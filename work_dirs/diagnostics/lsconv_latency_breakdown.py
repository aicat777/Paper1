"""Fixed-input latency attribution; does not modify or queue any experiment."""
from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import statistics
from types import MethodType

import torch
from torch import nn
from torch.profiler import record_function
from torch.nn.utils.fusion import fuse_conv_bn_eval

from custom_models.lsconv.operator import LSConv, ConvBN
from custom_models.lsconv.aggregation import small_kernel_aggregation

ROOT = Path('/home/pwt/Paper1')
spec = importlib.util.spec_from_file_location('ls_scope_audit', ROOT / 'work_dirs/diagnostics/lsconv_scope_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fuse_independently(module):
    # BN differs by scale: folding must create a separate convolution at each use.
    last = None
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            last = name
        elif isinstance(child, nn.modules.batchnorm._BatchNorm) and last is not None:
            module._modules[last] = fuse_conv_bn_eval(module._modules[last], child)
            module._modules[name] = nn.Identity()
            last = None
        else:
            fuse_independently(child)
            last = None
    return module


def time_eager(fn, repeats=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        output = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats, output


def time_graph(fn, repeats=80):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            reference = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fn()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / repeats
    if isinstance(captured, torch.Tensor):
        torch.testing.assert_close(captured, reference, atol=1e-3, rtol=1e-3)
    else:
        for left_group, right_group in zip(captured, reference):
            for left, right in zip(left_group, right_group):
                torch.testing.assert_close(left, right, atol=1e-3, rtol=1e-3)
    del graph
    return ms


def staged_time(model, x, repeats=20):
    values = defaultdict(list)
    for _ in range(repeats):
        points = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        points[0].record()
        features = model.backbone(x)
        points[1].record()
        features = model.neck(features)
        points[2].record()
        model.bbox_head.head_module(features)
        points[3].record()
        torch.cuda.synchronize()
        for i, name in enumerate(('backbone', 'neck', 'head')):
            values[name].append(points[i].elapsed_time(points[i + 1]))
    return {k: dict(median_ms=statistics.median(v), samples_ms=v) for k, v in values.items()}


def feature_caches(model, x):
    cache, handles = {}, []
    for name, op in model.named_modules():
        if not isinstance(op, LSConv):
            continue
        cache[name] = dict(op=op)
        def inputs(module, args, name=name):
            cache[name]['x'] = args[0].detach()
        def weights(module, args, output, name=name):
            cache[name]['w'] = output.detach()
        def norm_input(module, args, name=name):
            cache[name]['pre_gn'] = args[0].detach()
        handles.extend((op.register_forward_pre_hook(inputs),
                        op.lkp.register_forward_hook(weights),
                        op.lkp.norm.register_forward_pre_hook(norm_input)))
    model._forward(x)
    for handle in handles:
        handle.remove()
    return cache


def annotate(model):
    def lkp_forward(self, x):
        with record_function('COMPONENT.lkp_pw1'):
            y = self.cv1(x)
        with record_function('COMPONENT.lkp_relu'):
            y = self.act(y)
        with record_function('COMPONENT.lkp_dw7'):
            y = self.cv2(y)
        with record_function('COMPONENT.lkp_pw2'):
            y = self.cv3(y)
        with record_function('COMPONENT.lkp_relu'):
            y = self.act(y)
        with record_function('COMPONENT.lkp_weight_projection'):
            y = self.cv4(y)
        with record_function('COMPONENT.lkp_groupnorm'):
            y = self.norm(y)
        b, _, h, w = y.shape
        return y.reshape(b, x.shape[1] // 8, 9, h, w)
    def ls_forward(self, x):
        weights = self.lkp(x)
        with record_function('COMPONENT.ska'):
            y = small_kernel_aggregation(x, weights)
        with record_function('COMPONENT.output_bn'):
            y = self.bn(y)
        with record_function('COMPONENT.residual_add'):
            y = x + y
        return y
    inner = set()
    for module in model.modules():
        if isinstance(module, LSConv):
            inner.update(id(child) for child in module.lkp.modules())
            module.lkp.forward = MethodType(lkp_forward, module.lkp)
            module.forward = MethodType(ls_forward, module)
    for module in model.modules():
        if isinstance(module, ConvBN) and id(module) not in inner:
            original = module.forward
            def projection(self, x, original=original):
                with record_function('COMPONENT.explicit_channel_projection'):
                    return original(x)
            module.forward = MethodType(projection, module)


def profile(model, x, label, repeats=5):
    trace_path = Path('/tmp') / ('paper1_lsconv_' + label + '_trace.json')
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        for i in range(repeats):
            with record_function('FRAME'):
                with record_function('REGION.backbone'):
                    features = model.backbone(x)
                with record_function('REGION.neck'):
                    features = model.neck(features)
                with record_function('REGION.head'):
                    model.bbox_head.head_module(features)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())['traceEvents']
    # Torch 2.0's kernel External id is a CUDA correlation id, not a CPU op id.
    launches = {str(e.get('args', {}).get('correlation')): e for e in events
                if e.get('cat') in ('cuda_runtime', 'cuda_driver')}
    scopes = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith(('COMPONENT.', 'REGION.'))]
    gpu = sorted((e for e in events if e.get('cat') == 'kernel'), key=lambda e: e['ts'])
    totals, counts = defaultdict(float), Counter()
    for idx, kernel in enumerate(gpu):
        owner = launches.get(str(kernel.get('args', {}).get('correlation')))
        groups = []
        if owner is None and kernel['name'].startswith('_forward_kernel'):
            # The driver launch may be absent from this Torch trace. SKA is
            # immediately followed by its output BN on the same stream.
            for following in gpu[idx + 1:]:
                owner = launches.get(str(following.get('args', {}).get('correlation')))
                if owner is not None:
                    break
        if owner is not None:
            t = owner['ts']
            groups = [e for e in scopes if e['ts'] <= t <= e['ts'] + e.get('dur', 0)]
        for prefix in ('COMPONENT.', 'REGION.'):
            matches = [e for e in groups if e['name'].startswith(prefix)]
            category = min(matches, key=lambda e: e['dur'])['name'] if matches else prefix + 'unattributed'
            if prefix == 'COMPONENT.' and kernel['name'].startswith('_forward_kernel'):
                category = 'COMPONENT.ska'
            totals[category] += kernel['dur'] / 1000 / repeats
            counts[category] += 1 / repeats
    cpu_rows = []
    for event in prof.key_averages():
        if event.key.startswith('COMPONENT.') or event.key.startswith('REGION.'):
            cpu_rows.append(dict(name=event.key, calls_per_frame=event.count / repeats,
                                 self_cpu_ms=event.self_cpu_time_total / 1000 / repeats,
                                 inclusive_cpu_ms=event.cpu_time_total / 1000 / repeats))
    return dict(kernel_count_per_frame=len(gpu) / repeats,
                active_kernel_ms_per_frame=sum(e['dur'] for e in gpu) / 1000 / repeats,
                category_kernel_ms=dict(totals), category_kernel_counts=dict(counts),
                cpu_scopes=cpu_rows,
                largest_kernel_names=Counter(e['name'] for e in gpu).most_common(15),
                trace=str(trace_path), caveat='Profiler perturbs timing; CUDA-kernel durations exclude gaps between launches')


def main():
    free, _ = torch.cuda.mem_get_info()
    assert free / 2**30 > 10
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cudnn.benchmark = False
    report = dict(status='running', batch=1, input=[1, 3, 640, 640], amp=True,
                  conditions='Shared GPU0. Independently fold each Conv/BN use. Fixed input. Autocast cache retained throughout graph replay.',
                  formal_experiments_changed=False, cases={})
    models = {}
    for label, mode in (('baseline', 'baseline'), ('expanded', 'with_channel_projections')):
        model, _, _ = audit.build(mode)
        model.cuda()
        model.eval()
        models[label] = fuse_independently(model)
    x = torch.randn(1, 3, 640, 640, device='cuda')
    # Keep cached half-precision parameter tensors alive during all graph replays.
    with torch.no_grad(), torch.autocast('cuda'):
        for label, model in models.items():
            eager, _ = time_eager(lambda: model._forward(x))
            graph = time_graph(lambda: model._forward(x))
            report['cases'][label] = dict(eager_ms=eager, cuda_graph_ms=graph,
                                         stage_timings=staged_time(model, x))
            print(label, json.dumps(report['cases'][label]), flush=True)
        cache = feature_caches(models['expanded'], x)
        report['lsconv_runtime_calls'] = len(cache)
        report['dynamic_weight_elements'] = sum(row['w'].numel() for row in cache.values())
        report['dynamic_weight_mib'] = sum(row['w'].numel() * row['w'].element_size() for row in cache.values()) / 2**20
        report['dynamic_weight_dtypes'] = dict(Counter(str(row['w'].dtype) for row in cache.values()))
        grouped = defaultdict(list)
        for name, row in cache.items():
            grouped[tuple(row['x'].shape)].append((name, row))
        micro = []
        for shape, rows in grouped.items():
            name, row = rows[0]
            op, values, weights = row['op'], row['x'], row['w']
            agg = small_kernel_aggregation(values, weights)
            functions = {'lkp_full': lambda op=op, values=values: op.lkp(values),
                         'groupnorm_only': lambda op=op, row=row: op.lkp.norm(row['pre_gn']),
                         'ska': lambda values=values, weights=weights: small_kernel_aggregation(values, weights),
                         'output_bn_and_add': lambda op=op, agg=agg, values=values: values + op.bn(agg)}
            entry = dict(shape=list(shape), calls_per_frame=len(rows), representative=name, parts={})
            for key, fn in functions.items():
                eager, _ = time_eager(fn, repeats=40)
                graph = time_graph(fn)
                entry['parts'][key] = dict(eager_ms=eager, graph_ms=graph)
            micro.append(entry)
        report['representative_microbenchmarks'] = micro
        for label, model in models.items():
            if label == 'expanded':
                annotate(model)
            report['cases'][label]['profiler'] = profile(model, x, label)
    report['status'] = 'passed'
    target = ROOT / 'work_dirs/diagnostics/lsconv_latency_breakdown.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    for label, case in report['cases'].items():
        print(label, case['eager_ms'], case['cuda_graph_ms'], case['profiler']['active_kernel_ms_per_frame'],
              case['profiler']['kernel_count_per_frame'], flush=True)
    print('runtime LSConv calls', len(cache), 'dynamic kernel MiB', report['dynamic_weight_mib'], flush=True)


if __name__ == '__main__':
    main()
