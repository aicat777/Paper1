"""Read-only checkpoint and same-object view diagnostics, not detection AP."""
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic_multiscale.data import square_object_crop
from custom_models.instance_semantic_multiscale.normalization import crop_normalization
from custom_models.instance_semantic_multiscale.preprocessor import ObjectCropDataPreprocessor
from custom_models.instance_semantic_multiscale.roi import CropSemanticLoss

ROOT = Path('/home/pwt/Paper1')
OUT = Path(__file__).resolve().parent
SCALES = (32, 64, 128, 256, 512, 800)
register_all_modules()
torch.set_num_threads(2)
free, total = torch.cuda.mem_get_info()
assert free > total * .3 + 2 * 2**30
torch.cuda.set_per_process_memory_fraction(.3)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def summarize(logits, labels):
    return dict(n=len(labels), correct=int((logits.argmax(1) == labels).sum()),
                ce_sum=float(F.cross_entropy(logits, labels, reduction='sum')),
                true_probability_sum=float(logits.softmax(1).gather(1, labels[:, None]).sum()))


def classification(model, images, rows, helper):
    levels = helper.levels(rows[:, 2:])
    pyramid = model.extract_feat(images)
    maps = {}
    for level in levels.unique().tolist():
        x = pyramid[level]
        for layer in model.bbox_head.head_module.cls_convs[level]:
            x = layer(x)
        maps[level] = x
    rois = torch.cat((rows[:, :1], rows[:, 2:]), 1)
    vectors = helper.pool_instances(maps, rois, levels)
    return model.bbox_head.instance_semantic.classifier(vectors)


annotation = json.loads((ROOT / 'datasets/DIOR/coco_annotations/instances_val.json').read_text())
metadata = {row['id']: row for row in annotation['images']}
by_class = defaultdict(list)
for row in annotation['annotations']:
    if not row.get('iscrowd', 0) and min(row['bbox'][2:]) > 0:
        by_class[row['category_id']].append(row)
generator = np.random.default_rng(42)
selected, used = [], set()
for category in annotation['categories']:
    rows = by_class[category['id']]
    count = 0
    for index in generator.permutation(len(rows)):
        row = rows[index]
        if row['image_id'] in used:
            continue
        selected.append(row)
        used.add(row['image_id'])
        count += 1
        if count == 5:
            break
    assert count == 5
# Mixed categories per crop BN batch; maintain identical ordering for both models.
selected = [selected[index] for index in generator.permutation(len(selected))]
classes = {category['id']: i for i, category in enumerate(annotation['categories'])}
samples = []
for row in selected:
    meta = metadata[row['image_id']]
    raw = cv2.imread(str(ROOT / 'datasets/DIOR' / meta['file_name']))
    assert raw is not None and raw.shape[:2] == (800, 800)
    x, y, width, height = row['bbox']
    box = np.array([x, y, x + width, y + height], np.float32)
    crop, relative, _ = square_object_crop(raw, box)
    samples.append(dict(image=torch.from_numpy(cv2.resize(raw, (640, 640)).transpose(2, 0, 1).copy()),
        box=torch.from_numpy(box * .8), crop=torch.from_numpy(crop.transpose(2, 0, 1).copy()),
        crop_box=torch.from_numpy(relative), label=classes[row['category_id']], image_id=row['image_id']))

report = dict(sample='100 held-out val images, 5 selected targets/class, one target/image',
              inference_ap=False, images=[row['image_id'] for row in selected], models={})
helper = CropSemanticLoss(96, 20).cuda()
for tag, folder in (('old', 'instance_semantic_rtmdet_tiny_dior_640_gpu1_bs36'),
                    ('new', 'instance_semantic_multiscale_rtmdet_tiny_dior_640_gpu1_bs36')):
    directory = ROOT / 'work_dirs' / folder
    cfg = Config.fromfile(str(next(directory.glob('*.py'))))
    saved = torch.load(directory / 'epoch_300.pth', map_location='cpu', weights_only=False)
    optimizer = saved['optimizer']
    state = optimizer.get('state', optimizer.get('optimizer', {}).get('state', {}))
    steps = sorted({int(value['step']) for value in state.values() if 'step' in value})
    bad = [key + '.' + name for key in ('state_dict', 'ema_state_dict')
           for name, value in saved[key].items()
           if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all()]
    assert not bad
    model = MODELS.build(cfg.model)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.cuda()
    model.eval()
    before = {name: value.clone() for name, value in model.named_buffers()}
    data = dict(epoch=saved['meta']['epoch'], iterations=saved['meta']['iter'], optimizer_steps=steps,
        amp=optimizer.get('loss_scaler'), nonfinite_weights=bad,
        raw_bn_counts=sorted({int(value) for name, value in saved['ema_state_dict'].items()
                              if name.endswith('num_batches_tracked')}), views={})
    del saved
    measurements = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(samples), 8):
            batch = samples[start:start + 8]
            labels = torch.tensor([item['label'] for item in batch], device='cuda')
            index = torch.arange(len(batch), device='cuda', dtype=torch.float32)
            boxes = torch.stack([item['box'] for item in batch]).cuda()
            rows = torch.cat((index[:, None], labels[:, None], boxes), 1)
            images = torch.stack([item['image'] for item in batch]).cuda().float()
            pre = model.data_preprocessor
            if pre._channel_conversion:
                images = images[:, [2, 1, 0]]
            if pre._enable_normalize:
                images = (images - pre.mean) / pre.std
            measurements['whole_image_running_bn'].append(summarize(classification(model, images, rows, helper), labels))
            crops = dict(images=[item['crop'].cuda() for item in batch],
                boxes=torch.stack([item['crop_box'] for item in batch]).cuda(), labels=labels)
            for scale in SCALES:
                images, crop_rows = ObjectCropDataPreprocessor.prepare_crops(pre, crops, scale)
                for own in (False, True):
                    name = f'crop_{scale}_' + ('own_bn' if own else 'running_bn')
                    with crop_normalization(model) if own else nullcontext():
                        logits = classification(model, images, crop_rows, helper)
                    measurements[name].append(summarize(logits, labels))
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0, msg=name)
    for name, values in measurements.items():
        n = sum(value['n'] for value in values)
        data['views'][name] = dict(n=n, accuracy=sum(value['correct'] for value in values) / n,
            ce=sum(value['ce_sum'] for value in values) / n,
            mean_true_probability=sum(value['true_probability_sum'] for value in values) / n)
    report['models'][tag] = data
    print(tag, 'INTEGRITY', {k: v for k, v in data.items() if k != 'views'}, flush=True)
    print(tag, 'VIEWS', data['views'], flush=True)
    del model, before, optimizer, state
    torch.cuda.empty_cache()
(OUT / 'trained_views.json').write_text(json.dumps(report, indent=2) + '\n')
