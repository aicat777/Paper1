"""RTMDet-tiny from scratch: original DIOR train/val split, one GPU."""
import os.path as osp

_base_ = '../mmyolo/configs/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py'

root = osp.abspath('{{ fileDirname }}/..')
# Clear the inherited ImageNet checkpoint variable as well as its init_cfg.
checkpoint = None
classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
           'chimney', 'dam', 'expressway-service-area', 'expressway-toll-station',
           'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
           'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
           'windmill')
data_root = root + '/datasets/DIOR/'
num_classes = 20
max_epochs = 300
base_lr = 0.004 * 16 / 256

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(data_root=data_root,
                 ann_file='coco_annotations/instances_train.json',
                 data_prefix=dict(img=''),
                 metainfo=dict(classes=classes)))

batch_shapes_cfg = dict(type='BatchShapePolicy', batch_size=16,
                        img_size=640, size_divisor=32, extra_pad_ratio=0.5)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(data_root=data_root,
                 ann_file='coco_annotations/instances_val.json',
                 data_prefix=dict(img=''),
                 metainfo=dict(classes=classes),
                 batch_shapes_cfg=batch_shapes_cfg))
val_evaluator = dict(
    ann_file=data_root + 'coco_annotations/instances_val.json',
    proposal_nums=(1, 10, 100),
    classwise=True)
val_cfg = dict(type='ValLoop', fp16=False)

# _decide_current_val_interval uses the next epoch's milestone. Thus epoch 1
# validates once; epochs 10..270 validate every 10; epochs 280..300 every epoch.
train_cfg = dict(max_epochs=max_epochs, val_interval=1,
                 dynamic_intervals=[(3, 10), (281, 1)])

# No test-set evaluation during model selection.
test_dataloader = None
test_cfg = None
test_evaluator = None

optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic',
                     optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=150, end=300, T_max=150, by_epoch=True,
         convert_to_iter_based=True)]
default_hooks = dict(
    logger=dict(interval=50),
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_last=True,
                    save_best=['coco/bbox_mAP', 'coco/bbox_mAP_50'],
                    rule='greater'))
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)

# Inherited EMA and PipelineSwitchHook retain the original 280/20 schedule.
