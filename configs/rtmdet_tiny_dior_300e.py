"""RTMDet-tiny: DIOR train/val, 640 pixels, VOC and COCO, from scratch."""
import os.path as osp

_base_ = '../mmyolo/configs/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py'

root = osp.abspath('{{ fileDirname }}/..')
checkpoint = None
classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
           'chimney', 'dam', 'expressway-service-area', 'expressway-toll-station',
           'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
           'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
           'windmill')
data_root = root + '/datasets/DIOR/'
num_classes = 20
img_scale = (640, 640)
max_epochs = 300
train_batch_size_per_gpu = 12
val_batch_size_per_gpu = 8
base_lr = 0.004 * train_batch_size_per_gpu / 256
work_dir = root + '/work_dirs/rtmdet_tiny_dior_640_300e_gpu0'

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

# Rebuild both stages: overriding img_scale alone does not update base lists.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic', img_scale=img_scale, use_cached=True,
         max_cached_images=20, random_pop=False, pad_val=114.0),
    dict(type='mmdet.RandomResize', scale=(img_scale[0] * 2, img_scale[1] * 2),
         ratio_range=(0.5, 2.0), resize_type='mmdet.Resize', keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='YOLOv5MixUp', use_cached=True, random_pop=False,
         max_cached_images=10, prob=0.5),
    dict(type='mmdet.PackDetInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.RandomResize', scale=img_scale,
         ratio_range=(0.5, 2.0), resize_type='mmdet.Resize', keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False,
         pad_val=dict(img=114)),
    # Load GT after image resizing so VOC compares original coordinates.
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(data_root=data_root,
                 ann_file='coco_annotations/instances_train.json',
                 data_prefix=dict(img=''),
                 metainfo=dict(classes=classes),
                 pipeline=train_pipeline))

batch_shapes_cfg = None
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(data_root=data_root,
                 ann_file='coco_annotations/instances_val.json',
                 data_prefix=dict(img=''),
                 metainfo=dict(classes=classes),
                 batch_shapes_cfg=None,
                 pipeline=test_pipeline))
val_evaluator = [
    dict(type='mmdet.VOCMetric', iou_thrs=0.5, metric='mAP',
         eval_mode='11points', prefix='pascal_voc'),
    dict(type='mmdet.CocoMetric',
         ann_file=data_root + 'coco_annotations/instances_val.json',
         metric='bbox', proposal_nums=(1, 10, 100), classwise=True)
]
val_cfg = dict(type='ValLoop', fp16=False)

train_cfg = dict(max_epochs=max_epochs, val_interval=10,
                 dynamic_intervals=[])

# Configured for tools/test.py; training evaluates val only.
test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(data_root=data_root,
                 ann_file='coco_annotations/instances_test.json',
                 data_prefix=dict(img=''),
                 metainfo=dict(classes=classes),
                 batch_shapes_cfg=None,
                 pipeline=test_pipeline))
test_cfg = dict(type='TestLoop')
test_evaluator = [
    dict(type='mmdet.VOCMetric', iou_thrs=0.5, metric='mAP',
         eval_mode='11points', prefix='pascal_voc'),
    dict(type='mmdet.CocoMetric',
         ann_file=data_root + 'coco_annotations/instances_test.json',
         metric='bbox', proposal_nums=(1, 10, 100), classwise=True)
]

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
                    save_best=['pascal_voc/mAP', 'coco/bbox_mAP'],
                    rule='greater'))
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, strict_load=False, priority=49),
    dict(type='mmdet.PipelineSwitchHook', switch_epoch=280,
         switch_pipeline=train_pipeline_stage2)
]
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)
