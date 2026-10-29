_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
backend_args = None
base_lr = 0.0005625000000000001
batch_shapes_cfg = None
checkpoint = None
classes = (
    'airplane',
    'airport',
    'baseballfield',
    'basketballcourt',
    'bridge',
    'chimney',
    'dam',
    'expressway-service-area',
    'expressway-toll-station',
    'golffield',
    'groundtrackfield',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'storagetank',
    'tenniscourt',
    'trainstation',
    'vehicle',
    'windmill',
)
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=280,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                resize_type='mmdet.Resize',
                scale=(
                    640,
                    640,
                ),
                type='mmdet.RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='mmdet.RandomCrop'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='mmdet.Pad'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'custom_models.scdown_p3_g4_semantic',
        'custom_models.instance_semantic_gradscale',
    ])
data_root = '/home/pwt/Paper1/datasets/DIOR/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 0.167
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,
        rule='greater',
        save_best=[
            'pascal_voc/mAP',
            'coco/bbox_mAP',
        ],
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
dsl_topk = 13
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 2.0
loss_cls_weight = 1.0
lr_start_factor = 1e-05
max_epochs = 300
max_keep_ckpts = 3
mixup_max_cached_images = 10
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.167,
        expand_ratio=0.5,
        init_cfg=None,
        norm_cfg=dict(type='BN'),
        type='SCDownP3G4CSPNeXt',
        widen_factor=0.375),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            feat_channels=256,
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=256,
            norm_cfg=dict(type='BN'),
            num_classes=20,
            pred_kernel_size=1,
            share_conv=True,
            stacked_convs=2,
            type='SemanticFeatureHeadModule',
            widen_factor=0.375),
        loss_bbox=dict(loss_weight=2.0, type='mmdet.GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True),
        prior_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        semantic_feature_gradient_scale=0.25,
        semantic_finest_scale=56,
        semantic_loss_weight=0.1,
        type='GradientScaledSemanticRTMDetHead'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.167,
        expand_ratio=0.5,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(type='BN'),
        num_csp_blocks=3,
        out_channels=256,
        type='SCDownG4PAFPN',
        widen_factor=0.375),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            num_classes=20,
            topk=13,
            type='BatchDynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='YOLODetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.65, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosaic_max_cached_images = 20
norm_cfg = dict(type='BN')
num_classes = 20
num_epochs_stage2 = 20
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(lr=0.0005625000000000001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=150,
        begin=150,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=2.8125000000000006e-05,
        type='CosineAnnealingLR'),
]
persistent_workers = True
qfl_beta = 2.0
random_resize_ratio_range = (
    0.5,
    2.0,
)
randomness = dict(deterministic=False, seed=42)
resume = False
root = '/home/pwt/Paper1'
save_checkpoint_intervals = 10
strides = [
    8,
    16,
    32,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='coco_annotations/instances_test.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root='/home/pwt/Paper1/datasets/DIOR/',
        metainfo=dict(
            classes=(
                'airplane',
                'airport',
                'baseballfield',
                'basketballcourt',
                'bridge',
                'chimney',
                'dam',
                'expressway-service-area',
                'expressway-toll-station',
                'golffield',
                'groundtrackfield',
                'harbor',
                'overpass',
                'ship',
                'stadium',
                'storagetank',
                'tenniscourt',
                'trainstation',
                'vehicle',
                'windmill',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        eval_mode='11points',
        iou_thrs=0.5,
        metric='mAP',
        prefix='pascal_voc',
        type='mmdet.VOCMetric'),
    dict(
        ann_file=
        '/home/pwt/Paper1/datasets/DIOR/coco_annotations/instances_test.json',
        classwise=True,
        metric='bbox',
        proposal_nums=(
            1,
            10,
            100,
        ),
        type='mmdet.CocoMetric'),
]
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 36
train_cfg = dict(
    dynamic_intervals=[],
    max_epochs=300,
    type='EpochBasedTrainLoop',
    val_interval=10)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=36,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        ann_file='coco_annotations/instances_train.json',
        data_prefix=dict(img=''),
        data_root='/home/pwt/Paper1/datasets/DIOR/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'airplane',
                'airport',
                'baseballfield',
                'basketballcourt',
                'bridge',
                'chimney',
                'dam',
                'expressway-service-area',
                'expressway-toll-station',
                'golffield',
                'groundtrackfield',
                'harbor',
                'overpass',
                'ship',
                'stadium',
                'storagetank',
                'tenniscourt',
                'trainstation',
                'vehicle',
                'windmill',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                max_cached_images=20,
                pad_val=114.0,
                random_pop=False,
                type='Mosaic',
                use_cached=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                resize_type='mmdet.Resize',
                scale=(
                    1280,
                    1280,
                ),
                type='mmdet.RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='mmdet.RandomCrop'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='mmdet.Pad'),
            dict(
                max_cached_images=10,
                prob=0.5,
                random_pop=False,
                type='YOLOv5MixUp',
                use_cached=True),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 10
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        max_cached_images=20,
        pad_val=114.0,
        random_pop=False,
        type='Mosaic',
        use_cached=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        resize_type='mmdet.Resize',
        scale=(
            1280,
            1280,
        ),
        type='mmdet.RandomResize'),
    dict(crop_size=(
        640,
        640,
    ), type='mmdet.RandomCrop'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            640,
            640,
        ),
        type='mmdet.Pad'),
    dict(
        max_cached_images=10,
        prob=0.5,
        random_pop=False,
        type='YOLOv5MixUp',
        use_cached=True),
    dict(type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        resize_type='mmdet.Resize',
        scale=(
            640,
            640,
        ),
        type='mmdet.RandomResize'),
    dict(crop_size=(
        640,
        640,
    ), type='mmdet.RandomCrop'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            640,
            640,
        ),
        type='mmdet.Pad'),
    dict(type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 8
val_cfg = dict(fp16=False, type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='coco_annotations/instances_val.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root='/home/pwt/Paper1/datasets/DIOR/',
        metainfo=dict(
            classes=(
                'airplane',
                'airport',
                'baseballfield',
                'basketballcourt',
                'bridge',
                'chimney',
                'dam',
                'expressway-service-area',
                'expressway-toll-station',
                'golffield',
                'groundtrackfield',
                'harbor',
                'overpass',
                'ship',
                'stadium',
                'storagetank',
                'tenniscourt',
                'trainstation',
                'vehicle',
                'windmill',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        eval_mode='11points',
        iou_thrs=0.5,
        metric='mAP',
        prefix='pascal_voc',
        type='mmdet.VOCMetric'),
    dict(
        ann_file=
        '/home/pwt/Paper1/datasets/DIOR/coco_annotations/instances_val.json',
        classwise=True,
        metric='bbox',
        proposal_nums=(
            1,
            10,
            100,
        ),
        type='mmdet.CocoMetric'),
]
val_interval_stage2 = 1
val_num_workers = 10
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.05
widen_factor = 0.375
work_dir = '/home/pwt/Paper1/work_dirs/scdown_p3_g4_semantic_gradscale_rtmdet_tiny_dior_640_bs36'
