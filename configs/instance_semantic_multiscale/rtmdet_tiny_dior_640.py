"""Clean object views at one random scale per step; main batch36 is unchanged."""
import copy

_base_ = '../instance_semantic/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.instance_semantic_multiscale'], allow_failed_imports=False)
model = dict(type='ObjectCropSemanticDetector',
             crop_scales=(32, 64, 128, 256, 512, 800), crop_scale_seed=42,
             data_preprocessor=dict(type='ObjectCropDataPreprocessor'),
             bbox_head=dict(type='CropSemanticRTMDetHead',
                            head_module=dict(type='CropSemanticHeadModule')))

train_pipeline = copy.deepcopy(_base_.train_pipeline)
train_pipeline.insert(2, dict(type='SnapshotObjectCrop', seed=42, pad_value=114))
train_pipeline[-1]['type'] = 'PackObjectCropInputs'
train_pipeline_stage2 = copy.deepcopy(_base_.train_pipeline_stage2)
train_pipeline_stage2.insert(2, dict(type='SnapshotObjectCrop', seed=42, pad_value=114))
train_pipeline_stage2[-1]['type'] = 'PackObjectCropInputs'
train_dataloader = dict(collate_fn=dict(type='object_crop_collate'), dataset=dict(pipeline=train_pipeline))
custom_hooks = copy.deepcopy(_base_.custom_hooks)
custom_hooks[1]['switch_pipeline'] = train_pipeline_stage2
optim_wrapper = dict(type='ObjectCropAmpOptimWrapper')
work_dir = _base_.root + '/work_dirs/instance_semantic_multiscale_rtmdet_tiny_dior_640_bs36'
