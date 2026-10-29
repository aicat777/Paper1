"""All GTs train the auxiliary classifier; matched GTs train shared features."""
_base_ = '../instance_semantic/rtmdet_tiny_dior_640.py'
custom_imports = dict(imports=['custom_models.instance_semantic_split'], allow_failed_imports=False)
model = dict(bbox_head=dict(type='SplitGradientInstanceSemanticRTMDetHead'),
             train_cfg=dict(assigner=dict(type='ROISemanticIndexedAssigner')))
work_dir = _base_.root + '/work_dirs/instance_semantic_split_rtmdet_tiny_dior_640_bs36'
