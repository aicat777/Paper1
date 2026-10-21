"""E: random valid GTs, count from current DSL matches; all-GT denominator."""
_base_ = '../instance_semantic/rtmdet_tiny_dior_640.py'
custom_imports = dict(imports=['custom_models.instance_semantic_random'], allow_failed_imports=False)
model = dict(bbox_head=dict(type='RandomInstanceSemanticRTMDetHead', semantic_selection_seed=42),
             train_cfg=dict(assigner=dict(type='ROISemanticIndexedAssigner')))
work_dir = _base_.root + '/work_dirs/instance_semantic_random_rtmdet_tiny_dior_640_bs36'
