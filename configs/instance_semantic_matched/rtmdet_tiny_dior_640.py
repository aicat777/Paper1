"""D: ROI CE on matched valid GTs; denominator remains all valid GTs."""
_base_ = '../instance_semantic/rtmdet_tiny_dior_640.py'
custom_imports = dict(imports=['custom_models.instance_semantic_matched'], allow_failed_imports=False)
model = dict(bbox_head=dict(type='MatchedInstanceSemanticRTMDetHead'),
             train_cfg=dict(assigner=dict(type='ROISemanticIndexedAssigner')))
work_dir = _base_.root + '/work_dirs/instance_semantic_matched_rtmdet_tiny_dior_640_bs36'
