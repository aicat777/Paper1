"""C: all GTs, same semantic classifier; one bilinear GT-center sample."""
_base_ = '../instance_semantic/rtmdet_tiny_dior_640.py'
custom_imports = dict(imports=['custom_models.instance_semantic_center'], allow_failed_imports=False)
model = dict(bbox_head=dict(type='CenterInstanceSemanticRTMDetHead'))
work_dir = _base_.root + '/work_dirs/instance_semantic_center_rtmdet_tiny_dior_640_bs36'
