"""Keep all-GT ROI supervision and add per-instance detector-logit ranking."""
from .assigner import IndexedDynamicSoftLabelAssigner
from .head import InstanceSemanticRankRTMDetHead
from .loss import InstanceCategoryRankLoss

__all__ = ['IndexedDynamicSoftLabelAssigner', 'InstanceSemanticRankRTMDetHead',
           'InstanceCategoryRankLoss']
