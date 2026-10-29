"""A private training-only ROI encoder branching from the neck outputs."""
from .head import PrivateROISemanticRTMDetHead
from .roi import PrivateInstanceSemanticLoss

__all__ = ['PrivateROISemanticRTMDetHead', 'PrivateInstanceSemanticLoss']
