"""Route the original ROI auxiliary gradient to neck and classification only."""
from .detector import NeckHeadROISemanticDetector
from .routing import route_parameter_gradients

__all__ = ['NeckHeadROISemanticDetector', 'route_parameter_gradients']
