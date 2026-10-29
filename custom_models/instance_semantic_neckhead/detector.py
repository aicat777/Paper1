"""One original detector forward; block only ROI gradients to the backbone."""
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS

from custom_models.instance_semantic_gradscale.head import GradientScaledSemanticRTMDetHead
from .routing import route_parameter_gradients


@MODELS.register_module()
class NeckHeadROISemanticDetector(YOLODetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.bbox_head, GradientScaledSemanticRTMDetHead):
            raise TypeError('Neck/head routing requires the original gradient-scaled ROI head')

    def roi_gradient_parameters(self):
        """All eligible neck parameters, classifier tower and auxiliary readout.

        The original ROI graph does not supervise the regression tower or the
        final detection predictors. Preserve that boundary. Module.parameters
        deduplicates the classification convolutions shared across scales.
        """
        return tuple(self.neck.parameters()) + tuple(
            self.bbox_head.head_module.cls_convs.parameters()) + tuple(
            self.bbox_head.instance_semantic.parameters())

    def loss(self, batch_inputs, batch_data_samples):
        losses = super().loss(batch_inputs, batch_data_samples)
        losses['loss_instance_semantic'] = route_parameter_gradients(
            losses['loss_instance_semantic'], self.roi_gradient_parameters())
        return losses
