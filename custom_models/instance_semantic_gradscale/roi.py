"""Preserve the original ROI objective and scale only its feature gradient."""
import math

from torch import nn

from custom_models.instance_semantic.roi import InstanceSemanticLoss


def validate_gradient_scale(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('ROI feature gradient scale must be finite and in [0, 1]')
    return value


def scale_feature_gradient(vector, scale):
    """Identity forward; multiply the incoming feature gradient by scale.

    Keep a connected zero at scale=0. At scale=1 use the original tensor and
    graph, allowing an exact control against the unmodified ROI loss.
    """
    if scale == 1:
        return vector
    frozen = vector.detach()
    return frozen + scale * (vector - frozen)


class GradientScaledInstanceSemanticLoss(InstanceSemanticLoss):
    def __init__(self, reference, feature_gradient_scale=.25):
        """Reuse the already initialized original classifier, without RNG use.

        Pooling, valid-GT filtering, FP32 CE, DDP normalization and empty-GT
        handling remain the inherited implementation. No original source or
        classifier parameter object is modified.
        """
        scale = validate_gradient_scale(feature_gradient_scale)
        if type(reference) is not InstanceSemanticLoss:
            raise TypeError('Expected the original all-valid-GT ROI loss')
        nn.Module.__init__(self)
        self.strides = reference.strides
        self.loss_weight = reference.loss_weight
        self.finest_scale = reference.finest_scale
        self.output_size = reference.output_size
        self.classifier = reference.classifier
        self.feature_gradient_scale = scale
        self.train(reference.training)

    def pool_instances(self, features, rois):
        pooled = super().pool_instances(features, rois)
        return scale_feature_gradient(pooled, self.feature_gradient_scale)
