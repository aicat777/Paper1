"""All-GT ROI CE through its own MLP; shared-feature gradients stay scaled."""
import torch
from torch import nn

from custom_models.instance_semantic_gradscale.roi import GradientScaledInstanceSemanticLoss


class PrivateInstanceSemanticLoss(GradientScaledInstanceSemanticLoss):
    def __init__(self, reference, feature_gradient_scale=.25, hidden_channels=192):
        super().__init__(reference, feature_gradient_scale)
        if not isinstance(hidden_channels, int) or hidden_channels < 1:
            raise ValueError('The private ROI width must be a positive integer')
        channels = self.classifier.in_features
        # Extra training parameters must not perturb any detector initialization.
        with torch.random.fork_rng(devices=[]):
            self.encoder = nn.Sequential(
                nn.Linear(channels, hidden_channels), nn.SiLU(),
                nn.Linear(hidden_channels, channels))
            for layer in self.encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def pool_instances(self, features, rois):
        # Scale BEFORE the private encoder: only the shared neck/backbone get
        # alpha-scaled auxiliary derivatives. The MLP/classifier get full CE.
        vector = super().pool_instances(features, rois)
        return self.encoder(vector)
