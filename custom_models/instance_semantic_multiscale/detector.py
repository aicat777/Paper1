"""Sequential auxiliary/main graphs share weights and one optimizer update."""
import torch
from mmengine.dist import get_world_size
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS

from .normalization import crop_normalization
from .optimizer import AuxiliaryBackward


@MODELS.register_module()
class ObjectCropSemanticDetector(YOLODetector):
    def __init__(self, *args, crop_scales=(32, 64, 128, 256, 512, 800), crop_scale_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_scales = tuple(crop_scales)
        if not self.crop_scales or any(side < 32 or side % 32 for side in self.crop_scales):
            raise ValueError('Crop resolutions must be positive multiples of stride32')
        generator = torch.Generator(device='cpu').manual_seed(crop_scale_seed)
        self.register_buffer('crop_scale_rng_state', generator.get_state())

    def sample_crop_scale(self):
        generator = torch.Generator(device='cpu')
        generator.set_state(self.crop_scale_rng_state.detach().cpu())
        index = int(torch.randint(len(self.crop_scales), (1,), generator=generator))
        self.crop_scale_rng_state.copy_(generator.get_state().to(self.crop_scale_rng_state.device))
        return self.crop_scales[index]

    def auxiliary_loss(self, crops, scale):
        images, rows = self.data_preprocessor.prepare_crops(crops, scale)
        if not len(rows):
            return sum(p.sum() * 0 for p in self.bbox_head.instance_semantic.parameters())
        with crop_normalization(self):
            features = self.extract_feat(images)
            return self.bbox_head.crop_loss(features, rows)

    def loss(self, batch_inputs, batch_data_samples):
        if get_world_size() != 1:
            raise NotImplementedError('Object-crop two-branch training currently supports one GPU')
        return super().loss(batch_inputs, batch_data_samples)

    def train_step(self, data, optim_wrapper):
        if not isinstance(optim_wrapper, AuxiliaryBackward):
            raise TypeError('Use ObjectCropAmpOptimWrapper for the two-branch schedule')
        with optim_wrapper.optim_context(self):
            prepared = self.data_preprocessor(data, training=True)
            crops = prepared.pop('object_crops')
            scale = self.sample_crop_scale()
            auxiliary = self.auxiliary_loss(crops, scale)
        optim_wrapper.backward_auxiliary(auxiliary)
        auxiliary_value = auxiliary.detach()
        del auxiliary
        with optim_wrapper.optim_context(self):
            detection = self._run_forward(prepared, mode='loss')
        main_loss, log_vars = self.parse_losses(detection)
        optim_wrapper.update_params(main_loss)
        log_vars['loss_instance_semantic'] = auxiliary_value
        log_vars['loss'] = log_vars['loss'] + auxiliary_value
        return log_vars
