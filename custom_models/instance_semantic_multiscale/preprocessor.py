"""Normalize clean crops once, independently of the detection augmentations."""
import torch
from torch.nn import functional as F
from mmyolo.models.data_preprocessors import YOLOv5DetDataPreprocessor
from mmyolo.registry import MODELS


@MODELS.register_module()
class ObjectCropDataPreprocessor(YOLOv5DetDataPreprocessor):
    def forward(self, data, training=False):
        if not training:
            return super().forward(data, training=False)
        samples = dict(data['data_samples'])
        crops = samples.pop('object_crops')
        main = dict(data, data_samples=samples)
        output = super().forward(main, training=True)
        output['object_crops'] = self.cast_data(crops)
        return output

    def prepare_crops(self, crops, scale):
        images, boxes = [], []
        for image, box in zip(crops['images'], crops['boxes']):
            if image.shape[-1] != image.shape[-2]:
                raise ValueError('The saved object crop must be square')
            side = image.shape[-1]
            image = image.float()
            if self._channel_conversion:
                image = image[[2, 1, 0]]
            image = F.interpolate(image[None], size=(scale, scale), mode='bilinear', align_corners=False)[0]
            if self._enable_normalize:
                image = (image - self.mean) / self.std
            images.append(image)
            boxes.append(box.float() * (scale / side))
        if not images:
            device = crops['boxes'].device
            return torch.empty(0, 3, scale, scale, device=device), torch.empty(0, 6, device=device)
        indices = torch.arange(len(images), device=boxes[0].device, dtype=torch.float32)
        rows = torch.cat((indices[:, None], crops['labels'].float()[:, None], torch.stack(boxes)), dim=1)
        return torch.stack(images), rows
