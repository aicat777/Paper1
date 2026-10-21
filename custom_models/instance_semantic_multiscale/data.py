"""Clean GT crops travel beside the unchanged strong/weak detection pipeline."""
import math

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.dataset import COLLATE_FUNCTIONS
from mmdet.datasets.transforms import PackDetInputs
from mmyolo.datasets.utils import yolov5_collate
from mmyolo.registry import TRANSFORMS


def square_object_crop(image, box, pad_value=114):
    """Enclose a fractional xyxy box in an integer square; retain its true ROI."""
    height, width = image.shape[:2]
    box = np.asarray(box, dtype=np.float32).copy()
    box[0::2] = box[0::2].clip(0, width)
    box[1::2] = box[1::2].clip(0, height)
    if not np.isfinite(box).all() or not (box[2:] > box[:2]).all():
        raise ValueError('A clean crop needs a finite positive-area visible GT')
    x1, y1 = np.floor(box[:2]).astype(np.int64)
    x2, y2 = np.ceil(box[2:]).astype(np.int64)
    side = int(max(x2 - x1, y2 - y1))
    left = math.floor((int(x1) + int(x2) - side) / 2)
    top = math.floor((int(y1) + int(y2) - side) / 2)
    crop = np.full((side, side, image.shape[2]), pad_value, dtype=image.dtype)
    sx1, sy1, sx2, sy2 = max(left, 0), max(top, 0), min(left + side, width), min(top + side, height)
    crop[sy1 - top:sy2 - top, sx1 - left:sx2 - left] = image[sy1:sy2, sx1:sx2]
    relative = box - np.asarray([left, top, left, top], dtype=np.float32)
    return crop, relative, (left, top)


@TRANSFORMS.register_module()
class SnapshotObjectCrop(BaseTransform):
    """Sample one GT before Mosaic/MixUp without consuming their NumPy RNG."""
    def __init__(self, seed=42, pad_value=114):
        self.seed, self.pad_value = seed, pad_value
        self._rng = None

    def transform(self, results):
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed ^ torch.initial_seed())
        boxes = results['gt_bboxes']
        boxes = boxes.tensor.cpu().numpy() if hasattr(boxes, 'tensor') else np.asarray(boxes)
        boxes = boxes.astype(np.float32, copy=True)
        height, width = results['img'].shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
        valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        if 'gt_ignore_flags' in results:
            valid &= ~np.asarray(results['gt_ignore_flags'], dtype=bool)
        indices = np.flatnonzero(valid)
        if not len(indices):
            results['object_crop'] = None
            return results
        index = int(self._rng.choice(indices))
        crop, relative, origin = square_object_crop(results['img'], boxes[index], self.pad_value)
        results['object_crop'] = dict(image=crop, box=relative,
            label=int(results['gt_bboxes_labels'][index]), source_img_id=results.get('img_id', -1),
            source_gt_index=index, source_box=boxes[index].copy(), origin=origin,
            source_shape=results['img'].shape[:2])
        return results


@TRANSFORMS.register_module()
class PackObjectCropInputs(PackDetInputs):
    def transform(self, results):
        if 'object_crop' not in results:
            raise KeyError('SnapshotObjectCrop must run before detection augmentations')
        output = super().transform(results)
        crop = results['object_crop']
        if crop is not None:
            crop = dict(crop)
            crop['image'] = torch.from_numpy(np.ascontiguousarray(crop['image'].transpose(2, 0, 1)))
            crop['box'] = torch.as_tensor(crop['box'], dtype=torch.float32)
        output['data_samples'].set_field(crop, 'object_crop')
        return output


@COLLATE_FUNCTIONS.register_module()
def object_crop_collate(data_batch):
    output = yolov5_collate(data_batch)
    crops = [sample['data_samples'].object_crop for sample in data_batch]
    valid = [(index, crop) for index, crop in enumerate(crops) if crop is not None]
    output['data_samples']['object_crops'] = dict(
        images=[crop['image'] for _, crop in valid],
        boxes=torch.stack([crop['box'] for _, crop in valid]) if valid else torch.empty(0, 4),
        labels=torch.tensor([crop['label'] for _, crop in valid], dtype=torch.long),
        batch_indices=torch.tensor([index for index, _ in valid], dtype=torch.long),
        metadata=[{key: value for key, value in crop.items() if key not in ('image', 'box', 'label')}
                  for _, crop in valid])
    return output
