"""Two branch backwards, one MMEngine iteration/scaler/optimizer update."""
from mmengine.optim import AmpOptimWrapper, OptimWrapper
from mmyolo.registry import OPTIM_WRAPPERS


class AuxiliaryBackward:
    def backward_auxiliary(self, loss):
        if self._accumulative_counts != 1:
            raise ValueError('Object-crop training currently requires no gradient accumulation')
        # Only update_params(main_loss) advances _inner_count. Calling the
        # wrapper.backward here would incorrectly count two training iterations.
        if isinstance(self, AmpOptimWrapper):
            self.loss_scaler.scale(loss).backward()
        else:
            loss.backward()


@OPTIM_WRAPPERS.register_module()
class ObjectCropAmpOptimWrapper(AuxiliaryBackward, AmpOptimWrapper):
    pass


@OPTIM_WRAPPERS.register_module()
class ObjectCropOptimWrapper(AuxiliaryBackward, OptimWrapper):
    """FP32 counterpart for explicit gradient/update equivalence checks."""
