"""Original all-GT ROI supervision with selective region-to-point transfer."""
from .head import TransferInstanceSemanticRTMDetHead
from .supervision import RegionToPointTransfer, original_roi_readout, valid_rows_and_lookup

__all__ = ['TransferInstanceSemanticRTMDetHead', 'RegionToPointTransfer',
           'original_roi_readout', 'valid_rows_and_lookup']
