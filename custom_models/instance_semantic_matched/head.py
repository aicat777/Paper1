from mmyolo.registry import MODELS

from custom_models.roi_supervision_common import SelectedROISemanticHead
from .selection import MatchedGTSelection


@MODELS.register_module()
class MatchedInstanceSemanticRTMDetHead(SelectedROISemanticHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection = MatchedGTSelection()
