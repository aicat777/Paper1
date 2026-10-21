from mmyolo.registry import MODELS

from custom_models.roi_supervision_common import SelectedROISemanticHead
from .selection import RandomGTSelection


@MODELS.register_module()
class RandomInstanceSemanticRTMDetHead(SelectedROISemanticHead):
    def __init__(self, *args, semantic_selection_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection = RandomGTSelection(semantic_selection_seed)
