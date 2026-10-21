from torch import nn


class MatchedGTSelection(nn.Module):
    def forward(self, matched):
        return matched.clone()
