"""Uniform GT subset of the current model's matched-GT count, with private RNG."""
import torch
from torch import nn


class RandomGTSelection(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.register_buffer('rng_state', generator.get_state())

    def forward(self, matched):
        count = int(matched.sum())
        if count == 0 or count == len(matched):
            return matched.clone()
        generator = torch.Generator(device='cpu')
        generator.set_state(self.rng_state.detach().cpu())
        indices = torch.randperm(len(matched), generator=generator)[:count]
        self.rng_state.copy_(generator.get_state().to(self.rng_state.device))
        selected = torch.zeros_like(matched)
        selected[indices.to(matched.device)] = True
        return selected
