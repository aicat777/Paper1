"""First-order loss routing with the outer backward's actual AMP multiplier."""
import torch
from torch.autograd.function import once_differentiable


class _ParameterGradientRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, *parameters):
        ctx.save_for_backward(loss, *parameters)
        # Keep the reported CE unchanged without an ordinary backward edge
        # from this output to the original loss's feature/backbone graph.
        return loss.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx, multiplier):
        loss, *parameters = ctx.saved_tensors
        # The multiplier includes GradScaler and accumulation normalization.
        # Evaluating an unscaled VJP during forward can underflow AMP feature
        # gradients; wait for the official optimizer's backward instead.
        gradients = torch.autograd.grad(
            loss, parameters, grad_outputs=multiplier, allow_unused=True,
            retain_graph=True, create_graph=False)
        # Do not propagate through the original loss edge. Return the ROI
        # derivatives only on the selected parameter edges. The ordinary main
        # loss graph remains intact and supplies all backbone gradients.
        return (None, *gradients)


def route_parameter_gradients(loss, parameters):
    """Preserve scalar value and deliver its gradients only to these parameters.

    Intended for first-order training (the project's single-GPU AdamW/AMP).
    No .grad mutation, optimizer/scaler replacement, forward hooks, feature
    detachment, additional network forward or additional loss term is used.
    """
    parameters = tuple(p for p in parameters if p.requires_grad)
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError('Auxiliary routing parameters must be unique')
    if not torch.is_grad_enabled() or not loss.requires_grad:
        return loss
    if not parameters:
        return loss.detach()
    return _ParameterGradientRoute.apply(loss, *parameters)
