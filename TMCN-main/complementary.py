"""Shared/complementary reconstruction experiment, not a disentanglement guarantee.

Inspired by GMAE's joint reconstruction and detached shared decorrelation.
We use a batch/dimension-normalized squared cross-covariance, not its raw L1.
"""
import torch


def decorrelation_loss(common, specific):
    if common.ndim != 2 or specific.ndim != 2 or len(common) != len(specific):
        raise ValueError('Expected two [batch, feature] tensors with matching batch size')
    if len(common) < 2:
        return specific.sum() * 0
    shared = common.detach()
    shared = shared - shared.mean(dim=0, keepdim=True)
    specific = specific - specific.mean(dim=0, keepdim=True)
    cross_cov = shared.T @ specific / (len(common) - 1)
    return cross_cov.square().mean()


@torch.no_grad()
def branch_diagnostics(model, xs, common, specifics, joint):
    """First-batch diagnostics; mean replacement measures decoder sensitivity only."""
    mse = torch.nn.functional.mse_loss
    original = sum(mse(y, x) for y, x in zip(joint, xs))
    mean_specs = [s.mean(0, keepdim=True).expand_as(s) for s in specifics]
    mean_common = common.mean(0, keepdim=True).expand_as(common)
    no_spec = model.joint_reconstruct(common, mean_specs)
    no_common = model.joint_reconstruct(mean_common, specifics)
    stats = {
        'common_variance': common.var(0, unbiased=False).mean().item(),
        'mean_specific_rec_gap': (sum(mse(y, x) for y, x in zip(no_spec, xs)) - original).item(),
        'mean_common_rec_gap': (sum(mse(y, x) for y, x in zip(no_common, xs)) - original).item(),
    }
    for v, spec in enumerate(specifics):
        stats['specific_variance_' + str(v)] = spec.var(0, unbiased=False).mean().item()
    return stats
