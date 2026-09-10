"""Batch-local neighbor contrast, inspired by DMVCS; no label supervision."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def pure_hops(adjacency, hops):
    """Disjoint shortest-path shells of an undirected binary graph."""
    if hops not in (1, 2, 3):
        raise ValueError('hops must be 1, 2, or 3')
    adjacency = adjacency.bool().clone()
    adjacency.fill_diagonal_(False)
    seen = torch.eye(len(adjacency), dtype=torch.bool, device=adjacency.device)
    frontier = adjacency
    shells = []
    for distance in range(hops):
        shell = frontier & ~seen
        shells.append(shell)
        seen |= shell
        if distance + 1 < hops:
            frontier = (shell.float() @ adjacency.float()) > 0
    return shells


@torch.no_grad()
def neighbor_weights(z, topk=10, hops=1, min_sim=0.5):
    if topk < 1 or not -1 <= min_sim <= 1:
        raise ValueError('topk >= 1 and min_sim in [-1, 1] required')
    n = len(z)
    sim = F.normalize(z.detach(), dim=1) @ F.normalize(z.detach(), dim=1).T
    sim.fill_diagonal_(-float('inf'))
    adjacency = torch.zeros_like(sim, dtype=torch.bool)
    if n > 1:
        values, indices = sim.topk(min(topk, n - 1), dim=1)
        adjacency.scatter_(1, indices, values >= min_sim)
    adjacency = adjacency & adjacency.T  # mutual neighbors only
    shells = pure_hops(adjacency, hops)
    weights = torch.zeros_like(sim)
    for distance, shell in enumerate(shells):
        weights += shell.to(sim.dtype) * (0.5 ** distance)
    return weights


def neighbor_contrastive_loss(z, weights, temperature=0.5):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if len(z) < 2:
        return z.sum() * 0
    weights = weights.detach().clone().to(z)
    weights.fill_diagonal_(0)
    mass = weights.sum(1)
    valid = mass > 0
    if not valid.any():
        return z.sum() * 0
    z = F.normalize(z, dim=1)
    logits = z @ z.T / temperature
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -float('inf')), dim=1)
    log_prob = logits - denominator[:, None]
    return -((weights * log_prob).sum(1)[valid] / mass[valid]).mean()


def mnc_weight(epoch, maximum, start=10, ramp=20):
    """epoch is one-based within fine-tuning; first `start` epochs are off."""
    if epoch <= start:
        return 0.0
    return maximum * min(1.0, (epoch - start) / max(1, ramp))
