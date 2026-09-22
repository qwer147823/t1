"""Opt-in one-hop contrast on the original projected view features (hs).

The graphs are detached. The losses update hs, their shared projection and
their encoders; they do not directly update the common/Mamba projection.
This is not a guarantee of disentanglement or better clustering accuracy.
"""
import torch

from mnc import neighbor_weights, neighbor_contrastive_loss


@torch.no_grad()
def view_neighbor_weights(common, views, mode='shared', topk=10, min_sim=0.5):
    """Return per-view one-hop mutual-KNN graphs and label-free diagnostics.

    shared: each view uses the common graph.
    consensus: intersect the common graph with that view's own graph.
    Both graphs use the same topk/min_sim; neither uses the shared MNC hops.
    """
    if mode not in ('shared', 'consensus'):
        raise ValueError('view neighbor mode must be shared or consensus')
    if common.ndim != 2 or not views:
        raise ValueError('Expected common [batch, dim] and at least one view')
    if any(h.ndim != 2 or len(h) != len(common) or h.device != common.device
           for h in views):
        raise ValueError('Views must have the same batch size and device as common')
    common_graph = neighbor_weights(common, topk=topk, hops=1, min_sim=min_sim)
    n = len(common)
    candidates = common_graph.sum().item()
    stats = {'view_common_neighbors': candidates / n if n else 0.}
    graphs = []
    for v, h in enumerate(views, start=1):
        graph = common_graph
        if mode == 'consensus':
            local_graph = neighbor_weights(h, topk=topk, hops=1, min_sim=min_sim)
            graph = common_graph * (local_graph > 0)
        graphs.append(graph)
        counts = graph.sum(1)
        stats['view_neighbors_' + str(v)] = counts.mean().item() if n else 0.
        stats['view_active_anchor_fraction_' + str(v)] = (counts > 0).float().mean().item() if n else 0.
        stats['view_neighbor_retention_' + str(v)] = graph.sum().item() / candidates if candidates else 0.
    return graphs, stats


def view_neighbor_loss(common, views, mode='shared', topk=10, min_sim=0.5,
                       temperature=0.5):
    """Average across all views; an empty-neighbor view contributes zero.

    Only anchors with positives contribute within each view (same as MNC).
    All other non-self samples remain in the contrastive denominator.
    """
    graphs, stats = view_neighbor_weights(common, views, mode, topk, min_sim)
    losses = [neighbor_contrastive_loss(h, graph, temperature)
              for h, graph in zip(views, graphs)]
    for v, loss in enumerate(losses, start=1):
        stats['view_loss_' + str(v)] = loss.detach().item()
    return torch.stack(losses).mean(), stats
