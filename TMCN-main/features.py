"""Feature assembly shared by training and evaluation; no learnable parameters."""
import math
import torch


def validate_alpha(alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('fusion_alpha must be finite and in [0, 1]')
    return alpha


def fusion_features(common, views, mode='concat', alpha=None):
    """Inputs are the model's L2-normalized c and original hs, not new specifics.

    Weighted concat gives squared-distance weights alpha and (1-alpha)/V.
    None selects equal block weights. Legacy concat is returned without scaling.
    """
    if mode not in ('common', 'concat', 'weighted_concat'):
        raise ValueError('Unknown fusion mode: ' + str(mode))
    if mode != 'weighted_concat' and alpha is not None:
        raise ValueError('fusion_alpha applies only to weighted_concat')
    if mode == 'common':
        return common
    if not views:
        raise ValueError('At least one view feature is required')
    if mode == 'concat':
        return torch.cat([common, *views], dim=1)
    alpha = validate_alpha(1.0 / (len(views) + 1) if alpha is None else alpha)
    view_scale = math.sqrt((1 - alpha) / len(views))
    return torch.cat([math.sqrt(alpha) * common,
                      *[view_scale * h for h in views]], dim=1)
