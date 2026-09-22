"""Evaluate a run using its saved architecture and seed."""
import argparse
import json
from pathlib import Path
import torch
from network import TMCN
from dataloader import load_data
from metric import valid
from features import validate_alpha

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--feature_mode', choices=['common', 'concat', 'weighted_concat', 'complement', 'both', 'all', 'views'], default='common',
                        help='views evaluates each original h separately; all keeps its legacy modes')
    parser.add_argument('--fusion_alpha', type=float, nargs='+', default=None,
                        help='One or more shared distance weights, only for weighted_concat')
    args = parser.parse_args()
    if args.fusion_alpha is not None:
        if args.feature_mode != 'weighted_concat':
            parser.error('--fusion_alpha requires --feature_mode weighted_concat')
        try:
            for alpha in args.fusion_alpha:
                validate_alpha(alpha)
        except ValueError as exc:
            parser.error(str(exc))
    folder = Path(args.run_dir)
    config = json.loads((folder / 'config.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, dims, views, size, classes = load_data(config['dataset'])
    model = TMCN(views, dims, config['low_feature_dim'], config['high_feature_dim'], device,
                 complementary=config.get('complementary', False)).to(device)
    model.load_state_dict(torch.load(folder / 'model.pth', map_location=device, weights_only=True))
    modes = ['common', 'concat'] if args.feature_mode == 'both' else [args.feature_mode]
    if args.feature_mode == 'all':
        modes = ['common', 'concat'] + (['complement'] if model.complementary else [])
    if 'complement' in modes and not model.complementary:
        parser.error('This checkpoint has no complementary branch; use common/concat/both/all')
    results = {}
    evaluations = [(mode, None, None) for mode in modes]
    if args.feature_mode == 'views':
        evaluations = [('view', None, v) for v in range(views)]
    if args.feature_mode == 'weighted_concat':
        # Follow the trained weighted geometry unless explicitly overridden.
        saved_alpha = config.get('fusion_alpha') if config.get('mnc_feature_mode') == 'weighted_concat' else None
        alphas = args.fusion_alpha if args.fusion_alpha is not None else [saved_alpha]
        alphas = list(dict.fromkeys(1.0 / (views + 1) if a is None else a for a in alphas))
        try:
            for alpha in alphas:
                validate_alpha(alpha)
        except ValueError as exc:
            parser.error(str(exc))
        evaluations = [('weighted_concat', alpha, None) for alpha in alphas]
    sweep = []
    view_reports = []
    for mode, alpha, view_index in evaluations:
        metrics = valid(model, device, dataset, views, size, classes,
                        seed=config['seed'], feature_mode=mode, fusion_alpha=alpha,
                        view_index=view_index)
        key = mode if alpha is None else mode + '_alpha_' + str(alpha).replace('.', 'p')
        if mode == 'view':
            key = 'view_' + str(view_index + 1)
        results[key] = metrics
        report = dict(feature_mode=mode, seed=config['seed'], n_init=100,
                      feature_dim=config['high_feature_dim'] * (1 if mode in ('common', 'view') else views + 1),
                      fusion_alpha=alpha,
                      mnc_feature_mode=config.get('mnc_feature_mode', 'common'),
                      training_fusion_alpha=config.get('fusion_alpha'),
                      lambda_view=config.get('lambda_view', 0.0),
                      view_neighbor_mode=config.get('view_neighbor_mode', 'shared'),
                      metrics=metrics)
        if mode == 'view':
            report['view_number'] = view_index + 1  # h1 is hs[0]; reports are one-based.
            view_reports.append(report)
        output = folder / ('eval_' + key + '.json')
        output.write_text(json.dumps(report, indent=2))
        print('Saved evaluation:', output)
        if mode == 'weighted_concat':
            sweep.append(report)
    if sweep:
        (folder / 'eval_weighted_concat_sweep.json').write_text(json.dumps(sweep, indent=2))
    if view_reports:
        (folder / 'eval_views.json').write_text(json.dumps(view_reports, indent=2))
    if 'common' in results and 'concat' in results:
        print('concat - common:', {k: results['concat'][k] - results['common'][k]
                                  for k in results['common']})
