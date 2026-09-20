"""Evaluate a run using its saved architecture and seed."""
import argparse
import json
from pathlib import Path
import torch
from network import TMCN
from dataloader import load_data
from metric import valid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--feature_mode', choices=['common', 'concat', 'complement', 'both', 'all'], default='common',
                        help='common=c; concat=[c,*hs]; complement=[c,*specs]; all=all supported modes')
    args = parser.parse_args()
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
    for mode in modes:
        metrics = valid(model, device, dataset, views, size, classes,
                        seed=config['seed'], feature_mode=mode)
        results[mode] = metrics
        report = dict(feature_mode=mode, seed=config['seed'], n_init=100,
                      feature_dim=config['high_feature_dim'] * (views + 1 if mode in ('concat', 'complement') else 1),
                      metrics=metrics)
        (folder / ('eval_' + mode + '.json')).write_text(json.dumps(report, indent=2))
    if len(results) == 2:
        print('concat - common:', {k: results['concat'][k] - results['common'][k]
                                  for k in results['common']})

