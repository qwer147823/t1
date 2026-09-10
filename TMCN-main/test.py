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
    args = parser.parse_args()
    folder = Path(args.run_dir)
    config = json.loads((folder / 'config.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, dims, views, size, classes = load_data(config['dataset'])
    model = TMCN(views, dims, config['low_feature_dim'], config['high_feature_dim'], device).to(device)
    model.load_state_dict(torch.load(folder / 'model.pth', map_location=device, weights_only=True))
    valid(model, device, dataset, views, size, classes, seed=config['seed'])
