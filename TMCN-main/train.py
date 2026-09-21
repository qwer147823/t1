import torch
from network import TMCN
from complementary import decorrelation_loss, branch_diagnostics
import math
from metric import valid
import numpy as np
import argparse
import random
from loss import Loss
from mnc import neighbor_weights, neighbor_contrastive_loss, mnc_weight
from features import fusion_features, validate_alpha
import json
from pathlib import Path
from dataloader import load_data
Dataname = 'Hdigit'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5, type=float)
parser.add_argument("--learning_rate", default=0.0003, type=float)
parser.add_argument("--weight_decay", default=0., type=float)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--rec_epochs", default=200, type=int)
parser.add_argument("--fine_tune_epochs", default=100, type=int)
parser.add_argument("--low_feature_dim", default=512, type=int)
parser.add_argument("--high_feature_dim", default=128, type=int)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--run_name', default='baseline')
parser.add_argument('--lambda_mnc', type=float, default=0.0)
parser.add_argument('--mnc_hops', type=int, choices=[1, 2, 3], default=1)
parser.add_argument('--mnc_topk', type=int, default=10)
parser.add_argument('--mnc_min_sim', type=float, default=0.5)
parser.add_argument('--mnc_start', type=int, default=10)
parser.add_argument('--mnc_ramp', type=int, default=20)
parser.add_argument('--mnc_endpoint_min_sim', type=float, default=None,
                    help='Optional cosine threshold on endpoints of pure 2/3-hop neighbors')
parser.add_argument('--mnc_feature_mode', choices=['common', 'concat', 'weighted_concat'], default='common',
                    help='MNC loss space; the detached graph is always built on commonz')
parser.add_argument('--fusion_alpha', type=float, default=None,
                    help='Shared distance weight for weighted_concat MNC; default equal block weights')
parser.add_argument('--complementary', action='store_true',
                    help='Enable independent complementary heads and joint decoders')
parser.add_argument('--lambda_joint', type=float, default=1.0)
parser.add_argument('--lambda_dec', type=float, default=0.0,
                    help='Maximum weight on mean squared cross-covariance (experimental)')
parser.add_argument('--dec_start', type=int, default=20)
parser.add_argument('--dec_ramp', type=int, default=20)
parser.add_argument('--old_rec_weight', type=float, default=1.0,
                    help='Original reconstruction weight during fine-tuning only')
args = parser.parse_args()
if args.fusion_alpha is not None:
    try:
        validate_alpha(args.fusion_alpha)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mnc_feature_mode != 'weighted_concat':
        parser.error('--fusion_alpha requires --mnc_feature_mode weighted_concat')
if args.mnc_endpoint_min_sim is not None:
    if not -1 <= args.mnc_endpoint_min_sim <= 1:
        parser.error('--mnc_endpoint_min_sim must be finite and in [-1, 1]')
    if args.mnc_hops == 1:
        parser.error('--mnc_endpoint_min_sim requires --mnc_hops 2 or 3')
if args.lambda_mnc == 0 and (args.mnc_endpoint_min_sim is not None or args.mnc_feature_mode != 'common'):
    parser.error('MNC experiments require --lambda_mnc > 0')
for name in ('lambda_joint', 'lambda_dec', 'old_rec_weight', 'lambda_mnc',
             'temperature_f', 'learning_rate', 'weight_decay'):
    value = getattr(args, name)
    if not math.isfinite(value) or value < 0:
        parser.error(name + ' must be finite and nonnegative')
if args.learning_rate == 0 or args.low_feature_dim < 1 or args.high_feature_dim < 1:
    parser.error('Learning rate and feature dimensions must be positive')
if args.dec_start < 0 or args.dec_ramp < 0:
    parser.error('Decorrelation schedule must be nonnegative')
if args.complementary and args.lambda_joint <= 0:
    parser.error('--complementary requires a positive --lambda_joint')
if not args.complementary and (args.lambda_dec != 0 or args.old_rec_weight != 1 or args.lambda_joint != 1):
    parser.error('Complementary loss options require --complementary')
if args.lambda_mnc < 0 or args.mnc_topk < 1 or args.mnc_start < 0 or args.mnc_ramp < 0:
    parser.error('Invalid MNC weight, topk, or schedule')
if not -1 <= args.mnc_min_sim <= 1 or args.temperature_f <= 0:
    parser.error('Invalid similarity threshold or temperature')
if args.batch_size < 2 or args.low_feature_dim % 8:
    parser.error('batch_size >= 2 and low_feature_dim divisible by 8 required')
if args.rec_epochs < 0 or args.fine_tune_epochs < 1:
    parser.error('rec_epochs >= 0 and fine_tune_epochs >= 1 required')
if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
    parser.error('run_name must be a simple directory name')
run_dir = Path('runs') / args.run_name
run_dir.mkdir(parents=True, exist_ok=False)
(run_dir / 'config.json').write_text(json.dumps(vars(args), indent=2))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

if len(data_loader) == 0:
    raise ValueError('batch_size exceeds dataset size')

def pre_train(epoch):
    model.train()
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def fine_tune(epoch):
    model.train()
    tot_loss = 0.
    total_mnc = 0.
    total_neighbors = 0.
    components = dict(rec_old=0., ascl=0., rec_joint=0., dec_raw=0.)
    diagnostics = {}
    graph_stats = {}
    dec_weight = mnc_weight(epoch - args.rec_epochs, args.lambda_dec, args.dec_start, args.dec_ramp)
    weight = mnc_weight(epoch - args.rec_epochs, args.lambda_mnc, args.mnc_start, args.mnc_ramp)
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs = model(xs)
        commonz, S = model.TMCNF(xs)
        # AsCL remains on original hs; MNC loss space is independently selectable.
        rec_terms = [mes(x, xr) for x, xr in zip(xs, xrs)]
        align_terms = [criterion.Structure_guided_Contrastive_Loss(h, commonz, S) for h in hs]
        rec_old, ascl = sum(rec_terms), sum(align_terms)
        # Preserve original interleaved summation order for the disabled baseline.
        loss = sum(term for pair in zip(align_terms, rec_terms)
                   for term in (pair[0], args.old_rec_weight * pair[1]))
        components['rec_old'] += rec_old.item()
        components['ascl'] += ascl.item()
        if args.complementary:
            specs = model.complementary_features(zs)
            joint = model.joint_reconstruct(commonz, specs)
            rec_joint = sum(mes(x, xr) for x, xr in zip(xs, joint))
            dec = sum(decorrelation_loss(commonz, spec) for spec in specs)
            loss = loss + args.lambda_joint * rec_joint + dec_weight * dec
            components['rec_joint'] += rec_joint.item()
            components['dec_raw'] += dec.item()
            if batch_idx == 0:
                diagnostics = branch_diagnostics(model, xs, commonz, specs, joint)
        if weight > 0:
            W, batch_graph_stats = neighbor_weights(
                commonz, args.mnc_topk, args.mnc_hops, args.mnc_min_sim,
                endpoint_min_sim=args.mnc_endpoint_min_sim, return_stats=True)
            for key, value in batch_graph_stats.items():
                graph_stats[key] = graph_stats.get(key, 0.) + value
            mnc_features = fusion_features(commonz, hs, args.mnc_feature_mode, args.fusion_alpha)
            loss_mnc = neighbor_contrastive_loss(mnc_features, W, args.temperature_f)
            loss = loss + weight * loss_mnc
            total_mnc += loss_mnc.item()
            total_neighbors += (W > 0).float().sum(1).mean().item()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    stats = dict(epoch=epoch, loss=tot_loss/len(data_loader), mnc=total_mnc/len(data_loader),
                 mnc_weight=weight, neighbors=total_neighbors/len(data_loader))
    stats.update({k: v / len(data_loader) for k, v in components.items()})
    stats.update(diagnostics)
    stats.update({key: value / len(data_loader) for key, value in graph_stats.items()})
    stats.update(dec_weight=dec_weight, dec_weighted=dec_weight * stats['dec_raw'],
                 joint_weighted=args.lambda_joint * stats['rec_joint'])
    print(stats)
    with (run_dir / 'training.jsonl').open('a') as f:
        f.write(json.dumps(stats) + '\n')

model = TMCN(view, dims, args.low_feature_dim, args.high_feature_dim, device,
             complementary=args.complementary)
print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = Loss(args.batch_size, args.temperature_f, device).to(device)
epoch = 1
while epoch <= args.rec_epochs:
    pre_train(epoch)
    epoch += 1
while epoch <= args.rec_epochs + args.fine_tune_epochs:
    fine_tune(epoch)
    if epoch == args.rec_epochs + args.fine_tune_epochs:
        torch.save(model.state_dict(), run_dir / 'model.pth')
        metrics = valid(model, device, dataset, view, data_size, class_num, seed=args.seed)
        (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print('Saving model...')
    epoch += 1


