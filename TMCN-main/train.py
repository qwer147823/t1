import torch
from network import TMCN
from metric import valid
import numpy as np
import argparse
import random
from loss import Loss
from mnc import neighbor_weights, neighbor_contrastive_loss, mnc_weight
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
args = parser.parse_args()
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
    weight = mnc_weight(epoch - args.rec_epochs, args.lambda_mnc, args.mnc_start, args.mnc_ramp)
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, hs = model(xs)
        commonz, S = model.TMCNF(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion.Structure_guided_Contrastive_Loss(hs[v], commonz, S))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        if weight > 0:
            W = neighbor_weights(commonz, args.mnc_topk, args.mnc_hops, args.mnc_min_sim)
            loss_mnc = neighbor_contrastive_loss(commonz, W, args.temperature_f)
            loss = loss + weight * loss_mnc
            total_mnc += loss_mnc.item()
            total_neighbors += (W > 0).float().sum(1).mean().item()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    stats = dict(epoch=epoch, loss=tot_loss/len(data_loader), mnc=total_mnc/len(data_loader),
                 mnc_weight=weight, neighbors=total_neighbors/len(data_loader))
    print(stats)
    with (run_dir / 'training.jsonl').open('a') as f:
        f.write(json.dumps(stats) + '\n')

model = TMCN(view, dims, args.low_feature_dim, args.high_feature_dim, device)
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
        metrics = valid(model, device, dataset, view, data_size, class_num, seed=args.seed)
        (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        state = model.state_dict()
        torch.save(state, run_dir / 'model.pth')
        print('Saving model...')
    epoch += 1


