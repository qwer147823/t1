"""Start a fresh view-neighbor ablation using an existing run's configuration."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def training_command(config, run_name, mode, weight=0.01, seed=None,
                     topk=10, min_sim=0.5, temperature=0.5, start=10, ramp=20):
    """Copy training settings, override only run name/seed and new view options."""
    if not isinstance(config, dict) or not config:
        raise ValueError('Source config must be a nonempty JSON object')
    if mode not in ('off', 'shared', 'consensus'):
        raise ValueError('Mode must be off, shared, or consensus')
    if not math.isfinite(weight) or weight < 0 or (mode != 'off' and weight == 0):
        raise ValueError('View weight must be finite and positive when enabled')
    options = dict(config)
    options.update(run_name=run_name, lambda_view=0. if mode == 'off' else weight,
                   view_neighbor_mode='shared' if mode == 'off' else mode,
                   view_topk=topk, view_min_sim=min_sim, view_temperature=temperature,
                   view_start=start, view_ramp=ramp)
    if seed is not None:
        options['seed'] = seed
    command = [sys.executable, str(Path(__file__).resolve().with_name('train.py'))]
    for key, value in options.items():
        if not isinstance(key, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', key):
            raise ValueError('Invalid training option name: ' + str(key))
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append('--' + key)
        elif isinstance(value, (str, int, float)):
            command.extend(['--' + key, str(value)])
        else:
            raise ValueError('Expected a scalar training option: ' + key)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run', required=True, help='Existing run containing config.json')
    parser.add_argument('--run_name', required=True, help='New run name; train.py refuses existing runs')
    parser.add_argument('--mode', choices=['off', 'shared', 'consensus'], required=True)
    parser.add_argument('--lambda_view', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=None, help='Default: keep the source training seed')
    parser.add_argument('--view_topk', type=int, default=10)
    parser.add_argument('--view_min_sim', type=float, default=0.5)
    parser.add_argument('--view_temperature', type=float, default=0.5)
    parser.add_argument('--view_start', type=int, default=10)
    parser.add_argument('--view_ramp', type=int, default=20)
    parser.add_argument('--dry_run', action='store_true', help='Print the command without training')
    args = parser.parse_args()
    try:
        config = json.loads((Path(args.source_run) / 'config.json').read_text(encoding='utf-8-sig'))
        command = training_command(config, args.run_name, args.mode, args.lambda_view, args.seed,
                                   args.view_topk, args.view_min_sim, args.view_temperature,
                                   args.view_start, args.view_ramp)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print('Fresh training using source settings; source model.pth is NOT loaded.', flush=True)
    print(subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command), flush=True)
    if not args.dry_run:
        # List arguments and shell=False preserve paths/options safely on Windows and Linux.
        subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)


if __name__ == '__main__':
    main()
