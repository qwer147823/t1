"""Tests for per-view graphs, gradients, legacy inference and ablation entrypoints."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from metric import inference
from network import TMCN
from run_view_ablation import training_command
from test_complementary import SyntheticViews
from view_neighbors import view_neighbor_weights, view_neighbor_loss


def points(degrees):
    angles = torch.tensor(degrees, dtype=torch.float64) * math.pi / 180
    return torch.stack([angles.cos(), angles.sin()], 1).requires_grad_()


class ViewNeighborTest(unittest.TestCase):
    def test_shared_and_consensus_graphs_have_known_edges(self):
        c = points([0, 10, 80, 90])
        views = [points([0, 12, 80, -80]), points([0, 80, 10, 90]), c.clone()]
        expected = torch.tensor([[0., 1, 0, 0], [1., 0, 0, 0],
                                 [0., 0, 0, 1], [0., 0, 1, 0]], dtype=torch.float64)
        shared, stats = view_neighbor_weights(c, views, 'shared', 1, .9)
        for graph in shared:
            torch.testing.assert_close(graph, expected)
            self.assertFalse(graph.requires_grad)
        self.assertEqual(stats['view_common_neighbors'], 1.)
        consensus, stats = view_neighbor_weights(c, views, 'consensus', 1, .9)
        first = expected.clone()
        first[2:, 2:] = 0
        torch.testing.assert_close(consensus[0], first)
        torch.testing.assert_close(consensus[1], torch.zeros_like(expected))
        torch.testing.assert_close(consensus[2], expected)
        self.assertEqual(stats['view_active_anchor_fraction_1'], .5)
        self.assertEqual(stats['view_neighbor_retention_1'], .5)
        self.assertEqual(stats['view_active_anchor_fraction_2'], 0.)
        for graph in consensus:
            self.assertFalse(graph.requires_grad)
            self.assertTrue(torch.equal(graph, graph.T))
            self.assertEqual(graph.diag().sum().item(), 0.)

    def test_average_over_all_views_with_analytic_uniform_loss(self):
        c = torch.ones(4, 2, requires_grad=True)
        good = torch.ones(4, 2, requires_grad=True)
        empty = torch.eye(4, requires_grad=True)
        loss, stats = view_neighbor_loss(c, [good, empty], 'consensus', 3, .9)
        # Every non-self logit is equal in the first view: each probability is 1/3.
        # The empty second view contributes zero, but still counts in the average.
        self.assertAlmostEqual(loss.item(), math.log(3) / 2, places=6)
        self.assertAlmostEqual(stats['view_loss_1'], math.log(3), places=6)
        self.assertEqual(stats['view_loss_2'], 0.)
        loss.backward()
        self.assertIsNone(c.grad)
        self.assertTrue(torch.equal(empty.grad, torch.zeros_like(empty)))

    def test_feature_gradients_leave_graph_source_detached(self):
        for mode in ('shared', 'consensus'):
            c = points([0, 10, 80, 90])
            views = [points([0, 12, 79, 93]), points([-3, 11, 78, 92])]
            loss, _ = view_neighbor_loss(c, views, mode, 1, .9)
            loss.backward()
            self.assertIsNone(c.grad)
            for h in views:
                self.assertTrue(torch.isfinite(h.grad).all())
                self.assertGreater(h.grad.abs().sum().item(), 0.)

    def test_model_gradients_reach_view_projection_and_encoders(self):
        torch.manual_seed(31)
        model = TMCN(2, [8, 6], 8, 4, 'cpu', complementary=True)
        xs = [torch.randn(4, 8), torch.randn(4, 6)]
        _, _, hs = model(xs)
        c, _ = model.TMCNF(xs)
        loss, _ = view_neighbor_loss(c, hs, topk=1, min_sim=-1)
        loss.backward()
        for module in [*model.encoders, model.Specific_view]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.)
        for name in ('MambaEncoder', 'Common_view', 'decoders', 'complement_heads', 'joint_decoders'):
            self.assertTrue(all(p.grad is None for p in getattr(model, name).parameters()), name)

    def test_empty_and_singleton_batches_are_finite(self):
        for n in (1, 4):
            for mode in ('shared', 'consensus'):
                c = torch.eye(n, requires_grad=True)
                views = [torch.randn(n, 3, requires_grad=True) for _ in range(3)]
                loss, stats = view_neighbor_loss(c, views, mode, min_sim=.9)
                self.assertEqual(loss.item(), 0.)
                self.assertTrue(all(np.isfinite(value) for value in stats.values()))
                loss.backward()
                self.assertIsNone(c.grad)
                for h in views:
                    self.assertTrue(torch.equal(h.grad, torch.zeros_like(h)))

    def test_invalid_graph_inputs(self):
        c = torch.randn(4, 3)
        for options in ({'mode': 'unknown'}, {'topk': 0}, {'min_sim': float('nan')}):
            with self.assertRaises(ValueError):
                view_neighbor_weights(c, [c], **options)
        for views in ([], [torch.randn(3, 3)]):
            with self.assertRaises(ValueError):
                view_neighbor_weights(c, views)

    def test_single_view_inference_uses_hs_and_preserves_parameters(self):
        torch.manual_seed(4)
        model = TMCN(2, [8, 6], 8, 4, 'cpu', complementary=True)
        loader = DataLoader(SyntheticViews(), batch_size=3)
        before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        labels, combined = inference(loader, model, 'cpu', 2, 8, 'concat')
        with patch.object(model, 'TMCNF', side_effect=AssertionError('Unused in single-view mode')):
            for v in range(2):
                actual_labels, features = inference(loader, model, 'cpu', 2, 8, 'view', view_index=v)
                np.testing.assert_array_equal(actual_labels, labels)
                np.testing.assert_array_equal(features, combined[:, (v+1)*4:(v+2)*4])
        self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))
        for index in (None, -1, 2):
            with self.assertRaises(ValueError):
                inference(loader, model, 'cpu', 2, 8, 'view', view_index=index)

    def test_ablation_command_keeps_source_options_without_mutation(self):
        config = {'seed': 10, 'run_name': 'old', 'complementary': True,
                  'lambda_joint': .001, 'lambda_dec': 1000, 'mnc_hops': 3,
                  'lambda_mnc': .05, 'mnc_feature_mode': 'common', 'fusion_alpha': None}
        before = dict(config)
        command = training_command(config, 'new', 'consensus', seed=20)
        self.assertEqual(config, before)
        self.assertIn('--complementary', command)
        self.assertNotIn('--fusion_alpha', command)
        for key, value in {'seed': '20', 'run_name': 'new', 'lambda_joint': '0.001',
                           'lambda_dec': '1000', 'mnc_hops': '3', 'lambda_mnc': '0.05',
                           'view_neighbor_mode': 'consensus', 'lambda_view': '0.01'}.items():
            self.assertEqual(command[command.index('--' + key) + 1], value)
        for weight in (-1, float('nan'), float('inf'), 0):
            with self.assertRaises(ValueError):
                training_command(config, 'new', 'shared', weight)

    def test_invalid_cli_options_fail_before_creating_a_run(self):
        root = Path(__file__).resolve().parent
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                for options in (['--lambda_view', 'nan'], ['--view_temperature', '0'],
                                ['--view_topk', '0'], ['--view_min_sim', 'nan'], ['--view_start', '-1']):
                    with patch.object(sys, 'argv', ['train.py'] + options), \
                         contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                        runpy.run_path(str(root / 'train.py'), run_name='__main__')
                    self.assertEqual(exc.exception.code, 2)
                    self.assertFalse(Path('runs').exists())
            finally:
                os.chdir(previous)

    def test_train_save_reload_views_and_weighted_evaluation(self):
        root = Path(__file__).resolve().parent
        data = SyntheticViews()
        def invoke(script, options):
            with patch('dataloader.load_data', return_value=(data, [8, 6], 2, 8, 2)), \
                 patch('torch.cuda.is_available', return_value=False), \
                 patch.object(sys, 'argv', [script] + options), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(root / script), run_name='__main__')
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                for mode in ('off', 'shared', 'consensus'):
                    options = ['--run_name', mode, '--batch_size', '4', '--rec_epochs', '1',
                               '--fine_tune_epochs', '3', '--low_feature_dim', '8', '--high_feature_dim', '4',
                               '--lambda_mnc', '.05', '--mnc_hops', '3', '--mnc_min_sim', '-1',
                               '--mnc_start', '0', '--mnc_ramp', '1', '--complementary',
                               '--lambda_joint', '.001', '--lambda_dec', '1000', '--dec_start', '0',
                               '--dec_ramp', '1', '--view_start', '1', '--view_ramp', '2',
                               '--view_topk', '3', '--view_min_sim', '-1']
                    if mode != 'off':
                        options += ['--lambda_view', '.01', '--view_neighbor_mode', mode]
                    with patch('view_neighbors.view_neighbor_loss', wraps=view_neighbor_loss) as view_loss:
                        invoke('train.py', options)
                        self.assertEqual(view_loss.call_count, 0 if mode == 'off' else 4)
                    folder_run = Path('runs') / mode
                    rows = [json.loads(line) for line in (folder_run / 'training.jsonl').read_text().splitlines()]
                    self.assertEqual([r['view_weight'] for r in rows], [0., 0., 0.] if mode == 'off' else [0., .005, .01])
                    self.assertTrue(all(np.isfinite(v) for row in rows for v in row.values()))
                    if mode != 'off':
                        for row in rows[1:]:
                            self.assertAlmostEqual(row['view_loss'], (row['view_loss_1'] + row['view_loss_2']) / 2, places=6)
                            self.assertAlmostEqual(row['view_weighted'], row['view_loss'] * row['view_weight'])
                            self.assertEqual(row['view_active_anchor_fraction_1'], 1.)
                    config_file = folder_run / 'config.json'
                    config = json.loads(config_file.read_text())
                    if mode == 'off':
                        # Existing checkpoints did not save any of the new training options.
                        config = {k: v for k, v in config.items() if k != 'lambda_view' and not k.startswith('view_')}
                        config_file.write_text(json.dumps(config))
                    checkpoint = (folder_run / 'model.pth').read_bytes()
                    invoke('test.py', ['--run_dir', str(folder_run), '--feature_mode', 'all'])
                    concat_report = (folder_run / 'eval_concat.json').read_bytes()
                    invoke('test.py', ['--run_dir', str(folder_run), '--feature_mode', 'weighted_concat', '--fusion_alpha', '.2'])
                    invoke('test.py', ['--run_dir', str(folder_run), '--feature_mode', 'views'])
                    reports = json.loads((folder_run / 'eval_views.json').read_text())
                    self.assertEqual([r['view_number'] for r in reports], [1, 2])
                    self.assertTrue(all(r['feature_dim'] == 4 and r['n_init'] == 100 for r in reports))
                    for v in (1, 2):
                        self.assertTrue((folder_run / ('eval_view_' + str(v) + '.json')).exists())
                    self.assertEqual(concat_report, (folder_run / 'eval_concat.json').read_bytes())
                    self.assertEqual(checkpoint, (folder_run / 'model.pth').read_bytes())
                with patch('subprocess.run') as launch:
                    invoke('run_view_ablation.py', ['--source_run', 'runs/off', '--run_name', 'preview',
                                                   '--mode', 'shared', '--dry_run'])
                    launch.assert_not_called()
                    self.assertFalse(Path('runs/preview').exists())
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
