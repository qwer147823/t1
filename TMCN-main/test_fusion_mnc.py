"""Geometry, graph filtering, gradient routing and CLI regression checks."""
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
import torch.nn.functional as F
from features import fusion_features
from metric import inference
from mnc import neighbor_weights, neighbor_contrastive_loss
from network import TMCN
from test_complementary import SyntheticViews


class FusionMNCTest(unittest.TestCase):
    def test_weighted_distance_and_equal_weight_reference(self):
        torch.manual_seed(18)
        blocks = [F.normalize(torch.randn(5, 4, dtype=torch.float64), dim=1) for _ in range(4)]
        legacy = fusion_features(blocks[0], blocks[1:])
        self.assertTrue(torch.equal(legacy, torch.cat(blocks, 1)))
        torch.testing.assert_close(fusion_features(blocks[0], blocks[1:], 'weighted_concat'), legacy / 2)
        for alpha in (0., .2, .5, 1.):
            fused = fusion_features(blocks[0], blocks[1:], 'weighted_concat', alpha)
            distances = torch.cdist(fused, fused).square()
            expected = alpha * torch.cdist(blocks[0], blocks[0]).square()
            expected += (1-alpha) / 3 * sum(torch.cdist(h, h).square() for h in blocks[1:])
            torch.testing.assert_close(distances, expected)

    def test_invalid_alpha_and_mode(self):
        c = torch.randn(3, 4)
        for alpha in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                fusion_features(c, [c], 'weighted_concat', alpha)
        with self.assertRaises(ValueError):
            fusion_features(c, [c], 'concat', .5)

    def test_endpoint_filter_keeps_one_hop_and_pure_shell_weights(self):
        angles = torch.arange(4, dtype=torch.float64) * math.pi / 6
        z = torch.stack([angles.cos(), angles.sin()], 1).requires_grad_()
        original = neighbor_weights(z, 2, 3, .8)
        distance = (torch.arange(4)[:, None] - torch.arange(4)[None, :]).abs()
        expected = (distance == 1).double() + .5 * (distance == 2) + .25 * (distance == 3)
        torch.testing.assert_close(original, expected)
        filtered, stats = neighbor_weights(z, 2, 3, .8, endpoint_min_sim=.4, return_stats=True)
        torch.testing.assert_close(filtered, expected - .25 * (distance == 3))
        self.assertTrue(torch.equal(original == 1, filtered == 1))
        self.assertTrue(torch.equal(filtered, filtered.T))
        self.assertFalse(filtered.requires_grad)
        self.assertEqual(stats['neighbors_candidate_hop3'], .5)
        self.assertEqual(stats['neighbors_kept_hop3'], 0.)
        self.assertEqual(stats['mnc_active_anchor_fraction'], 1.)
        self.assertTrue(torch.equal(original, neighbor_weights(z, 2, 3, .8, endpoint_min_sim=None)))
        self.assertTrue(torch.equal(original, neighbor_weights(z, 2, 3, .8, endpoint_min_sim=-1)))

    def test_filter_empty_and_singleton_are_finite(self):
        for n in (1, 4):
            z = torch.eye(n, requires_grad=True)
            w, stats = neighbor_weights(z, 10, 3, .9, endpoint_min_sim=.9, return_stats=True)
            loss = neighbor_contrastive_loss(z, w)
            loss.backward()
            self.assertEqual(loss.item(), 0.)
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertEqual(stats['mnc_active_anchor_fraction'], 0.)
        with self.assertRaises(ValueError):
            neighbor_weights(z, endpoint_min_sim=float('nan'))

    def test_mnc_gradient_reaches_selected_feature_blocks(self):
        for mode in ('common', 'concat', 'weighted_concat'):
            torch.manual_seed(7)
            leaves = [torch.randn(4, 5, requires_grad=True) for _ in range(3)]
            blocks = [F.normalize(x, dim=1) for x in leaves]
            w = torch.tensor([[0., 1, 0, 0], [1., 0, 0, 0], [0., 0, 0, 1], [0., 0, 1, 0]])
            fused = fusion_features(blocks[0], blocks[1:], mode, .5 if mode == 'weighted_concat' else None)
            neighbor_contrastive_loss(fused, w).backward()
            for i, leaf in enumerate(leaves):
                if mode == 'common' and i > 0:
                    self.assertIsNone(leaf.grad)
                else:
                    self.assertTrue(torch.isfinite(leaf.grad).all())
                    self.assertGreater(leaf.grad.abs().sum().item(), 0.)

    def test_inference_uses_original_hs_without_mutating_checkpoint(self):
        from torch.utils.data import DataLoader
        model = TMCN(2, [8, 6], 8, 4, 'cpu')
        loader = DataLoader(SyntheticViews(), batch_size=3)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        labels, raw = inference(loader, model, 'cpu', 2, 8, 'concat')
        weighted_labels, weighted = inference(loader, model, 'cpu', 2, 8, 'weighted_concat', .5)
        expected = raw * np.array([math.sqrt(.5)] * 4 + [.5] * 8)
        np.testing.assert_allclose(weighted, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(labels, weighted_labels)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_training_and_evaluation_cli_with_saved_alpha_and_sweep(self):
        root = Path(__file__).resolve().parent
        data = SyntheticViews()
        def load_data(_):
            return data, [8, 6], 2, 8, 2
        def invoke(script, argv):
            with patch('dataloader.load_data', load_data), patch('torch.cuda.is_available', return_value=False), \
                 patch.object(sys, 'argv', [script] + argv), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(root / script), run_name='__main__')
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                for mode in ('common', 'concat', 'weighted_concat'):
                    options = ['--run_name', mode, '--batch_size', '4', '--rec_epochs', '1',
                               '--fine_tune_epochs', '1', '--low_feature_dim', '8', '--high_feature_dim', '4',
                               '--lambda_mnc', '.05', '--mnc_hops', '3', '--mnc_min_sim', '-1',
                               '--mnc_start', '0', '--mnc_ramp', '1', '--mnc_feature_mode', mode]
                    if mode != 'concat':
                        options += ['--mnc_endpoint_min_sim', '.5']
                    if mode == 'weighted_concat':
                        options += ['--fusion_alpha', '.7', '--complementary', '--lambda_joint', '.001']
                    invoke('train.py', options)
                    run_dir = Path('runs') / mode
                    config = json.loads((run_dir / 'config.json').read_text())
                    self.assertEqual(config['mnc_feature_mode'], mode)
                    row = json.loads((run_dir / 'training.jsonl').read_text().strip())
                    self.assertTrue(all(np.isfinite(v) for v in row.values()))
                    self.assertLessEqual(row['neighbors_kept_hop3'], row['neighbors_candidate_hop3'])
                    invoke('test.py', ['--run_dir', str(run_dir), '--feature_mode', 'all'])
                    original_report = (run_dir / 'eval_concat.json').read_bytes()
                    # Simulate an old checkpoint's config with no new fields.
                    if mode == 'common':
                        for key in ('mnc_feature_mode', 'mnc_endpoint_min_sim', 'fusion_alpha'):
                            config.pop(key)
                        (run_dir / 'config.json').write_text(json.dumps(config))
                    invoke('test.py', ['--run_dir', str(run_dir), '--feature_mode', 'weighted_concat'])
                    report = json.loads((run_dir / 'eval_weighted_concat_sweep.json').read_text())
                    self.assertEqual(report[0]['fusion_alpha'], .7 if mode == 'weighted_concat' else 1/3)
                    invoke('test.py', ['--run_dir', str(run_dir), '--feature_mode', 'weighted_concat',
                                      '--fusion_alpha', '.2', '.5'])
                    sweep = json.loads((run_dir / 'eval_weighted_concat_sweep.json').read_text())
                    self.assertEqual([r['fusion_alpha'] for r in sweep], [.2, .5])
                    self.assertEqual(original_report, (run_dir / 'eval_concat.json').read_bytes())
                    self.assertTrue((run_dir / 'eval_weighted_concat_alpha_0p2.json').exists())
                    self.assertTrue((run_dir / 'eval_weighted_concat_alpha_0p5.json').exists())
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
