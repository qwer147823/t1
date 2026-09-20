"""Gradient, backward compatibility and train/evaluate smoke checks on synthetic data."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from network import TMCN
from complementary import decorrelation_loss, branch_diagnostics
from metric import inference
from mnc import mnc_weight


class SyntheticViews(Dataset):
    def __init__(self):
        self.xs = [torch.randn(8, 8), torch.randn(8, 6)]
    def __len__(self):
        return 8
    def __getitem__(self, i):
        return [x[i] for x in self.xs], i % 2, i


class ComplementaryTest(unittest.TestCase):
    def model(self, enabled=True):
        return TMCN(2, [8, 6], 8, 4, 'cpu', complementary=enabled).to('cpu')

    def test_decorrelation_reference_and_gradient(self):
        c = torch.tensor([[1., 2.], [3., 0.], [2., 4.]], requires_grad=True)
        s = torch.tensor([[2., 1.], [0., 2.], [4., 0.]], requires_grad=True)
        cov = np.cov(c.detach().numpy().T, s.detach().numpy().T, ddof=1)[:2, 2:]
        loss = decorrelation_loss(c, s)
        self.assertAlmostEqual(loss.item(), float((cov ** 2).mean()), places=6)
        loss.backward()
        self.assertIsNone(c.grad)
        self.assertGreater(s.grad.abs().sum().item(), 0)
        one = torch.randn(1, 4, requires_grad=True)
        decorrelation_loss(torch.randn(1, 4), one).backward()
        self.assertTrue(torch.equal(one.grad, torch.zeros_like(one)))
        self.assertEqual(mnc_weight(20, 100, 20, 20), 0)
        self.assertEqual(mnc_weight(30, 100, 20, 20), 50)
        self.assertEqual(mnc_weight(40, 100, 20, 20), 100)

    def test_initialization_preserves_backbone_and_rng(self):
        torch.manual_seed(10)
        base = self.model(False)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(10)
        extended = self.model()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, tensor in base.state_dict().items():
            self.assertTrue(torch.equal(tensor, extended.state_dict()[key]), key)
        old_copy = self.model(False)
        old_copy.load_state_dict(base.state_dict(), strict=True)
        self.assertNotEqual(extended.complement_heads[0].weight.data_ptr(),
                            extended.complement_heads[1].weight.data_ptr())

    def test_joint_and_dec_gradient_routes(self):
        model = self.model()
        xs = [torch.randn(4, 8), torch.randn(4, 6)]
        _, zs, _ = model(xs)
        c, _ = model.TMCNF(xs)
        specs = model.complementary_features(zs)
        joint = model.joint_reconstruct(c, specs)
        loss = sum((x-y).square().mean() for x,y in zip(xs,joint))
        loss += 100 * sum(decorrelation_loss(c, s) for s in specs)
        loss.backward()
        for name in ('encoders', 'MambaEncoder', 'Common_view', 'complement_heads', 'joint_decoders'):
            grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, name)
        self.assertTrue(all(p.grad is None for p in model.Specific_view.parameters()))
        stats = branch_diagnostics(model, xs, c, specs, joint)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))

    def test_feature_order_and_legacy_guard(self):
        model = self.model()
        data = SyntheticViews()
        loader = DataLoader(data, batch_size=3, shuffle=False)
        labels, common = inference(loader, model, 'cpu', 2, 8, 'common')
        _, combined = inference(loader, model, 'cpu', 2, 8, 'complement')
        np.testing.assert_array_equal(common, combined[:, :4])
        expected = []
        with torch.no_grad():
            for xs, _, _ in loader:
                _, zs, _ = model(xs)
                expected.append(torch.cat(model.complementary_features(zs), 1).numpy())
        np.testing.assert_array_equal(combined[:, 4:], np.concatenate(expected))
        self.assertEqual(combined.shape, (8, 12))
        np.testing.assert_array_equal(labels, np.arange(8) % 2)
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            inference(loader, self.model(False), 'cpu', 2, 8, 'complement')

    def test_synthetic_train_checkpoint_and_all_evaluations(self):
        # Exercise real CLI entrypoints without loading Hdigit or using labels in training.
        root = Path(__file__).resolve().parent
        data = SyntheticViews()
        def load_data(_):
            return data, [8, 6], 2, 8, 2
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                for enabled, dec, old_rec in ((False, 0, 1), (True, 0, 1), (True, 100, 0)):
                    name = 'run_' + str(enabled) + '_' + str(dec)
                    argv = ['train.py', '--run_name', name, '--batch_size', '4',
                            '--rec_epochs', '1', '--fine_tune_epochs', '2',
                            '--low_feature_dim', '8', '--high_feature_dim', '4',
                            '--lambda_mnc', '0.05', '--mnc_hops', '3',
                            '--mnc_min_sim', '-1', '--mnc_start', '0', '--mnc_ramp', '1']
                    if enabled:
                        argv += ['--complementary', '--lambda_dec', str(dec),
                                 '--old_rec_weight', str(old_rec), '--dec_start', '0', '--dec_ramp', '1']
                    with patch('dataloader.load_data', load_data), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(str(root/'train.py'), run_name='__main__')
                    report_dir = Path('runs') / name
                    rows = [json.loads(line) for line in (report_dir/'training.jsonl').read_text().splitlines()]
                    self.assertEqual(len(rows), 2)
                    self.assertTrue(all(np.isfinite(v) for row in rows for v in row.values()))
                    self.assertEqual(rows[-1]['dec_weight'], dec)
                    # Config without the new flag simulates pre-change run metadata.
                    if not enabled:
                        config = json.loads((report_dir/'config.json').read_text())
                        config.pop('complementary')
                        (report_dir/'config.json').write_text(json.dumps(config))
                    argv = ['test.py', '--run_dir', str(report_dir), '--feature_mode', 'all']
                    with patch('dataloader.load_data', load_data), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(str(root/'test.py'), run_name='__main__')
                    for mode in ['common', 'concat'] + (['complement'] if enabled else []):
                        report = json.loads((report_dir/('eval_'+mode+'.json')).read_text())
                        self.assertEqual(report['feature_dim'], 4 if mode == 'common' else 12)
                        self.assertTrue(all(np.isfinite(v) for v in report['metrics'].values()))
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
