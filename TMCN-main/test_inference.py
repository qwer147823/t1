"""Check checkpoint-compatible feature selection without training."""
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from metric import inference
from network import TMCN


class TinyViews(Dataset):
    def __init__(self):
        self.xs = [torch.randn(5, 8), torch.randn(5, 6)]

    def __len__(self):
        return 5

    def __getitem__(self, i):
        return [x[i] for x in self.xs], i, i


class InferenceTest(unittest.TestCase):
    def test_concat_preserves_common_and_view_order_across_batches(self):
        torch.manual_seed(10)
        model = TMCN(2, [8, 6], 8, 4, torch.device('cpu'))
        data = TinyViews()
        loader = DataLoader(data, batch_size=2, shuffle=False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        labels, common = inference(loader, model, 'cpu', 2, 5)
        labels_concat, combined = inference(loader, model, 'cpu', 2, 5, 'concat')
        self.assertEqual(combined.shape, (5, 12))
        np.testing.assert_array_equal(labels, np.arange(5))
        np.testing.assert_array_equal(labels_concat, labels)
        np.testing.assert_array_equal(combined[:, :4], common)
        expected = []
        with torch.no_grad():
            for xs, _, _ in loader:
                _, _, hs = model(xs)
                expected.append(torch.cat(hs, 1).numpy())
        np.testing.assert_array_equal(combined[:, 4:], np.concatenate(expected))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_unknown_mode_fails(self):
        with self.assertRaises(ValueError):
            inference(None, None, 'cpu', 2, 5, 'typo')


if __name__ == '__main__':
    unittest.main()
