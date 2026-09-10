import unittest
import torch
from mnc import pure_hops, neighbor_weights, neighbor_contrastive_loss, mnc_weight
from network import TMCN
from loss import Loss

class MNCTest(unittest.TestCase):
    def test_chain_shortest_paths(self):
        a = torch.zeros(4, 4, dtype=torch.bool)
        for i in range(3):
            a[i, i+1] = a[i+1, i] = True
        expected = (torch.arange(4)[:, None] - torch.arange(4)[None, :]).abs()
        for i, shell in enumerate(pure_hops(a, 3)):
            self.assertTrue(torch.equal(shell, expected == i+1))

    def test_mutual_no_self_detached(self):
        z = torch.tensor([[1., 0.], [.99, .01], [-1., 0.]], requires_grad=True)
        w = neighbor_weights(z, topk=1, hops=3, min_sim=.5)
        self.assertFalse(w.requires_grad)
        self.assertEqual(w[0, 1].item(), 1)
        self.assertEqual(w.sum().item(), 2)

    def test_empty_and_singleton(self):
        for n in (1, 3):
            z = torch.randn(n, 4, requires_grad=True)
            loss = neighbor_contrastive_loss(z, torch.zeros(n, n))
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertTrue(torch.isfinite(z.grad).all())

    def test_probability_reference(self):
        z = torch.eye(3, requires_grad=True)
        w = torch.ones(3, 3) - torch.eye(3)
        loss = neighbor_contrastive_loss(z, w)
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.)).item(), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_small_batch_and_schedule(self):
        w = neighbor_weights(torch.randn(2, 4), topk=10)
        self.assertEqual(tuple(w.shape), (2, 2))
        self.assertEqual(mnc_weight(10, .1), 0)
        self.assertAlmostEqual(mnc_weight(20, .1), .05)
        self.assertEqual(mnc_weight(100, .1), .1)

    def test_tmcn_combined_backward_cpu(self):
        torch.manual_seed(1)
        model = TMCN(2, [8, 6], 8, 4, torch.device('cpu'))
        xs = [torch.randn(4, 8), torch.randn(4, 6)]
        xrs, _, hs = model(xs)
        z, s = model.TMCNF(xs)
        criterion = Loss(4, .5, torch.device('cpu'))
        loss = sum((xr-x).square().mean() for xr, x in zip(xrs, xs))
        loss += sum(criterion.Structure_guided_Contrastive_Loss(h, z, s) for h in hs)
        loss += .1 * neighbor_contrastive_loss(z, neighbor_weights(z, 2, 3, -1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

if __name__ == '__main__':
    unittest.main()
