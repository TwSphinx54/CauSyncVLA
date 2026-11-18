"""CPU contract checks using no checkpoint or simulator."""
import unittest
from pathlib import Path
import torch
from openpi.dual.config import load_config
from openpi.dual.history import condition_features
from openpi.dual.model import UpdaterNorm
from openpi.dual.rose_tfws import resolve_initial_latent, shift_repeat_endpoint


class ControllerContracts(unittest.TestCase):
    def test_shift_matches_consumed_nodes(self):
        endpoint = torch.arange(10.).view(1, 10, 1)
        shifted = shift_repeat_endpoint(endpoint, 5)
        self.assertEqual(shifted.flatten().tolist(), [5, 6, 7, 8, 9, 9, 9, 9, 9, 9])
        noise = torch.ones_like(endpoint)
        warm, time, used = resolve_initial_latent(noise, endpoint, enabled=True, replan_steps=5)
        torch.testing.assert_close(warm, .5 * shifted + .5 * noise)
        self.assertEqual((time, used), (.5, True))

    def test_history_ignores_unobserved_padding(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/libero.json')
        h = cfg.updater.history
        anchor, measured = torch.zeros(1, 8), torch.ones(1, 8)
        states, actions = torch.full((1, h, 8), float('nan')), torch.full((1, h, 7), float('nan'))
        states[:, 0] = measured
        actions[:, 0] = 2.
        norm = UpdaterNorm(torch.ones(2,18), torch.ones(2,18), torch.zeros(8), torch.ones(8))
        feature = condition_features(anchor, measured, states, actions, torch.tensor([1]), norm, h)
        self.assertEqual(tuple(feature.shape), (1, 745))
        self.assertTrue(torch.isfinite(feature).all())
        torch.testing.assert_close(feature[0, -7*h:-7*(h-1)], actions[0, 0])
        self.assertEqual(feature[0, -7*(h-1):].count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
