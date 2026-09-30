import copy
import unittest

import torch
import torch.nn as nn

from local_watch import LocalWatchAgent, fuse_local_and_global_risk, take_watch_sample


def make_data(n=600, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 5, generator=g)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).float()
    return X, y


def train(model, X, y, steps=60):
    # a few plain SGD steps, the same kind of local training the banks do
    model = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(model(X).squeeze(-1), y).backward()
        opt.step()
    return model


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


class LocalWatchAgentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.X, self.y = make_data()
        self.global_model = train(Squeeze(), self.X[:100], self.y[:100], steps=5)
        wx, wy = take_watch_sample(self.X, self.y, n=200)
        self.agent = LocalWatchAgent(client_id=1, watch_X=wx, watch_y=wy)

    def test_honest_training_is_not_flagged(self):
        honest = train(self.global_model, self.X, self.y)
        report = self.agent.assess(self.global_model, honest)
        self.assertEqual(report.status, "normal")
        self.assertLess(report.loss_ratio, 1.0)

    def test_label_flip_is_caught_in_the_first_round(self):
        poisoned = train(self.global_model, self.X, 1 - self.y)
        report = self.agent.assess(self.global_model, poisoned)
        self.assertEqual(report.round_number, 1)
        self.assertEqual(report.status, "poisoned")

    def test_report_carries_no_data(self):
        report = self.agent.assess(self.global_model, self.global_model).to_dict()
        self.assertEqual(set(report), {"client_id", "round_number", "loss_ratio", "status"})

    def test_ratio_limit_must_be_above_one(self):
        with self.assertRaises(ValueError):
            LocalWatchAgent(0, self.X[:10], self.y[:10], ratio_limit=0.9)

    def test_risk_fusion_never_hides_high_risk_layer(self):
        self.assertGreaterEqual(fuse_local_and_global_risk(.8, .1), .8)
        self.assertGreaterEqual(fuse_local_and_global_risk(.1, .8), .8)


if __name__ == "__main__":
    unittest.main()
