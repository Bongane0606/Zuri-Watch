"""Zuri Local Watch: a watch agent inside each bank.

Idea by Symphorose Tshibombi. Every bank runs its own agent that starts each round
from the global model and asks one question about the bank's freshly trained model:

    "Did this round of training make the model WORSE on my own correctly labelled data?"

Honest local training minimises loss on the bank's own data, so the answer for an
honest bank is always no. A bank whose training pipeline has been poisoned (flipped
labels, scaled updates) ends up with a model that does badly on its true labels, and
the agent sees that straight away, in the first round of the attack. Because the
agent compares against the global model it started from, and not against its own
history, it needs no warm-up rounds.

How it works in practice:
  * Before training starts, the agent keeps a small sample of the bank's rows with
    their true labels (a "watch sample"). It never leaves the bank.
  * Each round it measures the loss of the global model and of the new local model on
    that sample and reports only one number, the ratio between the two.
  * The server quarantines a bank whose ratio goes above the limit (1.5 by default).

Trust boundary: the agent is only useful if the attacker cannot tamper with it (for
example it runs as a separate, locked-down service). A fully compromised bank could
fake its report. That is why a report can only ever REMOVE trust from its own bank,
never add trust or accuse another bank. A lying bank simply falls back to being
judged by Zuri Guard alone, so the combined system is never weaker than Zuri Guard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LocalWatchReport:
    """The only thing that leaves the bank each round: no rows, no labels, no features."""

    client_id: int
    round_number: int
    loss_ratio: float
    status: str

    def to_dict(self):
        return asdict(self)


class LocalWatchAgent:
    """Watches one bank. Holds a private sample of that bank's correctly labelled rows."""

    def __init__(self, client_id: int, watch_X: torch.Tensor, watch_y: torch.Tensor,
                 ratio_limit: float = 1.5):
        if len(watch_X) == 0:
            raise ValueError("the watch sample needs at least one row")
        if ratio_limit <= 1.0:
            raise ValueError("ratio_limit must be above 1, honest training lowers the loss")
        self.client_id = client_id
        self.watch_X = watch_X
        self.watch_y = watch_y.float()
        self.ratio_limit = ratio_limit
        self.round_number = 0

    def _loss(self, model) -> float:
        model.eval()
        with torch.no_grad():
            return F.binary_cross_entropy_with_logits(model(self.watch_X), self.watch_y).item()

    def assess(self, global_model, local_model) -> LocalWatchReport:
        """Compare the model this bank just trained with the global model it started from."""
        self.round_number += 1
        ratio = self._loss(local_model) / max(self._loss(global_model), 1e-12)
        status = "poisoned" if ratio > self.ratio_limit else "normal"
        return LocalWatchReport(self.client_id, self.round_number, round(ratio, 4), status)


def take_watch_sample(X: torch.Tensor, y: torch.Tensor, n: int = 512, seed: int = 7):
    """Pick the bank's private watch sample before any training (and any poisoning) happens."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    idx = torch.as_tensor(idx)
    return X[idx], y[idx]


def fuse_local_and_global_risk(local_risk: float, global_risk: float) -> float:
    """Combine a bank's own report with Zuri Guard's view. Either layer alone can raise the risk,
    neither can lower what the other one found."""
    if not (0.0 <= local_risk <= 1.0 and 0.0 <= global_risk <= 1.0):
        raise ValueError("risk scores must be between 0 and 1")
    return round(1.0 - (1.0 - local_risk) * (1.0 - global_risk), 4)
