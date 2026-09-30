# Zuri Guard: Catching a Poisoned Bank Without Silencing the Honest Outliers

### Step-normalised federated aggregation for skewed banks, plus magnitude-based quarantine of malicious updates

**Team:** Zuri Watch (WeThinkCode_) &nbsp;|&nbsp; **Tracks:** 🔴 Advanced (primary) and 🟡 Intermediate

## Headline results (same held-out NSL-KDD test split, `evaluate()` unchanged)

| Track | Before | After | Change |
|---|---|---|---|
| 🔴 **Advanced** (bank 1, scale x15), seed 42 | FedAvg under attack **0.0511** | **Zuri Guard 0.7760** | **+0.7249** |
| 🔴 Advanced, mean of 5 seeds | 0.2461 | 0.7639 | +0.5178 |
| 🟡 **Intermediate** (non-IID, no attack), seed 42 | naive FedAvg **0.7117** | **ours 0.7821** | **+0.0704** |
| 🟡 Intermediate, mean of 5 seeds | 0.7147 | 0.7787 | +0.0640 |

`model_scripted.pt` in our repo is the seed 42 Zuri Guard model (F1 0.776). For reference, the
coordinate-median baseline scores 0.6855 on the same attack, and removing the attacker by hand
(the best any detect-and-exclude defense can do) scores 0.764 over 5 seeds.

FedAvg under attack swings between 0.00 and 0.40 from round to round, so the 5-seed means are the fairer picture.

## 1. Intermediate: why FedAvg breaks, and our fix

**Diagnosis first.** The banks differ in two ways. Banks 0 to 2 are 68 to 86% attack traffic and
banks 3 and 4 are mostly normal. The banks also differ a lot in *size*: bank 2 holds 48k rows, bank 0 only 6k. With
batch size 256 and one local epoch, bank 2 takes **189 SGD steps** per round and bank 0 takes **25**.

FedAvg gives bank 2 43% of the vote, and bank 2's update is also the longest because it took 7x more
steps. The effects multiply, so the global model mostly becomes bank 2's model. FedNova (Wang et al., 2020)
calls this *objective inconsistency*.

We first tried the notebook's suggestion of re-weighting banks so the pooled label mix is 50/50.
It barely helped (0.7147 to 0.7180), because the pool is already close to balanced. So label balance
was not the real problem here.

**Our aggregation:**
1. Divide each bank's update by its number of local steps, so we average *directions per step*, not distances that grow with dataset size.
2. Give every bank **one equal vote**.
3. Rescale by the average step count so the global model still moves a normal-sized step.

**Ablation (5 seeds, mean F1):**

| Method | F1 |
|---|---|
| IID reference, FedAvg | 0.7556 |
| Non-IID, naive FedAvg | 0.7147 |
| Label-balanced weights | 0.7180 |
| Equal weights only | 0.7491 |
| Step-normalised only | 0.7476 |
| **Both (ours)** | **0.7787** |
| Ours on IID data | 0.7556 (identical to FedAvg) |

Each idea closes most of the gap alone; together they pass the IID reference. On IID data our method
equals FedAvg exactly (all banks are the same size), so the gain comes from correcting skew, not a hidden
"bigger step" effect.

**What changed in behaviour:** recall rises from 0.57 to 0.68. Probe attacks go from 58% caught to
97%, and the rare R2L/U2R family (mostly attack types never seen in training) from 2.5% to 18.7%.
These patterns live mainly in the small attack-heavy banks FedAvg was drowning out. The cost: false alarms on
normal traffic rise from 3.7% to 8.1% (precision 0.95 to 0.92). We think that trade is worth it for intrusion detection, but it is a real trade.

**A security bonus:** FedAvg's weights come from `sizes`, a number each bank *reports about itself*.
A malicious bank can claim a million rows and take over the average. With one bank one vote, it can't.

## 2. Advanced: Zuri Guard

### What did not work, and why that mattered

Our first defense scored each bank by how well its update *pointed* the same way as the
coordinate-wise median update (the FLTrust / Krum idea), with a reputation across rounds. It reached
about 0.69 (3 seeds), no better than the median baseline, and it **flagged honest banks on clean
runs**. Late in training, the honest attack-heavy banks (0 and 1) point *away* from the median
because they pull towards "attack" while the normal-heavy banks pull towards "normal".

That was the key lesson: **under non-IID data, an honest bank with unusual customers looks like an
attacker if you only look at direction.** Banning it would silence the exact minority traffic the
Intermediate track taught us to protect.

### What does work: magnitude

We measured it. Across 5 clean runs, no honest bank's raw update was ever more than **1.72x** the
median bank's. The scale attack starts at about **9x** and grows past 70x as training goes on. Magnitude separates attackers
from honest outliers where direction cannot.

### The defense, each round

1. **Measure** each bank's update size relative to the median bank. The median stays honest while fewer than half the banks are malicious.
2. **Accumulate suspicion**: add `log2(ratio / 2.5)` whenever a bank is above 2.5x. The 2.5x line leaves a margin over the 1.72x honest maximum.
3. **Quarantine permanently** once suspicion passes 1.0. A compromised bank can't behave for one round and slip back in. Every round's ratios, suspicion and quarantine list are logged, so operators get an audit trail, not just a better number.
4. **Aggregate** the trusted banks with our Intermediate method, after **clipping** each per-step update to 3x the median trusted bank. The clip covers attackers that stay under the detection line, since the most they can do is a small, bounded push.

In the official scenario, bank 1 is quarantined in **round 1** (ratio 8.7x) and never comes back.
F1 reaches 0.76 by round 2 and holds.

### Stress test (5 seeds each, mean F1)

A defense tuned to one attack isn't worth much, so we changed who attacks, how hard, and when:

| Scenario | FedAvg | Median | Multi-Krum | **Zuri Guard** |
|---|---|---|---|---|
| Clean (no attack) | 0.715 | 0.729 | 0.720 | **0.779** |
| Scale x15, bank 1 (official) | 0.246 | 0.680 | 0.697 | **0.764** |
| Scale x3, bank 1 (stealthy) | 0.659 | 0.678 | 0.697 | **0.762** |
| Scale x50, bank 1 | 0.533 | 0.680 | 0.697 | **0.764** |
| Label flip only, bank 1 | 0.688 | 0.677 | 0.708 | **0.722** |
| Scale x15, bank 2 (largest) | 0.515 | 0.776 | 0.719 | **0.783** |
| Scale x15, bank 4 | 0.677 | **0.800** | 0.789 | 0.791 |
| Scale x15, banks 1 and 3 | 0.198 | 0.746 | 0.736 | **0.774** |
| Scale x15, bank 1 from round 4 | 0.172 | 0.696 | 0.712 | **0.764** |

**Detection:** across all 45 runs, Zuri Guard caught every scaled attacker, including both attackers
at once and the bank that turned malicious partway through, with **zero false positives**. On clean
data it quarantines nobody, so it scores exactly the same as our Intermediate aggregation: the security layer costs no
accuracy when nobody is attacking. Under the official attack it matches the hand-removed oracle (0.764).

### Honest limitations

* **Zuri Guard alone misses pure label flipping** (no size anomaly). We closed this with **Zuri Local Watch**
  (Symphorose's idea, in the repo with its own evaluation script): each bank runs an agent that starts from the
  global model and keeps a private sample of its correctly labelled rows. Assuming the agent is protected, if training made the model *worse* on
  them (loss ratio above 1.5), the bank is quarantined. Honest banks never exceeded 0.83; attackers were always
  2.3 or more. Over 10 scenarios x 5 seeds it caught every attacker in its first attack round with zero false
  alarms, lifting label flip on bank 1 from 0.722 to 0.764. A report can only remove trust from its own bank,
  so a bank faking its report is judged by Zuri Guard alone (we tested this: never worse than Zuri Guard).
* When bank 4 is the attacker, coordinate-median is slightly better (0.800 vs 0.791).
* An attacker that knows the 2.5x rule can stay just under it. Our x3 test shows it still gets caught
  (suspicion builds over rounds), and clipping bounds anything smaller, but a patient adaptive attacker is the next thing to test.
* The threshold assumes fewer than half the banks are malicious, because it compares against the median.

## 3. Reproducibility

The repo has the fully run notebook, `model_scripted.pt`, `submission.json`, `results.json` (every number here) and setup
steps. Harness changes: we reseed before every run (the original seeds once, so results depended on run order), and added an
optional `attack_start` argument (default is original behaviour). `evaluate()`, model, data and test split are untouched. Full run: about 8 minutes on CPU.

## What we take away

Under non-IID data, treating "different" as "dangerous" excludes honest outliers. Use signals attackers need but honest heterogeneity doesn't produce.
