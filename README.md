# Zuri Watch: Secure Federated Intrusion Detection

CAIRLab Secure AI Hackathon 2026, Days 2 and 3. Team **Zuri Watch** (WeThinkCode_).

Five banks train one intrusion detector on NSL-KDD without pooling their data. We fixed
federated averaging for skewed (non-IID) banks, then built a defense that detects and
quarantines a bank that starts poisoning the model.

**Primary track:** 🔴 Advanced &nbsp;&nbsp; **Also submitted:** 🟡 Intermediate

## Results

All numbers are F1 on the organizers' held-out NSL-KDD test split (`KDDTest+`), produced by
`evaluate()` in the notebook, unchanged.

| Track | Before | After | Change |
|---|---|---|---|
| 🔴 **Advanced**: bank 1 poisons (scale x15), seed 42 | naive FedAvg under attack **0.0511** | Zuri Guard **0.7760** | **+0.7249** |
| 🔴 Advanced, mean of 5 seeds | 0.2461 | 0.7639 | +0.5178 |
| 🟡 **Intermediate**: non-IID, no attack, seed 42 | naive FedAvg **0.7117** | our aggregation **0.7821** | **+0.0704** |
| 🟡 Intermediate, mean of 5 seeds | 0.7147 | 0.7787 | +0.0640 |

The coordinate-median baseline defense reaches 0.6855 on the same attack (seed 42).
The best any detect-and-exclude defense can do (bank 1 removed by hand) is 0.764, averaged over 5 seeds.

`model_scripted.pt` and `submission.json` are the Zuri Guard model from the seed 42 Advanced
run (F1 0.776). The notebook's last cell reloads the file and checks it reproduces that F1.

## What we built

**Intermediate: step-normalised, one bank one vote.** Bank 2 holds 48k rows and takes 189 SGD
steps per round, while bank 0 takes 25. FedAvg weights bank 2 by size *and* bank 2's update is
already the longest, so the global model mostly becomes bank 2's model. We divide each update by
its number of local steps (the FedNova idea), give every bank an equal vote, then rescale to a normal
step. On IID data this is exactly FedAvg, which shows the gain comes from handling skew.

**Advanced: Zuri Guard.** Under non-IID data, honest banks with unusual traffic *point* in
unusual directions, so direction-based detection (cosine, Krum) confuses them with attackers.
Update *size* does not have that problem: no honest bank ever exceeded 1.72x the median bank, while the
poisoner's update was about 9x the median or more. Zuri Guard
1. measures each bank's update size relative to the median bank,
2. adds suspicion whenever that ratio is above 2.5x,
3. quarantines a bank for good once its suspicion passes 1.0, and logs it,
4. aggregates the trusted banks with the Intermediate method, after clipping each update to 3x the typical trusted bank.

It was stress tested on nine scenarios (clean, stealthy x3, loud x50, pure label flip, attacker is the
largest bank, two attackers, attack starting at round 4). It catches every scaled attacker with zero
false positives across all 45 runs, and costs nothing on clean data. On its own, Zuri Guard does not
detect pure label flipping, because that attack doesn't make the update bigger.

**Zuri Local Watch: an agent inside each bank (closes the label-flip gap).** Symphorose's idea: every
bank runs a watch agent that starts from the global model each round. It keeps a small private sample of the
bank's correctly labelled rows and checks one thing: *did this round of training make the model worse on my
own data?* Honest training never does (highest ratio seen 0.83), a poisoned run always does (2.3 or more).
A ratio above 1.5 quarantines the bank. Over 10 scenarios and 5 seeds, it caught **every attacker in the first
round it attacked, with zero false alarms**, and lifts label flipping on bank 1 from 0.722 to 0.764 F1. A report
can only remove trust from its own bank, so a bank that fakes its report is judged by Zuri Guard alone: the
combined system is never weaker than Zuri Guard. Details, results and trust boundary:
[`docs/SECURITY_ARCHITECTURE.md`](docs/SECURITY_ARCHITECTURE.md). The submitted `model_scripted.pt` is still the
Zuri Guard model; on the official scenario Local Watch gives the identical result (0.7639 over 5 seeds).

## Reproduce

**Kaggle or Colab (easiest):** upload `zuri_watch_day2.ipynb`, then Run All. Internet must be on, because
the notebook downloads NSL-KDD from GitHub (`jmnwong/NSL-KDD-Dataset`), exactly like the organizers' notebook.
A full run takes about 8 minutes on CPU (most of it is the 5-seed stress test).

**Locally:**

```bash
git clone https://github.com/Bongane0606/Zuri-Watch.git
cd Zuri-Watch
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 zuri_watch_day2.ipynb
```

This regenerates `model_scripted.pt`, `submission.json`, `results.json` and everything in `figures/`.

To reproduce the Local Watch results (about 4 minutes) and run its unit tests:

```bash
python local_watch_eval.py
python -m unittest discover -s tests -v
```

Every run is reseeded right before it starts (`seeded_run`), so numbers do not depend on what ran
earlier in the notebook. Results were produced on CPU with torch 2.14, scikit-learn 1.8, pandas 3.0 and
numpy 2.4. Other versions can shift the last decimal place.

## Verify the exported model

```python
import torch, json
model = torch.jit.load("model_scripted.pt")   # input: 41 preprocessed features, as in Section 1
print(json.load(open("submission.json")))
```

## Repo contents

| File | What it is |
|---|---|
| `zuri_watch_day2.ipynb` | The organizers' Day 2 notebook with our work added, fully run with outputs |
| `model_scripted.pt` | Final exported model (Zuri Guard, Advanced track) |
| `submission.json` | Metrics file produced alongside it |
| `results.json` | Every number in the writeup: ablation, stress test, detection logs |
| `figures/` | Cover image, F1-over-rounds chart, stress test chart, detection timeline |
| `WRITEUP.md` | The Kaggle writeup text |
| `local_watch.py` | Zuri Local Watch: the per-bank watch agent |
| `local_watch_eval.py` | Evaluates Zuri Guard with Local Watch on ten attack scenarios |
| `local_watch_results.json` | Local Watch results over five seeds |
| `tests/test_local_watch.py` | Unit tests for the watch agent |
| `docs/SECURITY_ARCHITECTURE.md` | How the two layers fit together, results and trust boundary |

## Changes to the organizers' harness

1. `seeded_run()` reseeds torch and numpy before every run, for fair and reproducible comparisons.
2. `run_fl()` has an optional `attack_start` argument (default 0, the original behaviour), used only in the stress test.
3. Fixed a `"\\n"` print typo.

`evaluate()`, `WeakMLP`, the data pipeline, partitions and test set are untouched.

## References

* McMahan et al., 2017. *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg).
* Wang et al., 2020. *Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization* (FedNova).
* Blanchard et al., 2017. *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (Krum).
* Yin et al., 2018. *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates* (coordinate median, trimmed mean).
* Cao et al., 2021. *FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping.*
* Tavallaee et al., 2009. *A Detailed Analysis of the KDD CUP 99 Data Set* (NSL-KDD).

## Team

Clen Bongane (team lead), Zuko Gutyungwa, Symphorose Tshibombi. Team Zuri Watch, WeThinkCode_.

## Verification

WTC-R8WUGX63
