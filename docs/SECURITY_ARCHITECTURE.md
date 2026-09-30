# Zuri Watch Security Architecture

Zuri Watch keeps **learning** separate from **watching the learner**. Five banks train one shared
intrusion detector without pooling their data, and two independent checkpoints watch the training:

```text
Global model
    |
    +--> Bank 0 -- local training -- Local Watch agent --+
    +--> Bank 1 -- local training -- Local Watch agent --+
    +--> Bank 2 -- local training -- Local Watch agent --+--> Zuri Guard --> clip / quarantine --> aggregate
    +--> Bank 3 -- local training -- Local Watch agent --+
    +--> Bank 4 -- local training -- Local Watch agent --+
```

## 1. Local Watch (inside each bank): "Did training make me worse on my own data?"

The Local Watch idea is Symphorose Tshibombi's: every bank runs its own agent, starting from the
global model each round.

* Before any training happens, the agent keeps a small private **watch sample** of the bank's rows with
  their true labels (512 rows by default). It never leaves the bank.
* After local training, it measures the loss of the **global model** and of the **new local model** on
  that sample, and sends the server one number: `new loss / global loss`.
* Honest local training minimises loss on the bank's own data, so this ratio stays **below 1**. Across
  5 clean runs the highest honest value we saw was 0.83. A poisoned training run makes the model worse
  on the true labels: every attacker in our tests was at **2.3 or higher**, already in the first round of
  the attack.
* A ratio above **1.5** quarantines that bank for the rest of training.

**Why the first version was changed.** The first version compared each bank with its own history
(update size and validation score over past rounds). In testing it needed three warm-up rounds, so an
attacker present from round 1 became part of the "normal" baseline. It also raised 18 false alarms over
5 clean runs, because honest updates naturally shrink as training settles. Comparing against the global
model the bank started from needs no history, so both problems disappear.

## 2. Zuri Guard (at the server): "Is this bank unusual compared with the others?"

Zuri Guard compares the size of each bank's update with the median bank, builds suspicion above 2.5x,
quarantines persistent outliers, clips the rest and aggregates with one vote per bank. It catches
scaled attacks without needing anything from the banks.

## Trust boundary

Local Watch only helps if the attacker cannot tamper with the agent, for example because it runs as a
separate locked-down service, or inside a trusted execution environment. A fully compromised bank could
send a fake "all clear". So the design follows two rules:

1. **A report can only remove trust from its own bank.** It can never add trust, and it can never accuse
   another bank.
2. **Zuri Guard never listens to reports.** It judges every bank independently.

So a lying bank is simply judged by Zuri Guard alone. We test this directly: in every scenario, the
"bank lies" column equals the Zuri Guard column exactly. The combined system is never weaker than
Zuri Guard.

## Results (5 seeds each, mean F1 on the held-out test set)

Run `python local_watch_eval.py` to reproduce. Full numbers are in `local_watch_results.json`.

| Scenario | Zuri Guard | + Local Watch | Attacker fakes its report | Local Watch detection |
|---|---|---|---|---|
| Clean (no attack) | 0.7787 | 0.7787 | 0.7787 | 0 false alarms |
| Scale x15, bank 1 (official) | 0.7639 | 0.7639 | 0.7639 | 5/5, round 1 |
| Scale x3, bank 1 (stealthy) | 0.7617 | **0.7639** | 0.7617 | 5/5, round 1 |
| **Label flip only, bank 1** | 0.7215 | **0.7639** | 0.7215 | 5/5, round 1 |
| Label flip only, bank 2 | 0.7856 | 0.7828 | 0.7856 | 5/5, round 1 |
| Label flip only, bank 4 | 0.8046 | 0.7912 | 0.8046 | 5/5, round 1 |
| Label flip only, banks 1 and 3 | 0.7793 | 0.7740 | 0.7793 | 10/10, round 1 |
| Scale x15, banks 1 and 3 | 0.7740 | 0.7740 | 0.7740 | 10/10, round 1 |
| Scale x15, bank 1 from round 4 | 0.7642 | 0.7642 | 0.7642 | 5/5, round 4 |
| **Label flip, bank 1 from round 4** | 0.7294 | **0.7642** | 0.7294 | 5/5, round 4 |

**Detection:** every attacker in every scenario was caught in the first round it attacked, with zero
false alarms across all 50 runs. This closes the gap Zuri Guard alone had with quiet label flipping.

**An honest caveat.** When the label-flipping bank is bank 2 or bank 4, or banks 1 and 3 together,
test F1 is slightly lower (by 0.003 to 0.013) with the attacker removed than with it kept in and clipped.
Our reading is that on those banks, the clipped flipped update happens to push the model towards
flagging more traffic as attack, which the test set rewards. We still think removing a bank that is
provably poisoning the model is the right security decision, and we report the numbers as they are.

## Design note

The layering was inspired by a general operations lesson (the 2012 Knight Capital incident): automated
systems need independent monitoring and a way to contain a component that goes wrong. That is a design
inspiration, not evidence for the algorithm; the evidence is the table above.
