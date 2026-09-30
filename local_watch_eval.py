"""Evaluate Zuri Local Watch together with Zuri Guard.

Run:  python local_watch_eval.py      (about 10 minutes on CPU, needs internet for NSL-KDD)

The data pipeline, partitions, model, FL harness, our aggregation and Zuri Guard below are
copied unchanged from zuri_watch_day2.ipynb, so the numbers line up with the notebook.
Results are written to local_watch_results.json.
"""


# ---------- copied from the notebook ----------

# torch, scikit-learn, pandas, numpy and matplotlib are preinstalled on Kaggle and Colab
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy, json, os, hashlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("data", exist_ok=True)
print("Setup complete.")

import matplotlib
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)

TRAIN_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
TEST_URL  = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"

COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty",
]

train_raw = pd.read_csv(TRAIN_URL, names=COLS)
test_raw  = pd.read_csv(TEST_URL, names=COLS)

def clean(df):
    df = df.drop(columns=["difficulty"]).copy()
    df["binary_label"] = (df["label"] != "normal").astype(int)
    return df

train_df = clean(train_raw)
test_df  = clean(test_raw)

CAT_COLS = ["protocol_type", "service", "flag"]
encoders = {}
for c in CAT_COLS:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[c], test_df[c]], axis=0))
    train_df[c] = le.transform(train_df[c])
    test_df[c]  = le.transform(test_df[c])
    encoders[c] = le

FEATURE_COLS = [c for c in train_df.columns if c not in ["label", "binary_label"]]

scaler = StandardScaler()
train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

print("features:", len(FEATURE_COLS))

N_CLIENTS = 5
HOLDOUT_SIZE = 15000

_rng = np.random.RandomState(SEED)
_perm = _rng.permutation(len(train_df))
_holdout_idx = _perm[:HOLDOUT_SIZE]
_pool_idx = _perm[HOLDOUT_SIZE:]
train_pool = train_df.iloc[_pool_idx].reset_index(drop=True)

def make_iid_partition(df, n_clients=N_CLIENTS, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    chunks = np.array_split(idx, n_clients)
    return [df.iloc[c].reset_index(drop=True) for c in chunks]

FAMILY_MAP = {
    "normal": "normal",
    "neptune": "dos", "back": "dos", "land": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos",
    "worm": "dos", "mailbomb": "dos",
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
}

def make_noniid_partition(df, n_clients=N_CLIENTS, alpha=0.3, seed=SEED):
    '''Dirichlet-skewed split by attack family. Low alpha = strong skew.'''
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["family"] = df["label"].map(lambda x: FAMILY_MAP.get(x, "r2l_u2r_other"))
    client_indices = [[] for _ in range(n_clients)]
    for fam in df["family"].unique():
        fam_idx = df.index[df["family"] == fam].to_numpy().copy()
        rng.shuffle(fam_idx)
        proportions = rng.dirichlet(alpha=[alpha] * n_clients)
        split_points = (np.cumsum(proportions) * len(fam_idx)).astype(int)[:-1]
        for i, s in enumerate(np.split(fam_idx, split_points)):
            client_indices[i].extend(s.tolist())
    out = []
    for ci in client_indices:
        rng.shuffle(ci)
        out.append(df.loc[ci].drop(columns=["family"]).reset_index(drop=True))
    return out

iid_clients = make_iid_partition(train_pool)
noniid_clients = make_noniid_partition(train_pool)


N_FEATURES = len(FEATURE_COLS)

class WeakMLP(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

X_test = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
y_test = torch.tensor(test_df["binary_label"].values, dtype=torch.float32)

def df_to_tensors(df):
    X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
    y = torch.tensor(df["binary_label"].values, dtype=torch.float32)
    return X, y

def local_train(model, X, y, epochs=1, lr=0.05, batch_size=256):
    model = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
    return model

def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view(p.shape))
        i += n

def fedavg(models, weights):
    weights = np.array(weights, dtype=float) / np.sum(weights)
    flats = torch.stack([get_flat_params(m) for m in models])
    avg = (flats * torch.tensor(weights, dtype=torch.float32).unsqueeze(1)).sum(dim=0)
    out = copy.deepcopy(models[0])
    set_flat_params(out, avg)
    return out

def coordinate_median(models):
    '''A simple robust-aggregation baseline: take the per-coordinate median of all
    client updates instead of the (weighted) mean. Outliers (e.g. a poisoned update
    with an inflated magnitude) influence the median far less than the mean.'''
    flats = torch.stack([get_flat_params(m) for m in models])
    med = flats.median(dim=0).values
    out = copy.deepcopy(models[0])
    set_flat_params(out, med)
    return out

def evaluate(model):
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_test)) > 0.5).float()
    return {
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
    }

def run_fl(client_dfs, model_fn=lambda: WeakMLP(), rounds=8, epochs=1,
           aggregation="fedavg", agg_fn=None,
           malicious_clients=None, attack="scale", scale_factor=15.0, verbose=True,
           attack_start=0):
    '''
    aggregation: "fedavg" | "median" | "custom" (use agg_fn for "custom")
    malicious_clients: list of client indices that send poisoned updates (advanced track)
    attack: "label_flip" or "scale" (scale = label_flip + inflated update magnitude)
    attack_start: round index where the malicious clients switch on (0 = from the start, as originally)
    '''
    malicious_clients = malicious_clients or []
    global_model = model_fn()
    client_tensors = [df_to_tensors(df) for df in client_dfs]

    history = []
    for r in range(rounds):
        local_models, sizes = [], []
        global_flat = get_flat_params(global_model)

        for cid, (X, y) in enumerate(client_tensors):
            is_malicious = cid in malicious_clients and r >= attack_start
            if is_malicious:
                y = 1 - y  # label-flip
            lm = local_train(global_model, X, y, epochs=epochs)
            if is_malicious and attack == "scale":
                delta = get_flat_params(lm) - global_flat
                set_flat_params(lm, global_flat + scale_factor * delta)
            local_models.append(lm)
            sizes.append(len(X))

        prev_global_model = global_model
        if aggregation == "fedavg":
            global_model = fedavg(local_models, sizes)
        elif aggregation == "median":
            global_model = coordinate_median(local_models)
        elif aggregation == "custom":
            global_model = agg_fn(local_models, sizes, prev_global_model)
        else:
            raise ValueError(f"unknown aggregation: {aggregation}")

        metrics = evaluate(global_model)
        history.append(metrics)
        if verbose:
            print(f"round {r+1:>2}: {metrics}")

    return global_model, history


# Added by Zuri Watch: reseed right before each run so every comparison starts from the
# same initial model and the same client shuffles, no matter what ran before it.
def seeded_run(client_dfs, seed=SEED, **kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return run_fl(client_dfs, **kwargs)

# Small helpers our aggregators share
def model_from_flat(flat, like):
    out = copy.deepcopy(like)
    set_flat_params(out, flat)
    return out

def local_steps(sizes, batch_size=256):
    # how many SGD steps each bank took this round (the server knows this from the sizes)
    return np.ceil(np.asarray(sizes, dtype=float) / batch_size)

# 🔧 YOUR TURN — intermediate track
# Step-normalised aggregation (FedNova style) with one vote per bank.

def my_agg_fn(local_models, sizes, prev_global_model):
    global_flat = get_flat_params(prev_global_model)
    steps = local_steps(sizes)

    # each bank's update divided by how many SGD steps it took, so size stops inflating it
    per_step = torch.stack([
        (get_flat_params(m) - global_flat) / s for m, s in zip(local_models, steps)
    ])

    # every bank gets the same vote, then we scale back up to a normal sized global step
    weights = np.ones(len(local_models)) / len(local_models)
    avg_direction = (per_step * torch.tensor(weights, dtype=torch.float32).unsqueeze(1)).sum(dim=0)
    effective_steps = float((weights * steps).sum())

    return model_from_flat(global_flat + effective_steps * avg_direction, local_models[0])



# 🔧 YOUR TURN — advanced track
# Zuri Guard: magnitude-based detection with sticky quarantine, then clipped,
# step-normalised, one-vote-per-bank aggregation over the banks we still trust.

def make_zuri_guard(ratio_flag=2.5, suspicion_budget=1.0, clip_k=3.0, log=None):
    state = {"suspicion": None, "quarantined": None, "round": 0}

    def zuri_guard(local_models, sizes, prev_global_model):
        n = len(local_models)
        global_flat = get_flat_params(prev_global_model)
        steps = local_steps(sizes)
        updates = torch.stack([get_flat_params(m) - global_flat for m in local_models])

        # 1. how big is each bank's update compared to the median bank?
        raw_norms = updates.norm(dim=1).numpy()
        ratio = raw_norms / np.median(raw_norms)

        # 2 and 3. suspicion builds up above the threshold, and quarantine never resets
        if state["suspicion"] is None:
            state["suspicion"] = np.zeros(n)
            state["quarantined"] = np.zeros(n, dtype=bool)
        state["suspicion"] += np.maximum(0.0, np.log2(ratio / ratio_flag))
        state["quarantined"] |= state["suspicion"] > suspicion_budget
        trusted = ~state["quarantined"]
        if not trusted.any():
            trusted[:] = True  # never aggregate an empty set

        # 4. per-step updates, clipped to a few times the typical trusted bank
        per_step = updates / torch.tensor(steps, dtype=torch.float32).unsqueeze(1)
        per_step_norm = per_step.norm(dim=1)
        cap = clip_k * per_step_norm[torch.tensor(trusted)].median()
        per_step = per_step * torch.clamp(cap / (per_step_norm + 1e-12), max=1.0).unsqueeze(1)

        weights = trusted.astype(float) / trusted.sum()
        direction = (per_step * torch.tensor(weights, dtype=torch.float32).unsqueeze(1)).sum(dim=0)
        effective_steps = float((weights * steps).sum())

        state["round"] += 1
        if log is not None:
            log.append({"round": state["round"],
                        "norm_ratio": np.round(ratio, 2).tolist(),
                        "suspicion": np.round(state["suspicion"], 2).tolist(),
                        "quarantined": [int(i) for i in np.where(state["quarantined"])[0]]})
        return model_from_flat(global_flat + effective_steps * direction, local_models[0])

    return zuri_guard



# ---------- Zuri Local Watch ----------

from local_watch import LocalWatchAgent, take_watch_sample

def make_agents(client_dfs):
    # each bank takes its private watch sample before training starts, so poisoning later
    # in the training pipeline can't touch those labels
    agents = []
    for cid, df in enumerate(client_dfs):
        X, y = df_to_tensors(df)
        wx, wy = take_watch_sample(X, y)
        agents.append(LocalWatchAgent(cid, wx, wy))
    return agents

def make_guard_with_local_watch(agents, lying_banks=(), log=None):
    """Zuri Guard, plus: a bank whose own watch agent reports poisoning is quarantined for good.
    lying_banks simulates a fully compromised bank that fakes a clean report."""
    guard_state = {"suspicion": None, "quarantined": None}
    watch_quarantine = set()

    def agg(local_models, sizes, prev_global_model):
        n = len(local_models)
        for cid, m in enumerate(local_models):
            report = agents[cid].assess(prev_global_model, m)
            if cid in lying_banks:
                continue  # the compromised bank sends a fake "all clear", so nothing is learned
            if report.status == "poisoned":
                watch_quarantine.add(cid)

        # Zuri Guard exactly as in the notebook, with the watch quarantine added on top
        global_flat = get_flat_params(prev_global_model)
        steps = local_steps(sizes)
        updates = torch.stack([get_flat_params(m) - global_flat for m in local_models])
        raw_norms = updates.norm(dim=1).numpy()
        ratio = raw_norms / np.median(raw_norms)
        if guard_state["suspicion"] is None:
            guard_state["suspicion"] = np.zeros(n)
            guard_state["quarantined"] = np.zeros(n, dtype=bool)
        guard_state["suspicion"] += np.maximum(0.0, np.log2(ratio / 2.5))
        guard_state["quarantined"] |= guard_state["suspicion"] > 1.0
        trusted = ~guard_state["quarantined"]
        trusted[list(watch_quarantine)] = False
        if not trusted.any():
            trusted[:] = True

        per_step = updates / torch.tensor(steps, dtype=torch.float32).unsqueeze(1)
        per_step_norm = per_step.norm(dim=1)
        cap = 3.0 * per_step_norm[torch.tensor(trusted)].median()
        per_step = per_step * torch.clamp(cap / (per_step_norm + 1e-12), max=1.0).unsqueeze(1)
        weights = trusted.astype(float) / trusted.sum()
        direction = (per_step * torch.tensor(weights, dtype=torch.float32).unsqueeze(1)).sum(dim=0)
        if log is not None:
            log.append({"watch_quarantined": sorted(watch_quarantine),
                        "guard_quarantined": [int(i) for i in np.where(guard_state["quarantined"])[0]]})
        return model_from_flat(global_flat + float((weights * steps).sum()) * direction, local_models[0])

    return agg


SCENARIOS = {
    "clean (no attack)":                 dict(),
    "scale x15, bank 1 (official)":      dict(malicious_clients=[1], attack="scale", scale_factor=15.0),
    "scale x3, bank 1 (stealthy)":       dict(malicious_clients=[1], attack="scale", scale_factor=3.0),
    "label flip only, bank 1":           dict(malicious_clients=[1], attack="label_flip"),
    "label flip only, bank 2":           dict(malicious_clients=[2], attack="label_flip"),
    "label flip only, bank 4":           dict(malicious_clients=[4], attack="label_flip"),
    "label flip only, banks 1 and 3":    dict(malicious_clients=[1, 3], attack="label_flip"),
    "scale x15, banks 1 and 3":          dict(malicious_clients=[1, 3], attack="scale", scale_factor=15.0),
    "scale x15, bank 1 from round 4":    dict(malicious_clients=[1], attack="scale", scale_factor=15.0, attack_start=3),
    "label flip, bank 1 from round 4":   dict(malicious_clients=[1], attack="label_flip", attack_start=3),
}
SEEDS = [0, 1, 2, 3, 4]

def final_f1(scenario, agg_fn, seed):
    m, _ = seeded_run(noniid_clients, seed=seed, rounds=8, verbose=False, **scenario,
                      aggregation="custom", agg_fn=agg_fn)
    return evaluate(m)["f1"]

if __name__ == "__main__":
    results = {}
    print(f"{'scenario':34s}{'Zuri Guard':>11s}{'+ Local Watch':>15s}{'bank lies':>11s}   detection")
    for name, scen in SCENARIOS.items():
        bad = set(scen.get("malicious_clients", []))
        guard_f1, lw_f1, lying_f1, caught, false_alarms, rounds = [], [], [], 0, 0, []
        for s in SEEDS:
            guard_f1.append(final_f1(scen, make_zuri_guard(), s))
            log = []
            lw_f1.append(final_f1(scen, make_guard_with_local_watch(make_agents(noniid_clients), log=log), s))
            flagged = set(log[-1]["watch_quarantined"])
            caught += len(flagged & bad)
            false_alarms += len(flagged - bad)
            rounds.append(next((i + 1 for i, e in enumerate(log) if bad and bad <= set(e["watch_quarantined"])), None))
            lying_f1.append(final_f1(scen, make_guard_with_local_watch(make_agents(noniid_clients), lying_banks=bad), s))
        results[name] = {"zuri_guard_f1": round(float(np.mean(guard_f1)), 4),
                         "with_local_watch_f1": round(float(np.mean(lw_f1)), 4),
                         "attacker_fakes_report_f1": round(float(np.mean(lying_f1)), 4),
                         "attackers_caught": f"{caught}/{len(bad) * len(SEEDS)}",
                         "false_alarms": false_alarms,
                         "round_caught_per_seed": rounds}
        r = results[name]
        print(f"{name:34s}{r['zuri_guard_f1']:11.4f}{r['with_local_watch_f1']:15.4f}{r['attacker_fakes_report_f1']:11.4f}"
              f"   caught {r['attackers_caught']}, false alarms {false_alarms}, round {rounds}", flush=True)
    with open("local_watch_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved local_watch_results.json")
