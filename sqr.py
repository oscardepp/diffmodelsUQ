# ===============================================
#  Single-Model Quantile Regression (SQR) for UCI
#  - Uses CARD-style UCI splits (data.txt + index_* files)
#  - Metrics on standardized scale + raw width (like deep ensembles)
#  - Architecture matches Facebook SingleModelUncertainty ConditionalQuantile
#  - Now also logs RAW pinball losses for SQR
# ===============================================

import os, sys, time, math, argparse, datetime, random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------
# CLI
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", required=True,
                    help="Name of the UCI Dataset directory, e.g., bostonHousing, blog, YearPredictionMSD.")
parser.add_argument("--root", type=str, default="/home/otd8990/UCI_Datasets",
                    help="Root folder containing <dataset>/data.")
parser.add_argument("--n_hidden_layers", type=int, default=1)
parser.add_argument("--n_hidden_units", type=int, default=64)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--n_ens", type=int, default=1,
                    help="Number of SQR models in the ensemble (1 = pure single-model).")
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--seed_base", type=int, default=3,
                    help="Base random seed; actual seed = seed_base + split_index.")
args = parser.parse_args()

DATASET = args.dataset
ROOT    = args.root

# ------------------------------
# Paths (CARD-style)
# ------------------------------
DATA_DIR = os.path.join(ROOT, DATASET, "data")
RES_DIR  = os.path.join(ROOT, DATASET, "results_sqr")
os.makedirs(RES_DIR, exist_ok=True)

# Timestamped .out/.err (in addition to SLURM capture)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(RES_DIR, f"sqr_{ts}.out")
ERR_PATH = os.path.join(RES_DIR, f"sqr_{ts}.err")

class _Tee(object):
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
    def write(self, data):
        self.stream1.write(data); self.stream1.flush()
        self.stream2.write(data); self.stream2.flush()
    def flush(self):
        self.stream1.flush(); self.stream2.flush()

# Tee stdout/stderr to files as well
_out_f = open(OUT_PATH, "w")
_err_f = open(ERR_PATH, "w")
sys.stdout = _Tee(sys.stdout, _out_f)
sys.stderr = _Tee(sys.stderr, _err_f)

print("torch:", torch.__version__)
print("CUDA visible:", torch.cuda.is_available())

# ------------------------------
# Config / defaults
# ------------------------------
ALPHA = args.alpha  # for 1-ALPHA coverage intervals

def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")

# Optional target×100 scaling (same as deep ensemble)
SCALE_YX100 = {_norm("kin8nm"), _norm("naval"), _norm("naval-propulsion-plant")}

# ------------------------------
# Files & IO
# ------------------------------
def fpath(name): return os.path.join(DATA_DIR, name)

def read_n_splits():
    p = fpath("n_splits.txt")
    if os.path.exists(p):
        try:
            return int(float(np.loadtxt(p)))
        except Exception:
            pass
    return 1

def load_split(seed_idx: int):
    data = np.loadtxt(fpath("data.txt"))
    idx_x = np.loadtxt(fpath("index_features.txt")).astype(int)
    idx_y = int(np.loadtxt(fpath("index_target.txt")))
    X_all = data[:, idx_x].astype(np.float32)
    y_all = data[:, idx_y].astype(np.float32).reshape(-1,1)

    if _norm(DATASET) in SCALE_YX100:
        print("[info] Scaling y by 100 for this dataset.")
        y_all *= 100.0

    idx_tr = np.loadtxt(fpath(f"index_train_{seed_idx}.txt")).astype(int)
    idx_te = np.loadtxt(fpath(f"index_test_{seed_idx}.txt")).astype(int)
    X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
    X_te, y_te = X_all[idx_te], y_all[idx_te]

    # Standardize like CARD
    xs = StandardScaler().fit(X_tr)
    ys = StandardScaler().fit(y_tr)

    X_tr_s = xs.transform(X_tr).astype(np.float32)
    X_te_s = xs.transform(X_te).astype(np.float32)
    y_tr_s = ys.transform(y_tr).astype(np.float32)
    y_te_s = ys.transform(y_te).astype(np.float32)
    return (X_tr_s, y_tr_s, X_te_s, y_te_s, xs, ys)

# ------------------------------
# SQR architecture (from FB SingleModelUncertainty)
# ------------------------------
class QuantileLoss(torch.nn.Module):
    """
    Quantile regression loss (pinball / check loss)
    """
    def __init__(self):
        super(QuantileLoss, self).__init__()
    def forward(self, yhat, y, tau):
        diff = yhat - y
        mask = (diff.ge(0).float() - tau).detach()
        return (mask * diff).mean()

class Perceptron(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_layers,
                 n_hiddens,
                 alpha,
                 dropout):
        super(Perceptron, self).__init__()

        layers = []
        if n_layers == 0:
            layers.append(torch.nn.Linear(n_inputs, n_outputs))
        else:
            layers.append(torch.nn.Linear(n_inputs, n_hiddens))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            for _ in range(n_layers - 1):
                layers.append(torch.nn.Linear(n_hiddens, n_hiddens))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(n_hiddens, n_outputs))

        self.perceptron = torch.nn.Sequential(*layers)
        self.loss_function = None

    def loss(self, x, y):
        return self.loss_function(self.perceptron(x), y)

class ConditionalQuantile(Perceptron):
    """
    Single-Model Quantile Regression (SQR) head:
    - Takes x and tau as input (tau encoded as (tau-0.5)*12)
    - Trains simultaneously on lower and upper quantiles
    """
    def __init__(self, **kwargs):
        super(ConditionalQuantile, self).__init__(**kwargs)
        self.loss_function = QuantileLoss()
        self.alpha = kwargs["alpha"]

    def predict(self, x):
        # Use lower and upper quantiles as in the original code
        tau_l = torch.zeros(x.size(0), 1) + self.alpha / 2
        tau_u = torch.zeros(x.size(0), 1) + (1 - self.alpha / 2)

        preds_l = self.perceptron(torch.cat((x, (tau_l - 0.5) * 12), 1)).detach()
        preds_u = self.perceptron(torch.cat((x, (tau_u - 0.5) * 12), 1)).detach()
        return (preds_l + preds_u) / 2.0, preds_l, preds_u

    def loss(self, x, y):
        tau_l = torch.zeros(x.size(0), 1) + self.alpha / 2
        tau_u = torch.zeros(x.size(0), 1) + (1 - self.alpha / 2)

        preds_l = self.perceptron(torch.cat((x, (tau_l - 0.5) * 12), 1))
        preds_u = self.perceptron(torch.cat((x, (tau_u - 0.5) * 12), 1))

        return self.loss_function(preds_l, y, tau_l) + self.loss_function(preds_u, y, tau_u)

class Ensemble(torch.nn.Module):
    """
    Wrapper to allow n_ens independent SQR models (n_ens=1 => pure single-model).
    Uses the same aggregation rule as FB code.
    """
    def __init__(self,
                 n_ens,
                 n_inputs,
                 n_outputs,
                 n_layers,
                 n_hiddens,
                 alpha,
                 dropout):
        super(Ensemble, self).__init__()
        self.alpha = alpha
        self.learners = torch.nn.ModuleList()
        extra_inputs = 1  # tau
        extra_outputs = 0 # SQR outputs only y
        for _ in range(n_ens):
            self.learners.append(
                ConditionalQuantile(
                    n_inputs=n_inputs + extra_inputs,
                    n_outputs=n_outputs + extra_outputs,
                    n_layers=n_layers,
                    n_hiddens=n_hiddens,
                    alpha=alpha,
                    dropout=dropout,
                )
            )

    def predict(self, x):
        # x is standardized features tensor
        m = len(self.learners)
        preds_mean = torch.zeros(m, x.size(0), 1)
        preds_low  = torch.zeros(m, x.size(0), 1)
        preds_high = torch.zeros(m, x.size(0), 1)

        for l, learner in enumerate(self.learners):
            pm, pl, ph = learner.predict(x)
            preds_mean[l] = pm
            preds_low[l]  = pl
            preds_high[l] = ph

        # aggregate as in FB: mean +- z * std across learners
        from scipy.stats import norm
        threshold = norm.ppf(self.alpha / 2)

        mean_ens = preds_mean.mean(0)
        low_ens  = preds_low.mean(0)  - threshold * preds_low.std(0, m > 1)
        high_ens = preds_high.mean(0) + threshold * preds_high.std(0, m > 1)
        return mean_ens, low_ens, high_ens

    def loss(self, x, y):
        loss = 0.0
        for learner in self.learners:
            loss = loss + learner.loss(x, y)
        return loss / len(self.learners)

# ------------------------------
# Metrics (std scale + raw width)
# ------------------------------
def eval_from_intervals_std_and_rawwidth(y_true_std, lo_std, hi_std, point_std, y_scale):
    """
    Inputs: 1D numpy arrays on standardized scale.
    y_scale: scaler.scale_[0] so that raw_width = y_scale * std_width
    """
    y  = y_true_std.reshape(-1)
    lo = lo_std.reshape(-1)
    hi = hi_std.reshape(-1)
    pt = point_std.reshape(-1)

    cov = float(np.mean((y >= lo) & (y <= hi)))
    wid_std = float(np.mean(hi - lo))
    wid_raw = float(y_scale) * wid_std

    r2 = float(r2_score(y, pt))
    rmse = float(math.sqrt(mean_squared_error(y, pt)))

    # approximate NLL via Gaussian plug-in using the interval width
    z = 1.959963984540054  # ~95% quantile
    sigma = np.maximum((hi - lo) / (2 * z), 1e-6)
    nll = float(np.mean(0.5 * np.log(2 * math.pi * sigma**2) + 0.5 * ((y - pt)**2) / (sigma**2)))

    return dict(cov=cov, wid_std=wid_std, wid_raw=wid_raw, r2=r2, rmse=rmse, nll=nll)

def pinball_loss_np(y_true, y_pred, q):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))

def _fmt_pct(mean_std_tuple):
    m, s = mean_std_tuple
    return f"{m*100:.2f}% ± {s*100:.2f}%"

# ------------------------------
# One split
# ------------------------------
def reset_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_one_split(split_idx: int):
    print(f"\n=== Split seed {split_idx} ===")
    X_tr, y_tr, X_te, y_te, xs, ys = load_split(split_idx)
    print(f"   train: X={X_tr.shape} y={y_tr.shape} | test: X={X_te.shape} y={y_te.shape}")

    input_dim = X_tr.shape[1]
    print(f"   SQR: layers={args.n_hidden_layers} | width={args.n_hidden_units} | "
          f"batch={args.batch} | epochs={args.epochs} | n_ens={args.n_ens} | alpha={ALPHA}")

    x_tr_tensor = torch.Tensor(X_tr)
    y_tr_tensor = torch.Tensor(y_tr)
    x_te_tensor = torch.Tensor(X_te)
    y_te_tensor = torch.Tensor(y_te)

    reset_seeds(args.seed_base + split_idx)
    model = Ensemble(
        n_ens=args.n_ens,
        n_inputs=input_dim,
        n_outputs=1,
        n_layers=args.n_hidden_layers,
        n_hiddens=args.n_hidden_units,
        alpha=ALPHA,
        dropout=args.dropout,
    )

    loader_tr = DataLoader(TensorDataset(x_tr_tensor, y_tr_tensor),
                           shuffle=True,
                           batch_size=args.batch)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Train
    t_train0 = time.perf_counter()
    for epoch in range(args.epochs):
        for xb, yb in loader_tr:
            optimizer.zero_grad()
            loss = model.loss(xb, yb)
            loss.backward()
            optimizer.step()
    t_train1 = time.perf_counter()
    train_time = t_train1 - t_train0

    # Predict
    t_infer0 = time.perf_counter()
    with torch.no_grad():
        p_mean_te, p_low_te, p_high_te = model.predict(x_te_tensor)
    t_infer1 = time.perf_counter()
    infer_time = t_infer1 - t_infer0
    total_time = train_time + infer_time
    print(f"   infer: {x_te_tensor.shape[0]} points in {infer_time:.2f}s | "
          f"train {train_time:.2f}s | total {total_time:.2f}s")

    # Metrics on standardized scale + raw width
    y_te_std = y_te_tensor.numpy().reshape(-1)
    y_scale = float(ys.scale_[0])
    y_mean  = float(ys.mean_[0])

    mean_std = p_mean_te.numpy().reshape(-1)
    low_std  = p_low_te.numpy().reshape(-1)
    high_std = p_high_te.numpy().reshape(-1)

    metrics = eval_from_intervals_std_and_rawwidth(y_te_std, low_std, high_std, mean_std, y_scale)

    # --------------------------
    # Quantile pinball losses
    # --------------------------
    q_low  = ALPHA / 2.0
    q_high = 1.0 - ALPHA / 2.0

    # (1) Standardized scale pinball
    qpin_low_std  = pinball_loss_np(y_te_std, low_std,  q_low)
    qpin_med_std  = pinball_loss_np(y_te_std, mean_std, 0.5)
    qpin_high_std = pinball_loss_np(y_te_std, high_std, q_high)
    qpin_mean_std = (qpin_low_std + qpin_med_std + qpin_high_std) / 3.0

    # (2) RAW scale pinball (using inverse standardization: raw = std * scale + mean)
    y_te_raw   = y_te_std * y_scale + y_mean
    low_raw    = low_std  * y_scale + y_mean
    high_raw   = high_std * y_scale + y_mean
    mean_raw   = mean_std * y_scale + y_mean

    qpin_low_raw  = pinball_loss_np(y_te_raw, low_raw,  q_low)
    qpin_med_raw  = pinball_loss_np(y_te_raw, mean_raw, 0.5)
    qpin_high_raw = pinball_loss_np(y_te_raw, high_raw, q_high)
    qpin_mean_raw = (qpin_low_raw + qpin_med_raw + qpin_high_raw) / 3.0

    metrics.update(dict(
        qpin_low=qpin_low_std,
        qpin_med=qpin_med_std,
        qpin_high=qpin_high_std,
        qpin_mean=qpin_mean_std,
        qpin_low_raw=qpin_low_raw,
        qpin_med_raw=qpin_med_raw,
        qpin_high_raw=qpin_high_raw,
        qpin_mean_raw=qpin_mean_raw,
    ))

    return {
        "seed": split_idx,
        "metrics": metrics,
        "t_train_total": float(train_time),
        "t_infer": float(infer_time),
        "t_total": float(total_time),
        "epochs": args.epochs,
        "batch": args.batch,
        "n_hidden_layers": args.n_hidden_layers,
        "n_hidden_units": args.n_hidden_units,
        "n_ens": args.n_ens,
    }

# ------------------------------
# Aggregation & printing
# ------------------------------
def summarise(rs):
    keys = [
        "cov","wid_std","wid_raw","r2","rmse","nll",
        "qpin_low","qpin_med","qpin_high","qpin_mean",
        "qpin_low_raw","qpin_med_raw","qpin_high_raw","qpin_mean_raw"
    ]
    out = {}
    for k in keys:
        vals = [r["metrics"][k] for r in rs]
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    out["t_train_total"] = (float(np.mean([r["t_train_total"] for r in rs])),
                            float(np.std ([r["t_train_total"] for r in rs])))
    out["t_infer"]       = (float(np.mean([r["t_infer"] for r in rs])),
                            float(np.std ([r["t_infer"] for r in rs])))
    out["t_total"]       = (float(np.mean([r["t_total"] for r in rs])),
                            float(np.std ([r["t_total"] for r in rs])))
    meta = rs[-1]
    out["_meta"] = {
        "epochs": meta["epochs"],
        "batch": meta["batch"],
        "n_hidden_layers": meta["n_hidden_layers"],
        "n_hidden_units": meta["n_hidden_units"],
        "n_ens": meta["n_ens"],
    }
    return out

def pretty_print(tag, s):
    print(f"\n===== {tag} =====")
    print("Dataset :", DATASET)
    print("Layers  :", s['_meta']['n_hidden_layers'])
    print("Width   :", s['_meta']['n_hidden_units'])
    print("Batch   :", s['_meta']['batch'])
    print("Epochs  :", s['_meta']['epochs'])
    print("n_ens   :", s['_meta']['n_ens'])
    print(f"COV     : {_fmt_pct(s['cov'])}")
    print(f"WIDTH   : std={s['wid_std'][0]:.3f}±{s['wid_std'][1]:.3f} | raw={s['wid_raw'][0]:.3f}±{s['wid_raw'][1]:.3f}")
    print(f"R²      : {_fmt_pct(s['r2'])}")
    print(f"RMSE    : {s['rmse'][0]:.3f} ± {s['rmse'][1]:.3f}  (standardized units)")
    print(f"NLL     : {s['nll'][0]:.3f} ± {s['nll'][1]:.3f}  (standardized units, Gaussian plug-in)")
    print("Pinball (std): "
          f"q={ALPHA/2:.3f}: {s['qpin_low'][0]:.4f}±{s['qpin_low'][1]:.4f}; "
          f"q=0.500: {s['qpin_med'][0]:.4f}±{s['qpin_med'][1]:.4f}; "
          f"q={1-ALPHA/2:.3f}: {s['qpin_high'][0]:.4f}±{s['qpin_high'][1]:.4f}; "
          f"avg: {s['qpin_mean'][0]:.4f}±{s['qpin_mean'][1]:.4f}")
    print("Pinball (raw): "
          f"q={ALPHA/2:.3f}: {s['qpin_low_raw'][0]:.4f}±{s['qpin_low_raw'][1]:.4f}; "
          f"q=0.500: {s['qpin_med_raw'][0]:.4f}±{s['qpin_med_raw'][1]:.4f}; "
          f"q={1-ALPHA/2:.3f}: {s['qpin_high_raw'][0]:.4f}±{s['qpin_high_raw'][1]:.4f}; "
          f"avg: {s['qpin_mean_raw'][0]:.4f}±{s['qpin_mean_raw'][1]:.4f}")
    print(f"Train T : {s['t_train_total'][0]:.2f}s ± {s['t_train_total'][1]:.2f}s")
    print(f"Infer T : {s['t_infer'][0]:.2f}s ± {s['t_infer'][1]:.2f}s")
    print(f"Total T : {s['t_total'][0]:.2f}s ± {s['t_total'][1]:.2f}s")

# ------------------------------
# Result files (per split; mirror deep ensemble style)
# ------------------------------
FP_COV        = os.path.join(RES_DIR, "test_coverage_sqr.txt")
FP_WID_STD    = os.path.join(RES_DIR, "test_avg_width_std_sqr.txt")
FP_WID_RAW    = os.path.join(RES_DIR, "test_avg_width_raw_sqr.txt")
FP_R2         = os.path.join(RES_DIR, "test_r2_sqr.txt")
FP_RMSE_STD   = os.path.join(RES_DIR, "test_rmse_std_sqr.txt")
FP_NLL_STD    = os.path.join(RES_DIR, "test_ll_std_sqr.txt")   # negative log-likelihood (std-scale)

FP_QPIN_LOW_STD   = os.path.join(RES_DIR, "test_qpin_low_std_sqr.txt")
FP_QPIN_MED_STD   = os.path.join(RES_DIR, "test_qpin_med_std_sqr.txt")
FP_QPIN_HIGH_STD  = os.path.join(RES_DIR, "test_qpin_high_std_sqr.txt")
FP_QPIN_MEAN_STD  = os.path.join(RES_DIR, "test_qpin_mean_std_sqr.txt")

FP_QPIN_LOW_RAW   = os.path.join(RES_DIR, "test_qpin_low_raw_sqr.txt")
FP_QPIN_MED_RAW   = os.path.join(RES_DIR, "test_qpin_med_raw_sqr.txt")
FP_QPIN_HIGH_RAW  = os.path.join(RES_DIR, "test_qpin_high_raw_sqr.txt")
FP_QPIN_MEAN_RAW  = os.path.join(RES_DIR, "test_qpin_mean_raw_sqr.txt")

FP_TRAIN_T    = os.path.join(RES_DIR, "test_train_time_sqr.txt")
FP_INFER_T    = os.path.join(RES_DIR, "test_infer_time_sqr.txt")
FP_TOTAL_T    = os.path.join(RES_DIR, "test_total_time_sqr.txt")
FP_LOG        = os.path.join(RES_DIR, "log_sqr.txt")

def _safe_rm(p):
    try:
        if os.path.exists(p): os.remove(p)
    except Exception as e:
        print(f"[warn] could not remove {p}: {e}")

def reset_result_files():
    for p in [
        FP_COV, FP_WID_STD, FP_WID_RAW, FP_R2, FP_RMSE_STD, FP_NLL_STD,
        FP_QPIN_LOW_STD, FP_QPIN_MED_STD, FP_QPIN_HIGH_STD, FP_QPIN_MEAN_STD,
        FP_QPIN_LOW_RAW, FP_QPIN_MED_RAW, FP_QPIN_HIGH_RAW, FP_QPIN_MEAN_RAW,
        FP_TRAIN_T, FP_INFER_T, FP_TOTAL_T, FP_LOG
    ]:
        _safe_rm(p)
    print("Result files initialized in:", RES_DIR)

def append_split_results(r):
    m = r["metrics"]
    with open(FP_COV, "a") as f:          f.write(f"{m['cov']}\n")
    with open(FP_WID_STD, "a") as f:      f.write(f"{m['wid_std']}\n")
    with open(FP_WID_RAW, "a") as f:      f.write(f"{m['wid_raw']}\n")
    with open(FP_R2, "a") as f:           f.write(f"{m['r2']}\n")
    with open(FP_RMSE_STD, "a") as f:     f.write(f"{m['rmse']}\n")
    with open(FP_NLL_STD, "a") as f:      f.write(f"{m['nll']}\n")

    # standardized pinball
    with open(FP_QPIN_LOW_STD, "a") as f:  f.write(f"{m['qpin_low']}\n")
    with open(FP_QPIN_MED_STD, "a") as f:  f.write(f"{m['qpin_med']}\n")
    with open(FP_QPIN_HIGH_STD, "a") as f: f.write(f"{m['qpin_high']}\n")
    with open(FP_QPIN_MEAN_STD, "a") as f: f.write(f"{m['qpin_mean']}\n")

    # raw pinball
    with open(FP_QPIN_LOW_RAW, "a") as f:  f.write(f"{m['qpin_low_raw']}\n")
    with open(FP_QPIN_MED_RAW, "a") as f:  f.write(f"{m['qpin_med_raw']}\n")
    with open(FP_QPIN_HIGH_RAW, "a") as f: f.write(f"{m['qpin_high_raw']}\n")
    with open(FP_QPIN_MEAN_RAW, "a") as f: f.write(f"{m['qpin_mean_raw']}\n")

    with open(FP_TRAIN_T, "a") as f:      f.write(f"{r['t_train_total']}\n")
    with open(FP_INFER_T, "a") as f:      f.write(f"{r['t_infer']}\n")
    with open(FP_TOTAL_T, "a") as f:      f.write(f"{r['t_total']}\n")

def append_final_log(summary):
    def _stats_line(name, tup, pct=False):
        m, s = tup
        if pct:
            return f"{name} {_fmt_pct(tup)}\n"
        return f"{name} {m:.6f} +- {s:.6f}\n"
    with open(FP_LOG, "a") as f:
        f.write(f"Dataset {DATASET}\n")
        f.write(_stats_line("COV", summary["cov"], pct=True))
        f.write(f"WIDTH_std {summary['wid_std'][0]:.6f} +- {summary['wid_std'][1]:.6f}\n")
        f.write(f"WIDTH_raw {summary['wid_raw'][0]:.6f} +- {summary['wid_raw'][1]:.6f}\n")
        f.write(_stats_line("R2", summary["r2"], pct=True))
        f.write(f"RMSE_std {summary['rmse'][0]:.6f} +- {summary['rmse'][1]:.6f}\n")
        f.write(f"NLL_std {summary['nll'][0]:.6f} +- {summary['nll'][1]:.6f}\n")

        f.write(f"QPIN_low_std {summary['qpin_low'][0]:.6f} +- {summary['qpin_low'][1]:.6f}\n")
        f.write(f"QPIN_med_std {summary['qpin_med'][0]:.6f} +- {summary['qpin_med'][1]:.6f}\n")
        f.write(f"QPIN_high_std {summary['qpin_high'][0]:.6f} +- {summary['qpin_high'][1]:.6f}\n")
        f.write(f"QPIN_mean_std {summary['qpin_mean'][0]:.6f} +- {summary['qpin_mean'][1]:.6f}\n")

        f.write(f"QPIN_low_raw {summary['qpin_low_raw'][0]:.6f} +- {summary['qpin_low_raw'][1]:.6f}\n")
        f.write(f"QPIN_med_raw {summary['qpin_med_raw'][0]:.6f} +- {summary['qpin_med_raw'][1]:.6f}\n")
        f.write(f"QPIN_high_raw {summary['qpin_high_raw'][0]:.6f} +- {summary['qpin_high_raw'][1]:.6f}\n")
        f.write(f"QPIN_mean_raw {summary['qpin_mean_raw'][0]:.6f} +- {summary['qpin_mean_raw'][1]:.6f}\n")

        f.write(f"Train_total_s {summary['t_train_total'][0]:.6f} +- {summary['t_train_total'][1]:.6f}\n")
        f.write(f"Infer_s {summary['t_infer'][0]:.6f} +- {summary['t_infer'][1]:.6f}\n")
        f.write(f"Total_s {summary['t_total'][0]:.6f} +- {summary['t_total'][1]:.6f}\n")

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    print(f"DATA_DIR: {DATA_DIR}")
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")
    n_splits = read_n_splits()
    SEEDS = list(range(n_splits))
    print(f"Detected n_splits={n_splits} → SEEDS={SEEDS}")

    reset_result_files()

    results = []
    for i, seed_idx in enumerate(SEEDS, 1):
        r = train_one_split(seed_idx)
        results.append(r)
        append_split_results(r)
        summary = summarise(results)
        pretty_print(f"After seed {seed_idx} (n={i})", summary)

    # final consolidated log
    append_final_log(summary)

    # Close tee files cleanly
    _out_f.close(); _err_f.close()
