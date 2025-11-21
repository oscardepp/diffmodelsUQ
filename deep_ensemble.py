# ===============================================
#  Deep Ensembles (heteroscedastic MLP) for UCI
#  – Paths & result files aligned to MC-dropout script
#  – Metrics on standardized scale (CARD-style) + raw width
#  – Per-split files + aggregate printing; .out/.err logging
# ===============================================

import os, sys, time, math, argparse, datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------
# CLI
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", required=True,
                    help="Name of the UCI Dataset directory, e.g., bostonHousing, blog, YearPredictionMSD.")
parser.add_argument("--ensemble", "-E", type=int, default=5, help="Ensemble size.")
parser.add_argument("--samples", "-K", type=int, default=200, help="Samples per member at test time.")
parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
args = parser.parse_args()

DATASET = args.dataset
ENSEMBLE_SIZE = args.ensemble
SAMPLES_PER_MEMBER = args.samples
LEARNING_RATE = args.lr

# ------------------------------
# Paths (match MC-dropout layout)
# ------------------------------
ROOT = "/home/otd8990/UCI_Datasets"
DATA_DIR = os.path.join(ROOT, DATASET, "data")
RES_DIR  = os.path.join(ROOT, DATASET, "results_de")
os.makedirs(RES_DIR, exist_ok=True)

# Timestamped .out/.err (in addition to SLURM capture)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(RES_DIR, f"deep_ensemble_{ts}.out")
ERR_PATH = os.path.join(RES_DIR, f"deep_ensemble_{ts}.err")

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

print("TF:", tf.__version__)
print("GPUs visible:", tf.config.list_physical_devices('GPU'))
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

# ------------------------------
# Config / defaults
# ------------------------------
ALPHA = 0.05  # 95% intervals

# CARD-ish per-dataset knobs
def _norm(name:str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")

ARCH_DEFS = {
    "a": [(50,  "relu", 0.0)],
    "b": [(100, "relu", 0.0)],
    "c": [(100, "relu", 0.0), (50, "relu", 0.0)],
}
ARCH_BY_DATASET = {
    _norm("bostonHousing"): "a",
    _norm("concrete"): "c",
    _norm("energy"): "a",
    _norm("kin8nm"): "a",
    _norm("naval"): "a",
    _norm("naval-propulsion-plant"): "a",
    _norm("power-plant"): "a",
    _norm("protein-tertiary-structure"): "b",
    _norm("wine-quality-red"): "a",
    _norm("yacht"): "a",
    _norm("YearPredictionMSD"): "b",
    _norm("blog"): "b",
}
BATCH_BY_DATASET = {
    _norm("bostonHousing"): 32,
    _norm("concrete"): 32,
    _norm("energy"): 32,
    _norm("kin8nm"): 64,
    _norm("naval"): 64,
    _norm("naval-propulsion-plant"): 64,
    _norm("power-plant"): 64,
    _norm("protein-tertiary-structure"): 100,
    _norm("wine-quality-red"): 32,
    _norm("yacht"): 32,
    _norm("YearPredictionMSD"): 100,
    _norm("blog"): 100,
}
EPOCHS_BY_DATASET = { k:100 for k in ARCH_BY_DATASET.keys() }

# Optional target×100 scaling (applied BEFORE standardization)
SCALE_YX100 = {_norm("kin8nm"), _norm("naval"), _norm("naval-propulsion-plant")}

def _hp_for(name:str):
    k = _norm(name)
    arch  = ARCH_BY_DATASET.get(k, "a")
    batch = BATCH_BY_DATASET.get(k, 32)
    epochs= EPOCHS_BY_DATASET.get(k, 100)
    return arch, batch, epochs

# ------------------------------
# Files & IO
# ------------------------------
def fpath(name): return os.path.join(DATA_DIR, name)

def read_n_splits():
    p = fpath("n_splits.txt")
    if os.path.exists(p):
        try:
            return int(float(np.loadtxt(p)))
        except:
            pass
    return 1

def load_split(seed:int):
    data = np.loadtxt(fpath("data.txt"))
    idx_x = np.loadtxt(fpath("index_features.txt")).astype(int)
    idx_y = int(np.loadtxt(fpath("index_target.txt")))
    X_all = data[:, idx_x].astype(np.float32)
    y_all = data[:, idx_y].astype(np.float32).reshape(-1,1)

    if _norm(DATASET) in SCALE_YX100:
        print("[info] Scaling y by 100 for this dataset.")
        y_all *= 100.0

    idx_tr = np.loadtxt(fpath(f"index_train_{seed}.txt")).astype(int)
    idx_te = np.loadtxt(fpath(f"index_test_{seed}.txt")).astype(int)
    X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
    X_te, y_te = X_all[idx_te], y_all[idx_te]

    xs = StandardScaler().fit(X_tr)
    ys = StandardScaler().fit(y_tr)

    X_tr_s = xs.transform(X_tr).astype(np.float32)
    X_te_s = xs.transform(X_te).astype(np.float32)
    y_tr_s = ys.transform(y_tr).astype(np.float32)
    y_te_s = ys.transform(y_te).astype(np.float32)
    return (X_tr_s, y_tr_s, X_te_s, y_te_s, xs, ys)

# ------------------------------
# Model (heteroscedastic: μ + log σ²)
# ------------------------------
def _act_layer(act, name=None):
    if act is None: return None
    if isinstance(act, str):
        a = act.lower()
        if a == "relu": fn = tf.nn.relu
        elif a in ("leaky_relu","lrelu"): fn = lambda x: tf.nn.leaky_relu(x, alpha=0.2)
        elif a == "tanh": fn = tf.nn.tanh
        elif a == "sigmoid": fn = tf.nn.sigmoid
        elif a == "gelu": fn = getattr(tf.nn, "gelu", tf.keras.activations.gelu)
        elif a == "swish": fn = getattr(tf.nn, "silu", getattr(tf.nn, "swish", tf.keras.activations.swish))
        elif a == "elu": fn = tf.nn.elu
        elif a == "selu": fn = tf.nn.selu
        else: fn = lambda x: x
    elif callable(act):
        fn = act
    else:
        fn = lambda x: x
    return layers.Lambda(fn, name=name)

def build_pnn(input_dim: int, arch_code: str):
    arch = ARCH_DEFS[arch_code]
    x_in = layers.Input(shape=(input_dim,), name="x")
    h = x_in
    for i,(units,act,drop) in enumerate(arch):
        h = layers.Dense(units, activation=None, name=f"d{i}")(h)
        A = _act_layer(act, name=f"act{i}")
        if A is not None: h = A(h)
        if drop and drop > 0: h = layers.Dropout(drop, name=f"drop{i}")(h)
    mu      = layers.Dense(1, activation=None, name="mu")(h)
    log_var = layers.Dense(1, activation=None, name="log_var_raw")(h)
    log_var = layers.Lambda(lambda t: tf.clip_by_value(t, -7.0, 7.0), name="log_var")(log_var)
    y_out   = layers.Concatenate(name="y_out")([mu, log_var])
    return models.Model(inputs=x_in, outputs=y_out, name="PNN_Hetero")

def nll_gauss_from_concat(y_true, y_pred):
    mu = y_pred[:, :1]
    log_var = y_pred[:, 1:]
    inv_var = tf.exp(-log_var)
    return tf.reduce_mean(0.5*(tf.math.log(2*math.pi) + log_var + tf.square(y_true - mu)*inv_var))

def mae_on_mu(y_true, y_pred): return tf.reduce_mean(tf.abs(y_true - y_pred[:, :1]))
def mse_on_mu(y_true, y_pred): return tf.reduce_mean(tf.square(y_true - y_pred[:, :1]))

# ------------------------------
# Train / Predict
# ------------------------------
def train_one_member(X_tr, y_tr, input_dim, arch_code, batch_size, epochs, seed, member_idx):
    tf.keras.utils.set_random_seed(seed)
    model = build_pnn(input_dim, arch_code)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=nll_gauss_from_concat,
                  metrics=[mae_on_mu, mse_on_mu])
    print(f"   [member {member_idx}] Training {epochs} epochs, batch={batch_size}, lr={LEARNING_RATE:g}")
    t0 = time.perf_counter()
    hist = model.fit(X_tr, y_tr, batch_size=batch_size, epochs=epochs, verbose=0)
    t1 = time.perf_counter()
    print(f"   [member {member_idx}] done in {t1 - t0:.2f}s | "
          f"final: mae={hist.history['mae_on_mu'][-1]:.4f}, "
          f"mse={hist.history['mse_on_mu'][-1]:.4f}, "
          f"nll={hist.history['loss'][-1]:.4f}")
    return model, (t1 - t0)

def predict_member(model, X):
    y = model(tf.convert_to_tensor(X), training=False).numpy()
    mu = y[:, :1]
    log_var = y[:, 1:]
    sigma = np.sqrt(np.exp(log_var))
    return mu, sigma

def sample_from_ensemble(members, X, K=SAMPLES_PER_MEMBER):
    mus, sigs = [], []
    for m in members:
        mu, sig = predict_member(m, X)
        mus.append(mu.reshape(-1))
        sigs.append(sig.reshape(-1))
    mus = np.stack(mus, axis=1)     # (n, E)
    sigs = np.stack(sigs, axis=1)   # (n, E)
    n, E = mus.shape
    z = np.random.randn(n, E, K).astype(np.float32)
    samp = mus[..., None] + sigs[..., None] * z  # (n, E, K)
    return samp.reshape(n, E*K)

# ------------------------------
# Metrics (std scale + raw width + quantile loss)
# ------------------------------
def _pinball_loss_np(y, q, tau):
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))

def eval_from_samples_std_and_rawwidth(y_true_std, samples_std, y_scale):
    # Predictive mean & std on standardized scale
    mu  = samples_std.mean(axis=1)
    std = samples_std.std(axis=1, ddof=1)

    # Interval quantiles on standardized scale
    tau_low  = 0.5 * ALPHA
    tau_high = 1.0 - 0.5 * ALPHA
    lo  = np.quantile(samples_std, tau_low,  axis=1)
    hi  = np.quantile(samples_std, tau_high, axis=1)

    cov     = np.mean((y_true_std >= lo) & (y_true_std <= hi))
    wid_std = np.mean(hi - lo)                 # standardized width (CARD-style)
    wid_raw = float(y_scale) * wid_std         # raw width via linear scaling

    r2   = r2_score(y_true_std, mu)            # affine-invariant
    rmse = math.sqrt(mean_squared_error(y_true_std, mu))  # std-units

    std_safe = np.maximum(std, 1e-6)
    nll = np.mean(0.5*np.log(2*math.pi*std_safe**2) + 0.5*((y_true_std - mu)**2)/(std_safe**2))

    # -------- Quantile pinball losses (std + raw) --------
    q_med_std = np.quantile(samples_std, 0.5, axis=1)

    ql025_std = _pinball_loss_np(y_true_std, lo,        tau_low)
    ql500_std = _pinball_loss_np(y_true_std, q_med_std, 0.5)
    ql975_std = _pinball_loss_np(y_true_std, hi,        tau_high)

    scale = abs(float(y_scale))
    ql025_raw = scale * ql025_std
    ql500_raw = scale * ql500_std
    ql975_raw = scale * ql975_std

    qlavg_std = (ql025_std + ql500_std + ql975_std) / 3.0
    qlavg_raw = (ql025_raw + ql500_raw + ql975_raw) / 3.0

    return dict(
        cov=cov,
        wid_std=wid_std,
        wid_raw=wid_raw,
        r2=r2,
        rmse=rmse,
        nll=nll,
        ql025_std=ql025_std,
        ql500_std=ql500_std,
        ql975_std=ql975_std,
        ql025_raw=ql025_raw,
        ql500_raw=ql500_raw,
        ql975_raw=ql975_raw,
        qlavg_std=qlavg_std,
        qlavg_raw=qlavg_raw,
    )

def _fmt_pct(mean_std_tuple):
    m, s = mean_std_tuple
    return f"{m*100:.2f}% ± {s*100:.2f}%"

# ------------------------------
# One split
# ------------------------------
def run_once(seed:int):
    print(f"\n=== Split seed {seed} ===")
    X_tr, y_tr, X_te, y_te, xs, ys = load_split(seed)
    print(f"   train: X={X_tr.shape} y={y_tr.shape} | test: X={X_te.shape} y={y_te.shape}")

    input_dim = X_tr.shape[1]
    arch_code, batch, epochs = _hp_for(DATASET)
    print(f"   arch={arch_code} (a=[50], b=[100], c=[100,50]) | batch={batch} | epochs={epochs} | ensemble={ENSEMBLE_SIZE}")

    # Train ensemble
    members, train_times = [], []
    t_train_start = time.perf_counter()
    for i in range(ENSEMBLE_SIZE):
        member_seed = seed * 1000 + i
        m, t = train_one_member(X_tr, y_tr, input_dim, arch_code, batch, epochs, member_seed, i)
        members.append(m); train_times.append(t)
    t_train_end = time.perf_counter()
    train_time_total = t_train_end - t_train_start

    # Inference
    t0 = time.perf_counter()
    samples_std = sample_from_ensemble(members, X_te, K=SAMPLES_PER_MEMBER)
    t1 = time.perf_counter()
    infer_time = t1 - t0
    total_time = train_time_total + infer_time
    print(f"   inference sampling: {samples_std.shape} samples in {infer_time:.2f}s "
          f"| total train {train_time_total:.2f}s | total {total_time:.2f}s")

    # Metrics on standardized scale + raw width + quantile losses
    y_te_std = y_te.reshape(-1)
    y_scale = float(ys.scale_[0])
    metrics = eval_from_samples_std_and_rawwidth(y_te_std, samples_std, y_scale)

    return {
        "seed": seed,
        "metrics": metrics,
        "t_train_total": float(train_time_total),
        "t_infer": float(infer_time),
        "t_total": float(total_time),
        "t_train_mean_member": float(np.mean(train_times)),
        "t_train_std_member":  float(np.std(train_times)),
        "epochs": epochs,
        "batch": batch,
        "arch": arch_code,
        "members": ENSEMBLE_SIZE
    }

# ------------------------------
# Aggregation & printing
# ------------------------------
def summarise(rs):
    keys = [
        "cov","wid_std","wid_raw","r2","rmse","nll",
        "ql025_std","ql500_std","ql975_std",
        "ql025_raw","ql500_raw","ql975_raw",
        "qlavg_std","qlavg_raw",
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
    out["_meta"] = {k: meta[k] for k in ["epochs","batch","arch","members"]}
    return out

def pretty_print(tag, s):
    print(f"\n===== {tag} =====")
    print("Dataset :", DATASET)
    print("Arch    :", s['_meta']['arch'], "(a=[50], b=[100], c=[100,50])")
    print("Batch   :", s['_meta']['batch'])
    print("Epochs  :", s['_meta']['epochs'])
    print("Members :", s['_meta']['members'])
    print(f"COV     : {_fmt_pct(s['cov'])}")
    print(f"WIDTH   : std={s['wid_std'][0]:.3f}±{s['wid_std'][1]:.3f} | raw={s['wid_raw'][0]:.3f}±{s['wid_raw'][1]:.3f}")
    print(f"R²      : {_fmt_pct(s['r2'])}")
    print(f"RMSE    : {s['rmse'][0]:.3f} ± {s['rmse'][1]:.3f}  (standardized units)")
    print(f"NLL     : {s['nll'][0]:.3f} ± {s['nll'][1]:.3f}  (standardized units)")
    print(
        "  Quantile pinball losses (raw): "
        f"q=0.025: {s['ql025_raw'][0]:.4f}±{s['ql025_raw'][1]:.4f}; "
        f"q=0.500: {s['ql500_raw'][0]:.4f}±{s['ql500_raw'][1]:.4f}; "
        f"q=0.975: {s['ql975_raw'][0]:.4f}±{s['ql975_raw'][1]:.4f}"
    )
    print(
        f"  Avg quantile pinball (raw, over 0.025/0.5/0.975): "
        f"{s['qlavg_raw'][0]:.4f}±{s['qlavg_raw'][1]:.4f}"
    )
    print(f"Train T : {s['t_train_total'][0]:.2f}s ± {s['t_train_total'][1]:.2f}s")
    print(f"Infer T : {s['t_infer'][0]:.2f}s ± {s['t_infer'][1]:.2f}s")
    print(f"Total T : {s['t_total'][0]:.2f}s ± {s['t_total'][1]:.2f}s")

# ------------------------------
# Result files (per split; mirror MC-dropout style)
# ------------------------------
FP_COV        = os.path.join(RES_DIR, "test_coverage_de.txt")
FP_WID_STD    = os.path.join(RES_DIR, "test_avg_width_std_de.txt")
FP_WID_RAW    = os.path.join(RES_DIR, "test_avg_width_raw_de.txt")
FP_R2         = os.path.join(RES_DIR, "test_r2_de.txt")
FP_RMSE_STD   = os.path.join(RES_DIR, "test_rmse_std_de.txt")
FP_NLL_STD    = os.path.join(RES_DIR, "test_ll_std_de.txt")   # negative log-likelihood (std-scale)
FP_TRAIN_T    = os.path.join(RES_DIR, "test_train_time_de.txt")
FP_INFER_T    = os.path.join(RES_DIR, "test_infer_time_de.txt")
FP_TOTAL_T    = os.path.join(RES_DIR, "test_total_time_de.txt")
FP_LOG        = os.path.join(RES_DIR, "log_de.txt")

# per-quantile pinball loss files (std & raw)
FP_QL025_STD  = os.path.join(RES_DIR, "test_qpin_0.025_std_de.txt")
FP_QL500_STD  = os.path.join(RES_DIR, "test_qpin_0.500_std_de.txt")
FP_QL975_STD  = os.path.join(RES_DIR, "test_qpin_0.975_std_de.txt")
FP_QL025_RAW  = os.path.join(RES_DIR, "test_qpin_0.025_raw_de.txt")
FP_QL500_RAW  = os.path.join(RES_DIR, "test_qpin_0.500_raw_de.txt")
FP_QL975_RAW  = os.path.join(RES_DIR, "test_qpin_0.975_raw_de.txt")
FP_QLAVG_STD  = os.path.join(RES_DIR, "test_qpin_avg_std_de.txt")
FP_QLAVG_RAW  = os.path.join(RES_DIR, "test_qpin_avg_raw_de.txt")

def _safe_rm(p):
    try:
        if os.path.exists(p): os.remove(p)
    except Exception as e:
        print(f"[warn] could not remove {p}: {e}")

def reset_result_files():
    for p in [
        FP_COV, FP_WID_STD, FP_WID_RAW, FP_R2, FP_RMSE_STD, FP_NLL_STD,
        FP_TRAIN_T, FP_INFER_T, FP_TOTAL_T, FP_LOG,
        FP_QL025_STD, FP_QL500_STD, FP_QL975_STD,
        FP_QL025_RAW, FP_QL500_RAW, FP_QL975_RAW,
        FP_QLAVG_STD, FP_QLAVG_RAW,
    ]:
        _safe_rm(p)
    print("Result files initialized in:", RES_DIR)

def append_split_results(r):
    m = r["metrics"]
    with open(FP_COV, "a") as f:      f.write(f"{m['cov']}\n")
    with open(FP_WID_STD, "a") as f:  f.write(f"{m['wid_std']}\n")
    with open(FP_WID_RAW, "a") as f:  f.write(f"{m['wid_raw']}\n")
    with open(FP_R2, "a") as f:       f.write(f"{m['r2']}\n")
    with open(FP_RMSE_STD, "a") as f: f.write(f"{m['rmse']}\n")
    with open(FP_NLL_STD, "a") as f:  f.write(f"{m['nll']}\n")
    with open(FP_TRAIN_T, "a") as f:  f.write(f"{r['t_train_total']}\n")
    with open(FP_INFER_T, "a") as f:  f.write(f"{r['t_infer']}\n")
    with open(FP_TOTAL_T, "a") as f:  f.write(f"{r['t_total']}\n")
    # quantile pinball losses
    with open(FP_QL025_STD, "a") as f: f.write(f"{m['ql025_std']}\n")
    with open(FP_QL500_STD, "a") as f: f.write(f"{m['ql500_std']}\n")
    with open(FP_QL975_STD, "a") as f: f.write(f"{m['ql975_std']}\n")
    with open(FP_QL025_RAW, "a") as f: f.write(f"{m['ql025_raw']}\n")
    with open(FP_QL500_RAW, "a") as f: f.write(f"{m['ql500_raw']}\n")
    with open(FP_QL975_RAW, "a") as f: f.write(f"{m['ql975_raw']}\n")
    with open(FP_QLAVG_STD, "a") as f: f.write(f"{m['qlavg_std']}\n")
    with open(FP_QLAVG_RAW, "a") as f: f.write(f"{m['qlavg_raw']}\n")

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
        f.write(f"QPin_std_0.025 {summary['ql025_std'][0]:.6f} +- {summary['ql025_std'][1]:.6f}\n")
        f.write(f"QPin_std_0.500 {summary['ql500_std'][0]:.6f} +- {summary['ql500_std'][1]:.6f}\n")
        f.write(f"QPin_std_0.975 {summary['ql975_std'][0]:.6f} +- {summary['ql975_std'][1]:.6f}\n")
        f.write(f"QPin_raw_0.025 {summary['ql025_raw'][0]:.6f} +- {summary['ql025_raw'][1]:.6f}\n")
        f.write(f"QPin_raw_0.500 {summary['ql500_raw'][0]:.6f} +- {summary['ql500_raw'][1]:.6f}\n")
        f.write(f"QPin_raw_0.975 {summary['ql975_raw'][0]:.6f} +- {summary['ql975_raw'][1]:.6f}\n")
        f.write(f"QPin_std_avg {summary['qlavg_std'][0]:.6f} +- {summary['qlavg_std'][1]:.6f}\n")
        f.write(f"QPin_raw_avg {summary['qlavg_raw'][0]:.6f} +- {summary['qlavg_raw'][1]:.6f}\n")
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
    for i, seed in enumerate(SEEDS, 1):
        r = run_once(seed)
        results.append(r)
        append_split_results(r)
        s = summarise(results)
        pretty_print(f"After seed {seed} (n={i})", s)

    # final consolidated log
    append_final_log(s)

    # Close tee files cleanly
    _out_f.close(); _err_f.close()
