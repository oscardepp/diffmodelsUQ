# cdm.py
# ===============================================
#  UCI regression: CDM + Split Conformal + CQR
#  - Server friendly (ASCII prints)
#  - CARD-style standardization & evaluation (metrics on ORIGINAL scale)
#  - Reads <ROOT>/<dataset>/data/ with pre-made split indices
#  - Writes results to <ROOT>/<dataset>/results_cdmonly/
# ===============================================

import os, math, time, json, argparse
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ---------- CLI ----------
def _parse_args():
    p = argparse.ArgumentParser(description="CDM + Split Conformal + CQR (UCI)")
    p.add_argument("--dataset", "--dir", dest="dataset", type=str, required=True,
                   help="Dataset folder under ROOT (e.g., blog, kin8nm, concrete).")
    p.add_argument("--root", type=str, default="/home/otd8990/UCI_Datasets",
                   help="Root path containing <dataset>/data (default: /home/otd8990/UCI_Datasets).")
    p.add_argument("--results_subdir", type=str, default="results_cdm_only",
                   help="Subfolder under <ROOT>/<dataset>/ to write results (default: results_cdm_only).")

    # toggles
    g = p.add_mutually_exclusive_group()
    g.add_argument("--run_cdm", action="store_true", help="Enable diffusion model block.")
    g.add_argument("--no_cdm",  action="store_true", help="Disable diffusion model block.")

    # split control
    p.add_argument("--all_splits", action="store_true", help="Run all splits from n_splits.txt.")
    p.add_argument("--seeds", type=str,
                   help="Comma-separated split indices to run (e.g., 0,1,2). Overrides --all_splits.")
    p.add_argument("--cal_frac", type=float, default=0.1, help="Calibration fraction from train (default 0.1).")
    p.add_argument("--alpha", type=float, default=0.05, help="Miscoverage level (0.05 -> 95%% interval).")

    # SC/CQR MLP
    p.add_argument("--mlp_epochs", type=int, default=200)
    p.add_argument("--mlp_lr", type=float, default=1e-3)
    p.add_argument("--mlp_width", type=int, default=128)
    p.add_argument("--mlp_layers", type=int, default=2,
                   help="Number of hidden layers in SC/CQR MLPs (default: 2).")

    # Batch
    p.add_argument("--batch_override", type=int,
                   help="Force batch size, overrides dataset default (CARD-like table).")

    # CDM hyperparams
    p.add_argument("--epochs", type=int, default=200, help="CDM epochs.")
    p.add_argument("--timesteps", type=int, default=500, help="CDM diffusion steps.")
    p.add_argument("--samples", type=int, default=200, help="Samples per test point for CDM.")
    p.add_argument("--lr", type=float, default=1e-3, help="CDM learning rate.")
    p.add_argument("--weight_decay", type=float, default=0.0, help="CDM weight decay.")
    p.add_argument("--dropout", type=float, default=0.10, help="CDM dropout.")
    p.add_argument("--cond_dim", type=int, default=64, help="CDM conditioning embedding dim.")
    p.add_argument("--time_emb_dim", type=int, default=32, help="CDM time embedding dim.")
    p.add_argument("--clip_grad_norm", type=float, default=1.0, help="CDM grad clip.")
    p.add_argument("--cdm_blocks", type=int, default=2,
                   help="Number of ResidualBlocks in CDM backbone (default: 2).")

    return p.parse_args()

args = _parse_args()

# -----------------------------
# Config (from CLI)
# -----------------------------
ROOT               = args.root
data_directory     = args.dataset
RESULTS_SUBDIR     = args.results_subdir

# Split control
USE_ALL_SPLITS = True
SEEDS = [0]
if args.seeds:
    USE_ALL_SPLITS = False
    SEEDS = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
elif args.all_splits:
    USE_ALL_SPLITS = True

# SC/CQR
ALPHA               = args.alpha
CAL_FRAC_OF_TRAIN   = args.cal_frac
MLP_EPOCHS          = args.mlp_epochs
MLP_LR              = args.mlp_lr
MLP_WIDTH           = args.mlp_width
MLP_LAYERS          = args.mlp_layers

# CDM toggle & hyperparams
RUN_CDM = True
if args.no_cdm:
    RUN_CDM = False
if args.run_cdm:
    RUN_CDM = True

COND_DIM       = args.cond_dim
TIME_EMB_DIM   = args.time_emb_dim
DROP_OUT       = args.dropout
EPOCHS         = args.epochs
TIMESTEPS      = args.timesteps
N_SAMPLES      = args.samples
DIFF_LR        = args.lr
WEIGHT_DECAY   = args.weight_decay
CLIP_GRAD_NORM = args.clip_grad_norm
CDM_BLOCKS     = args.cdm_blocks

# Quantile levels for CDM quantile loss (tied to ALPHA)
QUANTILE_LEVELS = [ALPHA / 2.0, 0.5, 1.0 - ALPHA / 2.0]

# Batch defaults (CARD-ish) with override
def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")

CARD_BATCH = {
    _norm("bostonHousing"): 32,
    _norm("concrete"): 32,
    _norm("energy"): 32,
    _norm("kin8nm"): 64,
    _norm("naval-propulsion-plant"): 64,
    _norm("power-plant"): 64,
    _norm("protein-tertiary-structure"): 512,
    _norm("wine-quality-red"): 32,
    _norm("yacht"): 32,
    _norm("yearpredictionmsd"): 1024,
    _norm("blog"): 512,
}

BATCH_DEFAULT = 512
BATCH_OVERRIDE = args.batch_override

def batch_for(name: str) -> int:
    if BATCH_OVERRIDE is not None:
        return BATCH_OVERRIDE
    return CARD_BATCH.get(_norm(name), BATCH_DEFAULT)

# -----------------------------
# Device (ASCII print)
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

PIN_MEMORY  = False
NUM_WORKERS = 0
TORCH_DTYPE = torch.float32

# Optional diffusion deps only if RUN_CDM
if RUN_CDM:
    from diffusers import DDPMScheduler
    from torch.amp import GradScaler, autocast
USE_AMP = (device.type == "cuda") and RUN_CDM

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = os.path.join(ROOT, data_directory, "data")
FILES = dict(
    data   = os.path.join(DATA_DIR, "data.txt"),
    idx_x  = os.path.join(DATA_DIR, "index_features.txt"),
    idx_y  = os.path.join(DATA_DIR, "index_target.txt"),
    nsplit = os.path.join(DATA_DIR, "n_splits.txt"),
)
RESULTS_DIR = os.path.join(ROOT, data_directory, RESULTS_SUBDIR)
os.makedirs(RESULTS_DIR, exist_ok=True)
print("DATA_DIR:", DATA_DIR)
print("Detected/Will write results to:", RESULTS_DIR)

def _get_index_train_test_path(split_num, train=True):
    fname = ("index_train_" if train else "index_test_") + str(int(split_num)) + ".txt"
    return os.path.join(DATA_DIR, fname)

# -----------------------------
# Load arrays & splits
# -----------------------------
def read_splits():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError("Data folder not found: %s" % DATA_DIR)
    if not os.path.exists(FILES["data"]):
        raise FileNotFoundError("Missing data file: %s" % FILES["data"])
    n_splits = int(float(np.loadtxt(FILES["nsplit"])))
    return n_splits

def load_data_arrays():
    data = np.loadtxt(FILES["data"])
    idx_x = np.loadtxt(FILES["idx_x"]).astype(int)
    idx_y = int(np.loadtxt(FILES["idx_y"]))
    X = data[:, idx_x].astype(np.float32)
    y = data[:, idx_y].astype(np.float32).reshape(-1, 1)
    return X, y

# -----------------------------
# Torch modules
# -----------------------------
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=timesteps.device) * -(math.log(10000) / (half - 1)))
    args = timesteps[:, None] * freqs[None, :]
    emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2: emb = F.pad(emb, (0, 1, 0, 0))
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class ConditionalDenoiser(nn.Module):
    def __init__(self, x_dim, cond_dim=64, time_embed_dim=32, drop_out=0.2, n_blocks=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        self.x_embed = nn.Sequential(
            nn.Linear(x_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # Build backbone with variable number of ResidualBlocks
        main_layers = [
            nn.Linear(1 + 128 + cond_dim, 256),
            nn.SiLU(),
            nn.Dropout(drop_out),
        ]
        for _ in range(max(0, n_blocks)):
            main_layers.append(ResidualBlock(256, p=drop_out))
            main_layers.append(nn.SiLU())
            main_layers.append(nn.Dropout(drop_out))
        main_layers.append(nn.Linear(256, 1))
        self.main = nn.Sequential(*main_layers)

        self.time_embed_dim = time_embed_dim

    def forward(self, y_t, t, x_cond):
        emb_t = self.time_mlp(timestep_embedding(t, self.time_embed_dim))
        emb_x = self.x_embed(x_cond)
        return self.main(torch.cat([y_t, emb_t, emb_x], dim=1))

class RegressionMLP(nn.Module):
    def __init__(self, d, width=128, depth=2):
        super().__init__()
        layers = []
        if depth <= 0:
            layers.append(nn.Linear(d, 1))
        else:
            # first hidden layer
            layers.append(nn.Linear(d, width))
            layers.append(nn.SiLU())
            # additional hidden layers
            for _ in range(depth - 1):
                layers.append(nn.Linear(width, width))
                layers.append(nn.SiLU())
            # output
            layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class QuantileMLP(nn.Module):
    def __init__(self, d, width=128, depth=2):
        super().__init__()
        layers = []
        if depth <= 0:
            layers.append(nn.Linear(d, 1))
        else:
            layers.append(nn.Linear(d, width))
            layers.append(nn.SiLU())
            for _ in range(depth - 1):
                layers.append(nn.Linear(width, width))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def pinball_loss(pred, y, q):
    e = y - pred
    return torch.mean(torch.maximum(q*e, (q-1)*e))

# -----------------------------
# CDM training & sampling (timed separately)
# -----------------------------
def train_cdm(diff_model, train_loader, epochs, timesteps, lr, weight_decay=0.0, clip=1.0,
              val_loader=None, patience=10, checkpoint_path=None):
    noise_sched = DDPMScheduler(
        num_train_timesteps=timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False
    )
    opt = optim.AdamW(diff_model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=USE_AMP)

    diff_model.train()
    total_steps = 0
    total_wall_time = 0.0
    total_compute_time = 0.0

    # Early stopping state
    best_val_loss = float("inf") if val_loader is not None else None
    epochs_since_improve = 0
    best_state = None

    for epoch_idx in range(epochs):
        ep_compute_time = 0.0
        for xb, yb in train_loader:
            wall_start = time.perf_counter()
            xb = xb.to(device, non_blocking=False)
            yb = yb.to(device, non_blocking=False)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            compute_start = time.perf_counter()

            with autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                t = torch.randint(0, timesteps, (xb.size(0),), device=device)
                noise = torch.randn_like(yb)
                y_t = noise_sched.add_noise(yb, noise, t)
                pred = diff_model(y_t, t, xb)
                loss = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_((diff_model.parameters()), clip)
            scaler.step(opt)
            scaler.update()
            lr_sched.step()

            if torch.cuda.is_available(): torch.cuda.synchronize()
            compute_end = time.perf_counter()
            wall_end = time.perf_counter()

            total_compute_time += (compute_end - compute_start)
            total_wall_time    += (wall_end    - wall_start)
            total_steps        += 1
            ep_compute_time    += (compute_end - compute_start)
        # End epoch
        # Optionally run validation
        val_loss = None
        if val_loader is not None:
            diff_model.eval()
            losses = []
            with torch.no_grad():
                for vxb, vyb in val_loader:
                    vxb = vxb.to(device, non_blocking=False)
                    vyb = vyb.to(device, non_blocking=False)
                    t = torch.randint(0, timesteps, (vxb.size(0),), device=device)
                    noise = torch.randn_like(vyb)
                    y_t = noise_sched.add_noise(vyb, noise, t)
                    pred = diff_model(y_t, t, vxb)
                    losses.append(F.mse_loss(pred, noise).item())
            if len(losses) > 0:
                val_loss = float(np.mean(losses))
            diff_model.train()

            # Early stopping logic
            if val_loss is not None:
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    epochs_since_improve = 0
                    # keep best state in memory
                    best_state = deepcopy(diff_model.state_dict())
                    # optionally persist to disk
                    if checkpoint_path is not None:
                        try:
                            torch.save(best_state, checkpoint_path)
                        except Exception:
                            pass
                else:
                    epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch_idx+1} (no improvement for {patience} epochs).")
                    break

        # end optional validation

    train_steps_per_sec_wall    = total_steps / max(1e-9, total_wall_time)
    train_steps_per_sec_compute = total_steps / max(1e-9, total_compute_time)
    total_train_wall_seconds    = total_wall_time

    # restore best weights if we recorded them
    if best_state is not None:
        try:
            diff_model.load_state_dict(best_state)
        except Exception:
            # best_state may have been saved to disk; attempt load from checkpoint_path
            if checkpoint_path is not None and os.path.exists(checkpoint_path):
                try:
                    diff_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                except Exception:
                    pass

    return dict(
        steps_per_sec_wall=train_steps_per_sec_wall,
        steps_per_sec_compute=train_steps_per_sec_compute,
        avg_time_per_step_wall=1.0/train_steps_per_sec_wall,
        total_train_wall_s=total_train_wall_seconds,
        noise_sched=noise_sched,
        best_val_loss=best_val_loss,
    )

@torch.no_grad()
def sample_cdm(diff_model, noise_sched, x_norm, timesteps, n_samples):
    sched = deepcopy(noise_sched); sched.set_timesteps(timesteps)
    B, D = x_norm.shape; S = n_samples
    cond = x_norm.unsqueeze(1).repeat(1, S, 1).view(B*S, D)
    y = torch.randn(B*S, 1, device=device) * sched.init_noise_sigma
    for t in sched.timesteps:
        t_b = torch.full((B*S,), t, device=device, dtype=torch.long)
        eps = diff_model(y, t_b, cond)
        y = sched.step(eps, t, y).prev_sample
    return y.view(B, S)

# -----------------------------
# MLP training (point & quantiles)
# -----------------------------
def train_point_mlp(x_train, y_train, epochs, lr, width, batch, depth):
    model = RegressionMLP(x_train.shape[1], width=width, depth=depth).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr)
    ds    = TensorDataset(x_train, y_train)
    dl    = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_quantile_mlp(x_train, y_train, q, epochs, lr, width, batch, depth):
    model = QuantileMLP(x_train.shape[1], width=width, depth=depth).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr)
    ds    = TensorDataset(x_train, y_train)
    dl    = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = pinball_loss(pred, yb, q)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def predict_point(model, x):
    model.eval()
    ys = []
    dl = DataLoader(TensorDataset(x, torch.zeros(len(x),1)), batch_size=1024, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    for xb,_ in dl:
        ys.append(model(xb.to(device)).cpu())
    return torch.cat(ys, 0)

@torch.no_grad()
def predict_quantile(model, x):
    model.eval()
    ys = []
    dl = DataLoader(TensorDataset(x, torch.zeros(len(x),1)), batch_size=1024, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    for xb,_ in dl:
        ys.append(model(xb.to(device)).cpu())
    return torch.cat(ys, 0)

# -----------------------------
# Conformal utilities (SC & CQR)
# -----------------------------
def quantile_level_index(n, alpha):
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return min(max(k, 1), n)

# -----------------------------
# Metrics (ORIGINAL scale)
# -----------------------------
def pack_interval_metrics(y_true_raw, lo_raw, hi_raw, point_raw, also_gaussian_nll=False):
    y_np  = y_true_raw.reshape(-1)
    lo_np = lo_raw.reshape(-1)
    hi_np = hi_raw.reshape(-1)
    pt_np = point_raw.reshape(-1)

    cov  = float(np.mean((y_np >= lo_np) & (y_np <= hi_np)))
    wid  = float(np.mean(hi_np - lo_np))
    r2   = float(r2_score(y_np, pt_np))
    rmse = float(math.sqrt(mean_squared_error(y_np, pt_np)))
    out = dict(cov=cov, wid=wid, r2=r2, rmse=rmse)

    if also_gaussian_nll:
        # crude plug-in: convert interval width to sigma via 95% normal quantile
        z = 1.959963984540054
        sigma = np.maximum((hi_np - lo_np) / (2*z), 1e-12)
        nll = 0.5*np.log(2*np.pi*sigma**2) + 0.5*((y_np - pt_np)**2)/(sigma**2)
        out["nll"] = float(np.mean(nll))
    return out

def quantile_pinball_loss(y_true, qhat, q):
    """Scalar pinball loss for one quantile level q."""
    e = y_true - qhat
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))

def pack_samples_metrics(y_true_raw, samples_raw, samples_std=None, quantile_levels=QUANTILE_LEVELS):
    """
    Metrics from CDM samples.

    - y_true_raw: true target in ORIGINAL scale.
    - samples_raw: samples in ORIGINAL scale (N x S).
    - samples_std: (optional) samples in STANDARDIZED scale (N x S) for normalized width.
    """
    mu_np  = samples_raw.mean(1)
    std_np = samples_raw.std(1, ddof=1)
    lo_np  = np.quantile(samples_raw, ALPHA/2, axis=1)
    hi_np  = np.quantile(samples_raw, 1-ALPHA/2, axis=1)
    y_np   = y_true_raw.reshape(-1)

    cov  = float(np.mean((y_np >= lo_np) & (y_np <= hi_np)))
    wid  = float(np.mean(hi_np - lo_np))
    r2   = float(r2_score(y_np, mu_np))
    rmse = float(math.sqrt(mean_squared_error(y_np, mu_np)))

    std_safe = np.maximum(std_np, 1e-12)
    nll = float(np.mean(0.5*np.log(2*np.pi*std_safe**2) + 0.5*((y_np - mu_np)**2)/(std_safe**2)))
    out = dict(cov=cov, wid=wid, r2=r2, rmse=rmse, nll=nll)

    # Normalized interval width (in standardized y-space) if samples_std provided
    if samples_std is not None:
        lo_std = np.quantile(samples_std, ALPHA/2, axis=1)
        hi_std = np.quantile(samples_std, 1-ALPHA/2, axis=1)
        wid_std = float(np.mean(hi_std - lo_std))
        out["wid_norm"] = wid_std

    # Quantile pinball losses for CDM samples (on original scale)
    if quantile_levels is not None:
        q_loss = {}
        for q in quantile_levels:
            qhat = np.quantile(samples_raw, q, axis=1)
            q_loss[q] = quantile_pinball_loss(y_np, qhat, q)
        out["q_loss"] = q_loss

    return out

# -----------------------------
# One split: run_once(seed)
# -----------------------------
def run_once(split_seed: int, BATCH: int):
    print("\n=== Split seed %d ===" % split_seed)
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)

    # ---- load arrays ----
    X_all, Y_all = load_data_arrays()
    idx_train = np.loadtxt(_get_index_train_test_path(split_seed, train=True)).astype(int)
    idx_test  = np.loadtxt(_get_index_train_test_path(split_seed, train=False)).astype(int)

    X_tr = X_all[idx_train]; Y_tr = Y_all[idx_train]
    X_te = X_all[idx_test];  Y_te = Y_all[idx_test]

    # ---- standardize per CARD ----
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(Y_tr)

    X_tr_s   = x_scaler.transform(X_tr).astype(np.float32)
    X_te_s   = x_scaler.transform(X_te).astype(np.float32)
    Y_tr_s   = y_scaler.transform(Y_tr).astype(np.float32)
    Y_te_s   = y_scaler.transform(Y_te).astype(np.float32)

    xb_tr = torch.from_numpy(X_tr_s).to(TORCH_DTYPE)
    yb_tr = torch.from_numpy(Y_tr_s).to(TORCH_DTYPE)
    xb_te = torch.from_numpy(X_te_s).to(TORCH_DTYPE)
    yb_te = torch.from_numpy(Y_te_s).to(TORCH_DTYPE)

    print("   train: X=%s y=%s | test: X=%s y=%s" % (X_tr.shape, Y_tr.shape, X_te.shape, Y_te.shape))
    print("   batch=%d | epochs=%d | timesteps=%d | samples/pt=%d" % (BATCH, EPOCHS, TIMESTEPS, N_SAMPLES))
    print("   ALPHA=%.3f | CAL_FRAC_OF_TRAIN=%.3f" % (ALPHA, CAL_FRAC_OF_TRAIN))
    print("   CDM blocks=%d | MLP layers=%d" % (CDM_BLOCKS, MLP_LAYERS))

    # ---- calibration subset ----
    rng = np.random.RandomState(split_seed)
    n_tr = len(xb_tr)
    n_cal = max(1, int(round(CAL_FRAC_OF_TRAIN*n_tr)))
    cal_idx = rng.choice(n_tr, size=n_cal, replace=False)
    tr_mask = np.ones(n_tr, dtype=bool); tr_mask[cal_idx] = False
    tr_idx = np.where(tr_mask)[0]

    xb_tr_sc  = xb_tr[tr_idx]
    yb_tr_sc  = yb_tr[tr_idx]
    xb_cal    = xb_tr[cal_idx]
    yb_cal    = yb_tr[cal_idx]

    # ---- invert helper ----
    y_scale = float(y_scaler.scale_[0]); y_mean = float(y_scaler.mean_[0])
    def inv_y(t): return (t.cpu().numpy().reshape(-1,1) * y_scale + y_mean).reshape(-1)

    # Split/CQR removed here; CDM only
    y_true_raw = inv_y(yb_te)

    result = dict(
        seed=split_seed,
        batch=BATCH, epochs=EPOCHS
    )

    # ===========
    # DIFF (timed)
    # ===========
    if RUN_CDM:
        train_loader = DataLoader(TensorDataset(xb_tr, yb_tr),
                                  batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)

        diff_model = ConditionalDenoiser(
            x_dim=xb_tr.shape[1],
            cond_dim=COND_DIM,
            time_embed_dim=TIME_EMB_DIM,
            drop_out=DROP_OUT,
            n_blocks=CDM_BLOCKS,
        ).to(device)

        # Validation loader uses the calibration subset (acts as validation for early stopping)
        val_loader = DataLoader(TensorDataset(xb_cal, yb_cal),
                                batch_size=min(len(xb_cal), BATCH), shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

        t_diff_train0 = time.perf_counter()
        checkpoint_path = os.path.join(RESULTS_DIR, f"best_diff_model_seed{split_seed}.pt")
        train_info = train_cdm(
            diff_model=diff_model, train_loader=train_loader,
            epochs=EPOCHS, timesteps=TIMESTEPS, lr=DIFF_LR, weight_decay=WEIGHT_DECAY, clip=CLIP_GRAD_NORM,
            val_loader=val_loader, patience=10, checkpoint_path=checkpoint_path
        )
        t_diff_train1 = time.perf_counter()
        diff_train_wall = t_diff_train1 - t_diff_train0

        noise_sched = train_info.pop("noise_sched")

        # inference timing
        diff_model.eval()
        infer_times = []
        all_samples_raw = []
        all_samples_std = []
        test_loader = DataLoader(TensorDataset(xb_te, yb_te),
                                 batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
        for (xb, _) in test_loader:
            xb = xb.to(device, non_blocking=False)
            t0 = time.perf_counter()
            samp_std = sample_cdm(diff_model, noise_sched, xb, TIMESTEPS, N_SAMPLES)  # standardized samples
            if torch.cuda.is_available(): torch.cuda.synchronize()
            infer_times.append(time.perf_counter() - t0)

            # store standardized samples for normalized width
            all_samples_std.append(samp_std.cpu().numpy())

            # unnormalize for original-scale metrics
            samp_raw = (samp_std.cpu().numpy() * y_scale + y_mean)
            all_samples_raw.append(samp_raw)

        infer_time_total = float(np.sum(infer_times))
        samples_cat_raw = np.concatenate(all_samples_raw, axis=0)
        samples_cat_std = np.concatenate(all_samples_std, axis=0)

        metrics_diff = pack_samples_metrics(
            y_true_raw=y_true_raw,
            samples_raw=samples_cat_raw,
            samples_std=samples_cat_std
        )

        result.update(dict(
            diff=metrics_diff,
            t_diff_train_s=diff_train_wall,
            t_diff_infer_s=infer_time_total,
            t_diff_total_s=diff_train_wall + infer_time_total,
            t_train_steps_per_sec_wall=train_info["steps_per_sec_wall"],
            t_train_steps_per_sec_compute=train_info["steps_per_sec_compute"],
            t_train_avg_time_per_step_wall=train_info["avg_time_per_step_wall"],
            t_train_total_wall_s=train_info["total_train_wall_s"],
        ))
    else:
        result.update(dict(diff=None))

    # cleanup
    del xb_tr_sc, yb_tr_sc, xb_cal, yb_cal
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

# -----------------------------
# Aggregate, print, and write
# -----------------------------
def _fmt_pct(ms):
    m, s = ms
    return "%.2f%% +/- %.2f%%" % (m*100.0, s*100.0)

def summarise(rs):
    out = {}
    # metrics for diff only
    if any(r.get("diff") is not None for r in rs):
        rs_diff = [r for r in rs if r.get("diff") is not None]
        if len(rs_diff) > 0:
            out["diff"] = {}
            for k in ["cov","wid","wid_norm","r2","rmse","nll"]:
                vals = [r["diff"][k] for r in rs_diff]
                out["diff"][k] = (float(np.mean(vals)), float(np.std(vals)))
            # quantile losses, if present
            if "q_loss" in rs_diff[0]["diff"]:
                q_keys = sorted(rs_diff[0]["diff"]["q_loss"].keys())
                out["diff"]["q_loss"] = {}
                for q in q_keys:
                    vals = [r["diff"]["q_loss"][q] for r in rs_diff]
                    out["diff"]["q_loss"][q] = (float(np.mean(vals)), float(np.std(vals)))
            # times
            out["t_diff_train_s"] = (float(np.mean([r["t_diff_train_s"] for r in rs_diff])),
                                     float(np.std ([r["t_diff_train_s"] for r in rs_diff])))
            out["t_diff_infer_s"] = (float(np.mean([r["t_diff_infer_s"] for r in rs_diff])),
                                     float(np.std ([r["t_diff_infer_s"] for r in rs_diff])))
            out["t_diff_total_s"] = (float(np.mean([r["t_diff_total_s"] for r in rs_diff])),
                                     float(np.std ([r["t_diff_total_s"] for r in rs_diff])))
        else:
            out["diff"] = None
    else:
        out["diff"] = None

    out["_meta"] = dict(
        dataset=data_directory,
        batch=rs[-1]["batch"],
        epochs=rs[-1]["epochs"],
        n_runs=len(rs),
        run_cdm=(rs[-1].get("diff") is not None),
        cdm_blocks=CDM_BLOCKS,
        mlp_layers=MLP_LAYERS,
    )
    return out

def pretty_print(tag, s, last_split_times=None):
    print("\n===== %s =====" % tag)
    print("Dataset  :", s['_meta']['dataset'])
    print("Batch    :", s['_meta']['batch'])
    print("Epochs   :", s['_meta']['epochs'])
    print("Runs     :", s['_meta']['n_runs'])
    print("CDM run? :", s['_meta']['run_cdm'])
    print("CDM blocks:", s['_meta'].get("cdm_blocks"))
    print("MLP layers:", s['_meta'].get("mlp_layers"))

    for lbl in ["diff"]:
        block = s.get(lbl)
        if block is None:
            print("\n%-6s : (not run)" % lbl.upper()); continue
        if block is not None:
            line = ("\n%-6s: " % lbl.upper() +
                    "COV %s | " % _fmt_pct(block['cov']) +
                    "WIDTH_raw %.3f+/-%.3f | " % (block['wid'][0], block['wid'][1]) +
                    "WIDTH_norm %.3f+/-%.3f | " % (block['wid_norm'][0], block['wid_norm'][1]) +
                    "R2 %s | " % _fmt_pct(block['r2']) +
                    "RMSE %.3f+/-%.3f" % (block['rmse'][0], block['rmse'][1]))
            if "nll" in block:
                line += " | NLL %.3f+/-%.3f" % (block['nll'][0], block['nll'][1])
            print(line)
            # Print quantile pinball losses if present
            if "q_loss" in block:
                q_strs = []
                for q, ms in block["q_loss"].items():
                    m, sd = ms
                    q_strs.append("q=%.3f: %.4f+/-%.4f" % (q, m, sd))
                print("  Quantile pinball losses:", "; ".join(q_strs))

    # times (aggregated)
    print("\nTiming (mean +/- std over completed splits):")
    if s.get("diff") is not None:
        td_tr = s["t_diff_train_s"]; td_inf = s["t_diff_infer_s"]; td = s["t_diff_total_s"]
        print("  DIFF train            : %.2fs +/- %.2fs" % (td_tr[0], td_tr[1]))
        print("  DIFF infer            : %.2fs +/- %.2fs" % (td_inf[0], td_inf[1]))
        print("  DIFF total            : %.2fs +/- %.2fs" % (td[0], td[1]))

    # times (last split)
    if last_split_times is not None:
        print("\nLast split times:")
        if "t_diff_total_s" in last_split_times:
            print("  DIFF train            : %.2fs" % last_split_times["t_diff_train_s"])
            print("  DIFF infer            : %.2fs" % last_split_times["t_diff_infer_s"])
            print("  DIFF total            : %.2fs" % last_split_times["t_diff_total_s"])

def write_split_results(res, results_dir):
    """Append per-split metrics to text files (mirrors your style)."""
    def w(fname, val):
        with open(os.path.join(results_dir, fname), "a") as f:
            f.write(str(val) + "\n")

    # diff (if present)
    if res.get("diff") is not None:
        w("diff_cov.txt",          res["diff"]["cov"])
        w("diff_width.txt",        res["diff"]["wid"])
        w("diff_width_norm.txt",   res["diff"].get("wid_norm", float("nan")))
        w("diff_r2.txt",           res["diff"]["r2"])
        w("diff_rmse.txt",         res["diff"]["rmse"])
        w("diff_nll.txt",          res["diff"]["nll"])
        w("diff_time_train_s.txt", res["t_diff_train_s"])
        w("diff_time_infer_s.txt", res["t_diff_infer_s"])
        w("diff_time_total_s.txt", res["t_diff_total_s"])
        # You could also log q_loss per-quantile here in separate files if desired.

def write_final_log(tag, summary, results_dir):
    log_path = os.path.join(results_dir, "aggregate_log.txt")
    with open(log_path, "a") as f:
        f.write("\n===== %s =====\n" % tag)
        f.write("Dataset  : %s\n" % summary['_meta']['dataset'])
        f.write("Batch    : %s\n" % summary['_meta']['batch'])
        f.write("Epochs   : %s\n" % summary['_meta']['epochs'])
        f.write("Runs     : %s\n" % summary['_meta']['n_runs'])
        f.write("CDM run? : %s\n" % summary['_meta']['run_cdm'])
        f.write("CDM blocks: %s\n" % summary['_meta'].get("cdm_blocks"))
        f.write("MLP layers: %s\n" % summary['_meta'].get("mlp_layers"))

        def line(lbl):
            block = summary.get(lbl)
            if block is None: 
                f.write("\n%s : (not run)\n" % lbl.upper()); return
            def pct(ms): m, s = ms; return "%.2f%% +/- %.2f%%" % (m*100.0, s*100.0)
            out = ("\n%s: " % lbl.upper() +
                   "COV %s | " % pct(block['cov']) +
                   "WIDTH_raw %.3f+/-%.3f | " % (block['wid'][0], block['wid'][1]) +
                   "WIDTH_norm %.3f+/-%.3f | " % (block['wid_norm'][0], block['wid_norm'][1]) +
                   "R2 %s | " % pct(block['r2']) +
                   "RMSE %.3f+/-%.3f" % (block['rmse'][0], block['rmse'][1]))
            if "nll" in block:
                out += " | NLL %.3f+/-%.3f" % (block['nll'][0], block['nll'][1])
            f.write(out + "\n")
            if "q_loss" in block:
                f.write("  Quantile pinball losses:\n")
                for q, ms in block["q_loss"].items():
                    m, sd = ms
                    f.write("    q=%.3f: %.4f+/-%.4f\n" % (q, m, sd))

        for lbl in ["diff"]:
            line(lbl)

        f.write("\nTiming (mean +/- std):\n")
        if summary.get("diff") is not None:
            td_tr = summary["t_diff_train_s"]; td_inf = summary["t_diff_infer_s"]; td = summary["t_diff_total_s"]
            f.write("  DIFF train            : %.2fs +/- %.2fs\n" % (td_tr[0], td_tr[1]))
            f.write("  DIFF infer            : %.2fs +/- %.2fs\n" % (td_inf[0], td_inf[1]))
            f.write("  DIFF total            : %.2fs +/- %.2fs\n" % (td[0], td[1]))

# -----------------------------
# Run sweep
# -----------------------------
n_splits = read_splits()
if USE_ALL_SPLITS:
    SEEDS = list(range(n_splits))
print("Detected n_splits=%d -> SEEDS=%s" % (n_splits, SEEDS))

BATCH = batch_for(data_directory)
print("Batch size:", BATCH)

all_results = []
for k, seed in enumerate(SEEDS, 1):
    res = run_once(seed, BATCH)
    # write per-split append-only files
    write_split_results(res, RESULTS_DIR)

    all_results.append(res)
    last_times = {}
    if res.get("diff") is not None:
        last_times.update(
            t_diff_train_s=res["t_diff_train_s"],
            t_diff_infer_s=res["t_diff_infer_s"],
            t_diff_total_s=res["t_diff_total_s"],
        )

    summary = summarise(all_results)
    pretty_print("After seed %d  (n=%d)" % (seed, k), summary, last_split_times=last_times)

# final summary to file
final_summary = summarise(all_results)
write_final_log("FINAL", final_summary, RESULTS_DIR)

# also dump JSON summary
with open(os.path.join(RESULTS_DIR, "aggregate_summary.json"), "w") as jf:
    json.dump(final_summary, jf, indent=2)
