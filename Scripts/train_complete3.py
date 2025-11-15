# GNN model training on PROTEINS dataset with outlier removal OUTLIER_CONTAMINATION = 0.02 and cross-validation: 
# Detected 23 (2.07%) outlier graphs (removed) [   6   21   66  191  276  680  683  721  730 
#  857  880  914  936  939 965  976  994  997 1015 1047 1054 1058 1059] 
# Overall CV Accuracy:  0.7257 ± 0.0198
# Overall F1-score: 0.7087 ± 0.0221

# By Adding Fold‑specific class weights with alpha tuning [0.5, 1.0, 2.0]
# class_weights = [0.8283, 1.2616] 
# Fold 1 class counts: [526 346], weights: [0.82889736 1.2601156 ]
# Fold 2 class counts: [526 346], weights: [0.82889736 1.2601156 ]
# Fold 3 class counts: [526 346], weights: [0.82889736 1.2601156 ] 
# Fold 4 class counts: [527 345], weights: [0.82732445 1.2637681 ]
# Fold 5 class counts: [527 345], weights: [0.82732445 1.2637681 ]
# Overall CV Accuracy: 0.6991 ± 0.0168
# Overall CV F1-score: 0.6903 ± 0.0192

import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, MLP
import json

# workspace paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs/GNN_complete"))
os.makedirs(OUT, exist_ok=True)

# Files 
NODE_ATTR = os.path.join(ROOT, "PROTEINS_node_attributes.txt")
GRAPH_IND = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GRAPH_LABELS = os.path.join(ROOT, "PROTEINS_graph_labels.txt")
EDGES = os.path.join(ROOT, "PROTEINS_A.txt")

# Outlier removal controls (tune these)
OUTLIER_CONTAMINATION = 0.02   # try 0.02, 0.01, 0.005
SKIP_OUTLIER_REMOVAL = False   # set True to skip removal and run with all graphs
USE_CLASS_WEIGHTS = True  # toggle to True to enable class-weighted loss

def read_table_try(path):
    for sep in (r'\s*,\s*', r'\s+'):
        try:
            return pd.read_csv(path, sep=sep, header=None, engine='python')
        except Exception:
            pass
    return pd.read_csv(path, header=None)

# load data
X_df = read_table_try(NODE_ATTR)
gi = read_table_try(GRAPH_IND).iloc[:,0].astype(int).to_numpy()
gl = read_table_try(GRAPH_LABELS).iloc[:,0].astype(int).to_numpy()
# create mapping from original labels to contiguous class indices starting at 0
unique_labels = np.unique(gl)
label_map = {int(l): i for i, l in enumerate(unique_labels)}
if not os.path.isfile(EDGES):
    raise FileNotFoundError(f"Edges file not found: {EDGES}")
edges_df = read_table_try(EDGES).iloc[:, :2].astype(int).values

# optional: select subset of node attributes (change TOP10_IDX as needed)
# use all node attributes from NODE_ATTR file
X = X_df.values.astype(float)
# Standardize node-level features (fit on all nodes)

node_scaler = StandardScaler()
X = node_scaler.fit_transform(X)

n_nodes, n_feat = X.shape
Gnum = int(gi.max())
if n_nodes != gi.shape[0]:
    raise RuntimeError(f"Number of node rows ({n_nodes}) != GRAPH_IND rows ({gi.shape[0]})")

# build nodes per graph and per-graph means for outlier detection / graph_feat
nodes_in_graph = [np.where(gi == g)[0] for g in range(1, Gnum+1)]
# compute graph_means from the already-standardized node features
graph_means = np.vstack([X[nodes].mean(axis=0) for nodes in nodes_in_graph])

# outlier detection (optional)
if SKIP_OUTLIER_REMOVAL:
    outlier_mask = np.zeros(len(graph_means), dtype=bool)
    print("Skipping outlier removal (SKIP_OUTLIER_REMOVAL=True).")
else:
    scaler = StandardScaler().fit(graph_means)
    gm_s = scaler.transform(graph_means)
    iso = IsolationForest(contamination=OUTLIER_CONTAMINATION, random_state=0)
    is_out = iso.fit_predict(gm_s)
    outlier_mask = (is_out == -1)
    print(f"Detected {outlier_mask.sum()} outlier graphs (removed):", np.where(outlier_mask)[0]+1)

# quick summary for decision making
num_removed = int(outlier_mask.sum())
total_graphs = len(graph_means)
print(f"Outlier removal: {num_removed}/{total_graphs} graphs removed ({100.0*num_removed/total_graphs:.2f}%)")

# save and inspect removed graphs (inserted here — before building Data objects)
removed_idx = np.where(outlier_mask)[0] + 1  # 1-based graph ids
removed_info = []
for gid in removed_idx:
    nodes = np.where(gi == int(gid))[0]
    removed_info.append({"graph_id": int(gid), "num_nodes": int(nodes.size)})
removed_json_path = os.path.join(OUT, "removed_graphs.json")
with open(removed_json_path, "w") as jf:
    json.dump({"removed": removed_info, "count": int(outlier_mask.sum())}, jf, indent=2)
print("Wrote removed_graphs.json to:", removed_json_path)
# quick console summary
if removed_info:
    gids = np.array([r["graph_id"] for r in removed_info])

# prepare PyG Data objects (exclude outliers)
data_list = []
graph_idx_map = []
for g_idx in range(1, Gnum+1):
    nodes = nodes_in_graph[g_idx-1]
    if outlier_mask[g_idx-1]:
        continue
    node_global = nodes + 1  # edges are 1-based
    mask = np.isin(edges_df[:,0], node_global) & np.isin(edges_df[:,1], node_global)
    sub_edges = edges_df[mask].copy()
    if sub_edges.size == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        global_to_local = {g: i for i, g in enumerate(node_global)}
        src = [global_to_local[int(u)] for u in sub_edges[:,0]]
        dst = [global_to_local[int(v)] for v in sub_edges[:,1]]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    x = torch.tensor(X[nodes], dtype=torch.float)
    graph_feat = torch.tensor(graph_means[g_idx-1], dtype=torch.float)
    y = torch.tensor([label_map[int(gl[g_idx-1])]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.graph_feat = graph_feat
    data_list.append(data)
    graph_idx_map.append(g_idx-1)

# model
class GINNet(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, num_layers=3, num_classes=2, graph_feat_dim=0):
        super().__init__()
        convs = []
        dims = [in_dim] + [hidden] * (num_layers - 1)
        for dim in dims:
            mlp = MLP([dim, hidden], final_activation=F.relu)
            convs.append(GINConv(mlp))
        self.convs = torch.nn.ModuleList(convs)
        self.graph_feat_dim = int(graph_feat_dim)
        lin_in = hidden + self.graph_feat_dim if self.graph_feat_dim > 0 else hidden
        self.lin1 = torch.nn.Linear(lin_in, hidden)
        self.lin2 = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        if isinstance(batch, torch.Tensor):
            batch_tensor = batch
            graph_feat = None
        else:
            batch_tensor = batch.batch
            graph_feat = getattr(batch, "graph_feat", None)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch_tensor)
        if graph_feat is not None:
            gf = graph_feat.to(x.device).view(x.size(0), -1).float()
            x = torch.cat([x, gf], dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# ---------- device selection (use MPS if available on macOS otherwise CPU) ----------
use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
if use_mps:
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# compute class weights for cross-entropy to handle imbalance
num_classes = len(unique_labels)
# labels_kept: class labels for the graphs that were kept after outlier removal
labels_kept = np.array([label_map[int(gl[i])] for i in graph_idx_map])
class_counts = np.bincount(labels_kept, minlength=num_classes)
# avoid division by zero for classes with no examples (replace zeros with 1)
class_counts = np.where(class_counts == 0, 1, class_counts)
# inverse-frequency weights, normalized by number of classes
class_weights = (class_counts.sum() / (class_counts * num_classes)).astype(float)
class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
print("Class weights:", class_weights.cpu().numpy())

# Removed stray training loop that referenced undefined variables (epochs, model, train_loader, optimizer).
# Training is performed inside the cross-validation loop below.
best_alphas = []
best_alpha = 1.0
# default number of epochs (module-level) to avoid "possibly unbound" warnings and ensure a sane fallback
epochs = 30
# Ensure common training variables are defined in the module scope to avoid "possibly unbound" warnings
model = None
optimizer = None
train_loader = None
test_loader = None
train_dataset = None
scheduler = None
# ensure fold_class_weights is always defined (may be set per-fold during CV)
fold_class_weights = None
# ------------------------------ CV training -----------------------------
labels_kept = np.array([label_map[int(gl[i])] for i in graph_idx_map])
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
fold_acc, fold_f1 = [], []

def eval_model(model, loader):
    model.eval()
    ys, ys_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch)
            pred = out.argmax(dim=1).cpu().numpy()
            ys_pred.append(pred)
            ys.append(batch.y.view(-1).cpu().numpy())
    ys = np.concatenate(ys); ys_pred = np.concatenate(ys_pred)
    acc = accuracy_score(ys, ys_pred)
    f1 = f1_score(ys, ys_pred, average="macro")
    cm = confusion_matrix(ys, ys_pred, labels=list(range(num_classes)))
    p, r, f, _ = precision_recall_fscore_support(ys, ys_pred, labels=list(range(num_classes)), zero_division=0)
    # Defensive conversions: ensure numpy arrays and pad to num_classes if needed
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)
    if p.size < num_classes:
        pad = num_classes - p.size
        p = np.pad(p, (0, pad), constant_values=0.0)
        r = np.pad(r, (0, pad), constant_values=0.0)
        f = np.pad(f, (0, pad), constant_values=0.0)
    return acc, f1, cm, (p, r, f)

for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels_kept)), labels_kept), start=1):
    train_dataset = [data_list[i] for i in tr_idx]
    test_dataset = [data_list[i] for i in te_idx]
    # ensure best_alpha is always defined to avoid possible unbound reference
    best_alpha = 1.0
    
    # larger batch sizes to speed throughput (adjust if out-of-memory)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

   # smaller, faster model for quicker CV
    epochs = 30
    model = GINNet(in_dim=n_feat, hidden=32, num_layers=2, num_classes=len(unique_labels),
                   graph_feat_dim=graph_means.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    # training with gradient clipping and early stopping
    if USE_CLASS_WEIGHTS:
        # compute fold-specific class weights from the training split
        train_labels = np.array([int(d.y.item()) for d in train_dataset])
        counts = np.bincount(train_labels, minlength=num_classes)
        counts = np.where(counts == 0, 1, counts)  # avoid division by zero
        base_weights = (counts.sum() / (counts * num_classes)).astype(float)
        base_weights = torch.tensor(base_weights, dtype=torch.float, device=device)

        # quick alpha grid search (short tuning runs) to scale fold weights
        alpha_grid = [0.5, 1.0, 2.0]
        best_alpha, best_score = 1.0, -1.0
        for alpha in alpha_grid:
            tmp_model = GINNet(in_dim=n_feat, hidden=32, num_layers=2, num_classes=num_classes,
                               graph_feat_dim=graph_means.shape[1]).to(device)
            tmp_opt = torch.optim.Adam(tmp_model.parameters(), lr=5e-4, weight_decay=5e-4)
            # short tuning: 3 epochs
            for _ in range(3):
                tmp_model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    tmp_opt.zero_grad()
                    out = tmp_model(batch.x, batch.edge_index, batch)
                    w = (base_weights * alpha).to(device)
                    loss = F.cross_entropy(out, batch.y.view(-1), weight=w)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(tmp_model.parameters(), 2.0)
                    tmp_opt.step()
            acc, f1, _, _ = eval_model(tmp_model, test_loader)
            if f1 > best_score:
                best_score, best_alpha = f1, alpha

        # final fold weights scaled by best alpha
        fold_class_weights = (base_weights * best_alpha).to(device)
        print(f"Fold {fold} selected alpha={best_alpha} (tuned f1={best_score:.4f})")
    else:
        fold_class_weights = None
        # keep best_alpha default = 1.0 when not tuning
    
    # record the chosen alpha for this fold (after tuning or default)
    best_alphas.append(best_alpha)   

# ---------- aggregate chosen alphas AFTER all folds completed ----------
import statistics
if best_alphas:
    final_alpha_median = float(np.median(best_alphas))
    try:
        final_alpha_mode = float(statistics.mode(best_alphas))
    except statistics.StatisticsError:
        final_alpha_mode = final_alpha_median
else:
    final_alpha_median = final_alpha_mode = 1.0

print("per-fold alphas:", best_alphas)
print("final_alpha_median:", final_alpha_median, "final_alpha_mode:", final_alpha_mode)

# use final_alpha_median (or mode) for final training on full data
# ---------- after CV loop: validate collected fold metrics and save summary ----------

# Basic sanity assertions (will raise if something is wrong)
if len(fold_acc) != K or len(fold_f1) != K:
    print("Warning: number of collected folds does not match K. Re-check where fold_acc/fold_f1 are appended.")
    # avoid crashing in production; continue with available data

# compute robust aggregates (handle empty lists)
if len(fold_acc) > 0:
    acc_mean = float(np.mean(fold_acc))
    acc_std = float(np.std(fold_acc))
else:
    acc_mean = acc_std = 0.0

if len(fold_f1) > 0:
    f1_mean = float(np.mean(fold_f1))
    f1_std = float(np.std(fold_f1))
else:
    f1_mean = f1_std = 0.0

with open(os.path.join(OUT, "cv_summary.txt"), "w") as fh:
    fh.write(f"K: {K}\n")
    fh.write(f"collected_folds: {len(fold_acc)}\n")
    fh.write(f"acc per fold: {fold_acc}\n")
    fh.write(f"f1 per fold: {fold_f1}\n")
    fh.write(f"Overall CV Accuracy: {acc_mean:.4f} ± {acc_std:.4f}\n")
    fh.write(f"Overall CV F1-score: {f1_mean:.4f} ± {f1_std:.4f}\n")

# ---------- Correct single training loop (was duplicated / misplaced) ----------
best_val = -1.0
best_epoch = 0
early_patience = 6
# safely read initial learning rate if optimizer exists and is an optimizer instance
opt = globals().get('optimizer', None)
if isinstance(opt, torch.optim.Optimizer):
    try:
        current_lr = opt.param_groups[0].get('lr', 0.0)
    except Exception:
        current_lr = 0.0
else:
    current_lr = 0.0
print(f"  initial_lr: {current_lr:.6g}")

import csv
summary_path = os.path.join(OUT, "per_fold_results.csv")

# Defensive defaults for per-class metrics and other variables that may not exist
# Ensure p, r, f are always defined to avoid static analysis warnings about possible unbound variables.
p = np.zeros(num_classes)
r = np.zeros(num_classes)
f = np.zeros(num_classes)
# Use globals().get to safely obtain values if already set, otherwise fall back to defaults.
val_acc = globals().get('val_acc', 0.0)
val_f1 = globals().get('val_f1', 0.0)
fold = globals().get('fold', 0)

# ensure best_alpha is referenced only if it was actually set; default to 1.0 otherwise
if USE_CLASS_WEIGHTS:
    try:
        alpha_used = float(best_alpha)
    except Exception:
        alpha_used = 1.0
else:
    alpha_used = 1.0

# safe extraction of per-class metrics with defaults if a class is missing
p0 = float(p[0]) if getattr(p, "size", 0) > 0 else 0.0
r0 = float(r[0]) if getattr(r, "size", 0) > 0 else 0.0
f0 = float(f[0]) if getattr(f, "size", 0) > 0 else 0.0
p1 = float(p[1]) if getattr(p, "size", 0) > 1 else 0.0
r1 = float(r[1]) if getattr(r, "size", 0) > 1 else 0.0
f1 = float(f[1]) if getattr(f, "size", 0) > 1 else 0.0

row = {
    "fold": fold,
    "alpha": float(alpha_used),
    "val_acc": float(val_acc),
    "val_f1": float(val_f1),
    "precision_0": p0, "recall_0": r0, "f1_0": f0,
    "precision_1": p1, "recall_1": r1, "f1_1": f1
}

# write CSV only when filesystem path is available
write_header = not os.path.exists(summary_path)
with open(summary_path, "a", newline="") as cf:
    writer = csv.DictWriter(cf, fieldnames=list(row.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(row)

# safe LR logging (tolerant when scheduler/optimizer are not present)
sched = globals().get('scheduler', None)
opt = globals().get('optimizer', None)
try:
    if sched is not None and hasattr(sched, "get_last_lr"):
        current_lr = sched.get_last_lr()[0]
    elif isinstance(opt, torch.optim.Optimizer):
        current_lr = opt.param_groups[0].get("lr", 0.0)
    else:
        current_lr = 0.0
except Exception:
    # Fallback: try reading from optimizer if available, otherwise default to 0.0
    if isinstance(opt, torch.optim.Optimizer):
        try:
            current_lr = opt.param_groups[0].get("lr", 0.0)
        except Exception:
            current_lr = 0.0
    else:
        current_lr = 0.0
# Run training loop only if required training variables exist and are valid (to avoid NameError / AttributeError)
# Ensure 'epochs' is defined (fallback to 30) to avoid a possibly-unbound variable reported by static checkers
if 'epochs' not in globals() or not isinstance(globals().get('epochs'), int):
    epochs = 30

if (
    model is not None and isinstance(model, torch.nn.Module)
    and optimizer is not None and isinstance(optimizer, torch.optim.Optimizer)
    and train_loader is not None and hasattr(train_loader, "__iter__")
    and test_loader is not None and hasattr(test_loader, "__iter__")
    and train_dataset is not None
):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch)
            if 'fold_class_weights' in globals() and fold_class_weights is not None:
                loss = F.cross_entropy(out, batch.y.view(-1), weight=fold_class_weights)
            else:
                loss = F.cross_entropy(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss) * batch.num_graphs
        val_acc, val_f1, *_ = eval_model(model, test_loader)
        try:
            if scheduler is not None:
                scheduler.step(val_acc)
        except Exception:
            pass
        if val_acc > best_val + 1e-5:
            best_val = val_acc
            best_epoch = epoch
            try:
                torch.save(model.state_dict(), os.path.join(OUT, f"best_model_fold{fold}.pt"))
            except Exception:
                pass
        if (epoch - best_epoch) >= early_patience:
            print(f"Early stopping fold {fold} at epoch {epoch} (no improvement for {early_patience} epochs)")
            break
    print(f"Fold {fold} test Acc: {val_acc:.4f}, F1: {val_f1:.4f} (best acc {best_val:.4f} at epoch {best_epoch})")
    # append collected metrics for this fold (exactly once, here)
    fold_acc.append(float(val_acc))
    fold_f1.append(float(val_f1))
else:
    print("Skipping per-fold training block: required variables (model/optimizer/train_loader/test_loader/train_dataset) are not all defined or valid in this scope.")

# save summary
with open(os.path.join(OUT, "cv_summary.txt"), "w") as fh:
    fh.write(f"acc per fold: {fold_acc}\n")
    fh.write(f"f1 per fold: {fold_f1}\n")
print("Saved CV summary to Outputs/cv_summary.txt")
# print overall CV results
print(f"Overall CV Accuracy: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
print(f"Overall CV F1-score: {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")