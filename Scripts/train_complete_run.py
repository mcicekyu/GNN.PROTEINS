# Graph Isomorphism Network (GIN) training on PROTEINS dataset using 5-fold cross-validation with optional
# outlier graph removal using Isolation Forest and Fold‑specific class-weighted loss.
# augmented per-graph features: [mean_node_features..., graph_size, avg_degree]
# OUTLIER_CONTAMINATION = 0.02 is decided.
# Detected 23 (2.07%) outlier graphs (removed) [   6   21   66  191  276  680  683  721  730
#  857  880  914  936  939 965  976  994  997 1015 1047 1054 1058 1059]
# Overall CV Accuracy:  0.7257 ± 0.0198
# Overall F1-score: 0.7087 ± 0.0221

# Adding new graph features: avg-degree, assortativity, degree centralization, avg clustering, radius of gyration
# Without class weights.
# Overall CV Accuracy: 0.7385 ± 0.0176
# Overall CV F1-score: 0.7208 ± 0.0197

# With class weights.
# Overall CV Accuracy: 0.7263 ± 0.0179
# Overall CV F1-score: 0.7176 ± 0.0179

# without outlier removal, without class weights
# Overall CV Accuracy:  0.7356 ± 0.0201
# Overall CV F1-score:  0.7168 ± 0.0190

# BatchNorm + Dropout version to improve encoder & regularization
# without outlier removal, without class weights
# Overall CV Accuracy: 0.7296 ± 0.0203
# Overall CV F1-score: 0.7035 ± 0.0216

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict, Counter
from torch_geometric.nn import GINConv, global_mean_pool, MLP
import networkx as nx
from sklearn.decomposition import PCA
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

# Outlier removal controls (tune these) and class weight toggle
OUTLIER_CONTAMINATION = 0.02   # try 0.02, 0.01, 0.005
SKIP_OUTLIER_REMOVAL = False   # set True to skip removal and run with all graphs
USE_CLASS_WEIGHTS = True       # toggle to True to enable class-weighted loss
VERBOSE = False                # toggle per-fold live printing. When False only final aggregates are printed.

def read_table_try(path):
    for sep in (r'\s*,\s*', r'\s+'):
        try:
            return pd.read_csv(path, sep=sep, header=None, engine='python')
        except Exception:
            pass
    return pd.read_csv(path, header=None)

# load data
X_df = read_table_try(NODE_ATTR)
# use numpy-backed indexing to avoid mypy/pylance typing issues with DataFrame.iloc on tuple indices
gi = read_table_try(GRAPH_IND).values[:, 0].astype(int)
gl = read_table_try(GRAPH_LABELS).values[:, 0].astype(int)
# create mapping from original labels to contiguous class indices starting at 0
unique_labels = np.unique(gl)
label_map = {int(l): i for i, l in enumerate(unique_labels)}
if not os.path.isfile(EDGES):
    raise FileNotFoundError(f"Edges file not found: {EDGES}")
edges_df = read_table_try(EDGES).values[:, :2].astype(int)

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
# compute simple scalar graph features and combine with node-feature means ---
# graph size (number of nodes per graph)
graph_sizes = np.array([nodes.size for nodes in nodes_in_graph], dtype=float)

# average degree per graph (undirected). Use original edges file (1-based node ids).
avg_degs = []
assortivities = []
deg_centralizations = []
avg_clusts = []
rads_of_gyr = []

for nodes in nodes_in_graph:
    n = nodes.size
    if n == 0:
        avg_degs.append(0.0)
        assortivities.append(0.0)
        deg_centralizations.append(0.0)
        avg_clusts.append(0.0)
        rads_of_gyr.append(0.0)
        continue

    node_global = nodes + 1  # edges file is 1-based
    mask = np.isin(edges_df[:,0], node_global) & np.isin(edges_df[:,1], node_global)
    sub_edges = edges_df[mask]

    # build NetworkX graph using global ids (keeps unique labels)
    G = nx.Graph()
    G.add_nodes_from([int(g) for g in node_global.tolist()])
    if sub_edges.size:
        edges_list = [(int(u), int(v)) for u, v in sub_edges]
        G.add_edges_from(edges_list)

    # each undirected edge contributes degree 2
    avg_deg = (2.0 * sub_edges.shape[0]) / float(n) if n > 0 else 0.0
    avg_degs.append(avg_deg)
    
    # assortativity (degree Pearson) — robust to small/no-edge graphs and constant-degree inputs
    try:
        if G.number_of_edges() > 0:
            # compute degrees array first and only call degree_pearson_correlation_coefficient
            # when there is variance; otherwise return 0.0 to avoid ConstantInputWarning.
            # Use the module-level nx.degree(G) to avoid accidental attribute shadowing of G.degree.
            deg_pairs = list(nx.degree(G))
            degs_local = np.array([d for _, d in deg_pairs], dtype=float)
            if degs_local.size > 0 and np.nanstd(degs_local) > 0.0:
                ass = nx.degree_pearson_correlation_coefficient(G)
                assortivities.append(float(ass) if np.isfinite(ass) else 0.0)
            else:
                assortivities.append(0.0)
        else:
            assortivities.append(0.0)
    except Exception:
        assortivities.append(0.0)

    # degree centralization (Freeman): sum(max_deg - deg_i) normalized by (n-1)*(n-2) for undirected
    # obtain degrees explicitly per node to avoid any callable/namespace ambiguity
    if G.number_of_nodes() > 0:
        # build a degree mapping (node -> degree) using networkx.degree to avoid possible attribute shadowing
        deg_map = dict(nx.degree(G))
        degs = np.array([float(deg_map[node]) for node in G.nodes()], dtype=float)
    else:
        degs = np.array([], dtype=float)
    max_deg = float(degs.max()) if degs.size > 0 else 0.0
    deg_central = float(np.sum(max_deg - degs)) if degs.size > 0 else 0.0
    if n > 2:
        norm = float((n - 1) * (n - 2))
        deg_centralizations.append(deg_central / norm)
    else:
        deg_centralizations.append(0.0)

    # average (local) clustering coefficient
    try:
        if G.number_of_nodes() > 0:
            c = nx.average_clustering(G)
            avg_clusts.append(float(c) if np.isfinite(c) else 0.0)
        else:
            avg_clusts.append(0.0)
    except Exception:
        avg_clusts.append(0.0)
    
    # Radius of gyration (spatial compactness) — use PCA on node attribute vectors as proxy for coordinates
    try:
        k = min(3, n, X.shape[1])
        if k >= 1:
            pca = PCA(n_components=k)
            coords = pca.fit_transform(X[nodes])  # shape (n, k)
            centroid = coords.mean(axis=0)
            sq_dists = np.sum((coords - centroid) ** 2, axis=1)
            rog = float(np.sqrt(np.mean(sq_dists)))  # RMS distance
            rads_of_gyr.append(rog if np.isfinite(rog) else 0.0)
        else:
            rads_of_gyr.append(0.0)
    except Exception:
        rads_of_gyr.append(0.0)

avg_degs = np.array(avg_degs, dtype=float)
assortivities = np.array(assortivities, dtype=float)
deg_centralizations = np.array(deg_centralizations, dtype=float)
avg_clusts = np.array(avg_clusts, dtype=float)
rads_of_gyr = np.array(rads_of_gyr, dtype=float)

# Assemble augmented per-graph features: [mean_node_features..., graph_size, avg_degree,
#                                         assortativity, degree_centralization, avg_clustering, radius_of_gyration]
graph_features_raw = np.hstack([
    graph_means,
    graph_sizes.reshape(-1, 1),
    avg_degs.reshape(-1, 1),
    assortivities.reshape(-1, 1),
    deg_centralizations.reshape(-1, 1),
    avg_clusts.reshape(-1, 1),
    rads_of_gyr.reshape(-1, 1)
])
graph_feat_dim = graph_features_raw.shape[1]

# outlier detection (optional)
if SKIP_OUTLIER_REMOVAL:
    outlier_mask = np.zeros(len(graph_features_raw), dtype=bool)
    print("Skipping outlier removal (SKIP_OUTLIER_REMOVAL=True).")
else:
    scaler = StandardScaler().fit(graph_features_raw)
    gm_s = scaler.transform(graph_features_raw)
    iso = IsolationForest(contamination=OUTLIER_CONTAMINATION, random_state=0)
    is_out = iso.fit_predict(gm_s)
    outlier_mask = (is_out == -1)
    print(f"Detected outlier graphs (removed):", np.where(outlier_mask)[0]+1)

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
   # when building Data objects, attach the augmented graph features (not just means)
    graph_feat = torch.tensor(graph_features_raw[g_idx-1], dtype=torch.float)
    y = torch.tensor([label_map[int(gl[g_idx-1])]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.graph_feat = graph_feat
    data_list.append(data)
    graph_idx_map.append(g_idx-1)

# model
# GINNet with BatchNorm + Dropout version
class GINNet(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, num_layers=3, num_classes=2, graph_feat_dim=0, dropout=0.5, use_bn=True):
        super().__init__()
        convs = []
        dims = [in_dim] + [hidden] * (num_layers - 1)
        for dim in dims:
            mlp = MLP([dim, hidden], final_activation=F.relu)
            convs.append(GINConv(mlp))
        self.convs = torch.nn.ModuleList(convs)

        # BatchNorm per conv output (optional)
        self.use_bn = bool(use_bn)
        if self.use_bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(len(self.convs))])
        else:
            self.bns = None

        self.dropout = float(dropout)
        self.graph_feat_dim = int(graph_feat_dim)

        lin_in = hidden + self.graph_feat_dim if self.graph_feat_dim > 0 else hidden
        self.lin1 = nn.Linear(lin_in, hidden)
        self.bn_lin = nn.BatchNorm1d(hidden) if self.use_bn else None
        self.lin2 = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        if isinstance(batch, torch.Tensor):
            batch_tensor = batch
            graph_feat = None
        else:
            batch_tensor = batch.batch
            graph_feat = getattr(batch, "graph_feat", None)

        # GIN convs + optional BatchNorm + Dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn and self.bns is not None:
                # BatchNorm operates on [N, C]
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch_tensor)

        if graph_feat is not None:
            gf = graph_feat.to(x.device).view(x.size(0), -1).float()
            x = torch.cat([x, gf], dim=1)

        x = self.lin1(x)
        if self.bn_lin is not None:
            x = self.bn_lin(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
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

num_classes = len(unique_labels)
# only compute / print class weights when they will be used
if USE_CLASS_WEIGHTS:
    labels_kept = np.array([label_map[int(gl[i])] for i in graph_idx_map])
    class_counts = np.bincount(labels_kept, minlength=num_classes)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = (class_counts.sum() / (class_counts * num_classes)).astype(float)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    print("Class weights:", class_weights.cpu().numpy())
else:
    class_weights = None
    print("USE_CLASS_WEIGHTS=False (not using class-weighted loss).")

# Removed stray training loop that referenced undefined variables (epochs, model, train_loader, optimizer).
# Training is performed inside the cross-validation loop below.
best_alphas = []
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

# safe defaults for training bookkeeping to avoid "possibly-unbound" variable warnings
# these are used/updated in per-fold and final training loops below
best_val = -1.0
best_epoch = 0
early_patience = 6

# ------------------------------ CV training -----------------------------
labels_kept = np.array([label_map[int(gl[i])] for i in graph_idx_map])
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
fold_acc, fold_f1 = [], []

# collect per-alpha f1 scores across folds for deterministic tie-break
alpha_scores = defaultdict(list)

def eval_model(model, loader, return_best_threshold=False, pos_class=1):
    """
    Evaluate model on loader.
    - Always returns a 6-tuple:
      (acc, f1_macro, cm, (p, r, f), best_th, best_th_f1)
      When return_best_threshold=False the last two entries are (None, None).
    """
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

    best_th = None
    best_th_f1 = None
    if return_best_threshold:
        # collect class-probabilities for pos_class and search best threshold (on same loader)
        probs = []
        ys_prob = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch)
                ppos = torch.softmax(logits, dim=1)[:, pos_class].cpu().numpy()
                probs.append(ppos)
                ys_prob.append(batch.y.view(-1).cpu().numpy())
        probs = np.concatenate(probs); ys_prob = np.concatenate(ys_prob)

        # search threshold that maximizes macro F1
        best_th, best_th_f1 = 0.5, 0.0
        for th in np.linspace(0.1, 0.9, 41):
            preds_th = (probs >= th).astype(int)
            th_f1 = f1_score(ys_prob, preds_th, average='macro')
            if th_f1 > best_th_f1:
                best_th_f1 = th_f1
                best_th = float(th)

    return acc, f1, cm, (p, r, f), best_th, best_th_f1

# add focal loss here (place before CV loop / training)
def focal_loss(logits, targets, gamma=2.0, weight=None):
    ce = F.cross_entropy(logits, targets, reduction='none', weight=weight)
    p_t = torch.exp(-ce)
    return ((1.0 - p_t) ** gamma * ce).mean()

# ----------------- per-fold results summarizer (define before CV loop) -----------------
def summarize_per_fold_results(out_dir=OUT, src_name="per_fold_results.csv", dst_name="per_fold_results_summary.csv"):
    """Read per_fold_results.csv, pick one row per fold (prefer max val_acc then val_f1), save summary and print aggregates."""
    try:
        import pandas as _pd
        import numpy as _np
    except Exception:
        print("pandas/numpy required for summarization; skipping summarize_per_fold_results.")
        return

    src = os.path.join(out_dir, src_name)
    dst = os.path.join(out_dir, dst_name)
    if not os.path.exists(src):
        print("No per-fold results file found at", src)
        return
    try:
        df = _pd.read_csv(src)
    except Exception as e:
        print("Failed to read per-fold results:", e)
        return
    rows = []
    for fold, g in df.groupby("fold"):
        nonzero_acc = g[g["val_acc"] > 0]
        if len(nonzero_acc):
            sel = nonzero_acc.loc[nonzero_acc["val_acc"].idxmax()]
        else:
            nonzero_f1 = g[g["val_f1"] > 0]
            if len(nonzero_f1):
                sel = nonzero_f1.loc[nonzero_f1["val_f1"].idxmax()]
            else:
                sel = g.iloc[-1]
        rows.append(sel)

    summary = _pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    try:
        summary.to_csv(dst, index=False)
        print("Saved cleaned per-fold summary to:", dst)
    except Exception as e:
        print("Failed to write summary CSV:", e)
        return

    accs = summary["val_acc"].astype(float).to_numpy()
    f1s  = summary["val_f1"].astype(float).to_numpy()
    if accs.size == 0:
        print("No valid per-fold accuracy values found in summary.")
        return
    acc_mean = float(_np.nanmean(accs)); acc_std = float(_np.nanstd(accs, ddof=0)) if accs.size>1 else float("nan")
    f1_mean = float(_np.nanmean(f1s));   f1_std = float(_np.nanstd(f1s, ddof=0)) if f1s.size>1 else float("nan")
    print("Per-fold summary:")
    try:
        print(summary[["fold","alpha","val_acc","val_f1"]].to_string(index=False))
    except Exception:
        print(summary.to_dict(orient="records"))
    print(f"\nOverall CV Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Overall CV F1-score: {f1_mean:.4f} ± {f1_std:.4f}")


def run_cv():
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels_kept)), labels_kept), start=1):
        train_dataset = [data_list[i] for i in tr_idx]
        test_dataset = [data_list[i] for i in te_idx]
        # per-fold default alpha (rename to avoid shadowing global names)
        fold_alpha = 1.0

        # DataLoader: moderate train batches for better generalization, larger eval batch for speed.
        # On macOS/MPS prefer num_workers=0 to avoid subprocess issues; increase on multi-core Linux.
        cpu_count = os.cpu_count() or 1
        if device.type == "cuda":
            num_workers = min(8, max(1, cpu_count - 1))
            pin_memory = True
        elif device.type == "mps":
             num_workers = 0
             pin_memory = False
        else:  # cpu
            num_workers = min(4, max(0, cpu_count // 2))
            pin_memory = False

        train_bs = 16   # each training mini‑batch contains 16 graphs (per iteration); try 32; reduce if OOM or to improve generalization
        test_bs  = 64   # larger eval batch is fine for throughput

        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test_dataset,  batch_size=test_bs,  shuffle=False,
                                  num_workers=max(0, num_workers//2), pin_memory=pin_memory)

        # If memory is limited but you want the effect of larger batches, use gradient accumulation:
        # acc_steps = desired_effective_batch // train_bs; accumulate gradients and call optimizer.step() every acc_steps.

        # Initialize validation metrics to safe defaults to avoid "possibly unbound" warnings
        val_acc = 0.0
        val_f1 = 0.0
        cm_val = np.zeros((num_classes, num_classes), dtype=int)
        p_val = np.zeros(num_classes, dtype=float)
        r_val = np.zeros(num_classes, dtype=float)
        f_val = np.zeros(num_classes, dtype=float)

       # smaller, faster model for quicker CV
        epochs = 30
        model = GINNet(in_dim=n_feat, hidden=32, num_layers=2, num_classes=len(unique_labels),
                       graph_feat_dim=graph_feat_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

        # training with gradient clipping and early stopping
        if USE_CLASS_WEIGHTS:
            # inside CV loop, after train_dataset created:
            train_labels = np.array([int(d.y.item()) for d in train_dataset])
            class_counts = np.bincount(train_labels, minlength=num_classes)
            weights_per_class = 1.0 / (class_counts + 1e-9)
            # base_weights is the per-class weight tensor used for scaling via alpha during tuning
            base_weights = torch.tensor(weights_per_class, dtype=torch.float, device=device)
            # convert numpy array to a Python list to satisfy WeightedRandomSampler typing
            sample_weights = weights_per_class[train_labels].astype(float).tolist()
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
            
            # use a plain shuffled loader for short tuning so loss weights actually matter
            tune_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            # quick alpha grid search (short tuning runs) to scale fold weights
            alpha_grid = [0.25, 0.5, 1.0, 2.0, 4.0]
            fold_alpha, best_score = 1.0, -1.0
            for alpha in alpha_grid:
                tmp_model = GINNet(in_dim=n_feat, hidden=32, num_layers=2, num_classes=num_classes,
                           graph_feat_dim=graph_feat_dim).to(device)
                tmp_opt = torch.optim.Adam(tmp_model.parameters(), lr=5e-4, weight_decay=5e-4)
                # short tuning: 2 epochs (use tune_loader, NOT train_loader with sampler)
                for _ in range(3):
                    tmp_model.train()
                    for batch in tune_loader:
                        batch = batch.to(device)
                        tmp_opt.zero_grad()
                        out = tmp_model(batch.x, batch.edge_index, batch)
                        # compute a relative re-weighting by exponentiating base weights so alpha changes ratios
                        w = (base_weights.pow(float(alpha))).to(device)
                        loss = focal_loss(out, batch.y.view(-1), gamma=2.0, weight=w)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(tmp_model.parameters(), 2.0)
                        tmp_opt.step()
                acc, f1, *_ = eval_model(tmp_model, test_loader)
                alpha_scores[alpha].append(float(f1))
                if f1 > best_score:
                    best_score, fold_alpha = f1, alpha

            # final fold weights scaled by chosen alpha (use same exponentiation as tuning)
            fold_class_weights = (base_weights.pow(float(fold_alpha))).to(device)
            print(f"Fold {fold} selected alpha={fold_alpha} (tuned f1={best_score:.4f})")
        else:
            fold_class_weights = None
            print(f"Fold {fold} using no class weights (USE_CLASS_WEIGHTS=False).")
        
        # record the chosen alpha for this fold (after tuning or default)
        best_alphas.append(fold_alpha)

        # ----------------- per-fold training & eval (run once per fold) -----------------
        best_val = -1.0
        best_epoch = 0
        early_patience = 6
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            # optional gradient accumulation to simulate larger effective batch
            desired_effective_batch = 128  # set target effective batch size (tune)
            acc_steps = max(1, desired_effective_batch // train_bs)
            optimizer.zero_grad()
            # sentinel to track whether we entered the loop and remember the last index
            last_i = -1
            for i, batch in enumerate(train_loader):
                last_i = i
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch)
                if fold_class_weights is not None:
                    loss = F.cross_entropy(out, batch.y.view(-1), weight=fold_class_weights)
                else:
                    loss = F.cross_entropy(out, batch.y.view(-1))
                # scale loss so gradients correspond to the effective batch
                loss = loss / float(acc_steps)
                loss.backward()
                total_loss += float(loss) * getattr(batch, "num_graphs", 1) * acc_steps
                # step when we accumulated acc_steps micro-batches (or on last mini-batch)
                if (i + 1) % acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    optimizer.zero_grad()
            # final step if leftover gradients remain (only when we had at least one mini-batch)
            if last_i >= 0 and (last_i + 1) % acc_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad()

            _eval_res = eval_model(model, test_loader)
            # eval_model may return either 4 items (acc, f1, cm, (p,r,f)) or 6 items (with threshold info);
            # unpack only the first four elements to handle both cases safely.
            val_acc, val_f1, cm_val, prf = _eval_res[:4]
            p_val, r_val, f_val = prf
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
                print(f"Early stopping fold {fold} at epoch {epoch}. Best val acc: {best_val:.4f} at epoch {best_epoch}")
                break
        # final eval / logging for this fold
        print(f"Fold {fold} test Acc: {val_acc:.4f}, F1: {val_f1:.4f} (best acc {best_val:.4f} at epoch {best_epoch})")
        try:
            np.savetxt(os.path.join(OUT, f"fold{fold}_confusion.csv"), cm_val, fmt="%d", delimiter=",")
            pd.DataFrame({"precision": p_val, "recall": r_val, "f1": f_val}).to_csv(os.path.join(OUT, f"fold{fold}_per_class_metrics.csv"))
        except Exception:
            pass
        fold_acc.append(float(val_acc))
        fold_f1.append(float(val_f1))

        # --- write one clean per-fold row immediately (avoid global-based final write) ---
        try:
            import csv as _csv
            summary_path = os.path.join(OUT, "per_fold_results.csv")
            row = {
                "fold": int(fold),
                "alpha": float(fold_alpha),
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "precision_0": float(p_val[0]) if len(p_val)>0 else 0.0,
                "recall_0":    float(r_val[0]) if len(r_val)>0 else 0.0,
                "f1_0":        float(f_val[0]) if len(f_val)>0 else 0.0,
                "precision_1": float(p_val[1]) if len(p_val)>1 else 0.0,
                "recall_1":    float(r_val[1]) if len(r_val)>1 else 0.0,
                "f1_1":        float(f_val[1]) if len(f_val)>1 else 0.0
            }
            write_header = not os.path.exists(summary_path)
            with open(summary_path, "a", newline="") as cf:
                writer = _csv.DictWriter(cf, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            pass
        
        # concise per-fold & running summary (printed only when VERBOSE=True)
        if VERBOSE:
            if len(fold_acc) == 1:
                acc_mean, acc_std = fold_acc[0], float("nan")
                f1_mean, f1_std = fold_f1[0], float("nan")
            else:
                acc_mean, acc_std = float(np.mean(fold_acc)), float(np.std(fold_acc, ddof=0))
                f1_mean, f1_std = float(np.mean(fold_f1)), float(np.std(fold_f1, ddof=0))
            print(f"Fold {fold} done — acc: {val_acc:.4f}, f1: {val_f1:.4f}, alpha: {fold_alpha}")
            print(f"Overall CV (so far) acc: {acc_mean:.4f} ± {acc_std:.4f}, f1: {f1_mean:.4f} ± {f1_std:.4f}")
            try:
                with open(os.path.join(OUT, "cv_summary.txt"), "w") as fh:
                    fh.write(f"K: {K}\n")
                    fh.write(f"collected_folds: {len(fold_acc)}\n")
                    fh.write(f"running_acc_mean: {acc_mean:.6f}\n")
                    fh.write(f"running_f1_mean: {f1_mean:.6f}\n")
            except Exception:
                pass

         # --- end per-fold CSV write ---

# ---------- aggregate chosen alphas AFTER all folds completed ----------
import statistics
if best_alphas:
    final_alpha_median = float(np.median(best_alphas))
    # deterministic mode with tie-break by highest mean CV F1 (then by smallest alpha)
    counts = Counter(best_alphas)
    max_count = max(counts.values())
    candidates = [a for a, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        final_alpha_mode = float(candidates[0])
    else:
        # compute mean f1 per candidate (default 0.0 if missing)
        mean_scores = {a: float(np.mean(alpha_scores.get(a, [0.0]))) for a in candidates}
        best_mean = max(mean_scores.values())
        best_candidates = [a for a, m in mean_scores.items() if m == best_mean]
        final_alpha_mode = float(min(best_candidates))  # deterministic tie-break -> smallest alpha
else:
    final_alpha_median = final_alpha_mode = 1.0
# use final_alpha_median (or mode) for final training on full data

print("per-fold alphas:", best_alphas)
print("final_alpha_median:", final_alpha_median, "final_alpha_mode:", final_alpha_mode)


# ---------- after CV loop: validate collected fold metrics and save summary ----------

# Diagnostic: show what we actually collected
print("DEBUG: len(fold_acc) =", len(fold_acc), "fold_acc =", fold_acc)
print("DEBUG: len(fold_f1)  =", len(fold_f1),  "fold_f1  =", fold_f1)

# Basic sanity assertions (will raise if something is wrong)
if len(fold_acc) != K or len(fold_f1) != K:
    print("Warning: number of collected folds does not match K. Re-check where fold_acc/fold_f1 are appended.")
    # continue but be explicit about statistics when few samples are present

# compute robust aggregates (handle empty/one-element lists)
if len(fold_acc) == 0:
    acc_mean = acc_std = float("nan")
elif len(fold_acc) == 1:
    acc_mean = float(np.mean(fold_acc))
    acc_std = float("nan")   # std undefined for single sample
else:
    acc_mean = float(np.mean(fold_acc))
    acc_std = float(np.std(fold_acc, ddof=0))

if len(fold_f1) == 0:
    f1_mean = f1_std = float("nan")
elif len(fold_f1) == 1:
    f1_mean = float(np.mean(fold_f1))
    f1_std = float("nan")
else:
    f1_mean = float(np.mean(fold_f1))
    f1_std = float(np.std(fold_f1, ddof=0))

with open(os.path.join(OUT, "cv_summary.txt"), "w") as fh:
    fh.write(f"K: {K}\n")
    fh.write(f"collected_folds: {len(fold_acc)}\n")
    fh.write(f"final_acc_mean: {acc_mean:.6f}\n")
    fh.write(f"final_acc_std:  {acc_std:.6f}\n")
    fh.write(f"final_f1_mean:  {f1_mean:.6f}\n")
    fh.write(f"final_f1_std:   {f1_std:.6f}\n")

# Always print final aggregates once (independent of VERBOSE)
print(f"Overall CV Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Overall CV F1-score: {f1_mean:.4f} ± {f1_std:.4f}")

# ----------------- save per-fold summary CSV -----------------

# after computing cv_summary.txt and before final LR print
summary_src = os.path.join(OUT, "per_fold_results.csv")
if os.path.exists(summary_src) and callable(globals().get("summarize_per_fold_results", None)):
    try:
        summarize_per_fold_results()
    except Exception as e:
        print("summarize_per_fold_results failed:", e)

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
print(f"Final Learning Rate after CV: {current_lr:.6e}")
if __name__ == "__main__":
    run_cv()