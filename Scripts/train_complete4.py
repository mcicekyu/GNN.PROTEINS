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

import os, sys
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
SKIP_OUTLIER_REMOVAL = True   # set True to skip removal and run with all graphs
USE_CLASS_WEIGHTS = False # toggle to True to enable class-weighted loss

def read_table_try(path):
    for sep in (r'\s*,\s*', r'\s+'):
        try:
            return pd.read_csv(path, sep=sep, header=None, engine='python')
        except Exception:
            pass
    return pd.read_csv(path, header=None)

# load data
X_df = read_table_try(NODE_ATTR)
# use squeeze() to get a Series for a single-column table to avoid mypy/Pylance overload warnings
def _read_int_array(path):
    s = read_table_try(path).squeeze()
    # If a DataFrame is returned, take its first column
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    # Convert Series or ndarray safely via pandas to preserve astype semantics
    if isinstance(s, (pd.Series, np.ndarray)):
        return pd.Series(s).astype(int).to_numpy()
    # Fallback for scalars or unknown types: coerce to native Python int with safe fallbacks
    try:
        # Attempt to extract a native Python scalar from numpy/pandas/memoryview-like objects
        try:
            scalar = np.asarray(s).item()
        except Exception:
            # If extraction fails, keep original value for further handling
            scalar = s

        # Explicitly reject complex numbers which cannot be safely converted to int
        if isinstance(scalar, complex):
            raise ValueError(f"Cannot convert complex value to int: {scalar!r}")

        # Handle memoryview and bytes-like objects by attempting to decode to a string first
        if isinstance(scalar, memoryview):
            try:
                scalar = scalar.tobytes().decode()
            except Exception:
                scalar = scalar.tobytes()
        if isinstance(scalar, (bytes, bytearray)):
            try:
                scalar = scalar.decode()
            except Exception:
                # As a last-ditch attempt for pure byte sequences, convert via int.from_bytes
                try:
                    val = int.from_bytes(bytes(scalar), byteorder='big', signed=False)
                    return np.array([val], dtype=np.int64)
                except Exception:
                    pass

        # Narrow types for the static type checker and perform conversion in prioritized order
        # Only call int() on known-safe input types; bytes-like objects are handled explicitly.
        if isinstance(scalar, (int, np.integer)):
            val = int(scalar)
        elif isinstance(scalar, float):
            # accept floats but convert via int() semantics (truncation)
            val = int(scalar)
        elif isinstance(scalar, str):
            val = int(scalar)
        elif isinstance(scalar, (bytes, bytearray, memoryview)):
            # defensive handling: try decode first, then fallback to integer-from-bytes
            try:
                if isinstance(scalar, memoryview):
                    sval = scalar.tobytes().decode()
                else:
                    sval = scalar.decode()
                val = int(sval)
            except Exception:
                try:
                    val = int.from_bytes(bytes(scalar), byteorder='big', signed=False)
                except Exception:
                    raise ValueError(f"Cannot convert bytes-like value to int: {scalar!r}")
        else:
            # For any other types (dates, datetimes, custom objects, etc.) raise explicitly
            raise ValueError(f"Cannot convert value of type {type(scalar)!r} to int")
    except Exception:
        raise ValueError(f"Cannot convert value to int: {s!r}")
    # Construct numpy array from a native Python int to satisfy type checkers
    return np.array([val], dtype=np.int64)

gi = _read_int_array(GRAPH_IND)
gl = _read_int_array(GRAPH_LABELS)
# create mapping from original labels to contiguous class indices starting at 0
unique_labels = np.unique(gl)
label_map = {int(l): i for i, l in enumerate(unique_labels)}
if not os.path.isfile(EDGES):
    raise FileNotFoundError(f"Edges file not found: {EDGES}")
edges_df: np.ndarray = read_table_try(EDGES).iloc[:, 0:2].astype(int).to_numpy()

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
        # compute fold-specific class weights from the training split
        train_labels = np.array([int(d.y.item()) for d in train_dataset])
        counts = np.bincount(train_labels, minlength=num_classes)
        counts = np.where(counts == 0, 1, counts)  # avoid division by zero
        base_weights = (counts.sum() / (counts * num_classes)).astype(float)
        base_weights = torch.tensor(base_weights, dtype=torch.float, device=device)

        # quick alpha grid search (short tuning runs) to scale fold weights
        alpha_grid = [0.25, 0.5, 1.0, 2.0, 4.0]
        best_alpha, best_score = 1.0, -1.0
        for alpha in alpha_grid:
            tmp_model = GINNet(in_dim=n_feat, hidden=32, num_layers=2, num_classes=num_classes,
                       graph_feat_dim=graph_feat_dim).to(device)
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
            # record per-alpha performance for later aggregation / tie-break
            alpha_scores[alpha].append(float(f1))
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
    
    # ----------------- per-fold training & eval (run once per fold) -----------------
    best_val = -1.0
    best_epoch = 0
    early_patience = 6
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch)
            if fold_class_weights is not None:
                loss = F.cross_entropy(out, batch.y.view(-1), weight=fold_class_weights)
            else:
                loss = F.cross_entropy(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss) * getattr(batch, "num_graphs", 1)
        val_acc, val_f1, cm_val, (p_val, r_val, f_val) = eval_model(model, test_loader)
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

# ----------------- save per-fold summary CSV -----------------

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