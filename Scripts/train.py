"""
Train a GNN for graph classification (PyTorch Geometric) with:
 - per-graph outlier detection (IsolationForest) on aggregated node features
 - stratified K-Fold CV
 - simple GIN model + global pooling
Requirements: torch, torch_geometric, scikit-learn, pandas, numpy
Run from repo root (conda env active).
"""
import os, sys, math

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, Sequential, MLP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs/GNN_complete"))
os.makedirs(OUT, exist_ok=True)

# Files (edit if needed)
NODE_ATTR = os.path.join(ROOT, "PROTEINS_node_attributes.cleaned.txt")
GRAPH_IND = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GRAPH_LABELS = os.path.join(ROOT, "PROTEINS_graph_labels.txt")
EDGES = os.path.join(ROOT, "PROTEINS_A.txt") 

# ---------- helpers: load raw files ----------
def read_table_try(path):
    for sep in (r'\s*,\s*', r'\s+'):
        try:
            return pd.read_csv(path, sep=sep, header=None, engine='python')
        except Exception:
            pass
    return pd.read_csv(path, header=None)

X_df = read_table_try(NODE_ATTR)
gi = read_table_try(GRAPH_IND).iloc[:,0].astype(int).to_numpy()
gl = read_table_try(GRAPH_LABELS).iloc[:,0].astype(int).to_numpy()
# edges: two columns of 1-based node indices; if file missing, raise
if not os.path.isfile(EDGES):
    raise FileNotFoundError(f"Edges file not found: {EDGES}")
edges_df = read_table_try(EDGES).iloc[:, :2].astype(int).values

# --- select top-10 attributes (provided indices, zero-based) ---
TOP10_IDX = [23, 19, 0, 1, 21, 4, 22, 24, 6, 12]
X_full = X_df.values.astype(float)
# validate indices
max_idx = X_full.shape[1] - 1
bad = [i for i in TOP10_IDX if i < 0 or i > max_idx]
if bad:
    raise IndexError(f"Top feature indices out of range for loaded attributes: {bad}")
X = X_full[:, TOP10_IDX].astype(float)
n_nodes, n_feat = X.shape
print("Selected top-10 feature indices:", TOP10_IDX)
print("Loaded nodes, selected features:", n_nodes, n_feat)
G = int(gi.max())
print("Number of graphs (G):", G)

# map graph labels to 0..K-1
unique_labels = np.unique(gl)
label_map = {v: i for i, v in enumerate(sorted(unique_labels))}
y_graphs = np.array([label_map[v] for v in gl], dtype=int)

# build per-graph node lists
nodes_in_graph = [np.where(gi == g)[0] for g in range(1, G+1)]

# build per-graph aggregated features (mean) for outlier detection
graph_means = np.vstack([X[nodes].mean(axis=0) for nodes in nodes_in_graph])

# ---------- outlier detection ----------
# scale then IsolationForest
scaler = StandardScaler().fit(graph_means)
gm_s = scaler.transform(graph_means)
iso = IsolationForest(contamination=0.02, random_state=0)  # tune contamination
is_out = iso.fit_predict(gm_s)  # -1 -> outlier, 1 -> inlier
outlier_mask = (is_out == -1)
print(f"Detected {outlier_mask.sum()} outlier graphs (will remove from training):", np.where(outlier_mask)[0]+1)

# ---------- prepare PyG Data objects ----------
# prepare dataset objects, excluding outliers
data_list = []
graph_idx_map = []  # map kept graph idx -> original graph id (1-based)
# precompute per-graph means (already have graph_means variable)
for kept_local_idx, g_idx in enumerate(range(1, G+1)):
    nodes = nodes_in_graph[g_idx-1]
    if outlier_mask[g_idx-1]:
        continue
    # select edges internal to this graph (edges are 1-based)
    node_global = nodes + 1
    mask = np.isin(edges_df[:,0], node_global) & np.isin(edges_df[:,1], node_global)
    sub_edges = edges_df[mask].copy()
    if sub_edges.size == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        global_to_local = {g: i for i, g in enumerate(node_global)}
        src = [global_to_local[int(u)] for u in sub_edges[:,0]]
        dst = [global_to_local[int(v)] for v in sub_edges[:,1]]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)  # make undirected
    x = torch.tensor(X[nodes], dtype=torch.float)
    # attach per-graph summary (mean of selected node features)
    graph_feat = torch.tensor(graph_means[g_idx-1], dtype=torch.float)
    y = torch.tensor([label_map[int(gl[g_idx-1])]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    # custom attribute gets batched automatically by PyG
    data.graph_feat = graph_feat
    data_list.append(data)
    graph_idx_map.append(g_idx-1)

# ---------- simple GIN model (updated) ----------
class GINNet(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, num_layers=3, num_classes=2, graph_feat_dim=0):
        super().__init__()
        convs = []
        dims = [in_dim] + [hidden] * (num_layers - 1)
        for dim in dims:
            mlp = MLP([dim, hidden], final_activation=F.relu)
            conv = GINConv(mlp)
            convs.append(conv)
        self.convs = torch.nn.ModuleList(convs)
        # if we concatenate graph-level features, adjust linear input
        self.graph_feat_dim = int(graph_feat_dim)
        lin_in = hidden + self.graph_feat_dim if self.graph_feat_dim > 0 else hidden
        self.lin1 = torch.nn.Linear(lin_in, hidden)
        self.lin2 = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        # Accept either a batch tensor (torch.Tensor) or a PyG Batch/Data object
        if isinstance(batch, torch.Tensor):
            batch_tensor = batch
            graph_feat = None
        else:
            # batch is a Batch object: use its .batch tensor and optional graph_feat attr
            batch_tensor = batch.batch
            graph_feat = getattr(batch, "graph_feat", None)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch_tensor)  # (num_graphs, hidden)

        if graph_feat is not None:
            # ensure proper device and shape (num_graphs, graph_feat_dim)
            gf = graph_feat.to(x.device).view(x.size(0), -1).float()
            x = torch.cat([x, gf], dim=1)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


# ---------- CV training ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
labels_kept = np.array([int(gl[i]) for i in graph_idx_map])
fold_acc = []
fold_f1 = []

# small training helper
def train_epoch(model, loader, opt):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y.view(-1))
        loss.backward()
        opt.step()
        total_loss += float(loss) * batch.num_graphs
    return total_loss / len(loader.dataset)

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
    ys = np.concatenate(ys)
    ys_pred = np.concatenate(ys_pred)
    return accuracy_score(ys, ys_pred), f1_score(ys, ys_pred, average="macro")

# ...existing code...
for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels_kept)), labels_kept), start=1):
    
    # compute class weights for loss using training split (ensure a torch.Tensor, not a numpy array)
    kept_labels = labels_kept[tr_idx]  # labels for training graphs in this fold
    classes, counts = np.unique(kept_labels, return_counts=True)
    weights_np = (len(kept_labels) / (len(classes) * counts)).astype(float)
    class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)

    # prepare train / test datasets and loaders for this fold
    train_dataset = [data_list[i] for i in tr_idx]
    test_dataset = [data_list[i] for i in te_idx]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = GINNet(in_dim=n_feat, hidden=64, num_layers=3, num_classes=len(unique_labels),
                   graph_feat_dim=graph_means.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, verbose=False)

    def train_epoch(model, loader, opt):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            # forward accepts batch.graph_feat automatically via model
            out = model(batch.x, batch.edge_index, batch)
            loss = F.cross_entropy(out, batch.y.view(-1), weight=class_weights)
            loss.backward()
            opt.step()
            total_loss += float(loss) * batch.num_graphs
        return total_loss / len(loader.dataset)

    # in epoch loop, call scheduler.step(best_val) after validation metric
    best_val = 0.0
    # ensure acc/f1 are always defined (avoids "possibly unbound" warnings)
    acc, f1 = 0.0, 0.0
    for epoch in range(1, 101):
        loss = train_epoch(model, train_loader, opt)
        acc, f1 = eval_model(model, test_loader)
        scheduler.step(acc)
        if acc > best_val:
            best_val = acc
        if epoch == 1 or epoch % 10 == 0 or epoch == 100:
            print(f"Fold {fold} Epoch {epoch:03d} Loss: {loss:.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}")

    # ...existing code...    print(f"Fold {fold} test Acc: {acc:.4f}, F1: {f1:.4f}")
    fold_acc.append(acc)
    fold_f1.append(f1)
    print(f"Fold {fold} test Acc: {acc:.4f}, F1: {f1:.4f}")
# save summary
with open(os.path.join(OUT, "cv_summary.txt"), "w") as fh:
    fh.write(f"acc per fold: {fold_acc}\n")
    fh.write(f"f1 per fold: {fold_f1}\n")
print("Saved CV summary to Outputs/cv_summary.txt")