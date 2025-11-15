import os, sys
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
from torch_geometric.nn import GINConv, global_mean_pool, MLP

# workspace paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs/GNN_complete"))
os.makedirs(OUT, exist_ok=True)

# Files 
NODE_ATTR = os.path.join(ROOT, "PROTEINS_node_attributes.txt")
GRAPH_IND = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GRAPH_LABELS = os.path.join(ROOT, "PROTEINS_graph_labels.txt")
EDGES = os.path.join(ROOT, "PROTEINS_A.txt")

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
from sklearn.preprocessing import StandardScaler
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
# ...existing code...
# outlier detection (optional)
scaler = StandardScaler().fit(graph_means)
gm_s = scaler.transform(graph_means)
iso = IsolationForest(contamination=0.02, random_state=0)
is_out = iso.fit_predict(gm_s)
outlier_mask = (is_out == -1)
print(f"Detected {outlier_mask.sum()} outlier graphs (removed):", np.where(outlier_mask)[0]+1)

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
# ---------- CV training ----------
labels_kept = np.array([label_map[int(gl[i])] for i in graph_idx_map])
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
fold_acc, fold_f1 = [], []
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
    return accuracy_score(ys, ys_pred), f1_score(ys, ys_pred, average="macro")

for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels_kept)), labels_kept), start=1):
    train_dataset = [data_list[i] for i in tr_idx]
    test_dataset = [data_list[i] for i in te_idx]
    # larger batch sizes to speed throughput (adjust if out-of-memory)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # smaller, faster model for quicker CV
    epochs = 30
    model = GINNet(in_dim=n_feat, hidden=32, num_layers=2, num_classes=len(unique_labels),
                   graph_feat_dim=graph_means.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=False)

    # training with gradient clipping and early stopping
    best_val = 0.0
    best_epoch = 0
    early_patience = 6
    val_acc, val_f1 = 0.0, 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch)
            loss = F.cross_entropy(out, batch.y.view(-1))
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss) * batch.num_graphs
        val_acc, val_f1 = eval_model(model, test_loader)
        scheduler.step(val_acc)
        # early stopping tracking
        if val_acc > best_val + 1e-5:
            best_val = val_acc
            best_epoch = epoch
            # save best model for this fold
            torch.save(model.state_dict(), os.path.join(OUT, f"best_model_fold{fold}.pt"))
        # verbose logging
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            avg_loss = total_loss / max(1, len(train_dataset))
            print(f"Fold {fold} Epoch {epoch:03d} Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} (best {best_val:.4f} @epoch {best_epoch})")
        if (epoch - best_epoch) >= early_patience:
            print(f"Early stopping fold {fold} at epoch {epoch} (no improvement for {early_patience} epochs)")
            break
    print(f"Fold {fold} test Acc: {val_acc:.4f}, F1: {val_f1:.4f} (best acc {best_val:.4f} at epoch {best_epoch})")
    fold_acc.append(val_acc); fold_f1.append(val_f1)

# save summary
with open(os.path.join(OUT, "cv_summary.txt"), "w") as fh:
    fh.write(f"acc per fold: {fold_acc}\n")
    fh.write(f"f1 per fold: {fold_f1}\n")
print("Saved CV summary to Outputs/cv_summary.txt")
# print overall CV results
print(f"Overall CV Accuracy: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
print(f"Overall CV F1-score: {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")