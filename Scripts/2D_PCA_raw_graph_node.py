import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs"))
os.makedirs(OUT, exist_ok=True)

INPATH = os.path.join(ROOT, "PROTEINS_node_attributes.txt")
GI_PATH = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GL_PATH = os.path.join(ROOT, "PROTEINS_graph_labels.txt")

# read node attributes (try comma, then whitespace)
def read_nodes(path):
    for sep in (r'\s*,\s*', r'\s+'):
        try:
            import pandas as pd
            df = pd.read_csv(path, sep=sep, engine='python', header=None, comment='#')
            return df.values.astype(float)
        except Exception:
            pass
    # final fallback: plain numpy (comma)
    try:
        return np.loadtxt(path, delimiter=',').astype(float)
    except Exception as e:
        raise RuntimeError(f"Failed to read node attributes from {path}: {e}")

X = read_nodes(INPATH)               # nodes x features
gi = np.loadtxt(GI_PATH, dtype=int)  # node -> graph id (1..G)
gl = np.loadtxt(GL_PATH, dtype=int)  # graph labels (len = G)

# node-level features (raw)
X_nodes = X.astype(float)

# per-graph aggregated features (mean over nodes)
G = int(gi.max())
graph_feats = np.vstack([X[gi == g].mean(axis=0) for g in range(1, G+1)])

# map node -> graph label (assumes gl indexed 1..G)
node_labels = np.array([int(gl[g-1]) for g in gi], dtype=int)

# Standardize and PCA(2) for both representations
scaler_nodes = StandardScaler().fit(X_nodes)
Xn_s = scaler_nodes.transform(X_nodes)
pca_nodes = PCA(n_components=2).fit(Xn_s)
pcs_nodes = pca_nodes.transform(Xn_s)
expl_nodes = pca_nodes.explained_variance_ratio_

scaler_graph = StandardScaler().fit(graph_feats)
Xg_s = scaler_graph.transform(graph_feats)
pca_graph = PCA(n_components=2).fit(Xg_s)
pcs_graph = pca_graph.transform(Xg_s)
expl_graph = pca_graph.explained_variance_ratio_

# plotting colors based on graph labels
unique_labels, counts = np.unique(node_labels, return_counts=True)
unique_labels = unique_labels.astype(int)
cmap = plt.get_cmap("tab10")
color_map = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}

# side-by-side 2D scatter: left = node-level, right = per-graph
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

ax = axes[0]
for lab in unique_labels:
    mask = (node_labels == lab)
    ax.scatter(pcs_nodes[mask, 0], pcs_nodes[mask, 1], s=6, alpha=0.5,
               color=color_map[lab], label=f"class {lab} ({mask.sum()})")
ax.set_xlabel(f"PC1 ({expl_nodes[0]*100:.2f}%)")
ax.set_ylabel(f"PC2 ({expl_nodes[1]*100:.2f}%)")
ax.set_title("Node-level PCA (raw features)")
ax.grid(True); ax.legend(fontsize="small")

ax = axes[1]
for lab in np.unique(gl.astype(int)):
    mask = (gl == lab)
    ax.scatter(pcs_graph[mask, 0], pcs_graph[mask, 1], s=50, alpha=0.9,
               color=color_map.get(int(lab), "k"), label=f"class {int(lab)} ({mask.sum()})")
ax.set_xlabel(f"PC1 ({expl_graph[0]*100:.2f}%)")
ax.set_ylabel(f"PC2 ({expl_graph[1]*100:.2f}%)")
ax.set_title("Per-graph PCA (per-graph means of raw features)")
ax.grid(True); ax.legend(fontsize="small")

out_png = os.path.join(OUT, "pca_side_by_side_raw_2d.png")
plt.savefig(out_png, dpi=200)
plt.close()

# save PC scores
nodes_out = os.path.join(OUT, "PCA_nodes_raw_2d_scores.csv")
graphs_out = os.path.join(OUT, "PCA_graphs_raw_2d_scores.csv")
np.savetxt(nodes_out, pcs_nodes, delimiter=',', header="PC1,PC2", comments='')
np.savetxt(graphs_out, pcs_graph, delimiter=',', header="PC1,PC2", comments='')

print("Saved plot:", out_png)
print("Node scores:", nodes_out)
print("Graph scores:", graphs_out)
print("Explained variance (nodes) PC1,PC2:", expl_nodes[0], expl_nodes[1])
print("Explained variance (graphs) PC1,PC2:", expl_graph[0], expl_graph[1])
