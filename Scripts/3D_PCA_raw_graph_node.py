import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs"))
os.makedirs(OUT, exist_ok=True)

INPATH = os.path.join(ROOT, "PROTEINS_node_attributes.txt")
GI_PATH = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GL_PATH = os.path.join(ROOT, "PROTEINS_graph_labels.txt")

def read_nodes(path):
    try:
        import pandas as pd
        # try common separators
        for sep in (r'\s*,\s*', r'\s+'):
            try:
                df = pd.read_csv(path, sep=sep, engine='python', header=None, comment='#')
                return df.values.astype(float)
            except Exception:
                pass
    except Exception:
        pass
    # fallback to numpy
    try:
        return np.loadtxt(path, delimiter=',').astype(float)
    except Exception as e:
        raise RuntimeError(f"Failed to read node attributes: {e}")

X = read_nodes(INPATH)                       # nodes x features (raw)
gi = np.loadtxt(GI_PATH, dtype=int)          # node -> graph id (1..G)
gl = np.loadtxt(GL_PATH, dtype=int)          # graph labels (len = G)

# node-level features (raw)
X_nodes = X.astype(float)

# per-graph aggregated features (mean over nodes)
G = int(gi.max())
graph_feats = np.vstack([X[gi == g].mean(axis=0) for g in range(1, G+1)])

# map node -> graph label (assumes gl indexed 1..G)
node_labels = np.array([int(gl[g-1]) for g in gi], dtype=int)

# Standardize and PCA(3) for both representations
scaler_nodes = StandardScaler().fit(X_nodes)
Xn_s = scaler_nodes.transform(X_nodes)
pca_nodes = PCA(n_components=3).fit(Xn_s)
pcs_nodes = pca_nodes.transform(Xn_s)
expl_nodes = pca_nodes.explained_variance_ratio_

scaler_graph = StandardScaler().fit(graph_feats)
Xg_s = scaler_graph.transform(graph_feats)
pca_graph = PCA(n_components=3).fit(Xg_s)
pcs_graph = pca_graph.transform(Xg_s)
expl_graph = pca_graph.explained_variance_ratio_

# plotting colors based on graph labels
unique_labels = np.unique(gl.astype(int))
cmap = plt.get_cmap("tab10")
color_map = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}

# side-by-side 3D scatter: left = node-level, right = per-graph
fig = plt.figure(figsize=(14, 6), constrained_layout=True)

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
for lab in unique_labels:
    mask = (node_labels == lab)
    if mask.sum() == 0:
        continue
    ax1.scatter(pcs_nodes[mask, 0], pcs_nodes[mask, 1], pcs_nodes[mask, 2],
                s=4, alpha=0.4, color=color_map[lab], label=f"class {int(lab)} ({mask.sum()})")
ax1.set_xlabel(f"PC1 ({expl_nodes[0]*100:.2f}%)")
ax1.set_ylabel(f"PC2 ({expl_nodes[1]*100:.2f}%)")
ax1.set_zlabel(f"PC3 ({expl_nodes[2]*100:.2f}%)")
ax1.set_title("Node-level PCA (raw features)")
ax1.legend(fontsize="small", loc="upper left")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for lab in unique_labels:
    mask = (gl == lab)
    if mask.sum() == 0:
        continue
    ax2.scatter(pcs_graph[mask, 0], pcs_graph[mask, 1], pcs_graph[mask, 2],
                s=60, alpha=0.9, color=color_map[lab], label=f"class {int(lab)} ({mask.sum()})")
ax2.set_xlabel(f"PC1 ({expl_graph[0]*100:.2f}%)")
ax2.set_ylabel(f"PC2 ({expl_graph[1]*100:.2f}%)")
ax2.set_zlabel(f"PC3 ({expl_graph[2]*100:.2f}%)")
ax2.set_title("Per-graph PCA (per-graph means of raw features)")
ax2.legend(fontsize="small", loc="upper left")

png_out = os.path.join(OUT, "pca_side_by_side_raw_3d.png")
plt.savefig(png_out, dpi=200)
plt.close()

# save PC scores
nodes_out = os.path.join(OUT, "PCA_nodes_raw_3d_scores.csv")
graphs_out = os.path.join(OUT, "PCA_graphs_raw_3d_scores.csv")
np.savetxt(nodes_out, pcs_nodes, delimiter=',', header="PC1,PC2,PC3", comments='')
np.savetxt(graphs_out, pcs_graph, delimiter=',', header="PC1,PC2,PC3", comments='')

print("Saved plot:", png_out)
print("Node scores:", nodes_out)
print("Graph scores:", graphs_out)
print("Explained variance (nodes) PC1,PC2,PC3:", tuple(expl_nodes[:3]))
print("Explained variance (graphs) PC1,PC2,PC3:", tuple(expl_graph[:3]))
