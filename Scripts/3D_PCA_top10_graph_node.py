"""
3D PCA side-by-side:
 - left: node-level PCA (top-10 features)
 - right: per-graph PCA (per-graph means of top-10 features)

Saves: Outputs/pca_side_by_side_3d.png and PC score CSVs.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs"))
os.makedirs(OUT, exist_ok=True)

INPATH = os.path.join(ROOT, "PROTEINS_node_attributes.cleaned.txt")
GI_PATH = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GL_PATH = os.path.join(ROOT, "PROTEINS_graph_labels.txt")

# load data
X = np.loadtxt(INPATH, delimiter=',')          # nodes x features
gi = np.loadtxt(GI_PATH, dtype=int)           # node -> graph id (1..G)
gl = np.loadtxt(GL_PATH, dtype=int)           # graph labels (len = G)

top10 = [22, 18, 3, 20, 21, 23, 0, 5, 1, 10]
# node-level data using top10
X_nodes = X[:, top10].astype(float)

# per-graph means using top10
G = int(gi.max())
graph_feats = np.vstack([X[gi == g][:, top10].mean(axis=0) for g in range(1, G+1)])

# Standardize and PCA(3) for both
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

# map node -> graph label
node_labels = np.array([int(gl[g-1]) for g in gi], dtype=int)
unique_labels = np.unique(gl.astype(int))
cmap = plt.get_cmap("tab10")
color_map = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}

# Plot side-by-side 3D scatter
fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
for lab in unique_labels:
    mask = (node_labels == lab)
    ax1.scatter(pcs_nodes[mask,0], pcs_nodes[mask,1], pcs_nodes[mask,2],
                s=6, alpha=0.5, color=color_map[lab], label=f"class {int(lab)} ({mask.sum()})")
ax1.set_title(f"Node-level (top-10) — PC1 {expl_nodes[0]*100:.2f}%, PC2 {expl_nodes[1]*100:.2f}%")
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.set_zlabel("PC3")
ax1.legend(loc="upper left", fontsize="small")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for lab in unique_labels:
    mask = (gl == lab)
    ax2.scatter(pcs_graph[mask,0], pcs_graph[mask,1], pcs_graph[mask,2],
                s=50, alpha=0.9, color=color_map[lab], label=f"class {int(lab)} ({mask.sum()})")
ax2.set_title(f"Per-graph (top-10 means) — PC1 {expl_graph[0]*100:.2f}%, PC2 {expl_graph[1]*100:.2f}%")
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")
ax2.legend(loc="upper left", fontsize="small")

plt.tight_layout()
png_out = os.path.join(OUT, "pca_side_by_side_3d.png")
plt.savefig(png_out, dpi=200)
plt.close()

# save PC scores
nodes_out = os.path.join(OUT, "PCA_nodes_top10_3d_scores.csv")
graph_out = os.path.join(OUT, "PCA_graphs_top10_3d_scores.csv")
np.savetxt(nodes_out, pcs_nodes, delimiter=',', header="PC1,PC2,PC3", comments='')
np.savetxt(graph_out, pcs_graph, delimiter=',', header="PC1,PC2,PC3", comments='')

print("Saved:", png_out)
print("Node scores:", nodes_out, "Graph scores:", graph_out)
print("Explained variance (nodes):", expl_nodes.tolist())
print("Explained variance (graphs):", expl_graph.tolist())