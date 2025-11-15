# after preprocessing, do PCA on per-graph feature vectors (mean over nodes in graph)
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs"))
os.makedirs(OUT, exist_ok=True)

X = np.loadtxt(os.path.join(ROOT, "PROTEINS_node_attributes.cleaned.txt"), delimiter=',')
gi = np.loadtxt(os.path.join(ROOT, "PROTEINS_graph_indicator.txt"), dtype=int)  # node -> graph id (1..G)
gl = np.loadtxt(os.path.join(ROOT, "PROTEINS_graph_labels.txt"), dtype=int)     # graph labels (len G)

G = int(gi.max())
# one vector per graph: mean over nodes in that graph
graph_feats = np.vstack([X[gi == g].mean(axis=0) for g in range(1, G+1)])

# standardize + PCA
graph_feats_s = StandardScaler().fit_transform(graph_feats)
pca = PCA(n_components=2).fit(graph_feats_s)
pcs = pca.transform(graph_feats_s)

# plot, color by graph label
plt.figure(figsize=(7,6))
for lab in np.unique(gl):
    mask = (gl == lab)
    plt.scatter(pcs[mask,0], pcs[mask,1], s=30, alpha=0.8, label=f"class {int(lab)} ({mask.sum()})")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
plt.legend()
plt.title("PCA on per-graph feature vectors")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pca_graphs_per_graph.png"), dpi=200)
plt.close()
print("Saved:", os.path.join(OUT, "pca_graphs_per_graph.png"))