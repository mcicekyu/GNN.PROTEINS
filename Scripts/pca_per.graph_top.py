import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs"))
os.makedirs(OUT, exist_ok=True)

# paths
INPATH = os.path.join(ROOT, "PROTEINS_node_attributes.cleaned.txt")
GI_PATH = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GL_PATH = os.path.join(ROOT, "PROTEINS_graph_labels.txt")

# load node attributes and graph mapping/labels
X = np.loadtxt(INPATH, delimiter=',')           # node x features
gi = np.loadtxt(GI_PATH, dtype=int)            # node -> graph id (1..G)
gl = np.loadtxt(GL_PATH, dtype=int)            # graph labels (len G)

# top-10 feature indices (0-based) requested
top10 = [22, 18, 3, 20, 21, 23, 0, 5, 1, 10]

G = int(gi.max())
# per-graph feature vectors using only top-10 (one row per graph)
graph_feats = np.vstack([X[gi == g][:, top10].mean(axis=0) for g in range(1, G+1)])

# standardize and PCA (2 components)
scaler = StandardScaler()
graph_feats_s = scaler.fit_transform(graph_feats)
pca = PCA(n_components=2)
pcs = pca.fit_transform(graph_feats_s)
expl = pca.explained_variance_ratio_

# prepare colors by graph label and counts for legend
unique_labels, counts = np.unique(gl.astype(int), return_counts=True)
label_map = {lab: int(lab) for lab in unique_labels}
cmap = plt.get_cmap("tab10")
colors = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}

# scatter plot (one point per graph)
plt.figure(figsize=(7,6))
for lab in unique_labels:
    mask = (gl == lab)
    plt.scatter(pcs[mask, 0], pcs[mask, 1], s=40, alpha=0.8, label=f"class {int(lab)} ({mask.sum()})", color=colors[lab])
plt.xlabel(f"PC1 ({expl[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({expl[1]*100:.2f}%)")
plt.title("PCA on per-graph means (top-10 features)")
plt.legend(title="graph label", fontsize="small")
plt.grid(True, linestyle=':', linewidth=0.4)
plt.tight_layout()

plot_out = os.path.join(OUT, "pca_graphs_top10_per_graph.png")
scores_out = os.path.join(OUT, "PCA_graph_top10_scores.csv")
np.savetxt(scores_out, pcs, delimiter=',', header="PC1,PC2", comments='')
plt.savefig(plot_out, dpi=200)
plt.close()

print("Saved per-graph PCA plot:", plot_out)
print("Saved per-graph PCA scores (PC1,PC2):", scores_out)
print("Explained variance PC1,PC2:", expl[0], expl[1])
