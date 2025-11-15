# PCA per-node (selected vs all features) and color by graph labels (from PROTEINS_graph_labels.txt)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs"))
os.makedirs(OUT, exist_ok=True)

INPATH = os.path.join(ROOT, "PROTEINS_node_attributes.cleaned.txt")
X = pd.read_csv(INPATH, sep=r'\s*,\s*', engine='python', header=None).values.astype(float)

# load graph indicator and graph labels to color nodes by their graph's class
GI_PATH = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GL_PATH = os.path.join(ROOT, "PROTEINS_graph_labels.txt")
gi = np.loadtxt(GI_PATH, dtype=int)           # length == number of nodes; values 1..G
gl = np.loadtxt(GL_PATH, dtype=int)           # length == number of graphs; values are raw labels

# create per-node labels by mapping node -> graph id -> graph label
# assumes gl is indexed 1..G (so gl[g-1] is label for graph g)
node_labels = np.array([int(gl[g-1]) for g in gi], dtype=int)

top10 = [22, 18, 3, 20, 21, 23, 0, 5, 1, 10]  # 0-based indices

def run_pca(X_sub, name):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_sub)
    pca = PCA(n_components=min(10, Xs.shape[1]))
    pcs = pca.fit_transform(Xs)
    expl = pca.explained_variance_ratio_
    print(f"{name}: features={X_sub.shape[1]}  PC1={expl[0]:.4f}  PC2={expl[1]:.4f}  cum(2)={(expl[:2].sum()):.4f}")
    return pcs, expl, pca

pcs_top, expl_top, pca_top = run_pca(X[:, top10], "Top-10")
pcs_all, expl_all, pca_all = run_pca(X, "All-cleaned")

# prepare colors and legend labels based on PROTEINS_graph_labels.txt
unique_labels, counts = np.unique(node_labels, return_counts=True)
unique_labels = unique_labels.astype(int)
label_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
print("Per-node label counts:", label_counts)

# choose a color map with enough distinct colors
cmap = plt.get_cmap("tab10")
color_map = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}

# side-by-side PC1/PC2 scatter colored by graph label
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

ax = axes[0]
for lab in unique_labels:
    mask = (node_labels == lab)
    ax.scatter(pcs_top[mask, 0], pcs_top[mask, 1], s=12, alpha=0.6, label=f"class {int(lab)} ({mask.sum()})", color=color_map[lab])
ax.set_title(f"Top-10 (PC1 {expl_top[0]*100:.2f}%, PC2 {expl_top[1]*100:.2f}%)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True)
ax.legend(title="graph label", fontsize="small", markerscale=1.5)

ax = axes[1]
for lab in unique_labels:
    mask = (node_labels == lab)
    ax.scatter(pcs_all[mask, 0], pcs_all[mask, 1], s=12, alpha=0.6, label=f"class {int(lab)} ({mask.sum()})", color=color_map[lab])
ax.set_title(f"All cleaned (PC1 {expl_all[0]*100:.2f}%, PC2 {expl_all[1]*100:.2f}%)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(True)
ax.legend(title="graph label", fontsize="small", markerscale=1.5)

plot_path = os.path.join(OUT, "compare_pca_top10_vs_all_colored.png")
plt.savefig(plot_path, dpi=200)
plt.close()
print("Saved plot:", plot_path)

# save numeric results
np.savetxt(os.path.join(OUT, "pcs_top10.csv"), pcs_top, delimiter=",")
np.savetxt(os.path.join(OUT, "pcs_all.csv"), pcs_all, delimiter=",")
pd.DataFrame({"expl_top": expl_top}).to_csv(os.path.join(OUT, "expl_top10.csv"), index=False)
pd.DataFrame({"expl_all": expl_all}).to_csv(os.path.join(OUT, "expl_all.csv"), index=False)

# additional check: fraction of total variance captured by top-2 and top-5
print("Top-10 cum var: top2, top5, top10 =", expl_top[:2].sum(), expl_top[:5].sum(), expl_top[:10].sum())
print("All-cleaned cum var: top2, top5, top10 =", expl_all[:2].sum(), expl_all[:5].sum(), expl_all[:10].sum())
