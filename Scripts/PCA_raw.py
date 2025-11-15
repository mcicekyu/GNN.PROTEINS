import os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs"))
os.makedirs(OUT, exist_ok=True)

# load node attributes (raw), graph indicator and graph labels
X = np.loadtxt(os.path.join(ROOT, "PROTEINS_node_attributes.txt"), delimiter=',')
print("Loaded X.shape:", X.shape)
gi = np.loadtxt(os.path.join(ROOT, "PROTEINS_graph_indicator.txt"), dtype=int)
gl_raw = np.loadtxt(os.path.join(ROOT, "PROTEINS_graph_labels.txt"), dtype=int)  # raw labels (e.g. 0,1)

# compute per-graph means (G x F)
G = int(gi.max())
means = np.array([X[gi==g].mean(axis=0) for g in range(1, G+1)])

# standardize then PCA
scaler = StandardScaler()
means_s = scaler.fit_transform(means)
pca = PCA(n_components=2)
pcs = pca.fit_transform(means_s)
expl = pca.explained_variance_ratio_

# save scores with labels
df_out = pd.DataFrame({
    "PC1": pcs[:,0], "PC2": pcs[:,1],
    "label": gl_raw.astype(int)
})
df_out.to_csv(os.path.join(OUT, "PCA_graph_scores.csv"), index=False)

# plot colored by raw label
colors = {int(v):c for v,c in zip(np.unique(gl_raw), ["#1f77b4","#ff7f0e","#2ca02c"])}
plt.figure(figsize=(7,6))
for lab in np.unique(gl_raw):
    lab_int = int(lab)
    mask = df_out["label"]==lab_int
    plt.scatter(df_out.loc[mask,"PC1"].to_numpy(), df_out.loc[mask,"PC2"].to_numpy(), s=18, alpha=0.7, label=f"class {lab_int}", color=colors.get(lab_int,"k"))
plt.xlabel(f"PC1 ({expl[0]*100:.2f}%)"); plt.ylabel(f"PC2 ({expl[1]*100:.2f}%)")
plt.legend(); plt.title("PCA (standardized)")
plt.grid(True, linestyle=':', linewidth=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pca_graph_scatter.png"), dpi=200)
plt.close()
print("Saved:", os.path.join(OUT, "pca_graph_scatter.png"))
print("Explained variance PC1,PC2:", expl[:2])