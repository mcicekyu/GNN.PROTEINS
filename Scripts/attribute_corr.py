# Computes correlation between graph-level attributes and graph labels
import os, re, math
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
NA_PATH = os.path.join(ROOT, "PROTEINS_node_attributes.txt")
GI_PATH = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GL_PATH = os.path.join(ROOT, "PROTEINS_graph_labels.txt")
README = os.path.join(ROOT, "REMARKS.txt")

float_re = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def load_numeric_table(path):
    rows = []
    with open(path, "r") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            nums = float_re.findall(s)
            if not nums:
                parts = re.split(r'[,\s]+', s)
                nums = [p for p in parts if p]
            rows.append([float(x) for x in nums])
    return np.array(rows, dtype=float)

def main():
    if not os.path.exists(NA_PATH):
        print("Node-attributes missing:", NA_PATH); return
    if not os.path.exists(GI_PATH) or not os.path.exists(GL_PATH):
        print("Required graph_indicator or graph_labels missing."); return

    print("Loading node attributes (may take a moment)...")
    X = load_numeric_table(NA_PATH)         # shape: (N_nodes, F)
    gi = np.loadtxt(GI_PATH, dtype=int)     # length N_nodes, values 1..G
    gl = np.loadtxt(GL_PATH, dtype=float)   # length G (one label per graph)

    N, F = X.shape
    G = int(gi.max())
    print(f"Nodes: {N}, Features: {F}, Graphs: {G}")

    # aggregate per-graph means and stds
    means = np.zeros((G, F), dtype=float)
    stds = np.zeros((G, F), dtype=float)
    counts = np.zeros(G, dtype=int)
    for gid in range(1, G+1):
        idx = np.where(gi == gid)[0]
        counts[gid-1] = idx.size
        if idx.size > 0:
            sub = X[idx, :]
            means[gid-1, :] = sub.mean(axis=0)
            stds[gid-1, :] = sub.std(axis=0, ddof=0)
        else:
            means[gid-1, :] = np.nan
            stds[gid-1, :] = np.nan

    # ensure labels align (if labels are 1-based class ids, map to 0..C-1)
    if gl.size == G:
        labels = gl
    else:
        print("Warning: graph_labels length != number of graphs; proceeding with available labels")
        labels = gl[:G]

    # compute Pearson correlation between each feature (per-graph mean) and graph labels
    corrs = []
    lab = labels.astype(float)
    for f in range(F):
        col = means[:, f]
        valid = ~np.isnan(col)
        if valid.sum() < 2 or np.allclose(col[valid], col[valid][0]):
            corr = float('nan')
        else:
            corr = np.corrcoef(col[valid], lab[valid])[0,1]
        corrs.append(corr)
    corrs = np.array(corrs, dtype=float)

    # show top features by absolute correlation
    order = np.argsort(-np.abs(corrs))
    print("\nTop 10 features by |corr| with graph label (feature_index, corr, mean, std):")
    for i in order[:10]:
        print(f"  f{i}: corr={corrs[i]:.4f}, global_mean={np.nanmean(means[:,i]):.4g}, global_std={np.nanstd(means[:,i]):.4g}")

    print("\nCounts per graph: min, mean, max:", counts.min(), counts.mean(), counts.max())
    if os.path.exists(README):
        print("\nREADME (first 20 lines):")
        with open(README) as f:
            for ln in f.readlines()[:20]:
                print(" ", ln.rstrip())

if __name__ == '__main__':
    main()
