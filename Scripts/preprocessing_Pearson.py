# Drops near-duplicate attribute columns, (keeping first occurrence): |r|>=0.999: [1, 8, 20, 27]
# PrintsOriginal->new column mapping (kept columns) saving cleaned attributes to Outputs
# Computes Pearson correlation of each feature with graph labels
# Prints top features correlated with graph labels: Top features by |corr|: [22, 18, 3, 20, 21, 23, 0, 5, 1, 10]
# Plots feature histograms and boxplots of top features and scatter plot of top 2 features (f22 and f18) saving to Outputs


import os, re
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
NA_PATH = os.path.join(ROOT, "PROTEINS_node_attributes.txt")
GI_PATH = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GL_PATH = os.path.join(ROOT, "PROTEINS_graph_labels.txt")
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs"))
os.makedirs(OUT_DIR, exist_ok=True)

float_re = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def load_numeric_table(path):
    rows = []
    with open(path, "r") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            nums = float_re.findall(s)
            if not nums:
                parts = re.split(r'[,\s]+', s)
                nums = [p for p in parts if p]
            rows.append([float(x) for x in nums])
    return np.array(rows, dtype=float)

def per_graph_aggregates(X, gi):
    G = int(gi.max())
    means = np.zeros((G, X.shape[1]), dtype=float)
    counts = np.zeros(G, dtype=int)
    for gid in range(1, G+1):
        idx = np.where(gi == gid)[0]
        counts[gid-1] = idx.size
        if idx.size:
            means[gid-1] = X[idx].mean(axis=0)
        else:
            means[gid-1] = np.nan
    return means, counts

def top_feature_indices_by_corr(means, labels_mapped, topk=10):
    corrs = []
    lab = labels_mapped.astype(float)
    for f in range(means.shape[1]):
        col = means[:, f]
        valid = ~np.isnan(col)
        if valid.sum() < 2 or np.allclose(col[valid], col[valid][0]):
            corrs.append(np.nan)
        else:
            corrs.append(np.corrcoef(col[valid], lab[valid])[0,1])
    corrs = np.array(corrs, dtype=float)
    order = np.argsort(-np.abs(corrs))
    return order[:topk], corrs

def plot_feature_distributions(means, labels_raw, feat_idx, out_dir):
    a = means[:, feat_idx]
    unique_raw = np.unique(labels_raw)
    # use matplotlib default color cycle first two colors so hist and box use same palette
    colors = ['#1f77b4', '#ff7f0e']

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    for i, raw in enumerate(unique_raw):
        mask = (labels_raw == raw)
        col = colors[i % len(colors)]
        plt.hist(a[mask & ~np.isnan(a)], bins=40, alpha=0.6, label=f'class {int(raw)}', color=col)
    plt.legend(); plt.title(f"Feature f{feat_idx} histogram")

    plt.subplot(1,2,2)
    groups = [a[labels_raw==raw][~np.isnan(a[labels_raw==raw])] for raw in unique_raw]
    # some matplotlib versions do not accept a 'labels' kwarg for plt.boxplot;
    # create the boxplot first and then set tick labels with plt.xticks so it's compatible.
    boxes = plt.boxplot(groups, patch_artist=True)
    plt.xticks(np.arange(1, len(unique_raw) + 1), [f"{int(raw)}" for raw in unique_raw])
    # color boxes to match histograms
    for i, patch in enumerate(boxes['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_edgecolor('k')
    # unify other boxplot element colors
    for median in boxes.get('medians', []):
        median.set_color('k')
    for whisker in boxes.get('whiskers', []):
        whisker.set_color('k')
    for cap in boxes.get('caps', []):
        cap.set_color('k')
    for flier in boxes.get('fliers', []):
        try:
            flier.set_markeredgecolor('k')
        except Exception:
            pass

    plt.title(f"Feature f{feat_idx} boxplot")
    path = os.path.join(out_dir, f"feature_f{feat_idx:02d}_dist.png")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def plot_scatter_top2(means, labels_raw, i0, i1, out_dir):
    x = means[:, i0]; y = means[:, i1]
    valid = ~np.isnan(x) & ~np.isnan(y)
    unique_raw = np.unique(labels_raw)
    plt.figure(figsize=(6,6))
    for raw in unique_raw:
        mask = valid & (labels_raw == raw)
        plt.scatter(x[mask], y[mask], s=12, alpha=0.6, label=f'class {int(raw)}')
    plt.xlabel(f"f{i0}"); plt.ylabel(f"f{i1}"); plt.legend()
    path = os.path.join(out_dir, f"scatter_f{i0:02d}_f{i1:02d}.png")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    return path

def main():
    # load original attributes for duplicate detection
    X_orig = load_numeric_table(NA_PATH)
    # Drop near-duplicate columns by Pearson correlation threshold.
    # Set thr to desired sensitivity (0.999 very strict, 0.995 looser).
    thr = 0.999
    F = X_orig.shape[1]
    cleaned_path = os.path.join(ROOT, "PROTEINS_node_attributes.cleaned.txt")
    if F > 1:
        C = np.corrcoef(X_orig, rowvar=False)  # (F,F)
        to_drop = set()
        kept = []
        for i in range(F):
            if i in to_drop:
                continue
            kept.append(i)
            for j in range(i+1, F):
                if j in to_drop:
                    continue
                if np.isnan(C[i, j]):
                    continue
                if abs(C[i, j]) >= thr:
                    to_drop.add(j)
        if to_drop:
            print(f"Dropping near-duplicate columns with |r|>={thr}: {sorted(to_drop)}")
            X_clean = X_orig[:, kept]
            mapping = {orig: new for new, orig in enumerate(kept)}
            print("Original->new column mapping (kept columns):", mapping)

            # write cleaned file with ", " delimiter
            np.savetxt(cleaned_path, X_clean, delimiter=", ", fmt="%.6g")
            print("Saved cleaned attributes to PROTEINS_node_attributes.cleaned.txt")
        else:
            print("No near-duplicate columns found (thr={} ).".format(thr))
            # no cleaned file written; treat cleaned == original
            X_clean = X_orig
    else:
        print("Single-column attributes: skipping duplicate removal.")
        X_clean = X_orig
    # Now use the cleaned file if present (per your request)
    if os.path.exists(cleaned_path):
        X = load_numeric_table(cleaned_path)
        print("Using cleaned attributes from:", cleaned_path)
    else:
        X = X_clean
        print("Using original attributes (no cleaned file):", NA_PATH)

    gi = np.loadtxt(GI_PATH, dtype=int)
    gl_raw = np.loadtxt(GL_PATH, dtype=int)  # raw labels as in file (e.g. 0,1 or 1,2)
    # If raw labels are already 0..C-1 (contiguous, zero-based), use them directly.
    unique = np.unique(gl_raw)
    if unique.min() == 0 and np.array_equal(unique, np.arange(unique.size)):
        gl_mapped = gl_raw
        label_map = {int(v): int(v) for v in unique}
    else:
        # Otherwise map sorted unique raw labels to 0..C-1 for numeric computations
        label_map = {int(val): i for i, val in enumerate(unique)}
        gl_mapped = np.vectorize(label_map.get)(gl_raw)

    print("Label mapping (raw -> mapped):", label_map)

    means, counts = per_graph_aggregates(X, gi)
    topk = 10
    top_idx, corrs = top_feature_indices_by_corr(means, gl_mapped, topk=topk)
    print("Top features by |corr|:", top_idx.tolist())

    saved = []
    for idx in top_idx:
        p = plot_feature_distributions(means, gl_raw, int(idx), OUT_DIR)
        saved.append(p)
    if top_idx.size >= 2:
        s = plot_scatter_top2(means, gl_raw, int(top_idx[0]), int(top_idx[1]), OUT_DIR)
        saved.append(s)
    print("Saved plots:", "\n  ".join(saved))

if __name__ == "__main__":
    main()
