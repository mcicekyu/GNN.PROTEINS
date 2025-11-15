# Drops near-duplicate attribute columns by Spearman correlation, (keeping first occurrence): |r|>=0.999: [1, 20, 27]
# PrintsOriginal->new column mapping (kept columns) saving cleaned attributes to Outputs
# Computes Spearman correlation of each feature with graph labels                    
# Prints top features correlated with graph labels: Top features by |corr|: [23, 19, 0, 1, 21, 4, 22, 24, 6, 12]
# Plots feature histograms and boxplots of top features and scatter plot of top 2 features (f23 and f19) saving to Outputs


import os, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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

def top_feature_indices_by_corr(means, labels_mapped, topk=10, method="pearson"):
    """
    Return topk feature indices by absolute correlation with labels.
    method: "pearson" (default), "spearman", "pointbiserial", "mutual_info".
    Also computes alternative scores and prints a short reminder when default is used.
    Returns (top_idx, scores_dict) where scores_dict contains arrays for:
      "pearson", "spearman", "pointbiserial", "mutual_info", and "primary".
    """
    # local imports to avoid adding global deps if unused
    from scipy.stats import spearmanr
    try:
        from scipy.stats import pointbiserialr
    except Exception:
        pointbiserialr = None
    try:
        from sklearn.feature_selection import mutual_info_classif
    except Exception:
        mutual_info_classif = None

    nfeat = means.shape[1]
    lab = labels_mapped.astype(float)

    # Pearson (default)
    pearson_scores = np.full(nfeat, np.nan, dtype=float)
    for f in range(nfeat):
        col = means[:, f]
        valid = ~np.isnan(col)
        if valid.sum() < 2 or np.allclose(col[valid], col[valid][0]):
            pearson_scores[f] = np.nan
        else:
            pearson_scores[f] = np.corrcoef(col[valid], lab[valid])[0, 1]

    # Spearman
    spearman_scores = np.full(nfeat, np.nan, dtype=float)
    for f in range(nfeat):
        col = means[:, f]
        valid = ~np.isnan(col)
        if valid.sum() < 2 or np.allclose(col[valid], col[valid][0]):
            spearman_scores[f] = np.nan
        else:
            res = spearmanr(col[valid], lab[valid], nan_policy='omit')
            sc = getattr(res, "correlation", None)
            if sc is None:
                if isinstance(res, (tuple, list)) and len(res) >= 1:
                    sc = res[0]
                else:
                    sc = res
            spearman_scores[f] = float(np.asarray(sc).item())

    # point-biserial (if binary labels and available)
    pb_scores = None
    if pointbiserialr is not None and np.unique(lab).size == 2:
        pb_scores = np.full(nfeat, np.nan, dtype=float)
        for f in range(nfeat):
            col = means[:, f]
            valid = ~np.isnan(col)
            if valid.sum() < 2 or np.allclose(col[valid], col[valid][0]):
                pb_scores[f] = np.nan
            else:
                try:
                    res = pointbiserialr(col[valid], lab[valid])
                    # extract correlation robustly: prefer `.correlation`, else take first tuple element or the result itself
                    corr = getattr(res, "correlation", None)
                    if corr is None:
                        if isinstance(res, (tuple, list)) and len(res) >= 1:
                            corr = res[0]
                        else:
                            corr = res
                    # convert to a plain float (handles numpy scalars, etc.)
                    pb_scores[f] = float(np.asarray(corr).item())
                except Exception:
                    pb_scores[f] = np.nan

    # mutual information (if sklearn available)
    mi_scores = None
    if mutual_info_classif is not None:
        valid_rows = ~np.isnan(means).any(axis=1)
        if valid_rows.sum() >= 2:
            try:
                mi = mutual_info_classif(means[valid_rows], lab[valid_rows].astype(int),
                                         discrete_features=False, random_state=0)
                mi_scores = np.asarray(mi, dtype=float)
            except Exception:
                mi_scores = None

    # choose primary scoring method
    method = method.lower() if method is not None else "pearson"
    if method == "pearson":
        primary = pearson_scores
        reminder = "Default: Pearson correlation used. Alternatives: Spearman (rank), point-biserial (binary labels), mutual information (non-linear)."
    elif method == "spearman":
        primary = spearman_scores
        reminder = "Used Spearman rank correlation. Alternatives: Pearson, point-biserial (binary labels), mutual information."
    elif method == "pointbiserial":
        if pb_scores is None:
            raise RuntimeError("Point-biserial not available or labels not binary.")
        primary = pb_scores
        reminder = "Used point-biserial correlation. Alternatives: Pearson, Spearman, mutual information."
    elif method == "mutual_info":
        if mi_scores is None:
            raise RuntimeError("mutual_info_classif not available or insufficient valid rows.")
        primary = mi_scores
        reminder = "Used mutual information. Alternatives: Pearson, Spearman, point-biserial (binary)."
    else:
        raise ValueError("Unknown method: " + str(method))

    # print reminder when default (or any) is used
    print(reminder)

    # rank by absolute primary score (NaNs go last)
    order = np.argsort(-np.abs(np.nan_to_num(primary, nan=-np.inf)))
    top_idx = order[:topk]

    scores = {
        "pearson": pearson_scores,
        "spearman": spearman_scores,
        "pointbiserial": pb_scores,
        "mutual_info": mi_scores,
        "primary": primary
    }
    return top_idx, scores

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
        # compute Spearman rank correlation between feature columns (handles ties/nans)
        corr_res = spearmanr(X_orig, axis=0, nan_policy='omit')
        # scipy.stats.spearmanr may return different shapes/types depending on scipy version and input:
        corr_attr = getattr(corr_res, "correlation", None)
        if corr_attr is not None:
            C = np.asarray(corr_attr)
        elif isinstance(corr_res, (tuple, list)) and len(corr_res) >= 1:
            C = np.asarray(corr_res[0])
        else:
            C = np.asarray(corr_res)
        # If spearmanr returned a scalar (degenerate case), make it a 2D array
        if C.ndim == 0:
            C = np.array([[C]])
        # ensure a 2D numpy array for subsequent indexing
        C = np.atleast_2d(C)
        
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
            print(f"Dropping near-duplicate columns with |Spearman corr|>={thr}: {sorted(to_drop)}")
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
    # choose scoring method: default Pearson; alternatives: "spearman", "pointbiserial", "mutual_info"
    method = "pearson"
    top_idx, scores = top_feature_indices_by_corr(means, gl_mapped, topk=topk, method=method)
    print(f"Top features by |{method} corr|:", top_idx.tolist())
    # also print the primary scores for the selected top features (rounded)
    print("Top feature primary scores:", np.round(scores["primary"][top_idx], 6).tolist())

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
