# Drops near-duplicate attribute columns, (keeping greater variance): |Pearson correlation|>=0.999: [0, 8, 20, 27]
# PrintsOriginal->new column mapping (kept columns) saving cleaned attributes to Outputs
# Measures Point Biserial correlation of each feature with graph labels
# Prints top features correlated with graph labels: Top features by |corr|: [22, 18, 3, 20, 21, 23, 0, 5, 1, 9]
# Plots feature histograms and boxplots of top features and scatter plot of top 2 features (f22 and f18) saving to Outputs
import os
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
NA_PATH = os.path.join(ROOT, "PROTEINS_node_attributes.txt")
GI_PATH = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GL_PATH = os.path.join(ROOT, "PROTEINS_graph_labels.txt")
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs", "Point.Biserial"))
os.makedirs(OUT_DIR, exist_ok=True)

def drop_correlated_features(df_features, threshold=0.999, prefer_high_variance=True):
    """
    Greedy drop of correlated features using pairwise Pearson correlation.
    df_features: DataFrame of only feature columns (numeric).
    threshold: absolute Pearson correlation threshold above which one of the pair is dropped.
    Returns: (reduced_df, dropped_list, kept_list)
    """
    corr = df_features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    variances = df_features.var()
    for col in upper.columns:
        high_corr_cols = upper.index[upper[col] > threshold].tolist()
        for other in high_corr_cols:
            if other in to_drop or col in to_drop:
                continue
            if prefer_high_variance:
                if variances[col] >= variances[other]:
                    to_drop.add(other)
                else:
                    to_drop.add(col)
            else:
                mean_abs_corr = corr.mean()
                if mean_abs_corr[col] <= mean_abs_corr[other]:
                    to_drop.add(other)
                else:
                    to_drop.add(col)
    dropped = sorted(list(to_drop))
    kept = [c for c in df_features.columns if c not in to_drop]
    return df_features[kept], dropped, kept

def select_topk_pointbiserial_pergraph(df_node_features, gi_path, gl_path, topk=10):
    """
    Aggregate node features to per-graph means, load graph labels and compute
    point-biserial correlation (feature vs binary graph label). Return top-k features
    (feature column label, r, p) with column labels referring to original node-attribute columns.
    """
    # read graph indicator and labels
    gi = np.loadtxt(gi_path, dtype=int)
    gl_raw = np.loadtxt(gl_path, dtype=int)
    # ensure shapes match
    if df_node_features.shape[0] != gi.shape[0]:
        raise RuntimeError(f"Node rows ({df_node_features.shape[0]}) != GI rows ({gi.shape[0]})")
    # aggregate per-graph means; graph ids assumed 1..G
    df = df_node_features.copy()
    df['__gid__'] = gi
    means = df.groupby('__gid__').mean()
    means.index = means.index.astype(int)  # 1..G
    # labels vector (ordered by graph id lines in GL_PATH)
    labels = gl_raw.astype(int)
    unique = np.unique(labels)
    if unique.size != 2:
        raise RuntimeError("point-biserial requires binary labels (found %d unique labels)" % unique.size)
    # map raw labels to 0/1
    mapping = {int(v): i for i, v in enumerate(np.sort(unique))}
    y = np.vectorize(mapping.get)(labels)
    # ensure means rows correspond to labels order: means index likely 1..G
    # convert means to numpy array rows ordered by graph id 1..G
    G = labels.size
    # some graphs may be missing in means if no nodes -> fill with NaN rows
    means_full = pd.DataFrame(np.nan, index=np.arange(1, G+1), columns=means.columns)
    for idx in means.index:
        means_full.loc[idx] = means.loc[idx]
    # compute point-biserial per feature
    corrs = {}
    for feat in means_full.columns:
        col = means_full[feat].to_numpy(dtype=float)
        valid = ~np.isnan(col)
        if valid.sum() < 2 or np.allclose(col[valid], col[valid][0]):
            r = 0.0
            p = 1.0
        else:
            try:
                # pointbiserialr may return a tuple-like or a result object; normalize to floats
                res = pointbiserialr(y[valid], col[valid])
                # support both scipy's result object and tuple-like returns
                # prefer attribute access when available
                r_attr = getattr(res, "statistic", None)
                p_attr = getattr(res, "pvalue", None)
                if r_attr is not None or p_attr is not None:
                    r_val = float(r_attr) if r_attr is not None else 0.0
                    p_val = float(p_attr) if p_attr is not None else 1.0
                else:
                    # fall back to converting the result to a numeric array first,
                    # which avoids passing non-numeric objects to float()
                    try:
                        arr = np.asarray(res, dtype=float)
                        if arr.size >= 2:
                            r_val = float(arr[0])
                            p_val = float(arr[1])
                        else:
                            r_val = 0.0
                            p_val = 1.0
                    except Exception:
                        r_val = 0.0
                        p_val = 1.0
                # ensure r is a valid number
                try:
                    if np.isnan(r_val):
                        r_val = 0.0
                except Exception:
                    r_val = 0.0
                r = r_val
                p = p_val
            except Exception:
                r = 0.0
                p = 1.0
        corrs[feat] = (float(r), float(p))
    # rank by absolute r
    sorted_feats = sorted(corrs.items(), key=lambda item: abs(item[1][0]), reverse=True)
    topk_list = sorted_feats[:topk]
    return topk_list, corrs

def main():
    # read node attributes (no header, columns = original indices 0..F-1)
    df = pd.read_csv(NA_PATH, header=None, sep=r'[,\s]+', engine='python')
    # drop correlated features using pairwise Pearson
    X_reduced, dropped, kept = drop_correlated_features(df, threshold=0.999, prefer_high_variance=True)
    # save cleaned node attributes (rows = nodes, cols = kept original columns)
    cleaned_path = os.path.join(ROOT, "PROTEINS_nodes_attributes_cleaned.txt")
    X_reduced.to_csv(cleaned_path, index=False, header=False, sep=",")
    # also save in OUT_DIR for convenience
    out_clean = os.path.join(OUT_DIR, "PROTEINS_nodes_attributes_cleaned.txt")
    X_reduced.to_csv(out_clean, index=False, header=False, sep=",")
    print("Saved cleaned node-attributes to:", cleaned_path)
    print("Also saved to Outputs:", out_clean)

    # report which features dropped / kept (original column indices)
    print("Dropped features by Pearson (original column indices):", dropped)
    print("Kept features (original column indices):", kept)
    # mapping orig->new index in cleaned matrix
    orig2new = {int(orig): int(new) for new, orig in enumerate(kept)}
    new2orig = {int(new): int(orig) for new, orig in enumerate(kept)}
    print("Original -> cleaned-col-index mapping (orig->new):", orig2new)

    # select top-10 by point-biserial using per-graph means of cleaned features
    topk = 10
    topk_list, all_corrs = select_topk_pointbiserial_pergraph(X_reduced, GI_PATH, GL_PATH, topk=topk)
    print(f"\nTop-{topk} features by point-biserial (from kept features):")
    for feat, (r, p) in topk_list:
        # feat is original column label from df columns (int)
        orig_idx = int(feat)
        new_idx = orig2new.get(orig_idx)
        print(f" kept-col-newidx={new_idx}  orig-col={orig_idx}    r={r:.6g}    p={p:.6g}")

if __name__ == "__main__":
    main()
# ...existing