import os, re
import numpy as np
from scipy.stats import spearmanr
from collections import OrderedDict

# optional sklearn imports
try:
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.metrics import mutual_info_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import make_scorer, f1_score
    SKLEARN_AVAILABLE = True
except Exception:
    # Ensure names exist even if sklearn is not available to avoid "possibly unbound" warnings.
    KBinsDiscretizer = None
    mutual_info_score = None
    LogisticRegression = None
    StratifiedKFold = None
    cross_val_score = None
    make_scorer = None
    f1_score = None
    SKLEARN_AVAILABLE = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
NA_PATH = os.path.join(ROOT, "PROTEINS_node_attributes.txt")
GI_PATH = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
GL_PATH = os.path.join(ROOT, "PROTEINS_graph_labels.txt")
CLEANED_PATH = os.path.join(ROOT, "PROTEINS_node_attributes.cleaned.txt")

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
    for gid in range(1, G+1):
        idx = np.where(gi == gid)[0]
        if idx.size:
            means[gid-1] = X[idx].mean(axis=0)
        else:
            means[gid-1] = np.nan
    return means

def pairwise_pearson_matrix(X):
    F = X.shape[1]
    C = np.full((F, F), np.nan, dtype=float)
    for i in range(F):
        xi = X[:, i]
        for j in range(i, F):
            xj = X[:, j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if mask.sum() < 2:
                continue
            try:
                r = np.corrcoef(xi[mask], xj[mask])[0,1]
            except Exception:
                r = np.nan
            C[i, j] = r
            C[j, i] = r
    return C

def pairwise_spearman_matrix(X):
    # scipy.stats.spearmanr handles pairwise NaNs when given full matrix with nan_policy='omit'
    res = spearmanr(X, axis=0, nan_policy='omit')
    corr = getattr(res, "correlation", None)
    if corr is not None:
        C = np.asarray(corr)
    elif isinstance(res, (tuple, list)) and len(res) >= 1:
        C = np.asarray(res[0])
    else:
        C = np.asarray(res)
    return np.atleast_2d(C)

def pairwise_mi_matrix_binned(X, n_bins=16, min_samples=20):
    # discretize by quantiles then use mutual_info_score pairwise
    if not SKLEARN_AVAILABLE:
        return None
    # defensive: ensure the sklearn symbols we expect were actually imported
    if KBinsDiscretizer is None or mutual_info_score is None:
        return None
    F = X.shape[1]
    C = np.full((F, F), np.nan, dtype=float)
    # fit discretizer on rows without any NaN
    nonnan_rows = ~np.isnan(X).any(axis=1)
    if nonnan_rows.sum() < 2:
        return C
    try:
        kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        Xb = np.full_like(X, np.nan)
        Xb[nonnan_rows] = kb.fit_transform(X[nonnan_rows])
    except Exception:
        # if discretization fails for any reason, return the empty matrix
        return C
    for i in range(F):
        xi = Xb[:, i]
        for j in range(i, F):
            xj = Xb[:, j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if mask.sum() < min_samples:
                continue
            try:
                mi = mutual_info_score(xi[mask].astype(int), xj[mask].astype(int))
            except Exception:
                mi = np.nan
            C[i, j] = mi
            C[j, i] = mi
    return C

def drop_near_duplicates(C, thr=0.999):
    if C is None:
        return None, []
    F = C.shape[0]
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
    return np.sort(np.array(kept, dtype=int)), sorted(to_drop)

def top_feature_indices_by_corr(means, labels_mapped, topk=10, method="pearson"):
    # compute candidate scores (simple implementations)
    nfeat = means.shape[1]
    lab = labels_mapped.astype(float)
    pearson = np.full(nfeat, np.nan)
    for f in range(nfeat):
        col = means[:, f]; valid = ~np.isnan(col)
        if valid.sum() < 2 or np.allclose(col[valid], col[valid][0]):
            continue
        pearson[f] = np.corrcoef(col[valid], lab[valid])[0,1]
    spearman = np.full(nfeat, np.nan)
    for f in range(nfeat):
        col = means[:, f]; valid = ~np.isnan(col)
        if valid.sum() < 2 or np.allclose(col[valid], col[valid][0]):
            continue
        res = spearmanr(col[valid], lab[valid], nan_policy='omit')
        sc = getattr(res, "correlation", None)
        if sc is None:
            sc = res[0] if isinstance(res, (tuple,list)) else res
        spearman[f] = float(np.asarray(sc).item())
    pb = None
    mi = None
    scores = {"pearson": pearson, "spearman": spearman, "pointbiserial": pb, "mutual_info": mi}
    primary = scores.get(method)
    if primary is None:
        raise RuntimeError(f"Requested method '{method}' not available")
    order = np.argsort(-np.abs(np.nan_to_num(primary, nan=-np.inf)))
    return order[:topk], scores

def eval_classifier_on_means(means, labels, top_idxs, topk=10):
    if not SKLEARN_AVAILABLE:
        return None
    # perform local imports to ensure the sklearn objects are available (and to satisfy static checkers)
    try:
        from sklearn.linear_model import LogisticRegression as _LogisticRegression
        from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
        from sklearn.model_selection import cross_val_score as _cross_val_score
        from sklearn.metrics import make_scorer as _make_scorer, f1_score as _f1_score
    except Exception:
        # sklearn not usable at runtime even if SKLEARN_AVAILABLE was set
        return None

    Xg = means[:, top_idxs]
    mask = ~np.isnan(Xg).any(axis=1)
    Xg = Xg[mask]; y = labels[mask]
    if Xg.shape[0] < 5:
        return None
    clf = _LogisticRegression(max_iter=2000, solver='liblinear')
    skf = _StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc = _cross_val_score(clf, Xg, y, cv=skf, scoring='accuracy', n_jobs=1)
    f1 = _cross_val_score(clf, Xg, y, cv=skf, scoring=_make_scorer(_f1_score, average='macro'), n_jobs=1)
    return acc.mean(), acc.std(), f1.mean(), f1.std()

def main():
    import json
    X_orig = load_numeric_table(NA_PATH)
    gi = np.loadtxt(GI_PATH, dtype=int)
    gl_raw = np.loadtxt(GL_PATH, dtype=int)
    unique = np.unique(gl_raw)
    if unique.min() == 0 and np.array_equal(unique, np.arange(unique.size)):
        gl_mapped = gl_raw
    else:
        label_map = {int(val): i for i, val in enumerate(unique)}
        gl_mapped = np.vectorize(label_map.get)(gl_raw)

    methods = OrderedDict()
    methods['pearson'] = pairwise_pearson_matrix
    methods['spearman'] = pairwise_spearman_matrix
    if SKLEARN_AVAILABLE:
        methods['mutual_info'] = pairwise_mi_matrix_binned

    thr = 0.999
    topk = 10
    results = {}

    for name, func in methods.items():
        try:
            C = func(X_orig)
        except Exception:
            C = None
        if C is None:
            print(f"{name}: matrix not available / failed.")
            results[name] = {"kept": None, "dropped": None, "perf": None, "kept_idxs": None}
            continue
        kept, dropped = drop_near_duplicates(C, thr=thr)
        kept_arr = np.asarray(kept, dtype=int) if kept is not None else np.array([], dtype=int)
        X_clean = X_orig[:, kept_arr]
        means = per_graph_aggregates(X_clean, gi)
        try:
            top_idxs, _ = top_feature_indices_by_corr(means, gl_mapped, topk=topk, method="pearson")
        except Exception:
            top_idxs = np.arange(min(topk, means.shape[1]))
        perf = eval_classifier_on_means(means, gl_mapped, top_idxs, topk=topk)
        results[name] = {
            "kept": int(kept_arr.size),
            "dropped": len(dropped),
            "dropped_list": dropped,
            "perf": perf,
            "kept_idxs": kept_arr
        }
        print(f"{name}: kept {kept_arr.size}, dropped {len(dropped)} -> top{topk} idxs {top_idxs.tolist() if top_idxs is not None else None}, perf {perf}")

    # summary and choose best cleaning method
    best_method = None
    best_score = -np.inf
    for name, info in results.items():
        perf = info.get("perf")
        if perf is None:
            continue
        acc_mean = perf[0]
        if acc_mean is not None and acc_mean > best_score:
            best_score = acc_mean
            best_method = name

    if best_method is None:
        best_method = 'spearman' if 'spearman' in results else next(iter(results.keys()))
    print("\nMethod performances summary:")
    for name, info in results.items():
        perf = info.get("perf")
        if perf is None:
            pstr = "no perf"
        else:
            pstr = f"acc {perf[0]:.4f}±{perf[1]:.4f}, f1 {perf[2]:.4f}±{perf[3]:.4f}"
        print(f" - {name}: kept={info.get('kept')}, dropped={info.get('dropped')}, {pstr}")

    print(f"\nSelected cleaning method: {best_method} (best CV acc mean={best_score:.4f})")

    # Save cleaned attributes using chosen method
    chosen_func = methods.get(best_method)
    # guard against missing/unsupported method (e.g., mutual_info when sklearn isn't available)
    if chosen_func is None:
        Cchosen = None
    else:
        try:
            Cchosen = chosen_func(X_orig)
        except Exception:
            # if the chosen function fails for any reason, treat as unavailable
            Cchosen = None
    kept, dropped = drop_near_duplicates(Cchosen, thr=thr)
    kept_arr = np.asarray(kept, dtype=int) if kept is not None else np.arange(X_orig.shape[1], dtype=int)
    X_chosen_clean = X_orig[:, kept_arr]
    np.savetxt(CLEANED_PATH, X_chosen_clean, delimiter=", ", fmt="%.6g")
    print("Saved cleaned attributes to:", CLEANED_PATH)
    print("Dropped columns (indices, w.r.t. original attributes):", dropped)

    # write mapping files
    orig2new = {int(orig): int(new) for new, orig in enumerate(kept_arr)}
    new2orig = {int(new): int(orig) for new, orig in enumerate(kept_arr)}
    mapping = {"orig2new": orig2new, "new2orig": new2orig, "dropped": dropped, "method": best_method}
    try:
        outmap = os.path.join(os.path.dirname(CLEANED_PATH), "cleaning_mapping.json")
        with open(outmap, "w") as jf:
            json.dump(mapping, jf, indent=2)
        print("Wrote cleaning mapping to", outmap)
    except Exception:
        pass

    # Now use cleaned attributes to compute top-k by Pearson and Spearman and map back to original indices
    means_used = per_graph_aggregates(X_chosen_clean, gi)
    for method in ('pearson', 'spearman'):
        try:
            top_idxs_clean, scores = top_feature_indices_by_corr(means_used, gl_mapped, topk=topk, method=method)
            top_idxs_orig = [int(new2orig[int(i)]) for i in top_idxs_clean]
            print(f"\nTop-{topk} by {method} (indices in cleaned matrix): {top_idxs_clean.tolist()}")
            print(f"Top-{topk} by {method} (mapped to original attribute indices): {top_idxs_orig}")
            primary_scores = scores.get(method) if scores is not None else None
            if primary_scores is not None:
                print("Primary scores (rounded):", np.round(primary_scores[top_idxs_clean], 6).tolist())
        except Exception as e:
            print(f"Top selection by {method} failed: {e}")

if __name__ == "__main__":
    main()