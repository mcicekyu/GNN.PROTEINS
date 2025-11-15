# After preprocessing, analyze node attributes with respect to graph labels.
# Analyze graph-wise means of a specific node attribute (f5 here)  and their relation to graph labels

import os
import numpy as np
# try to import scipy.stats, but provide a minimal fallback so the script runs even if scipy is missing
try:
    import importlib
    st = importlib.import_module('scipy.stats')
except Exception:
    # minimal Welch's t-test fallback with best-effort p-value (uses normal approx if scipy missing)
    class _SimpleStats:
        @staticmethod
        def ttest_ind(a, b, equal_var=False):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            a = a[~np.isnan(a)]
            b = b[~np.isnan(b)]
            na = a.size
            nb = b.size
            if na < 1 or nb < 1:
                class _EmptyRes:
                    def __init__(self):
                        self.statistic = np.nan
                        self.pvalue = np.nan
                return _EmptyRes()
            ma = a.mean()
            mb = b.mean()
            sa = a.var(ddof=1) if na > 1 else 0.0
            sb = b.var(ddof=1) if nb > 1 else 0.0
            if equal_var and na + nb - 2 > 0:
                s_pooled = ((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2)
                se = np.sqrt(s_pooled * (1.0 / na + 1.0 / nb))
                df = na + nb - 2
            else:
                se = np.sqrt(sa / na + sb / nb)
                # Welch-Satterthwaite approximation
                num = (sa / na + sb / nb) ** 2
                den = 0.0
                if na > 1:
                    den += (sa * sa) / (na * na * (na - 1))
                if nb > 1:
                    den += (sb * sb) / (nb * nb * (nb - 1))
                df = num / den if den > 0 else np.inf
            if se == 0:
                t = np.inf if (ma - mb) != 0 else 0.0
            else:
                t = (ma - mb) / se
            # attempt to compute p-value using scipy if present, otherwise fall back to normal approx
            try:
                import importlib as _importlib
                _sc = _importlib.import_module('scipy.stats')
                p = _sc.t.sf(abs(t), df) * 2
            except Exception:
                # normal approximation (reasonable for moderate/large sample sizes)
                from math import erf, sqrt
                z = t
                p = (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))) * 2.0
            class _Res:
                def __init__(self, statistic, pvalue):
                    self.statistic = statistic
                    self.pvalue = pvalue
            return _Res(t, p)
    st = _SimpleStats()

ROOT = "/Users/mihribancicekyurt/Desktop/GNN.PROTEINS/DATA/PROTEINS"
# NOTE: the cleaned file saved earlier was named "PROTEINS_node_attributes.cleaned.txt"
ATTR_PATH = os.path.join(ROOT, "PROTEINS_node_attributes.cleaned.txt")
if not os.path.exists(ATTR_PATH):
    ATTR_PATH = os.path.join(ROOT, "PROTEINS_node_attributes_cleaned.txt")
    if not os.path.exists(ATTR_PATH):
        raise FileNotFoundError("Cleaned node-attributes file not found; create or adjust ATTR_PATH")
X = np.loadtxt(ATTR_PATH, delimiter=',')
gi = np.loadtxt(os.path.join(ROOT,"PROTEINS_graph_indicator.txt"), dtype=int)
gl = np.loadtxt(os.path.join(ROOT,"PROTEINS_graph_labels.txt"), dtype=int)
G = int(gi.max())
# compute per-graph means robustly (use nanmean in case of empty groups)
means = np.array([np.nanmean(X[gi==g, 5]) for g in range(1, G+1)])  # f5 index = 5, change as needed
# per-class stats
unique = np.unique(gl)
mapping = {int(v): i for i, v in enumerate(unique)}
labels = np.vectorize(mapping.get)(gl)
print("Label mapping (raw -> mapped):", mapping)
for c in np.unique(labels):
     a = means[labels==c]
     print(f"class {unique[c]}: n={a.size}, mean={a.mean():.4g}, median={np.median(a):.4g}, std={a.std():.4g}")
if np.sum(labels==0) > 1 and np.sum(labels==1) > 1:
    print("t-test p:", st.ttest_ind(means[labels==0], means[labels==1], equal_var=False).pvalue)
else:
    print("Not enough samples for t-test.")
# list extreme graphs
outs = np.where((means < -2) | (means > 2))[0] + 1
print("Extreme-mean graph ids:", outs.tolist())