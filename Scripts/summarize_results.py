import os
import pandas as pd
import numpy as np

# paths (adjust if needed)
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Outputs", "GNN_complete")
src = os.path.join(OUT, "per_fold_results.csv")
dst = os.path.join(OUT, "per_fold_results_summary.csv")

df = pd.read_csv(src)

# choose one representative row per fold:
rows = []
for fold, g in df.groupby("fold"):
    # prefer row with max val_acc (non-zero); fallback to max val_f1; final fallback = last row
    nonzero_acc = g[g["val_acc"] > 0]
    if len(nonzero_acc):
        sel = nonzero_acc.loc[nonzero_acc["val_acc"].idxmax()]
    else:
        nonzero_f1 = g[g["val_f1"] > 0]
        if len(nonzero_f1):
            sel = nonzero_f1.loc[nonzero_f1["val_f1"].idxmax()]
        else:
            sel = g.iloc[-1]
    rows.append(sel)

summary = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
summary.to_csv(dst, index=False)

# compute final aggregates
accs = summary["val_acc"].astype(float).to_numpy()
f1s  = summary["val_f1"].astype(float).to_numpy()

acc_mean, acc_std = np.nanmean(accs), np.nanstd(accs, ddof=0) if len(accs)>1 else (accs.mean(), np.nan)
f1_mean, f1_std   = np.nanmean(f1s), np.nanstd(f1s, ddof=0) if len(f1s)>1 else (f1s.mean(), np.nan)

print("Saved cleaned per-fold summary to:", dst)
print("Per-fold summary:")
print(summary[["fold","alpha","val_acc","val_f1"]].to_string(index=False))
print(f"\nOverall CV Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Overall CV F1-score: {f1_mean:.4f} ± {f1_std:.4f}")