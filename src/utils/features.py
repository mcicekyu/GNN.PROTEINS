import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import os

def eval_model(model, loader, device, num_classes, return_best_threshold=False, pos_class=1):
    model.eval()
    ys, ys_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch)
            pred = out.argmax(dim=1).cpu().numpy()
            ys_pred.append(pred)
            ys.append(batch.y.view(-1).cpu().numpy())
    ys = np.concatenate(ys); ys_pred = np.concatenate(ys_pred)
    acc = accuracy_score(ys, ys_pred)
    f1 = f1_score(ys, ys_pred, average="macro")
    cm = confusion_matrix(ys, ys_pred, labels=list(range(num_classes)))
    p, r, f, _ = precision_recall_fscore_support(ys, ys_pred, labels=list(range(num_classes)), zero_division=0)
    p = np.asarray(p, dtype=float); r = np.asarray(r, dtype=float); f = np.asarray(f, dtype=float)
    if p.size < num_classes:
        pad = num_classes - p.size
        p = np.pad(p, (0, pad), constant_values=0.0)
        r = np.pad(r, (0, pad), constant_values=0.0)
        f = np.pad(f, (0, pad), constant_values=0.0)

    best_th = None; best_th_f1 = None
    if return_best_threshold:
        probs = []; ys_prob = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch)
                ppos = torch.softmax(logits, dim=1)[:, pos_class].cpu().numpy()
                probs.append(ppos)
                ys_prob.append(batch.y.view(-1).cpu().numpy())
        probs = np.concatenate(probs); ys_prob = np.concatenate(ys_prob)
        best_th, best_th_f1 = 0.5, 0.0
        for th in np.linspace(0.1, 0.9, 41):
            preds_th = (probs >= th).astype(int)
            th_f1 = f1_score(ys_prob, preds_th, average='macro')
            if th_f1 > best_th_f1:
                best_th_f1 = th_f1; best_th = float(th)

    return acc, f1, cm, (p, r, f), best_th, best_th_f1

def focal_loss(logits, targets, gamma=2.0, weight=None):
    ce = F.cross_entropy(logits, targets, reduction='none', weight=weight)
    p_t = torch.exp(-ce)
    return ((1.0 - p_t) ** gamma * ce).mean()

def summarize_per_fold_results(out_dir, src_name="per_fold_results.csv", dst_name="per_fold_results_summary.csv"):
    src = os.path.join(out_dir, src_name)
    dst = os.path.join(out_dir, dst_name)
    if not os.path.exists(src):
        print("No per-fold results file found at", src); return
    try:
        df = pd.read_csv(src)
    except Exception as e:
        print("Failed to read per-fold results:", e); return
    rows = []
    for fold, g in df.groupby("fold"):
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
    try:
        summary.to_csv(dst, index=False)
        print("Saved cleaned per-fold summary to:", dst)
    except Exception as e:
        print("Failed to write summary CSV:", e)
        return
    accs = summary["val_acc"].astype(float).to_numpy()
    f1s  = summary["val_f1"].astype(float).to_numpy()
    if accs.size == 0:
        print("No valid per-fold accuracy values found in summary."); return
    import numpy as _np
    acc_mean = float(_np.nanmean(accs)); acc_std = float(_np.nanstd(accs, ddof=0)) if accs.size>1 else float("nan")
    f1_mean = float(_np.nanmean(f1s));   f1_std = float(_np.nanstd(f1s, ddof=0)) if f1s.size>1 else float("nan")
    try:
        print(summary[["fold","alpha","val_acc","val_f1"]].to_string(index=False))
    except Exception:
        print(summary.to_dict(orient="records"))
    print(f"\nOverall CV Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Overall CV F1-score: {f1_mean:.4f} ± {f1_std:.4f}")