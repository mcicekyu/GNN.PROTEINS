# Graph Isomorphism Network (GIN) training on PROTEINS dataset using 5-fold cross-validation with optional
# outlier graph removal using Isolation Forest and Fold‑specific class-weighted loss.
# augmented per-graph features: [mean_node_features..., graph_size, avg_degree]
# OUTLIER_CONTAMINATION = 0.02 is decided.
# Detected 23 (2.07%) outlier graphs (removed) [   6   21   66  191  276  680  683  721  730
#  857  880  914  936  939 965  976  994  997 1015 1047 1054 1058 1059]
# Overall CV Accuracy:  0.7257 ± 0.0198
# Overall F1-score: 0.7087 ± 0.0221


import os, sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold

# Ensure the repository root is in sys.path so relative imports work when run as script
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# direct imports from package modules (no long fallback)
from src.data.dataset import prepare_data
from src.models.gnn import GINNet
from src.utils.features import eval_model, focal_loss, summarize_per_fold_results

# workspace paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs/GNN_complete"))
os.makedirs(OUT, exist_ok=True)

# controls
OUTLIER_CONTAMINATION = 0.02
SKIP_OUTLIER_REMOVAL = False
USE_CLASS_WEIGHTS = False
VERBOSE = False

# prepare data (returns data_list, graph_idx_map, n_feat, graph_feat_dim, unique_labels, gl, gi, label_map)
prepared = prepare_data(ROOT, OUT,
                        outlier_contamination=OUTLIER_CONTAMINATION,
                        skip_outlier=SKIP_OUTLIER_REMOVAL)
data_list = prepared["data_list"]
graph_idx_map = prepared["graph_idx_map"]
n_feat = prepared["n_feat"]
graph_feat_dim = prepared["graph_feat_dim"]
unique_labels = prepared["unique_labels"]
gl = prepared["gl"]
gi = prepared["gi"]
label_map = prepared["label_map"]

num_classes = len(unique_labels)

# save graph-level scaler if prepare_data produced one (useful for inference)
if "graph_scaler" in prepared and prepared["graph_scaler"] is not None:
    try:
        with open(os.path.join(OUT, "graph_scaler.pkl"), "wb") as fh:
            pickle.dump(prepared["graph_scaler"], fh)
    except Exception:
        pass

# quick sanity checks for the new graph-level feature
# ensure prepare_data actually produced a graph_feat_dim and each Data has matching graph_feat
if graph_feat_dim is None:
    raise RuntimeError("prepare_data did not provide 'graph_feat_dim' — update dataset.prepare_data to return it.")
# check Data objects
_mismatch = [i for i, d in enumerate(data_list) if getattr(d, "graph_feat", None) is None or int(getattr(d.graph_feat, "numel", lambda: -1)()) != graph_feat_dim]
if _mismatch:
    print(f"Warning: {len(_mismatch)} Data objects missing graph_feat or with unexpected size. Indices (in kept list): {_mismatch[:10]}{'...' if len(_mismatch)>10 else ''}")


# device selection (prefer MPS on macOS, then CUDA, else CPU)
use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
if use_mps:
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# CV settings and bookkeeping
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
labels_kept = np.array([label_map[int(gl[i])] for i in graph_idx_map])
fold_acc, fold_f1 = [], []
alpha_scores = defaultdict(list)
best_alphas = []

# safe defaults
epochs = 30

def run_cv():
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels_kept)), labels_kept), start=1):
        train_dataset = [data_list[i] for i in tr_idx]
        test_dataset = [data_list[i] for i in te_idx]
        fold_alpha = 1.0

        # DataLoader workers / pin_memory
        cpu_count = os.cpu_count() or 1
        if device.type == "cuda":
            num_workers = min(8, max(1, cpu_count - 1))
            pin_memory = True
        elif device.type == "mps":
            num_workers = 0
            pin_memory = False
        else:
            num_workers = min(4, max(0, cpu_count // 2))
            pin_memory = False

        train_bs = 16
        test_bs = 64

        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False,
                                 num_workers=max(0, num_workers // 2), pin_memory=pin_memory)

        # metrics defaults
        val_acc = 0.0
        val_f1 = 0.0
        cm_val = np.zeros((num_classes, num_classes), dtype=int)
        p_val = np.zeros(num_classes, dtype=float)
        r_val = np.zeros(num_classes, dtype=float)
        f_val = np.zeros(num_classes, dtype=float)

        # model, optimizer, scheduler for fold
        epochs = 30
        model = GINNet(in_dim=n_feat, hidden=32, num_layers=2, num_classes=num_classes,
                       graph_feat_dim=graph_feat_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

        # class-weight tuning & sampler
        if USE_CLASS_WEIGHTS:
            train_labels = np.array([int(d.y.item()) for d in train_dataset])
            class_counts = np.bincount(train_labels, minlength=num_classes)
            weights_per_class = 1.0 / (class_counts + 1e-9)
            base_weights = torch.tensor(weights_per_class, dtype=torch.float, device=device)
            # build per-sample weights for the sampler (keep on CPU / list)
            sample_weights = weights_per_class[train_labels].astype(float)
            sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
            # Use the same train_bs / num_workers / pin_memory as the fold defaults
            train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=sampler,
                                      num_workers=num_workers, pin_memory=pin_memory)

            # tuning loader should also preserve same workers/pin_memory and batch size
            tune_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True,
                                     num_workers=num_workers, pin_memory=pin_memory)

            alpha_grid = [0.25, 0.5, 1.0, 2.0, 4.0]
            # cheaper, safer alpha tuning: reuse a single tmp_model state and reload initial weights
            fold_alpha, best_score = 1.0, -1.0
            tmp_model = GINNet(in_dim=n_feat, hidden=32, num_layers=2, num_classes=num_classes,
                               graph_feat_dim=graph_feat_dim).to(device)
            initial_state = {k: v.cpu().clone() for k, v in tmp_model.state_dict().items()}
            # ensure tmp_opt is always bound to silence potential "possibly unbound" warnings
            tmp_opt = None
            tune_epochs = 1  # keep small to reduce memory/time
            for alpha in alpha_grid:
                # restore initial weights
                for k, v in initial_state.items():
                    tmp_model.state_dict()[k].copy_(v.to(device))
                tmp_opt = torch.optim.Adam(tmp_model.parameters(), lr=5e-4, weight_decay=5e-4)
                for _ in range(tune_epochs):
                    tmp_model.train()
                    for batch in tune_loader:
                        batch = batch.to(device)
                        # defensive check: ensure optimizer is initialized to avoid "possibly unbound" warnings
                        if tmp_opt is None:
                            tmp_opt = torch.optim.Adam(tmp_model.parameters(), lr=5e-4, weight_decay=5e-4)
                        tmp_opt.zero_grad()
                        out = tmp_model(batch.x, batch.edge_index, batch)
                        w = (base_weights.pow(float(alpha))).to(device)
                        loss = focal_loss(out, batch.y.view(-1), gamma=2.0, weight=w)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(tmp_model.parameters(), 2.0)
                        tmp_opt.step()
                    acc, f1, *_ = eval_model(tmp_model, test_loader, device, num_classes)
                    alpha_scores[alpha].append(float(f1))
                    try:
                        # keep tmp_opt bound (set to None) instead of deleting the name to avoid "possibly unbound" warnings
                        tmp_opt = None
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if f1 > best_score:
                        best_score, fold_alpha = f1, alpha
            # delete tmp_model once after finishing the alpha tuning loop to avoid unbinding it mid-loop
            try:
                del tmp_model
            except Exception:
                pass

            fold_class_weights = (base_weights.pow(float(fold_alpha))).to(device)
            print(f"Fold {fold} selected alpha={fold_alpha} (tuned f1={best_score:.4f})")
        else:
            fold_class_weights = None
            print(f"Fold {fold} using no class weights (USE_CLASS_WEIGHTS=False).")

        best_alphas.append(fold_alpha)

        # training loop for fold
        best_val = -1.0
        best_epoch = 0
        early_patience = 6
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            desired_effective_batch = 128
            # compute acc_steps from actual loader batch size (handles sampler overrides)
            actual_bs = getattr(train_loader, "batch_size", train_bs) or train_bs
            acc_steps = max(1, desired_effective_batch // int(actual_bs))
            optimizer.zero_grad()
            last_i = -1
            for i, batch in enumerate(train_loader):
                last_i = i
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch)
                if fold_class_weights is not None:
                    loss = F.cross_entropy(out, batch.y.view(-1), weight=fold_class_weights)
                else:
                    loss = F.cross_entropy(out, batch.y.view(-1))
                loss = loss / float(acc_steps)
                loss.backward()
                total_loss += float(loss) * getattr(batch, "num_graphs", 1) * acc_steps
                if (i + 1) % acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    optimizer.zero_grad()
            if last_i >= 0 and (last_i + 1) % acc_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad()

            val_acc, val_f1, cm_val, prf, *_ = eval_model(model, test_loader, device, num_classes)
            p_val, r_val, f_val = prf
            try:
                if scheduler is not None:
                    scheduler.step(val_acc)
            except Exception:
                pass
            if val_acc > best_val + 1e-5:
                best_val = val_acc
                best_epoch = epoch
                try:
                    torch.save(model.state_dict(), os.path.join(OUT, f"best_model_fold{fold}.pt"))
                except Exception:
                    pass
            if (epoch - best_epoch) >= early_patience:
                print(f"Early stopping fold {fold} at epoch {epoch}. Best val acc: {best_val:.4f} at epoch {best_epoch}")
                break

        # final logging for fold
        print(f"Fold {fold} test Acc: {val_acc:.4f}, F1: {val_f1:.4f} (best acc {best_val:.4f} at epoch {best_epoch})")
        try:
            np.savetxt(os.path.join(OUT, f"fold{fold}_confusion.csv"), cm_val, fmt="%d", delimiter=",")
            pd.DataFrame({"precision": p_val, "recall": r_val, "f1": f_val}).to_csv(os.path.join(OUT, f"fold{fold}_per_class_metrics.csv"))
        except Exception:
            pass
        fold_acc.append(float(val_acc))
        fold_f1.append(float(val_f1))

        # write per-fold CSV row
        try:
            import csv as _csv
            summary_path = os.path.join(OUT, "per_fold_results.csv")
            row = {
                "fold": int(fold),
                "alpha": float(fold_alpha),
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "precision_0": float(p_val[0]) if len(p_val) > 0 else 0.0,
                "recall_0":    float(r_val[0]) if len(r_val) > 0 else 0.0,
                "f1_0":        float(f_val[0]) if len(f_val) > 0 else 0.0,
                "precision_1": float(p_val[1]) if len(p_val) > 1 else 0.0,
                "recall_1":    float(r_val[1]) if len(r_val) > 1 else 0.0,
                "f1_1":        float(f_val[1]) if len(f_val) > 1 else 0.0
            }
            write_header = not os.path.exists(summary_path)
            with open(summary_path, "a", newline="") as cf:
                writer = _csv.DictWriter(cf, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            pass

        # running summary (optional)
        if VERBOSE:
            if len(fold_acc) == 1:
                acc_mean, acc_std = fold_acc[0], float("nan")
                f1_mean, f1_std = fold_f1[0], float("nan")
            else:
                acc_mean, acc_std = float(np.mean(fold_acc)), float(np.std(fold_acc, ddof=0))
                f1_mean, f1_std = float(np.mean(fold_f1)), float(np.std(fold_f1, ddof=0))
            print(f"Fold {fold} done — acc: {val_acc:.4f}, f1: {val_f1:.4f}, alpha: {fold_alpha}")
            print(f"Overall CV (so far) acc: {acc_mean:.4f} ± {acc_std:.4f}, f1: {f1_mean:.4f} ± {f1_std:.4f}")
            try:
                with open(os.path.join(OUT, "cv_summary.txt"), "w") as fh:
                    fh.write(f"K: {K}\n")
                    fh.write(f"collected_folds: {len(fold_acc)}\n")
                    fh.write(f"running_acc_mean: {acc_mean:.6f}\n")
                    fh.write(f"running_f1_mean: {f1_mean:.6f}\n")
            except Exception:
                pass

    # aggregate chosen alphas after CV
    if best_alphas:
        final_alpha_median = float(np.median(best_alphas))
        counts = Counter(best_alphas)
        max_count = max(counts.values())
        candidates = [a for a, c in counts.items() if c == max_count]
        if len(candidates) == 1:
            final_alpha_mode = float(candidates[0])
        else:
            mean_scores = {a: float(np.mean(alpha_scores.get(a, [0.0]))) for a in candidates}
            best_mean = max(mean_scores.values())
            best_candidates = [a for a, m in mean_scores.items() if m == best_mean]
            final_alpha_mode = float(min(best_candidates))
    else:
        final_alpha_median = final_alpha_mode = 1.0

    print("per-fold alphas:", best_alphas)
    print("final_alpha_median:", final_alpha_median, "final_alpha_mode:", final_alpha_mode)

    # final CV summary / save
    print("DEBUG: len(fold_acc) =", len(fold_acc), "fold_acc =", fold_acc)
    print("DEBUG: len(fold_f1)  =", len(fold_f1),  "fold_f1  =", fold_f1)
    if len(fold_acc) != K or len(fold_f1) != K:
        print("Warning: number of collected folds does not match K. Re-check where fold_acc/fold_f1 are appended.")

    if len(fold_acc) == 0:
        acc_mean = acc_std = float("nan")
    elif len(fold_acc) == 1:
        acc_mean = float(np.mean(fold_acc)); acc_std = float("nan")
    else:
        acc_mean = float(np.mean(fold_acc)); acc_std = float(np.std(fold_acc, ddof=0))

    if len(fold_f1) == 0:
        f1_mean = f1_std = float("nan")
    elif len(fold_f1) == 1:
        f1_mean = float(np.mean(fold_f1)); f1_std = float("nan")
    else:
        f1_mean = float(np.mean(fold_f1)); f1_std = float(np.std(fold_f1, ddof=0))

    with open(os.path.join(OUT, "cv_summary.txt"), "w") as fh:
        fh.write(f"K: {K}\n")
        fh.write(f"collected_folds: {len(fold_acc)}\n")
        fh.write(f"final_acc_mean: {acc_mean:.6f}\n")
        fh.write(f"final_acc_std:  {acc_std:.6f}\n")
        fh.write(f"final_f1_mean:  {f1_mean:.6f}\n")
        fh.write(f"final_f1_std:   {f1_std:.6f}\n")

    print(f"Overall CV Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Overall CV F1-score: {f1_mean:.4f} ± {f1_std:.4f}")

    # summarize per-fold CSV if present
    summary_src = os.path.join(OUT, "per_fold_results.csv")
    if os.path.exists(summary_src):
        try:
            summarize_per_fold_results(OUT)
        except Exception as e:
            print("summarize_per_fold_results failed:", e)

    # safe LR logging
    sched = globals().get('scheduler', None)
    opt = globals().get('optimizer', None)
    try:
        if sched is not None and hasattr(sched, "get_last_lr"):
            current_lr = sched.get_last_lr()[0]
        elif isinstance(opt, torch.optim.Optimizer):
            current_lr = opt.param_groups[0].get("lr", 0.0)
        else:
            current_lr = 0.0
    except Exception:
        if isinstance(opt, torch.optim.Optimizer):
            try:
                current_lr = opt.param_groups[0].get("lr", 0.0)
            except Exception:
                current_lr = 0.0
        else:
            current_lr = 0.0
    print(f"Final Learning Rate after CV: {current_lr:.6e}")


if __name__ == "__main__":
    # set deterministic seeds where possible and run cross-validation
    try:
        torch.manual_seed(0)
        np.random.seed(0)
        # if CUDA is available, also seed it
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    except Exception:
        pass

    try:
        run_cv()
    except Exception as e:
        print("run_cv failed:", e)
