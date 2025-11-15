# Reads and summarizes PROTEINS_full dataset raw files after downloading 
# from TU repository as PROTEINS_full.zip and removing the _full suffix.

# Data Manipulation for graph labels
# Run these in shell to map graph labels 1->0 and 2->1 by changing directory:
# cp /Users/mc/Desktop/GNN.PROTEINS/DATA/PROTEINS/PROTEINS_graph_labels.txt \
#   /Users/mc/Desktop/GNN.PROTEINS/DATA/PROTEINS/PROTEINS_graph_labels.txt.bak
# awk '{ if($1==1) print 0; else if($1==2) print 1; else print $1 }' \
#  /Users/mc/Desktop/GNN.PROTEINS/DATA/PROTEINS/PROTEINS_graph_labels.txt.bak \
#  > /tmp/PROTEINS_graph_labels.txt && mv /tmp/PROTEINS_graph_labels.txt \
#  /Users/mc/Desktop/GNN.PROTEINS/DATA/PROTEINS/PROTEINS_graph_labels.txt
# wc -l /Users/mc/Desktop/GNN.PROTEINS/DATA/PROTEINS/PROTEINS_graph_labels.txt

import os, re
from collections import Counter

ROOT = "/Users/mc/Desktop/GNN.PROTEINS/DATA"
NAME = "PROTEINS"


def _read_int_list(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def _read_node_attr_dim(path):
    # return number of columns in node attributes (0 if file missing)
    if not path or not os.path.exists(path):
        return 0
    num_re = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # find all numeric tokens on the line (works with commas or whitespace)
            nums = num_re.findall(s)
            if nums:
                return len(nums)
            # fallback: try comma splitting then whitespace
            if ',' in s:
                parts = [p for p in s.split(',') if p.strip()]
                if parts:
                    return len(parts)
            return len(s.split())
    return 0

def _find_raw_dir(root, name):
    cand = os.path.join(root, name, "raw")
    if os.path.isdir(cand):
        return cand
    # also accept root/name (some users extract directly)
    cand2 = os.path.join(root, name)
    if os.path.isdir(cand2) and any(p.endswith(".txt") for p in os.listdir(cand2)):
        return cand2
    raise RuntimeError(
        f"Raw dataset directory not found. Expected either:\n"
        f"  {os.path.join(root, name, 'raw')}\n"
        f"or\n"
        f"  {os.path.join(root, name)}\n\n"
        "Place the TU PROTEINS raw files (PROTEINS_A.txt, PROTEINS_graph_indicator.txt, "
        "PROTEINS_graph_labels.txt, PROTEINS_node_attributes.txt, ...) into one of those paths."
    )


def _find_file_variants(raw_dir, name, stem):
    # check common variants: exact stem, NAME_stem, stem without .txt, with .txt
    candidates = []
    for s in [stem, f"{name}_{stem}"]:
        for variant in [s, f"{s}.txt"]:
            p = os.path.join(raw_dir, variant)
            if os.path.exists(p):
                candidates.append(p)
    return candidates[0] if candidates else None


def summarize_PROTEINS(root=ROOT, name=NAME):
    raw_dir = _find_raw_dir(root, name)

    A_path = _find_file_variants(raw_dir, name, "A") or _find_file_variants(raw_dir, name, "A.txt")
    gi_path = _find_file_variants(raw_dir, name, "graph_indicator")
    gl_path = _find_file_variants(raw_dir, name, "graph_labels")
    na_path = _find_file_variants(raw_dir, name, "node_attributes")
    nl_path = _find_file_variants(raw_dir, name, "node_labels")

    files_used = {
        "edges": A_path,
        "graph_indicator": gi_path,
        "graph_labels": gl_path,
        "node_attributes": na_path,
        "node_labels": nl_path,
    }

    if not (A_path and gi_path and gl_path):
        raise RuntimeError(
            "Required TU raw files missing in " + raw_dir + ".\n"
            "Found files: " + ", ".join(sorted(os.listdir(raw_dir))) + "\n"
            "Required: A (edges), graph_indicator, graph_labels (with or without PROTEINS_ prefix)."
        )

    # graph counts
    indicator = _read_int_list(gi_path)
    total = max(indicator) if indicator else 0

    # node feature dim
    feat_dim = _read_node_attr_dim(na_path)

    # graph labels and distribution
    raw_labels = _read_int_list(gl_path)
    try:
        labs = [int(x) for x in raw_labels]
        if labs and min(labs) == 1:
            labs = [l - 1 for l in labs]
    except Exception:
        labs = raw_labels
    dist = dict(Counter(labs))
    num_classes = len(dist)

    raw_files = sorted(os.listdir(raw_dir))

    return {
        "total_graphs": int(total),
        "num_node_features": int(feat_dim),
        "num_classes": int(num_classes),
        "class_distribution": dist,
        "raw_dir": raw_dir,
        "raw_files": raw_files,
        "files_used": files_used,
    }


if __name__ == "__main__":
    try:
        summary = summarize_PROTEINS()
        print(f"Total graphs: {summary['total_graphs']}")
        print(f"Num node features: {summary['num_node_features']}")
        print(f"Num classes: {summary['num_classes']}")
        print(f"Class distribution: {summary['class_distribution']}")
        print(f"Raw dir used: {summary['raw_dir']}")
        print("Raw files present:")
        for fn in summary["raw_files"]:
            print("  ", fn)
        print("Files used (resolved paths):")
        for k, v in summary["files_used"].items():
            print(f"  {k}: {v if v else '(not found)'}")
    except Exception as e:
        print("Error:", e)
