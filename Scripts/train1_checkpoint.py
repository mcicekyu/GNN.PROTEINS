import os, json, numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs", "GNN_complete"))
GI = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
REMOVED_JSON = os.path.join(OUT, "removed_graphs.json")

if not os.path.exists(REMOVED_JSON):
    print("removed_graphs.json not found:", REMOVED_JSON); raise SystemExit(1)

with open(REMOVED_JSON) as f:
    info = json.load(f)

removed = [r["graph_id"] for r in info.get("removed", [])]
print("Removed graph ids count:", len(removed))
if os.path.exists(GI):
    gi = np.loadtxt(GI, dtype=int)
    for r in info.get("removed", []):
        gid = int(r["graph_id"])
        nodes = np.where(gi == gid)[0]
        print(f"graph {gid}: num_nodes={nodes.size}")
else:
    print("GI file not available to count nodes:", GI)

# quick summaries
if removed:
    gids = np.array(removed)
    print("min, max removed graph id:", gids.min(), gids.max())
    print("removed list (first 50):", removed[:50])