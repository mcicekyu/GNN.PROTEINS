import os, sys, json
import numpy as np
import pandas as pd
import networkx as nx

# ensure repo root importable
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.data import dataset as ds   # reuse functions in src/data/dataset.py
from src.data.dataset import prepare_data

OUT_DIR = os.path.join(REPO, "Outputs", "GNN_complete")
REMOVED_JSON = os.path.join(OUT_DIR, "removed_graphs.json")
DATA_ROOT = os.path.join(REPO, "DATA", "PROTEINS")
INSPECT_OUT = os.path.join(OUT_DIR, "outlier_inspect")
os.makedirs(INSPECT_OUT, exist_ok=True)

def load_raw_tables():
    X_df = ds.read_table_try(os.path.join(DATA_ROOT, "PROTEINS_node_attributes.txt"))
    gi = ds.read_table_try(os.path.join(DATA_ROOT, "PROTEINS_graph_indicator.txt")).values[:,0].astype(int)
    gl = ds.read_table_try(os.path.join(DATA_ROOT, "PROTEINS_graph_labels.txt")).values[:,0].astype(int)
    edges = ds.read_table_try(os.path.join(DATA_ROOT, "PROTEINS_A.txt")).values[:, :2].astype(int)
    return X_df.values.astype(float), gi, gl, edges

def build_subgraph(nodes, edges_df):
    node_global = (nodes + 1).tolist()
    mask = np.isin(edges_df[:,0], node_global) & np.isin(edges_df[:,1], node_global)
    sub = edges_df[mask]
    G = nx.Graph()
    G.add_nodes_from(node_global)
    if sub.size:
        G.add_edges_from([(int(u), int(v)) for u,v in sub])
    return G, sub

def main():
    if not os.path.exists(REMOVED_JSON):
        print("removed_graphs.json not found at:", REMOVED_JSON)
        return
    removed = json.load(open(REMOVED_JSON)).get("removed", [])
    removed_ids = [int(r["graph_id"]) for r in removed]
    print("Removed graph ids:", removed_ids)

    # get full dataset objects (skip_outlier=True to get all graphs)
    prepared_all = prepare_data(DATA_ROOT, OUT_DIR, outlier_contamination=0.02, skip_outlier=True)
    gl = prepared_all["gl"]
    gi = prepared_all["gi"]
    Gnum = int(gi.max())

    X, gi_raw, gl_raw, edges_df = load_raw_tables()

    rows = []
    for gid in range(1, Gnum+1):
        nodes = np.where(gi_raw == gid)[0]
        G, sub = build_subgraph(nodes, edges_df)
        n_nodes = nodes.size
        n_edges = int(sub.shape[0])
        avg_deg = (2.0 * n_edges) / n_nodes if n_nodes>0 else 0.0
        label = int(gl_raw[gid-1])
        rows.append({"graph_id": gid, "n_nodes": n_nodes, "n_edges": n_edges, "avg_deg": avg_deg, "label": label})

    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(INSPECT_OUT, "all_graph_basic_metrics.csv"), index=False)

    df_removed = df_all[df_all["graph_id"].isin(removed_ids)].sort_values("graph_id")
    df_removed.to_csv(os.path.join(INSPECT_OUT, "removed_graphs_basic_metrics.csv"), index=False)

    print("Saved CSVs to", INSPECT_OUT)
    print("Removed label distribution:\n", df_removed["label"].value_counts())
    print("All label distribution:\n", df_all["label"].value_counts())
    print("Removed sizes summary:\n", df_removed["n_nodes"].describe())

if __name__ == "__main__":
    main()