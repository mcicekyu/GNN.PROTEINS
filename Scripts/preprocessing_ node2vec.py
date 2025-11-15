import os,re, numpy as np, pandas as pd
import networkx as nx

# robust runtime import: install node2vec if missing
try:
    from node2vec import Node2Vec
except Exception:
    import subprocess, sys, importlib
    print("node2vec not found — attempting to install via pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "node2vec"])
        importlib.invalidate_caches()
        from node2vec import Node2Vec
        print("node2vec installed and imported successfully.")
    except Exception as e:
        raise RuntimeError("Failed to install or import 'node2vec'. "
                           "Install it manually (pip install node2vec) or check your environment.") from e

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
NA_PATH = os.path.join(ROOT, "PROTEINS_node_attributes.txt")
GI_PATH = os.path.join(ROOT, "PROTEINS_graph_indicator.txt")
OUT_PATH = os.path.join(ROOT, "PROTEINS_node2vec_node_emb.npy")
# load GI and build per-graph adjacency (if you have edges file, build from that instead).
# For demo: build graph per dataset if edge list available; otherwise run node2vec on full graph.
# Example: run node2vec on full graph then aggregate node embeddings by graph id:
gi = np.loadtxt(GI_PATH, dtype=int)
# build graph from edge list (PROTEINS_A.txt) if available, else create empty graph with nodes
G = nx.Graph()
n_nodes = gi.size
G.add_nodes_from(range(1, n_nodes+1))

A_PATH = os.path.join(ROOT, "PROTEINS_A.txt")
if os.path.exists(A_PATH):
    with open(A_PATH, "r") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            parts = re.split(r'[,\s]+', ln)
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except ValueError:
                continue
            # ensure edges reference existing nodes
            if 1 <= u <= n_nodes and 1 <= v <= n_nodes:
                G.add_edge(u, v)
    # node2vec expects string node identifiers for gensim Word2Vec keys
    mapping = {n: str(n) for n in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

# now run node2vec on G
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4, seed=0)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
emb = np.zeros((n_nodes, 64), dtype=float)
for i in range(1, n_nodes+1):
    emb[i-1] = model.wv.get_vector(str(i)) if str(i) in model.wv else np.zeros(64)
np.save(OUT_PATH, emb)

# aggregate per-graph means
import math
Gcount = int(gi.max())
means = np.zeros((Gcount, emb.shape[1]), dtype=float)
for gid in range(1, Gcount+1):
    idx = np.where(gi==gid)[0]
    if idx.size:
        means[gid-1] = emb[idx].mean(axis=0)
    else:
        means[gid-1] = np.nan
np.savetxt(os.path.join(ROOT, "PROTEINS_graph_node2vec_means.txt"), means, delimiter=", ")