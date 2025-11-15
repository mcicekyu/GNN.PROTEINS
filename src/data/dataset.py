# ...existing code...
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import torch
from torch_geometric.data import Data
import networkx as nx

# added imports for spectral / sparse ops
from scipy.sparse.linalg import eigsh
from scipy import sparse

def read_table_try(path):
    for sep in (r'\s*,\s*', r'\s+'):
        try:
            return pd.read_csv(path, sep=sep, header=None, engine='python')
        except Exception:
            pass
    return pd.read_csv(path, header=None)

# --- new helpers for graph structural features ---
def compute_laplacian_eigs(G, k=5):
    """Compute k smallest non-trivial Laplacian eigenvalues (fallback to zeros)."""
    try:
        n = G.number_of_nodes()
        if n <= 1:
            return np.zeros(k, dtype=float)
        L = nx.laplacian_matrix(G).astype(float)
        # ensure csr for eigsh
        if not sparse.isspmatrix_csr(L):
            L = L.tocsr()
        # eigsh requires k < n, so use at most n-1
        k_eff = min(k, max(0, n - 1))
        if k_eff <= 0:
            vals = np.array([], dtype=float)
        else:
            vals = eigsh(L, k=k_eff, which='SM', return_eigenvectors=False)
            vals = np.asarray(vals, dtype=float).ravel()
            vals = np.sort(np.real(vals))
        if k_eff < k:
            vals = np.pad(vals, (0, k - k_eff), 'constant', constant_values=0.0)
    except Exception:
        vals = np.zeros(k, dtype=float)
    return vals

def safe_graph_diameter(G):
    """Compute graph diameter with safe fallbacks / approximations."""
    try:
        if G.number_of_nodes() == 0:
            return 0.0
        if G.number_of_nodes() <= 200:
            return float(nx.diameter(G))
        # For large graphs, avoid networkx.approximation.eccentricity (not available
        # in some NX versions). Approximate diameter by sampling nodes and taking
        # the maximum eccentricity among samples.
        nodes = list(G.nodes())
        sample_size = min(50, len(nodes))
        import random
        rnd = random.Random(0)
        samples = rnd.sample(nodes, sample_size)
        max_ecc = 0
        for s in samples:
            lengths = nx.single_source_shortest_path_length(G, s)
            if lengths:
                ecc = max(lengths.values())
                if ecc > max_ecc:
                    max_ecc = ecc
        return float(max_ecc)
    except Exception:
        return 0.0

def safe_avg_shortest_path(G):
    """Average shortest path length with fallback: exact on small graphs (per connected component)
    and sample-based approximation for large graphs.
    """
    try:
        n = G.number_of_nodes()
        if n <= 1:
            return 0.0

        # For small graphs, compute exact average shortest path length by aggregating over connected components.
        if n <= 200:
            comps = list(nx.connected_components(G))
            total_pairs = 0.0
            weighted_sum = 0.0
            for comp in comps:
                sz = len(comp)
                if sz <= 1:
                    continue
                sub = G.subgraph(comp)
                avg_sub = nx.average_shortest_path_length(sub)
                pairs = sz * (sz - 1) / 2.0
                total_pairs += pairs
                weighted_sum += avg_sub * pairs
            if total_pairs == 0.0:
                return 0.0
            return float(weighted_sum / total_pairs)

        # For large graphs, approximate by sampling source nodes and averaging finite shortest paths.
        nodes = list(G.nodes())
        sample_size = min(50, len(nodes))
        import random
        rnd = random.Random(0)
        samples = rnd.sample(nodes, sample_size)
        total = 0.0
        count = 0
        for s in samples:
            lengths = nx.single_source_shortest_path_length(G, s)
            # exclude the source itself (distance 0)
            if len(lengths) <= 1:
                continue
            total += sum(lengths.values()) - 0  # source included as 0, safe to subtract or ignore
            count += len(lengths) - 1
        if count == 0:
            return 0.0
        return float(total / count)
    except Exception:
        return 0.0

def degree_histogram_binned(G, bins=(0, 2, 4, 6, 10, 100)):
    """Degree histogram over fixed bins, normalized."""
    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.size == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    hist = np.histogram(degs, bins=bins)[0].astype(float)
    return hist / (degs.size + 1e-9)

def prepare_data(root, out_dir,
                 outlier_contamination=0.02,
                 skip_outlier=False,
                 lap_k=5):
    """
    Loads PROTEINS files from `root`, computes augmented per-graph features,
    removes outliers (unless skip_outlier=True) and returns:
      data_list, graph_idx_map, n_feat, graph_feat_dim, unique_labels, gi, gl
    """
    NODE_ATTR = os.path.join(root, "PROTEINS_node_attributes.txt")
    GRAPH_IND = os.path.join(root, "PROTEINS_graph_indicator.txt")
    GRAPH_LABELS = os.path.join(root, "PROTEINS_graph_labels.txt")
    EDGES = os.path.join(root, "PROTEINS_A.txt")
    if not os.path.isfile(EDGES):
        raise FileNotFoundError(f"Edges file not found: {EDGES}")

    X_df = read_table_try(NODE_ATTR)
    gi = read_table_try(GRAPH_IND).values[:, 0].astype(int)
    gl = read_table_try(GRAPH_LABELS).values[:, 0].astype(int)
    unique_labels = np.unique(gl)
    label_map = {int(l): i for i, l in enumerate(unique_labels)}
    edges_df = read_table_try(EDGES).values[:, :2].astype(int)

    X = X_df.values.astype(float)
    node_scaler = StandardScaler()
    X = node_scaler.fit_transform(X)
    n_nodes, n_feat = X.shape
    Gnum = int(gi.max())
    nodes_in_graph = [np.where(gi == g)[0] for g in range(1, Gnum+1)]

    # graph_means and sizes (kept)
    graph_means = np.vstack([X[nodes].mean(axis=0) if nodes.size > 0 else np.zeros((X.shape[1],), dtype=float) for nodes in nodes_in_graph])
    graph_sizes = np.array([nodes.size for nodes in nodes_in_graph], dtype=float)

    # compute new structural features per graph
    all_graph_feats = []
    for nodes in nodes_in_graph:
        n = nodes.size
        if n == 0:
            # construct zero-feature vector of expected length later via padding
            node_mean = np.zeros((X.shape[1],), dtype=float)
            graph_size = 0.0
            avg_deg = 0.0
            num_edges = 0.0
            density = 0.0
            diameter = 0.0
            avg_path = 0.0
            lap_eigs = np.zeros(lap_k, dtype=float)
            num_cc = 0.0
            deg_hist = np.zeros(len((0,2,4,6,10,100)) - 1, dtype=float)
        else:
            node_global = nodes + 1
            mask = np.isin(edges_df[:,0], node_global) & np.isin(edges_df[:,1], node_global)
            sub_edges = edges_df[mask]
            G = nx.Graph()
            G.add_nodes_from([int(g) for g in node_global.tolist()])
            if sub_edges.size:
                edges_list = [(int(u), int(v)) for u, v in sub_edges]
                G.add_edges_from(edges_list)

            node_mean = X[nodes].mean(axis=0) if nodes.size > 0 else np.zeros((X.shape[1],), dtype=float)
            graph_size = float(n)
            num_edges = float(G.number_of_edges())
            avg_deg = (2.0 * num_edges / graph_size) if graph_size > 0 else 0.0
            if graph_size > 1:
                density = 2.0 * num_edges / (graph_size * (graph_size - 1))
            else:
                density = 0.0
            diameter = safe_graph_diameter(G)
            avg_path = safe_avg_shortest_path(G)
            lap_eigs = compute_laplacian_eigs(G, k=lap_k)
            num_cc = float(nx.number_connected_components(G))
            deg_hist = degree_histogram_binned(G)

        gfeat = np.concatenate([
            np.array([
                graph_size, avg_deg, num_edges, density,
                diameter, avg_path, num_cc
            ], dtype=float),
            lap_eigs.astype(float),
            deg_hist.astype(float)
        ])
        all_graph_feats.append(gfeat)

    # Pad features to equal length and stack
    max_len = max(len(f) for f in all_graph_feats)
    graph_features_raw = np.zeros((len(all_graph_feats), max_len), dtype=float)
    for i, feat in enumerate(all_graph_feats):
        graph_features_raw[i, :len(feat)] = feat
    # scale graph-level features (fit on all graphs) and reuse scaler for outlier detection
    graph_scaler = StandardScaler().fit(graph_features_raw)
    graph_features_scaled = graph_scaler.transform(graph_features_raw)
    graph_feat_dim = graph_features_scaled.shape[1]

    # outlier detection (using the scaled graph-level features)
    if skip_outlier:
        outlier_mask = np.zeros(len(graph_features_scaled), dtype=bool)
    else:
        iso = IsolationForest(contamination=outlier_contamination, random_state=0)
        is_out = iso.fit_predict(graph_features_scaled)
        outlier_mask = (is_out == -1)

    # save removed graphs info
    removed_idx = np.where(outlier_mask)[0] + 1
    removed_info = []
    for gid in removed_idx:
        nodes = np.where(gi == int(gid))[0]
        removed_info.append({"graph_id": int(gid), "num_nodes": int(nodes.size)})
    os.makedirs(out_dir, exist_ok=True)
    removed_json_path = os.path.join(out_dir, "removed_graphs.json")
    with open(removed_json_path, "w") as jf:
        json.dump({"removed": removed_info, "count": int(outlier_mask.sum())}, jf, indent=2)

    # build PyG Data objects (excluding outliers)
    data_list = []
    graph_idx_map = []
    for g_idx in range(1, Gnum+1):
        nodes = nodes_in_graph[g_idx-1]
        if outlier_mask[g_idx-1]:
            continue
        node_global = nodes + 1
        mask = np.isin(edges_df[:,0], node_global) & np.isin(edges_df[:,1], node_global)
        sub_edges = edges_df[mask].copy()
        if sub_edges.size == 0:
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            global_to_local = {g: i for i, g in enumerate(node_global)}
            src = [global_to_local[int(u)] for u in sub_edges[:,0]]
            dst = [global_to_local[int(v)] for v in sub_edges[:,1]]
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        x = torch.tensor(X[nodes], dtype=torch.float)
        graph_feat = torch.tensor(graph_features_scaled[g_idx-1], dtype=torch.float)
        y = torch.tensor([label_map[int(gl[g_idx-1])]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.graph_feat = graph_feat
        data_list.append(data)
        graph_idx_map.append(g_idx-1)

    return {
        "data_list": data_list,
        "graph_idx_map": graph_idx_map,
        "n_feat": n_feat,
        "graph_feat_dim": graph_feat_dim,
        "unique_labels": unique_labels,
        "gl": gl,
        "gi": gi,
        "label_map": label_map
    }
