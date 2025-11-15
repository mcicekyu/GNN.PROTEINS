# ...existing code...
import os
import argparse
import math
from typing import Any

try:
    import importlib
    cm = importlib.import_module("matplotlib.cm")
    plt = importlib.import_module("matplotlib.pyplot")
    _HAS_MATPLOTLIB = True
except Exception:
    cm = None

    class _MissingPlt:
        def _err(self, *args, **kwargs):
            raise RuntimeError("matplotlib is required for plotting; install it with 'pip install matplotlib'")

        def figure(self, *args, **kwargs):
            self._err()

        def savefig(self, *args, **kwargs):
            self._err()

        def show(self, *args, **kwargs):
            self._err()

        def close(self, *args, **kwargs):
            self._err()

        def title(self, *args, **kwargs):
            self._err()

        def axis(self, *args, **kwargs):
            self._err()

    plt = _MissingPlt()  # type: Any
    _HAS_MATPLOTLIB = False

import networkx as nx

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))


def _read_int_list(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def _read_edges(path):
    edges = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in line.replace(",", " ").split()]
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u - 1, v - 1))  # convert to 0-based
    return edges


def _read_node_attributes(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 1 and " " in parts[0]:
                parts = parts[0].split()
            try:
                rows.append([float(x) for x in parts])
            except Exception:
                rows.append([float(x) for x in parts if x])
    return rows


def build_graph_for_gid(gid, edges, graph_indicator, node_attrs=None):
    # gid: 1-based graph id
    nodes_global = [i for i, g in enumerate(graph_indicator) if g == gid]
    if not nodes_global:
        raise ValueError(f"No nodes found for graph id {gid}")
    G = nx.Graph()
    local_map = {glob: i for i, glob in enumerate(nodes_global)}
    for glob in nodes_global:
        G.add_node(local_map[glob], global_id=glob)
        if node_attrs:
            G.nodes[local_map[glob]]['feat'] = node_attrs[glob]
    s_nodes = set(nodes_global)
    for u, v in edges:
        if u in s_nodes and v in s_nodes:
            G.add_edge(local_map[u], local_map[v])
    return G


def plot_graph(G, node_attr_idx=0, title=None, save_path=None, show=True):
    if not _HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting; install it with 'pip install matplotlib' to use plot_graph.")
    pos = nx.spring_layout(G, seed=42)
    node_colors = None
    if node_attr_idx is not None and all('feat' in G.nodes[n] for n in G.nodes):
        vals = [G.nodes[n]['feat'][node_attr_idx] for n in G.nodes]
        mn, mx = min(vals), max(vals)
        if math.isclose(mx, mn):
            node_colors = [0.5 for _ in vals]
        else:
            node_colors = [(v - mn) / (mx - mn) for v in vals]
        assert cm is not None
        cmap = cm.get_cmap("viridis")
    else:
        assert cm is not None
        cmap = cm.get_cmap("tab10")

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_edges(G, pos, alpha=0.6)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=80,
        node_color=node_colors if node_colors is not None else "tab:blue",
        cmap=cmap if node_colors is not None else None,
        linewidths=0.5,
        edgecolors='k'
    )
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# ...existing code...
def main():
    parser = argparse.ArgumentParser(description="Plot an example PROTEINS graph.")
    parser.add_argument("--gid", type=int, default=1, help="1-based graph id to plot (default: 1)")
    parser.add_argument("--attr", type=int, default=0, help="node attribute column index to color by (default: 0)")
    parser.add_argument("--data", type=str, default=DATA_DIR, help="PROTEINS data dir")
    parser.add_argument("--save", type=str, default=None, help="path to save PNG (optional)")
    parser.add_argument("--no-show", action="store_true", help="do not show the plot interactively")
    args = parser.parse_args()

    data_dir = args.data
    edges_path = os.path.join(data_dir, "PROTEINS_A.txt")
    gi_path = os.path.join(data_dir, "PROTEINS_graph_indicator.txt")
    na_path = os.path.join(data_dir, "PROTEINS_node_attributes.txt")

    if not os.path.exists(edges_path) or not os.path.exists(gi_path):
        raise RuntimeError(f"Required files not found in {data_dir}")

    edges = _read_edges(edges_path)
    graph_indicator = _read_int_list(gi_path)  # values are 1-based
    node_attrs = _read_node_attributes(na_path)

    # build graph and plot
    G = build_graph_for_gid(args.gid, edges, graph_indicator, node_attrs)
    title = f"PROTEINS graph id={args.gid}  nodes={G.number_of_nodes()} edges={G.number_of_edges()}"

    # default Outputs folder at project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_dir = os.path.join(project_root, "Outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    if args.save:
        save_path = args.save
    else:
        save_path = os.path.join(outputs_dir, f"graph_{args.gid:04d}.png")

    plot_graph(G, node_attr_idx=args.attr, title=title, save_path=save_path, show=not args.no_show)
    print("Saved plot to", save_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
