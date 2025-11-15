# ...existing code...
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import torch
import os
import numpy as np

# try to load existing PCA node scores from Outputs (3D or 2D), else make random embeddings
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "Outputs")
os.makedirs(OUT, exist_ok=True)

def load_pca_scores():
    candidates = [
        os.path.join(OUT, "PCA_nodes_raw_3d_scores.csv"),
        os.path.join(OUT, "PCA_nodes_top10_3d_scores.csv"),
        os.path.join(OUT, "PCA_nodes_raw_2d_scores.csv"),
        os.path.join(OUT, "PCA_nodes_top10_2d_scores.csv"),
        os.path.join(OUT, "pcs_all.csv")
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                arr = np.loadtxt(p, delimiter=',', skiprows=0)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, arr.shape[0])
                return arr.astype(np.float32)
            except Exception:
                continue
    # fallback: small random embedding
    return np.random.randn(100, 3).astype(np.float32)

emb_np = load_pca_scores()
emb = torch.tensor(emb_np)

# try to attach per-node metadata (graph labels) if files exist
meta = None
GI = os.path.join(ROOT, "DATA", "PROTEINS", "PROTEINS_graph_indicator.txt")
GL = os.path.join(ROOT, "DATA", "PROTEINS", "PROTEINS_graph_labels.txt")
if os.path.isfile(GI) and os.path.isfile(GL):
    try:
        gi = np.loadtxt(GI, dtype=int)
        gl = np.loadtxt(GL, dtype=int)
        # map node -> graph label
        meta = [str(int(gl[g-1])) for g in gi[: emb.shape[0]]]
    except Exception:
        meta = None

# ensure a deterministic, absolute run-dir inside the Scripts folder
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs", "embeds")
os.makedirs(RUNS_DIR, exist_ok=True)

# write embeddings to TensorBoard using absolute path
writer = SummaryWriter(log_dir=RUNS_DIR)
step = int(datetime.now().timestamp())
writer.add_embedding(emb, metadata=meta, tag="pca_embeddings", global_step=step)
writer.flush()
writer.close()

print("Wrote embedding to:", RUNS_DIR)