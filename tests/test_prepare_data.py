import os
import pytest
from src.data.dataset import prepare_data

# Use the same default data location as your script; skip if dataset not present
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "PROTEINS"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Outputs", "GNN_complete"))

def test_prepare_data_smoke_or_skip():
    if not os.path.exists(ROOT):
        pytest.skip(f"Dataset not present at {ROOT} — skipping prepare_data smoke test.")
    prepared = prepare_data(ROOT, OUT, outlier_contamination=0.0, skip_outlier=True)
    # basic contract checks
    assert isinstance(prepared, dict)
    required_keys = {"data_list", "graph_idx_map", "n_feat", "graph_feat_dim", "unique_labels", "gl", "gi", "label_map"}
    assert required_keys.issubset(set(prepared.keys()))
    # sanity on types
    assert isinstance(prepared["data_list"], list)
    assert hasattr(prepared["label_map"], "__getitem__")