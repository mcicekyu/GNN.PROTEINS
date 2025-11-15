
import torch
from torch_geometric.data import Data, Batch
from src.models.gnn import GINNet

def make_dummy_batch(n_nodes=3, n_feat=4, graph_feat_dim=2):
    x = torch.randn(n_nodes, n_feat)
    edge_index = torch.tensor([[0,1,1],[1,0,2]], dtype=torch.long)
    y = torch.tensor([0], dtype=torch.long)
    d = Data(x=x, edge_index=edge_index, y=y)
    d.graph_feat = torch.randn(graph_feat_dim)
    return Batch.from_data_list([d])

# ...existing code...
def test_ginnet_forward_cpu():
    device = torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else torch.device("cpu")
    batch = make_dummy_batch()

    # build a new Batch with tensors moved to the target device (avoids assigning to Batch attributes)
    data_list = []
    for d in batch.to_data_list():
        nd = Data(x=d.x.to(device), edge_index=d.edge_index.to(device))
        if getattr(d, "y", None) is not None:
            nd.y = d.y.to(device)
        if getattr(d, "graph_feat", None) is not None:
            nd.graph_feat = d.graph_feat.to(device)
        data_list.append(nd)
    batch = Batch.from_data_list(data_list)
    
    # ensure internal batch index lives on the same device as the per-data tensors
    from typing import Any, cast
    batch = cast(Any, batch)
    setattr(batch, "batch", getattr(batch, "batch").to(device))

    model = GINNet(in_dim=4, hidden=8, num_layers=2, num_classes=2, graph_feat_dim=2).to(device)

    # avoid BatchNorm complaining about batch size == 1 by using eval() + no_grad()
    model.eval()
    with torch.no_grad():
        # use the first Data object's tensors (Data has typed attributes) to avoid analyzer warnings
        first = data_list[0]
        out = model(first.x, first.edge_index, batch)
    assert out.shape == (1, 2)
