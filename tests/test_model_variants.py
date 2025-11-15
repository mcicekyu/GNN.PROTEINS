import torch
from typing import cast, Sequence
from torch_geometric.data import Data, Batch
from src.models.gnn import GINNet

# select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_datum(n_nodes=3, n_feat=4, graph_feat_dim=2, label=0):
    x = torch.randn(n_nodes, n_feat)
    edge_index = torch.tensor([[0,1,1],[1,0,2]], dtype=torch.long)
    d = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    d.graph_feat = torch.randn(graph_feat_dim)
    return d

def _move_batch_tensors(batch, device):
    batch.x = batch.x.to(device)
    batch.edge_index = batch.edge_index.to(device)
    if getattr(batch, "y", None) is not None:
        batch.y = batch.y.to(device)
    if getattr(batch, "graph_feat", None) is not None:
        batch.graph_feat = batch.graph_feat.to(device)
    return batch

import pytest
@pytest.mark.parametrize("graph_feat_dim,num_classes", [(2,2), (2,3), (1,2)])
def test_model_forward_shapes(graph_feat_dim, num_classes):
    data_list: Sequence[Data] = [make_datum(graph_feat_dim=graph_feat_dim, n_feat=4, label=i % num_classes) for i in range(2)]
    batch = Batch.from_data_list(cast(list, data_list))
    model = GINNet(in_dim=4, hidden=16, num_layers=2, num_classes=num_classes, graph_feat_dim=graph_feat_dim).to(device)
    batch = _move_batch_tensors(batch, device)
    b = cast(Data, batch)
    out = model(b.x, b.edge_index, b)
    assert out.shape == (len(data_list), num_classes)