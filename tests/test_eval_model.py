import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from src.models.gnn import GINNet
from src.utils.features import eval_model

def make_simple_batch(num_graphs=2, n_nodes=3, n_feat=4, graph_feat_dim=2, num_classes=2):
    lst = []
    for g in range(num_graphs):
        x = torch.randn(n_nodes, n_feat)
        edge_index = torch.tensor([[0,1,1],[1,0,2]], dtype=torch.long)
        y = torch.tensor([g % num_classes], dtype=torch.long)
        d = Data(x=x, edge_index=edge_index, y=y)
        d.graph_feat = torch.randn(graph_feat_dim)
        lst.append(d)
    return Batch.from_data_list(lst)

def test_eval_model_returns_expected_types():
    device = torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else torch.device("cpu")
    batch = make_simple_batch(num_graphs=4)
    # convert Batch to a list of Data objects and move tensors to device
    dataset = batch.to_data_list()
    for d in dataset:
        d.x = d.x.to(device)
        d.edge_index = d.edge_index.to(device)
        d.y = d.y.to(device)
        d.graph_feat = d.graph_feat.to(device)
    loader = DataLoader(dataset, batch_size=1)
    model = GINNet(in_dim=4, hidden=8, num_layers=2, num_classes=2, graph_feat_dim=2).to(device)
    acc, f1, cm, prf, *_ = eval_model(model, loader, device, num_classes=2)
    assert isinstance(acc, float)
    assert isinstance(f1, float)
    assert hasattr(cm, "shape")
    assert len(prf) == 3  # precision, recall, f1 arrays