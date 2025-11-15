import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, MLP

class GINNet(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, num_layers=3, num_classes=2, graph_feat_dim=0, dropout=0.5, use_bn=True):
        super().__init__()
        convs = []
        dims = [in_dim] + [hidden] * (num_layers - 1)
        for dim in dims:
            mlp = MLP([dim, hidden], final_activation=F.relu)
            convs.append(GINConv(mlp))
        self.convs = torch.nn.ModuleList(convs)

        self.use_bn = bool(use_bn)
        if self.use_bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(len(self.convs))])
        else:
            self.bns = None

        self.dropout = float(dropout)
        self.graph_feat_dim = int(graph_feat_dim)

        lin_in = hidden + self.graph_feat_dim if self.graph_feat_dim > 0 else hidden
        self.lin1 = nn.Linear(lin_in, hidden)
        self.bn_lin = nn.BatchNorm1d(hidden) if self.use_bn else None
        self.lin2 = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        if isinstance(batch, torch.Tensor):
            batch_tensor = batch
            graph_feat = None
        else:
            batch_tensor = batch.batch
            graph_feat = getattr(batch, "graph_feat", None)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn and self.bns is not None:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch_tensor)

        if graph_feat is not None:
            gf = graph_feat.to(x.device).view(x.size(0), -1).float()
            x = torch.cat([x, gf], dim=1)

        x = self.lin1(x)
        if self.bn_lin is not None:
            x = self.bn_lin(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x