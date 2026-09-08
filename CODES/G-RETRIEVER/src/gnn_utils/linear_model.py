"""Modèle linéaire multi-couches (remplacement du GraphTransformer).
Même interface : forward(x, edge_index, edge_attr, batch) -> [B, out_dim].
Utilise des Linear + LayerNorm au lieu de TransformerConv.
Intègre les edge_attr : concaténation aux node features après agrégation.
"""
import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class LinearModel(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 1024,
        out_dim: int = 1024,
        edge_dim: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # edge encoder : edge_attr -> edge_dim, puis scatter_mean aux noeuds
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # première couche : in_dim + edge_dim -> hidden_dim
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim + edge_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # couches intermédiaires
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # dernière couche : hidden_dim -> out_dim
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.norms.append(nn.LayerNorm(out_dim))

        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_attr.size(0) > 0:
            edge_feats = self.edge_net(edge_attr)
            src = edge_index[0]
            agg_edge = scatter(
                edge_feats, src, dim=0, dim_size=x.size(0), reduce="mean"
            )
            x = torch.cat([x, agg_edge], dim=-1)
        else:
            pad = torch.zeros(
                x.size(0), edge_attr.size(1), device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, pad], dim=-1)

        for i in range(self.num_layers):
            x_res = x
            x = self.layers[i](x)
            x = self.norms[i](x)
            x = self.act(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            if x_res.shape == x.shape:
                x = x + x_res

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        graph_emb = scatter(x, batch, dim=0, dim_size=num_graphs, reduce="mean")
        return graph_emb
