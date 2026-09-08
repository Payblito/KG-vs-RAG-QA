"""GraphTransformer pour G-Retriever (inspiré de He et al., NeurIPS 2024).

On encode un sous-graphe (issu du PCST) en une séquence d'embeddings de
noeuds, puis on pool (mean) pour obtenir un unique "graph token".
"""
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import scatter


class GraphTransformer(nn.Module):
    """GraphTransformer multi-couches avec edge features.

    Args:
        in_dim:    dim des embeddings de noeuds (Jina v3 = 1024)
        hidden_dim: dim cachée
        out_dim:   dim de sortie (avant projection vers l'espace LLM)
        edge_dim:  dim des embeddings d'aretes (Jina v3 = 1024)
        num_layers: nombre de couches TransformerConv
        num_heads: nombre de tetes d'attention
        dropout:   dropout
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 1024,
        out_dim: int = 1024,
        edge_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim doit etre divisible par num_heads"
        self.num_layers = num_layers
        self.dropout = dropout

        head_dim = hidden_dim // num_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 1ere couche : in_dim -> hidden_dim
        self.convs.append(
            TransformerConv(
                in_dim, head_dim, heads=num_heads,
                concat=True, beta=True, edge_dim=edge_dim, dropout=dropout,
            )
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # couches intermediaires : hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    hidden_dim, head_dim, heads=num_heads,
                    concat=True, beta=True, edge_dim=edge_dim, dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # derniere couche : hidden_dim -> out_dim (concat=False => moyenne des tetes)
        self.convs.append(
            TransformerConv(
                hidden_dim, out_dim, heads=num_heads,
                concat=False, beta=True, edge_dim=edge_dim, dropout=dropout,
            )
        )
        self.norms.append(nn.LayerNorm(out_dim))

        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        """
        x:          [N, in_dim]   embeddings de noeuds (concat de tous les sous-graphes du batch)
        edge_index: [2, E]
        edge_attr:  [E, edge_dim]
        batch:      [N]           indice de graphe pour chaque noeud (0..B-1)

        Retourne: [B, out_dim]    un graph token par sous-graphe
        """
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = self.act(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            # connexion residuelle si dims compatibles
            if x_res.shape == x.shape:
                x = x + x_res

        # mean pooling par sous-graphe -> [B, out_dim]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        graph_emb = scatter(x, batch, dim=0, dim_size=num_graphs, reduce="mean")
        return graph_emb
