"""Dataset MCQ pour G-Retriever : reconstruit le sous-graphe PCST cache."""
import torch
from torch.utils.data import Dataset

LETTERS = ["A", "B", "C", "D"]


def _options_from_row(row):
    return {L: str(row[f"option {L}"]) for L in LETTERS}


class GRetrieverMCQDataset(Dataset):
    """Reconstruit (x, edge_index, edge_attr) du sous-graphe PCST a partir
    des indices globaux caches, en remappant les indices d'aretes -> [0, N-1]
    (exactement comme ton _build_subgraph)."""

    LETTERS = LETTERS

    def __init__(self, df_questions, pcst_cache, graph):
        self.df = df_questions.reset_index(drop=True)
        self.cache = pcst_cache
        self.graph = graph
        assert len(self.df) == len(self.cache), \
            "df et cache PCST doivent avoir la meme longueur / ordre"

    def __len__(self):
        return len(self.df)

    def _extract_subgraph(self, sel_nodes, sel_edges):
        sel_nodes = torch.as_tensor(sel_nodes, dtype=torch.long)
        x = self.graph.x[sel_nodes].float()                      # [N, d]

        if len(sel_edges) > 0:
            sel_edges_t = torch.as_tensor(sel_edges, dtype=torch.long)
            edge_attr = self.graph.edge_attr[sel_edges_t].float()  # [E, d]
            ei = self.graph.edge_index[:, sel_edges_t]             # [2, E] globaux
            # remap indices globaux -> locaux [0, N-1]
            remap = {int(g): i for i, g in enumerate(sel_nodes.tolist())}
            ei_local = torch.tensor(
                [[remap[int(s)] for s in ei[0].tolist()],
                 [remap[int(d)] for d in ei[1].tolist()]],
                dtype=torch.long,
            )
            edge_index = ei_local
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.graph.edge_attr.size(1)),
                                    dtype=torch.float)
        return x, edge_index, edge_attr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        c = self.cache[idx]
        opts = _options_from_row(row)

        x, edge_index, edge_attr = self._extract_subgraph(
            c["selected_nodes"], c["selected_edges"]
        )

        correct_letter = str(row["correct_letter"]).strip().upper()
        targets = {L: f"{L}) {opts[L]}" for L in LETTERS}

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": x.size(0),
            "num_edges": edge_index.size(1),
            "desc": c["desc"],
            "question": row["question"],
            "options": opts,
            "targets": targets,
            "correct_letter": correct_letter,
            "article_id": str(row["article_id"]),
        }


def mcq_collate(batch):
    """Concatene les sous-graphes (style PyG) + garde le reste en listes."""
    xs, eis, eas, batch_vec = [], [], [], []
    node_offset = 0
    for i, b in enumerate(batch):
        n = b["num_nodes"]
        xs.append(b["x"])
        eas.append(b["edge_attr"])
        if b["edge_index"].numel() > 0:
            eis.append(b["edge_index"] + node_offset)
        batch_vec.append(torch.full((n,), i, dtype=torch.long))
        node_offset += n

    x = torch.cat(xs, dim=0)
    edge_attr = torch.cat(eas, dim=0) if eas else torch.zeros((0, x.size(1)))
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.long)
    batch_vec = torch.cat(batch_vec, dim=0)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch_vec,
        "desc": [b["desc"] for b in batch],
        "question": [b["question"] for b in batch],
        "options": [b["options"] for b in batch],
        "targets": [b["targets"] for b in batch],
        "correct_letter": [b["correct_letter"] for b in batch],
        "num_nodes": [b["num_nodes"] for b in batch],
        "num_edges": [b["num_edges"] for b in batch],
        "article_id": [b["article_id"] for b in batch],
    }
