import torch
from torch_geometric.data import Data
import pysmiles
from typing import Optional


class SmileConverter:
    def __init__(self):
        pass

    @staticmethod
    def smile_to_data(smile: str) -> Optional[Data]:
        try:
            graph = pysmiles.read_smiles(smile, explicit_hydrogen=False)
        except Exception as e:
            return None

        # Node feature extraction:
        # In the feature we can add some features from dataset to each node (like mass, ...)
        atom_features = [[i] for i in range(len(graph.nodes))]

        x = torch.tensor(atom_features, dtype=torch.float)

        node_mapping = {node: i for i, node in enumerate(graph.nodes())}

        edge_index = []
        for src, dst, _ in graph.edges(data=True):
            # Add edges in both directions because molecular graphs are undirected
            edge_index.append([node_mapping[src], node_mapping[dst]])
            edge_index.append([node_mapping[dst], node_mapping[src]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)
