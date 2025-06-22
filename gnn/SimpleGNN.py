import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
        self.linear1 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(0.05)
        self.linear2 = nn.Linear(output_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # Aggregate data from all nodes to one vector to represent whole graph
        x = global_mean_pool(x, batch)
        
        x = self.linear1(x).relu()
        x = self.dropout(x)
        x = self.linear2(x)

        return x

