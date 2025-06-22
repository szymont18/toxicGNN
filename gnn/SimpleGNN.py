import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(SimpleGNN, self).__init__()
        self.layer1 = GCNConv(input_dim, hidden_dim)
        self.layer2 = GCNConv(hidden_dim, output_dim)
        self.lin_layer = torch.nn.Linear(output_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.layer1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.layer2(x, edge_index)
        x = F.relu(x)

        # Aggregate data from all nodes to one vector to represent whole graph
        x = global_mean_pool(x, batch)

        return self.lin_layer(x)

