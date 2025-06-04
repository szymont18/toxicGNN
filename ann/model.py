import torch
import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ANNModel, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.layer_out = nn.Linear(hidden_dim2, output_dim)
        self.softmax = nn.Softmax(dim=1) # Use Softmax for multi-class classification

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu1(x)
        x = self.layer_2(x)
        x = self.relu2(x)
        x = self.layer_out(x)
        x = self.softmax(x)
        return x 