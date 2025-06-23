import torch
import torch.nn as nn

class ANNModel(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for classification tasks.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5):
        """
        Initializes the ANN model with two hidden layers.

        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim1 (int): The number of neurons in the first hidden layer.
            hidden_dim2 (int): The number of neurons in the second hidden layer.
            output_dim (int): The number of output classes.
            dropout_rate (float): The dropout rate for regularization.
        """
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.output_layer(x)
        return x 