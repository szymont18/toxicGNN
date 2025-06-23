import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm
import numpy as np
import os
import sys

# Choose SmilesConverter
# from smiles.SmilesConverter import SmileConverter
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from smiles.BetterSmilesConverter import SmileConverter


class AdvancedGNN(torch.nn.Module):
    """
    A Graph Attention Network (GAT) model for graph-level classification.
    This model uses GATConv layers, which allow nodes to weigh the importance
    of their neighbors' features. BatchNorm layers are included for stable training.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes, num_heads=4):
        super(AdvancedGNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.2)
        self.bn1 = BatchNorm(hidden_dim * num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.2)
        self.bn2 = BatchNorm(hidden_dim * num_heads)

        self.linear1 = nn.Linear(hidden_dim * num_heads, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(output_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x)

        # Global mean pooling aggregates node features into a single graph-level feature vector.
        x = global_mean_pool(x, batch)

        x = self.linear1(x).relu()
        x = self.dropout(x)
        x = self.linear2(x)

        return x


def predict_toxicity(smiles_string, model_path="./models/best_model_gnn_binary.pth"):
    """
    Loads a pre-trained model and predicts the toxicity of a given SMILES string.

    Args:
        smiles_string (str): The SMILES string of the molecule to be evaluated.
        model_path (str): The path to the saved model state dictionary.

    Returns:
        str: A string indicating whether the substance is "Toxic" or "Non-toxic".
    """
    # --- 1. Load the Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The model must be instantiated with the same parameters as when it was trained.
    # From the original script: input_dim=10, hidden_dim=64, output_dim=64, num_classes=2
    model = AdvancedGNN(input_dim=10, hidden_dim=64, output_dim=64, num_classes=2).to(device)

    # Load the saved state dictionary
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode

    # --- 2. Preprocess the Input SMILES ---
    converter = SmileConverter()
    graph_data = converter.smile_to_data(smiles_string.strip().upper())

    if graph_data is None:
        return "Error: Invalid SMILES string. Could not be parsed."

    # The DataLoader in the original script would batch data. Here, we create a
    # pseudo-batch for a single graph.
    graph_data = graph_data.to(device)
    from torch_geometric.loader import DataLoader
    loader = DataLoader([graph_data], batch_size=1, shuffle=False)
    batched_data = next(iter(loader))


    # --- 3. Perform Inference ---
    with torch.no_grad():
        output = model(batched_data)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    # --- 4. Output the Result ---
    # Assuming "inactive" (0) is non-toxic and "active" (1) is toxic.
    if prediction == 0:
        return "Prediction: Non-toxic"
    else:
        return "Prediction: Toxic"

if __name__ == '__main__':
    # Example Usage:
    # To run this, you must have the 'best_model.pth' file in the same directory,
    # and the required 'smiles' package must be accessible.

    # This SMILES string is for Aspirin, which is generally considered non-toxic in this context
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    print(f"Analyzing SMILES: {aspirin_smiles}")
    result = predict_toxicity(aspirin_smiles)
    print(result)

    print("-" * 30)

    # This SMILES is for a known toxic compound (e.g., a simplified representation of a toxic substance)
    toxic_example_smiles = "C(C(F)(F)F)(C(F)(F)F)O" # Example for a highly fluorinated, potentially toxic compound
    print(f"Analyzing SMILES: {toxic_example_smiles}")
    result = predict_toxicity(toxic_example_smiles)
    print(result)

    print("-" * 30)

    # Interactive input from the user
    try:
        while True:
            user_smiles = input("Enter a SMILES string to predict its toxicity (or 'quit' to exit): ")
            if user_smiles.lower() == 'quit':
                break
            if user_smiles:
                prediction = predict_toxicity(user_smiles)
                print(prediction)
            else:
                print("Please enter a valid SMILES string.")
    except KeyboardInterrupt:
        print("\nExiting.")