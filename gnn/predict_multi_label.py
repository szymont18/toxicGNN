import torch
import torch.nn as nn
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional, List

# --- GNN Model Definition ---
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class AdvancedGNN(torch.nn.Module):
    """
    A Graph Attention Network (GAT) model for graph-level classification.
    This model must be identical to the one used for training.
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
        x = global_mean_pool(x, batch)
        x = self.linear1(x).relu()
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# --- Feature Engineering and SMILES Converter ---
ATOM_FEATURES_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I', 'Na', 'K', 'other']
CHIRAL_TAG_LIST = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]
NODE_FEATURE_DIM = 10 + len(ATOM_FEATURES_LIST) + len(CHIRAL_TAG_LIST)

def one_hot_encode(value, allowed_set):
    if value not in allowed_set:
        value = allowed_set[-1]
    return [1 if s == value else 0 for s in allowed_set]

class SmileConverter:
    """
    Handles the conversion of SMILES strings into graph data structures.
    This must be identical to the one used for training.
    """
    @staticmethod
    def smile_to_data(smile: str, label: Optional[List[float]] = None) -> Optional[Data]:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        AllChem.ComputeGasteigerCharges(mol)
        node_features = []
        for atom in mol.GetAtoms():
            try:
                gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
                if not np.isfinite(gasteiger_charge):
                    gasteiger_charge = 0.0
            except (KeyError, ValueError):
                gasteiger_charge = 0.0
            features = [
                atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
                float(atom.GetHybridization()), atom.GetIsAromatic(),
                atom.GetNumRadicalElectrons(), atom.GetMass(), atom.IsInRing(),
                gasteiger_charge, atom.HasProp('_ChiralityPossible'),
                *one_hot_encode(atom.GetSymbol(), ATOM_FEATURES_LIST),
                *one_hot_encode(atom.GetChiralTag(), CHIRAL_TAG_LIST),
            ]
            node_features.append(features)
        x = torch.tensor(node_features, dtype=torch.float)
        if x.shape[1] != NODE_FEATURE_DIM:
            raise ValueError(f"Node feature dimension mismatch: expected {NODE_FEATURE_DIM}, got {x.shape[1]}")
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        if label is not None:
            y = torch.tensor(label, dtype=torch.float)
            return Data(x=x, edge_index=edge_index, y=y)
        return Data(x=x, edge_index=edge_index)


def predict_toxicity(smiles_string: str, model_path="./models/best_multilabel_model_gnn.pth", cache_dir="./cached_data"):
    """
    Loads a pre-trained multi-label GNN model and predicts the toxicity profile
    for a given SMILES string.
    """
    # --- 1. Setup and Sanity Checks ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    label_columns_file = os.path.join(cache_dir, "label_columns_new_format.txt")
    if not os.path.exists(model_path) or not os.path.exists(label_columns_file):
        print(f"Error: Required files not found.")
        print(f"Make sure '{model_path}' and '{label_columns_file}' are present.")
        return

    # --- 2. Load Labels and Model ---
    with open(label_columns_file, "r") as f:
        label_columns = [line.strip() for line in f]
    num_classes = len(label_columns)

    print(f"Loading model for {num_classes} toxicity assays...")
    model = AdvancedGNN(input_dim=NODE_FEATURE_DIM, hidden_dim=64, output_dim=128, num_classes=num_classes).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- 3. Preprocess Input SMILES ---
    converter = SmileConverter()
    graph_data = converter.smile_to_data(smiles_string.strip())

    if graph_data is None:
        print("Error: Invalid SMILES string provided.")
        return

    # Create a pseudo-batch for the single graph
    loader = DataLoader([graph_data.to(device)], batch_size=1, shuffle=False)
    batched_data = next(iter(loader))

    # --- 4. Perform Inference ---
    with torch.no_grad():
        logits = model(batched_data)
        # Apply sigmoid to get probabilities for each class (assay)
        probabilities = torch.sigmoid(logits)
        # Get binary predictions based on a 0.5 threshold
        predictions = (probabilities > 0.5).int().squeeze().cpu().numpy()

    # --- 5. Display Results ---
    print("\n--- Toxicity Prediction Results ---")
    print(f"SMILES: {smiles_string}")
    for i, assay_name in enumerate(label_columns):
        print(f"- {assay_name:<30}: {probabilities[0, i]*100:.2f}%")

    # The most toxic assay is the one with the highest probability
    most_toxic_assay = label_columns[np.argmax(probabilities[0])]
    print(f"\nThe most toxic assay is: {most_toxic_assay}")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # To run this script, you must have the following files in their respective directories:
    # 1. 'best_multilabel_model_new_format.pth' (in the same directory as this script)
    # 2. 'cached_data/label_columns_new_format.txt' (generated by the training script)

    # Example SMILES string for Aspirin
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    # Run prediction for the example
    predict_toxicity(aspirin_smiles)

    print("\n" + "="*50 + "\n")

    # Interactive loop for user input
    try:
        while True:
            user_input = input("Enter a SMILES string to predict (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
            if user_input:
                predict_toxicity(user_input)
                print("\n" + "="*50 + "\n")
            else:
                print("Please enter a valid SMILES string.")
    except KeyboardInterrupt:
        print("\nExiting.")