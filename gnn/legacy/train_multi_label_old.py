import os
import sys
import time
import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional, List

# --- New, more powerful GNN model ---
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm


# SUGGESTION: Use a more powerful GNN architecture.
# GATConv allows the model to learn attention weights for neighbors.
# BatchNorm helps stabilize training.
class AdvancedGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes, num_heads=4):
        super(AdvancedGNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.2)
        self.bn1 = BatchNorm(hidden_dim * num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.2)
        self.bn2 = BatchNorm(hidden_dim * num_heads)

        self.linear1 = nn.Linear(hidden_dim * num_heads, output_dim)
        # SUGGESTION: Increase dropout for better regularization in the final layers.
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(output_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x)

        # Global pooling aggregates node features into a single graph-level representation.
        x = global_mean_pool(x, batch)

        x = self.linear1(x).relu()
        x = self.dropout(x)
        x = self.linear2(x)

        return x


from torch.utils.data import Dataset

class SmilesDataset(Dataset):
    """Custom PyTorch Dataset for SMILES graphs and their labels."""
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

# --- Feature Engineering Constants ---
ATOM_FEATURES_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I', 'Na', 'K', 'other']
BOND_TYPE_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_STEREO_LIST = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
CHIRAL_TAG_LIST = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]


# SUGGESTION: Add more informative atom features.
# We are adding Gasteiger partial charges and chirality.
# 10 base features + length of one-hot encoded atom types + length of one-hot encoded chiral tags
NODE_FEATURE_DIM = 10 + len(ATOM_FEATURES_LIST) + len(CHIRAL_TAG_LIST)


def one_hot_encode(value, allowed_set):
    """Helper function for one-hot encoding."""
    if value not in allowed_set:
        value = allowed_set[-1] # Default to 'other'
    return [1 if s == value else 0 for s in allowed_set]


class SmileConverter:
    """
    Handles the conversion of SMILES strings into graph data structures
    suitable for PyTorch Geometric, using RDKit for feature extraction.
    """
    @staticmethod
    def smile_to_data(smile: str, label: Optional[List[float]] = None) -> Optional[Data]:
        """Converts a SMILES string to a PyTorch Geometric Data object."""
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None

        # SUGGESTION: Compute Gasteiger charges before feature extraction.
        AllChem.ComputeGasteigerCharges(mol)

        node_features = []
        for atom in mol.GetAtoms():
            # --- MODIFICATION: Sanitize Gasteiger charge feature ---
            try:
                gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
                if not np.isfinite(gasteiger_charge):
                    gasteiger_charge = 0.0  # Replace NaN or inf with 0
            except (KeyError, ValueError):
                gasteiger_charge = 0.0 # If property doesn't exist or fails conversion

            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                float(atom.GetHybridization()),
                atom.GetIsAromatic(),
                atom.GetNumRadicalElectrons(),
                atom.GetMass(),
                atom.IsInRing(),
                # --- New Features ---
                gasteiger_charge, # Use the sanitized Gasteiger charge
                atom.HasProp('_ChiralityPossible'),
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


def convert_and_cache_smiles(file_path, cache_dir="./cached_data", force_reprocess=False):
    """Processes the raw tox21.csv file and caches the results."""
    os.makedirs(cache_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # Use a new cache name to reflect the fixed features
    graph_file = os.path.join(cache_dir, f"{file_name}_graphs_advanced_fixed.pt")
    label_file = os.path.join(cache_dir, f"{file_name}_labels_advanced_fixed.pt")
    label_columns_file = os.path.join(cache_dir, "label_columns_advanced.txt")

    if os.path.exists(graph_file) and os.path.exists(label_file) and not force_reprocess:
        print(f"Loading cached data: {file_name}")
        graphs = torch.load(graph_file, weights_only=False)
        labels = torch.load(label_file, weights_only=False)
        with open(label_columns_file, "r") as f:
            label_columns = [line.strip() for line in f]
    else:
        print(f"Processing and caching data: {file_name}")
        df = pd.read_csv(file_path, index_col=False)
        df.columns = df.columns.str.strip()

        label_columns = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
            'SR-MMP', 'SR-p53'
        ]

        df = df.dropna(subset=['smiles'])
        df[label_columns] = df[label_columns].fillna(0)
        df["smiles"] = df["smiles"].str.strip()

        graphs, labels = [], []
        converter = SmileConverter()
        for _, row in df.iterrows():
            graph = converter.smile_to_data(row["smiles"])
            if graph is not None:
                graphs.append(graph)
                labels.append(row[label_columns].values.astype(float))

        torch.save(graphs, graph_file)
        torch.save(labels, label_file)
        with open(label_columns_file, "w") as f:
            for col in label_columns:
                f.write(f"{col}\n")

    return graphs, labels, label_columns


def preprocessing(file_path, test_size=0.2, batch_size=32, cache_dir="./cached_data"):
    """Full preprocessing pipeline."""
    graphs, labels, label_columns = convert_and_cache_smiles(
        file_path, cache_dir=cache_dir, force_reprocess=False # Set to True to re-process features
    )

    labels_tensor = torch.tensor(np.array(labels), dtype=torch.float)

    X_train, X_test, y_train, y_test = train_test_split(
        graphs, labels_tensor, test_size=test_size, random_state=42
    )

    train_dataset = SmilesDataset(X_train, y_train)
    test_dataset = SmilesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    pos_weights = []
    for i in range(labels_tensor.shape[1]):
        n_pos = (labels_tensor[:, i] == 1).sum()
        n_neg = labels_tensor.shape[0] - n_pos
        weight = n_neg / (n_pos + 1e-6)
        pos_weights.append(weight)

    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float)

    # --- MODIFICATION: Cap the positive weights to prevent extreme values ---
    pos_weight_tensor = torch.clamp(pos_weight_tensor, max=100)

    print(f"Calculated and capped positive class weights for loss function: {pos_weight_tensor}")

    return train_loader, test_loader, len(label_columns), pos_weight_tensor

def train(model, train_loader, optimizer, criterion, device, grad_clip_value=1.0):
    """Trains the model for one epoch with gradient clipping."""
    model.train()
    total_loss = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, labels)

        # --- MODIFICATION: Check for NaN loss as a safeguard ---
        if torch.isnan(loss):
            print("Warning: NaN loss detected. Skipping this batch.")
            continue

        loss.backward()

        # --- MODIFICATION: Add Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


def test(model, test_loader, criterion, device):
    """Evaluates the model on the test set."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            out = model(data)
            
            # Check for NaN outputs from the model
            if torch.isnan(out).any():
                print("Warning: NaN output detected during testing.")
                # Handle NaN outputs, e.g., by creating zero predictions
                # This prevents the script from crashing but indicates a problem
                preds = torch.zeros_like(out, dtype=torch.int)
            else:
                loss = criterion(out, labels)
                total_loss += loss.item() * data.num_graphs
                preds = (torch.sigmoid(out) > 0.5).int()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu().int())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    avg_loss = total_loss / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, accuracy, f1, precision, recall


def main():
    """Main function with improved training process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    file_path = "./data/tox21.csv"
    num_epochs = 100
    patience_epochs = 15 # For early stopping
    learning_rate = 0.001
    # --- End Configuration ---

    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}.")
        # As a fallback for environments like Google Colab, let's try to download it.
        print("Attempting to download Tox21 dataset...")
        try:
            import deepchem as dc
            _, (dataset, _), _ = dc.molnet.load_tox21()
            df = dataset.to_dataframe()
            os.makedirs("./data", exist_ok=True)
            df.to_csv(file_path, index=False)
            print("Download successful.")
        except Exception as e:
            print(f"Could not download dataset. Please place tox21.csv in the ./data/ directory. Error: {e}")
            return


    # Setting force_reprocess=True for the first run after fixing SmileConverter is a good idea
    train_loader, test_loader, num_classes, pos_weights = preprocessing(file_path)

    print(f"\nNumber of classes (assays): {num_classes}")
    print(f"Initializing model with Node Feature Dimension: {NODE_FEATURE_DIM}")
    
    model = AdvancedGNN(input_dim=NODE_FEATURE_DIM, hidden_dim=64, output_dim=128, num_classes=num_classes).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    best_f1 = 0
    best_state = None
    patience_counter = 0

    print("\nStarting model training...")
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_f1, test_prec, test_recall = test(model, test_loader, criterion, device)

        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"Train | Loss: {train_loss:.4f}")
        print(f"Test  | Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

        scheduler.step(test_f1)

        if test_f1 > best_f1:
            best_f1 = test_f1
            patience_counter = 0 # Reset patience
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1_score": best_f1,
            }
        else:
            patience_counter += 1
            if patience_counter >= patience_epochs:
                print(f"\nEarly stopping! F1 score did not improve for {patience_epochs} epochs.")
                break

    if best_state:
        torch.save(best_state, "best_multilabel_model_advanced.pth")
        print(f"\nTraining complete. Saved best model from epoch {best_state['epoch']+1} (F1 Score: {best_f1:.4f}) to best_multilabel_model_advanced.pth")
    else:
        print("\nTraining complete. No model was saved as no improvement was seen.")


if __name__ == "__main__":
    main()
