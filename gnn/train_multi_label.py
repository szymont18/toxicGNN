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

# --- GNN Model Definition ---
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm

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


from torch.utils.data import Dataset

class SmilesDataset(Dataset):
    """Custom PyTorch Dataset for SMILES graphs and their labels."""
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # The DataLoader will handle batching of graph Data objects.
        # We return the graph and its corresponding label tensor.
        return self.graphs[idx], self.labels[idx]

# --- Feature Engineering Constants ---
ATOM_FEATURES_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I', 'Na', 'K', 'other']
BOND_TYPE_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_STEREO_LIST = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
CHIRAL_TAG_LIST = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]

# Define the dimensionality of node features based on the selected atomic properties.
# 10 base numeric features + one-hot encoding for atom type + one-hot encoding for chiral tag.
NODE_FEATURE_DIM = 10 + len(ATOM_FEATURES_LIST) + len(CHIRAL_TAG_LIST)


def one_hot_encode(value, allowed_set):
    """Helper function for one-hot encoding a value within a predefined set."""
    if value not in allowed_set:
        value = allowed_set[-1] # Default to the 'other' category if not found
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
            # If RDKit cannot parse the SMILES string, skip it.
            return None

        # Compute Gasteiger partial charges for atoms, which can be a useful feature.
        AllChem.ComputeGasteigerCharges(mol)

        node_features = []
        for atom in mol.GetAtoms():
            # Sanitize Gasteiger charge feature to prevent errors from non-numeric values.
            try:
                gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
                if not np.isfinite(gasteiger_charge):
                    gasteiger_charge = 0.0  # Replace NaN or inf with 0
            except (KeyError, ValueError):
                gasteiger_charge = 0.0 # If property doesn't exist or fails conversion

            # Assemble the feature vector for each atom (node).
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                float(atom.GetHybridization()),
                atom.GetIsAromatic(),
                atom.GetNumRadicalElectrons(),
                atom.GetMass(),
                atom.IsInRing(),
                gasteiger_charge,
                atom.HasProp('_ChiralityPossible'),
                *one_hot_encode(atom.GetSymbol(), ATOM_FEATURES_LIST),
                *one_hot_encode(atom.GetChiralTag(), CHIRAL_TAG_LIST),
            ]
            node_features.append(features)
        x = torch.tensor(node_features, dtype=torch.float)

        # Sanity check for feature dimensions.
        if x.shape[1] != NODE_FEATURE_DIM:
             raise ValueError(f"Node feature dimension mismatch: expected {NODE_FEATURE_DIM}, got {x.shape[1]}")

        # Extract edge information (bonds) for the graph.
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            # Add edges in both directions for an undirected graph.
            edge_indices.extend([[i, j], [j, i]])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        if label is not None:
            y = torch.tensor(label, dtype=torch.float)
            return Data(x=x, edge_index=edge_index, y=y)
        return Data(x=x, edge_index=edge_index)


def convert_and_cache_smiles(file_path, cache_dir="./cached_data", force_reprocess=False):
    """Processes the raw CSV file and caches the generated graph data."""
    os.makedirs(cache_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # --- MODIFICATION: Use new cache filenames to reflect the different data format ---
    graph_file = os.path.join(cache_dir, f"{file_name}_graphs_new_format.pt")
    label_file = os.path.join(cache_dir, f"{file_name}_labels_new_format.pt")
    label_columns_file = os.path.join(cache_dir, "label_columns_new_format.txt")

    if os.path.exists(graph_file) and os.path.exists(label_file) and not force_reprocess:
        print(f"Loading cached data from: {graph_file} and {label_file}")
        graphs = torch.load(graph_file, weights_only=False)
        labels = torch.load(label_file, weights_only=False)
        with open(label_columns_file, "r") as f:
            label_columns = [line.strip() for line in f]
    else:
        print(f"Processing and caching data from: {file_path}")
        df = pd.read_csv(file_path, index_col=False)
        df.columns = df.columns.str.strip() # Sanitize column names

        # --- MODIFICATION: Updated list of label columns based on user-provided format ---
        label_columns = [
            'ahr-p1', 'ap1-agonist-p1', 'ar-bla-agonist-p1', 'ar-bla-antagonist-p1',
            'ar-mda-kb2-luc-agonist-p1', 'ar-mda-kb2-luc-agonist-p3',
            'ar-mda-kb2-luc-antagonist-p1', 'ar-mda-kb2-luc-antagonist-p2',
            'are-bla-p1', 'aromatase-p1', 'car-agonist-p1', 'car-antagonist-p1',
            'dt40-p1', 'elg1-luc-agonist-p1', 'er-bla-agonist-p2', 'er-bla-antagonist-p1',
            'er-luc-bg1-4e2-agonist-p2', 'er-luc-bg1-4e2-agonist-p4',
            'er-luc-bg1-4e2-antagonist-p1', 'er-luc-bg1-4e2-antagonist-p2',
            'erb-bla-antagonist-p1', 'erb-bla-p1', 'err-p1', 'esre-bla-p1',
            'fxr-bla-agonist-p2', 'fxr-bla-antagonist-p1', 'gh3-tre-agonist-p1',
            'gh3-tre-antagonist-p1', 'gr-hela-bla-agonist-p1', 'gr-hela-bla-antagonist-p1',
            'h2ax-cho-p2', 'hdac-p1', 'hre-bla-agonist-p1', 'hse-bla-p1',
            'luc-biochem-p1', 'mitotox-p1'
        ]
        
        # --- MODIFICATION: Using 'SMILES' column as specified by the new format ---
        smiles_col = 'SMILES'

        # Validate that all required columns exist in the dataframe
        all_expected_cols = label_columns + [smiles_col]
        missing_cols = [col for col in all_expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following required columns are missing from the CSV file: {missing_cols}")

        df = df.dropna(subset=[smiles_col])
        df[label_columns] = df[label_columns].fillna(0) # Fill missing labels with 0 (inactive)
        df[smiles_col] = df[smiles_col].str.strip()

        graphs, labels = [], []
        converter = SmileConverter()
        print("Converting SMILES to graphs...")
        for _, row in df.iterrows():
            # --- MODIFICATION: Use the correct 'SMILES' column name ---
            graph = converter.smile_to_data(row[smiles_col])
            if graph is not None:
                graphs.append(graph)
                labels.append(row[label_columns].values.astype(float))
        
        print(f"Successfully converted {len(graphs)} molecules.")
        torch.save(graphs, graph_file)
        torch.save(labels, label_file)
        with open(label_columns_file, "w") as f:
            for col in label_columns:
                f.write(f"{col}\n")
        print(f"Saved processed data to {cache_dir}")

    return graphs, labels, label_columns


def preprocessing(file_path, test_size=0.2, batch_size=32, cache_dir="./cached_data"):
    """Full preprocessing pipeline: load, cache, split, and create DataLoaders."""
    graphs, labels, label_columns = convert_and_cache_smiles(
        file_path, cache_dir=cache_dir, force_reprocess=True # Set to True to re-process file
    )

    labels_tensor = torch.tensor(np.array(labels), dtype=torch.float)

    X_train, X_test, y_train, y_test = train_test_split(
        graphs, labels_tensor, test_size=test_size, random_state=42
    )

    train_dataset = SmilesDataset(X_train, y_train)
    test_dataset = SmilesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calculate weights for each class to handle imbalance in the dataset.
    pos_weights = []
    for i in range(labels_tensor.shape[1]):
        n_pos = (labels_tensor[:, i] == 1).sum()
        n_neg = labels_tensor.shape[0] - n_pos
        weight = n_neg / (n_pos + 1e-6) # Add epsilon to avoid division by zero
        pos_weights.append(weight)

    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float)
    pos_weight_tensor = torch.clamp(pos_weight_tensor, max=100) # Cap weights to prevent instability

    print(f"Calculated positive class weights for loss function: {pos_weight_tensor}")

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

        if torch.isnan(loss):
            print("Warning: NaN loss detected. Skipping this batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value) # Prevent exploding gradients
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
            
            if torch.isnan(out).any():
                print("Warning: NaN output detected during testing. Substituting with zeros.")
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
    """Main function to run the training and evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # IMPORTANT: Change this to the path of your CSV file.
    file_path = "./data/tox21_summary.csv" 
    num_epochs = 100
    patience_epochs = 100 # For early stopping
    learning_rate = 0.001

    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'.")
        print("Please ensure your data file is at the correct location and the 'file_path' variable is set correctly.")
        print("The CSV file must contain a 'SMILES' column and the specified assay columns for the model to work.")
        return


    # Setting force_reprocess=True might be needed if you change the data file
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

    print("\n--- Starting model training ---")
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_f1, test_prec, test_recall = test(model, test_loader, criterion, device)
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.2f}s] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")

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
    
    total_training_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    if best_state:
        torch.save(best_state, "./models/best_multilabel_model_gnn.pth")
        print(f"\nTraining complete. Saved best model from epoch {best_state['epoch']+1} (F1 Score: {best_f1:.4f}) to 'best_multilabel_model_gnn.pth'")
    else:
        print("\nTraining complete. No model was saved as no improvement was seen.")


if __name__ == "__main__":
    main()
