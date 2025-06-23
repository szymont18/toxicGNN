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

# --- Improved GNN Model Definition ---
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, BatchNorm

class ImprovedGNN(torch.nn.Module):
    """
    An improved Graph Attention Network (GATv2) model for graph-level classification.
    This model uses more expressive GATv2Conv layers that incorporate edge features,
    has increased depth, and uses concatenated pooling for a richer graph representation.
    """
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_classes, num_heads=4, dropout=0.3):
        super(ImprovedGNN, self).__init__()
        self.dropout_rate = dropout

        # --- GNN Layers ---
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=num_heads, edge_dim=edge_dim, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim * num_heads)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=edge_dim, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim * num_heads)
        self.conv3 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=edge_dim, dropout=dropout)
        self.bn3 = BatchNorm(hidden_dim * num_heads)

        # --- Readout and Classifier Layers ---
        # Concatenated pooling doubles the input size to the linear layers
        self.linear1 = nn.Linear(hidden_dim * num_heads * 2, output_dim)
        self.dropout = nn.Dropout(0.5) # Higher dropout in the dense layers
        self.linear2 = nn.Linear(output_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # --- Graph convolutions ---
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_attr).relu()
        x = self.bn3(x)

        # --- Enhanced Pooling ---
        # Concatenate mean and max pooling to create a more expressive graph embedding
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # --- Classifier ---
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
# --- Atom Features ---
ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I', 'Na', 'K', 'other']
CHIRAL_TAGS = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]
NODE_FEATURE_DIM = 10 + len(ATOM_SYMBOLS) + len(CHIRAL_TAGS) # Base features + one-hot encodings

# --- Bond Features ---
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
EDGE_FEATURE_DIM = len(BOND_TYPES) + 2 # one-hot for bond type + is_conjugated + is_in_ring


def one_hot_encode(value, allowed_set):
    """Helper function for one-hot encoding."""
    if value not in allowed_set:
        # For atoms, default to 'other'. For bonds, this case should be handled carefully.
        if allowed_set == ATOM_SYMBOLS:
            value = allowed_set[-1]
        else:
            # Fallback for unexpected bond types, though RDKit should be consistent.
            return [0] * len(allowed_set)
    return [1 if s == value else 0 for s in allowed_set]


class SmileConverter:
    """
    Handles the conversion of SMILES strings into graph data structures
    with both node and edge features.
    """
    @staticmethod
    def smile_to_data(smile: str, label: Optional[List[float]] = None) -> Optional[Data]:
        """Converts a SMILES string to a PyTorch Geometric Data object."""
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None

        AllChem.ComputeGasteigerCharges(mol)

        # --- Node Feature Extraction ---
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
                *one_hot_encode(atom.GetSymbol(), ATOM_SYMBOLS),
                *one_hot_encode(atom.GetChiralTag(), CHIRAL_TAGS),
            ]
            node_features.append(features)
        x = torch.tensor(node_features, dtype=torch.float)

        if x.shape[1] != NODE_FEATURE_DIM:
             raise ValueError(f"Node feature dimension mismatch: expected {NODE_FEATURE_DIM}, got {x.shape[1]}")

        # --- Edge and Edge Feature Extraction ---
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_features = [
                *one_hot_encode(bond.GetBondType(), BOND_TYPES),
                bond.GetIsConjugated(),
                bond.IsInRing()
            ]
            edge_indices.extend([[i, j], [j, i]])
            edge_attrs.extend([bond_features, bond_features])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if label is not None:
            data.y = torch.tensor(label, dtype=torch.float)

        return data


def convert_and_cache_smiles(file_path, cache_dir="./cached_data", force_reprocess=False):
    """Processes the raw CSV file and caches the generated graph data."""
    os.makedirs(cache_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Use new cache filenames to reflect the improved data format
    graph_file = os.path.join(cache_dir, f"{file_name}_graphs_improved.pt")
    label_file = os.path.join(cache_dir, f"{file_name}_labels_improved.pt")
    label_columns_file = os.path.join(cache_dir, "label_columns_improved.txt")

    if os.path.exists(graph_file) and os.path.exists(label_file) and not force_reprocess:
        print(f"Loading cached data from: {graph_file} and {label_file}")
        graphs = torch.load(graph_file, weights_only=False)
        labels = torch.load(label_file, weights_only=False)
        with open(label_columns_file, "r") as f:
            label_columns = [line.strip() for line in f]
    else:
        print(f"Processing and caching data from: {file_path}")
        df = pd.read_csv(file_path, index_col=False)
        df.columns = df.columns.str.strip()

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
        
        smiles_col = 'SMILES'
        all_expected_cols = label_columns + [smiles_col]
        missing_cols = [col for col in all_expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following required columns are missing from the CSV file: {missing_cols}")

        df = df.dropna(subset=[smiles_col])
        df[label_columns] = df[label_columns].fillna(0)
        df[smiles_col] = df[smiles_col].str.strip()

        graphs, labels = [], []
        converter = SmileConverter()
        print("Converting SMILES to graphs with node and edge features...")
        for _, row in df.iterrows():
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


def preprocessing(file_path, test_size=0.2, batch_size=64, cache_dir="./cached_data", force_reprocess=False):
    """Full preprocessing pipeline: load, cache, split, and create DataLoaders."""
    graphs, labels, label_columns = convert_and_cache_smiles(
        file_path, cache_dir=cache_dir, force_reprocess=force_reprocess
    )

    labels_tensor = torch.tensor(np.array(labels), dtype=torch.float)

    # Check for dataset imbalance
    print("\nDataset Class Distribution (Positive Samples):")
    for i, col in enumerate(label_columns):
        pos_count = (labels_tensor[:, i] == 1).sum().item()
        print(f"- {col}: {pos_count}/{len(labels_tensor)} ({(pos_count/len(labels_tensor))*100:.2f}%)")


    X_train, X_test, y_train, y_test = train_test_split(
        graphs, labels_tensor, test_size=test_size
    )

    train_dataset = SmilesDataset(X_train, y_train)
    test_dataset = SmilesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calculate weights for each class to handle imbalance
    n_total = labels_tensor.shape[0]
    pos_weights = torch.tensor([
        (n_total - (labels_tensor[:, i] == 1).sum()) / ((labels_tensor[:, i] == 1).sum() + 1e-6)
        for i in range(labels_tensor.shape[1])
    ], dtype=torch.float)
    
    pos_weight_tensor = torch.clamp(pos_weights, max=100) # Cap weights

    print(f"\nCalculated positive class weights for loss function: {pos_weight_tensor}")

    return train_loader, test_loader, len(label_columns), pos_weight_tensor


def train(model, train_loader, optimizer, criterion, device, grad_clip_value=1.0):
    """Trains the model for one epoch."""
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
            
            # Ensure output is valid before calculating loss
            if not torch.isnan(out).any():
                loss = criterion(out, labels)
                total_loss += loss.item() * data.num_graphs

            preds = (torch.sigmoid(out) > 0.5).int()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu().int())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    avg_loss = total_loss / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
    
    # Using 'micro' average for F1 is often better for multi-label classification
    f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)

    return avg_loss, accuracy, f1, precision, recall


def main():
    """Main function to run the training and evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    # IMPORTANT: Change this to the path of your CSV file.
    file_path = "./data/tox21_summary.csv"
    num_epochs = 100
    patience_epochs = 100
     # Early stopping patience
    learning_rate = 5e-4 # Adjusted learning rate
    weight_decay = 5e-5 # L2 regularization
    # Set to True to re-process file if you change featurization, False to use cache
    force_reprocess_data = False 
    # --- End Configuration ---

    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'.")
        print("Please ensure your data file is at the correct location.")
        return

    train_loader, test_loader, num_classes, pos_weights = preprocessing(
        file_path, force_reprocess=force_reprocess_data
    )

    print(f"\nNumber of classes (assays): {num_classes}")
    print(f"Node Feature Dimension: {NODE_FEATURE_DIM}")
    print(f"Edge Feature Dimension: {EDGE_FEATURE_DIM}")
    
    model = ImprovedGNN(
        input_dim=NODE_FEATURE_DIM, 
        edge_dim=EDGE_FEATURE_DIM,
        hidden_dim=128,          # Increased hidden dim
        output_dim=256,          # Increased output dim
        num_classes=num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    best_f1 = 0
    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    print("\n--- Starting model training ---")
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_f1, test_prec, test_recall = test(model, test_loader, criterion, device)
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.2f}s] | "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Test F1 (micro): {test_f1:.4f} | Test Acc: {test_acc:.4f}")

        # Learning rate scheduler step based on test loss
        scheduler.step(test_loss)

        # Early stopping logic based on test loss
        if test_loss < best_loss:
            best_loss = test_loss
            best_f1 = test_f1
            patience_counter = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1_score": best_f1,
                "loss": best_loss,
            }
        else:
            patience_counter += 1
            if patience_counter >= patience_epochs:
                print(f"\nEarly stopping! Test loss did not improve for {patience_epochs} epochs.")
                break
    
    total_training_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    if best_state:
        torch.save(best_state, "best_multilabel_model_improved.pth")
        print(f"\nTraining complete. Saved best model from epoch {best_state['epoch']+1} "
              f"(Test Loss: {best_state['loss']:.4f}, F1 Score: {best_state['f1_score']:.4f}) "
              f"to 'best_multilabel_model_improved.pth'")
    else:
        print("\nTraining complete. No model was saved as no improvement was seen.")


if __name__ == "__main__":
    main()