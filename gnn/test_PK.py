import os
import sys
import time
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm
from collections import Counter

# Assuming SmilesDataset is in the specified path
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from smiles.SmilesDataset import SmilesDataset

# Choose SmilesConverter
# from smiles.SmilesConverter import SmileConverter
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


def convert_to_smiles(df):
    """
    Converts the 'SMILES' column of a DataFrame into graph data objects.
    """
    n = len(df)
    converter = SmileConverter()

    df.columns = df.columns.str.strip()
    df = df[df["SMILES"].notnull()]
    df["SMILES"] = df["SMILES"].str.strip().str.upper()

    print("Converting SMILES to graph data...")
    start = time.time()
    df["SMILES_DATA"] = df["SMILES"].map(converter.smile_to_data)
    df = df[df["SMILES_DATA"].notnull()]
    end = time.time()

    print(f"Successfully parsed SMILES: {len(df)} / {n}")
    print(f"Conversion Time: {end - start:.2f} sec")

    return df


def convert_and_cache_smiles(files, cache_dir="./cached_data", force_reprocess=False):
    """
    Processes raw data files, converts SMILES to graphs, and caches the results.
    Loads from cache if available unless reprocessing is forced.
    """
    os.makedirs(cache_dir, exist_ok=True)
    all_graphs = []
    all_labels = []

    for file_path in files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        graph_file = os.path.join(cache_dir, f"{file_name}_graphs.pt")
        label_file = os.path.join(cache_dir, f"{file_name}_labels.pt")

        if os.path.exists(graph_file) and os.path.exists(label_file) and not force_reprocess:
            print(f"Loading cached data: {file_name}")
            graphs = torch.load(graph_file, weights_only=False)
            labels = torch.load(label_file, weights_only=False)
        else:
            print(f"Processing and caching data: {file_name}")
            df = pd.read_csv(file_path, sep="\t", index_col=False)
            df = convert_to_smiles(df)
            df = df.dropna(subset=["ASSAY_OUTCOME"])

            df["ASSAY_OUTCOME"] = df["ASSAY_OUTCOME"].str.strip().str.lower()
            df = df[df["ASSAY_OUTCOME"].isin(["inactive", "active agonist", "active antagonist"])]
            df["ASSAY_OUTCOME"] = df["ASSAY_OUTCOME"].replace({"active agonist": "active", "active antagonist": "active"})

            label_to_idx = {"inactive": 0, "active": 1}
            df["label_numeric"] = df["ASSAY_OUTCOME"].map(label_to_idx)

            graphs = df["SMILES_DATA"].tolist()
            labels = df["label_numeric"].tolist()

            torch.save(graphs, graph_file)
            torch.save(labels, label_file)

        all_graphs.extend(graphs)
        all_labels.extend(labels)

    return all_graphs, all_labels


def preprocessing(file_paths, test_size=0.2, batch_size=32, cache_dir="./cached_data"):
    """
    Full preprocessing pipeline: loads data, splits into train/test sets,
    creates data loaders, and computes class weights for handling imbalance.
    """
    graphs, labels = convert_and_cache_smiles(
        file_paths, cache_dir=cache_dir, force_reprocess=False # Set to False to use cache
    )

    print("Label distribution:", Counter(labels))

    X_train, X_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=test_size, random_state=42, stratify=labels
    )

    train_dataset = SmilesDataset(X_train, y_train)
    test_dataset = SmilesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return train_loader, test_loader, len(set(labels)), class_weights


def train(model, train_loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_correct += (out.argmax(dim=1) == labels).sum().item()
        total_examples += data.num_graphs

    return total_loss / total_examples, total_correct / total_examples


def test(model, test_loader, criterion, device):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            out = model(data)
            loss = criterion(out, labels)
            total_loss += loss.item() * data.num_graphs
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, accuracy, f1, precision, recall


def count_assay_outcome(file_path):
    """
    Utility function to print the distribution of assay outcomes in a file.
    """
    df = pd.read_csv(file_path, sep="\t", index_col=False)
    df.columns = df.columns.str.strip()
    counts = df["ASSAY_OUTCOME"].value_counts(dropna=False)
    print(f"Distribution of ASSAY_OUTCOME in {os.path.basename(file_path)}:")
    for label, count in counts.items():
        print(f"- {label}: {count}")
    print("-" * 30)


def main():
    """
    Main function to run the data processing, model training, and evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    file_paths = [
        "./data/tox21-ache-p3.aggregrated.txt",
        "./data/tox21-ap1-agonist-p1.aggregrated.txt",
    ]
    num_epochs = 10
    # --- End Configuration ---


    for file_path in file_paths:
        if os.path.exists(file_path):
            count_assay_outcome(file_path)
        else:
            print(f"Warning: Data file not found at {file_path}. Skipping.")
            return

    train_loader, test_loader, num_classes, class_weights = preprocessing(file_paths)

    print(f"Number of classes: {num_classes}")
    print(f"Class weights for loss function: {class_weights}")

    # Note: Assuming the input_dim from the data converter is 10.
    # The hidden_dim for GATConv with multiple heads is multiplied by num_heads.
    model = AdvancedGNN(input_dim=10, hidden_dim=64, output_dim=64, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_f1 = 0
    best_state = None

    print("\nStarting model training...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_f1, test_prec, test_recall = test(model, test_loader, criterion, device)

        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"Train | Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Test  | Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1_score": best_f1,
            }

    if best_state:
        torch.save(best_state, "best_model.pth")
        print(f"\nTraining complete. Saved best model from epoch {best_state['epoch']+1} (F1 Score: {best_f1:.4f}) to best_model.pth")
    else:
        print("\nTraining complete. No model was saved as no improvement was seen.")


if __name__ == "__main__":
    main()