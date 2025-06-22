import os
import sys
import time
import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from SimpleGNN import SimpleGNN
from smiles.SmilesDataset import SmilesDataset
from smiles.SmilesConverter import SmileConverter


def convert_to_smiles(df):
    n = len(df)
    converter = SmileConverter()

    df.columns = df.columns.str.strip()
    df = df[df["SMILES"].notnull()]
    df["SMILES"] = df["SMILES"].str.strip().str.upper()

    start = time.time()

    df["SMILES_DATA"] = df["SMILES"].map(converter.smile_to_data)
    df = df[df["SMILES_DATA"].notnull()]

    end = time.time()

    print(f"Parsed SMILES: {len(df)} / {n}")
    print(f"Time: {end - start:.2f} sec")

    return df


def convert_and_cache_smiles(files, cache_dir="./cached_data", force_reprocess=False):
    os.makedirs(cache_dir, exist_ok=True)

    all_graphs = []
    all_labels = []

    for file_path in files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        graph_file = os.path.join(cache_dir, f"{file_name}_graphs.pt")
        label_file = os.path.join(cache_dir, f"{file_name}_labels.pt")

        if (
            os.path.exists(graph_file)
            and os.path.exists(label_file)
            and not force_reprocess
        ):
            print(f"Loading cached: {file_name}")
            graphs = torch.load(graph_file)
            labels = torch.load(label_file)
        else:
            print(f"Processing: {file_name}")
            df = pd.read_csv(file_path, sep="\t", index_col=False)
            df = convert_to_smiles(df)
            df = df.dropna(subset=["ASSAY_OUTCOME"])

            df["ASSAY_OUTCOME"] = df["ASSAY_OUTCOME"].str.strip().str.lower()

            df = df[
                df["ASSAY_OUTCOME"].isin(
                    ["inactive", "active agonist", "active antagonist"]
                )
            ]

            df["ASSAY_OUTCOME"] = df["ASSAY_OUTCOME"].replace(
                {"active agonist": "active", "active antagonist": "active"}
            )

            label_to_idx = {"inactive": 0, "active": 1}
            df["label_numeric"] = df["ASSAY_OUTCOME"].map(label_to_idx)
            #            ================================

            graphs = df["SMILES_DATA"].tolist()
            labels = df["label_numeric"].tolist()

            torch.save(graphs, graph_file)
            torch.save(labels, label_file)

        all_graphs.extend(graphs)
        all_labels.extend(labels)

    return all_graphs, all_labels


def preprocessing(file_paths, test_size=0.2, batch_size=32, cache_dir="./cached_data"):
    graphs, labels = convert_and_cache_smiles(
        file_paths, cache_dir=cache_dir, force_reprocess=False
    )

    print("Label distribution:", Counter(labels))

    X_train, X_test, y_train, y_test = train_test_split(
        graphs, labels, test_size=test_size, random_state=42, stratify=labels
    )

    train_dataset = SmilesDataset(X_train, y_train)
    test_dataset = SmilesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return train_loader, test_loader, len(set(labels)), class_weights


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

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
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            out = model(data)
            loss = criterion(out, labels)
            total_loss += loss.item() * data.num_graphs

            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return (
        total_loss / len(test_loader.dataset),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        recall_score(all_labels, all_preds, average="weighted", zero_division=0),
    )


def count_assay_outcome(file_path):
    df = pd.read_csv(file_path, sep="\t")
    df.columns = df.columns.str.strip()
    counts = df["ASSAY_OUTCOME"].value_counts(dropna=False)
    print(f"Distribution at ASSAY_OUTCOME in {file_path}:")
    for label, count in counts.items():
        print(f"{label}: {count}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_paths = [
        "./data/tox21-ap1-agonist-p1.aggregrated.txt",
        "./data/tox21-ar-bla-agonist-p1.aggregrated.txt",
    ]

    for file_path in file_paths:
        count_assay_outcome(file_path)

    train_loader, test_loader, num_classes, class_weights = preprocessing(file_paths)

    print(f"Num of classes: {num_classes}")
    print(f"Class weights: {class_weights}")

    model = SimpleGNN(
        input_dim=1, hidden_dim=64, output_dim=64, num_classes=num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    num_epochs = 5
    best_f1 = 0
    best_state = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_f1, test_prec, test_recall = test(
            model, test_loader, criterion, device
        )

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train  - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(
            f"Test   - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_recall:.4f}"
        )

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
        print(
            f"\nSaved the best from epoch {best_state['epoch']+1} (F1: {best_f1:.4f})"
        )


if __name__ == "__main__":
    main()
