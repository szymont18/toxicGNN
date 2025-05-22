from SimpleGNN import SimpleGNN
from smiles.SmilesConvertJob import convert_to_smiles
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
from torch_geometric.loader import DataLoader
from smiles.SmilesDataset import SmilesDataset
from sklearn.model_selection import train_test_split
import torch


def preprocessing():
    file_path = '../data/tox21-ache-p3.aggregrated.txt'

    # client = Client()

    df = dd.read_csv(file_path, index_col=False, sep='\t', dtype={'FLAG': 'object',
                                                                  'PUBCHEM_CID': 'float64'})

    df = df.repartition(npartitions=8)

    df = convert_to_smiles(df)

    unique_labels = df["ASSAY_OUTCOME"].dropna().unique()
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    df["label_numeric"] = df["ASSAY_OUTCOME"].map(label_to_idx)

    df = df.dropna(subset=["SMILES_DATA", "label_numeric"])

    X = df["SMILES_DATA"].tolist()
    y = df["label_numeric"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_dataset = SmilesDataset(X_train, y_train)
    test_dataset = SmilesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # client.close()
    return train_loader, test_loader, len(unique_labels)


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

        preds = out.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_examples += data.num_graphs

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, n_classes = preprocessing()

    model = SimpleGNN(input_dim=1, hidden_dim=64, output_dim=64, num_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 20
    for epoch in range(num_epochs):
        loss, acc = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()
