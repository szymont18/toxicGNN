import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from SimpleGNN import SimpleGNN
from smiles.SmilesConvertJob import convert_to_smiles
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
from torch_geometric.loader import DataLoader
from smiles.SmilesDataset import SmilesDataset
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler



def preprocessing():
    # file_path = './data/tox21-ache-p3.aggregrated.txt'
    # file_path = './data/tox21-ap1-agonist-p1.txt'
    file_path = './data/tox21-ap1-agonist-p1.aggregrated.txt'

    # client = Client()

    df = dd.read_csv(file_path, index_col=False, sep='\t', dtype={'FLAG': 'object',
                                                                  'PUBCHEM_CID': 'float64',
                                                                  'PURITY_RATING_4M': 'object'})

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
    
    print("Label distribution:", Counter(y))

    train_dataset = SmilesDataset(X_train, y_train)
    test_dataset = SmilesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # client.close()
    label_counts = Counter(y)
    total = sum(label_counts.values())
    num_classes = len(label_counts)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float)


    return train_loader, test_loader, num_classes, class_weights



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


def test(model, test_loader, criterion, device):
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            out = model(data)
            loss = criterion(out, labels)
            
            total_loss += loss.item() * data.num_graphs
            
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # print(classification_report(all_labels, all_preds, digits=4))

    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    avg_loss = total_loss / len(test_loader.dataset)
    
    return avg_loss, accuracy, f1, precision, recall


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, n_classes, class_weights = preprocessing()
    
    print(f"Class weights: {class_weights}")

    model = SimpleGNN(input_dim=1, hidden_dim=64, output_dim=64, num_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))


    num_epochs = 20
    best_f1 = 0
    best_acc = 0
    best_precision = 0
    best_recall = 0
    best_model_state = None
    best_optimizer_state = None
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # Testing
        test_loss, test_acc, test_f1, test_precision, test_recall = test(model, test_loader, criterion, device)
        print(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print(f"F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_acc = test_acc
            best_precision = test_precision
            best_recall = test_recall
            best_model_state = model.state_dict().copy()
            best_optimizer_state = optimizer.state_dict().copy()
            best_epoch = epoch
    
    if best_model_state is not None:
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'f1_score': best_f1,
        }, 'best_model.pth')
    
    print(f"Best model from epoch {best_epoch + 1} with:")
    print(f"F1 Score: {best_f1:.4f}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")


if __name__ == '__main__':
    main()
