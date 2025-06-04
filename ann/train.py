import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from model import ANNModel
from smiles.SmilesConvertJob import convert_to_smiles # Not strictly needed if we only use SMILES
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

NBITS = 2048 # Define nBits for fingerprints globally or pass as param

def smiles_to_fingerprint(smiles_list, radius=2, nBits=NBITS):
    """Convert SMILES strings to Morgan fingerprints using MorganGenerator"""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = mfpgen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
        else:
            fingerprints.append(np.zeros(nBits, dtype=int))
    return np.array(fingerprints)

def preprocessing():
    file_path = './data/tox21-ache-p3.aggregrated.txt'

    df = dd.read_csv(file_path, index_col=False, sep='\t', dtype={'FLAG': 'object',
                                                                  'PUBCHEM_CID': 'float64'})
    df = df.repartition(npartitions=8)
    df = df.compute()
    
    unique_labels = df["ASSAY_OUTCOME"].dropna().unique()
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    df["label_numeric"] = df["ASSAY_OUTCOME"].map(label_to_idx)
    df = df.dropna(subset=["SMILES", "label_numeric"])

    X = df["SMILES"].tolist()
    y = df["label_numeric"].tolist()

    X_train_smiles, X_test_smiles, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train_fp = smiles_to_fingerprint(X_train_smiles)
    X_test_fp = smiles_to_fingerprint(X_test_smiles)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_fp, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_fp, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, len(unique_labels)

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_examples += labels.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy

def test_model(model, test_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_examples = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_examples += labels.size(0)
    
    avg_loss = total_loss / total_examples
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1, precision, recall

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader, n_classes = preprocessing()

    # Define model, optimizer, and criterion
    input_dim = NBITS # Number of bits in Morgan fingerprint
    hidden_dim1 = 128
    hidden_dim2 = 64
    model = ANNModel(input_dim, hidden_dim1, hidden_dim2, n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20 # Can be adjusted
    best_f1 = 0
    best_acc = 0
    best_precision = 0
    best_recall = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        test_loss, test_acc, test_f1, test_precision, test_recall = test_model(model, test_loader, criterion, device)
        print(f"  Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print(f"  F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_acc = test_acc
            best_precision = test_precision
            best_recall = test_recall
            # Optionally save the best model
            # torch.save(model.state_dict(), 'best_ann_model.pth')
            # print(f"    New best model saved with F1: {best_f1:.4f}")

    print("Training finished.")
    print(f"F1 Score: {best_f1:.4f}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")


if __name__ == '__main__':
    main() 