import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# --- Model Definition ---
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5):
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x

# --- Preprocessing and Fingerprint Generation ---
def smiles_to_fingerprint(smiles_list, radius=2, nBits=2048):
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

def preprocessing(file_path, nBits=2048):
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
    df = df.dropna(subset=[smiles_col])
    df[label_columns] = df[label_columns].fillna(0)
    df[smiles_col] = df[smiles_col].str.strip()
    X = df[smiles_col].tolist()
    y = df[label_columns].values.astype(int)
    X_train_smiles, X_test_smiles, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Converting SMILES to Morgan fingerprints...")
    X_train = smiles_to_fingerprint(X_train_smiles, nBits=nBits)
    X_test = smiles_to_fingerprint(X_test_smiles, nBits=nBits)
    return X_train, X_test, y_train, y_test, label_columns

# --- Training and Evaluation Logic ---
def train_and_evaluate(model, X_train, X_test, y_train, y_test, label_columns, epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Training ANN model...")
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    print("\nMaking predictions...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            outputs = model(batch_X.to(device))
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"\nOverall Model F1 Score (weighted): {f1:.4f}")

def main():
    """Main function to run the training, evaluation, and model saving pipeline."""
    file_path = "./gnn/tox21_summary.csv"
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'.")
        return

    # --- Hyperparameters ---
    NBITS = 2048
    HIDDEN_DIM1 = 512
    HIDDEN_DIM2 = 256
    DROPOUT_RATE = 0.5
    LEARNING_RATE = 0.001
    EPOCHS = 20
    BATCH_SIZE = 64

    # --- Data Loading and Preprocessing ---
    X_train, X_test, y_train, y_test, label_columns = preprocessing(file_path, nBits=NBITS)

    # --- Model Initialization ---
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = ANNModel(input_dim=input_dim, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2, output_dim=output_dim, dropout_rate=DROPOUT_RATE)

    # --- Training and Evaluation ---
    train_and_evaluate(model, X_train, X_test, y_train, y_test, label_columns, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

    # --- ADDED: Save the trained model's state and label columns ---
    model_filename = "ann_multilabel_model.pth"
    print(f"\nSaving trained model and labels to {model_filename}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_columns': label_columns,
        'nbits': NBITS,
        'hidden_dim1': HIDDEN_DIM1,
        'hidden_dim2': HIDDEN_DIM2,
        'dropout_rate': DROPOUT_RATE
    }, model_filename)
    print("Model saved successfully.")
    # --- END ADDITION ---

if __name__ == '__main__':
    main()