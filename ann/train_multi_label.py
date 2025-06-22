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

# --- Model Definition from code2 ---
class ANNModel(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for classification tasks.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5):
        """
        Initializes the ANN model with two hidden layers.

        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim1 (int): The number of neurons in the first hidden layer.
            hidden_dim2 (int): The number of neurons in the second hidden layer.
            output_dim (int): The number of output classes.
            dropout_rate (float): The dropout rate for regularization.
        """
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.output_layer(x)
        return x

# --- Preprocessing and Fingerprint Generation (from code1) ---
def smiles_to_fingerprint(smiles_list, radius=2, nBits=2048):
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

def preprocessing(file_path):
    """
    Loads data from a CSV file, preprocesses it for multi-label classification,
    and converts SMILES strings to Morgan fingerprints.
    """
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

    X = df[smiles_col].tolist()
    y = df[label_columns].values.astype(int)

    X_train_smiles, X_test_smiles, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print("Converting SMILES to Morgan fingerprints for training set...")
    X_train = smiles_to_fingerprint(X_train_smiles)
    print("Converting SMILES to Morgan fingerprints for test set...")
    X_test = smiles_to_fingerprint(X_test_smiles)

    return X_train, X_test, y_train, y_test, label_columns

# --- Modified Training and Evaluation for PyTorch Model ---
def train_and_evaluate(model, X_train, X_test, y_train, y_test, label_columns, epochs, batch_size, learning_rate):
    """Trains the PyTorch model and evaluates its performance."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Convert data to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss() # Suitable for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Training ANN model...")
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    print("\nMaking predictions...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            
            # Convert logits to probabilities and then to binary predictions
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_test = np.concatenate(all_labels, axis=0)

    # Calculate overall metrics
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred) # Subset accuracy
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\nOverall Model Performance:")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Subset Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")

    # Calculate and print per-label metrics
    print("\nPerformance per label:")
    f1_scores = f1_score(y_test, y_pred, average=None, zero_division=0)
    precision_scores = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_scores = recall_score(y_test, y_pred, average=None, zero_division=0)

    for i, label_name in enumerate(label_columns):
        print(f"  {label_name}:")
        print(f"    F1: {f1_scores[i]:.4f}, Precision: {precision_scores[i]:.4f}, Recall: {recall_scores[i]:.4f}")


def main():
    """Main function to run the training and evaluation pipeline."""
    file_path = "./gnn/tox21_summary.csv" 
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'.")
        print("Please ensure your data file is at the correct location.")
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
    X_train, X_test, y_train, y_test, label_columns = preprocessing(file_path)

    # --- Model Initialization ---
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = ANNModel(
        input_dim=input_dim,
        hidden_dim1=HIDDEN_DIM1,
        hidden_dim2=HIDDEN_DIM2,
        output_dim=output_dim,
        dropout_rate=DROPOUT_RATE
    )
    
    # --- Training and Evaluation ---
    train_and_evaluate(
        model, 
        X_train, X_test, y_train, y_test, 
        label_columns,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

if __name__ == '__main__':
    # Add a check for the gnn directory for compatibility with original code structure
    if not os.path.exists("./gnn"):
        print("Creating directory 'gnn' for the data file.")
        os.makedirs("./gnn")
        print("Please place 'tox21_summary.csv' inside the 'gnn' directory.")
    
    # The original script had this sys.path append, which might be relevant in some structures
    # sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
    
    main()