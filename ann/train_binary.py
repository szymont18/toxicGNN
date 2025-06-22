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

# The original script had this sys.path append
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# --- PyTorch Model Definition ---
class ANNModel(nn.Module):
    """A simple multi-layer perceptron for binary classification."""
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate=0.5):
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer with a single neuron for binary classification
        self.output_layer = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.output_layer(x)
        return x

# --- Fingerprint Generation and Preprocessing ---
def smiles_to_fingerprint(smiles_list, radius=2, nBits=2048):
    """Convert SMILES strings to Morgan fingerprints."""
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

def preprocessing(file_paths, test_size=0.2):
    """
    Loads data from multiple files, processes it for binary classification,
    and converts SMILES to Morgan fingerprints.
    """
    all_dfs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: Data file not found at {file_path}. Skipping.")
            continue
        df = pd.read_csv(file_path, sep="\t", index_col=False)
        df.columns = df.columns.str.strip()
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError("No valid data files were found. Please check file_paths.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.dropna(subset=["SMILES", "ASSAY_OUTCOME"])

    combined_df["ASSAY_OUTCOME"] = combined_df["ASSAY_OUTCOME"].str.strip().str.lower()
    
    df_filtered = combined_df[combined_df["ASSAY_OUTCOME"].isin(["inactive", "active agonist", "active antagonist"])].copy()
    
    df_filtered.loc[:, "ASSAY_OUTCOME"] = df_filtered["ASSAY_OUTCOME"].replace({
        "active agonist": "active",
        "active antagonist": "active"
    })

    label_to_idx = {"inactive": 0, "active": 1}
    df_filtered.loc[:, "label_numeric"] = df_filtered["ASSAY_OUTCOME"].map(label_to_idx)
    df_filtered = df_filtered.dropna(subset=["label_numeric"])

    X = df_filtered["SMILES"].tolist()
    y = df_filtered["label_numeric"].astype(int).values # Return as numpy array

    X_train_smiles, X_test_smiles, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print("Converting SMILES to Morgan fingerprints for training and test sets...")
    X_train = smiles_to_fingerprint(X_train_smiles)
    X_test = smiles_to_fingerprint(X_test_smiles)

    return X_train, X_test, y_train, y_test

# --- New ANN Wrapper and Evaluation Logic ---
class ToxicANN:
    """A wrapper for the PyTorch ANN model."""
    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ANNModel(input_dim, hidden_dim1, hidden_dim2, dropout_rate).to(self.device)
        print(f"Using device: {self.device}")

    def train(self, X_train, y_train, epochs=20, batch_size=64, learning_rate=0.001):
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1) # Reshape for BCEWithLogitsLoss
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print("Training ANN model...")
        for epoch in range(epochs):
            self.model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X_test):
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            # Apply sigmoid and threshold to get binary predictions
            predicted = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        return predicted.flatten()

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """Train the model and evaluate its performance."""
    # The training process is now encapsulated within the model's class
    model.train(X_train, y_train)
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return accuracy, f1, precision, recall

def main():
    """Main function to run the training and evaluation pipeline."""
    # Create dummy directories and files for demonstration if they don't exist
    if not os.path.exists("./smiles_HK/data"):
        os.makedirs("./smiles_HK/data")
        print("Creating dummy data files for demonstration...")
        dummy_data1 = "SMILES\tASSAY_OUTCOME\nCCO\tinactive\nCNC\tactive agonist\nCCN\tactive antagonist"
        dummy_data2 = "SMILES\tASSAY_OUTCOME\nCCC\tinactive\nC=C\tactive agonist\nCC#N\tinactive"
        with open("./smiles_HK/data/tox21-ache-p3.aggregrated.txt", "w") as f:
            f.write(dummy_data1)
        with open("./smiles_HK/data/tox21-ap1-agonist-p1.aggregrated.txt", "w") as f:
            f.write(dummy_data2)
    
    file_paths = [
        "./smiles_HK/data/tox21-ache-p3.aggregrated.txt",
        "./smiles_HK/data/tox21-ap1-agonist-p1.aggregrated.txt",
    ]

    X_train, X_test, y_train, y_test = preprocessing(file_paths)

    # Initialize the new ANN model
    input_dim = X_train.shape[1]
    model = ToxicANN(input_dim=input_dim)
    
    # The train_and_evaluate function now works with the ToxicANN wrapper
    accuracy, f1, precision, recall = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    
    print("\nModel Performance:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Feature importances are not available for this type of model
    # and have been removed.

if __name__ == '__main__':
    main()