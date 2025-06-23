import os
import sys
import pandas as pd
import numpy as np
import joblib  # Import joblib to save the model

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

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
    )

    print("Converting SMILES to Morgan fingerprints for training set...")
    X_train = smiles_to_fingerprint(X_train_smiles)
    print("Converting SMILES to Morgan fingerprints for test set...")
    X_test = smiles_to_fingerprint(X_test_smiles)

    return X_train, X_test, y_train, y_test, label_columns

def train_and_evaluate(model, X_train, X_test, y_train, y_test, label_columns):
    """Trains the model and evaluates its performance."""
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\nOverall Model Performance:")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Subset Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")

def main():
    """Main function to run the training and evaluation pipeline."""
    file_path = "./data/tox21_summary.csv" 
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'.")
        print("Please ensure your data file is at the correct location.")
        return

    X_train, X_test, y_train, y_test, label_columns = preprocessing(file_path)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    
    train_and_evaluate(model, X_train, X_test, y_train, y_test, label_columns)
    
    saved_model_data = {
        'model': model,
        'label_columns': label_columns
    }
    model_filename = "./models/rf_multilabel_model.joblib"
    print(f"\nSaving trained model and labels to {model_filename}...")
    joblib.dump(saved_model_data, model_filename)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()