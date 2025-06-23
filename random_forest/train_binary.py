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
    """Convert SMILES strings to Morgan fingerprints."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = mfpgen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
        else:
            # Important: handle invalid SMILES by providing a zeroed array of the correct shape
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
    y = df_filtered["label_numeric"].astype(int).tolist()

    X_train_smiles, X_test_smiles, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y
    )

    print("Converting SMILES to Morgan fingerprints for training and test sets...")
    X_train = smiles_to_fingerprint(X_train_smiles)
    X_test = smiles_to_fingerprint(X_test_smiles)

    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """Train the model and evaluate its performance."""
    print("Training Random Forest model...")
    model.train(X_train, y_train)
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return accuracy, f1, precision, recall

class ToxicRandomForest:
    """A wrapper for the scikit-learn RandomForestClassifier."""
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def get_feature_importances(self):
        return self.model.feature_importances_

def main():
    """Main function to run the training, evaluation, and model saving pipeline."""
    file_paths = [
        "./data/tox21-ache-p3.aggregrated.txt",
        "./data/tox21-ap1-agonist-p1.aggregrated.txt",
    ]

    X_train, X_test, y_train, y_test = preprocessing(file_paths)

    model = ToxicRandomForest(n_estimators=100)
    
    accuracy, f1, precision, recall = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    
    print("\nModel Performance:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    model_filename = "./models/toxic_random_forest_model.joblib"
    print(f"\nSaving trained model to {model_filename}...")
    joblib.dump(model, model_filename)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()