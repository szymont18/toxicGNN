import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from model import ToxicRandomForest
from smiles.SmilesConvertJob import convert_to_smiles
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

def smiles_to_fingerprint(smiles_list, radius=2, nBits=2048):
    """Convert SMILES strings to Morgan fingerprints using MorganGenerator"""
    # Initialize the Morgan fingerprint generator
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Generate fingerprint using the generator
            fp = mfpgen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
        else:
            # If molecule is invalid, use zero vector
            fingerprints.append(np.zeros(nBits, dtype=int)) # Ensure dtype matches output of GetFingerprintAsBitVect
    return np.array(fingerprints)

def preprocessing():
    file_path = './data/tox21-ache-p3.aggregrated.txt'

    # client = Client()

    df = dd.read_csv(file_path, index_col=False, sep='\t', dtype={'FLAG': 'object',
                                                                  'PUBCHEM_CID': 'float64'})

    df = df.repartition(npartitions=8)

    # For Random Forest, we need the original SMILES strings, not the converted Data objects
    # So we'll use the SMILES column directly instead of converting to graph data
    
    # Compute the dataframe to convert from Dask to Pandas
    df = df.compute()
    
    unique_labels = df["ASSAY_OUTCOME"].dropna().unique()
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    df["label_numeric"] = df["ASSAY_OUTCOME"].map(label_to_idx)

    df = df.dropna(subset=["SMILES", "label_numeric"])

    X = df["SMILES"].tolist()
    y = df["label_numeric"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Convert SMILES to Morgan fingerprints
    X_train = smiles_to_fingerprint(X_train)
    X_test = smiles_to_fingerprint(X_test)

    # client.close()
    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.train(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return accuracy, f1, precision, recall

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = preprocessing()

    # Initialize and train the model
    model = ToxicRandomForest(n_estimators=100, max_depth=None, random_state=42)
    
    # Train and evaluate
    accuracy, f1, precision, recall = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    
    # Print results
    print("Model Performance:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Get feature importances
    feature_importances = model.get_feature_importances()
    print("\nTop 10 most important features:")
    top_indices = np.argsort(feature_importances)[-10:]
    for idx in top_indices:
        print(f"Feature {idx}: {feature_importances[idx]:.4f}")

if __name__ == '__main__':
    main()
