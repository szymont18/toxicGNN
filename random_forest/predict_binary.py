import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# --- Replicating necessary components from the training script ---

class ToxicRandomForest:
    """
    A wrapper for the scikit-learn RandomForestClassifier.
    This class definition is needed for joblib to correctly load the model object.
    """
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
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


def smiles_to_fingerprint(smiles_list, radius=2, nBits=2048):
    """
    Convert a list of SMILES strings to Morgan fingerprints.
    This function must be identical to the one used in training.
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = mfpgen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
        else:
            # Handle invalid SMILES by providing a zeroed array of the correct shape
            fingerprints.append(np.zeros(nBits, dtype=int))
    return np.array(fingerprints)


def predict_toxicity(smiles_string, model_path="toxic_random_forest_model.joblib"):
    """
    Loads a pre-trained RandomForest model and predicts the toxicity of a
    given SMILES string.
    """
    try:
        # Load the trained model
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please run the training script first to create and save the model.")
        return

    # Convert the single SMILES string to a fingerprint
    # The function expects a list, so we wrap the string in a list
    fingerprint = smiles_to_fingerprint([smiles_string])

    # Make a prediction
    prediction = model.predict(fingerprint)[0]  # Get the first (and only) prediction
    probabilities = model.predict_proba(fingerprint)[0] # Get probabilities for [inactive, active]

    # Interpret the result
    # In training, {"inactive": 0, "active": 1}
    if prediction == 1:
        result = "Toxic (Active)"
        confidence = probabilities[1]
    else:
        result = "Non-toxic (Inactive)"
        confidence = probabilities[0]
        
    print("\n--- Prediction Result ---")
    print(f"SMILES: {smiles_string}")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # To run this script, you must first run the modified training script
    # to generate the 'toxic_random_forest_model.joblib' file.

    # Example SMILES string for Aspirin
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    predict_toxicity(aspirin_smiles)

    print("\n" + "="*50 + "\n")

    # Interactive loop for user input
    try:
        while True:
            user_input = input("Enter a SMILES string to predict (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
            if user_input:
                predict_toxicity(user_input)
                print("\n" + "="*50 + "\n")
            else:
                print("Please enter a valid SMILES string.")
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"An error occurred: {e}")