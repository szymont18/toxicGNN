import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

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


def predict_toxicity_profile(smiles_string, model_path="./models/rf_multilabel_model.joblib"):
    """
    Loads a pre-trained multi-label RandomForest model and predicts the toxicity
    profile for a given SMILES string.
    """
    try:
        # Load the dictionary containing the model and label columns
        saved_model_data = joblib.load(model_path)
        model = saved_model_data['model']
        label_columns = saved_model_data['label_columns']
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please run the training script first to create and save the model.")
        return

    # Convert the single SMILES string to a fingerprint.
    # The function expects a list, so we wrap the string in a list.
    fingerprint = smiles_to_fingerprint([smiles_string])
    
    # Check if the SMILES was valid
    if np.all(fingerprint[0] == 0) and Chem.MolFromSmiles(smiles_string) is None:
        print("Error: Invalid SMILES string provided.")
        return

    # Make predictions and get probabilities
    predictions = model.predict(fingerprint)[0]  # Get the first (and only) prediction row
    # predict_proba returns a list of arrays, one per label
    probabilities = model.predict_proba(fingerprint)

    print("\n--- Toxicity Profile Prediction ---")
    print(f"SMILES: {smiles_string}")
    
    max_prob = 0
    most_toxic_assay = ""
    # Iterate through each label to display the result
    for i, assay_name in enumerate(label_columns):
        prob_active = probabilities[i][0, 1]
        if prob_active > max_prob:
            max_prob = prob_active
            most_toxic_assay = assay_name
        print(f"- {assay_name:<30}: {prob_active*100:.2f}%")

    print(f"\nThe most toxic assay is: {most_toxic_assay}")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # To run this script, you must first run the modified training script
    # to generate the 'rf_multilabel_model.joblib' file.

    # Example SMILES string for a common compound, e.g., Ibuprofen
    ibuprofen_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    predict_toxicity_profile(ibuprofen_smiles)

    print("\n" + "="*50 + "\n")

    # Interactive loop for user input
    try:
        while True:
            user_input = input("Enter a SMILES string for toxicity profiling (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
            if user_input:
                predict_toxicity_profile(user_input)
                print("\n" + "="*50 + "\n")
            else:
                print("Please enter a valid SMILES string.")
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"An error occurred: {e}")