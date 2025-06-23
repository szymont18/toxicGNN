import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

class ANNModel(nn.Module):
    """
    This architecture MUST MATCH the one used for training.
    """
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

def smiles_to_fingerprint(smiles_list, nBits=2048):
    """
    This function must be identical to the one used in training.
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nBits)
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = mfpgen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
        else:
            fingerprints.append(np.zeros(nBits, dtype=int))
    return np.array(fingerprints)

def predict_toxicity_profile(smiles_string, model_path="./models/ann_multilabel_model.pth"):
    """
    Loads a pre-trained multi-label ANN model and predicts the toxicity
    profile for a given SMILES string.
    """
    try:
        # --- 1. Load Model and Hyperparameters ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        label_columns = checkpoint['label_columns']
        nbits = checkpoint['nbits']
        
        # Re-create the model with the saved architecture
        model = ANNModel(
            input_dim=nbits,
            hidden_dim1=checkpoint['hidden_dim1'],
            hidden_dim2=checkpoint['hidden_dim2'],
            output_dim=len(label_columns),
            dropout_rate=checkpoint['dropout_rate']
        ).to(device)
        
        # Load the saved weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set the model to evaluation mode

    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please run the training script first to create and save the model.")
        return
        
    # --- 2. Preprocess the Input SMILES ---
    fingerprint = smiles_to_fingerprint([smiles_string], nBits=nbits)
    
    if np.all(fingerprint[0] == 0) and Chem.MolFromSmiles(smiles_string) is None:
        print("Error: Invalid SMILES string provided.")
        return

    fingerprint_tensor = torch.FloatTensor(fingerprint).to(device)

    # --- 3. Perform Inference ---
    with torch.no_grad():
        logits = model(fingerprint_tensor)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        predictions = (probabilities > 0.5).astype(int)

    # --- 4. Display Results ---
    print("\n--- Toxicity Profile Prediction ---")
    print(f"SMILES: {smiles_string}")
    
    for i, label in enumerate(label_columns):
        result = "Active" if predictions[i] == 1 else "Inactive"
        prob_active = probabilities[i]
        print(f"- {label:<30}: {prob_active*100:.2f}%")

    print(f"\nThe most toxic assay is: {label_columns[np.argmax(probabilities)]}")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # To run this script, you must first run the modified training script
    # to generate the 'ann_multilabel_model.pth' file.

    # Example SMILES string for Ibuprofen
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