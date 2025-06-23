import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

class ANNModel(nn.Module):
    """
    A simple multi-layer perceptron for binary classification.
    This architecture MUST MATCH the one used for training.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate=0.5):
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
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
            fingerprints.append(np.zeros(nBits, dtype=int))
    return np.array(fingerprints)

def predict_toxicity(smiles_string, model_path="./models/toxic_ann_model.pth", nBits=2048):
    """
    Loads a pre-trained PyTorch ANN model and predicts the toxicity of a
    given SMILES string.
    """
    try:
        # --- 1. Load the Model ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate the model with the same architecture as during training
        # input_dim must match the fingerprint size (nBits)
        model = ANNModel(input_dim=nBits, hidden_dim1=512, hidden_dim2=256).to(device)
        
        # Load the saved state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set the model to evaluation mode

    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please run the training script first to create and save the model.")
        return
        
    # --- 2. Preprocess the Input SMILES ---
    # The function expects a list, so we wrap the single string in a list
    fingerprint = smiles_to_fingerprint([smiles_string], nBits=nBits)
    
    # Check for invalid SMILES
    if np.all(fingerprint[0] == 0) and Chem.MolFromSmiles(smiles_string) is None:
        print("Error: Invalid SMILES string provided.")
        return

    fingerprint_tensor = torch.FloatTensor(fingerprint).to(device)

    # --- 3. Perform Inference ---
    with torch.no_grad():
        output = model(fingerprint_tensor)
        # Apply sigmoid to convert logits to probabilities
        probability = torch.sigmoid(output).item()
        # Get binary prediction based on a 0.5 threshold
        prediction = (probability > 0.5)

    # --- 4. Output the Result ---
    # In training, {"inactive": 0, "active": 1}
    if prediction:
        result = "Toxic (Active)"
        confidence = probability
    else:
        result = "Non-toxic (Inactive)"
        confidence = 1 - probability

    print("\n--- Prediction Result ---")
    print(f"SMILES: {smiles_string}")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # To run this script, you must first run the modified training script
    # to generate the 'toxic_ann_model.pth' file.

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