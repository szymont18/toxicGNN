# all credits to Gemini 2.5 Pro

import torch
from torch_geometric.data import Data
from rdkit import Chem
from typing import Optional


class SmileConverter:
    """
    Handles the conversion of SMILES strings into graph data structures
    suitable for PyTorch Geometric, using RDKit for feature extraction.
    """
    def __init__(self):
        pass

    @staticmethod
    def smile_to_data(smile: str, label: Optional[float] = None) -> Optional[Data]:
        """Converts a SMILES string to a PyTorch Geometric Data object with RDKit features."""
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            # If RDKit cannot parse the SMILES string, return None
            return None

        # 1. Node Features (Atom features)
        # Extract a feature vector for each atom in the molecule.
        node_features = []
        for atom in mol.GetAtoms():
            num_implicit_hs = atom.GetNumImplicitHs()
            explicit_valence = atom.GetTotalValence() - num_implicit_hs
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                atom.GetIsAromatic(),
                atom.GetNumRadicalElectrons(),
                num_implicit_hs,
                explicit_valence,
                atom.IsInRing(),
                atom.GetTotalNumHs(),
            ]
            node_features.append(features)
        x = torch.tensor(node_features, dtype=torch.float)

        # 2. Edge Index and Edge Features
        # Represent bonds as edges in the graph.
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]]) # Undirected graph

            bond_type = bond.GetBondTypeAsDouble()
            bond_features = [
                bond_type,
                bond.IsInRing(),
                bond.GetIsAromatic(),
                int(bond.GetStereo())
            ]
            edge_features.extend([bond_features, bond_features])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # 3. Create Data object with optional label
        if label is not None:
            # Standardize labels to numeric format (0.0 or 1.0)
            if isinstance(label, str):
                if label.lower() in ['active', '1', 'true', 'positive']:
                    label = 1.0
                else:
                    label = 0.0 # Default to inactive
            y = torch.tensor([float(label)], dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        else:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
