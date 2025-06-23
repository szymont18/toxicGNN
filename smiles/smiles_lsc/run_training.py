import os
import time
import pandas as pd
import numpy as np
from typing import Optional, List
from collections import Counter

# --- Dask Imports ---
import dask
import dask.dataframe as dd
from dask.distributed import Client

# --- PyTorch Imports ---
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# --- Scikit-learn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

# --- Inne ---
import pysmiles


# ==============================================================================
# 1. KLASA KONWERTERA SMILES -> GRAF
# (Niezmieniona, kluczowa dla konwersji)
# ==============================================================================
class SmileConverter:
    @staticmethod
    def smile_to_data(smile: str) -> Optional[Data]:
        try:
            graph = pysmiles.read_smiles(smile, explicit_hydrogen=False)
        except Exception:
            return None

        atom_features = [[float(i)] for i in range(len(graph.nodes))]
        x = torch.tensor(atom_features, dtype=torch.float)
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}

        edge_index = []
        for src, dst, _ in graph.edges(data=True):
            edge_index.append([node_mapping[src], node_mapping[dst]])
            edge_index.append([node_mapping[dst], node_mapping[src]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

# ==============================================================================
# 2. MODEL SIECI NEURONOWEJ GNN
# ==============================================================================
class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(SimpleGNN, self).__init__()
        self.layer1 = GCNConv(input_dim, hidden_dim)
        self.layer2 = GCNConv(hidden_dim, output_dim)
        self.lin_layer = torch.nn.Linear(output_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.layer1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layer2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.lin_layer(x)

# ==============================================================================
# 3. DATASET PYTORCH
# ==============================================================================
class SmilesDataset(Dataset):
    def __init__(self, graphs: List[Data], labels: List[int]):
        self.graphs = graphs
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[Data, torch.Tensor]:
        return self.graphs[idx], self.labels[idx]

# ==============================================================================
# 4. FUNKCJE PRZETWARZANIA DANYCH
# ==============================================================================
def process_smiles_partition(df: pd.DataFrame) -> pd.Series:
    """
    Przetwarza partycję Dask: konwertuje SMILES na obiekty grafowe PyG.
    Zamiast zapisywać do pliku, ZWRACA serię z obiektami Data.
    """
    converter = SmileConverter.smile_to_data
    graph_objects = []

    for smile_string in df["SMILES"]:
        if isinstance(smile_string, str):
            graph_data = converter(smile_string)
            graph_objects.append(graph_data)
        else:
            graph_objects.append(None) # Dodaj None, jeśli SMILES jest niepoprawny/pusty
            
    return pd.Series(graph_objects, index=df.index)

def preprocessing():
    """
    Główna funkcja do wczytania i przetworzenia danych.
    Łączy logikę obu skryptów w jeden potok.
    """
    file_path = './data/tox21-ap1-agonist-p1.aggregrated.txt'
    print(f"Rozpoczynanie przetwarzania pliku: {file_path}")

    # Używamy Dask do wczytania i wstępnego przetworzenia danych
    ddf = dd.read_csv(file_path, index_col=False, sep='\t', dtype={'FLAG': 'object',
                                                                  'PUBCHEM_CID': 'float64',
                                                                  'PURITY_RATING_4M': 'object'})
    ddf = ddf.repartition(npartitions=8)

    # Krok 1: Konwersja SMILES na obiekty grafowe za pomocą map_partitions
    print("Konwertowanie SMILES na grafy...")
    # 'meta' informuje Dask o typie danych wyjściowych, co jest kluczowe
    ddf['SMILES_DATA'] = ddf.map_partitions(process_smiles_partition, meta=('SMILES_DATA', 'object'))
    
    # Krok 2: Kodowanie etykiet (bardziej ogólne niż 0/1)
    print("Kodowanie etykiet...")
    # Musimy .compute() unikalne etykiety, aby stworzyć mapowanie
    unique_labels = ddf["ASSAY_OUTCOME"].dropna().unique().compute()
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    num_classes = len(label_to_idx)
    print(f"Znaleziono {num_classes} klasy: {label_to_idx}")
    
    ddf["label_numeric"] = ddf["ASSAY_OUTCOME"].map(label_to_idx, meta=('label_numeric', 'i4'))

    # Krok 3: Czyszczenie danych i pobranie wyników do pamięci
    print("Czyszczenie danych i pobieranie do pamięci...")
    ddf = ddf.dropna(subset=["SMILES_DATA", "label_numeric"])
    
    # .compute() wykonuje wszystkie zaplanowane operacje i zwraca ramkę Pandas
    computed_df = ddf[["SMILES_DATA", "label_numeric"]].compute()
    
    X = computed_df["SMILES_DATA"].tolist()
    y = computed_df["label_numeric"].astype(int).tolist()

    # Krok 4: Podział na zbiór treningowy i testowy
    print("Dzielenie danych na zbiór treningowy i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    
    print(f"Rozkład etykiet w całym zbiorze: {Counter(y)}")
    print(f"Liczba próbek treningowych: {len(X_train)}, testowych: {len(X_test)}")

    train_dataset = SmilesDataset(X_train, y_train)
    test_dataset = SmilesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Krok 5: Obliczenie wag klas dla niezbalansowanego zbioru
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return train_loader, test_loader, num_classes, class_weights

# ==============================================================================
# 5. FUNKCJE TRENINGU I TESTOWANIA
# ==============================================================================
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_examples = 0, 0, 0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_correct += (out.argmax(dim=1) == labels).sum().item()
        total_examples += data.num_graphs
    return total_loss / total_examples, total_correct / total_examples

def test(model, test_loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            out = model(data)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return acc, f1, prec, rec

# ==============================================================================
# 6. GŁÓWNA FUNKCJA ORKIESTRUJĄCA
# ==============================================================================
def main():
    # Uruchomienie klienta Dask (opcjonalne, ale zalecane dla większych plików)
    client = Client()
    print(f"Panel Dask dostępny pod adresem: {client.dashboard_link}")
    
    # Przygotowanie danych
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używane urządzenie: {device}")
    
    train_loader, test_loader, n_classes, class_weights = preprocessing()
    
    print(f"Wagi klas: {class_weights}")

    # Inicjalizacja modelu, optymalizatora i funkcji straty
    model = SimpleGNN(input_dim=1, hidden_dim=64, output_dim=64, num_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Pętla treningowa
    num_epochs = 20
    best_f1 = 0.0
    
    print("\n--- Rozpoczęcie treningu ---")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_acc, test_f1, test_prec, test_rec = test(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch + 1:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"-> Zapisano nowy najlepszy model z F1: {best_f1:.4f}")
            
    print("--- Zakończono trening ---")
    client.close()

if __name__ == '__main__':
    # Ustawienie multiprocessingu dla Dask na 'fork', co bywa pomocne
    dask.config.set(scheduler='processes')
    main()