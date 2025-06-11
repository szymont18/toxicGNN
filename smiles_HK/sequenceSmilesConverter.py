import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import pysmiles
from typing import List, Optional
import os
import time
from tqdm import tqdm  # Biblioteka do wizualizacji paska postępu

# ==============================================================================
# 1. KLASA KONWERTERA (bez zmian)
# ==============================================================================

class SmileConverter:
    """
    Konwertuje pojedynczy ciąg SMILES na obiekt grafu PyTorch Geometric.
    """
    @staticmethod
    def smile_to_data(smile: str) -> Optional[Data]:
        """
        Próbuje parsować SMILES i tworzy z niego graf.
        """
        try:
            # Używamy pysmiles do sparsowania ciągu SMILES
            graph = pysmiles.read_smiles(smile, explicit_hydrogen=False)
        except Exception:
            # W razie błędu w parsowaniu SMILES, zwracamy None
            return None

        # Proste cechy atomów: ich indeksy
        atom_features = [[float(i)] for i in range(len(graph.nodes))]
        x = torch.tensor(atom_features, dtype=torch.float)

        # Mapowanie węzłów na indeksy
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}

        # Tworzenie listy krawędzi w formacie PyTorch Geometric
        edge_index = []
        for src, dst, _ in graph.edges(data=True):
            edge_index.append([node_mapping[src], node_mapping[dst]])
            # Grafy są nieskierowane, więc dodajemy krawędź w obie strony
            edge_index.append([node_mapping[dst], node_mapping[src]])

        if not edge_index:
            # Jeśli nie ma krawędzi, tworzymy pusty tensor
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

# ==============================================================================
# 2. GŁÓWNY SKRYPT PRZETWARZAJĄCY (WERSJA SEKWENCYJNA)
# ==============================================================================

def main_sequential():
    """
    Główna funkcja, która wczytuje dane, przetwarza je sekwencyjnie
    i zapisuje wynik do jednego pliku, mierząc czas wykonania.
    """
    # POMIAR CZASU: Start całkowitego czasu
    total_script_start_time = time.time()

    # --- Konfiguracja ---
    INPUT_FILE = "./data/tox21-ap1-agonist-p1.txt"
    OUTPUT_DIR = "./processed_data_sequential"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_data.pt")
    LABEL_COLUMN = "ASSAY_OUTCOME"
    SMILES_COLUMN = "SMILES"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Wczytywanie danych (Pandas) ---
    print(f"Wczytywanie danych z {INPUT_FILE} za pomocą Pandas...")
    loading_start_time = time.time()
    
    # Używamy Pandas do wczytania całego pliku do jednej ramki danych
    try:
        df = pd.read_csv(INPUT_FILE, sep='\t', dtype={'FLAG': 'object',
                                                     'PUBCHEM_CID': 'float64',
                                                     'PURITY_RATING_4M': 'object'})
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku wejściowego: {INPUT_FILE}")
        return

    loading_end_time = time.time()
    print(f"Dane wczytane do pamięci.")
    print(f"-> Czas wczytywania: {loading_end_time - loading_start_time:.2f} s")
    print("-" * 40)

    total_records = len(df)
    print(f"Całkowita liczba rekordów do przetworzenia: {total_records}")

    # --- Przetwarzanie sekwencyjne ---
    print("Rozpoczynanie sekwencyjnego przetwarzania SMILES...")
    processing_start_time = time.time()

    all_graphs = []
    all_labels = []
    
    # Pętla przez wszystkie wiersze ramki danych z paskiem postępu tqdm
    for _, row in tqdm(df.iterrows(), total=total_records, desc="Przetwarzanie"):
        smile_string = row[SMILES_COLUMN]
        label_text = row[LABEL_COLUMN]
        
        # Sprawdzamy, czy SMILES nie jest pusty
        if not isinstance(smile_string, str):
            continue
            
        graph_data = SmileConverter.smile_to_data(smile_string)

        if graph_data is not None:
            # Mapowanie etykiet tekstowych na wartości binarne (0 lub 1)
            if isinstance(label_text, str) and 'active' in label_text:
                label_numeric = 1
            else:
                label_numeric = 0
            
            all_graphs.append(graph_data)
            all_labels.append(label_numeric)

    processing_end_time = time.time()
    
    # --- Zapisywanie wyników ---
    print(f"\nZapisywanie {len(all_graphs)} grafów do pliku {OUTPUT_FILE}...")
    saving_start_time = time.time()
    torch.save({'graphs': all_graphs, 'labels': torch.tensor(all_labels, dtype=torch.long)}, OUTPUT_FILE)
    saving_end_time = time.time()
    
    # --- Podsumowanie ---
    print("\n" + "=" * 40)
    print("--- ZAKOŃCZONO PRZETWARZANIE SEKWENCYJNE ---")
    print("=" * 40)
    
    # POMIAR CZASU: Wyświetlenie wyników
    print("\n--- PODSUMOWANIE CZASÓW ---")
    print(f"Czas wczytywania danych: {loading_end_time - loading_start_time:.2f} s")
    print(f"Czas przetwarzania (konwersja SMILES): {processing_end_time - processing_start_time:.2f} s")
    print(f"Czas zapisu do pliku: {saving_end_time - saving_start_time:.2f} s")
    total_script_end_time = time.time()
    print("-" * 20)
    print(f"CAŁKOWITY CZAS WYKONANIA SKRYPTU: {total_script_end_time - total_script_start_time:.2f} s")
    print("=" * 40)

# ==============================================================================
# 3. KLASA DATASET I FUNKCJA DO WCZYTYWANIA WYNIKÓW
# ==============================================================================

class SmilesDataset(Dataset):
    def __init__(self, graphs: List[Data], labels: torch.Tensor):
        self.graphs = graphs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[Data, torch.Tensor]:
        return self.graphs[idx], self.labels[idx]

def load_processed_data(processed_file: str) -> Optional[SmilesDataset]:
    # POMIAR CZASU: Start wczytywania gotowego pliku
    loading_start_time = time.time()

    print(f"\nWczytywanie przetworzonych danych z {processed_file}...")
    
    if not os.path.exists(processed_file):
        print(f"BŁĄD: Plik {processed_file} nie istnieje.")
        return None
        
    data = torch.load(processed_file)
    graphs = data["graphs"]
    labels = data["labels"]

    # POMIAR CZASU: Koniec wczytywania gotowego pliku
    loading_end_time = time.time()
    print(f"Wczytano pomyślnie {len(graphs)} grafów i {len(labels)} etykiet.")
    print(f"-> Czas wczytywania przetworzonego pliku: {loading_end_time - loading_start_time:.2f} s")

    return SmilesDataset(graphs=graphs, labels=labels)


if __name__ == "__main__":
    # Uruchomienie głównego procesu przetwarzania
    main_sequential()

    # Przykład, jak następnie wczytać dane do trenowania
    print("\n--- Przykład wczytania danych do trenowania (z pomiarem czasu) ---")
    
    final_dataset = load_processed_data("./processed_data_sequential/processed_data.pt")
    if final_dataset:
        print(f"Utworzono dataset. Liczba elementów: {len(final_dataset)}")
        print(f"Pierwszy element: {final_dataset[0]}")