import dask
import dask.dataframe as dd
from dask.distributed import Client, progress, get_worker
from dask.diagnostics import ProgressBar
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import pysmiles
import pandas as pd
from typing import Optional, List
import os
import time  # <-- Import modułu time
import glob

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

        if not edge_index:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)


# ==============================================================================
# 2. FUNKCJA PRZETWARZAJĄCA PARTYCJE (bez zmian)
# ==============================================================================


# Ta funkcja zawiera OBIE poprawki: dla błędu 'ValueError' i 'AttributeError'
def process_partition(df: pd.DataFrame, label_column: str):
    """
    Przetwarza partycję danych (Pandas DataFrame), konwertuje SMILES na grafy
    i zwraca listę grafów oraz tensor etykiet.
    """
    converter = SmileConverter.smile_to_data
    graphs = []
    labels = []

    for _, row in df.iterrows():
        smile_string = row["SMILES"]
        label_text = row[label_column]

        # Sprawdzamy, czy SMILES nie jest pusty/błędny
        if not isinstance(smile_string, str):
            continue

        graph_data = converter(smile_string)

        if graph_data is not None:
            # Mapowanie etykiet tekstowych na numeryczne
            if isinstance(label_text, str) and "active" in label_text:
                label_numeric = 1
            else:
                label_numeric = 0

            graphs.append(graph_data)
            labels.append(label_numeric)

    if not graphs:
        # Zwracamy puste listy/tensory, jeśli nie ma grafów w tej partycji
        return [], torch.empty(0, dtype=torch.long)

    return graphs, torch.tensor(labels, dtype=torch.long)


# ==============================================================================
# 3. GŁÓWNY SKRYPT ORKIESTRUJĄCY PRACĘ (z dodanym pomiarem czasu)
# ==============================================================================


def main():
    """
    Główna funkcja, która wczytuje dane, przetwarza je równolegle z Dask
    i zapisuje wyniki do plików, mierząc czas poszczególnych etapów.
    """
    # POMIAR CZASU: Start całkowitego czasu
    total_script_start_time = time.time()

    # --- Konfiguracja ---
    INPUT_FILE = "./data/tox21-ap1-agonist-p1.aggregrated.txt"
    OUTPUT_DIR = "./processed_data"
    LABEL_COLUMN = "ASSAY_OUTCOME"
    NUM_PARTITIONS = 16

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Uruchamianie klienta Dask...")
    client = Client()
    print(f"Panel Dask dostępny pod adresem: {client.dashboard_link}")
    print("-" * 40)

    # --- Wczytywanie danych ---
    print(f"Wczytywanie danych z {INPUT_FILE}...")
    # POMIAR CZASU: Start wczytywania i dystrybucji
    loading_start_time = time.time()

    ddf = dd.read_csv(
        INPUT_FILE,
        index_col=False,
        sep="\t",
        dtype={
            "FLAG": "object",
            "PUBCHEM_CID": "float64",
            "PURITY_RATING_4M": "object",
        },
    )

    ddf = ddf.repartition(npartitions=NUM_PARTITIONS).persist()

    # POMIAR CZASU: Koniec wczytywania i dystrybucji
    loading_end_time = time.time()
    print(f"Dane wczytane i rozproszone w pamięci klastra.")
    print(
        f"-> Czas wczytywania i dystrybucji: {loading_end_time - loading_start_time:.2f} s"
    )
    print("-" * 40)

    total_records = len(ddf)
    print(f"Całkowita liczba rekordów do przetworzenia: {total_records}")

    # --- Przetwarzanie równoległe ---
    print("Rozpoczynanie równoległego przetwarzania SMILES...")
    # POMIAR CZASU: Start głównej fazy obliczeniowej
    processing_start_time = time.time()

    processed_files = ddf.map_partitions(
        process_partition,
        label_column=LABEL_COLUMN,
        meta=pd.Series(dtype="str"),
    )

    #This turns a lazy Dask collection into a Dask collection with the same metadata,
    #but now with the results fully computed or actively computing in the background.
    processed_files = processed_files[processed_files != ""].persist()

    with ProgressBar():
        final_file_list = processed_files.compute()

    # POMIAR CZASU: Koniec głównej fazy obliczeniowej
    processing_end_time = time.time()

    # --- Podsumowanie ---
    print("\n" + "=" * 40)
    print("--- ZAKOŃCZONO PRZETWARZANIE ---")
    print("=" * 40)

    # POMIAR CZASU: Wyświetlenie wyników
    print("\n--- PODSUMOWANIE CZASÓW ---")
    print(
        f"Czas wczytywania i dystrybucji danych: {loading_end_time - loading_start_time:.2f} s"
    )
    print(
        f"Czas przetwarzania równoległego (obliczenia + zapis): {processing_end_time - processing_start_time:.2f} s"
    )
    total_script_end_time = time.time()
    print(
        f"CAŁKOWITY CZAS WYKONANIA SKRYPTU: {total_script_end_time - total_script_start_time:.2f} s"
    )
    print("=" * 40)

    client.close()


# ==============================================================================
# 4. KLASA DATASET I FUNKCJA DO WCZYTYWANIA WYNIKÓW (z dodanym pomiarem czasu)
# ==============================================================================


class SmilesDataset(Dataset):
    def __init__(self, graphs: List[Data], labels: torch.Tensor):
        self.graphs = graphs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[Data, torch.Tensor]:
        return self.graphs[idx], self.labels[idx]


def load_processed_data(processed_dir: str) -> SmilesDataset:
    # POMIAR CZASU: Start wczytywania gotowych plików
    loading_start_time = time.time()

    print(f"\nWczytywanie przetworzonych danych z {processed_dir}...")
    all_graphs = []
    all_labels = []

    file_paths = glob.glob(os.path.join(processed_dir, "*.pt"))

    for path in file_paths:
        data = torch.load(path)
        all_graphs.extend(data["graphs"])
        all_labels.append(data["labels"])

    if not all_labels:
        raise ValueError("Nie znaleziono żadnych przetworzonych danych.")

    final_labels = torch.cat(all_labels)

    # POMIAR CZASU: Koniec wczytywania gotowych plików
    loading_end_time = time.time()
    print(f"Wczytano pomyślnie {len(all_graphs)} grafów i {len(final_labels)} etykiet.")
    print(
        f"-> Czas wczytywania przetworzonych plików: {loading_end_time - loading_start_time:.2f} s"
    )

    return SmilesDataset(graphs=all_graphs, labels=final_labels)


if __name__ == "__main__":
    # Uruchomienie głównego procesu przetwarzania
    main()

    # Przykład, jak następnie wczytać dane do trenowania
    print("\n--- Przykład wczytania danych do trenowania (z pomiarem czasu) ---")
    try:
        final_dataset = load_processed_data("./processed_data")
        print(f"Utworzono dataset. Pierwszy element: {final_dataset[0]}")
    except ValueError as e:
        print(e)
