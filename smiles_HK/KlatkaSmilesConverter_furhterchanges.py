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
import time
import glob
from dask_jobqueue.slurm import SLURMCluster, SLURMRunner

class SmileConverter:
    """
    Konwertuje pojedynczy ciąg SMILES na obiekt grafu PyTorch Geometric.
    """
    @staticmethod
    def smile_to_data(smile: str) -> Optional[Data]:
        """
        Próbuje parsować SMILES i tworzy z niego graf. W tej wersji pętle zostały
        zastąpione operacjami wektorowymi dla większej wydajności.
        """
        try:
            graph = pysmiles.read_smiles(smile, explicit_hydrogen=False)
        except Exception:
            return None

        num_nodes = len(graph.nodes())
        x = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)

        if not graph.edges():
            return Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))

        # Pysmiles gwarantuje, że węzły są liczbami całkowitymi, więc możemy użyć
        # mapowania opartego na tensorach, które jest znacznie szybsze niż pętle w Pythonie.
        nodes = list(graph.nodes())
        node_keys = torch.tensor(nodes, dtype=torch.long)
        node_values = torch.arange(num_nodes, dtype=torch.long)

        # Tworzymy tensor mapujący, aby znormalizować indeksy węzłów do zakresu 0..N-1
        # To jest odpowiednik słownika `node_mapping` z poprzedniej wersji.
        mapping_tensor = torch.zeros(node_keys.max() + 1, dtype=torch.long)
        mapping_tensor[node_keys] = node_values

        # Pobieramy krawędzie i mapujemy je za pomocą tensora
        edges_tensor = torch.tensor(list(graph.edges()), dtype=torch.long)
        mapped_edges = mapping_tensor[edges_tensor]

        # Tworzymy krawędzie w obie strony (graf nieskierowany) i transponujemy
        edge_index = torch.cat([mapped_edges, mapped_edges.flip(1)], dim=0).t().contiguous()

        return Data(x=x, edge_index=edge_index)

def process_partition(df: pd.DataFrame, label_column: str, output_dir):
    converter = SmileConverter.smile_to_data

    valid_df = df[df["SMILES"].apply(lambda x: isinstance(x, str))]

    graphs = [converter(smile) for smile in valid_df["SMILES"]]

    filtered = [(g, label) for g, label in zip(graphs, valid_df[label_column]) if g is not None]

    if not filtered:
        return [], []

    graphs, labels_text = zip(*filtered)

    labels = [1 if isinstance(label, str) and "active" in label else 0 for label in labels_text]

    if not graphs:
        return [], torch.empty(0, dtype=torch.long)

    return list(graphs), torch.tensor(labels, dtype=torch.long)


file_name = "./logs5.txt"

def print_to_file(text):
    with open(file_name, "a") as f:
        f.write(text + "\n")

def main():
    """
    Główna funkcja, która wczytuje dane, przetwarza je równolegle z Dask
    i zapisuje wyniki do plików, mierząc czas poszczególnych etapów.
    """
    total_script_start_time = time.time()

    # --- Konfiguracja ---
    # INPUT_FILE = "./data/tox21-ap1-agonist-p1.aggregrated.txt"
    INPUT_FILE = os.path.expandvars("$SCRATCH/huge_tox21-ap1-agonist-p16.txt")
    # INPUT_FILE = os.path.expandvars("$SCRATCH/test.txt")
    OUTPUT_DIR = os.path.expandvars("$SCRATCH/processed_data5")
    LABEL_COLUMN = "ASSAY_OUTCOME"
    NUM_PARTITIONS = 16

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Uruchamianie klienta Dask...")
    print_to_file("Uruchamianie klienta Dask...")

    import socket
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print(f"Hostname: {hostname}")
    print(f"IP Address: http://{ip_address}:8787/status")
    print_to_file(f"Hostname: {hostname}")
    print_to_file(f"IP Address: http://{ip_address}:8787/status")

    cluster = SLURMRunner()
    client = Client(cluster)
    # client = Client(n_workers=4, threads_per_worker=2, processes=True, memory_limit='8GB')
    # ZALECENIE: Monitoruj postęp przez dashboard
    print(f"Panel Dask dostępny pod adresem: {client.dashboard_link}")
    print_to_file(f"Panel Dask dostępny pod adresem: {client.dashboard_link}")
    print("-" * 40)

    # --- Wczytywanie i dystrybucja danych ---
    print(f"Wczytywanie danych z {INPUT_FILE}...")
    print_to_file(f"Wczytywanie danych z {INPUT_FILE}...")
    loading_start_time = time.time()

    ddf = dd.read_csv(
        INPUT_FILE,
        index_col=False,
        sep="\t",
        dtype={
            "FLAG": "object", "PUBCHEM_CID": "float64", "PURITY_RATING_4M": "object"
        },
        # Lepsze zarządzanie błędami podczas parsowania CSV
        on_bad_lines='skip' 
    )

    # ZALECENIE: Użyj persist() po wczytaniu, aby rozproszyć dane w pamięci klastra.
    # Repartycjonujemy i od razu utrwalamy dane w pamięci workerów Dask.
    # To kluczowa optymalizacja, która przyspiesza wszystkie kolejne kroki.
    ddf = ddf.repartition(npartitions=NUM_PARTITIONS).persist()
    # Czekamy, aż operacja persist() się zakończy, aby dokładnie zmierzyć czas.
    progress(ddf)
    loading_end_time = time.time()
    print_to_file(f"Dane wczytane i rozproszone w pamięci klastra.")
    print(f"Dane wczytane i rozproszone w pamięci klastra.")
    print(f"-> Czas wczytywania i dystrybucji: {loading_end_time - loading_start_time:.2f} s")
    print_to_file(f"-> Czas wczytywania i dystrybucji: {loading_end_time - loading_start_time:.2f} s")
    print("-" * 40)

    total_records = len(ddf)
    print(f"Całkowita liczba rekordów do przetworzenia: {total_records}")
    print_to_file(f"Całkowita liczba rekordów do przetworzenia: {total_records}")
    # --- Przetwarzanie równoległe ---
    print("Rozpoczynanie równoległego przetwarzania SMILES...")
    processing_start_time = time.time()

    # ZALECENIE: Użyj map_partitions na całym Dask DataFrame.
    # Zamiast pętli, stosujemy funkcję `process_partition` do każdej partycji równolegle.
    # To leniwa operacja – Dask buduje graf zadań, ale jeszcze niczego nie liczy.
    processed_files_series = ddf.map_partitions(
        process_partition,
        label_column=LABEL_COLUMN,
        output_dir=OUTPUT_DIR,
        meta=pd.Series(dtype="object"),
    )
    print_to_file(f"Rozpoczynanie równoległego przetwarzania SMILES...")
    # ZALECENIE: Filtruj w ramach Dask DataFrame.
    # Usuwamy puste wyniki (z partycji bez poprawnych grafów) zanim przejdziemy do .compute().
    # To również jest operacja leniwa.
    processed_files_series = processed_files_series[processed_files_series != ""]
    print_to_file(f"Uruchamianie obliczeń na klastrze Dask...")
    print("Uruchamianie obliczeń na klastrze Dask...")
    # ZALECENIE: Wywołaj compute() raz na końcu i monitoruj postęp.
    # To jest jedyny moment, w którym uruchamiamy rzeczywiste obliczenia.
    # Dask wykonuje cały graf zadań: wczytanie -> konwersja -> zapis -> filtrowanie.
    with ProgressBar():
        final_file_list = processed_files_series.compute()
    print_to_file(f"Uruchamianie obliczeń na klastrze Dask...")
    processing_end_time = time.time()
    print(f"\nZakończono. Przetworzono i zapisano {len(final_file_list)} plików z partycjami.")
    print_to_file(f"\nZakończono. Przetworzono i zapisano {len(final_file_list)} plików z partycjami.")
    # --- Podsumowanie ---
    print("\n" + "=" * 40)
    print("--- ZAKOŃCZONO PRZETWARZANIE ---")
    print("=" * 40)
    print("\n--- PODSUMOWANIE CZASÓW ---")
    print(f"Czas wczytywania i dystrybucji danych: {loading_end_time - loading_start_time:.2f} s")
    print(f"Czas przetwarzania równoległego (obliczenia + zapis): {processing_end_time - processing_start_time:.2f} s")
    total_script_end_time = time.time()
    print(f"CAŁKOWITY CZAS WYKONANIA SKRYPTU: {total_script_end_time - total_script_start_time:.2f} s")
    print("=" * 40)
    print_to_file("--- ZAKOŃCZONO PRZETWARZANIE ---")
    print_to_file("=" * 40)
    print_to_file("\n--- PODSUMOWANIE CZASÓW ---")
    print_to_file(f"Czas wczytywania i dystrybucji danych: {loading_end_time - loading_start_time:.2f} s")
    print_to_file(f"Czas przetwarzania równoległego (obliczenia + zapis): {processing_end_time - processing_start_time:.2f} s")
    total_script_end_time = time.time()
    print_to_file(f"CAŁKOWITY CZAS WYKONANIA SKRYPTU: {total_script_end_time - total_script_start_time:.2f} s")
    print_to_file("=" * 40)
    client.close()

class SmilesDataset(Dataset):
    def __init__(self, graphs: List[Data], labels: torch.Tensor):
        self.graphs = graphs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[Data, torch.Tensor]:
        return self.graphs[idx], self.labels[idx]


def load_processed_data(processed_dir: str) -> SmilesDataset:
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
    loading_end_time = time.time()
    
    print(f"Wczytano pomyślnie {len(all_graphs)} grafów i {len(final_labels)} etykiet.")
    print(f"-> Czas wczytywania przetworzonych plików: {loading_end_time - loading_start_time:.2f} s")

    return SmilesDataset(graphs=all_graphs, labels=final_labels)


if __name__ == "__main__":
    # Uruchomienie głównego procesu przetwarzania
    main()

    # Przykład, jak następnie wczytać dane do trenowania
    print("\n--- Przykład wczytania danych do trenowania ---")
    try:
        final_dataset = load_processed_data(OUTPUT_DIR)
        print(f"Utworzono dataset. Pierwszy element: {final_dataset[0]}")
    except ValueError as e:
        print(e)
    except NameError:
        # Obsługa błędu, jeśli OUTPUT_DIR nie jest zdefiniowany globalnie
        print("Nie można załadować danych, ponieważ zmienna OUTPUT_DIR nie jest dostępna.")
        print("Uruchomienie `load_processed_data('./processed_data')` powinno zadziałać.")
