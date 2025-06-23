# SmilesConvertJob_Dask.py
import dask.dataframe as dd
from SmilesConverter import SmileConverter
from dask.distributed import Client
import time


def convert_to_smiles(df):
    n = len(df)
    converter = SmileConverter()

    df["SMILES"] = df["SMILES"].str.upper()
    df["SMILES_DATA"] = df["SMILES"].map(
        converter.smile_to_data, meta=("SMILES_DATA", object)
    )
    df = df[df["SMILES_DATA"].notnull()]

    start = time.time()
    result = df.compute()
    end = time.time()

    print(f"[DASK] Parsed SMILES: {len(result)} / {n}")
    print(f"[DASK] Time taken: {end - start:.2f} seconds")

    return result


def main():
    client = Client()

    file_path = "./data/tox21-ap1-agonist-p1.aggregrated.txt"

    df = dd.read_csv(file_path, sep="\t", dtype={"PUBCHEM_CID": "float64",
                                                 "CURVE_RANK": "float64",
                                                 "PURITY_RATING": "float64"})
    df = df.repartition(npartitions=8)

    convert_to_smiles(df)

    client.close()



if __name__ == "__main__":
    main()
