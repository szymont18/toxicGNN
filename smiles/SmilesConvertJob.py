import dask.dataframe as dd
from smiles.SmilesConverter import SmileConverter
from dask.distributed import Client
import time

def main():
    client = Client()

    file_path = '../data/tox21-ache-p3.txt'

    df = dd.read_csv(file_path, index_col=False, sep='\t', dtype={'PUBCHEM_CID': 'float64'})
    n = len(df)

    df = df.repartition(npartitions=8)

    converter = SmileConverter()

    df["SMILES"] = df["SMILES"].str.upper()
    df["SMILES_DATA"] = df["SMILES"].map(converter.smile_to_data, meta=("SMILES_DATA", object))
    df = df[df["SMILES_DATA"].notnull()]

    start = time.time()
    result = df.compute()
    end = time.time()

    print(f"Parsed Smiles: {len(result)} / {n}")
    print(f"Time: {end - start}")

    client.close()


if __name__ == "__main__":
    main()
