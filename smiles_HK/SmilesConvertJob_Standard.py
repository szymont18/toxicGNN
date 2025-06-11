# SmilesConvertJob_Standard.py
import pandas as pd
from SmilesConverter import SmileConverter
import time


def convert_to_smiles(df):
    n = len(df)
    converter = SmileConverter()

    df["SMILES"] = df["SMILES"].str.upper()

    start = time.time()
    df["SMILES_DATA"] = df["SMILES"].apply(converter.smile_to_data)
    df = df[df["SMILES_DATA"].notnull()]
    end = time.time()

    print(f"[STANDARD] Parsed SMILES: {len(df)} / {n}")
    print(f"[STANDARD] Time taken: {end - start:.2f} seconds")

    return df


def main():
    file_path = './data/tox21-ap1-agonist-p1.aggregrated.txt'

    df = pd.read_csv(file_path, sep='\t', dtype={"PUBCHEM_CID": "float64",
                                                 "CAS": "int64",
                                                 "CURVE_RANK": "float64",
                                                 "PURITY_RATING": "float64"})

    convert_to_smiles(df)


if __name__ == "__main__":
    main()
