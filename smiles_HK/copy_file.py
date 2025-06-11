import os


def multiply_data_to_file(input_file, output_file, target_size_gb):
    target_size_bytes = target_size_gb * 1024 * 1024 * 1024  # GB → B

    if not os.path.exists(input_file):
        print(f"Error: input file '{input_file}' not exists.")
        return

    with open(input_file, "rb") as f:
        data = f.read()

    data_size = len(data)
    if data_size == 0:
        print("Error: input file is empty.")
        return

    with open(output_file, "wb") as f_out:
        bytes_written = 0
        f_out.write(
            "SAMPLE_ID\tSAMPLE_DATA_ID\tPROTOCOL_NAME\tSAMPLE_DATA_TYPE\tASSAY_OUTCOME\tCURVE_CLASS2\tAC50\tEFFICACY\tZERO_ACTIVITY\tINF_ACTIVITY\tHILL_COEF\tR2\tP_HILL\tCHANNEL_OUTCOME\tDATA0\tDATA1\tDATA2\tDATA3\tDATA4\tDATA5\tDATA6\tDATA7\tDATA8\tDATA9\tDATA10\tDATA11\tDATA12\tDATA13\tDATA14\tDATA15\tCONC0\tCONC1\tCONC2\tCONC3\tCONC4\tCONC5\tCONC6\tCONC7\tCONC8\tCONC9\tCONC10\tCONC11\tCONC12\tCONC13\tCONC14\tCONC15\tCAS\tPUBCHEM_CID\tPUBCHEM_SID\tPURITY_RATING\tPURITY_RATING_4M\tSAMPLE_NAME\tSMILES\tTOX21_ID\tPURITY\n".encode(
                "utf-8"
            )
        )
        while bytes_written < target_size_bytes:
            f_out.write(data)
            bytes_written += data_size
            if bytes_written % (100 * 1024 * 1024) < data_size:
                print(f"Bytes written {bytes_written / (1024*1024):.2f} MB...")

    print(
        f"Output filr '{output_file}' got ~{bytes_written / (1024*1024*1024):.2f} GB."
    )


# Przykład użycia
multiply_data_to_file(
    input_file="./data/tox21-ap1-agonist-p1.txt",
    output_file="./data/huge_tox21-ap1-agonist-p1.txt",
    target_size_gb=1
)
# SAMPLE_ID	SAMPLE_DATA_ID	PROTOCOL_NAME	SAMPLE_DATA_TYPE	ASSAY_OUTCOME	CURVE_CLASS2	AC50	EFFICACY	ZERO_ACTIVITY	INF_ACTIVITY	HILL_COEF	R2	P_HILL	CHANNEL_OUTCOME	DATA0	DATA1	DATA2	DATA3	DATA4	DATA5	DATA6	DATA7	DATA8	DATA9	DATA10	DATA11	DATA12	DATA13	DATA14	DATA15	CONC0	CONC1	CONC2	CONC3	CONC4	CONC5	CONC6	CONC7	CONC8	CONC9	CONC10	CONC11	CONC12	CONC13	CONC14	CONC15	CAS	PUBCHEM_CID	PUBCHEM_SID	PURITY_RATING	PURITY_RATING_4M	SAMPLE_NAME	SMILES	TOX21_ID	PURITY
