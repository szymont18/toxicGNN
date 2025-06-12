import os


def multiply_data_to_file(input_file, output_file, target_size_gb):
    target_size_bytes = target_size_gb * 1024 * 1024 * 1024  # GB → B

    if not os.path.exists(input_file):
        print(f"Error: input file '{input_file}' not exists.")
        return

    output_file = os.path.expandvars(output_file)

    with open(input_file, "rb") as f:
        # Skip header
        header = f.readline()
        data = f.read()

    data_size = len(data)
    if data_size == 0:
        print("Error: input file is empty.")
        return

    with open(output_file, "wb") as f_out:
        bytes_written = 0
        f_out.write(header)
        while bytes_written < target_size_bytes:
            f_out.write(data)
            bytes_written += data_size
            if bytes_written % (100 * 1024 * 1024) < data_size:
                print(f"Bytes written {bytes_written / (1024*1024):.2f} MB...")

    print(
        f"Output file '{output_file}' got ~{bytes_written / (1024*1024*1024):.2f} GB."
    )


# Example usage
multiply_data_to_file(
    input_file="./data/tox21-ap1-agonist-p1.txt",
    output_file="$SCRATCH/huge_tox21-ap1-agonist-p16.txt",
    target_size_gb=1
)