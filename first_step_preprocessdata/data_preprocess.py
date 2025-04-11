import os
import scipy
import pandas as pd
import numpy as np

# Define input and output directories
input_folder = "/scratch/zczlyf7/dataset"
output_folder = "/scratch/zczlyf7/new_dataset"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


def convert_to_parquet(filename, out_filename):
    """Convert CSV files to Parquet format and save them in the output folder."""
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, out_filename + ".parquet")

    df = pd.read_csv(input_path)
    df.to_parquet(output_path)

    print(f"Saved: {output_path}")


def convert_h5_to_sparse_csr(filename, out_filename, chunksize=2500):
    """Convert large .h5 files to sparse .npz format in chunks and save them in the output folder."""
    input_path = os.path.join(input_folder, filename)
    npz_path = os.path.join(output_folder, out_filename + ".npz")
    idxcol_path = os.path.join(output_folder, out_filename + "_idxcol.npz")

    start = 0
    total_rows = 0  # Track total rows processed
    sparse_chunks_data_list = []
    chunks_index_list = []
    columns_name = None

    while True:
        df_chunk = pd.read_hdf(input_path, start=start, stop=start + chunksize)
        if len(df_chunk) == 0:
            break

        chunk_data_as_sparse = scipy.sparse.csr_matrix(df_chunk.to_numpy())
        sparse_chunks_data_list.append(chunk_data_as_sparse)
        chunks_index_list.append(df_chunk.index.to_numpy())

        if columns_name is None:
            columns_name = df_chunk.columns.to_numpy()
        else:
            assert np.all(columns_name == df_chunk.columns.to_numpy())

        total_rows += len(df_chunk)
        print(f"Processed {total_rows} rows")

        if len(df_chunk) < chunksize:
            del df_chunk
            break
        del df_chunk
        start += chunksize

    all_data_sparse = scipy.sparse.vstack(sparse_chunks_data_list)
    del sparse_chunks_data_list

    all_indices = np.hstack(chunks_index_list)

    scipy.sparse.save_npz(npz_path, all_data_sparse)
    np.savez(idxcol_path, index=all_indices, columns=columns_name)

    print(f"Saved: {npz_path}")
    print(f"Saved: {idxcol_path}")


# Convert CSV files to Parquet
convert_to_parquet("metadata.csv", "metadata")
convert_to_parquet("evaluation_ids.csv", "evaluation")
convert_to_parquet("sample_submission.csv", "sample_submission")

# Convert H5 files to Sparse CSR format
convert_h5_to_sparse_csr("train_multi_inputs.h5", "train_multi_inputs")
convert_h5_to_sparse_csr("train_multi_targets.h5", "train_multi_targets")
convert_h5_to_sparse_csr("train_cite_inputs.h5", "train_cite_inputs")
convert_h5_to_sparse_csr("train_cite_targets.h5", "train_cite_targets")
convert_h5_to_sparse_csr("test_cite_inputs.h5", "test_cite_inputs")
convert_h5_to_sparse_csr("test_multi_inputs.h5", "test_multi_inputs")
