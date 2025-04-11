import pandas as pd

h5_file = "/scratch/zczlyf7/dataset/train_multi_inputs.h5"

# Load entire index to get number of rows (this works for fixed-format)
df = pd.read_hdf(h5_file, start=0, stop=1)  # Load 1 row to get column count
num_rows = pd.read_hdf(h5_file).shape[0]
num_cols = df.shape[1]

print(f"H5 File Shape: ({num_rows}, {num_cols})")
