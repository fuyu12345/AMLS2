import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import joblib

# CONFIG 
TEST_INPUT_PATH = "/scratch/zczlyf7/new_dataset/test_cite_inputs.npz"
IDXCOL_PATH = "/scratch/zczlyf7/new_dataset/test_cite_inputs_idxcol.npz"
METADATA_PATH = "/scratch/zczlyf7/dataset/metadata.csv"

SAVE_DIR = "/scratch/zczlyf7/final_dataset"
ARTIFACT_DIR = os.path.join(SAVE_DIR, "artifacts_cite")
os.makedirs(SAVE_DIR, exist_ok=True)

#  LOAD ARTIFACTS 
print(" Loading preprocessing artifacts...")
svd = joblib.load(os.path.join(ARTIFACT_DIR, "input_svd.pkl"))
col_medians = np.load(os.path.join(ARTIFACT_DIR, "input_col_medians.npy"))
day_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "day_encoder.pkl"))

#  LOAD TEST DATA 
print(" Loading CITE test input...")
test_inputs_csr = sp.load_npz(TEST_INPUT_PATH)

print(" Loading cell IDs from idxcol...")
idx_data = np.load(IDXCOL_PATH, allow_pickle=True)
cell_ids = idx_data['index']

print(" Loading metadata...")
metadata = pd.read_csv(METADATA_PATH).set_index("cell_id")
metadata_aligned = metadata.loc[cell_ids]
print(" Metadata aligned shape:", metadata_aligned.shape)

#  PREPROCESSING 
print(" Preprocessing test CITE RNA input...")

# Step 1: Row-wise L2 normalization
test_inputs_csr = normalize(test_inputs_csr, norm="l2", axis=1)

# Step 2: Subtract training column medians
test_inputs_dense = test_inputs_csr.toarray()
test_inputs_dense -= col_medians

# Step 3: Apply SVD from training
test_inputs_svd = svd.transform(test_inputs_dense)
print(" Transformed test shape:", test_inputs_svd.shape)

# Step 4: One-hot encode 'day'
print(" One-hot encoding 'day'...")
day_encoded = day_encoder.transform(metadata_aligned[['day']])
print(" One-hot day shape:", day_encoded.shape)

# Step 5: Concatenate
test_inputs_final = np.concatenate([test_inputs_svd, day_encoded], axis=1)
print(" Final CITE test input shape:", test_inputs_final.shape)

#  SAVE 
np.save(os.path.join(SAVE_DIR, "test_cite_inputs.npy"), test_inputs_final)
print(" Saved preprocessed test input to test_cite_inputs.npy")
