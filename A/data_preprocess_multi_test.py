import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import joblib
from tqdm import tqdm

# CONFIG 
TEST_INPUT_PATH = "/scratch/zczlyf7/new_dataset/test_multi_inputs.npz"
IDXCOL_PATH = "/scratch/zczlyf7/new_dataset/test_multi_inputs_idxcol.npz"
METADATA_PATH = "/scratch/zczlyf7/dataset/metadata.csv"

# Directory  to save processed test input 
SAVE_DIR = "/scratch/zczlyf7/final_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# Directory containing preprocessing artifacts from training
ARTIFACT_DIR = "/scratch/zczlyf7/final_dataset/artifacts_multi"
TOPK_INDICES_PATH = os.path.join(ARTIFACT_DIR, "topk_indices.npy")
SVD_MODEL_PATH = os.path.join(ARTIFACT_DIR, "svd.pkl")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "day_encoder.pkl")

# LOAD ARTIFACTS FROM TRAINING 
print("Loading preprocessing artifacts...")
topk_indices = np.load(TOPK_INDICES_PATH)
svd = joblib.load(SVD_MODEL_PATH)
day_encoder = joblib.load(ENCODER_PATH)

#  LOAD TEST DATA 
print("Loading sparse test input...")
test_inputs_csr = sp.load_npz(TEST_INPUT_PATH)

print("Loading cell_ids from idxcol...")
idx_data = np.load(IDXCOL_PATH, allow_pickle=True)
cell_ids = idx_data['index']

print("Loading metadata...")
metadata = pd.read_csv(METADATA_PATH).set_index("cell_id")
metadata_aligned = metadata.loc[cell_ids]
print("Metadata aligned shape:", metadata_aligned.shape)

#  INPUT PREPROCESSING 
print("Preprocessing test ATAC input...")

# Step 1: Feature selection (use saved indices from training)
test_inputs_csr = test_inputs_csr[:, topk_indices]
print("After feature selection:", test_inputs_csr.shape)

# Step 2: Binarize nonzero
test_inputs_csr = test_inputs_csr.copy()
test_inputs_csr.data[:] = 1.0

# Step 3: Apply saved tSVD
print("Applying tSVD on test input...")
test_inputs_svd = svd.transform(test_inputs_csr)
print("Input SVD shape:", test_inputs_svd.shape)

# Step 4: One-hot encode 'day' using saved encoder
print("One-hot encoding 'day'...")
day_encoded = day_encoder.transform(metadata_aligned[['day']])
print("One-hot encoded day shape:", day_encoded.shape)

# Step 5: Concatenate SVD + One-hot
test_inputs_final = np.concatenate([test_inputs_svd, day_encoded], axis=1)
print("Final test input shape:", test_inputs_final.shape)

#  SAVE 
np.save(os.path.join(SAVE_DIR, "test_inputs.npy"), test_inputs_final)
print(" Saved processed test input to test_inputs.npy")
