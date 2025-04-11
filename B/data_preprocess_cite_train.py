import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, normalize
from tqdm import tqdm
import joblib

#  CONFIG 
INPUT_PATH = '/scratch/zczlyf7/new_dataset/train_cite_inputs.npz'
TARGET_PATH = '/scratch/zczlyf7/new_dataset/train_cite_targets.npz'
IDXCOL_PATH = '/scratch/zczlyf7/new_dataset/train_cite_inputs_idxcol.npz'
METADATA_PATH = '/scratch/zczlyf7/dataset/metadata.csv'
SAVE_DIR = '/scratch/zczlyf7/final_dataset'
ARTIFACTS_DIR = os.path.join(SAVE_DIR, "artifacts_cite")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# PROCESS FUNCTIONS 
def preprocess_cite_matrix(csr_mat, n_components=128, save_prefix=None):
    """
    Preprocess a CITE-seq matrix (RNA or protein) by:
      1) Row-wise L2 normalization,
      2) Column-wise median subtraction,
      3) TruncatedSVD dimension reduction.
    Optionally saves the SVD model and column medians.
    """
    print("  Raw shape:", csr_mat.shape)

    # Step 1: Row-wise L2 normalization
    csr_mat = normalize(csr_mat, norm='l2', axis=1)
    print("  After row-wise L2 normalization:", csr_mat.shape)

    # Step 2: Column-wise median subtraction
    dense_mat = csr_mat.toarray()
    col_medians = np.median(dense_mat, axis=0)
    dense_mat -= col_medians
    print("  After column-wise median subtraction:", dense_mat.shape)

    # Step 3: TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(dense_mat)
    print(f"  After tSVD (n_components={n_components}): {reduced.shape}\n")

    # Save artifacts 
    if save_prefix:
        np.save(os.path.join(ARTIFACTS_DIR, f"{save_prefix}_col_medians.npy"), col_medians)
        joblib.dump(svd, os.path.join(ARTIFACTS_DIR, f"{save_prefix}_svd.pkl"))

    return reduced


def preprocess_cite_matrix_target(csr_mat, n_components=128, save_prefix=None):
    """
    Preprocess a CITE-seq target  by:
      1) Row-wise L2 normalization,
      2) Column-wise median subtraction,
      3) TruncatedSVD dimension reduction.
    Optionally saves the SVD model and column medians.
    """
    print("  Raw shape:", csr_mat.shape)

    # Step 1: Row-wise L2 normalization
    csr_mat = normalize(csr_mat, norm='l2', axis=1)
    print("  After row-wise L2 normalization:", csr_mat.shape)

    # Step 2: Column-wise median subtraction
    dense_mat = csr_mat.toarray()
    col_medians = np.median(dense_mat, axis=0)
    dense_mat -= col_medians
    print("  After column-wise median subtraction:", dense_mat.shape)

    # Step 3: TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(dense_mat)
    print(f"  After tSVD (n_components={n_components}): {reduced.shape}\n")

    # Save artifacts
    if save_prefix:
        np.save(os.path.join(ARTIFACTS_DIR, f"{save_prefix}_col_medians.npy"), col_medians)
        joblib.dump(svd, os.path.join(ARTIFACTS_DIR, f"{save_prefix}_svd.pkl"))

    return reduced


# LOAD DATA 
print("Loading sparse CITE-seq input (RNA)...")
train_inputs_csr = sp.load_npz(INPUT_PATH)

print("Loading CITE-seq targets (protein)...")
train_targets_csr = sp.load_npz(TARGET_PATH)

print("Loading cell_ids...")
idx_data = np.load(IDXCOL_PATH, allow_pickle=True)
cell_ids = idx_data['index']

print("Loading metadata...")
metadata = pd.read_csv(METADATA_PATH).set_index("cell_id")
metadata_aligned = metadata.loc[cell_ids]
print("Metadata aligned shape:", metadata_aligned.shape)

# INPUT PREPROCESSING 
print("\n Preprocessing CITE RNA input...")
train_inputs_reduced = preprocess_cite_matrix(train_inputs_csr, n_components=128, save_prefix="input")

#  ONE-HOT ENCODE 'day' 
print("One-hot encoding 'day' for CITE...")
day_encoder = OneHotEncoder(
    categories=[[2, 3, 4, 7]],  # Include private test day 7
    sparse_output=False,
    dtype=np.float32,
    handle_unknown="ignore"
)
day_encoded = day_encoder.fit_transform(metadata_aligned[['day']])
print("  One-hot encoded day shape:", day_encoded.shape)

# Save day encoder
joblib.dump(day_encoder, os.path.join(ARTIFACTS_DIR, "day_encoder.pkl"))

# Final input
train_inputs_final = np.concatenate([train_inputs_reduced, day_encoded], axis=1)
print("Final CITE input shape:", train_inputs_final.shape)
np.save(os.path.join(SAVE_DIR, "train_cite_inputs.npy"), train_inputs_final)
print(" Saved train_cite_inputs.npy")

# TARGET PREPROCESSING 
print("\n Preprocessing CITE protein targets...")

#  if you want to save the raw target data as well, use the code below
# train_targets_final = preprocess_cite_matrix_target(train_targets_csr, n_components=128, save_prefix="target")
# print("Final CITE target shape:", train_targets_final.shape)

train_targets_final = train_targets_csr.toarray()  # Convert sparse to dense
print("Final CITE target shape (raw):", train_targets_final.shape)
np.save(os.path.join(SAVE_DIR, "train_cite_targets.npy"), train_targets_final)
print(" Saved train_cite_targets.npy")

