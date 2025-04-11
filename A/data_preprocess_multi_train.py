import os
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, normalize
import joblib
from tqdm import tqdm

# CONFIG 
INPUT_PATH = '/scratch/zczlyf7/new_dataset/train_multi_inputs.npz'
TARGET_PATH = '/scratch/zczlyf7/new_dataset/train_multi_targets.npz'
IDXCOL_PATH = '/scratch/zczlyf7/new_dataset/train_multi_inputs_idxcol.npz'
METADATA_PATH = '/scratch/zczlyf7/dataset/metadata.csv'
SAVE_DIR = '/scratch/zczlyf7/final_dataset'
ARTIFACTS_DIR = os.path.join(SAVE_DIR, "artifacts_multi")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

#  FUNCTIONS 
def rowwise_nonzero_median_scaling(csr_mat):
    """
    Perform row-wise median scaling on a sparse matrix.
    """
    indptr = csr_mat.indptr
    data = csr_mat.data
    for row in tqdm(range(csr_mat.shape[0]), desc="Row-wise median scaling"):
        start, end = indptr[row], indptr[row+1]
        if end > start:
            row_data = data[start:end]
            median = np.median(row_data)
            data[start:end] /= (median + 1e-10)
    return csr_mat

def binarize_nonzero(csr_mat):
    """
    Binarize the non-zero entries of a sparse matrix.
    """
    csr_mat = csr_mat.copy()
    csr_mat.data[:] = 1.0
    return csr_mat

def preprocess_targets(csr_mat, n_components=128):
    """
    Preprocess a target matrix (e.g., RNA) by:
      1) Row-wise L2 normalization,
      2) Column-wise median subtraction,
      3) TruncatedSVD dimension reduction.
    """

    print("Raw target shape:", csr_mat.shape)

    # Normalize
    csr_mat = normalize(csr_mat, norm='l2', axis=1)
    print("After row-wise normalization:", csr_mat.shape)

    # Subtract column medians
    dense_mat = csr_mat.toarray()           
    col_medians = np.median(dense_mat, axis=0)
    dense_mat -= col_medians
    print("After column-wise median subtraction:", dense_mat.shape)

    # Apply tSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(dense_mat)
    print("After tSVD:", reduced.shape)

    # Save artifacts
    np.save(os.path.join(ARTIFACTS_DIR, "target_col_medians.npy"), col_medians)
    joblib.dump(svd, os.path.join(ARTIFACTS_DIR, "target_svd.pkl"))

    return reduced


#  LOAD DATA 
print("Loading training inputs and targets...")
train_inputs_csr = sp.load_npz(INPUT_PATH)
train_targets_csr = sp.load_npz(TARGET_PATH)

print("Loading cell IDs from idxcol...")
idx_data = np.load(IDXCOL_PATH, allow_pickle=True)
cell_ids = idx_data['index']

print("Loading metadata...")
metadata = pd.read_csv(METADATA_PATH).set_index("cell_id")
metadata_aligned = metadata.loc[cell_ids]
print("Metadata aligned shape:", metadata_aligned.shape)

#  INPUT PREPROCESSING 
print("Preprocessing ATAC input...")
col_means = np.array(train_inputs_csr.mean(axis=0)).ravel()
col_meansq = np.array(train_inputs_csr.power(2).mean(axis=0)).ravel()
col_variances = col_meansq - col_means**2
topk_indices = np.argsort(col_variances)[-20000:]
train_inputs_csr = train_inputs_csr[:, topk_indices]
print("After feature selection:", train_inputs_csr.shape)

train_inputs_csr = binarize_nonzero(train_inputs_csr)

# Save topk_indices
np.save(os.path.join(ARTIFACTS_DIR, "topk_indices.npy"), topk_indices)

# Fit and apply tSVD
svd = TruncatedSVD(n_components=128, random_state=42)
train_inputs_svd = svd.fit_transform(train_inputs_csr)
print("SVD-reduced input shape:", train_inputs_svd.shape)

# Save SVD model
joblib.dump(svd, os.path.join(ARTIFACTS_DIR, "svd.pkl"))

# Encode day
print("One-hot encoding 'day'...")
day_encoder = OneHotEncoder(
    categories=[[2, 3, 4, 7, 10]],  # include all possible day values
    sparse_output=False,
    dtype=np.float32,
    handle_unknown="ignore"  # ignore unseen values safely
)
day_encoded = day_encoder.fit_transform(metadata_aligned[['day']])
print("One-hot encoded day shape:", day_encoded.shape)

# Save encoder
joblib.dump(day_encoder, os.path.join(ARTIFACTS_DIR, "day_encoder.pkl"))

# Concatenate input
train_inputs_final = np.concatenate([train_inputs_svd, day_encoded], axis=1)
print("Final input shape:", train_inputs_final.shape)

# Save preprocessed input
np.save(os.path.join(SAVE_DIR, "train_inputs.npy"), train_inputs_final)
print(" Saved input to train_inputs.npy")

# TARGET PREPROCESSING
print("Preprocessing target RNA data...")

train_targets_final = preprocess_targets(train_targets_csr, n_components=128)

# if you want to save the raw target data as well, use the code below
# train_targets_final = train_targets_csr.toarray()   

np.save(os.path.join(SAVE_DIR, "train_targets.npy"), train_targets_final)
print(" Saved target to train_targets.npy")


