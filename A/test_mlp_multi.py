import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import joblib

# CONFIG 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

TEST_INPUT_PATH = "/scratch/zczlyf7/final_dataset/test_inputs.npy"
MODEL_DIR = "/scratch/zczlyf7/parameter_multi"
IDXCOL_PATH = "/scratch/zczlyf7/new_dataset/test_multi_inputs_idxcol.npz"
EVAL_PARQUET_PATH = "/scratch/zczlyf7/new_dataset/evaluation.parquet"
TARGET_COLUMNS_PATH = "/scratch/zczlyf7/new_dataset/train_multi_targets_idxcol.npz"
SUBMISSION_PATH = "/home/zczlyf7/AMLS2/AMLS2_coursework/A/submission_MLP.csv"

OUTPUT_SIZE = 23418
BATCH_SIZE = 1024

# hyperparameters from training
best_params = {
    "hidden_dim": 128,
    "n_layers": 2,
    "lr": 0.003119759019743404,
    "wd": 0.00011038938790833339,
    "head": "none",
    "dropout": 0.12062068955145466,
}

# LOAD TEST INPUT 
test_inputs = np.load(TEST_INPUT_PATH)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(DEVICE)

# MODEL 
class MLP(nn.Module):
    """
    A simple MLP model with ReLU activations and optional dropout.
    """
    def __init__(self, layer_sizes, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def build_model(layer_sizes, dropout=0.0, head="none"):
    model = MLP(layer_sizes, dropout=dropout)
    if head == "softplus":
        model = nn.Sequential(model, nn.Softplus())
    return model.to(DEVICE)

# LOAD MODELS using best artifacts from training
layer_sizes = [test_inputs.shape[1]] + [best_params["hidden_dim"]] * best_params["n_layers"] + [OUTPUT_SIZE]
models = []
for fold in range(5):
    model = build_model(layer_sizes, dropout=best_params["dropout"], head=best_params["head"])
    model_path = os.path.join(MODEL_DIR, f"final_fold{fold}_best_params.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    models.append(model)
print(" Loaded all 5 CV models")

# PREDICT 
preds = []
with torch.no_grad():
    for i in tqdm(range(0, test_inputs.shape[0], BATCH_SIZE)):
        batch = test_inputs[i:i + BATCH_SIZE]
        batch_preds = [model(batch) for model in models]
        avg_pred = torch.stack(batch_preds).mean(dim=0)
        preds.append(avg_pred.cpu().numpy())

# POSTPROCESS 
print(" Inversing SVD and restoring original target space...")


SVD_PATH = "/scratch/zczlyf7/final_dataset/artifacts_multi/target_svd.pkl"
MEDIAN_PATH = "/scratch/zczlyf7/final_dataset/artifacts_multi/target_col_medians.npy"
svd = joblib.load(SVD_PATH)
col_medians = np.load(MEDIAN_PATH)
final_preds_full = svd.inverse_transform(np.vstack(preds))
final_preds_full = np.vstack(preds)
final_preds_full += col_medians


#MAP TO SUBMISSION 
print("Mapping predictions to submission format...")
eval_ids = pd.read_parquet(EVAL_PARQUET_PATH)
eval_ids.cell_id = eval_ids.cell_id.astype("category")
eval_ids.gene_id = eval_ids.gene_id.astype("category")

y_columns = np.load(TARGET_COLUMNS_PATH, allow_pickle=True)["columns"]
eval_ids = eval_ids[eval_ids.gene_id.isin(y_columns)].copy()

test_index = np.load(IDXCOL_PATH, allow_pickle=True)["index"]
cell_dict = {k: v for v, k in enumerate(test_index)}
gene_dict = {k: v for v, k in enumerate(y_columns)}

cell_nums = eval_ids.cell_id.map(cell_dict).to_numpy()
gene_nums = eval_ids.gene_id.map(gene_dict).to_numpy()
valid_mask = (cell_nums != -1) & (gene_nums != -1)

valid_cell_idx = cell_nums[valid_mask].astype(int)
valid_gene_idx = gene_nums[valid_mask].astype(int)
multi_pred = final_preds_full[valid_cell_idx, valid_gene_idx]

#  SUBMISSION 

submission = pd.Series(index=pd.MultiIndex.from_frame(eval_ids), dtype=np.float32, name="target")
submission.iloc[valid_mask] = multi_pred

# Save as DataFrame with reset index
# Prepare Kaggle-ready format
submission = submission.reset_index(drop=True)  # Drop MultiIndex
submission.index.name = "row_id"  # Name the index as row_id
submission = submission.to_frame()  # Convert Series to DataFrame with single 'target' column
submission.to_csv(SUBMISSION_PATH)

print(" Saved Multiome-only submission to:", SUBMISSION_PATH)






