# This script retrained  a  single multi-layer perceptron (MLP) model on the CITE dataset and generates predictions for the test set.
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import joblib

#  CONFIG 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f" Using device: {DEVICE}")

# Paths
TRAIN_INPUT_PATH = "/scratch/zczlyf7/final_dataset/train_cite_inputs.npy"
TRAIN_TARGET_PATH = "/scratch/zczlyf7/final_dataset/train_cite_targets.npy"
TEST_INPUT_PATH = "/scratch/zczlyf7/final_dataset/test_cite_inputs.npy"
SVD_PATH = "/scratch/zczlyf7/final_dataset/artifacts_cite/target_svd.pkl"
MEDIAN_PATH = "/scratch/zczlyf7/final_dataset/artifacts_cite/target_col_medians.npy"
EVAL_PARQUET_PATH = "/scratch/zczlyf7/new_dataset/evaluation.parquet"
IDXCOL_PATH = "/scratch/zczlyf7/new_dataset/test_cite_inputs_idxcol.npz"
TARGET_COLUMNS_PATH = "/scratch/zczlyf7/new_dataset/train_cite_targets_idxcol.npz"
SUBMISSION_PATH = "/home/zczlyf7/AMLS2/AMLS2_coursework/A/submission_MLP.csv"

# Hyperparameters
OUTPUT_SIZE = 140
BATCH_SIZE = 1024
EPOCHS = 200
PATIENCE = 10

best_params = {
    "hidden_dim": 256,
    "n_layers": 3,
    "lr": 0.0011,
    "dropout": 0.15,
}

#  MODEL 
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

def build_model(input_dim):
    layer_sizes = [input_dim] + [best_params["hidden_dim"]] * best_params["n_layers"] + [OUTPUT_SIZE]
    return MLP(layer_sizes, dropout=best_params["dropout"]).to(DEVICE)

#  LOAD DATA
print(" Loading CITE training & test data...")
X_train = np.load(TRAIN_INPUT_PATH)
Y_train = np.load(TRAIN_TARGET_PATH)
X_test = np.load(TEST_INPUT_PATH)

X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

#  TRAIN 
model = build_model(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
loss_fn = nn.MSELoss()

best_val_loss = float("inf")
patience_counter = 0

print(" Training MLP on CITE data...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, Y_train)
    loss.backward()
    optimizer.step()

    val_loss = loss.item()
    print(f"Epoch {epoch+1}: loss = {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(" Early stopping")
            break

model.load_state_dict(best_model)
model.eval()

#PREDICT 
print("🔮 Predicting on test set...")
preds = []
with torch.no_grad():
    for i in tqdm(range(0, X_test.shape[0], BATCH_SIZE)):
        batch = X_test[i:i + BATCH_SIZE]
        preds.append(model(batch).cpu().numpy())
preds = np.vstack(preds)

#  POSTPROCESS 
print(" Inversing SVD and restoring original target space...")
svd = joblib.load(SVD_PATH)
col_medians = np.load(MEDIAN_PATH)
final_preds_full = svd.inverse_transform(preds)
final_preds_full += col_medians
final_preds_full = np.vstack(preds)


#  MAP TO SUBMISSION 
print("Mapping predictions to submission format...")
eval_ids = pd.read_parquet(EVAL_PARQUET_PATH)
eval_ids.cell_id = eval_ids.cell_id.astype("category")
eval_ids.gene_id = eval_ids.gene_id.astype("category")
y_columns = np.load(TARGET_COLUMNS_PATH, allow_pickle=True)["columns"]
eval_ids = eval_ids[eval_ids.gene_id.isin(y_columns)].copy()

# Load test index and create dictionaries for mapping
test_index = np.load(IDXCOL_PATH, allow_pickle=True)["index"]
cell_dict = {k: v for v, k in enumerate(test_index)}
gene_dict = {k: v for v, k in enumerate(y_columns)}

# Map cell and gene IDs to indices
cell_nums = eval_ids.cell_id.map(cell_dict).to_numpy()
gene_nums = eval_ids.gene_id.map(gene_dict).to_numpy()
valid_mask = (cell_nums != -1) & (gene_nums != -1)
valid_cell_idx = cell_nums[valid_mask].astype(int)
valid_gene_idx = gene_nums[valid_mask].astype(int)
cite_pred = final_preds_full[valid_cell_idx, valid_gene_idx]

#  FINAL SUBMISSION 
print(" Preparing final submission...")
submission_series = pd.Series(index=pd.MultiIndex.from_frame(eval_ids), dtype=np.float32, name="target")
submission_series.iloc[valid_mask] = cite_pred

start_row_id = 58931360
submission_df = submission_series.reset_index(drop=True)
submission_df = submission_df.to_frame().reset_index()
submission_df["row_id"] = np.arange(start_row_id, start_row_id + len(submission_df))
submission_df = submission_df[["row_id", "target"]]

if os.path.exists(SUBMISSION_PATH):
    multi_df = pd.read_csv(SUBMISSION_PATH)
    print(" Loaded existing Multiome submission")
else:
    raise FileNotFoundError(f"{SUBMISSION_PATH} does not exist. Run test_multi.py first.")

final_df = pd.concat([multi_df, submission_df], axis=0)
final_df = final_df.sort_values("row_id").reset_index(drop=True)
final_df.index.name = "row_id"
final_df.to_csv(SUBMISSION_PATH, index=False)

print("Appended CITE predictions and saved final submission to:", SUBMISSION_PATH)






# # This script loads a pre-trained MLP model from traing CV, makes predictions on the test set, and prepares a submission file.
# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import joblib

# #  CONFIG 
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f" Using device: {DEVICE}")

# TEST_INPUT_PATH = "/scratch/zczlyf7/final_dataset/test_cite_inputs.npy"
# MODEL_DIR = "/scratch/zczlyf7/parameter_cite"
# IDXCOL_PATH = "/scratch/zczlyf7/new_dataset/test_cite_inputs_idxcol.npz"
# EVAL_PARQUET_PATH = "/scratch/zczlyf7/new_dataset/evaluation.parquet"
# TARGET_COLUMNS_PATH = "/scratch/zczlyf7/new_dataset/train_cite_targets_idxcol.npz"
# SUBMISSION_PATH = "/home/zczlyf7/AMLS2/AMLS2_coursework/A/submission_MLP.csv"

# OUTPUT_SIZE = 140
# BATCH_SIZE = 1024

# best_params = {
#     "hidden_dim": 256,
#     "n_layers": 4,
#     "lr": 0.004528652755340088,
#     "wd": 0.007602581446566692,
#     "head": "none",
#     "dropout": 0.1144671983843158,
# }

# # LOAD TEST INPUT 
# test_inputs = np.load(TEST_INPUT_PATH)
# test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(DEVICE)

# # MODEL 
# class MLP(nn.Module):
#     """
#     A simple MLP model with ReLU activations and optional dropout.
#     """
#     def __init__(self, layer_sizes, dropout=0.0):
#         super().__init__()
#         layers = []
#         for i in range(len(layer_sizes) - 1):
#             layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
#             if i < len(layer_sizes) - 2:
#                 layers.append(nn.ReLU())
#                 if dropout > 0:
#                     layers.append(nn.Dropout(dropout))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)

# def build_model(layer_sizes, dropout=0.0, head="none"):
#     model = MLP(layer_sizes, dropout=dropout)
#     if head == "softplus":
#         model = nn.Sequential(model, nn.Softplus())
#     return model.to(DEVICE)
    
# #  LOAD MODELS 
# layer_sizes = [test_inputs.shape[1]] + [best_params["hidden_dim"]] * best_params["n_layers"] + [OUTPUT_SIZE]
# models = []
# for fold in range(5):
#     model = build_model(layer_sizes, dropout=best_params["dropout"], head=best_params["head"])
#     model_path = os.path.join(MODEL_DIR, f"final_fold{fold}_best_params.pth")
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval()
#     models.append(model)
# print(" Loaded all 5 CV CITE models")

# # PREDICT 
# preds = []
# with torch.no_grad():
#     for i in tqdm(range(0, test_inputs.shape[0], BATCH_SIZE)):
#         batch = test_inputs[i:i + BATCH_SIZE]
#         batch_preds = [model(batch) for model in models]
#         avg_pred = torch.stack(batch_preds).mean(dim=0)
#         preds.append(avg_pred.cpu().numpy())

# #  POSTPROCESS
# print(" Inversing SVD and restoring original target space...")
# SVD_PATH = "/scratch/zczlyf7/final_dataset/artifacts_cite/target_svd.pkl"
# MEDIAN_PATH = "/scratch/zczlyf7/final_dataset/artifacts_cite/target_col_medians.npy"
# svd = joblib.load(SVD_PATH)
# col_medians = np.load(MEDIAN_PATH)

# final_preds_full = svd.inverse_transform(np.vstack(preds))
# final_preds_full += col_medians
# final_preds_full = np.vstack(preds)

# # MAP TO SUBMISSION 
# print(" Mapping predictions to submission format...")
# eval_ids = pd.read_parquet(EVAL_PARQUET_PATH)
# eval_ids.cell_id = eval_ids.cell_id.astype("category")
# eval_ids.gene_id = eval_ids.gene_id.astype("category")

# y_columns = np.load(TARGET_COLUMNS_PATH, allow_pickle=True)["columns"]
# eval_ids = eval_ids[eval_ids.gene_id.isin(y_columns)].copy()

# test_index = np.load(IDXCOL_PATH, allow_pickle=True)["index"]
# cell_dict = {k: v for v, k in enumerate(test_index)}
# gene_dict = {k: v for v, k in enumerate(y_columns)}

# cell_nums = eval_ids.cell_id.map(cell_dict).to_numpy()
# gene_nums = eval_ids.gene_id.map(gene_dict).to_numpy()
# valid_mask = (cell_nums != -1) & (gene_nums != -1)

# valid_cell_idx = cell_nums[valid_mask].astype(int)
# valid_gene_idx = gene_nums[valid_mask].astype(int)
# cite_pred = final_preds_full[valid_cell_idx, valid_gene_idx]

# # FINAL SUBMISSION 
# print(" Preparing final submission...")

# # Create a Series with MultiIndex to hold CITE predictions
# submission_series = pd.Series(index=pd.MultiIndex.from_frame(eval_ids), dtype=np.float32, name="target")
# submission_series.iloc[valid_mask] = cite_pred

# # Convert to DataFrame with proper row_id
# start_row_id = 58931360
# submission_df = submission_series.reset_index(drop=True)
# submission_df = submission_df.to_frame().reset_index()
# submission_df["row_id"] = np.arange(start_row_id, start_row_id + len(submission_df))
# submission_df = submission_df[["row_id", "target"]]


# # Load existing Multiome submission
# if os.path.exists(SUBMISSION_PATH):
#     multi_df = pd.read_csv(SUBMISSION_PATH)
#     print(" Loaded existing Multiome submission")
# else:
#     raise FileNotFoundError(f"{SUBMISSION_PATH} does not exist. Run test_multi.py first.")

# # Combine Multiome + CITE predictions
# final_df = pd.concat([multi_df, submission_df], axis=0)
# final_df = final_df.sort_values("row_id").reset_index(drop=True)
# final_df.index.name = "row_id"

# # Save the updated full submission
# final_df.to_csv(SUBMISSION_PATH, index=False)
# print(" Appended CITE predictions and saved final submission to:", SUBMISSION_PATH)


