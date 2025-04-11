import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GroupKFold
import optuna

#  CONFIG 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

DATA_DIR = "/scratch/zczlyf7/final_dataset"
METADATA_PATH = "/scratch/zczlyf7/dataset/metadata.csv"
IDXCOL_PATH = "/scratch/zczlyf7/new_dataset/train_cite_inputs_idxcol.npz"

train_inputs_np = np.load(os.path.join(DATA_DIR, "train_cite_inputs.npy"))
train_targets_np = np.load(os.path.join(DATA_DIR, "train_cite_targets.npy"))

train_inputs = torch.tensor(train_inputs_np, dtype=torch.float32).to(device)
train_targets = torch.tensor(train_targets_np, dtype=torch.float32).to(device)

INPUT_SIZE = train_inputs.shape[1]
OUTPUT_SIZE = train_targets.shape[1]
NB_EXAMPLES = train_inputs.shape[0]

print(f"Loaded input shape: {train_inputs.shape}")
print(f"Loaded target shape: {train_targets.shape}")

#  GROUPING FOR CROSS-VALIDATION 
idx_data = np.load(IDXCOL_PATH, allow_pickle=True)
cell_ids = idx_data["index"]
metadata = pd.read_csv(METADATA_PATH).set_index("cell_id")
metadata_aligned = metadata.loc[cell_ids]
groups = metadata_aligned["donor"].astype(str) + "_" + metadata_aligned["day"].astype(str)

# DATASET + LOADER 
class DenseDataset(Dataset):
    def __init__(self, X, Y, indices):
        self.X = X
        self.Y = Y
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.X[i], self.Y[i]

def make_dataloader(X, Y, indices, batch_size=2048, shuffle=False):
    return DataLoader(DenseDataset(X, Y, indices), batch_size=batch_size, shuffle=shuffle)

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

def build_model(layer_sizes, dropout=0.0, head="softplus"):
    model = MLP(layer_sizes, dropout=dropout)
    if head == "softplus":
        return nn.Sequential(model, nn.Softplus())
    return model

# LOSS
def correlation_loss(pred, tgt):
    """
    Correlation loss function.
    """
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)
    corr_num = (pred_centered * tgt_centered).sum(dim=1)
    corr_den = (pred_centered.std(dim=1) * tgt_centered.std(dim=1) + 1e-10) * (pred.shape[1] - 1)
    corr = corr_num / corr_den
    return -torch.mean(corr)

def train_fn(model, optimizer, dl):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    for X, Y in dl:
        optimizer.zero_grad()
        loss = correlation_loss(model(X), Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(dl.dataset)

def valid_fn(model, dl):
    """
    Validate the model.
    """
    model.eval()
    total_loss, total_corr = 0, 0
    with torch.no_grad():
        for X, Y in dl:
            pred = model(X)
            loss = correlation_loss(pred, Y)
            total_loss += loss.item() * X.size(0)

            pred_centered = pred - pred.mean(dim=1, keepdim=True)
            Y_centered = Y - Y.mean(dim=1, keepdim=True)
            corr_num = (pred_centered * Y_centered).sum(dim=1)
            corr_den = (pred_centered.std(dim=1) * Y_centered.std(dim=1) + 1e-10) * (pred.shape[1] - 1)
            total_corr += (corr_num / corr_den).sum().item()
    return {
        "loss": total_loss / len(dl.dataset),
        "score": total_corr / len(dl.dataset)
    }

#  OPTUNA TUNING
train_idxs, valid_idxs = train_test_split(np.arange(NB_EXAMPLES), test_size=0.2)
dl_train_optuna = make_dataloader(train_inputs, train_targets, train_idxs, shuffle=True)
dl_valid_optuna = make_dataloader(train_inputs, train_targets, valid_idxs, shuffle=False)

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 192, 256])
    n_layers = trial.suggest_int("n_layers", 2, 5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    wd = trial.suggest_loguniform("wd", 1e-5, 1e-2)
    head_type = trial.suggest_categorical("head", ["none", "softplus"])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Create model and optimizer
    layer_sizes = [INPUT_SIZE] + [hidden_dim]*n_layers + [OUTPUT_SIZE]
    model = build_model(layer_sizes, dropout=dropout, head=head_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val = None
    for epoch in range(10):
        train_fn(model, optimizer, dl_train_optuna)
        val = valid_fn(model, dl_valid_optuna)["score"]
        if best_val is None or val > best_val:
            best_val = val
    return best_val

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
best_params = study.best_trial.params
print("Best hyperparameters:", best_params)

# FINAL GROUPKFold CV 
def build_final_model(hparams):
    layer_sizes = [INPUT_SIZE] + [hparams["hidden_dim"]] * hparams["n_layers"] + [OUTPUT_SIZE]
    return build_model(layer_sizes, dropout=hparams["dropout"], head=hparams["head"]).to(device)

group_kfold = GroupKFold(n_splits=5)
folds = list(group_kfold.split(np.arange(NB_EXAMPLES), groups=groups))

save_dir = "/scratch/zczlyf7/parameter_cite"
os.makedirs(save_dir, exist_ok=True)

# Final training and validation
final_scores = []
for fold_num, (train_idx, valid_idx) in enumerate(folds):
    model = build_final_model(best_params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["wd"])

    dl_train = make_dataloader(train_inputs, train_targets, train_idx, shuffle=True)
    dl_valid = make_dataloader(train_inputs, train_targets, valid_idx, shuffle=False)

    best_score, best_state = None, None
    patience = 4
    for epoch in range(40):
        print(f"\n[Fold {fold_num+1}/5] Epoch {epoch+1}")
        train_fn(model, optimizer, dl_train)
        val_log = valid_fn(model, dl_valid)
        print(f"  Loss: {val_log['loss']:.4f} | Score: {val_log['score']:.4f}")

        if best_score is None or val_log["score"] > best_score:
            best_score = val_log["score"]
            best_state = model.state_dict()
            patience = 4
        else:
            patience -= 1
        if patience < 0:
            break

    torch.save(best_state, os.path.join(save_dir, f"final_fold{fold_num}_best_params.pth"))
    final_scores.append(best_score)

print("\n Final CV Scores:")
for i, s in enumerate(final_scores):
    print(f"  Fold {i+1}: {s:.4f}")
print(f"  Mean: {np.mean(final_scores):.4f} ± {np.std(final_scores):.4f}")
