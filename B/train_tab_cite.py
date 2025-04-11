# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$below are the Hyperparameter Tuning with Optuna process, using the tabtransformer from library$$$$$$$$$$$$$$$$$$$$$
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna
from tab_transformer_pytorch import TabTransformer
from tqdm import tqdm

#  CONFIG 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/scratch/zczlyf7/final_dataset"
METADATA_PATH = "/scratch/zczlyf7/dataset/metadata.csv"
IDXCOL_PATH = "/scratch/zczlyf7/new_dataset/train_cite_inputs_idxcol.npz"
BATCH_SIZE = 512
EPOCHS = 20
PATIENCE = 5

#  LOAD DATA 
X = np.load(f"{DATA_DIR}/train_cite_inputs.npy")  
Y = np.load(f"{DATA_DIR}/train_cite_targets.npy")  
idxcol = np.load(IDXCOL_PATH, allow_pickle=True)["index"]

metadata = pd.read_csv(METADATA_PATH).set_index("cell_id").loc[idxcol]
metadata = metadata[["day", "donor", "technology"]].copy()
metadata["technology"] = LabelEncoder().fit_transform(metadata["technology"])
metadata["donor"] = LabelEncoder().fit_transform(metadata["donor"])

#  SPLIT INPUT 
X_cont = X[:, :128]
X_cat = metadata[["day", "donor", "technology"]].values

X_cat = torch.tensor(X_cat, dtype=torch.long)
X_cont = torch.tensor(X_cont, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

Y = Y - Y.mean(dim=1, keepdim=True)
Y = Y / (Y.std(dim=1, keepdim=True) + 1e-6)

#  DATASET 
class TabDataset(Dataset):
    def __init__(self, cont, cat, y):
        self.cont = cont
        self.cat = cat
        self.y = y

    def __len__(self):
        return len(self.cont)

    def __getitem__(self, idx):
        return self.cont[idx], self.cat[idx], self.y[idx]

dataset = TabDataset(X_cont, X_cat, Y)

#  LOSS 
def correlation_loss(pred, target):
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = pred.norm(dim=1) * target.norm(dim=1) + 1e-6
    return -torch.mean(num / den)

# TRAIN / EVAL 
def train_model(model, optimizer, train_loader):
    model.train()
    total_loss = 0
    for x_cont, x_cat, y in train_loader:
        x_cont, x_cat, y = x_cont.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x_cat, x_cont)
        loss = correlation_loss(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_cont.size(0)
    return total_loss / len(train_loader.dataset)

def eval_model(model, val_loader):
    model.eval()
    all_corrs = []
    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont, x_cat, y = x_cont.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
            pred = model(x_cat, x_cont)
            pred = pred - pred.mean(dim=1, keepdim=True)
            y = y - y.mean(dim=1, keepdim=True)
            num = (pred * y).sum(dim=1)
            den = pred.norm(dim=1) * y.norm(dim=1) + 1e-6
            corr = num / den
            all_corrs.append(corr.cpu())
    return torch.cat(all_corrs).mean().item()

#  HYPERPARAMETER TUNING 
# Split the dataset into train and validation sets for hyperparameter tuning
groups = metadata["donor"].astype(str) + "_" + metadata["day"].astype(str)
train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=groups)

train_loader_tune = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader_tune = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
num_categories = [int(X_cat[:, i].max().item() + 1) for i in range(X_cat.shape[1])]

# Define the objective function for Optuna
def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    model = TabTransformer(
        categories=num_categories,
        num_continuous=X_cont.shape[1],
        dim=trial.suggest_categorical("dim", [16, 32, 64]),
        depth=trial.suggest_int("depth", 1, 4),
        heads=trial.suggest_categorical("heads", [2, 4, 8]),
        dim_out=Y.shape[1],
        mlp_hidden_mults=trial.suggest_categorical("mlp_hidden_mults", [(4,2), (2,1), (8,4)])
    ).to(DEVICE)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = None
    for _ in range(10):
        _ = train_model(model, optimizer, train_loader_tune)
        val_corr = eval_model(model, val_loader_tune)
        if best_val is None or val_corr > best_val:
            best_val = val_corr
    return best_val

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(" Best hyperparameters:", study.best_trial.params)
best_params = study.best_trial.params

#  CROSS-VALIDATION 
gkf = GroupKFold(n_splits=5)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(dataset)), groups=groups)):
    print(f"\n=== Fold {fold+1}/5 ===")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = TabTransformer(
        categories=num_categories,
        num_continuous=X_cont.shape[1],
        dim=best_params["dim"],
        depth=best_params["depth"],
        heads=best_params["heads"],
        dim_out=Y.shape[1],
        mlp_hidden_mults=best_params["mlp_hidden_mults"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    best_score = None
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_model(model, optimizer, train_loader)
        val_corr = eval_model(model, val_loader)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Corr={val_corr:.4f}")
        if best_score is None or val_corr > best_score:
            best_score = val_corr
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping!")
                break

    fold_scores.append(best_score)
    print(f" Fold {fold+1} Best Val Corr: {best_score:.4f}")

# FINAL REPORT
print("\n==== Final TabTransformer CV Results (CITE-seq) ====")
for i, score in enumerate(fold_scores):
    print(f"Fold {i+1} Corr: {score:.4f}")
print(f"Mean Corr: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")





