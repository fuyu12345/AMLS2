# # $$$$$$$$$$$$$$$$$$$$$$$$$$$$below are the Hyperparameter Tuning with Optuna process, using the tabtransformer from library$$$$$$$$$$$$$$$$$$$$$
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tab_transformer_pytorch import TabTransformer

# CONFIG 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/scratch/zczlyf7/final_dataset"
METADATA_PATH = "/scratch/zczlyf7/dataset/metadata.csv"
IDXCOL_PATH = "/scratch/zczlyf7/new_dataset/train_multi_inputs_idxcol.npz"
BATCH_SIZE = 512
EPOCHS = 20
PATIENCE = 5

# LOAD DATA 
X = np.load(f"{DATA_DIR}/train_inputs.npy")  
Y = np.load(f"{DATA_DIR}/train_targets.npy")  
idxcol = np.load(IDXCOL_PATH, allow_pickle=True)["index"]

metadata = pd.read_csv(METADATA_PATH).set_index("cell_id").loc[idxcol]
metadata = metadata[["day", "donor", "technology"]].copy()

metadata["technology"] = LabelEncoder().fit_transform(metadata["technology"])
metadata["donor"] = LabelEncoder().fit_transform(metadata["donor"])

# SPLIT INPUT
X_cont = X[:, :128]  # continuous part
X_cat = metadata[["day", "donor", "technology"]].values  # categorical part

X_cat = torch.tensor(X_cat, dtype=torch.long)
X_cont = torch.tensor(X_cont, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# Normalize targets (per sample)
Y = Y - Y.mean(dim=1, keepdim=True)
Y = Y / (Y.std(dim=1, keepdim=True) + 1e-6)
print(" Target normalized: mean =", Y.mean().item(), ", std =", Y.std().item())

# DATASET
class TabDataset(Dataset):
    """
    Custom dataset for tabular data with continuous and categorical features.
    """
    def __init__(self, cont, cat, y):
        self.cont = cont
        self.cat = cat
        self.y = y

    def __len__(self):
        return len(self.cont)

    def __getitem__(self, idx):
        return self.cont[idx], self.cat[idx], self.y[idx]

dataset = TabDataset(X_cont, X_cat, Y)

# correlation loss function
def correlation_loss(pred, target):
    # same correlation loss as before
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = pred.norm(dim=1) * target.norm(dim=1) + 1e-6
    return -torch.mean(num / den)

def train_model(model, optimizer, train_loader):
    """
    Train the model for one epoch.
    """
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
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    all_corrs = []
    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont, x_cat, y = x_cont.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
            pred = model(x_cat, x_cont)

            # compute correlation per sample
            pred = pred - pred.mean(dim=1, keepdim=True)
            y = y - y.mean(dim=1, keepdim=True)
            num = (pred * y).sum(dim=1)
            den = pred.norm(dim=1) * y.norm(dim=1) + 1e-6
            corr = num / den

            all_corrs.append(corr.cpu())
    return torch.cat(all_corrs).mean().item()

# GROUP K-FOLD CV 
groups = metadata["donor"].astype(str) + "_" + metadata["day"].astype(str)
gkf = GroupKFold(n_splits=5)

fold_scores = []
# get number of categories for each categorical column
num_categories = [int(X_cat[:, i].max().item() + 1) for i in range(X_cat.shape[1])]

for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(dataset)), groups=groups)):
    print(f"\n=== Fold {fold+1}/5 ===")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    # LIBRARY TABTRANSFORMER
    model = TabTransformer(
        categories = num_categories,              
        num_continuous = X_cont.shape[1],         # number of continuous features
        dim = 16,                                 # embedding dimension
        depth = 2,                                # number of transformer layers
        heads = 4,                                # number of attention heads
        dim_out = Y.shape[1],                    # output dimension
        mlp_hidden_mults = (4, 2)                
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

print("\n==== Cross-Validation Results ====")
for i, score in enumerate(fold_scores):
    print(f"Fold {i+1} Corr: {score:.4f}")
print(f"Mean Corr: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")




# $$$$$$$$$$$$$$$$$$$$$$$$$$$below are the self constructed tabtransformer, no tuning$$$$$$$$$$$$$$$$$$$$$

# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, Subset
# from sklearn.model_selection import GroupKFold
# from sklearn.preprocessing import LabelEncoder
# from tqdm import tqdm

# # CONFIG 
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DATA_DIR = "/scratch/zczlyf7/final_dataset"
# METADATA_PATH = "/scratch/zczlyf7/dataset/metadata.csv"
# IDXCOL_PATH = "/scratch/zczlyf7/new_dataset/train_multi_inputs_idxcol.npz"
# BATCH_SIZE = 512
# EPOCHS = 20
# PATIENCE = 5

# # LOAD DATA 
# X = np.load(f"{DATA_DIR}/train_inputs.npy")  
# Y = np.load(f"{DATA_DIR}/train_targets.npy")  
# idxcol = np.load(IDXCOL_PATH, allow_pickle=True)["index"]

# metadata = pd.read_csv(METADATA_PATH).set_index("cell_id").loc[idxcol]
# metadata = metadata[["day", "donor", "technology"]].copy()

# metadata["technology"] = LabelEncoder().fit_transform(metadata["technology"])
# metadata["donor"] = LabelEncoder().fit_transform(metadata["donor"])

# #  SPLIT INPUT 
# X_cont = X[:, :128]
# X_cat = metadata[["day", "donor", "technology"]].values

# X_cat = torch.tensor(X_cat, dtype=torch.long)
# X_cont = torch.tensor(X_cont, dtype=torch.float32)
# Y = torch.tensor(Y, dtype=torch.float32)

# Y = Y - Y.mean(dim=1, keepdim=True)
# Y = Y / (Y.std(dim=1, keepdim=True) + 1e-6)
# print(" Target normalized: mean =", Y.mean().item(), ", std =", Y.std().item())

# # DATASET 
# class TabDataset(Dataset):
#     def __init__(self, cont, cat, y):
#         self.cont = cont
#         self.cat = cat
#         self.y = y

#     def __len__(self):
#         return len(self.cont)

#     def __getitem__(self, idx):
#         return self.cont[idx], self.cat[idx], self.y[idx]

# dataset = TabDataset(X_cont, X_cat, Y)

# #  TABTRANSFORMER 
# class TabTransformer(nn.Module):
#     """
#     A simple TabTransformer model for tabular data.
#     """
#     def __init__(self, num_categories, embed_dim, cont_dim, output_dim):
#         super().__init__()
#         self.embeds = nn.ModuleList([
#             nn.Embedding(num_cat, embed_dim) for num_cat in num_categories
#         ])
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
#             num_layers=2
#         )
#         self.cont_proj = nn.Linear(cont_dim, embed_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim * (len(num_categories) + 1), 68),
#             nn.ReLU(),
#             nn.Linear(68, output_dim)
#         )

#     def forward(self, x_cont, x_cat):
#         cat_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
#         cat_stack = torch.stack(cat_emb, dim=1)
#         cat_out = self.transformer(cat_stack)
#         cont_out = self.cont_proj(x_cont).unsqueeze(1)
#         full = torch.cat([cat_out.flatten(1), cont_out.flatten(1)], dim=1)
#         return self.mlp(full)

# # = LOSS & TRAINING 
# def correlation_loss(pred, target):
#     pred = pred - pred.mean(dim=1, keepdim=True)
#     target = target - target.mean(dim=1, keepdim=True)
#     num = (pred * target).sum(dim=1)
#     den = pred.norm(dim=1) * target.norm(dim=1) + 1e-6
#     return -torch.mean(num / den)


# def train_model(model, optimizer, train_loader):
#     """
#     Train the model for one epoch.
#     """
#     model.train()
#     total_loss = 0
#     for x_cont, x_cat, y in train_loader:
#         x_cont, x_cat, y = x_cont.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
#         optimizer.zero_grad()
#         pred = model(x_cont, x_cat)
#         loss = correlation_loss(pred, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * x_cont.size(0)
#     return total_loss / len(train_loader.dataset)


# def eval_model(model, val_loader):
#     """
#     Evaluate the model on the validation set.
#     """
#     model.eval()
#     all_corrs = []
#     with torch.no_grad():
#         for x_cont, x_cat, y in val_loader:
#             x_cont, x_cat, y = x_cont.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
#             pred = model(x_cont, x_cat)

#             pred = pred - pred.mean(dim=1, keepdim=True)
#             y = y - y.mean(dim=1, keepdim=True)
#             num = (pred * y).sum(dim=1)
#             den = pred.norm(dim=1) * y.norm(dim=1) + 1e-6
#             corr = num / den

#             all_corrs.append(corr.cpu())

#     return torch.cat(all_corrs).mean().item()


# #  GROUP K-FOLD CV 
# groups = metadata["donor"].astype(str) + "_" + metadata["day"].astype(str)
# gkf = GroupKFold(n_splits=5)

# fold_scores = []
# num_categories = [int(X_cat[:, i].max().item() + 1) for i in range(X_cat.shape[1])]

# for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(dataset)), groups=groups)):
#     print(f"\n=== Fold {fold+1}/5 ===")

#     train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

#     model = TabTransformer(num_categories=num_categories, embed_dim=16, cont_dim=X_cont.shape[1], output_dim=Y.shape[1]).to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     best_score = None
#     best_state = None
#     patience_counter = 0
#     # train the model
#     for epoch in range(EPOCHS):
#         train_loss = train_model(model, optimizer, train_loader)
#         val_corr = eval_model(model, val_loader)
#         print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Corr={val_corr:.4f}")

#         if best_score is None or val_corr > best_score:
#             best_score = val_corr
#             best_state = model.state_dict()
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= PATIENCE:
#                 print("Early stopping!")
#                 break

#     fold_scores.append(best_score)
#     print(f" Fold {fold+1} Best Val Corr: {best_score:.4f}")

# print("\n==== Cross-Validation Results ====")
# for i, score in enumerate(fold_scores):
#     print(f"Fold {i+1} Corr: {score:.4f}")
# print(f"Mean Corr: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

