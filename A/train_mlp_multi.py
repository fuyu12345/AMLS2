import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GroupKFold
import optuna
import matplotlib.pyplot as plt



# CONFIGURATION AND ENV SETUP
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

DATA_DIR = "/scratch/zczlyf7/final_dataset"
METADATA_PATH = "/scratch/zczlyf7/dataset/metadata.csv"
IDXCOL_PATH = "/scratch/zczlyf7/new_dataset/train_multi_inputs_idxcol.npz"

# Load .npy data for Multiome 
train_inputs_np = np.load(os.path.join(DATA_DIR, "train_inputs.npy"))
train_targets_np = np.load(os.path.join(DATA_DIR, "train_targets.npy"))

train_inputs = torch.tensor(train_inputs_np, dtype=torch.float32).to(device)
train_targets = torch.tensor(train_targets_np, dtype=torch.float32).to(device)

INPUT_SIZE = train_inputs.shape[1]
OUTPUT_SIZE = train_targets.shape[1]
NB_EXAMPLES = train_inputs.shape[0]

print(f"Loaded input shape: {train_inputs.shape}")
print(f"Loaded target shape: {train_targets.shape}")

# Load metadata for grouping (donor+day)
idx_data = np.load(IDXCOL_PATH, allow_pickle=True)
cell_ids = idx_data["index"]
metadata = pd.read_csv(METADATA_PATH).set_index("cell_id")
metadata_aligned = metadata.loc[cell_ids]

# Create grouping labels
groups = metadata_aligned["donor"].astype(str) + "_" + metadata_aligned["day"].astype(str)


# DATASET, DATALOADER, AND UTILITY FUNCTIONS
class DenseDataset(Dataset):
    """
    Custom dataset for dense tensors.
    """
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
    ds = DenseDataset(X, Y, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)



#  MODEL DEFINITION
class MLP(nn.Module):
    """
    A simple MLP model with ReLU activations and optional dropout.
    """
    def __init__(self, layer_sizes, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
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



# TRAINING AND VALIDATION FUNCTIONS
def correlation_loss(pred, tgt):
    """
    Negative correlation as a loss: we want to maximize correlation.
    This function returns a *negative* value, so minimizing = maximizing correlation.
    """
    # pred, tgt: (batch_size, n_features)
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)
    corr_num = (pred_centered * tgt_centered).sum(dim=1)  # sum across features
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
        pred = model(X)
        loss = correlation_loss(pred, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(dl.dataset)

def valid_fn(model, dl):
    """
    Validate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    total_corr = 0
    with torch.no_grad():
        for X, Y in dl:
            pred = model(X)
            loss = correlation_loss(pred, Y)
            total_loss += loss.item() * X.size(0)
            
            # correlation (positive, for printing/evaluation)
            pred_centered = pred - pred.mean(dim=1, keepdim=True)
            Y_centered = Y - Y.mean(dim=1, keepdim=True)
            corr_num = (pred_centered * Y_centered).sum(dim=1)
            corr_den = (pred_centered.std(dim=1) * Y_centered.std(dim=1) + 1e-10) * (pred.shape[1] - 1)
            corr_val = corr_num / corr_den
            total_corr += corr_val.sum().item()
    avg_loss = total_loss / len(dl.dataset)
    avg_corr = total_corr / len(dl.dataset)
    return {"loss": avg_loss, "score": avg_corr}




# HYPERPARAM TUNING WITH OPTUNA (80/20 SPLIT)
# do a single 80/20 random split for the entire dataset
train_idxs, valid_idxs = train_test_split(
    np.arange(NB_EXAMPLES),
    test_size=0.2,
    # random_state=2
)
dl_train_optuna = make_dataloader(train_inputs, train_targets, train_idxs, shuffle=True)
dl_valid_optuna = make_dataloader(train_inputs, train_targets, valid_idxs, shuffle=False)

def objective(trial):
    """
    Optuna objective that proposes hyperparameters, trains on 80% data,
    and reports correlation on 20% data.
    """
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32,64, 128])
    n_layers = trial.suggest_int("n_layers", 2, 5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    wd = trial.suggest_loguniform("wd", 1e-5, 1e-2)
    head_type = trial.suggest_categorical("head", ["none", "softplus"])
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)
    # Build layer sizes
    layer_sizes = [INPUT_SIZE] + [hidden_dim]*n_layers + [OUTPUT_SIZE]
    model = build_model(layer_sizes, dropout=dropout_rate, head=head_type).to(device)
    if head_type == "softplus":
        model = nn.Sequential(model, nn.Softplus())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val_corr = None
    max_epochs = 10

    for epoch in range(max_epochs):
        tr_loss = train_fn(model, optimizer, dl_train_optuna)
        val_log = valid_fn(model, dl_valid_optuna)
        
        if best_val_corr is None or val_log["score"] > best_val_corr:
            best_val_corr = val_log["score"]
        # Optionally add early stopping if desired

    return best_val_corr  # Optuna tries to maximize this

# Create the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # number of trials can be increased
print("Best trial:", study.best_trial.params)

best_params = study.best_trial.params
print(f"Best hyperparameters from Optuna: {best_params}")


#  FINAL EVALUATION WITH GROUPKFold USING BEST HYPERPARAMS
def build_final_model(hparams):
    """
    Build the final model using the best hyperparameters.
    """
    hidden_dim = hparams["hidden_dim"]
    n_layers = hparams["n_layers"]
    dropout_rate = hparams.get("dropout", 0.0)
    head_type = hparams["head"]

    layer_sizes = [INPUT_SIZE] + [hidden_dim]*n_layers + [OUTPUT_SIZE]
    return build_model(layer_sizes, dropout=dropout_rate, head=head_type).to(device)


# Retrieve the best hyperparams from the study
best_hparams = study.best_trial.params
best_lr = best_hparams["lr"]
best_wd = best_hparams["wd"]



# make sure to set the directory for saving figures
FIGURE_DIR = "/home/zczlyf7/AMLS2/AMLS2_coursework/A/figures_2"
os.makedirs(FIGURE_DIR, exist_ok=True)

group_kfold = GroupKFold(n_splits=5)
folds_list = list(group_kfold.split(np.arange(NB_EXAMPLES), groups=groups))

final_scores = []

max_epochs_cv = 40  # you can use more epochs if desired
for fold_num, (train_idx, valid_idx) in enumerate(folds_list):
    print(f"\n========== Fold {fold_num+1}/5 ==========")

    # Create dataloaders
    dl_train = make_dataloader(train_inputs, train_targets, train_idx, shuffle=True)
    dl_valid = make_dataloader(train_inputs, train_targets, valid_idx, shuffle=False)

    # Build model with best hyperparams
    model = build_final_model(best_hparams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)

    best_score = None
    patience = 4  # early stopping
    best_params = None

    train_losses = []
    val_losses = []
    val_scores = []

    for epoch in range(max_epochs_cv):
        print(f"\n[Fold {fold_num+1}/5] Epoch {epoch+1}/{max_epochs_cv}")
        tr_loss = train_fn(model, optimizer, dl_train)
        val_log = valid_fn(model, dl_valid)

        print(f"Train Loss: {tr_loss:.4f}")
        print(f"Valid Loss: {val_log['loss']:.4f} | Score: {val_log['score']:.4f}")

        train_losses.append(tr_loss)
        val_losses.append(val_log["loss"])
        val_scores.append(val_log["score"])

        score = val_log["score"]
        if best_score is None or score > best_score:
            best_score = score
            best_params = model.state_dict()
            patience = 4
        else:
            patience -= 1
        if patience < 0:
            print("Early stopping in final CV")
            break

    # Save the best model for this fold
    save_dir = "/scratch/zczlyf7/parameter_multi/"
    os.makedirs(save_dir, exist_ok=True)
    fold_model_path = os.path.join(save_dir, f"final_fold{fold_num}_best_params.pth")
    torch.save(best_params, fold_model_path)

    print(f" Fold {fold_num+1} best score: {best_score:.4f}")
    final_scores.append(best_score)

    #  Plot learning curves for this fold
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve (Loss) - Fold {fold_num+1}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"fold{fold_num+1}_loss_curve.png"))
    plt.close()


    plt.figure(figsize=(10, 5))
    plt.plot(val_scores, label="Validation Correlation Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"Validation Score over Epochs - Fold {fold_num+1}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"fold{fold_num+1}_score_curve.png"))
    plt.close()

# Final bar plot of all fold scores
print("\n==== Final GroupKFold Results ====")
for i, sc in enumerate(final_scores):
    print(f"Fold {i+1} Score: {sc:.4f}")
print(f"Mean Score: {np.mean(final_scores):.4f} | Std: {np.std(final_scores):.4f}")

plt.figure(figsize=(8, 5))
plt.bar(range(1, 6), final_scores)
plt.xlabel("Fold")
plt.ylabel("Validation Score")
plt.title("Validation Score across GroupKFold Splits")
plt.ylim(0.0, 1.0)
plt.xticks(range(1, 6))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "groupkfold_scores_barplot.png"))
plt.close()






