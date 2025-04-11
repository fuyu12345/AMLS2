import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

#  Partial Correlation 
def partial_correlation_score(y_true, y_pred):
    """
    Computes the mean partial correlation over samples.
    y_true, y_pred: (n_samples, n_features)
    """
    y_true_centered = y_true - y_true.mean(axis=1, keepdims=True)
    y_pred_centered = y_pred - y_pred.mean(axis=1, keepdims=True)

    cov_tp = np.sum(y_true_centered * y_pred_centered, axis=1) / (y_true.shape[1] - 1)
    var_t = np.sum(y_true_centered ** 2, axis=1) / (y_true.shape[1] - 1)
    var_p = np.sum(y_pred_centered ** 2, axis=1) / (y_true.shape[1] - 1)

    return np.mean(cov_tp / (np.sqrt(var_t * var_p) + 1e-8))

# path
INPUT_PATH = '/scratch/zczlyf7/new_dataset/train_multi_inputs.npz'
TARGET_PATH = '/scratch/zczlyf7/new_dataset/train_multi_targets.npz'

# load
X = sp.load_npz(INPUT_PATH).tocsr().copy()   # make sure writable
Y = sp.load_npz(TARGET_PATH).toarray()

print("input shape:", X.shape)
print("target shape:", Y.shape)

# K-Fold 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"\n Fold {fold + 1}")

    X_train, X_valid = X[train_idx], X[valid_idx]
    Y_train, Y_valid = Y[train_idx], Y[valid_idx]

    #  MultiOutputRegressor + LightGBM 
    base_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        n_jobs=8
    )

    multi_model = MultiOutputRegressor(base_model, n_jobs=1)
    multi_model.fit(X_train, Y_train)

    # predict
    Y_pred = multi_model.predict(X_valid)

    #  partial correlation score
    score = partial_correlation_score(Y_valid, Y_pred)
    print(f" Fold {fold + 1} Partial Correlation Score: {score:.4f}")
    fold_scores.append(score)

print(f"\n✅ average Partial Correlation Score: {np.mean(fold_scores):.4f}")





# $$$$$$$$$$$$$$$$$$$$$$$$$$$$below are the Hyperparameter Tuning with Optuna process$$$$$$$$$$$$$$$$$$$$$


# import os
# import numpy as np
# import lightgbm as lgb
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.model_selection import KFold
# import optuna

# # Partial Correlation 
# def partial_correlation_score(y_true, y_pred):
#     """
#     Computes the mean partial correlation over samples.
#     y_true, y_pred: (n_samples, n_features)
#     """
#     y_true_centered = y_true - y_true.mean(axis=1, keepdims=True)
#     y_pred_centered = y_pred - y_pred.mean(axis=1, keepdims=True)

#     cov_tp = np.sum(y_true_centered * y_pred_centered, axis=1) / (y_true.shape[1] - 1)
#     var_t = np.sum(y_true_centered ** 2, axis=1) / (y_true.shape[1] - 1)
#     var_p = np.sum(y_pred_centered ** 2, axis=1) / (y_true.shape[1] - 1)

#     return np.mean(cov_tp / (np.sqrt(var_t * var_p) + 1e-8))


# #PATHS 
# DATA_DIR = "/scratch/zczlyf7/final_dataset"
# X_PATH = os.path.join(DATA_DIR, "train_inputs.npy")   # SVD-reduced features
# Y_PATH = os.path.join(DATA_DIR, "train_targets.npy")  # SVD-reduced targets

# # Load 
# X = np.load(X_PATH)  
# Y = np.load(Y_PATH)  

# print(" SVD-Reduced Input shape:", X.shape)
# print(" SVD-Reduced Target shape:", Y.shape)



# #  Hyperparameter Tuning with Optuna
# def objective(trial):
#     """
#     Objective function for Optuna hyperparameter tuning.
#     maximize the partial correlation score.
#     """

#     #   hyperparams 
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 200),
#         "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
#         "num_leaves": trial.suggest_int("num_leaves", 8, 128),
#         "n_jobs": 8,
#         "random_state": 42
#     }

#     #  small K-Fold (3 splits) for quick evaluation 
#     kf = KFold(n_splits=3, shuffle=True, random_state=42)
#     fold_scores = []
#     for train_idx, valid_idx in kf.split(X):
#         X_train, X_valid = X[train_idx], X[valid_idx]
#         Y_train, Y_valid = Y[train_idx], Y[valid_idx]

#         # Wrap LightGBM in MultiOutputRegressor
#         base_model = lgb.LGBMRegressor(**params)
#         multi_model = MultiOutputRegressor(base_model, n_jobs=1)
#         multi_model.fit(X_train, Y_train)

#         Y_pred = multi_model.predict(X_valid)
#         score = partial_correlation_score(Y_valid, Y_pred)
#         fold_scores.append(score)

#     # maximize partial correlation, return negative for "minimize" 
#     return -np.mean(fold_scores)


# # Create and run study 
# study = optuna.create_study(direction="minimize")  
# study.optimize(objective, n_trials=2, show_progress_bar=True)

# print("Best trial:")
# best_trial = study.best_trial
# print("  Value: ", -best_trial.value)  # negative of stored value = best partial correlation
# print("  Params: ")
# for key, val in best_trial.params.items():
#     print(f"    {key}: {val}")

# best_params = study.best_params
# best_params["n_jobs"] = 8
# best_params["random_state"] = 42
# print("\nBest hyperparameters from Optuna:", best_params)



# #  Final 5-Fold CV with best hyperparams
# #  We re-train and measure partial correlation.

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# fold_scores = []

# for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
#     print(f"\n Fold {fold + 1}")

#     X_train, X_valid = X[train_idx], X[valid_idx]
#     Y_train, Y_valid = Y[train_idx], Y[valid_idx]

#     base_model = lgb.LGBMRegressor(**best_params)
#     multi_model = MultiOutputRegressor(base_model, n_jobs=1)
#     multi_model.fit(X_train, Y_train)

#     Y_pred = multi_model.predict(X_valid)
#     score = partial_correlation_score(Y_valid, Y_pred)

#     print(f" Fold {fold+1} Partial Correlation Score: {score:.4f}")
#     fold_scores.append(score)

# print("\n Average Partial Correlation Score across 5 folds:", np.mean(fold_scores))
