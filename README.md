
# Open Problems - Multimodal Single-Cell Integration

In this project, we developed and systematically evaluated several machine learning models—including Multi-Layer Perceptron (MLP), LightGBM, and TabTransformer—for analyzing patterns and predicting gene expression and protein levels from single-cell multimodal data (Multiome and CITE-seq datasets). This work was carried out for the Kaggle competition: **Open Problems - Multimodal Single-Cell Integration**.

---

## 🗂 Project Organization

This project contains the following folders:

### 📁 Folder `A` – Multiome Task
Contains scripts related to the Multiome dataset.

- `GBM.py`: Training with LightGBM.
- `check.py`: Checks final submission format and missing data.
- `data_preprocess_multi_test.py`: Preprocessing for Multiome test data.
- `data_preprocess_multi_train.py`: Preprocessing for Multiome training data.
- `test_mlp_multi.py`: Final test using MLP, outputs submission CSV.
- `train_mlp_multi.py`: MLP training with hyperparameter tuning and 5-fold CV.
- `train_tab_multi.py`: TabTransformer training and validation (both custom and library versions), including tuning.

---

### 📁 Folder `B` – CITE-seq Task
Contains scripts related to the CITE-seq dataset.

- `data_preprocess_cite_test.py`: Preprocessing for CITE-seq test data.
- `data_preprocess_cite_train.py`: Preprocessing for CITE-seq training data.
- `test_mlp_cite.py`: Final prediction using MLP, creates submission file.
- `train_mlp_cite.py`: MLP training with tuning and cross-validation.
- `train_tab_cite.py`: TabTransformer training on CITE-seq data.

---

### 📁 Folder `Datasets`
This folder is initially empty. You need to manually download the dataset using:

```bash
kaggle competitions download -c open-problems-multimodal
```
## 📁 Folder `first_step_preprocessdata`

- `data_preprocess.py`: Initial preprocessing for all data; converts files to CSR format to reduce memory usage.
- `read_h5.py`: Simple utility for reading `.h5` files.

---

## 📄 `main.py`

Replace the paths in this file as needed to run the desired test scripts.

---

## 📄 `requirements.txt`

Contains all the required packages to run this project.

---

## ✅ Required Packages

The following Python packages are used in this project. You can install them using the `requirements.txt` file.

### Python Libraries

- pandas  
- numpy  
- scipy  
- scikit-learn  
- joblib  
- tqdm  
- lightgbm  
- torch  
- optuna  
- matplotlib  
- tab-transformer-pytorch  

---

## 🚀 Installation and Usage

To install all required packages:

```bash
pip install -r requirements.txt
```

## 🧪 Running the Code

Once your environment is set up, you can run different scripts by modifying the script paths as needed in your Python code.

To execute the main test pipeline, run:

```bash
python main.py
```
##⚠️ Important Note
This project involves working with different data types located in multiple folders.
Please ensure that all scripts have the correct paths configured for your local setup before running.
Many files require you to manually adjust file paths for loading and saving data.






