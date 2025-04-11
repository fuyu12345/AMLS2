import pandas as pd


EVAL_PARQUET_PATH = "/scratch/zczlyf7/new_dataset/evaluation.parquet"
SUBMISSION_PATH = "/home/zczlyf7/AMLS2/AMLS2_coursework/A/submission_MLP.csv"

# Load 
eval_df = pd.read_parquet(EVAL_PARQUET_PATH)
print(" Evaluation parquet shape:", eval_df.shape)
print("  - Expected submission rows:", len(eval_df))


sub_df = pd.read_csv(SUBMISSION_PATH)
print("Submission CSV shape:", sub_df.shape)
print("\n First 10 rows of submission:")
print(sub_df.head(30))

# Check for row_id issues
print("\nSubmission checks:")
print("  - row_id is unique:", sub_df['row_id'].is_unique)
print("  - row_id starts at 0:", sub_df['row_id'].min() == 0)
print("  - row_id max:", sub_df['row_id'].max())
print("  - Any NaNs in target?", sub_df['target'].isna().any())
