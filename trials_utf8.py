
# --- CELL ---
%pip install matplotlib seaborn

# --- CELL ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# --- CELL ---
df = pd.read_csv("../data/phishingData.csv")

df.sample(5)

# --- CELL ---
df.columns

# --- CELL ---
df.columns.str.lower()

# --- CELL ---
df.columns = df.columns.str.lower()

# --- CELL ---
df.result.value_counts()

# --- CELL ---
data = [
    {
        "column_name": col,
        "total_unique_values": df[col].nunique(),
        "allowed_unique_values": df[col].unique(),
        "data_type": df[col].dtype,
    }
    for col in df.columns
]

# --- CELL ---
column_data = pd.DataFrame(data=data)
column_data

# --- CELL ---
column_data["allowed_unique_values"] = column_data["allowed_unique_values"].apply(
    lambda x: x.tolist() if hasattr(x, "tolist") else x
)

# --- CELL ---
schema = {"columns": {}}

# --- CELL ---
for _, row in column_data.iterrows():
    schema["columns"][row["column_name"]] = {
        "total_unique_values": row["total_unique_values"],
        "allowed_unique_values": row["allowed_unique_values"],
        "data_type": str(row["data_type"]),
    }

# --- CELL ---
from phishdetector.utils.common import write_yaml
from pathlib import Path

# --- CELL ---
write_yaml(filepath=Path("../data/schema.yaml"), content=schema)

# --- CELL ---
results_df = pd.read_csv("../artifacts/reports/training_report.csv")

# --- CELL ---
results_df["param_clf"].value_counts()

# --- CELL ---
results_df.columns

# --- CELL ---
results_df.groupby("param_clf")[
    ["mean_test_score", "mean_fit_time", "rank_test_score"]
].max().sort_values(by="mean_test_score", ascending=False).reset_index()

# --- CELL ---
core_results = results_df[[
    'rank_test_score',
    'mean_test_score',
    'std_test_score',
    'mean_fit_time',
    'param_clf'
]].sort_values(by='rank_test_score').reset_index(drop=True)

print(core_results.head(10))

# --- CELL ---
# Robustly extract classifier names
def extract_clf_name(clf_str):
    if 'RandomForestClassifier' in clf_str:
        return 'RandomForest'
    elif 'KNeighborsClassifier' in clf_str:
        return 'KNeighbors'
    elif 'SVC' in clf_str:
        return 'SVC'
    elif 'RidgeClassifier' in clf_str:
        return 'Ridge'
    elif 'LogisticRegression' in clf_str:
        return 'Logistic'
    else:
        return 'Unknown'

# Add a column with just the classifier name
results_df['clf_name'] = results_df['param_clf'].apply(extract_clf_name)

# Get the row with the highest mean_test_score for each classifier
best_models = results_df.loc[
    results_df.groupby('clf_name')['mean_test_score'].idxmax()
].sort_values(by='mean_test_score', ascending=False)

# Show the best result for each classifier
print(best_models[['clf_name', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'params']])

# --- CELL ---
from phishdetector.utils import write_yaml

# --- CELL ---
def save_report(results_df: pd.DataFrame):
    results_df["clf_name"] = results_df["param_clf"].apply(extract_clf_name)
    best_models = results_df.loc[
        results_df.groupby("clf_name")["mean_test_score"].idxmax()
    ].sort_values(by="mean_test_score", ascending=False)
    
    report = best_models[
        [
            "clf_name",
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "params",
        ]
    ].to_dict(orient="records")
    
    print(report)
    
    # write_yaml(filepath="../artifacts/reports/training_report.yaml", content=report)

# --- CELL ---
save_report(results_df=results_df)