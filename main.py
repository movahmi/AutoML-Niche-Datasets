
# Project: AutoML for Niche Datasets

# Main Script
# This script is designed to provide an AutoML solution tailored to niche datasets.
# It leverages TPOT and Auto-Sklearn with added domain-specific preprocessing pipelines.

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from autosklearn.classification import AutoSklearnClassifier

# Define a function for loading and preprocessing niche datasets.
def load_and_preprocess_data(file_path, domain):
    """
    Load dataset and apply domain-specific preprocessing steps.

    :param file_path: Path to the dataset file.
    :param domain: Domain type to apply specific preprocessing.
    :return: Processed X (features) and y (target).
    """
    data = pd.read_csv(file_path)

    # Example preprocessing based on domain
    if domain == "finance":
        # Custom feature engineering for financial datasets
        data["return"] = data["price_end"] / data["price_start"] - 1
        data.drop(columns=["price_end", "price_start"], inplace=True)
    elif domain == "art":
        # Normalize features for cultural datasets
        data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.dtype != 'object' else x)

    X = data.drop(columns=["target"])
    y = data["target"]
    return X, y

# Define a function to run TPOT.
def run_tpot(X_train, X_test, y_train, y_test, generations=5, population_size=20):
    """Run TPOT AutoML on the given dataset."""
    tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)
    print(f"TPOT Score: {tpot.score(X_test, y_test)}")
    tpot.export("tpot_pipeline.py")

# Define a function to run Auto-Sklearn.
def run_autosklearn(X_train, X_test, y_train, y_test):
    """Run Auto-Sklearn AutoML on the given dataset."""
    automl = AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=60, random_state=42)
    automl.fit(X_train, y_train)
    print(f"Auto-Sklearn Score: {automl.score(X_test, y_test)}")

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML for Niche Datasets")
    parser.add_argument("--file", required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--domain", required=True, help="Domain type (e.g., finance, art, etc.)")
    parser.add_argument("--tool", required=True, choices=["tpot", "autosklearn"], help="AutoML tool to use.")
    args = parser.parse_args()

    # Load and preprocess data
    X, y = load_and_preprocess_data(args.file, args.domain)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run the selected AutoML tool
    if args.tool == "tpot":
        run_tpot(X_train, X_test, y_train, y_test)
    elif args.tool == "autosklearn":
        run_autosklearn(X_train, X_test, y_train, y_test)
