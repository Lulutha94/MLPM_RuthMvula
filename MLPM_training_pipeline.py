# -*- coding: utf-8 -*-
"""
Created on Tuesday Jan 30 17:27:07 2024

@author: Ruth Mvula

"""

import os, random
from sklearn.model_selection import StratifiedKFold, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from imblearn.metrics import geometric_mean_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.auto import tqdm
import json
from collections import Counter
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Convert input to a DataFrame
        X = pd.DataFrame(X)

        # Convert Date column to datetime and extract features
        X["Date"] = pd.to_datetime(X["Date"])
        X["Year"] = X["Date"].dt.year
        X["Month"] = X["Date"].dt.month
        X["Day"] = X["Date"].dt.day
        X["Weekday"] = X["Date"].dt.weekday

        # Drop the original Date column
        return X.drop(["Date"], axis=1)


def set_seed(seed=42069):
    # np and random
    random.seed(seed)
    np.random.seed(seed)

    # hash
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


set_seed()

df = pd.read_csv("simulated_failure_data.csv")
X = df.drop(["Failure", "Component_ID"], axis=1)
y = df["Failure"]

debug = False


if debug:
    X = X.sample(n=500, random_state=42)
    y = y.loc[X.index]

# Encode target labels for AUROC calculation
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define K-Fold cross-validation
# kf = StratifiedKFold(n_splits=5)
tscv = TimeSeriesSplit(n_splits=5)

# Define models and their parameter grids
models = {
    # 'RandomForest': (RandomForestClassifier(), {'classifier__n_estimators': [x for x in range(50, 500, 50)],
    #                                             'classifier__max_features': ['sqrt', 'log2'],
    #                                             # 'classifier__max_depth': [int(x) for x in np.linspace(10, 110, num = 20, dtype=int)],
    #                                             'classifier__min_samples_split': [2, 5, 10],
    #                                             'classifier__min_samples_leaf': [1, 2, 4],
    #                                             # 'classifier__bootstrap': [True, False],
    #                                             'classifier__criterion': ['gini', 'entropy']}),
    "DecisionTree": (
        DecisionTreeClassifier(),
        {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None]
            + [int(x) for x in np.linspace(10, 110, num=11)],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": [None, "sqrt", "log2"],
        },
    ),
    "SVM": (
        SVC(probability=True, C=1),
        {
            "classifier__kernel": ["linear", "rbf", "poly", "sigmoid"],
            "classifier__gamma": ["scale", "auto", 1, 0.1, 0.01, 0.001],
        },
    ),
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        {
            "classifier__n_estimators": [x for x in range(50, 500, 50)],
            # 'classifier__max_depth': [int(x) for x in np.linspace(10, 110, num = 20, dtype=int)],
            "classifier__learning_rate": np.linspace(0.01, 0.2, 5),
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        },
    ),
    "ExtraTrees": (
        ExtraTreesClassifier(),
        {
            "classifier__n_estimators": [x for x in range(50, 500, 50)],
            "classifier__max_features": ["sqrt", "log2"],
            # 'classifier__max_depth': [int(x) for x in np.linspace(10, 110, num = 20, dtype=int)],
            "classifier__max_leaf_nodes": [None, 10, 20, 50, 100],
            # 'classifier__min_samples_split': [2, 5, 10],
            # 'classifier__min_samples_leaf': [1, 2, 4],
            # 'classifier__bootstrap': [True, False],
            "classifier__criterion": ["gini", "entropy"],
        },
    ),
}

# Initialize results
results = {
    model: {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "gmean": [],
        "aucs": [],
    }
    for model in models
}
training_rates = {
    model: {
        "aucs": [],
        "tprs": [],
        "mean_fpr": np.linspace(0, 1, 100),
        "y_true": [],
        "y_pred": [],
        "y_proba": [],
        "best_params": [],
        "selected_features": [],
        "best_smote_tomek_params": [],
    }
    for model in models
}

colors = [
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
]  # Add more colors if more models


fig, ax = plt.subplots()
set_seed()
# Cross-validation loop
for fold, (train_index, test_index) in enumerate(
    tqdm(tscv.split(X, y_encoded), total=tscv.get_n_splits(), desc="Folds Progress")
):
    print(f"~~~~Processing Fold {fold+1}/{tscv.get_n_splits()}...~~~~")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    print("~~~~~~~~Applying date transformation...~~~~~~~~")

    date_transformer = DateTransformer()
    X_train = date_transformer.transform(X_train)
    X_test = date_transformer.transform(X_test)

    print("Number of training samples", Counter(y_train))
    print("Number of testing samples", Counter(y_test))

    cv = TimeSeriesSplit(n_splits=3)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [
                    col
                    for col in X.select_dtypes(include=["int64", "float64"]).columns
                    if col != "Date"
                ],
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                [
                    col
                    for col in X.select_dtypes(include=["object"]).columns
                    if col != "Date"
                ],
            ),
        ]
    )

    set_seed()
    for i, (model_name, (model, params)) in enumerate(models.items()):
        print(f"~~~~~~~~Training {model_name} in fold {fold+1}...~~~~~~~~")
        ###### OLD PIPELINE ######
        # pipeline = Pipeline([('preprocessor', preprocessor), ('select', SelectKBest()), ('classifier', model)])
        ########################

        ###### NEW PIPELINE ######
        pipeline = ImbPipeline(
            [
                ("preprocessor", preprocessor),
                ("resample", SMOTETomek()),  # Add the SMOTE-Tomek resampling step here
                ("select", SelectKBest()),
                ("classifier", model),
            ]
        )
        ########################
        grid_search = GridSearchCV(
            pipeline, params, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Evaluate on the test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Extract the best parameters and selected features
        best_params = grid_search.best_params_
        # Get the feature names after the preprocessing step (before SelectKBest)
        preprocessed_feature_names = (
            best_model.named_steps["preprocessor"].transformers_[0][2]
            + best_model.named_steps["preprocessor"].transformers_[1][2]
        )

        # Get the selected features by SelectKBest
        selector = best_model.named_steps["select"]
        selected_features_mask = selector.get_support()
        selected_feature_names = [
            name
            for (name, selected) in zip(
                preprocessed_feature_names, selected_features_mask
            )
            if selected
        ]

        # Extract the best parameters for SMOTE-Tomek
        smote_tomek_params = {k: v for k, v in best_params.items() if "resample__" in k}

        # Calculate metrics
        accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        gmean = geometric_mean_score(y_test, y_pred, average="macro")

        # Compute ROC curve and area the curve

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        tprs = interp(training_rates[model_name]["mean_fpr"], fpr, tpr)
        tprs[0] = 0.0
        roc_auc = auc(fpr, tpr)

        # Store results
        results[model_name]["accuracy"].append(accuracy)
        results[model_name]["precision"].append(precision)
        results[model_name]["recall"].append(recall)
        results[model_name]["f1"].append(f1)
        results[model_name]["gmean"].append(gmean)
        results[model_name]["aucs"].append(roc_auc)

        training_rates[model_name]["tprs"].append(tprs)
        training_rates[model_name]["aucs"].append(roc_auc)
        training_rates[model_name]["y_true"].append(y_test)
        training_rates[model_name]["y_pred"].append(y_pred)
        training_rates[model_name]["y_proba"].append(y_proba)
        training_rates[model_name]["best_params"].append(best_params)
        training_rates[model_name]["selected_features"].append(selected_feature_names)
        training_rates[model_name]["best_smote_tomek_params"].append(smote_tomek_params)

    print(f"Fold {fold+1} completed.\n")

    if debug:
        if fold + 1 >= 2:
            break

# Plotting mean ROC curve for each model
# Initialize the figure for the combined ROC curves
plt.figure(figsize=(10, 8))

# Plotting mean ROC curve for each model and saving individual ROC curves
for i, (model_name, data) in enumerate(training_rates.items()):
    mean_tpr = np.mean(data["tprs"], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(data["mean_fpr"], mean_tpr)
    std_auc = np.std(data["aucs"])

    # Plot for the combined ROC
    plt.plot(
        data["mean_fpr"],
        mean_tpr,
        color=colors[i % len(colors)],
        label=r"Mean ROC for %s (AUC = %0.2f $\pm$ %0.2f)"
        % (model_name, mean_auc, std_auc),
        lw=2,
    )

    # Create individual plot for each model
    plt.figure()
    plt.plot(
        data["mean_fpr"],
        mean_tpr,
        color=colors[i % len(colors)],
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray", label="Chance")
    plt.title(f"Mean ROC Curve for {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"Mean_ROC_Curve_{model_name}.png")
    plt.close()

# Add the chance line and labels for the combined ROC plot
plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Combined Mean ROC Curves")
plt.legend(loc="lower right")

# Save the combined ROC curve image
plt.savefig("Combined_Mean_ROC_Curves.png")
plt.close()

# (Optionally) Save metrics to files after all folds
for model_name, metrics in results.items():
    df = pd.DataFrame(metrics)
    df.to_csv(f"{model_name}_performance_metrics.csv", index=False)

for model_name, data in training_rates.items():
    if "tprs" in data:
        data["tprs"] = [tpr.tolist() for tpr in data["tprs"]]
    if "aucs" in data:
        data["aucs"] = [float(auc) for auc in data["aucs"]]
    if "mean_fpr" in data:
        data["mean_fpr"] = data["mean_fpr"].tolist()

    if "y_true" in data:
        data["y_true"] = [y_true.tolist() for y_true in data["y_true"]]
    if "y_pred" in data:
        data["y_pred"] = [y_pred.tolist() for y_pred in data["y_pred"]]
    if "y_proba" in data:
        data["y_proba"] = [y_proba.tolist() for y_proba in data["y_proba"]]


with open("model_training_rates.json", "w") as json_file:
    json.dump(training_rates, json_file, indent=4)

with open("model_results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)
