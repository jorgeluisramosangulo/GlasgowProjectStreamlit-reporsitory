# === Core Libraries ===
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# === Versioning ===
from packaging import version
from sklearn import __version__ as sklearn_version

# === Scikit-learn: Model Selection & Evaluation ===
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_predict
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# === Scikit-learn: Models ===
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression

# === Scikit-learn: Preprocessing & Feature Engineering ===
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.utils.multiclass import type_of_target

# === Imbalanced-learn ===
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler



# === Helper Functions ===

# === Label Decoder Helper for Later Use ===
# Converts numeric labels back to their original string form using the stored label_map_.
# Useful for displaying predictions in a human-readable format.
def decode_labels(y_encoded):
    reverse_map = {v: k for k, v in st.session_state["label_map_"].items()}
    return [reverse_map.get(val, val) for val in y_encoded]


# === Class 1 Probability Extractor ===
# Returns predicted probabilities for the positive class (label 1), regardless of class order.
# Ensures compatibility with models where class labels may be [0, 1] or [1, 0].
def get_class1_proba(model, X):
    """
    Returns predicted probability for the class the user mapped to 1.
    Handles cases where model.classes_ may not be in [0, 1] order.
    """
    if "label_map" not in st.session_state:
        raise ValueError("label_map not found in session state.")

    # This is the integer that represents class "1" in model.classes_
    mapped_class_1_value = 1

    # Now lookup this value in model.classes_ (e.g., classes_ = [1, 0] or [0, 1])
    class_idx = list(model.classes_).index(mapped_class_1_value)

    return model.predict_proba(X)[:, class_idx]



# === Ensure DataFrame and Series Format ===
# Converts NumPy arrays to pandas DataFrame (X) and Series (y) if needed.
# Helps standardize input formats before modeling or transformation.
def ensure_dataframe_and_series(X, y):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    return X, y


# === Initialize Session State Key ===
# Sets a default value for a Streamlit session state key if it doesn't already exist.
# Prevents KeyError when accessing session variables.
def init_session_key(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value

# === Metrics Logger and Display ===
# Saves metrics to session state and displays them under a subheader.
# Useful for tracking and presenting model evaluation results.
def log_metrics(metrics: dict, name: str = "Train Metrics"):
    st.session_state[f"{name}_metrics"] = metrics
    st.subheader(name)
    for k, v in metrics.items():
        st.write(f"{k}: {v:.4f}")

# === Cross-Validation Scores Display ===
# Neatly displays mean and standard deviation of cross-validation metrics.
# Helps evaluate model stability across folds.
def display_cv_scores(cv_scores: dict):
    if cv_scores:
        st.subheader("Cross-Validation Scores")
        for metric, (mean, std) in cv_scores.items():
            st.write(f"{metric}: {mean:.4f} ± {std:.4f}")


# === Final Train/Validation Set Selector ===
# Returns the final X_train and X_val sets, using PCA or resampled versions if available.
# Also removes the 'row_id' column if it exists.
def get_final_train_val_sets():
    """Returns the final X_train and X_val sets based on PCA and resampling status,
    and drops 'row_id' if present."""
    
    use_pca = st.session_state.get("use_pca", "No")
    pca_ready = st.session_state.get("pca_ready", False)

    # Get base X sets depending on PCA
    if use_pca == "Yes" and pca_ready:
        X_train = st.session_state["X_train_pca"]
        X_val = st.session_state["X_val_pca"]
    else:
        # Use resampled if available, else raw
        X_train = st.session_state.get("X_train_resampled", st.session_state["X_train"])
        X_val = st.session_state.get("X_val_resampled", st.session_state["X_val"])

    # Drop 'row_id' if still present
    X_train = X_train.drop(columns=["row_id"], errors="ignore")
    X_val = X_val.drop(columns=["row_id"], errors="ignore")

    return X_train, X_val



# === Final Target Set Selector ===
# Retrieves the final y_train and y_val sets, using resampled y_train if available.
# Ensures consistency with the selected training features.
def get_final_y_sets():
    """Returns the final y_train and y_val sets based on resampling status."""
    y_train = st.session_state.get("y_train_resampled", st.session_state["y_train"])
    y_val = st.session_state["y_val"]
    return y_train, y_val



# === Store Final Feature Sets ===
# Saves the final X_train and X_val datasets to session state for model training.
# Keeps the modeling pipeline consistent and accessible across app steps.
def set_final_datasets(X_train_final, X_val_final):
    """Store final modeling-ready X sets in session state."""
    st.session_state["X_train_final"] = X_train_final
    st.session_state["X_val_final"] = X_val_final

# === Transformation Logger ===
# Logs each transformation step with its name, transformer object, and target group.
# Useful for tracking and debugging the preprocessing pipeline.
def log_transformation(step_name, transformer, target="general"):
    if "transform_steps" not in st.session_state:
        st.session_state["transform_steps"] = []
    st.session_state["transform_steps"].append((step_name, transformer, target))


# === Model Trainer with Optional Tuning ===
# Fits a classification model with or without hyperparameter tuning (Grid or Randomized Search).
# Returns the trained model, predictions, predicted probabilities, evaluation metrics, and optional CV scores.
def train_model(
    model_name: str,
    base_model,
    param_grid: dict,
    X_train,
    y_train,
    enable_tuning: bool = False,
    search_method: str = "Grid Search",
    n_iter: int = 10,
    n_folds: int = 10,
    metric: str = "roc_auc",
    random_state: int = 42,
    max_iter: int = 1000,
):
    """
    Trains a classification model with optional hyperparameter tuning.
    Returns the fitted model, predictions, and probabilities.
    """


    if enable_tuning:
        if search_method == "Grid Search":
            search = GridSearchCV(
                base_model,
                param_grid=param_grid,
                cv=n_folds,
                scoring=metric,
                n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=n_folds,
                scoring=metric,
                n_jobs=-1,
                random_state=random_state
            )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = base_model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_train)
    y_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_train)

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_train, y_pred),
        "Precision": precision_score(y_train, y_pred),
        "Recall": recall_score(y_train, y_pred),
        "F1": f1_score(y_train, y_pred),
        "AUC": roc_auc_score(y_train, y_prob),
    }

    # Optional cross-validation
    cv_scores = None
    if n_folds > 1:
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_result = cross_validate(model, X_train, y_train, cv=n_folds, scoring=scoring, n_jobs=-1)
        cv_scores = {
            metric: (cv_result[f'test_{metric}'].mean(), cv_result[f'test_{metric}'].std())
            for metric in scoring
        }

    return model, y_pred, y_prob, metrics, cv_scores




# === Scaling Utilities ===
def apply_minmax_scaling(X_train, X_val, column):
    scaler = MinMaxScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_train_scaled[column] = scaler.fit_transform(X_train[[column]])
    X_val_scaled[column] = scaler.transform(X_val[[column]])
    return X_train_scaled, X_val_scaled, scaler

def apply_standard_scaling(X_train, X_val, column):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_train_scaled[column] = scaler.fit_transform(X_train[[column]])
    X_val_scaled[column] = scaler.transform(X_val[[column]])
    return X_train_scaled, X_val_scaled, scaler

# === Visualization of Before/After ===
def plot_before_after(before, after, title):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    sns.histplot(before, ax=ax[0], kde=True, color='skyblue')
    ax[0].set_title(f"Before {title}")
    ax[0].set_xlabel(before.name)
    ax[0].set_ylabel("Count")

    sns.histplot(after, ax=ax[1], kde=True, color='steelblue')
    ax[1].set_title(f"After {title}")
    ax[1].set_xlabel(after.name)
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    return fig  # ✅ Return fig for thread-safe use in Streamlit


# === Missing Value Step Logger ===
def log_missing_value_action(method, column, value=None):
    if "missing_value_steps" not in st.session_state:
        st.session_state["missing_value_steps"] = []
    st.session_state["missing_value_steps"].append((method, column, value))

# === Add Row ID Helper ===
def add_row_id(df):
    df = df.copy()
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(1, len(df) + 1))
    return df

# === Traffic Light Labeling ===
def get_traffic_light(pred, prob, threshold_0=0.85, threshold_1=0.85):
    if pred == 0 and (1 - prob) >= threshold_0:
        return "Green"
    elif pred == 1 and prob >= threshold_1:
        return "Red"
    else:
        return "Yellow"
    
# === Feature Creation Helper ===
# Applies a new feature operation to both training and validation sets
def create_new_feature(X_train, X_val, operation, col1, col2=None, new_col_name="new_feature"):
    if operation == "Add":
        new_train = X_train[col1] + X_train[col2]
        new_val = X_val[col1] + X_val[col2]
    elif operation == "Subtract":
        new_train = X_train[col1] - X_train[col2]
        new_val = X_val[col1] - X_val[col2]
    elif operation == "Multiply":
        new_train = X_train[col1] * X_train[col2]
        new_val = X_val[col1] * X_val[col2]
    elif operation == "Divide":
        new_train = X_train[col1] / (X_train[col2] + 1e-9)
        new_val = X_val[col1] / (X_val[col2] + 1e-9)
    elif operation == "Log":
        new_train = np.log1p(X_train[col1])
        new_val = np.log1p(X_val[col1])
    elif operation == "Square":
        new_train = X_train[col1] ** 2
        new_val = X_val[col1] ** 2
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    X_train[new_col_name] = new_train
    X_val[new_col_name] = new_val
    return X_train, X_val




def export_training_data_general(
    X_train_final,
    y_train_raw,
    model,
    row_ids=None,
    model_name="Model",
    use_original_labels=False,
    flip_outputs=False,
    label_map=None
):
    """
    General export function for training data:
    - Includes row_id, original target, model prediction, and probability
    - Supports flipping outputs for alignment
    - Supports original label restoration
    - Computes standard training metrics
    """

    def get_class1_proba(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            return model.decision_function(X)
        else:
            return None

    # Predict
    y_pred = model.predict(X_train_final)
    y_prob = get_class1_proba(model, X_train_final)

    # Flip if requested
    if flip_outputs:
        y_pred = 1 - y_pred
        y_prob = 1 - y_prob
        y_train_flipped = 1 - pd.Series(y_train_raw)
    else:
        y_train_flipped = pd.Series(y_train_raw)

    # Restore original labels for display if applicable
    if use_original_labels and label_map is not None:
        inverse_map = {v: k for k, v in label_map.items()}
        y_target_display = y_train_flipped.map(inverse_map)
        y_pred_display = pd.Series(y_pred).map(inverse_map)
    else:
        y_target_display = y_train_flipped
        y_pred_display = pd.Series(y_pred)

    # Assemble export DataFrame — include full training features
    df_export = X_train_final.copy().reset_index(drop=True)

    # Insert row_id as first column if available
    if row_ids is not None:
        df_export.insert(0, "row_id", pd.Series(row_ids).reset_index(drop=True))

    # Add target, prediction, and probability columns
    df_export["target"] = y_target_display.reset_index(drop=True)
    df_export[f"{model_name}_Prediction"] = y_pred_display.reset_index(drop=True)
    df_export[f"{model_name}_Prob"] = y_prob


    # Metrics using numeric labels
    if label_map is not None:
        y_true_numeric = y_train_flipped.map(label_map)
        y_pred_numeric = pd.Series(y_pred).map(label_map)
    else:
        y_true_numeric = y_train_flipped
        y_pred_numeric = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_true_numeric, y_pred_numeric),
        "Precision": precision_score(y_true_numeric, y_pred_numeric, pos_label=1),
        "Recall": recall_score(y_true_numeric, y_pred_numeric, pos_label=1),
        "F1-Score": f1_score(y_true_numeric, y_pred_numeric, pos_label=1),
        "AUC": roc_auc_score(y_true_numeric, y_prob) if y_prob is not None else None
    }

    return df_export, metrics



def export_training_data_pls_da(
    X_train_final,
    y_train_raw,
    model,
    row_ids=None,
    model_name="PLSDA",
    use_original_labels=False,
    flip_outputs=False,
    label_map=None
):
    """
    Export function for PLS-DA:
    - Handles shape (n_samples, 1) output from PLSRegression
    - Includes row_id, target, prediction, and probability
    - Supports flipped outputs and original label mapping
    - Returns export DataFrame and performance metrics
    """


    def get_class1_proba(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            return model.decision_function(X)
        else:
            return None

    # Predict and flatten PLS output
    y_pred = np.ravel(model.predict(X_train_final))
    y_pred = (y_pred >= 0.5).astype(int)  # Convert to binary labels
    y_prob = np.ravel(model.predict(X_train_final))  # Probabilities still needed for AUC

    # Flip if requested
    if flip_outputs:
        y_pred = 1 - y_pred
        y_prob = 1 - y_prob
        y_train_flipped = 1 - pd.Series(y_train_raw)
    else:
        y_train_flipped = pd.Series(y_train_raw)

    # Restore original labels for display if needed
    if use_original_labels and label_map is not None:
        inverse_map = {v: k for k, v in label_map.items()}
        y_target_display = y_train_flipped.map(inverse_map)
        y_pred_display = pd.Series(y_pred).map(inverse_map)
    else:
        y_target_display = y_train_flipped
        y_pred_display = pd.Series(y_pred)

    # Assemble export DataFrame
    df_export = X_train_final.copy().reset_index(drop=True)
    if row_ids is not None:
        df_export.insert(0, "row_id", pd.Series(row_ids).reset_index(drop=True))
    df_export["target"] = y_target_display.reset_index(drop=True)
    df_export[f"{model_name}_Prediction"] = y_pred_display.reset_index(drop=True)
    df_export[f"{model_name}_Prob"] = y_prob

    # Metrics using numeric labels
    if label_map is not None:
        y_true_numeric = y_train_flipped.map(label_map)
        y_pred_numeric = pd.Series(y_pred).map(label_map)
    else:
        y_true_numeric = y_train_flipped
        y_pred_numeric = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_true_numeric, y_pred_numeric),
        "Precision": precision_score(y_true_numeric, y_pred_numeric, pos_label=1),
        "Recall": recall_score(y_true_numeric, y_pred_numeric, pos_label=1),
        "F1-Score": f1_score(y_true_numeric, y_pred_numeric, pos_label=1),
        "AUC": roc_auc_score(y_true_numeric, y_prob) if y_prob is not None else None
    }

    return df_export, metrics



def apply_flipping(model_name, pred, prob, flip):
    if not flip:
        return pred, prob
    if model_name == "PLS-DA":
        return 1 - pred, -1 * prob
    return 1 - pred, 1 - prob
