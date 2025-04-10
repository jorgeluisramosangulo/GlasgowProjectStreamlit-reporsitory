import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.title("🤖 Binary Classification App")
st.info("This app builds a binary classification model!")

# === File Upload ===
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        elif file_type == "json":
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"✅ Successfully loaded {file_type.upper()} file.")
    st.write("Preview of your uploaded data:")
    st.dataframe(df)

    # === Dataset Overview ===
    st.markdown("### 📋 Dataset Overview")
    st.write(f"📃 **Rows:** {df.shape[0]} | 📄 **Columns:** {df.shape[1]}")

    col_dtype_df = pd.DataFrame({'Column': df.columns, 'Data Type': df.dtypes.values})
    st.markdown("#### 📌 Column Data Types")
    st.dataframe(col_dtype_df)

    dtype_counts_df = df.dtypes.value_counts().reset_index()
    dtype_counts_df.columns = ['Data Type', 'Count']
    st.markdown("#### 📊 Data Type Frequency")
    st.dataframe(dtype_counts_df)

    # === Target Selection ===
    target_column = st.selectbox("Select the target column:", df.columns)
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]

    if y_raw.nunique() != 2:
        st.error("❌ The selected target column must have exactly 2 unique values for binary classification.")
        st.stop()
    else:
        y_raw = pd.factorize(y_raw)[0]

    # === Train/Validation Split ===
    test_size_percent = st.slider("Select validation set size (%)", 10, 50, 20, 5)
    test_size = test_size_percent / 100.0
    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=test_size, random_state=42, shuffle=True)

    # === PCA Step ===
    use_pca = st.radio("Would you like to apply PCA?", ["No", "Yes"])
    if use_pca == "Yes":
        numeric_cols_train = X_train.select_dtypes(include=np.number).dropna(axis=1)
        numeric_cols_val = X_val[numeric_cols_train.columns]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(numeric_cols_train)
        X_val_scaled = scaler.transform(numeric_cols_val)

        pca = PCA()
        pca.fit(X_train_scaled)

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(cum_var)+1), cum_var, marker='o')
        ax.set_title("Cumulative Explained Variance")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance")
        st.pyplot(fig)

        n_components = st.slider("Select number of principal components to keep", 1, X_train_scaled.shape[1], 2)

        pca = PCA(n_components=n_components)
        X_train_final = pd.DataFrame(pca.fit_transform(X_train_scaled), columns=[f'PC{i+1}' for i in range(n_components)])
        X_val_final = pd.DataFrame(pca.transform(X_val_scaled), columns=[f'PC{i+1}' for i in range(n_components)])
        st.dataframe(X_train_final.head())
    else:
        X_train_final = X_train.copy()
        X_val_final = X_val.copy()

    # === Model Definitions ===
    models = []

    # Logistic Regression
    with st.expander("📊 Logistic Regression"):
        C = st.slider("C (inverse regularization)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C)
        models.append(("Logistic Regression", model))

    # Ridge
    with st.expander("📊 Ridge Classifier"):
        alpha = st.slider("Alpha (regularization strength)", 0.01, 10.0, 1.0)
        model = RidgeClassifier(alpha=alpha)
        models.append(("Ridge", model))

    # Lasso (via LogisticRegression with L1 penalty)
    with st.expander("📊 Lasso"):
        C = st.slider("C (inverse regularization)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C, penalty='l1', solver='liblinear')
        models.append(("Lasso", model))

    # Elastic Net
    with st.expander("📊 Elastic Net"):
        C = st.slider("C", 0.01, 10.0, 1.0)
        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
        model = LogisticRegression(C=C, penalty='elasticnet', solver='saga', l1_ratio=l1_ratio, max_iter=10000)
        models.append(("Elastic Net", model))

    # PLS-DA
    with st.expander("📊 PLS-DA"):
        n_components = st.slider("Number of PLS components", 1, min(10, X_train_final.shape[1]), 2)
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train_final, y_train)
        y_pred_pls = (pls.predict(X_train_final) > 0.5).astype(int).flatten()
        models.append(("PLS-DA", pls))

    # SVM
    with st.expander("📊 Support Vector Machine"):
        C = st.slider("C", 0.01, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf"])
        model = SVC(C=C, kernel=kernel, probability=True)
        models.append(("SVM", model))

    # Decision Tree
    with st.expander("🌳 Decision Tree"):
        max_depth = st.slider("Max depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)
        models.append(("Decision Tree", model))

    # Random Forest
    with st.expander("🌳 Random Forest"):
        n_estimators = st.slider("Number of trees", 10, 200, 100)
        max_depth = st.slider("Max depth", 1, 20, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        models.append(("Random Forest", model))

    # Gradient Boosting
    with st.expander("🌿 Gradient Boosting"):
        n_estimators = st.slider("Number of estimators", 10, 200, 100)
        learning_rate = st.slider("Learning rate", 0.01, 1.0, 0.1)
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        models.append(("GBM", model))

    # Neural Network
    with st.expander("🧠 Neural Network"):
        hidden_layer_sizes = st.slider("Hidden layer size", 1, 100, 50)
        alpha = st.slider("Regularization (alpha)", 0.0001, 0.1, 0.001)
        model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), alpha=alpha, max_iter=1000)
        models.append(("Neural Net", model))

    # === Final Metrics on Validation Set ===
    st.subheader("📊 Final Validation Set Comparison")
    summary_data = []
    for name, model in models:
        model.fit(X_train_final, y_train)
        if name == "PLS-DA":
            y_pred = (model.predict(X_val_final) > 0.5).astype(int).flatten()
            y_proba = model.predict(X_val_final)
        else:
            y_pred = model.predict(X_val_final)
            y_proba = model.predict_proba(X_val_final)[:, 1] if hasattr(model, "predict_proba") else None

        try:
            auc = roc_auc_score(y_val, y_proba) if y_proba is not None else np.nan
        except:
            auc = np.nan

        summary_data.append({
            "Model": name,
            "Accuracy": accuracy_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "Recall": recall_score(y_val, y_pred, zero_division=0),
            "F1 Score": f1_score(y_val, y_pred, zero_division=0),
            "AUC": auc
        })

    metrics_summary = pd.DataFrame(summary_data)
    st.dataframe(metrics_summary.style.format("{:.2f}"))

else:
    st.warning("📂 Please upload a CSV, Excel, or JSON file to proceed.")
