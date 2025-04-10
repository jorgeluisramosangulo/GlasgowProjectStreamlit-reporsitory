import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

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

    # === Train/Validation Split ===
    test_size_percent = st.slider("Select validation set size (%)", 10, 50, 20, 5)
    test_size = test_size_percent / 100.0
    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=test_size, random_state=42, shuffle=True)

    # === PCA Step ===
    use_pca = st.radio("Would you like to apply PCA?", ["No", "Yes"])
    if use_pca == "Yes":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=np.number))
        X_val_scaled = scaler.transform(X_val.select_dtypes(include=np.number))

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

    # === Logistic Regression ===
    with st.expander("📊 Logistic Regression"):
        st.write("**Hyperparameters**")
        C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
        max_iter = st.slider("Max iterations", 100, 1000, 100)

        lr_model = LogisticRegression(C=C, max_iter=max_iter)
        lr_model.fit(X_train_final, y_train)

        y_pred_lr = lr_model.predict(X_train_final)
        st.text("Classification Report (Training Set):")
        st.text(classification_report(y_train, y_pred_lr))

    # === Random Forest ===
    with st.expander("🌳 Random Forest"):
        st.write("**Hyperparameters**")
        n_estimators = st.slider("Number of trees", 10, 200, 100)
        max_depth = st.slider("Max depth", 1, 20, 5)

        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_train_final, y_train)

        y_pred_rf = rf_model.predict(X_train_final)
        st.text("Classification Report (Training Set):")
        st.text(classification_report(y_train, y_pred_rf))

    # === Validation Metrics Summary ===
    st.subheader("📊 Final Validation Set Comparison")
    y_val_pred_lr = lr_model.predict(X_val_final)
    y_val_pred_rf = rf_model.predict(X_val_final)

    acc_lr = accuracy_score(y_val, y_val_pred_lr)
    acc_rf = accuracy_score(y_val, y_val_pred_rf)

    summary_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy on Validation Set': [acc_lr, acc_rf]
    })
    st.dataframe(summary_df)

else:
    st.warning("📂 Please upload a CSV, Excel, or JSON file to proceed.")