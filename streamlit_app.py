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


##########################################################################################################################
######################################    Presentation   #################################################################
##########################################################################################################################

st.title("🤖 Binary Classification Appppppp")
st.info("This app builds a binary classification model!")


##########################################################################################################################
######################################    File Upload    #################################################################
##########################################################################################################################

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

##########################################################################################################################
#################################        Dataset Overview    #############################################################
##########################################################################################################################


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


##########################################################################################################################
################################       PCA Step    #######################################################################
##########################################################################################################################

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

##########################################################################################################################
###########################    Machine Learning Methods for Binary Classification     ####################################
##########################################################################################################################

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

    # === Ridge Logistic Regression ===
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )

    with st.expander("🧱 Ridge Logistic Regression (L2)"):
        st.write("**Hyperparameters**")
        ridge_C = st.slider("Ridge: Regularization strength (C)", 0.01, 10.0, 1.0)
        ridge_max_iter = st.slider("Ridge: Max iterations", 100, 2000, 1000)

        ridge_model = LogisticRegression(
            penalty='l2',
            C=ridge_C,
            solver='lbfgs',
            max_iter=ridge_max_iter,
            random_state=42
        )
        ridge_model.fit(X_train_final, y_train)

        y_pred_ridge_train = ridge_model.predict(X_train_final)
        y_prob_ridge_train = ridge_model.predict_proba(X_train_final)[:, 1]

        st.markdown("**📊 Training Set Performance**")
        st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_ridge_train):.4f}")
        st.text(f"Precision: {precision_score(y_train, y_pred_ridge_train):.4f}")
        st.text(f"Recall:    {recall_score(y_train, y_pred_ridge_train):.4f}")
        st.text(f"F1-Score:  {f1_score(y_train, y_pred_ridge_train):.4f}")
        st.text(f"AUC:       {roc_auc_score(y_train, y_prob_ridge_train):.4f}")


    # === Lasso Logistic Regression ===
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )

    with st.expander("🧊 Lasso Logistic Regression (L1)"):
        st.write("**Hyperparameters**")
        lasso_C = st.slider("Lasso: Regularization strength (C)", 0.01, 10.0, 1.0)
        lasso_max_iter = st.slider("Lasso: Max iterations", 100, 2000, 1000)

        lasso_model = LogisticRegression(
            penalty='l1',
            C=lasso_C,
            solver='liblinear',  # 'liblinear' supports L1
            max_iter=lasso_max_iter,
            random_state=42
        )
        lasso_model.fit(X_train_final, y_train)

        # Training performance
        y_pred_lasso_train = lasso_model.predict(X_train_final)
        y_prob_lasso_train = lasso_model.predict_proba(X_train_final)[:, 1]

        st.markdown("**📊 Training Set Performance**")
        st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_lasso_train):.4f}")
        st.text(f"Precision: {precision_score(y_train, y_pred_lasso_train):.4f}")
        st.text(f"Recall:    {recall_score(y_train, y_pred_lasso_train):.4f}")
        st.text(f"F1-Score:  {f1_score(y_train, y_pred_lasso_train):.4f}")
        st.text(f"AUC:       {roc_auc_score(y_train, y_prob_lasso_train):.4f}")

    # === Elastic Net Logistic Regression ===
    with st.expander("🧬 Elastic Net Logistic Regression"):
        st.write("**Hyperparameters**")
        enet_C = st.slider("Elastic Net: Regularization strength (C)", 0.01, 10.0, 1.0)
        enet_max_iter = st.slider("Elastic Net: Max iterations", 100, 2000, 1000)
        enet_l1_ratio = st.slider("Elastic Net: L1 Ratio (0=L2, 1=L1)", 0.0, 1.0, 0.5)

        enet_model = LogisticRegression(
            penalty='elasticnet',
            C=enet_C,
            l1_ratio=enet_l1_ratio,
            solver='saga',  # 'saga' supports elasticnet
            max_iter=enet_max_iter,
            random_state=42
        )
        enet_model.fit(X_train_final, y_train)

        # Training performance
        y_pred_enet_train = enet_model.predict(X_train_final)
        y_prob_enet_train = enet_model.predict_proba(X_train_final)[:, 1]

        st.markdown("**📊 Training Set Performance**")
        st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_enet_train):.4f}")
        st.text(f"Precision: {precision_score(y_train, y_pred_enet_train):.4f}")
        st.text(f"Recall:    {recall_score(y_train, y_pred_enet_train):.4f}")
        st.text(f"F1-Score:  {f1_score(y_train, y_pred_enet_train):.4f}")
        st.text(f"AUC:       {roc_auc_score(y_train, y_prob_enet_train):.4f}")

    # === Partial Least Squares Discriminant Analysis (PLS-DA) ===
    from sklearn.cross_decomposition import PLSRegression

    with st.expander("🧪 Partial Least Squares Discriminant Analysis (PLS-DA)"):
        pls_n_components = st.slider("PLS-DA: Number of Components", 1, min(X_train_final.shape[1], 10), 2)

        pls_model = PLSRegression(n_components=pls_n_components)
        pls_model.fit(X_train_final, y_train)

        # Training performance
        y_scores_train_pls = pls_model.predict(X_train_final).ravel()
        y_pred_train_pls = (y_scores_train_pls >= 0.5).astype(int)

        st.markdown("**📊 Training Set Performance**")
        st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_train_pls):.4f}")
        st.text(f"Precision: {precision_score(y_train, y_pred_train_pls):.4f}")
        st.text(f"Recall:    {recall_score(y_train, y_pred_train_pls):.4f}")
        st.text(f"F1-Score:  {f1_score(y_train, y_pred_train_pls):.4f}")
        st.text(f"AUC:       {roc_auc_score(y_train, y_scores_train_pls):.4f}")




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


##########################################################################################################################
######################################             Validation             ################################################
##########################################################################################################################

    # === Validation Metrics Summary ===
    st.subheader("📊 Final Validation Set Comparison (Full Metrics)")

    # Get predictions and probabilities for all models
    y_val_pred_lr = lr_model.predict(X_val_final)
    y_val_prob_lr = lr_model.predict_proba(X_val_final)[:, 1]

    y_val_pred_rf = rf_model.predict(X_val_final)
    y_val_prob_rf = rf_model.predict_proba(X_val_final)[:, 1]

    y_val_pred_ridge = ridge_model.predict(X_val_final)
    y_val_prob_ridge = ridge_model.predict_proba(X_val_final)[:, 1]

    y_val_pred_lasso = lasso_model.predict(X_val_final)
    y_val_prob_lasso = lasso_model.predict_proba(X_val_final)[:, 1]

    y_val_pred_enet = enet_model.predict(X_val_final)
    y_val_prob_enet = enet_model.predict_proba(X_val_final)[:, 1]

    y_val_scores_pls = pls_model.predict(X_val_final).ravel()
    y_val_pred_pls = (y_val_scores_pls >= 0.5).astype(int)

    # Helper function to compute metrics
    def compute_metrics(y_true, y_pred, y_prob, model_name):
        return {
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_prob)
        }

    # Collect metrics for each model
    metrics = [
        compute_metrics(y_val, y_val_pred_lr, y_val_prob_lr, "Logistic Regression"),
        compute_metrics(y_val, y_val_pred_rf, y_val_prob_rf, "Random Forest"),
        compute_metrics(y_val, y_val_pred_ridge, y_val_prob_ridge, "Ridge Logistic Regression"),
        compute_metrics(y_val, y_val_pred_lasso, y_val_prob_lasso, "Lasso Logistic Regression"),
        compute_metrics(y_val, y_val_pred_enet, y_val_prob_enet, "Elastic Net Logistic Regression"),
        compute_metrics(y_val, y_val_pred_pls, y_val_scores_pls, "PLS-DA")
    ]

    # Create DataFrame and display
    summary_df = pd.DataFrame(metrics)
    st.dataframe(summary_df.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}", "AUC": "{:.4f}"}))




##########################################################################################################################
######################################         Final Test File             ###############################################
##########################################################################################################################



    
    # === Final Test File Upload and Prediction ===
    st.markdown("## 🔍 Apply Models to New Test Data")

    test_file = st.file_uploader("Upload a test dataset (same structure as training data):", key="test_file")

    if test_file is not None:
        try:
            if test_file.name.endswith(".csv"):
                df_test = pd.read_csv(test_file)
            elif test_file.name.endswith((".xlsx", ".xls")):
                df_test = pd.read_excel(test_file)
            elif test_file.name.endswith(".json"):
                df_test = pd.read_json(test_file)
            else:
                st.error("Unsupported file type.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading test file: {e}")
            st.stop()

        st.success("✅ Test file loaded successfully.")
        st.dataframe(df_test.head())

        # Preserve target column if it exists (for reference in download)
        target_column_present = target_column in df_test.columns
        if target_column_present:
            df_test_target = df_test[[target_column]].copy()
            df_test = df_test.drop(columns=[target_column])


        try:
            if use_pca == "Yes":
                df_test_scaled = scaler.transform(df_test.select_dtypes(include=np.number))
                df_test_transformed = pd.DataFrame(pca.transform(df_test_scaled), columns=[f'PC{i+1}' for i in range(n_components)])
            else:
                df_test_transformed = df_test.copy()

            # Make Predictions
            test_pred_lr = lr_model.predict(df_test_transformed)
            test_pred_rf = rf_model.predict(df_test_transformed)
            test_pred_ridge = ridge_model.predict(df_test_transformed)
            test_pred_lasso = lasso_model.predict(df_test_transformed)
            test_pred_enet = enet_model.predict(df_test_transformed)
            test_scores_pls = pls_model.predict(df_test_transformed).ravel()


            # Prediction Probabilities
            prob_pred_lr = lr_model.predict_proba(df_test_transformed)[:, 1]
            prob_pred_rf = rf_model.predict_proba(df_test_transformed)[:, 1]
            prob_pred_ridge = ridge_model.predict_proba(df_test_transformed)[:, 1]
            prob_pred_lasso = lasso_model.predict_proba(df_test_transformed)[:, 1]
            prob_pred_enet = enet_model.predict_proba(df_test_transformed)[:, 1]
            test_pred_pls = (test_scores_pls >= 0.5).astype(int)



            # Combine predictions
            # Combine predictions (and reattach target column if present)
            df_results = df_test.copy()

            if target_column_present:
                df_results[target_column] = df_test_target

            df_results["Logistic_Prediction"] = test_pred_lr
            df_results["Logistic_Prob"] = prob_pred_lr

            df_results["RandomForest_Prediction"] = test_pred_rf
            df_results["RandomForest_Prob"] = prob_pred_rf

            df_results["Ridge_Prediction"] = test_pred_ridge
            df_results["Ridge_Prob"] = prob_pred_ridge

            df_results["Lasso_Prediction"] = test_pred_lasso
            df_results["Lasso_Prob"] = prob_pred_lasso

            df_results["ElasticNet_Prediction"] = test_pred_enet
            df_results["ElasticNet_Prob"] = prob_pred_enet

            df_results["PLSDA_Prediction"] = test_pred_pls
            df_results["PLSDA_Prob"] = test_scores_pls


            st.markdown("### 📄 Predictions on Uploaded Test Data")
            st.dataframe(df_results)

            # Download link
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="classified_results.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.warning("📂 Please upload a CSV, Excel, or JSON file to proceed.")
