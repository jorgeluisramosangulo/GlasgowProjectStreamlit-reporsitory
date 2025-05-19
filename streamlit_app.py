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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


##########################################################################################################################
######################################    Presentation   #################################################################
##########################################################################################################################

st.title("🤖 Binary Classification App")

st.markdown("""
**Author:** Jorge Ramos  
**Student ID:** 2599173  
**Project:** MSc Data Analytics – Binary Classification Dashboard  
""")

st.info("This app builds a binary classification model using 10 different machine learning different.")



##########################################################################################################################
######################################    File Upload    #################################################################
##########################################################################################################################

# === File Upload ===
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "xlsx":
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




    if "columns_confirmed" not in st.session_state:
        st.session_state["columns_confirmed"] = False

    # === Column Selection ===
    st.markdown("### 📌 Step 1: Select Columns to Include")

    selected_columns = st.multiselect(
        "Select the columns you want to use (you can leave out irrelevant or ID columns):",
        options=df.columns.tolist(),
        default=st.session_state.get("selected_columns", df.columns.tolist())
    )

    confirm_columns = st.button("✅ Confirm Column Selection")

    if confirm_columns and selected_columns:
        st.session_state["selected_columns"] = selected_columns
        st.session_state["columns_confirmed"] = True
        st.rerun()  # <== force rerun so next step shows

    if not st.session_state["columns_confirmed"]:
        st.info("👈 Please confirm column selection to continue.")
        st.stop()

    df = df[st.session_state["selected_columns"]]  # apply confirmed selection






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
    st.markdown("### 🎯 Step 2: Select Target Column")

    if "target_confirmed" not in st.session_state:
        st.session_state["target_confirmed"] = False
    if "target_column" not in st.session_state:
        st.session_state["target_column"] = None

    # UI
    target_column_input = st.selectbox("Select the target column:", df.columns)

    if st.button("✅ Confirm Target Selection"):
        st.session_state["target_column"] = target_column_input
        st.session_state["target_confirmed"] = True
        st.rerun()

    if not st.session_state["target_confirmed"]:
        st.info("👈 Please confirm target column to continue.")
        st.stop()

    # ✅ Use confirmed value
    target_column = st.session_state["target_column"]
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]



    # After confirmation, split data
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]

    # Optional: store in session_state
    st.session_state["target_column"] = target_column

    st.success(f"✅ Target column confirmed: `{target_column}`")


    # Convert target to integer labels
    y_raw = pd.factorize(y_raw)[0].astype('int64')  # Guarantees int64

    # Handle categorical features (one-hot encoding)
    X_encoded = pd.get_dummies(X_raw, drop_first=True)

    # Ensure float64 type
    X_encoded = X_encoded.astype('float64')



    # Final cleaned features
    X_raw = X_encoded

    # Check for and remove rows with missing or infinite values
    X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    invalid_rows = X_raw.isnull().any(axis=1)
    if invalid_rows.any():
        st.warning(f"⚠️ Removed {invalid_rows.sum()} rows with NaNs or infinite values.")
        X_raw = X_raw[~invalid_rows]
        y_raw = y_raw[~invalid_rows]

    # === Train/Validation Split ===
    test_size_percent = st.slider("Select validation set size (%)", 10, 50, 20, 5)
    test_size = test_size_percent / 100.0
    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=test_size, random_state=42, shuffle=True)


##########################################################################################################################
################################       PCA Step    #######################################################################
##########################################################################################################################

    # === Step 3: PCA Selection ===
    st.markdown("### 🧬 Step 3: PCA Dimensionality Reduction")

    # Ensure PCA confirmation state exists
    if "pca_confirmed" not in st.session_state:
        st.session_state["pca_confirmed"] = False
    if "use_pca" not in st.session_state:
        st.session_state["use_pca"] = "No"

    # Let user choose
    use_pca_input = st.radio("Would you like to apply PCA?", ["No", "Yes"], index=0)

    # Confirm PCA selection
    confirm_pca = st.button("✅ Confirm PCA Selection")

    # Save decision and rerun
    if confirm_pca:
        st.session_state["use_pca"] = use_pca_input
        st.session_state["pca_confirmed"] = True
        st.rerun()

    # Wait until confirmed
    if not st.session_state["pca_confirmed"]:
        st.info("👈 Please confirm PCA selection to continue.")
        st.stop()

    # === PCA logic ===
    use_pca = st.session_state["use_pca"]
    if use_pca == "Yes":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=np.number))
        X_val_scaled = scaler.transform(X_val.select_dtypes(include=np.number))

        # Fit PCA
        pca = PCA()
        pca.fit(X_train_scaled)

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
        ax.set_title("Cumulative Explained Variance")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance")
        st.pyplot(fig)

        # Component slider
        n_components = st.slider("Select number of principal components to keep", 1, X_train_scaled.shape[1], 2)
        pca = PCA(n_components=n_components)

        # Final transformation
        X_train_final = pd.DataFrame(pca.fit_transform(X_train_scaled), columns=[f'PC{i+1}' for i in range(n_components)])
        X_val_final = pd.DataFrame(pca.transform(X_val_scaled), columns=[f'PC{i+1}' for i in range(n_components)])

        # Show and store
        st.success(f"✅ PCA applied with {n_components} components.")
        st.dataframe(X_train_final.head())

        st.session_state["pca"] = pca
        st.session_state["scaler"] = scaler
        st.session_state["n_components"] = n_components

    else:
        X_train_final = X_train.copy()
        X_val_final = X_val.copy()
        st.success("✅ PCA skipped.")



##########################################################################################################################
###########################    Machine Learning Methods for Binary Classification     ####################################
##########################################################################################################################

    # === Step 4: Model Selection ===
    st.markdown("### 🧠 Step 4: Select ML Models to Train")

    # Initialize session state if not already done
    if "models_confirmed" not in st.session_state:
        st.session_state["models_confirmed"] = False
    if "selected_models" not in st.session_state:
        st.session_state["selected_models"] = []

    # Let user select models
    model_selection_input = st.multiselect(
        "Select models to include:",
        options=[
            "Logistic Regression",
            "Ridge Logistic Regression",
            "Lasso Logistic Regression",
            "ElasticNet Logistic Regression",
            "Random Forest",
            "Decision Tree",
            "Support Vector Machine",
            "Gradient Boosting",
            "PLS-DA",
            "Neural Network"
        ],
        default=st.session_state.get("selected_models", []) or []
    )

    # Confirm selection
    confirm_models = st.button("✅ Confirm Model Selection")

    # On confirmation, store and rerun
    if confirm_models:
        if not model_selection_input:
            st.warning("⚠️ Please select at least one model to continue.")
            st.stop()

        st.session_state["selected_models"] = model_selection_input
        st.session_state["models_confirmed"] = True
        st.rerun()

    # Wait until confirmed
    if not st.session_state["models_confirmed"]:
        st.info("👈 Please confirm your model selection to continue.")
        st.stop()

    # Use confirmed models
    selected_models = st.session_state["selected_models"]
    st.success(f"✅ {len(selected_models)} model(s) selected and confirmed.")





    # === Logistic Regression ===
    if "Logistic Regression" in selected_models:
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
    if "Ridge Logistic Regression" in selected_models:
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
    if "Lasso Logistic Regression" in selected_models:
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
    if "ElasticNet Logistic Regression" in selected_models:
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
    if "PLS-DA" in selected_models:
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

    # === Support Vector Machine (SVM) ===
    if "Support Vector Machine" in selected_models:
        from sklearn.svm import SVC

        with st.expander("🔲 Support Vector Machine (SVM)"):
            st.write("**Hyperparameters**")
            svm_kernel = st.selectbox("SVM: Kernel", ['linear', 'rbf', 'poly', 'sigmoid'], index=1)
            svm_C = st.slider("SVM: Regularization parameter (C)", 0.01, 10.0, 1.0)
            svm_gamma = st.selectbox("SVM: Gamma", ['scale', 'auto'])

            svm_model = SVC(
                C=svm_C,
                kernel=svm_kernel,
                gamma=svm_gamma,
                probability=True,
                random_state=42
            )
            svm_model.fit(X_train_final, y_train)

            y_pred_svm_train = svm_model.predict(X_train_final)
            y_prob_svm_train = svm_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_svm_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_svm_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_svm_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_svm_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_svm_train):.4f}")



    # === Decision Tree Classifier ===
    if "Decision Tree" in selected_models:
        from sklearn.tree import DecisionTreeClassifier

        with st.expander("🌲 Decision Tree"):
            st.write("**Hyperparameters**")
            tree_max_depth = st.slider("Decision Tree: Max Depth", 1, 20, 5)
            tree_min_samples_split = st.slider("Decision Tree: Min Samples Split", 2, 20, 2)
            tree_min_samples_leaf = st.slider("Decision Tree: Min Samples Leaf", 1, 20, 1)
            tree_criterion = st.selectbox("Decision Tree: Criterion", ['gini', 'entropy'])

            tree_model = DecisionTreeClassifier(
                max_depth=tree_max_depth,
                min_samples_split=tree_min_samples_split,
                min_samples_leaf=tree_min_samples_leaf,
                criterion=tree_criterion,
                random_state=42
            )
            tree_model.fit(X_train_final, y_train)

            y_pred_tree_train = tree_model.predict(X_train_final)
            y_prob_tree_train = tree_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_tree_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_tree_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_tree_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_tree_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_tree_train):.4f}")



    # === Random Forest ===
    if "Random Forest" in selected_models:
        with st.expander("🌳 Random Forest"):
            st.write("**Hyperparameters**")
            n_estimators = st.slider("Number of trees", 10, 200, 100)
            max_depth = st.slider("Max depth", 1, 20, 5)

            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf_model.fit(X_train_final, y_train)

            y_pred_rf = rf_model.predict(X_train_final)
            st.text("Classification Report (Training Set):")
            st.text(classification_report(y_train, y_pred_rf))






    # === Gradient Boosting Machine (GBM) ===
    if "Gradient Boosting" in selected_models:
        with st.expander("🚀 Gradient Boosting Machine (GBM)"):
            st.write("**Hyperparameters**")
            gbm_n_estimators = st.slider("GBM: Number of Estimators", 10, 500, 100)
            gbm_learning_rate = st.slider("GBM: Learning Rate", 0.01, 1.0, 0.1, step=0.01)
            gbm_max_depth = st.slider("GBM: Max Depth", 1, 10, 3)
            gbm_subsample = st.slider("GBM: Subsample", 0.1, 1.0, 1.0, step=0.1)
            gbm_min_samples_split = st.slider("GBM: Min Samples Split", 2, 20, 2)
            gbm_min_samples_leaf = st.slider("GBM: Min Samples Leaf", 1, 20, 1)

            gbm_model = GradientBoostingClassifier(
                n_estimators=gbm_n_estimators,
                learning_rate=gbm_learning_rate,
                max_depth=gbm_max_depth,
                subsample=gbm_subsample,
                min_samples_split=gbm_min_samples_split,
                min_samples_leaf=gbm_min_samples_leaf,
                random_state=42
            )
            gbm_model.fit(X_train_final, y_train)

            y_pred_gbm_train = gbm_model.predict(X_train_final)
            y_prob_gbm_train = gbm_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_gbm_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_gbm_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_gbm_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_gbm_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_gbm_train):.4f}")



    # === Neural Network (MLPClassifier) ===
    if "Neural Network" in selected_models:
        with st.expander("🧠 Neural Network (MLPClassifier)"):
            st.write("**Hyperparameters**")
            
            nn_hidden_units = st.number_input("NN: Units in Hidden Layer", min_value=1, max_value=500, value=50)
            nn_activation = st.selectbox("NN: Activation Function", ['relu', 'logistic', 'tanh'])
            nn_solver = st.selectbox("NN: Solver", ['adam', 'sgd', 'lbfgs'])
            nn_alpha = st.number_input("NN: L2 Penalty (alpha)", value=0.0001, format="%.5f")
            nn_learning_rate_init = st.number_input("NN: Initial Learning Rate", value=0.001, format="%.5f")
            nn_max_iter = st.slider("NN: Max Iterations", 100, 2000, 1000)

            nn_model = MLPClassifier(
                hidden_layer_sizes=(nn_hidden_units,),
                activation=nn_activation,
                solver=nn_solver,
                alpha=nn_alpha,
                learning_rate_init=nn_learning_rate_init,
                max_iter=nn_max_iter,
                random_state=42
            )
            nn_model.fit(X_train_final, y_train)

            y_pred_nn_train = nn_model.predict(X_train_final)
            y_prob_nn_train = nn_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_nn_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_nn_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_nn_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_nn_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_nn_train):.4f}")




##########################################################################################################################
######################################             Validation             ################################################
##########################################################################################################################



    st.subheader("📊 Final Validation Set Comparison (Full Metrics)")

    # Retrieve selected models
    selected_models = st.session_state.get("selected_models", [])

    # Define dictionary to store predictions and probabilities
    val_predictions = {}

    # === Conditional Predictions by Model ===
    if "Logistic Regression" in selected_models:
        y_val_pred_lr = lr_model.predict(X_val_final)
        y_val_prob_lr = lr_model.predict_proba(X_val_final)[:, 1]
        val_predictions["Logistic Regression"] = (y_val_pred_lr, y_val_prob_lr)

    if "Random Forest" in selected_models:
        y_val_pred_rf = rf_model.predict(X_val_final)
        y_val_prob_rf = rf_model.predict_proba(X_val_final)[:, 1]
        val_predictions["Random Forest"] = (y_val_pred_rf, y_val_prob_rf)

    if "Ridge Logistic Regression" in selected_models:
        y_val_pred_ridge = ridge_model.predict(X_val_final)
        y_val_prob_ridge = ridge_model.predict_proba(X_val_final)[:, 1]
        val_predictions["Ridge Logistic Regression"] = (y_val_pred_ridge, y_val_prob_ridge)

    if "Lasso Logistic Regression" in selected_models:
        y_val_pred_lasso = lasso_model.predict(X_val_final)
        y_val_prob_lasso = lasso_model.predict_proba(X_val_final)[:, 1]
        val_predictions["Lasso Logistic Regression"] = (y_val_pred_lasso, y_val_prob_lasso)

    if "ElasticNet Logistic Regression" in selected_models:
        y_val_pred_enet = enet_model.predict(X_val_final)
        y_val_prob_enet = enet_model.predict_proba(X_val_final)[:, 1]
        val_predictions["Elastic Net Logistic Regression"] = (y_val_pred_enet, y_val_prob_enet)

    if "PLS-DA" in selected_models:
        y_val_scores_pls = pls_model.predict(X_val_final).ravel()
        y_val_pred_pls = (y_val_scores_pls >= 0.5).astype(int)
        val_predictions["PLS-DA"] = (y_val_pred_pls, y_val_scores_pls)

    if "Support Vector Machine" in selected_models:
        y_val_pred_svm = svm_model.predict(X_val_final)
        y_val_prob_svm = svm_model.predict_proba(X_val_final)[:, 1]
        val_predictions["Support Vector Machine"] = (y_val_pred_svm, y_val_prob_svm)

    if "Decision Tree" in selected_models:
        y_val_pred_tree = tree_model.predict(X_val_final)
        y_val_prob_tree = tree_model.predict_proba(X_val_final)[:, 1]
        val_predictions["Decision Tree"] = (y_val_pred_tree, y_val_prob_tree)

    if "Gradient Boosting" in selected_models:
        y_val_pred_gbm = gbm_model.predict(X_val_final)
        y_val_prob_gbm = gbm_model.predict_proba(X_val_final)[:, 1]
        val_predictions["Gradient Boosting"] = (y_val_pred_gbm, y_val_prob_gbm)

    if "Neural Network" in selected_models:
        y_val_pred_nn = nn_model.predict(X_val_final)
        y_val_prob_nn = nn_model.predict_proba(X_val_final)[:, 1]
        val_predictions["Neural Network"] = (y_val_pred_nn, y_val_prob_nn)

    # === Helper to compute metrics ===
    def compute_metrics(y_true, y_pred, y_prob, model_name):
        return {
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_prob)
        }

    # === Build metrics list dynamically ===
    metrics = []
    for model_name, (y_pred, y_prob) in val_predictions.items():
        metrics.append(compute_metrics(y_val, y_pred, y_prob, model_name))

    # === Display ===
    summary_df = pd.DataFrame(metrics)
    st.dataframe(summary_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}",
        "Recall": "{:.4f}", "F1-Score": "{:.4f}", "AUC": "{:.4f}"
    }))





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
        elif test_file.name.endswith((".xlsx")):
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

    # Preserve original file
    df_test_original = df_test.copy()


    # Use the same columns as in training
    expected_columns = st.session_state.get("selected_columns")

    if expected_columns is None:
        st.error("Training columns not found. Please upload and process a training file first.")
        st.stop()

    # Check which expected columns are missing in test
    missing_test_cols = set(expected_columns) - set(df_test.columns)
    if missing_test_cols:
        st.warning(f"⚠️ These expected columns are missing in the test set: {missing_test_cols}")

    # Filter test set to only selected columns (fill missing with 0s)
    df_test = df_test.reindex(columns=expected_columns, fill_value=0)


    # Preserve target column if present
    target_column_present = target_column in df_test.columns
    if target_column_present:
        df_test_target = df_test[[target_column]].copy()
        df_test = df_test.drop(columns=[target_column])

    try:
        # One-hot encode using same structure as training data
        df_test_encoded = pd.get_dummies(df_test, drop_first=True)

        # Align with training columns (fill missing with 0)
        missing_cols = set(X_raw.columns) - set(df_test_encoded.columns)
        for col in missing_cols:
            df_test_encoded[col] = 0

        # Reorder columns to match training set
        df_test_encoded = df_test_encoded[X_raw.columns]

        
        # Ensure float64
        df_test_encoded = df_test_encoded.astype('float64')

        # Check for and remove rows with missing or infinite values
        df_test_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
        invalid_test_rows = df_test_encoded.isnull().any(axis=1)
        if invalid_test_rows.any():
            st.warning(f"⚠️ Removed {invalid_test_rows.sum()} rows with NaNs or infinite values in test data.")
            df_test_encoded = df_test_encoded[~invalid_test_rows]
            df_test_original = df_test_original[~invalid_test_rows]
            if target_column_present:
                df_test_target = df_test_target[~invalid_test_rows]



        # PCA transform if selected
        if use_pca == "Yes":
            df_test_scaled = scaler.transform(df_test_encoded)
            df_test_transformed = pd.DataFrame(
                pca.transform(df_test_scaled),
                columns=[f"PC{i+1}" for i in range(n_components)]
            )
        else:
            df_test_transformed = df_test_encoded.copy()

        # === Retrieve selected models ===
        selected_models = st.session_state.get("selected_models", [])

        # === Initialize results DataFrame ===
        df_results = df_test_original.copy()
        if target_column_present:
            df_results[target_column] = df_test_target

        # === Make Predictions and Add Columns Dynamically ===

        if "Logistic Regression" in selected_models:
            test_pred_lr = lr_model.predict(df_test_transformed)
            prob_pred_lr = lr_model.predict_proba(df_test_transformed)[:, 1]
            df_results["Logistic_Prediction"] = test_pred_lr
            df_results["Logistic_Prob"] = prob_pred_lr

        if "Random Forest" in selected_models:
            test_pred_rf = rf_model.predict(df_test_transformed)
            prob_pred_rf = rf_model.predict_proba(df_test_transformed)[:, 1]
            df_results["RandomForest_Prediction"] = test_pred_rf
            df_results["RandomForest_Prob"] = prob_pred_rf

        if "Ridge Logistic Regression" in selected_models:
            test_pred_ridge = ridge_model.predict(df_test_transformed)
            prob_pred_ridge = ridge_model.predict_proba(df_test_transformed)[:, 1]
            df_results["Ridge_Prediction"] = test_pred_ridge
            df_results["Ridge_Prob"] = prob_pred_ridge

        if "Lasso Logistic Regression" in selected_models:
            test_pred_lasso = lasso_model.predict(df_test_transformed)
            prob_pred_lasso = lasso_model.predict_proba(df_test_transformed)[:, 1]
            df_results["Lasso_Prediction"] = test_pred_lasso
            df_results["Lasso_Prob"] = prob_pred_lasso

        if "ElasticNet Logistic Regression" in selected_models:
            test_pred_enet = enet_model.predict(df_test_transformed)
            prob_pred_enet = enet_model.predict_proba(df_test_transformed)[:, 1]
            df_results["ElasticNet_Prediction"] = test_pred_enet
            df_results["ElasticNet_Prob"] = prob_pred_enet

        if "PLS-DA" in selected_models:
            test_scores_pls = pls_model.predict(df_test_transformed).ravel()
            test_pred_pls = (test_scores_pls >= 0.5).astype(int)
            df_results["PLSDA_Prediction"] = test_pred_pls
            df_results["PLSDA_Prob"] = test_scores_pls

        if "Support Vector Machine" in selected_models:
            test_pred_svm = svm_model.predict(df_test_transformed)
            prob_pred_svm = svm_model.predict_proba(df_test_transformed)[:, 1]
            df_results["SVM_Prediction"] = test_pred_svm
            df_results["SVM_Prob"] = prob_pred_svm

        if "Decision Tree" in selected_models:
            test_pred_tree = tree_model.predict(df_test_transformed)
            prob_pred_tree = tree_model.predict_proba(df_test_transformed)[:, 1]
            df_results["DecisionTree_Prediction"] = test_pred_tree
            df_results["DecisionTree_Prob"] = prob_pred_tree

        if "Gradient Boosting" in selected_models:
            test_pred_gbm = gbm_model.predict(df_test_transformed)
            prob_pred_gbm = gbm_model.predict_proba(df_test_transformed)[:, 1]
            df_results["GBM_Prediction"] = test_pred_gbm
            df_results["GBM_Prob"] = prob_pred_gbm

        if "Neural Network" in selected_models:
            test_pred_nn = nn_model.predict(df_test_transformed)
            prob_pred_nn = nn_model.predict_proba(df_test_transformed)[:, 1]
            df_results["NN_Prediction"] = test_pred_nn
            df_results["NN_Prob"] = prob_pred_nn


        # === Show and Download ===
        st.markdown("### 📄 Predictions on Uploaded Test Data")
        st.dataframe(df_results)

        # Select file format
        file_format = st.selectbox("Select file format for download:", ["CSV", "JSON"])

        # Generate downloadable data based on selection
        if file_format == "CSV":
            file_data = df_results.to_csv(index=False).encode("utf-8")
            file_name = "classified_results.csv"
            mime_type = "text/csv"

        elif file_format == "JSON":
            file_data = df_results.to_json(orient="records", indent=2).encode("utf-8")
            file_name = "classified_results.json"
            mime_type = "application/json"

        # Download button
        st.download_button(
            label=f"📥 Download Predictions as {file_format}",
            data=file_data,
            file_name=file_name,
            mime=mime_type,
        )


    except Exception as e:
        st.error(f"Error during prediction: {e}")
