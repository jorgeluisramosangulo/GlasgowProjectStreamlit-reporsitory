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
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns



##########################################################################################################################
######################################    Presentation   #################################################################
##########################################################################################################################

st.title("🤖 Binary Classification Appppppppppppppppppppp")

st.markdown("""
**Author:** Jorge Ramos  
**Student ID:** 2599173  
**Project:** MSc Data Analytics – Binary Classification Dashboard  
""")

st.info("This app trains up to 12 machine learning models for datasets with a binary target (0 or 1).")



##########################################################################################################################
######################################    File Upload    #################################################################
##########################################################################################################################

st.markdown("### 📂 Choose a sample dataset or upload your own")

use_sample = st.radio("How would you like to provide your dataset?", ["Use sample dataset", "Upload your own file"])
df = None

if use_sample == "Use sample dataset":
    dataset_names = ["titanic", "heart_disease", "breast cancer", "creditcard", "diabetes", "banknote"]
    format_options = ["csv", "xlsx", "json"]
    stage_options = ["train"]  # Only show train here

    dataset = st.selectbox("Select dataset", dataset_names)
    file_format = st.radio("File format", format_options, horizontal=True)

    filepath = f"Datasets/{file_format}/{dataset} train.{file_format}"

    try:
        if file_format == "csv":
            df = pd.read_csv(filepath)
        elif file_format == "xlsx":
            df = pd.read_excel(filepath)
        elif file_format == "json":
            df = pd.read_json(filepath)
        st.success(f"✅ Loaded sample: {dataset} ({file_format})")
    except Exception as e:
        st.error(f"❌ Could not load sample dataset: {e}")
        st.stop()

else:
    uploaded_file = st.file_uploader("Upload your data file (more than 40 rows recommended)", type=["csv", "xlsx", "json"])

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

# === Show preview if any data is loaded ===
if df is not None:
    st.markdown("### 🔍 Preview of Loaded Data")
    st.dataframe(df.head())




##########################################################################################################################
######################################    Delete Columns    ##############################################################
##########################################################################################################################


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
    
    # === Missing Values Summary ===
    st.markdown("#### ❗ Missing Values per Column")

    # Calculate missing values and percentages
    missing_counts = df.isnull().sum()
    missing_percent = 100 * df.isnull().mean()

    # Filter only columns with missing values
    missing_df = pd.DataFrame({
        "Column": missing_counts.index,
        "Missing Values": missing_counts.values,
        "% Missing": missing_percent.values
    }).query("`Missing Values` > 0").sort_values(by="Missing Values", ascending=False)

    # Display table
    if not missing_df.empty:
        st.dataframe(missing_df.style.format({"% Missing": "{:.2f}%"}))
        
        # Optional bar chart visualization
        if st.checkbox("📉 Show Missing Values Bar Chart"):
            st.bar_chart(missing_df.set_index("Column")["% Missing"])
    else:
        st.success("✅ No missing values in the dataset.")


##########################################################################################################################
#################################        Target Selection    #############################################################
##########################################################################################################################

    
    # === Target Selection ===
    st.markdown("### 🎯 Step 2: Select Target Column")

    if "target_confirmed" not in st.session_state:
        st.session_state["target_confirmed"] = False
    if "target_column" not in st.session_state:
        st.session_state["target_column"] = None

    # UI
    target_column_input = st.selectbox("Select the target column:", df.columns)

    # Confirm button
    if st.button("✅ Confirm Target Selection"):
        st.session_state["target_column"] = target_column_input
        st.session_state["target_confirmed"] = True
        st.rerun()

    # Require confirmation before continuing
    if not st.session_state["target_confirmed"]:
        st.info("👈 Please confirm target column to continue.")
        st.stop()

    # ✅ Use confirmed value
    target_column = st.session_state["target_column"]
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]

    # === Validate target column ===
    unique_vals = y_raw.dropna().unique()
    if len(unique_vals) != 2 or not set(unique_vals).issubset({0, 1}):
        st.error("❌ Target column must contain exactly two values: 0 and 1.")
        st.stop()

    # Save label mapping for test-time consistency (optional here but safe)
    st.session_state["label_classes_"] = sorted(unique_vals)

    # Show class distribution
    st.markdown("#### 📊 Target Value Distribution")

    target_counts = pd.Series(y_raw).value_counts().sort_index()
    target_percents = round(target_counts / len(y_raw) * 100, 2)

    target_summary_df = pd.DataFrame({
        "Class": target_counts.index,
        "Count": target_counts.values,
        "Percentage": target_percents.values
    })

    st.dataframe(target_summary_df)




##########################################################################################################################
#################################        Data Visualization    ###########################################################
##########################################################################################################################


    # === Optional: Data Visualization ===
    st.markdown("### 📊 Optional: Data Visualization")

    enable_vis = st.checkbox("🔍 Enable Data Visualization?", value=False)

    if enable_vis:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        
        st.markdown("#### 🎨 Visualization Options")

        # Histogram Matrix (Target Legend)
        if st.checkbox("1️⃣ Histogram Matrix (Target Legend)", value=False):
            hist_cols = st.multiselect("Select columns to include", numeric_cols, default=numeric_cols[:3], key="hist_target")
            if len(hist_cols) >= 2:
                fig = sns.pairplot(df, vars=hist_cols, hue=target_column, kind="hist")
                st.pyplot(fig)

        # Histogram Matrix (Custom Legend)
        if st.checkbox("2️⃣ Histogram Matrix (Custom Legend)", value=False):
            hist_cols_custom = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:3], key="hist_custom_cols")
            legend_col = st.selectbox("Choose categorical column for legend", categorical_cols, key="hist_custom_legend")
            if len(hist_cols_custom) >= 2:
                fig = sns.pairplot(df, vars=hist_cols_custom, hue=legend_col, kind="hist")
                st.pyplot(fig)

        # Scatter Matrix (Target)
        if st.checkbox("3️⃣.1 Scatter Matrix (Target Legend)", value=False):
            scatter_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:3], key="scat_target")
            if len(scatter_cols) >= 2:
                fig = sns.pairplot(df, vars=scatter_cols, hue=target_column)
                st.pyplot(fig)

        # Scatter Matrix (Custom Legend)
        if st.checkbox("3️⃣.2 Scatter Matrix (Custom Legend)", value=False):
            scatter_cols_custom = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:3], key="scat_custom_cols")
            legend_col_scat = st.selectbox("Legend column", categorical_cols, key="scat_custom_leg")
            if len(scatter_cols_custom) >= 2:
                fig = sns.pairplot(df, vars=scatter_cols_custom, hue=legend_col_scat)
                st.pyplot(fig)

        # Scatter Plot 2 Variables (Target)
        if st.checkbox("4️⃣.1 Scatter Plot (Target Legend)", value=False):
            xcol = st.selectbox("X Axis", numeric_cols, key="scatter_x_target")
            ycol = st.selectbox("Y Axis", numeric_cols, key="scatter_y_target")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=xcol, y=ycol, hue=target_column, ax=ax)
            st.pyplot(fig)

        # Scatter Plot 2 Variables (Custom Legend)
        if st.checkbox("4️⃣.2 Scatter Plot (Custom Legend)", value=False):
            xcol = st.selectbox("X Axis", numeric_cols, key="scatter_x_custom")
            ycol = st.selectbox("Y Axis", numeric_cols, key="scatter_y_custom")
            legend_col = st.selectbox("Legend Column", categorical_cols, key="scatter_leg_custom")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=xcol, y=ycol, hue=legend_col, ax=ax)
            st.pyplot(fig)

        # Correlation (2 vars, Target Filter)
        if st.checkbox("5️⃣.1 Correlation of Two Variables (Target Filter)", value=False):
            xcol = st.selectbox("X Column", numeric_cols, key="corr_x1")
            ycol = st.selectbox("Y Column", numeric_cols, key="corr_y1")
            filter_opt = st.radio("Filter By Target?", ["No Filter", "Target = 0", "Target = 1"], key="corr_filter1")
            df_corr = df.copy()
            if filter_opt == "Target = 0":
                df_corr = df[df[target_column] == 0]
            elif filter_opt == "Target = 1":
                df_corr = df[df[target_column] == 1]
            corr_val = df_corr[[xcol, ycol]].corr().iloc[0, 1]
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_corr, x=xcol, y=ycol, ax=ax)
            ax.set_title(f"Correlation: {corr_val:.2f}")
            st.pyplot(fig)

        # Correlation (2 vars, Custom Filter)
        if st.checkbox("5️⃣.2 Correlation of Two Variables (Custom Category Filter)", value=False):
            xcol = st.selectbox("X Column", numeric_cols, key="corr_x2")
            ycol = st.selectbox("Y Column", numeric_cols, key="corr_y2")
            cat_filter = st.selectbox("Category Column", categorical_cols, key="corr_cat_col")
            cat_val = st.selectbox("Filter Category", df[cat_filter].unique(), key="corr_cat_val")
            df_corr = df[df[cat_filter] == cat_val]
            corr_val = df_corr[[xcol, ycol]].corr().iloc[0, 1]
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_corr, x=xcol, y=ycol, ax=ax)
            ax.set_title(f"Correlation (filtered): {corr_val:.2f}")
            st.pyplot(fig)

        # Correlation Matrix of Selected Variables
        if st.checkbox("6️⃣ Correlation Matrix (All Combinations)", value=False):
            matrix_cols = st.multiselect("Choose numeric columns", numeric_cols, default=numeric_cols[:5], key="matrix_cols")
            filter_opt = st.radio("Filter?", ["No Filter", "Target = 0", "Target = 1"], key="matrix_filter")
            df_filt = df.copy()
            if filter_opt == "Target = 0":
                df_filt = df[df[target_column] == 0]
            elif filter_opt == "Target = 1":
                df_filt = df[df[target_column] == 1]
            if len(matrix_cols) >= 2:
                corr_matrix = df_filt[matrix_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Matrix")
                st.pyplot(fig)

        # === Summary Statistics & Box Plot for a Selected Column ===
        if st.checkbox("📦 Summary Stats + Box Plot for One Column", value=False):
            selected_column = st.selectbox("Select a numeric column:", numeric_cols, key="summary_col")

            if selected_column:
                st.markdown(f"### 📊 Summary Statistics for `{selected_column}`")

                stats = df[selected_column].describe().to_frame().T
                stats["median"] = df[selected_column].median()
                stats = stats.rename(columns={
                    "count": "Count",
                    "mean": "Mean",
                    "std": "Std Dev",
                    "min": "Min",
                    "25%": "25th Percentile",
                    "50%": "50th Percentile",
                    "75%": "75th Percentile",
                    "max": "Max"
                })
                st.dataframe(stats)

                st.markdown("### 📈 Box Plot")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[selected_column], ax=ax)
                ax.set_title(f"Box Plot of {selected_column}")
                st.pyplot(fig)


##########################################################################################################################
################################       Feature Importance    #############################################################
########################################################################################################################## 

    # === Feature Importance ===
    st.markdown("### 🧠 Optional: Explore Feature Importance")

    # Preprocess (basic encoding)
    X_encoded = pd.get_dummies(X_raw, drop_first=True).astype('float64')

    # Let user select importance method
    method = st.selectbox(
        "Choose importance method:",
        ["Random Forest", "Lasso (L1)", "Permutation (RF)", "Mutual Info"],
        index=0
    )

    # Train/Test split (just for internal importance calc)
    X_train_imp, X_val_imp, y_train_imp, y_val_imp = train_test_split(
        X_encoded, y_raw, test_size=0.2, random_state=42
    )

    # Compute importances
    importance_values = None
    model_used = None

    if method == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model_used = RandomForestClassifier(random_state=42)
        model_used.fit(X_train_imp, y_train_imp)
        importance_values = model_used.feature_importances_

    elif method == "Lasso (L1)":
        from sklearn.linear_model import LogisticRegression
        model_used = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        model_used.fit(X_train_imp, y_train_imp)
        importance_values = np.abs(model_used.coef_[0])

    elif method == "Permutation (RF)":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train_imp, y_train_imp)
        result = permutation_importance(rf, X_val_imp, y_val_imp, n_repeats=10, random_state=42)
        importance_values = result.importances_mean

    elif method == "Mutual Info":
        from sklearn.feature_selection import mutual_info_classif
        importance_values = mutual_info_classif(X_train_imp, y_train_imp, random_state=42)

    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": X_encoded.columns,
        "Importance": importance_values
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    st.dataframe(importance_df.style.format({"Importance": "{:.4f}"}))

    # Optional bar chart
    if st.checkbox("📊 Show Top 10 Features as Bar Chart"):
        st.bar_chart(importance_df.set_index("Feature").head(10))



##########################################################################################################################
#################################        Split Data Train and Validate    ################################################
##########################################################################################################################

    # After confirmation, split data
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]

    # Optional: store in session_state
    st.session_state["target_column"] = target_column

    st.success(f"✅ Target column confirmed: `{target_column}`")

    # Convert target to integer labels   
    y_raw, label_classes = pd.factorize(y_raw)
    y_raw = y_raw.astype('int64')
    st.session_state["label_classes_"] = label_classes.tolist()


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

    # ✅ Store processed training columns for later use in test section
    st.session_state["X_raw"] = X_raw.copy()

    # === Train/Validation Split ===
    test_size_percent = st.slider("Select validation set size (%)", 10, 50, 20, 5)
    test_size = test_size_percent / 100.0
    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=test_size, random_state=42, shuffle=True)



##########################################################################################################################
#################################        Data Transformation (Before PCA)       ##########################################
##########################################################################################################################

    st.markdown("### 🔧 Step 2.5: Optional Data Transformation")

    # ✅ Always initialize fallback copies
    X_train_resampled = X_train.copy()
    y_train_resampled = y_train.copy()

    apply_transformation = st.checkbox("🧪 Would you like to transform the data before PCA?", value=False)

    if apply_transformation:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler

        st.info("Transformations will apply to both training and validation sets consistently.")

        # Always start with current train/val from earlier step
        X_train_resampled = X_train.copy()
        y_train_resampled = y_train.copy()

        # 1. Centering + Scaling
        if st.checkbox("1️⃣ Centering + Scaling (MinMaxScaler)", value=False):
            col_to_scale = st.selectbox("Select column to scale", X_train.columns, key="scale_col")
            scaler = MinMaxScaler()
            X_train[col_to_scale] = scaler.fit_transform(X_train[[col_to_scale]])
            X_val[col_to_scale] = scaler.transform(X_val[[col_to_scale]])

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(X_raw[col_to_scale], ax=ax[0], kde=True).set(title="Before Scaling")
            sns.histplot(X_train[col_to_scale], ax=ax[1], kde=True).set(title="After Scaling")
            st.pyplot(fig)

        # 2. Standardization
        if st.checkbox("2️⃣ Standardization (Zero Mean, Unit Variance)", value=False):
            col_to_standardize = st.selectbox("Select column to standardize", X_train.columns, key="standardize_col")
            std_scaler = StandardScaler()
            X_train[col_to_standardize] = std_scaler.fit_transform(X_train[[col_to_standardize]])
            X_val[col_to_standardize] = std_scaler.transform(X_val[[col_to_standardize]])

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(X_raw[col_to_standardize], ax=ax[0], kde=True).set(title="Before Standardization")
            sns.histplot(X_train[col_to_standardize], ax=ax[1], kde=True).set(title="After Standardization")
            st.pyplot(fig)

        # 3. Create New Features
        if st.checkbox("3️⃣ Create New Variable", value=False):
            operation = st.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide", "Log", "Square"], key="op")
            col1 = st.selectbox("Select first column", X_train.columns, key="new_col1")
            col2 = st.selectbox("Select second column (if applicable)", X_train.columns, key="new_col2")

            if operation == "Add":
                new_col = X_train[col1] + X_train[col2]
            elif operation == "Subtract":
                new_col = X_train[col1] - X_train[col2]
            elif operation == "Multiply":
                new_col = X_train[col1] * X_train[col2]
            elif operation == "Divide":
                new_col = X_train[col1] / (X_train[col2] + 1e-9)
            elif operation == "Log":
                new_col = np.log1p(X_train[col1])
            elif operation == "Square":
                new_col = X_train[col1] ** 2

            new_col_name = st.text_input("New column name", value=f"{col1}_{operation}_{col2 if operation in ['Add', 'Subtract', 'Multiply', 'Divide'] else ''}")

            if st.button("➕ Add New Feature"):
                X_train[new_col_name] = new_col
                X_val[new_col_name] = new_col  # replicate same transformation
                st.success(f"Feature '{new_col_name}' added to both train and validation sets.")

        # 4. Remove or Fix Outliers
        if st.checkbox("4️⃣ Outlier Detection & Removal", value=False):
            outlier_col = st.selectbox("Select column", X_train.columns, key="outlier_col")
            method = st.selectbox("Outlier Method", ["IQR (1.5x)"], key="outlier_method")

            col_data = X_train[outlier_col]
            q1, q3 = col_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            outliers_train = ((col_data < lower) | (col_data > upper))
            st.write(f"Outliers detected in training set: {outliers_train.sum()} / {len(col_data)}")

            fig, ax = plt.subplots()
            sns.boxplot(x=col_data, ax=ax)
            ax.set_title("Boxplot with IQR Thresholds")
            st.pyplot(fig)

            if st.button("🧹 Remove Outliers (Train Only)"):
                keep_indices = ~outliers_train
                X_train = X_train[keep_indices]
                y_train = y_train[keep_indices]
                st.success("Outliers removed from training set.")

        # 5. Optional Class Imbalance Handling (ALWAYS visible if apply_transformation)
        st.markdown("### ⚖️ Optional: Handle Class Imbalance (Train Set Only)")

        imbalance_strategy = st.radio(
            "Choose a resampling method:",
            ["None", "Undersampling", "Oversampling", "SMOTE"],
            index=0
        )

        if imbalance_strategy == "Undersampling":
            sampler = RandomUnderSampler(random_state=42)
        elif imbalance_strategy == "Oversampling":
            sampler = RandomOverSampler(random_state=42)
        elif imbalance_strategy == "SMOTE":
            sampler = SMOTE(random_state=42)
        else:
            sampler = None

        if sampler:
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            st.success(f"✅ {imbalance_strategy} applied. New class distribution:")
            st.dataframe(pd.Series(y_train_resampled).value_counts().rename("Count"))

    # Save results (resampled or not)
    st.session_state["X_train"] = X_train_resampled.copy()
    st.session_state["y_train"] = y_train_resampled.copy()
    st.session_state["X_val"] = X_val.copy()
    st.session_state["y_val"] = y_val.copy()




##########################################################################################################################
################################       PCA Step    #######################################################################
##########################################################################################################################

    # === Step 3: PCA Selection ===
    st.markdown("### 🧬 Step 3: PCA Dimensionality Reduction")

    # Initialize session state
    if "pca_confirmed" not in st.session_state:
        st.session_state["pca_confirmed"] = False
    if "pca_ready" not in st.session_state:
        st.session_state["pca_ready"] = False
    if "use_pca" not in st.session_state:
        st.session_state["use_pca"] = "No"
    if "n_components_slider" not in st.session_state:
        st.session_state["n_components_slider"] = 2

    # === Step 3.1: Ask user if PCA should be used ===
    use_pca_input = st.radio("Would you like to apply PCA?", ["No", "Yes"], index=0)

    if st.button("✅ Confirm PCA Selection"):
        st.session_state["use_pca"] = use_pca_input
        st.session_state["pca_confirmed"] = True
        st.session_state["pca_ready"] = False  # Reset PCA confirmation
        st.rerun()

    if not st.session_state["pca_confirmed"]:
        st.info("👈 Please confirm PCA selection to continue.")
        st.stop()

    # === Step 3.2: Apply PCA ===
    use_pca = st.session_state["use_pca"]
    if use_pca == "Yes":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(st.session_state["X_train"].select_dtypes(include=np.number))
        X_val_scaled = scaler.transform(st.session_state["X_val"].select_dtypes(include=np.number))


        # Fit PCA to show variance plot
        pca_temp = PCA()
        pca_temp.fit(X_train_scaled)

        # Show explained variance plot
        cum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
        ax.set_title("Cumulative Explained Variance")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance")
        st.pyplot(fig)

            # === Show PCA component loadings (for interpretation) ===
        loadings = pd.DataFrame(
            pca_temp.components_.T,
            index=st.session_state["X_train"].select_dtypes(include=np.number).columns,
            columns=[f"PC{i+1}" for i in range(pca_temp.n_components_)]
        )

        st.markdown("### 📊 PCA Loadings: How Original Features Contribute to Each Principal Component")
        st.dataframe(loadings.round(4))


        # Select number of components without triggering computation
        st.session_state["n_components_slider"] = st.slider(
            "Select number of principal components to keep",
            1,
            X_train_scaled.shape[1],
            value=st.session_state["n_components_slider"]
        )

        # Button to apply PCA based on slider
        if st.button("✅ Confirm PCA Parameters"):
            n_components = st.session_state["n_components_slider"]
            final_pca = PCA(n_components=n_components)
            X_train_final = pd.DataFrame(final_pca.fit_transform(X_train_scaled), columns=[f'PC{i+1}' for i in range(n_components)])
            X_val_final = pd.DataFrame(final_pca.transform(X_val_scaled), columns=[f'PC{i+1}' for i in range(n_components)])

            # Store for later use
            st.session_state["pca"] = final_pca
            st.session_state["scaler"] = scaler
            st.session_state["n_components"] = n_components
            st.session_state["X_train_final"] = X_train_final
            st.session_state["X_val_final"] = X_val_final
            st.session_state["pca_ready"] = True
            st.success(f"✅ PCA applied with {n_components} components.")
            st.dataframe(X_train_final.head())

        if not st.session_state["pca_ready"]:
            st.info("👈 Please confirm number of components to apply PCA.")
            st.stop()
        else:
            X_train_final = st.session_state["X_train_final"]
            X_val_final = st.session_state["X_val_final"]

    else:
        X_train_final = st.session_state["X_train"].copy()
        X_val_final = st.session_state["X_val"].copy()
        st.success("✅ PCA skipped.")





##########################################################################################################################
###########################    Machine Learning Methods for Binary Classification     ####################################
##########################################################################################################################

    # === Step 4: Model Selection ===
    st.markdown("### 🧠 Step 4: Select ML Models to Train")

    # Initialize session state if not already set
    if "models_confirmed" not in st.session_state:
        st.session_state["models_confirmed"] = False
    if "selected_models" not in st.session_state:
        st.session_state["selected_models"] = []

    # Let user choose models
    model_selection_input = st.multiselect(
        "Select models to include:",
        options=[
            "Ridge Logistic Regression",
            "Lasso Logistic Regression",
            "ElasticNet Logistic Regression",
            "Random Forest",
            "Decision Tree",
            "Support Vector Machine",
            "Gradient Boosting",
            "PLS-DA",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Neural Network",
            "Voting Classifier"
        ],
        default=st.session_state["selected_models"]
    )

    # Button to confirm selection
    if st.button("✅ Confirm Model Selection"):

        if len(model_selection_input) < 4:
            st.warning("⚠️ Please select at least four models to continue.")
            st.stop()

        st.session_state["selected_models"] = model_selection_input
        st.session_state["models_confirmed"] = True
        st.rerun()

    # Halt here unless confirmed
    if not st.session_state["models_confirmed"]:
        st.info("👈 Please confirm model selection to continue.")
        st.stop()

    # Get confirmed list
    selected_models = st.session_state["selected_models"]
    st.success(f"✅ {len(selected_models)} model(s) selected and confirmed.")

    # === Train Models ===


    # === Ridge Logistic Regression (with CV + Tuning) ===
    if "Ridge Logistic Regression" in selected_models:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, make_scorer
        )

        with st.expander("🧱 Ridge Logistic Regression (L2)"):
            st.write("**Hyperparameters**")

            # === Optional: Enable tuning ===
            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning (Grid or Random Search)?", key="ridge_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="ridge_search_method")

                ridge_max_iter = st.slider("Ridge: Max Iterations", 100, 2000, 1000, step=100, key="ridge_max_iter")

                c_range = st.slider("C Range (log scale)", 0.01, 10.0, (0.1, 5.0), step=0.1, key="ridge_c_range")
                param_grid = {"C": np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), num=10)}

                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="ridge_cv_folds")

                with st.spinner("Running hyperparameter tuning..."):
                    base_model = LogisticRegression(
                        penalty='l2', solver='lbfgs', max_iter=ridge_max_iter, random_state=42
                    )

                    if search_method == "Grid Search":
                        ridge_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                    else:
                        ridge_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                        cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                    ridge_search.fit(X_train_final, y_train)
                    ridge_model = ridge_search.best_estimator_

                    st.success(f"Best C: {ridge_model.C:.4f}")
            else:
                # Manual hyperparameters
                ridge_C = st.slider("Ridge: Regularization strength (C)", 0.01, 10.0, 1.0, key="ridge_C_manual")
                ridge_max_iter = st.slider("Ridge: Max iterations", 100, 2000, 1000, step=100, key="ridge_iter_manual")

                ridge_model = LogisticRegression(
                    penalty='l2',
                    C=ridge_C,
                    solver='lbfgs',
                    max_iter=ridge_max_iter,
                    random_state=42
                )
                ridge_model.fit(X_train_final, y_train)

            # === Training performance on full training set ===
            y_pred_ridge_train = ridge_model.predict(X_train_final)
            y_prob_ridge_train = ridge_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_ridge_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_ridge_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_ridge_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_ridge_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_ridge_train):.4f}")

            # === Optional: 10-Fold Cross-Validation ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for Ridge?", key="ridge_run_cv"):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                with st.spinner("Running cross-validation..."):
                    cv_results = cross_validate(
                        ridge_model, X_train_final, y_train,
                        cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                    )

                st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                for metric in scoring:
                    mean_score = cv_results[f'test_{metric}'].mean()
                    std_score = cv_results[f'test_{metric}'].std()
                    st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")



    # === Lasso Logistic Regression (with CV + Tuning) ===
    if "Lasso Logistic Regression" in selected_models:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        with st.expander("🧊 Lasso Logistic Regression (L1)"):
            st.write("**Hyperparameters**")

            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning (Grid or Random Search)?", key="lasso_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="lasso_search_method")

                lasso_max_iter = st.slider("Lasso: Max Iterations", 100, 2000, 1000, step=100, key="lasso_max_iter")

                c_range = st.slider("C Range (log scale)", 0.01, 10.0, (0.1, 5.0), step=0.1, key="lasso_c_range")
                param_grid = {"C": np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), num=10)}

                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="lasso_cv_folds")

                with st.spinner("Running hyperparameter tuning..."):
                    base_model = LogisticRegression(
                        penalty='l1',
                        solver='liblinear',  # Required for L1
                        max_iter=lasso_max_iter,
                        random_state=42
                    )

                    if search_method == "Grid Search":
                        lasso_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                    else:
                        lasso_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                        cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                    lasso_search.fit(X_train_final, y_train)
                    lasso_model = lasso_search.best_estimator_

                    st.success(f"Best C: {lasso_model.C:.4f}")

            else:
                lasso_C = st.slider("Lasso: Regularization strength (C)", 0.01, 10.0, 1.0, key="lasso_C_manual")
                lasso_max_iter = st.slider("Lasso: Max iterations", 100, 2000, 1000, step=100, key="lasso_iter_manual")

                with st.spinner("Training Lasso Logistic Regression..."):
                    lasso_model = LogisticRegression(
                        penalty='l1',
                        C=lasso_C,
                        solver='liblinear',
                        max_iter=lasso_max_iter,
                        random_state=42
                    )
                    lasso_model.fit(X_train_final, y_train)

            # === Global metrics (always computed) ===
            y_pred_lasso_train = lasso_model.predict(X_train_final)
            y_prob_lasso_train = lasso_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_lasso_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_lasso_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_lasso_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_lasso_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_lasso_train):.4f}")

            # === Optional: 10-Fold Cross-Validation ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for Lasso?", key="lasso_run_cv"):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                with st.spinner("Running cross-validation..."):
                    cv_results = cross_validate(
                        lasso_model, X_train_final, y_train,
                        cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                    )

                st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                for metric in scoring:
                    mean_score = cv_results[f'test_{metric}'].mean()
                    std_score = cv_results[f'test_{metric}'].std()
                    st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")



    # === Elastic Net Logistic Regression (with CV + Tuning) ===
    if "ElasticNet Logistic Regression" in selected_models:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        with st.expander("🧬 Elastic Net Logistic Regression"):
            st.write("**Hyperparameters**")

            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning (Grid or Random Search)?", key="enet_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="enet_search_method")

                enet_max_iter = st.slider("Elastic Net: Max Iterations", 100, 2000, 1000, step=100, key="enet_max_iter")

                # Grid for C and l1_ratio
                c_range = st.slider("C Range (log scale)", 0.01, 10.0, (0.1, 5.0), step=0.1, key="enet_c_range")
                l1_range = st.slider("L1 Ratio Range", 0.0, 1.0, (0.2, 0.8), step=0.1, key="enet_l1_range")

                param_grid = {
                    "C": np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), num=5),
                    "l1_ratio": np.linspace(l1_range[0], l1_range[1], num=5)
                }

                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="enet_cv_folds")

                with st.spinner("Running hyperparameter tuning..."):
                    base_model = LogisticRegression(
                        penalty='elasticnet',
                        solver='saga',
                        max_iter=enet_max_iter,
                        random_state=42
                    )

                    if search_method == "Grid Search":
                        enet_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                    else:
                        enet_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                        cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                    enet_search.fit(X_train_final, y_train)
                    enet_model = enet_search.best_estimator_

                    st.success(f"Best C: {enet_model.C:.4f}, L1 Ratio: {enet_model.l1_ratio:.2f}")

            else:
                enet_C = st.slider("Elastic Net: Regularization strength (C)", 0.01, 10.0, 1.0, key="enet_C_manual")
                enet_max_iter = st.slider("Elastic Net: Max iterations", 100, 2000, 1000, step=100, key="enet_iter_manual")
                enet_l1_ratio = st.slider("Elastic Net: L1 Ratio (0=L2, 1=L1)", 0.0, 1.0, 0.5, step=0.01, key="enet_l1_ratio_manual")

                with st.spinner("Training Elastic Net Logistic Regression..."):
                    enet_model = LogisticRegression(
                        penalty='elasticnet',
                        C=enet_C,
                        l1_ratio=enet_l1_ratio,
                        solver='saga',
                        max_iter=enet_max_iter,
                        random_state=42
                    )
                    enet_model.fit(X_train_final, y_train)

            # === Global metrics (always computed) ===
            y_pred_enet_train = enet_model.predict(X_train_final)
            y_prob_enet_train = enet_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_enet_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_enet_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_enet_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_enet_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_enet_train):.4f}")

            # === Optional: 10-Fold Cross-Validation ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for Elastic Net?", key="enet_run_cv"):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                with st.spinner("Running cross-validation..."):
                    cv_results = cross_validate(
                        enet_model, X_train_final, y_train,
                        cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                    )

                st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                for metric in scoring:
                    mean_score = cv_results[f'test_{metric}'].mean()
                    std_score = cv_results[f'test_{metric}'].std()
                    st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")



    # === Partial Least Squares Discriminant Analysis (PLS-DA) ===
    if "PLS-DA" in selected_models:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_validate
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, make_scorer
        )

        with st.expander("🧪 Partial Least Squares Discriminant Analysis (PLS-DA)"):
            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for n_components?", key="pls_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="pls_search_method")

                max_comp_limit = min(X_train_final.shape[1], 10)
                n_components_range = st.slider(
                    "Range of Components to Search",
                    1, max_comp_limit, (2, max_comp_limit),
                    key="pls_comp_range"
                )

                comp_grid = {"n_components": list(range(n_components_range[0], n_components_range[1] + 1))}
                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="pls_cv_folds")

                with st.spinner("Running PLS-DA hyperparameter tuning..."):
                    pls_base = PLSRegression()
                    if search_method == "Grid Search":
                        pls_search = GridSearchCV(pls_base, comp_grid, cv=n_folds, scoring='r2')  # r2 as proxy
                    else:
                        pls_search = RandomizedSearchCV(pls_base, comp_grid, n_iter=5, cv=n_folds,
                                                        scoring='r2', random_state=42)

                    pls_search.fit(X_train_final, y_train)
                    pls_model = pls_search.best_estimator_
                    st.success(f"Best n_components: {pls_model.n_components}")

            else:
                pls_n_components = st.slider(
                    "PLS-DA: Number of Components",
                    1,
                    min(X_train_final.shape[1], 10),
                    2,
                    key="pls_n_components"
                )
                pls_model = PLSRegression(n_components=pls_n_components)
                pls_model.fit(X_train_final, y_train)

            # === Global metrics (always computed) ===
            y_scores_train_pls = pls_model.predict(X_train_final).ravel()
            y_pred_train_pls = (y_scores_train_pls >= 0.5).astype(int)

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_train_pls):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_train_pls):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_train_pls):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_train_pls):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_scores_train_pls):.4f}")

            # === Optional: 10-Fold Cross-Validation for PLS-DA ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for PLS-DA?", key="pls_run_cv"):
                with st.spinner("Running cross-validation..."):
                    # We'll use cross_val_predict to get scores for ROC AUC
                    y_scores_cv_pls = cross_val_predict(
                        pls_model, X_train_final, y_train,
                        cv=10, method="predict"
                    ).ravel()
                    y_pred_cv_pls = (y_scores_cv_pls >= 0.5).astype(int)

                    st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                    st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_cv_pls):.4f}")
                    st.text(f"Precision: {precision_score(y_train, y_pred_cv_pls):.4f}")
                    st.text(f"Recall:    {recall_score(y_train, y_pred_cv_pls):.4f}")
                    st.text(f"F1-Score:  {f1_score(y_train, y_pred_cv_pls):.4f}")
                    st.text(f"AUC:       {roc_auc_score(y_train, y_scores_cv_pls):.4f}")



    # === K-Nearest Neighbors (KNN) with CV + Tuning ===
    if "K-Nearest Neighbors" in selected_models:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        with st.expander("📍 K-Nearest Neighbors (KNN)"):
            st.write("**Hyperparameters**")

            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning?", key="knn_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="knn_search_method")

                n_range = st.slider("k Range", 1, 50, (3, 15), step=1, key="knn_k_range")
                weight_options = ["uniform", "distance"]
                metric_options = ["minkowski", "euclidean", "manhattan"]

                param_grid = {
                    "n_neighbors": list(range(n_range[0], n_range[1] + 1)),
                    "weights": weight_options,
                    "metric": metric_options
                }

                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="knn_cv_folds")

                with st.spinner("Running KNN hyperparameter tuning..."):
                    base_model = KNeighborsClassifier()
                    if search_method == "Grid Search":
                        knn_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                    else:
                        knn_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                        cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                    knn_search.fit(X_train_final, y_train)
                    knn_model = knn_search.best_estimator_

                    st.success(f"Best Parameters: k={knn_model.n_neighbors}, weights={knn_model.weights}, metric={knn_model.metric}")

            else:
                knn_n_neighbors = st.slider("KNN: Number of Neighbors (k)", min_value=1, max_value=50, value=5, key="knn_n_neighbors")
                knn_weights = st.selectbox("KNN: Weight Function", options=["uniform", "distance"], key="knn_weights")
                knn_metric = st.selectbox("KNN: Distance Metric", options=["minkowski", "euclidean", "manhattan"], key="knn_metric")

                with st.spinner("Training KNN..."):
                    knn_model = KNeighborsClassifier(
                        n_neighbors=knn_n_neighbors,
                        weights=knn_weights,
                        metric=knn_metric
                    )
                    knn_model.fit(X_train_final, y_train)

            # === Global metrics (always computed) ===
            y_pred_train_knn = knn_model.predict(X_train_final)
            y_prob_train_knn = knn_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_train_knn):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_train_knn):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_train_knn):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_train_knn):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_train_knn):.4f}")

            # === Optional: 10-Fold Cross-Validation ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for KNN?", key="knn_run_cv"):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                with st.spinner("Running cross-validation..."):
                    cv_results = cross_validate(
                        knn_model, X_train_final, y_train,
                        cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                    )

                st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                for metric in scoring:
                    mean_score = cv_results[f'test_{metric}'].mean()
                    std_score = cv_results[f'test_{metric}'].std()
                    st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")





    # === Naive Bayes (GaussianNB) with CV + Tuning ===
    if "Naive Bayes" in selected_models:
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        with st.expander("📦 Naive Bayes (GaussianNB)"):
            st.write("Naive Bayes assumes feature independence and models each feature using a normal distribution.")

            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for Naive Bayes?", key="nb_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="nb_search_method")

                # Log-space for numerical stability range
                smoothing_range = st.slider(
                    "Variance Smoothing Range (log scale)", -12, -2, (-9, -6),
                    key="nb_smoothing_range"
                )
                smoothing_values = np.logspace(smoothing_range[0], smoothing_range[1], num=5)
                param_grid = {"var_smoothing": smoothing_values}

                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="nb_cv_folds")

                with st.spinner("Running hyperparameter tuning for Naive Bayes..."):
                    base_model = GaussianNB()
                    if search_method == "Grid Search":
                        nb_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                    else:
                        nb_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=5,
                                                    cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                    nb_search.fit(X_train_final, y_train)
                    nb_model = nb_search.best_estimator_

                    st.success(f"Best var_smoothing: {nb_model.var_smoothing:.1e}")
            else:
                with st.spinner("Training Naive Bayes..."):
                    nb_model = GaussianNB()
                    nb_model.fit(X_train_final, y_train)

            # === Global metrics (always computed) ===
            y_pred_train_nb = nb_model.predict(X_train_final)
            y_prob_train_nb = nb_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_train_nb):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_train_nb):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_train_nb):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_train_nb):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_train_nb):.4f}")

            # === Optional: 10-Fold Cross-Validation ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for Naive Bayes?", key="nb_run_cv"):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                with st.spinner("Running cross-validation..."):
                    cv_results = cross_validate(
                        nb_model, X_train_final, y_train,
                        cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                    )

                st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                for metric in scoring:
                    mean_score = cv_results[f'test_{metric}'].mean()
                    std_score = cv_results[f'test_{metric}'].std()
                    st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")




    # === Support Vector Machine (SVM) with CV + Tuning ===
    if "Support Vector Machine" in selected_models:
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        with st.expander("🔲 Support Vector Machine (SVM)"):
            st.write("**Hyperparameters**")

            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for SVM?", key="svm_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="svm_search_method")

                c_range = st.slider("C Range", 0.01, 10.0, (0.1, 5.0), step=0.1, key="svm_c_range")
                kernel_options = ['linear', 'rbf', 'poly', 'sigmoid']
                gamma_options = ['scale', 'auto']

                param_grid = {
                    "C": np.linspace(c_range[0], c_range[1], 5),
                    "kernel": kernel_options,
                    "gamma": gamma_options
                }

                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="svm_cv_folds")

                with st.spinner("Running SVM hyperparameter tuning..."):
                    base_model = SVC(probability=True, random_state=42)

                    if search_method == "Grid Search":
                        svm_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                    else:
                        svm_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                        cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                    svm_search.fit(X_train_final, y_train)
                    svm_model = svm_search.best_estimator_

                    st.success(f"Best Parameters: C={svm_model.C}, kernel={svm_model.kernel}, gamma={svm_model.gamma}")

            else:
                svm_kernel = st.selectbox("SVM: Kernel", ['linear', 'rbf', 'poly', 'sigmoid'], index=1, key="svm_kernel")
                svm_C = st.slider("SVM: Regularization parameter (C)", 0.01, 10.0, 1.0, key="svm_C")
                svm_gamma = st.selectbox("SVM: Gamma", ['scale', 'auto'], key="svm_gamma")

                with st.spinner("Training Support Vector Machine..."):
                    svm_model = SVC(
                        C=svm_C,
                        kernel=svm_kernel,
                        gamma=svm_gamma,
                        probability=True,
                        random_state=42
                    )
                    svm_model.fit(X_train_final, y_train)

            # === Global metrics (always computed) ===
            y_pred_svm_train = svm_model.predict(X_train_final)
            y_prob_svm_train = svm_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_svm_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_svm_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_svm_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_svm_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_svm_train):.4f}")

            # === Optional: 10-Fold Cross-Validation ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for SVM?", key="svm_run_cv"):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                with st.spinner("Running cross-validation..."):
                    cv_results = cross_validate(
                        svm_model, X_train_final, y_train,
                        cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                    )

                st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                for metric in scoring:
                    mean_score = cv_results[f'test_{metric}'].mean()
                    std_score = cv_results[f'test_{metric}'].std()
                    st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")





    # === Decision Tree Classifier with CV + Tuning ===
    if "Decision Tree" in selected_models:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        with st.expander("🌲 Decision Tree"):
            st.write("**Hyperparameters**")

            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for Decision Tree?", key="tree_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="tree_search_method")

                depth_range = st.slider("Max Depth Range", 1, 20, (3, 10), key="tree_depth_range")
                split_range = st.slider("Min Samples Split Range", 2, 20, (2, 10), key="tree_split_range")
                leaf_range = st.slider("Min Samples Leaf Range", 1, 20, (1, 5), key="tree_leaf_range")
                criterion_options = ['gini', 'entropy']

                param_grid = {
                    "max_depth": list(range(depth_range[0], depth_range[1] + 1)),
                    "min_samples_split": list(range(split_range[0], split_range[1] + 1)),
                    "min_samples_leaf": list(range(leaf_range[0], leaf_range[1] + 1)),
                    "criterion": criterion_options
                }

                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="tree_cv_folds")

                with st.spinner("Running Decision Tree hyperparameter tuning..."):
                    base_model = DecisionTreeClassifier(random_state=42)

                    if search_method == "Grid Search":
                        tree_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                    else:
                        tree_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                        cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                    tree_search.fit(X_train_final, y_train)
                    tree_model = tree_search.best_estimator_

                    st.success(
                        f"Best Params: depth={tree_model.max_depth}, "
                        f"split={tree_model.min_samples_split}, leaf={tree_model.min_samples_leaf}, "
                        f"criterion={tree_model.criterion}"
                    )

            else:
                tree_max_depth = st.slider("Decision Tree: Max Depth", 1, 20, 5, key="tree_max_depth")
                tree_min_samples_split = st.slider("Decision Tree: Min Samples Split", 2, 20, 2, key="tree_min_samples_split")
                tree_min_samples_leaf = st.slider("Decision Tree: Min Samples Leaf", 1, 20, 1, key="tree_min_samples_leaf")
                tree_criterion = st.selectbox("Decision Tree: Criterion", ['gini', 'entropy'], key="tree_criterion")

                with st.spinner("Training Decision Tree..."):
                    tree_model = DecisionTreeClassifier(
                        max_depth=tree_max_depth,
                        min_samples_split=tree_min_samples_split,
                        min_samples_leaf=tree_min_samples_leaf,
                        criterion=tree_criterion,
                        random_state=42
                    )
                    tree_model.fit(X_train_final, y_train)

            # === Global metrics (always computed) ===
            y_pred_tree_train = tree_model.predict(X_train_final)
            y_prob_tree_train = tree_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_tree_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_tree_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_tree_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_tree_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_tree_train):.4f}")

            # === Optional: 10-Fold Cross-Validation ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for Decision Tree?", key="tree_run_cv"):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                with st.spinner("Running cross-validation..."):
                    cv_results = cross_validate(
                        tree_model, X_train_final, y_train,
                        cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                    )

                st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                for metric in scoring:
                    mean_score = cv_results[f'test_{metric}'].mean()
                    std_score = cv_results[f'test_{metric}'].std()
                    st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")




    # === Random Forest Classifier with CV + Tuning ===
    if "Random Forest" in selected_models:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        with st.expander("🌳 Random Forest"):
            st.write("**Hyperparameters**")

            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for Random Forest?", key="rf_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="rf_search_method")

                tree_range = st.slider("Number of Trees (n_estimators)", 10, 200, (50, 150), step=10, key="rf_n_range")
                depth_range = st.slider("Max Depth", 1, 20, (3, 10), key="rf_depth_range")
                leaf_range = st.slider("Min Samples Leaf", 1, 20, (1, 5), key="rf_leaf_range")

                param_grid = {
                    "n_estimators": list(range(tree_range[0], tree_range[1] + 1, 10)),
                    "max_depth": list(range(depth_range[0], depth_range[1] + 1, 1)),
                    "min_samples_leaf": list(range(leaf_range[0], leaf_range[1] + 1, 1))
                }

                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="rf_cv_folds")

                with st.spinner("Running Random Forest hyperparameter tuning..."):
                    base_model = RandomForestClassifier(random_state=42)

                    if search_method == "Grid Search":
                        rf_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                    else:
                        rf_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                    cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                    rf_search.fit(X_train_final, y_train)
                    rf_model = rf_search.best_estimator_

                    st.success(
                        f"Best Parameters: "
                        f"n_estimators={rf_model.n_estimators}, "
                        f"max_depth={rf_model.max_depth}, "
                        f"min_samples_leaf={rf_model.min_samples_leaf}"
                    )

            else:
                n_estimators = st.slider("Number of Trees", 10, 200, 100, key="rf_n_estimators")
                max_depth = st.slider("Max Depth", 1, 20, 5, key="rf_max_depth")

                with st.spinner("Training Random Forest..."):
                    rf_model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42
                    )
                    rf_model.fit(X_train_final, y_train)

            # === Global metrics (always computed) ===
            y_pred_rf = rf_model.predict(X_train_final)
            y_prob_rf = rf_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_rf):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_rf):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_rf):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_rf):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_rf):.4f}")

            # === Optional: 10-Fold Cross-Validation ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for Random Forest?", key="rf_run_cv"):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                with st.spinner("Running cross-validation..."):
                    cv_results = cross_validate(
                        rf_model, X_train_final, y_train,
                        cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                    )

                st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                for metric in scoring:
                    mean_score = cv_results[f'test_{metric}'].mean()
                    std_score = cv_results[f'test_{metric}'].std()
                    st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")



    # === Gradient Boosting Machine (GBM) with CV + Tuning ===
    if "Gradient Boosting" in selected_models:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        with st.expander("🚀 Gradient Boosting Machine (GBM)"):
            st.write("**Hyperparameters**")

            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for GBM?", key="gbm_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="gbm_search_method")

                n_estimators_range = st.slider("Number of Estimators", 10, 500, (50, 200), step=10, key="gbm_n_range")
                learning_rates = st.slider("Learning Rate Range", 0.01, 1.0, (0.05, 0.3), step=0.01, key="gbm_lr_range")
                max_depth_range = st.slider("Max Depth", 1, 10, (2, 6), key="gbm_depth_range")
                subsample_range = st.slider("Subsample Range", 0.1, 1.0, (0.5, 1.0), step=0.1, key="gbm_sub_range")
                split_range = st.slider("Min Samples Split", 2, 20, (2, 10), key="gbm_split_range")
                leaf_range = st.slider("Min Samples Leaf", 1, 20, (1, 5), key="gbm_leaf_range")

                param_grid = {
                    "n_estimators": list(range(n_estimators_range[0], n_estimators_range[1] + 1, 10)),
                    "learning_rate": np.linspace(learning_rates[0], learning_rates[1], 5),
                    "max_depth": list(range(max_depth_range[0], max_depth_range[1] + 1)),
                    "subsample": np.linspace(subsample_range[0], subsample_range[1], 5),
                    "min_samples_split": list(range(split_range[0], split_range[1] + 1)),
                    "min_samples_leaf": list(range(leaf_range[0], leaf_range[1] + 1))
                }

                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="gbm_cv_folds")

                with st.spinner("Running GBM hyperparameter tuning..."):
                    base_model = GradientBoostingClassifier(random_state=42)

                    if search_method == "Grid Search":
                        gbm_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                    else:
                        gbm_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                        cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                    gbm_search.fit(X_train_final, y_train)
                    gbm_model = gbm_search.best_estimator_

                    st.success("Best Parameters Selected via Tuning.")

            else:
                gbm_n_estimators = st.slider("GBM: Number of Estimators", 10, 500, 100, key="gbm_n_estimators")
                gbm_learning_rate = st.slider("GBM: Learning Rate", 0.01, 1.0, 0.1, step=0.01, key="gbm_learning_rate")
                gbm_max_depth = st.slider("GBM: Max Depth", 1, 10, 3, key="gbm_max_depth")
                gbm_subsample = st.slider("GBM: Subsample", 0.1, 1.0, 1.0, step=0.1, key="gbm_subsample")
                gbm_min_samples_split = st.slider("GBM: Min Samples Split", 2, 20, 2, key="gbm_min_samples_split")
                gbm_min_samples_leaf = st.slider("GBM: Min Samples Leaf", 1, 20, 1, key="gbm_min_samples_leaf")

                with st.spinner("Training Gradient Boosting Machine..."):
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

            # === Global metrics (always computed) ===
            y_pred_gbm_train = gbm_model.predict(X_train_final)
            y_prob_gbm_train = gbm_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_gbm_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_gbm_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_gbm_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_gbm_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_gbm_train):.4f}")

            # === Optional: 10-Fold Cross-Validation ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for GBM?", key="gbm_run_cv"):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                with st.spinner("Running cross-validation..."):
                    cv_results = cross_validate(
                        gbm_model, X_train_final, y_train,
                        cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                    )

                st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                for metric in scoring:
                    mean_score = cv_results[f'test_{metric}'].mean()
                    std_score = cv_results[f'test_{metric}'].std()
                    st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")





    # === Neural Network (MLPClassifier) with CV + Tuning ===
    if "Neural Network" in selected_models:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        with st.expander("🧠 Neural Network (MLPClassifier)"):
            st.write("**Hyperparameters**")

            enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for Neural Network?", key="nn_tuning")

            if enable_tuning:
                search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="nn_search_method")

                hidden_layer_options = [(32,), (64,), (128,), (64, 32), (128, 64)]
                activation_options = ['relu', 'logistic', 'tanh']
                alpha_range = st.slider("L2 Penalty (log scale)", -6, -1, (-4, -2), key="nn_alpha_range")
                lr_range = st.slider("Initial Learning Rate (log scale)", -5, -1, (-4, -2), key="nn_lr_range")
                max_iter = st.slider("Max Iterations", 100, 2000, 1000, key="nn_cv_max_iter")

                param_grid = {
                    "hidden_layer_sizes": hidden_layer_options,
                    "activation": activation_options,
                    "alpha": np.logspace(alpha_range[0], alpha_range[1], num=5),
                    "learning_rate_init": np.logspace(lr_range[0], lr_range[1], num=5),
                }

                n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="nn_cv_folds")

                with st.spinner("Running Neural Network hyperparameter tuning..."):
                    base_model = MLPClassifier(
                        solver='adam',  # consistent default
                        max_iter=max_iter,
                        random_state=42
                    )

                    if search_method == "Grid Search":
                        nn_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                    else:
                        nn_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                    cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                    nn_search.fit(X_train_final, y_train)
                    nn_model = nn_search.best_estimator_

                    st.success(
                        f"Best Parameters: layers={nn_model.hidden_layer_sizes}, "
                        f"activation={nn_model.activation}, "
                        f"alpha={nn_model.alpha:.5f}, lr={nn_model.learning_rate_init:.5f}"
                    )

            else:
                nn_hidden_units = st.number_input("NN: Units in Hidden Layer", min_value=1, max_value=500, value=50, key="nn_hidden_units")
                nn_activation = st.selectbox("NN: Activation Function", ['relu', 'logistic', 'tanh'], key="nn_activation")
                nn_solver = st.selectbox("NN: Solver", ['adam', 'sgd', 'lbfgs'], key="nn_solver")
                nn_alpha = st.number_input("NN: L2 Penalty (alpha)", value=0.0001, format="%.5f", key="nn_alpha")
                nn_learning_rate_init = st.number_input("NN: Initial Learning Rate", value=0.001, format="%.5f", key="nn_lr_init")
                nn_max_iter = st.slider("NN: Max Iterations", 100, 2000, 1000, key="nn_max_iter")

                with st.spinner("Training Neural Network..."):
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

            # === Global metrics (always computed) ===
            y_pred_nn_train = nn_model.predict(X_train_final)
            y_prob_nn_train = nn_model.predict_proba(X_train_final)[:, 1]

            st.markdown("**📊 Training Set Performance**")
            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_nn_train):.4f}")
            st.text(f"Precision: {precision_score(y_train, y_pred_nn_train):.4f}")
            st.text(f"Recall:    {recall_score(y_train, y_pred_nn_train):.4f}")
            st.text(f"F1-Score:  {f1_score(y_train, y_pred_nn_train):.4f}")
            st.text(f"AUC:       {roc_auc_score(y_train, y_prob_nn_train):.4f}")

            # === Optional: 10-Fold Cross-Validation ===
            if st.checkbox("🔁 Run 10-Fold Cross-Validation for Neural Network?", key="nn_run_cv"):
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                with st.spinner("Running cross-validation..."):
                    cv_results = cross_validate(
                        nn_model, X_train_final, y_train,
                        cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                    )

                st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                for metric in scoring:
                    mean_score = cv_results[f'test_{metric}'].mean()
                    std_score = cv_results[f'test_{metric}'].std()
                    st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")



    # === Voting Classifier (Soft Voting Only) with CV ===
    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )

    if "Voting Classifier" in selected_models:
        with st.expander("🗳️ Voting Classifier (Soft Voting Ensemble)"):
            st.write("**Soft voting averages predicted probabilities across models.**")
            st.write("All models included must support `predict_proba()`. At least 2 models required.")

            available_models = []
            model_names = []

            # === Add models if selected and defined ===
            if "Ridge Logistic Regression" in selected_models and 'ridge_model' in locals():
                available_models.append(("ridge", ridge_model))
                model_names.append("Ridge Logistic Regression")

            if "Lasso Logistic Regression" in selected_models and 'lasso_model' in locals():
                available_models.append(("lasso", lasso_model))
                model_names.append("Lasso Logistic Regression")

            if "ElasticNet Logistic Regression" in selected_models and 'enet_model' in locals():
                available_models.append(("elastic", enet_model))
                model_names.append("ElasticNet Logistic Regression")

            if "Random Forest" in selected_models and 'rf_model' in locals():
                available_models.append(("rf", rf_model))
                model_names.append("Random Forest")

            if "Decision Tree" in selected_models and 'tree_model' in locals():
                available_models.append(("dt", tree_model))
                model_names.append("Decision Tree")

            if "Support Vector Machine" in selected_models and 'svm_model' in locals():
                available_models.append(("svm", svm_model))
                model_names.append("Support Vector Machine")

            if "Gradient Boosting" in selected_models and 'gbm_model' in locals():
                available_models.append(("gb", gbm_model))
                model_names.append("Gradient Boosting")

            if "K-Nearest Neighbors" in selected_models and 'knn_model' in locals():
                available_models.append(("knn", knn_model))
                model_names.append("K-Nearest Neighbors")

            if "Naive Bayes" in selected_models and 'nb_model' in locals():
                available_models.append(("nb", nb_model))
                model_names.append("Naive Bayes")

            if "Neural Network" in selected_models and 'nn_model' in locals():
                available_models.append(("nn", nn_model))
                model_names.append("Neural Network")

            # === Train Voting Classifier if enough models are available ===
            if len(available_models) < 2:
                st.warning("Please select at least two trained models that support probability outputs.")
            else:
                with st.spinner("Training Voting Classifier (soft)..."):
                    voting_clf = VotingClassifier(
                        estimators=available_models,
                        voting="soft"
                    )
                    voting_clf.fit(X_train_final, y_train)

                    y_pred_vote_train = voting_clf.predict(X_train_final)
                    y_prob_vote_train = voting_clf.predict_proba(X_train_final)[:, 1]

                    st.markdown("**📊 Training Set Performance**")
                    st.text(f"Using Models: {', '.join(model_names)}")
                    st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_vote_train):.4f}")
                    st.text(f"Precision: {precision_score(y_train, y_pred_vote_train):.4f}")
                    st.text(f"Recall:    {recall_score(y_train, y_pred_vote_train):.4f}")
                    st.text(f"F1-Score:  {f1_score(y_train, y_pred_vote_train):.4f}")
                    st.text(f"AUC:       {roc_auc_score(y_train, y_prob_vote_train):.4f}")

                # === Optional: 10-Fold Cross-Validation ===
                if st.checkbox("🔁 Run 10-Fold Cross-Validation for Voting Classifier?", key="vote_run_cv"):
                    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                    with st.spinner("Running cross-validation..."):
                        cv_results = cross_validate(
                            voting_clf, X_train_final, y_train,
                            cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                        )

                    st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                    for metric in scoring:
                        mean_score = cv_results[f'test_{metric}'].mean()
                        std_score = cv_results[f'test_{metric}'].std()
                        st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")





##########################################################################################################################
######################################             Validation             ################################################
##########################################################################################################################



    st.subheader("📊 Final Validation Set Comparison (Full Metrics)")

    # Retrieve selected models
    selected_models = st.session_state.get("selected_models", [])

    # Define dictionary to store predictions and probabilities
    val_predictions = {}

    # === Conditional Predictions by Model ===

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

    if "K-Nearest Neighbors" in selected_models:
        y_val_pred_knn = knn_model.predict(X_val_final)
        y_val_prob_knn = knn_model.predict_proba(X_val_final)[:, 1]
        val_predictions["K-Nearest Neighbors"] = (y_val_pred_knn, y_val_prob_knn)

    if "Naive Bayes" in selected_models:
        y_val_pred_nb = nb_model.predict(X_val_final)
        y_val_prob_nb = nb_model.predict_proba(X_val_final)[:, 1]
        val_predictions["Naive Bayes"] = (y_val_pred_nb, y_val_prob_nb)

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

    if "Voting Classifier" in selected_models:
        y_val_pred_vote = voting_clf.predict(X_val_final)
        y_val_prob_vote = voting_clf.predict_proba(X_val_final)[:, 1]
        val_predictions["Voting Classifier"] = (y_val_pred_vote, y_val_prob_vote)


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

    # === Choose sample or upload ===
    use_sample_test = st.radio("Provide test data via:", ["Use sample test file", "Upload your own test file"], key="test_source")
    df_test = None

    if use_sample_test == "Use sample test file":
        dataset_names = ["titanic", "heart_disease", "breast cancer", "creditcard", "diabetes", "banknote"]
        format_options = ["csv", "xlsx", "json"]

        dataset = st.selectbox("Select test dataset", dataset_names, key="test_dataset")
        file_format = st.radio("Test file format", format_options, horizontal=True, key="test_format")

        filepath = f"Datasets/{file_format}/{dataset} test.{file_format}"

        try:
            if file_format == "csv":
                df_test = pd.read_csv(filepath)
            elif file_format == "xlsx":
                df_test = pd.read_excel(filepath)
            elif file_format == "json":
                df_test = pd.read_json(filepath)
            st.success(f"✅ Loaded sample test file: {dataset} ({file_format})")
        except Exception as e:
            st.error(f"❌ Could not load sample test file: {e}")
            st.stop()

    else:
        test_file = st.file_uploader("Upload a test dataset (same structure as training data):", type=["csv", "xlsx", "json"], key="test_file")

        if test_file is not None:
            try:
                if test_file.name.endswith(".csv"):
                    df_test = pd.read_csv(test_file)
                elif test_file.name.endswith(".xlsx"):
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

    # Optional: preview data after either loading method
    if df_test is not None:
        st.dataframe(df_test.head())


        df_test_original = df_test.copy()
        try:
            # === Columns from training ===
            expected_columns = st.session_state.get("selected_columns")
            if expected_columns is None:
                st.error("Training columns not found. Please run the training section first.")
                st.stop()

            # Filter test set to training columns (fill missing with 0s)
            df_test = df_test.reindex(columns=expected_columns, fill_value=0)

            # === Encode features ===
            df_test_encoded = pd.get_dummies(df_test, drop_first=True)

            # Align with training columns
            missing_cols = set(st.session_state["X_raw"].columns) - set(df_test_encoded.columns)
            for col in missing_cols:
                df_test_encoded[col] = 0
            df_test_encoded = df_test_encoded[st.session_state["X_raw"].columns].astype("float64")

            # Remove bad rows
            df_test_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
            invalid_test_rows = df_test_encoded.isnull().any(axis=1)
            if invalid_test_rows.any():
                st.warning(f"⚠️ Removed {invalid_test_rows.sum()} rows with NaNs/infs.")
                df_test_encoded = df_test_encoded[~invalid_test_rows]
                df_test = df_test[~invalid_test_rows]
                df_test_original = df_test_original[~invalid_test_rows]

            
            # === Handle target ===
            target_column = st.session_state.get("target_column", None)
            target_column_present = target_column is not None and target_column in df_test.columns

            if target_column_present:
                st.markdown(f"✅ Target column **`{target_column}`** detected in test set.")
                st.markdown("#### 📊 Test Set Target Value Distribution (Raw)")
                st.dataframe(df_test[target_column].value_counts())

                df_test_target = df_test[[target_column]].copy()

                if "label_classes_" in st.session_state:
                    label_classes = st.session_state["label_classes_"]

                    # Validate that test target contains exactly the same labels
                    test_labels = set(df_test_target[target_column].dropna().unique())
                    expected_labels = set(label_classes)

                    if test_labels != expected_labels:
                        st.error(
                            f"❌ Test set target labels must exactly match training labels {sorted(expected_labels)}.\n"
                            f"Found: {sorted(test_labels)}"
                        )
                        st.stop()

                    # Build consistent label mapping
                    label_map = {label: idx for idx, label in enumerate(label_classes)}

                    # Apply encoding
                    df_test_target["encoded_target"] = df_test_target[target_column].map(label_map)
                    df_test_target = df_test_target.dropna(subset=["encoded_target"]).astype({"encoded_target": "int64"})

                    # Align rows in all related dataframes
                    df_test_encoded = df_test_encoded.loc[df_test_target.index].reset_index(drop=True)
                    df_test_original = df_test_original.loc[df_test_target.index].reset_index(drop=True)

                    # Store encoded target for metrics
                    df_test_target_final = df_test_target["encoded_target"]

                    st.markdown("#### ✅ Encoded Target Value Distribution")
                    st.dataframe(df_test_target_final.value_counts())
                else:
                    st.warning("⚠️ Could not find label mapping from training. Skipping encoding.")
                    df_test_target_final = df_test_target[target_column]

            else:
                st.info("ℹ️ No target column found. Predictions will be made but metrics skipped.")
                target_column_present = False



            # === Apply transformations from training ===
            if "transform_pipeline" in st.session_state:
                df_test_transformed = st.session_state["transform_pipeline"].transform(df_test_encoded)
                df_test_transformed = pd.DataFrame(df_test_transformed, columns=df_test_encoded.columns)
            else:
                df_test_transformed = df_test_encoded.copy()


            # PCA transform if selected
            use_pca = st.session_state.get("use_pca", "No")

            if use_pca == "Yes":
                # Retrieve trained PCA and scaler objects from session state
                scaler = st.session_state["scaler"]
                pca = st.session_state["pca"]
                n_components = st.session_state["n_components"]

                # Scale and transform test data
                df_test_scaled = scaler.transform(df_test_encoded)
                df_test_transformed = pd.DataFrame(
                    pca.transform(df_test_scaled),
                    columns=[f"PC{i+1}" for i in range(n_components)]
                )
            else:
                df_test_transformed = df_test_encoded.copy()


            st.markdown("### 🎯 Traffic Light Thresholds")
            threshold_0 = st.slider("Confidence threshold for class 0 (Green)", 0.5, 1.0, 0.85, 0.01)
            threshold_1 = st.slider("Confidence threshold for class 1 (Red)", 0.5, 1.0, 0.85, 0.01)

            def get_traffic_light(pred, prob, threshold_0, threshold_1):
                if pred == 0 and (1 - prob) >= threshold_0:
                    return "Green"
                elif pred == 1 and prob >= threshold_1:
                    return "Red"
                else:
                    return "Yellow"


            # === Retrieve selected models ===
            selected_models = st.session_state.get("selected_models", [])

            # === Initialize results DataFrame ===
            df_results = df_test_original.copy()
            if target_column_present and 'df_test_target_final' in locals():
                df_results[target_column] = df_test_target_final.reset_index(drop=True)


            # === Make Predictions and Add Columns Dynamically ===

            if "Random Forest" in selected_models:
                test_pred_rf = rf_model.predict(df_test_transformed)
                prob_pred_rf = rf_model.predict_proba(df_test_transformed)[:, 1]
                df_results["RandomForest_Prediction"] = test_pred_rf
                df_results["RandomForest_Prob"] = prob_pred_rf
                df_results["RandomForest_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_rf, prob_pred_rf)
                ]


            if "Ridge Logistic Regression" in selected_models:
                test_pred_ridge = ridge_model.predict(df_test_transformed)
                prob_pred_ridge = ridge_model.predict_proba(df_test_transformed)[:, 1]
                df_results["Ridge_Prediction"] = test_pred_ridge
                df_results["Ridge_Prob"] = prob_pred_ridge
                df_results["Ridge_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_ridge, prob_pred_ridge)
                ]

            if "Lasso Logistic Regression" in selected_models:
                test_pred_lasso = lasso_model.predict(df_test_transformed)
                prob_pred_lasso = lasso_model.predict_proba(df_test_transformed)[:, 1]
                df_results["Lasso_Prediction"] = test_pred_lasso
                df_results["Lasso_Prob"] = prob_pred_lasso
                df_results["Lasso_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_lasso, prob_pred_lasso)
                ]

            if "ElasticNet Logistic Regression" in selected_models:
                test_pred_enet = enet_model.predict(df_test_transformed)
                prob_pred_enet = enet_model.predict_proba(df_test_transformed)[:, 1]
                df_results["ElasticNet_Prediction"] = test_pred_enet
                df_results["ElasticNet_Prob"] = prob_pred_enet
                df_results["ElasticNet_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_enet, prob_pred_enet)
                ]

            if "PLS-DA" in selected_models:
                test_scores_pls = pls_model.predict(df_test_transformed).ravel()
                test_pred_pls = (test_scores_pls >= 0.5).astype(int)
                df_results["PLSDA_Prediction"] = test_pred_pls
                df_results["PLSDA_Test_scores"] = test_scores_pls
                df_results["PLSDA_TrafficLight (no yellow)"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_pls, test_scores_pls)
                ]

            if "K-Nearest Neighbors" in selected_models:
                test_pred_knn = knn_model.predict(df_test_transformed)
                prob_pred_knn = knn_model.predict_proba(df_test_transformed)[:, 1]
                df_results["KNN_Prediction"] = test_pred_knn
                df_results["KNN_Prob"] = prob_pred_knn
                df_results["KNN_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_knn, prob_pred_knn)
                ]

            if "Naive Bayes" in selected_models:
                test_pred_nb = nb_model.predict(df_test_transformed)
                prob_pred_nb = nb_model.predict_proba(df_test_transformed)[:, 1]
                df_results["NB_Prediction"] = test_pred_nb
                df_results["NB_Prob"] = prob_pred_nb
                df_results["NB_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_nb, prob_pred_nb)
                ]

            if "Support Vector Machine" in selected_models:
                test_pred_svm = svm_model.predict(df_test_transformed)
                prob_pred_svm = svm_model.predict_proba(df_test_transformed)[:, 1]
                df_results["SVM_Prediction"] = test_pred_svm
                df_results["SVM_Prob"] = prob_pred_svm
                df_results["SVM_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_svm, prob_pred_svm)
                ]

            if "Decision Tree" in selected_models:
                test_pred_tree = tree_model.predict(df_test_transformed)
                prob_pred_tree = tree_model.predict_proba(df_test_transformed)[:, 1]
                df_results["DecisionTree_Prediction"] = test_pred_tree
                df_results["DecisionTree_Prob"] = prob_pred_tree
                df_results["DecisionTree_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_tree, prob_pred_tree)
                ]

            if "Gradient Boosting" in selected_models:
                test_pred_gbm = gbm_model.predict(df_test_transformed)
                prob_pred_gbm = gbm_model.predict_proba(df_test_transformed)[:, 1]
                df_results["GBM_Prediction"] = test_pred_gbm
                df_results["GBM_Prob"] = prob_pred_gbm
                df_results["GBM_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_gbm, prob_pred_gbm)
                ]

            if "Neural Network" in selected_models:
                test_pred_nn = nn_model.predict(df_test_transformed)
                prob_pred_nn = nn_model.predict_proba(df_test_transformed)[:, 1]
                df_results["NN_Prediction"] = test_pred_nn
                df_results["NN_Prob"] = prob_pred_nn
                df_results["NN_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_nn, prob_pred_nn)
                ]

            if "Voting Classifier" in selected_models:
                test_pred_vote = voting_clf.predict(df_test_transformed)
                prob_pred_vote = voting_clf.predict_proba(df_test_transformed)[:, 1]
                df_results["Vote_Prediction"] = test_pred_vote
                df_results["Vote_Prob"] = prob_pred_vote
                df_results["Vote_TrafficLight"] = [
                    get_traffic_light(pred, prob, threshold_0, threshold_1)
                    for pred, prob in zip(test_pred_vote, prob_pred_vote)
                ]




            # === Compute and Display Metrics on Test Data ===

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            test_predictions = {}

            if "Ridge Logistic Regression" in selected_models:
                test_predictions["Ridge Logistic Regression"] = (test_pred_ridge, prob_pred_ridge)

            if "Lasso Logistic Regression" in selected_models:
                test_predictions["Lasso Logistic Regression"] = (test_pred_lasso, prob_pred_lasso)

            if "ElasticNet Logistic Regression" in selected_models:
                test_predictions["ElasticNet Logistic Regression"] = (test_pred_enet, prob_pred_enet)

            if "PLS-DA" in selected_models:
                test_predictions["PLS-DA"] = (test_pred_pls, test_scores_pls)

            if "K-Nearest Neighbors" in selected_models:
                test_predictions["K-Nearest Neighbors"] = (test_pred_knn, prob_pred_knn)

            if "Naive Bayes" in selected_models:
                test_predictions["Naive Bayes"] = (test_pred_nb, prob_pred_nb)

            if "Support Vector Machine" in selected_models:
                test_predictions["Support Vector Machine"] = (test_pred_svm, prob_pred_svm)

            if "Decision Tree" in selected_models:
                test_predictions["Decision Tree"] = (test_pred_tree, prob_pred_tree)

            if "Random Forest" in selected_models:
                test_predictions["Random Forest"] = (test_pred_rf, prob_pred_rf)

            if "Gradient Boosting" in selected_models:
                test_predictions["Gradient Boosting"] = (test_pred_gbm, prob_pred_gbm)

            if "Neural Network" in selected_models:
                test_predictions["Neural Network"] = (test_pred_nn, prob_pred_nn)

            if "Voting Classifier" in selected_models:
                test_predictions["Voting Classifier"] = (test_pred_vote, prob_pred_vote)


            # === If target is present, compute performance metrics ===
            if target_column_present:
                st.markdown("### 📊 Test Set Performance Metrics")

                def compute_metrics(y_true, y_pred, y_prob, model_name):
                    # Sanity check: predicted probabilities should correlate positively with class 1
                    auc_score = roc_auc_score(y_true, y_prob)

                    # Optional: if AUC is inverted (less than 0.5), log a warning
                    if auc_score < 0.5:
                        st.warning(f"⚠️ AUC for {model_name} is {auc_score:.4f}, indicating a possible label inversion.")

                    return {
                        'Model': model_name,
                        'Accuracy': accuracy_score(y_true, y_pred),
                        'Precision': precision_score(y_true, y_pred),
                        'Recall': recall_score(y_true, y_pred),
                        'F1-Score': f1_score(y_true, y_pred),
                        'AUC': auc_score
                    }

                test_metrics = []
                for model_name, (y_pred, y_prob) in test_predictions.items():
                    test_metrics.append(compute_metrics(df_test_target_final, y_pred, y_prob, model_name))

                test_summary_df = pd.DataFrame(test_metrics)
                st.dataframe(test_summary_df.style.format({
                    "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                    "Recall": "{:.4f}", "F1-Score": "{:.4f}", "AUC": "{:.4f}"
                }))

            else:
                st.info("ℹ️ Target column not found in test data. Skipping performance metrics.")










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
