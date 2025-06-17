# === Core Imports ===
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# === Scikit-learn: Preprocessing, Feature Engineering, Models ===
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.utils.multiclass import type_of_target

# === Scikit-learn: Base Models ===
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier

# === Imbalanced-learn ===
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# === Custom Utility Functions ===
from ml_utils import (
    apply_flipping,
    get_class1_proba, ensure_dataframe_and_series,
    init_session_key, get_final_train_val_sets, set_final_datasets,
    log_transformation, apply_minmax_scaling, apply_standard_scaling,
    plot_before_after, create_new_feature,
    export_training_data_general, export_training_data_pls_da
)





##########################################################################################################################
######################################    Presentation   #################################################################
##########################################################################################################################

st.title("🤖 Binary Classification App")

st.markdown("""
**Author:** Jorge Ramos  
**Student ID:** 2599173  
**Project:** MSc Data Analytics – Binary Classification Dashboard  
""")

st.info("This app trains up to 15 machine learning models for datasets with a binary target (0 or 1).")



##########################################################################################################################
######################################    File Upload    #################################################################
##########################################################################################################################



st.markdown("### 📂 Choose a sample dataset or upload your own")

use_sample = st.radio("How would you like to provide your dataset?", ["Use sample dataset", "Upload your own file"])
df = None

if use_sample == "Use sample dataset":
    dataset_names = ["titanic", "heart_disease", "breast cancer", "creditcard", "diabetes", "banknote"]
    format_options = ["csv", "xlsx", "json"]
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
    uploaded_file = st.file_uploader("Upload your data file. It is recommended to have " \
    "more than 40 rows, especially if using cross-validation.", type=["csv", "xlsx", "json"])

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

# === Add row_id as first column and show download ===
if df is not None:
    df.insert(0, "row_id", np.arange(1, len(df) + 1))
    st.session_state["df_with_row_id"] = df.copy()

    st.markdown(
        "### 🔍 Preview of Loaded Data\n"
        "A `row_id` column has been added to track changes. "
        "During training, rows will be randomized and `row_id` will be ignored as an input feature.\n\n"
        "**Please do not delete or transform `row_id`.**"
    )

    st.dataframe(df.head())

    # 💾 Add download button
    csv_with_row_id = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Loaded Data with row_id",
        data=csv_with_row_id,
        file_name="loaded_data_with_row_id.csv",
        mime="text/csv"
    )





##########################################################################################################################
######################################    Delete Columns    ##############################################################
##########################################################################################################################


    if "columns_confirmed" not in st.session_state:
        st.session_state["columns_confirmed"] = False

    # === Column Selection ===
    st.markdown("### 📌 Step 1: Select Columns to Include")

    selected_columns = st.multiselect(
        "Select the columns you want to use. Do not delete row id:",
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


    # === Download Button ===
    st.markdown("### 📥 Download Filtered Dataset")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download CSV after column selection",
        data=csv,
        file_name="filtered_dataset.csv",
        mime="text/csv"
    )



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
#################################        Missing Values Treatment    #####################################################
##########################################################################################################################

    # === Missing Value Handling ===
    st.markdown("### 🧹 Step 2: Handle Missing Values")

    # Recalculate missing values in current df
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0].index.tolist()

    if not missing_cols:
        st.success("✅ No missing values to handle.")
    else:
        st.write(f"⚠️ Columns with missing values: {missing_cols}")

        # Select columns to handle
        selected_missing_cols = st.multiselect(
            "Select columns to handle missing values:",
            options=missing_cols,
            default=missing_cols
        )

        # Select action
        missing_action = st.radio("Select how to handle missing values:", [
            "Drop rows with missing values in selected columns",
            "Drop selected columns entirely",
            "Replace with 0",
            "Replace with mean (numeric columns only)",
            "Replace with median (numeric columns only)"
        ])

        apply_missing = st.button("🚀 Apply Missing Value Handling")

        if apply_missing and selected_missing_cols:

            # Initialize session state log
            if "missing_value_steps" not in st.session_state:
                st.session_state["missing_value_steps"] = []

            if missing_action == "Drop rows with missing values in selected columns":
                df.dropna(subset=selected_missing_cols, inplace=True)
                st.success("✅ Dropped rows with missing values in selected columns.")
                for col in selected_missing_cols:
                    st.session_state["missing_value_steps"].append(("drop_rows", col, None))

            elif missing_action == "Drop selected columns entirely":
                df.drop(columns=selected_missing_cols, inplace=True)
                st.success("✅ Dropped selected columns.")
                for col in selected_missing_cols:
                    st.session_state["missing_value_steps"].append(("drop_column", col, None))

            elif missing_action == "Replace with 0":
                df[selected_missing_cols] = df[selected_missing_cols].fillna(0)
                st.success("✅ Replaced missing values with 0.")
                for col in selected_missing_cols:
                    st.session_state["missing_value_steps"].append(("zero", col, 0))

            elif missing_action == "Replace with mean (numeric columns only)":
                for col in selected_missing_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        mean_val = df[col].mean()
                        df[col].fillna(mean_val, inplace=True)
                        st.session_state["missing_value_steps"].append(("mean", col, mean_val))
                st.success("✅ Replaced missing values with column mean (where numeric).")

            elif missing_action == "Replace with median (numeric columns only)":
                for col in selected_missing_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        st.session_state["missing_value_steps"].append(("median", col, median_val))
                st.success("✅ Replaced missing values with column median (where numeric).")

    # === Final Download Section ===
    st.markdown("### 💾 Download Final Transformed Dataset")
    final_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=final_csv,
        file_name="transformed_dataset.csv",
        mime="text/csv"
    )



##########################################################################################################################
#################################        Target Selection    #############################################################
##########################################################################################################################

    
    # === Target Selection ===
    st.markdown("### 🎯 Step 2: Select Target Column")

    # Initialization
    if "target_confirmed" not in st.session_state:
        st.session_state["target_confirmed"] = False
    if "target_column" not in st.session_state:
        st.session_state["target_column"] = None

    # Run selection UI only if not yet confirmed
    if not st.session_state["target_confirmed"]:
        target_column_input = st.selectbox("Select the target column:", df.columns)

        if st.button("✅ Confirm Target Selection"):
            st.session_state["target_column"] = target_column_input
            st.session_state["target_confirmed"] = True
            st.rerun()

        st.info("👈 Please confirm target column to continue.")
        st.stop()

    # ✅ Use confirmed value from here on
    target_column = st.session_state["target_column"]
    y_raw_original = df[target_column]
    X_raw = df.drop(columns=[target_column])

    # If y_raw already exists, skip remapping
    if "y_raw" not in st.session_state:

        # === Class Mapping ===
        unique_vals = y_raw_original.dropna().unique()

        if len(unique_vals) != 2:
            st.error("❌ Target column must contain exactly two unique values for binary classification.")
            st.stop()

        st.markdown("### 🔁 Map Target Classes to 0 and 1")

        class_0 = st.selectbox("Select class to map to **0**", unique_vals, key="map_class_0")
        class_1 = st.selectbox("Select class to map to **1**", [val for val in unique_vals if val != class_0], key="map_class_1")

        if st.button("🎯 Apply Class Mapping"):
            label_mapping = {class_0: 0, class_1: 1}
            y_raw = y_raw_original.map(label_mapping)

            if y_raw.isnull().any():
                st.error("❌ Error in mapping target classes. Please check selections.")
                st.stop()

            y_raw = pd.Series(y_raw.astype("int64"), name="target")
            st.session_state["target_mapping"] = label_mapping
            st.session_state["label_map"] = st.session_state["target_mapping"]
            st.write("Label map stored:", st.session_state["label_map"])
            st.session_state["label_classes_"] = [class_0, class_1]
            st.session_state["y_raw"] = y_raw
            st.success(f"✅ Class mapping applied: {class_0} → 0, {class_1} → 1")
            st.rerun()

        st.info("👈 Apply class mapping to continue.")
        st.stop()

    # Continue with mapped target
    y_raw = st.session_state["y_raw"]

    # === Target Class Summary ===
    st.markdown("#### 📊 Target Value Distribution")

    target_counts = y_raw.value_counts().sort_index()
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
    # Drop rows with missing values in X or y
    X_encoded, y_raw_clean = X_encoded.align(y_raw, join='inner', axis=0)
    X_encoded = X_encoded.dropna()
    y_raw_clean = y_raw_clean.loc[X_encoded.index]


    # Let user select importance method
    method = st.selectbox(
        "Choose importance method:",
        ["Random Forest", "Lasso (L1)", "Permutation (RF)", "Mutual Info"],
        index=0
    )

    # Train/Test split (just for internal importance calc)
    X_train_imp, X_val_imp, y_train_imp, y_val_imp = train_test_split(
        X_encoded, y_raw_clean, test_size=0.2, random_state=42
    )

    # Compute importances
    importance_values = None
    model_used = None

    if method == "Random Forest":
        model_used = RandomForestClassifier(random_state=42)
        model_used.fit(X_train_imp, y_train_imp)
        importance_values = model_used.feature_importances_

    elif method == "Lasso (L1)":
        model_used = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        model_used.fit(X_train_imp, y_train_imp)
        importance_values = np.abs(model_used.coef_[0])

    elif method == "Permutation (RF)":
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train_imp, y_train_imp)
        result = permutation_importance(rf, X_val_imp, y_val_imp, n_repeats=10, random_state=42)
        importance_values = result.importances_mean

    elif method == "Mutual Info":
        importance_values = mutual_info_classif(X_train_imp, y_train_imp, random_state=42)

    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": X_encoded.columns,
        "Importance": importance_values
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    st.dataframe(importance_df.style.format({"Importance": "{:.4f}"}))

    # Optional bar chart
    if st.checkbox("📊 Show Top 10 Features as Bar Chart"):
        top_10 = importance_df.sort_values("Importance", ascending=False).head(10)
        st.bar_chart(top_10.set_index("Feature"))




##########################################################################################################################
#################################        Split Data Train and Validate    ################################################
##########################################################################################################################

    # === Step: Split Data Train and Validate ===

    # === Feature Importance ===
    st.markdown("### Split Data Train and Validate")

    # Save original row_id
    if "row_id" in df.columns:
        st.session_state["row_id"] = df["row_id"].copy()
    else:
        st.session_state["row_id"] = pd.Series(np.arange(len(df)), name="row_id")


    # Drop target column
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]

    # Store in session_state
    st.session_state["target_column"] = target_column
    st.success(f"✅ Target column confirmed: `{target_column}`")

    # Add row_id for tracking (start at 1 if you prefer)
    row_ids = pd.Series(np.arange(len(df)), name="row_id")

    # Convert target to integer labels
    y_raw, label_classes = pd.factorize(y_raw)
    y_raw = pd.Series(y_raw.astype('int64'), name="target")
    st.session_state["label_classes_"] = label_classes.tolist()

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X_raw, drop_first=True).astype("float64")

    # Handle missing/infinite values
    X_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
    invalid_rows = X_encoded.isnull().any(axis=1)
    if invalid_rows.any():
        st.warning(f"⚠️ Removed {invalid_rows.sum()} rows with NaNs or infinite values.")
        X_encoded = X_encoded[~invalid_rows]
        y_raw = y_raw[~invalid_rows]
        row_ids = row_ids[~invalid_rows]

    # Store cleaned full data for test compatibility
    st.session_state["X_raw"] = X_encoded.copy()

    # === Train/Validation Split ===
    test_size = st.slider("Select validation set size (%)", 10, 50, 20, 5) / 100.0
    X_train, X_val, y_train, y_val, row_id_train, row_id_val = train_test_split(
        X_encoded, y_raw, row_ids, test_size=test_size, random_state=42, shuffle=True
    )

    # Keep copies without row_id for modeling
    st.session_state["X_train"] = X_train.copy()
    st.session_state["X_val"] = X_val.copy()
    st.session_state["y_train"] = y_train.copy()
    st.session_state["y_val"] = y_val.copy()
    st.session_state["row_id_train"] = row_id_train.copy()
    st.session_state["row_id_val"] = row_id_val.copy()

    # === Downloads with row_id as first column ===
    st.markdown("### 💾 Download Processed Train/Validation Splits")

    # Reinsert row_id as first column in features
    X_train_with_id = pd.concat([row_id_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
    X_val_with_id = pd.concat([row_id_val.reset_index(drop=True), X_val.reset_index(drop=True)], axis=1)

    y_train_df = pd.DataFrame({
        "row_id": row_id_train.reset_index(drop=True),
        "target": y_train.reset_index(drop=True)
    })
    y_val_df = pd.DataFrame({
        "row_id": row_id_val.reset_index(drop=True),
        "target": y_val.reset_index(drop=True)
    })

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download X_train.csv", X_train_with_id.to_csv(index=False).encode("utf-8"),
                        "X_train.csv", "text/csv")
        st.download_button("⬇️ Download y_train.csv", y_train_df.to_csv(index=False).encode("utf-8"),
                        "y_train.csv", "text/csv")
    with col2:
        st.download_button("⬇️ Download X_val.csv", X_val_with_id.to_csv(index=False).encode("utf-8"),
                        "X_val.csv", "text/csv")
        st.download_button("⬇️ Download y_val.csv", y_val_df.to_csv(index=False).encode("utf-8"),
                        "y_val.csv", "text/csv")

    # Save train indices
    st.session_state["train_idx"] = X_train.index



##########################################################################################################################
#################################        Data Transformation (Before PCA)       ##########################################
##########################################################################################################################




    st.markdown("### 🔧 Step 2.5: Optional Data Transformation")

    # Initialize session keys
    init_session_key("X_train_resampled", st.session_state["X_train"].copy())
    init_session_key("X_val_resampled", st.session_state["X_val"].copy())
    init_session_key("y_train_resampled", st.session_state["y_train"].copy())
    init_session_key("transform_steps", [])

    # Current working copies
    X_train_resampled = st.session_state["X_train_resampled"]
    X_val_resampled = st.session_state["X_val_resampled"]

    # Toggle transformation panel
    apply_transformation = st.checkbox("🧪 Would you like to transform the data before PCA?", value=False)

    if apply_transformation:
        st.info("Transformations will apply to both training and validation sets consistently.")

        # === 1️⃣ Centering + Scaling (MinMaxScaler) ===
        if st.checkbox("1️⃣ Centering + Scaling (MinMaxScaler)", value=False):
            numeric_cols = [col for col in X_train_resampled.select_dtypes(include=np.number).columns if col != "row_id"]
            col_to_scale = st.selectbox("Select column to scale", numeric_cols, key="scale_col")

            if st.button("✅ Confirm Scaling"):
                before = X_train_resampled[col_to_scale].copy()

                # Apply scaling using helper
                X_train_resampled, X_val_resampled, scaler = apply_minmax_scaling(
                    X_train_resampled, X_val_resampled, col_to_scale
                )

                # Save updated datasets
                st.session_state["X_train_resampled"] = X_train_resampled
                st.session_state["X_val_resampled"] = X_val_resampled

                # Log step
                log_transformation(f"minmax_{col_to_scale}", scaler, col_to_scale)

                # Show plots using helper
                after = X_train_resampled[col_to_scale]
                plot_before_after(before, after, title=f"MinMax Scaling ({col_to_scale})")

                # Optional download
                st.markdown("#### 📥 Download Transformed Training Set")
                csv_scaled = X_train_resampled.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Scaled Train Set", csv_scaled, "train_scaled.csv", "text/csv")




    # === 2️⃣ Standardization (Zero Mean, Unit Variance) ===
    if st.checkbox("2️⃣ Standardization (Zero Mean, Unit Variance)", value=False):
        numeric_cols = [col for col in X_train_resampled.select_dtypes(include=np.number).columns if col != "row_id"]
        col_to_standardize = st.selectbox("Select column to standardize", numeric_cols, key="standardize_col")

        if st.button("✅ Confirm Standardizing"):
            before = X_train_resampled[col_to_standardize].copy()

            # Apply transformation using helper
            X_train_resampled, X_val_resampled, std_scaler = apply_standard_scaling(
                X_train_resampled, X_val_resampled, col_to_standardize
            )

            # Save updated state
            st.session_state["X_train_resampled"] = X_train_resampled
            st.session_state["X_val_resampled"] = X_val_resampled

            # Log transformation step
            log_transformation(f"standard_{col_to_standardize}", std_scaler, col_to_standardize)

            # Plot before/after using helper
            after = X_train_resampled[col_to_standardize]
            fig = plot_before_after(before, after, "Standardization")
            st.pyplot(fig)

            # Optional download
            csv_std = X_train_resampled.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Standardized Train Set", csv_std, "train_standardized.csv", "text/csv")





    # === 3️⃣ Create New Feature ===
    if st.checkbox("3️⃣ Create New Feature", value=False):
        operation = st.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide", "Log", "Square"], key="op")

        numeric_cols = [col for col in X_train_resampled.select_dtypes(include=np.number).columns if col != "row_id"]
        col1 = st.selectbox("Select first column", numeric_cols, key="new_col1")

        col2 = None
        if operation in ["Add", "Subtract", "Multiply", "Divide"]:
            col2 = st.selectbox("Select second column", numeric_cols, key="new_col2")

        default_name = f"{col1}_{operation}_{col2}" if col2 else f"{col1}_{operation}"
        new_col_name = st.text_input("New column name", value=default_name)

        if st.button("➕ Add New Feature"):
            try:
                # Apply helper
                X_train_resampled, X_val_resampled = create_new_feature(
                    X_train_resampled, X_val_resampled, operation, col1, col2, new_col_name
                )

                # Save updated sets
                st.session_state["X_train_resampled"] = X_train_resampled
                st.session_state["X_val_resampled"] = X_val_resampled

                # Log transformation
                log_transformation("create_feature", {
                    "operation": operation,
                    "col1": col1,
                    "col2": col2,
                    "new_col": new_col_name
                }, target="custom")

                st.success(f"✅ Feature '{new_col_name}' successfully added.")

                # Optional download
                csv_feat = X_train_resampled.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Train Set with New Feature", csv_feat, f"train_{new_col_name}.csv", "text/csv")

            except Exception as e:
                st.error(f"⚠️ Error creating feature: {e}")






    # === 4️⃣ Outlier Detection & Removal ===
    if st.checkbox("4️⃣ Outlier Detection & Removal", value=False):

        # Restore current state of train data and labels
        X_train_resampled = st.session_state.get("X_train_resampled", st.session_state["X_train"]).copy()
        y_train_resampled = st.session_state.get("y_train_resampled", st.session_state["y_train"]).copy()

        numeric_cols = [col for col in X_train_resampled.select_dtypes(include=np.number).columns if col != "row_id"]
        outlier_col = st.selectbox("Select column", numeric_cols, key="outlier_col")
        method = st.selectbox("Outlier Method", ["IQR (1.5x)"], key="outlier_method")

        col_data = X_train_resampled[outlier_col]
        q1, q3 = col_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        outliers_train = (col_data < lower) | (col_data > upper)
        outlier_count = outliers_train.sum()

        st.write(f"📊 Outliers detected in training set: **{outlier_count}** out of **{len(col_data)}**")

        # Visualize boxplot
        fig, ax = plt.subplots()
        sns.boxplot(x=col_data, ax=ax)
        ax.axvline(lower, color="red", linestyle="--", label="Lower Threshold")
        ax.axvline(upper, color="red", linestyle="--", label="Upper Threshold")
        ax.set_title("Boxplot with IQR Thresholds")
        ax.legend()
        st.pyplot(fig)

        if st.button("🧹 Confirm Outlier Removal (Train Only)"):
            # Keep non-outlier rows
            keep_indices = ~outliers_train
            X_train_clean = X_train_resampled[keep_indices].reset_index(drop=True)
            y_train_clean = y_train_resampled[keep_indices].reset_index(drop=True)

            # Get row_ids removed
            removed_ids = X_train_resampled.loc[outliers_train, "row_id"].reset_index(drop=True)

            # Update session state
            st.session_state["X_train_resampled"] = X_train_clean
            st.session_state["y_train_resampled"] = y_train_clean
            st.session_state["removed_row_ids"] = removed_ids
            st.session_state["outlier_confirmed"] = True

            st.session_state["transform_steps"].append((
                "remove_outliers",
                {
                    "method": "IQR",
                    "column": outlier_col,
                    "thresholds": [float(lower), float(upper)],
                    "removed_count": int(outlier_count)
                },
                "row_filter"
            ))

            st.success("✅ Outliers removed and row IDs recorded.")

    # === Show download buttons if removal was confirmed ===
    if st.session_state.get("outlier_confirmed", False):
        st.markdown("#### 💾 Download Cleaned Data")
        csv_cleaned = st.session_state["X_train_resampled"].to_csv(index=False).encode("utf-8")
        csv_removed_ids = st.session_state["removed_row_ids"].to_frame(name="row_id").to_csv(index=False).encode("utf-8")

        st.download_button("⬇️ Download Train Set After Outlier Removal", csv_cleaned, "train_outliers_removed.csv", "text/csv")
        st.download_button("⬇️ Download Removed Outlier Row IDs", csv_removed_ids, "removed_outlier_row_ids.csv", "text/csv")






    # === 5️⃣ Class Imbalance Handling (Train Set Only) ===
    st.markdown("### ⚖️ Optional: Handle Class Imbalance (Train Set Only)")

    # ✅ Restore current training data with fallback
    X_train_resampled = st.session_state.get("X_train_resampled", st.session_state["X_train"]).copy()
    y_train_resampled = st.session_state.get("y_train_resampled", st.session_state["y_train"]).copy()

    # Resampling method selection
    imbalance_strategy = st.radio(
        "Choose a resampling method:",
        ["None", "Undersampling", "Oversampling", "SMOTE"],
        index=0
    )

    # Show current distribution
    st.write("📊 Current class distribution:")
    st.dataframe(pd.Series(y_train_resampled).value_counts().rename("Count"))

    # Initialize sampler based on selection
    sampler = None
    if imbalance_strategy == "Undersampling":
        sampler = RandomUnderSampler(random_state=42)
    elif imbalance_strategy == "Oversampling":
        sampler = RandomOverSampler(random_state=42)
    elif imbalance_strategy == "SMOTE":
        sampler = SMOTE(random_state=42)

    # Apply resampling if method is selected
    if sampler and st.button("✅ Apply Resampling"):
        X_train_new, y_train_new = sampler.fit_resample(X_train_resampled, y_train_resampled)

        # Remove previous resample steps
        st.session_state["transform_steps"] = [
            step for step in st.session_state.get("transform_steps", [])
            if step[0] != "resample"
        ]

        # Log transformation
        st.session_state["transform_steps"].append((
            "resample",
            {
                "strategy": imbalance_strategy,
                "sampler": type(sampler).__name__,
                "original_class_counts": pd.Series(y_train_resampled).value_counts().to_dict(),
                "resampled_class_counts": pd.Series(y_train_new).value_counts().to_dict()
            },
            "resample"
        ))

        # Save updated data
        st.session_state["X_train_resampled"] = X_train_new
        st.session_state["y_train_resampled"] = y_train_new

        st.success(f"✅ {imbalance_strategy} applied.")
        st.write("📊 New class distribution:")
        st.dataframe(pd.Series(y_train_new).value_counts().rename("Count"))

        # Optional download
        df_download = X_train_new.copy()
        df_download["Target"] = y_train_new
        csv_data = df_download.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Resampled Train Set",
            csv_data,
            f"train_{imbalance_strategy.lower()}.csv",
            "text/csv"
        )






    # === 5️⃣🆕 Optional: Drop Columns Manually Before PCA ===
    st.markdown("### 🧹 Optional: Drop Unwanted Columns Before PCA")
    st.write("You can remove any features (including engineered ones) before applying PCA or training models.")

    # ✅ Restore latest transformed state (important!)
    X_train_resampled = st.session_state.get("X_train_resampled", st.session_state["X_train"]).copy()
    X_val_resampled = st.session_state.get("X_val_resampled", st.session_state["X_val"]).copy()
    y_train_resampled = st.session_state.get("y_train_resampled", st.session_state["y_train"]).copy()

    # Show full list of columns (after all transformations)
    cols_to_consider = X_train_resampled.columns.tolist()

    cols_to_drop = st.multiselect(
        "Select columns to drop:",
        options=cols_to_consider,
        help="These columns will be removed from both the train and validation sets."
    )

    if st.button("✅ Confirm Column Drop"):
        if cols_to_drop:
            X_train_resampled.drop(columns=cols_to_drop, inplace=True)
            X_val_resampled.drop(columns=[col for col in cols_to_drop if col in X_val_resampled.columns], inplace=True)

            st.success(f"✅ Dropped {len(cols_to_drop)} column(s): {', '.join(cols_to_drop)}")

            # Save transformation step
            st.session_state["transform_steps"].append((
                "drop_columns",
                {
                    "columns_dropped": cols_to_drop
                },
                "cleanup"
            ))

            # ✅ Persist the updated state after transformation
            st.session_state["X_train_resampled"] = X_train_resampled
            st.session_state["X_val_resampled"] = X_val_resampled
            st.session_state["y_train_resampled"] = y_train_resampled

            # 📥 Download updated train set
            df_cleaned = X_train_resampled.copy()
            df_cleaned["Target"] = y_train_resampled
            csv_cleaned = df_cleaned.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Train Set After Column Deletion",
                csv_cleaned,
                "train_columns_dropped.csv",
                "text/csv"
            )

    # ✅ Final overwrite to main session keys (ensures PCA & Preview see latest version)
    st.session_state["X_train"] = X_train_resampled
    st.session_state["y_train"] = y_train_resampled
    st.session_state["X_val"] = X_val_resampled
    st.session_state["y_val"] = st.session_state["y_val"]  # Unchanged, but reassigned for safety



##########################################################################################################################
################################       PCA Step    #######################################################################
##########################################################################################################################

    # === Step 3: PCA Selection ===
    st.markdown("### 🧬 Step 3: PCA Dimensionality Reduction")

    # Initialization
    if "pca_confirmed" not in st.session_state:
        st.session_state["pca_confirmed"] = False
    if "pca_ready" not in st.session_state:
        st.session_state["pca_ready"] = False
    if "use_pca" not in st.session_state:
        st.session_state["use_pca"] = "No"
    if "n_components_slider" not in st.session_state:
        st.session_state["n_components_slider"] = 2

    # === Ask if PCA should be used ===
    use_pca_input = st.radio("Would you like to apply PCA?", ["No", "Yes"], index=0)

    if st.button("✅ Confirm PCA Selection"):
        st.session_state["use_pca"] = use_pca_input
        st.session_state["pca_confirmed"] = True
        st.session_state["pca_ready"] = False
        st.rerun()

    if not st.session_state["pca_confirmed"]:
        st.info("👈 Please confirm PCA selection to continue.")
        st.stop()

    # === Apply PCA ===
    use_pca = st.session_state["use_pca"]

    if use_pca == "Yes":
        X_train_df = st.session_state.get("X_train_resampled", st.session_state["X_train"])
        X_val_df = st.session_state.get("X_val_resampled", st.session_state["X_val"])

        numeric_cols = [col for col in X_train_df.select_dtypes(include=np.number).columns if col != "row_id"]

        # ✅ Save input columns used for PCA
        st.session_state["pca_input_columns"] = numeric_cols

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_df[numeric_cols])
        X_val_scaled = scaler.transform(X_val_df[numeric_cols])

        # Temporary PCA for visualization
        pca_temp = PCA()
        pca_temp.fit(X_train_scaled)

        cum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
        ax.set_title("Cumulative Explained Variance")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance")
        st.pyplot(fig)

        loadings = pd.DataFrame(
            pca_temp.components_.T,
            index=numeric_cols,
            columns=[f"PC{i+1}" for i in range(pca_temp.n_components_)]
        )
        st.markdown("### 📊 PCA Loadings")
        st.dataframe(loadings.round(4))

        st.session_state["scaler"] = scaler
        st.session_state["pca_temp"] = pca_temp

        st.session_state["n_components_slider"] = st.slider(
            "Select number of principal components to keep",
            1,
            X_train_scaled.shape[1],
            value=st.session_state["n_components_slider"]
        )

        if st.button("✅ Confirm PCA Parameters"):
            n_components = st.session_state["n_components_slider"]
            pca_input_cols = st.session_state.get("pca_input_columns")

            if not pca_input_cols:
                st.error("PCA input columns not found.")
                st.stop()

            # Re-transform
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_df[pca_input_cols])
            X_val_scaled = scaler.transform(X_val_df[pca_input_cols])

            final_pca = PCA(n_components=n_components)
            X_train_final = pd.DataFrame(final_pca.fit_transform(X_train_scaled), columns=[f"PC{i+1}" for i in range(n_components)])
            X_val_final = pd.DataFrame(final_pca.transform(X_val_scaled), columns=[f"PC{i+1}" for i in range(n_components)])

            # Add row_id
            X_train_final["row_id"] = X_train_df["row_id"].values
            X_val_final["row_id"] = X_val_df["row_id"].values

            # Save to session state
            st.session_state["pca"] = final_pca
            st.session_state["scaler"] = scaler
            st.session_state["n_components"] = n_components
            st.session_state["X_train_pca"] = X_train_final
            st.session_state["X_val_pca"] = X_val_final
            st.session_state["pca_ready"] = True

            st.session_state["transform_steps"].append((
                "pca",
                {
                    "n_components": n_components,
                    "scaler": "StandardScaler",
                    "original_columns": pca_input_cols
                },
                "dimensionality_reduction"
            ))

            st.success(f"✅ PCA applied with {n_components} components.")
            st.dataframe(X_train_final.head())

    # ✅ Get PCA or non-PCA final sets
    X_train_final, X_val_final = get_final_train_val_sets()
    set_final_datasets(X_train_final, X_val_final)

    # === Optional: Download PCA-transformed training set ===
    if use_pca == "Yes":
        train_pca_csv = X_train_final.copy()
        train_pca_csv["Target"] = pd.Series(st.session_state["y_train"]).reset_index(drop=True)
        csv = train_pca_csv.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download PCA-Transformed Train Set", csv, "train_pca.csv", "text/csv")







##########################################################################################################################
###########################    Data Preview before Model Training     ####################################################
##########################################################################################################################

    # === Step 3.5: Preview Final Data & Pipeline Before Modeling ===
    st.markdown("### 🔬 Final Check: Transformed Data & Pipeline Overview")

    # Ensure the final datasets are consistent with PCA or not
    X_train_final, X_val_final = get_final_train_val_sets()
    set_final_datasets(X_train_final, X_val_final)

    # Preview final training data
    st.subheader("📦 Preview of Final Training Data")
    st.dataframe(X_train_final.head())

    # Show shape info
    st.write("🔢 **Shape of Training and Validation Sets:**")
    st.write(f"- X_train_final: {X_train_final.shape}")
    st.write(f"- X_val_final: {X_val_final.shape}")
    st.write(f"- y_train: {st.session_state['y_train'].shape}")
    st.write(f"- y_val: {st.session_state['y_val'].shape}")

    # Preview pipeline
    st.subheader("🧪 Applied Transformation Pipeline (after train/validate split)")
    if "transform_steps" in st.session_state and st.session_state["transform_steps"]:
        for step_name, transformer, target in st.session_state["transform_steps"]:
            with st.expander(f"🔧 {step_name} on `{target}`"):
                st.write(f"**Transformer Type:** {type(transformer).__name__}")
                if isinstance(transformer, dict):
                    st.json(transformer)
                else:
                    try:
                        st.write(transformer.get_params())
                    except:
                        st.write("Parameters not available.")
    else:
        st.info("No transformation steps recorded.")

    # 📥 Download Transformed Datasets (Before Modeling)
    st.markdown("### 💾 Download Transformed Data Before Modeling")

    # Copy final datasets
    df_train_final = X_train_final.copy()
    df_val_final = X_val_final.copy()

    # Re-attach row_id if missing (safeguard)
    if "row_id" not in df_train_final.columns:
        df_train_final["row_id"] = st.session_state["X_train"]["row_id"].values
    if "row_id" not in df_val_final.columns:
        df_val_final["row_id"] = st.session_state["X_val"]["row_id"].values

    # Add target column
    df_train_final["Target"] = st.session_state["y_train"]
    df_val_final["Target"] = st.session_state["y_val"]

    # Encode and offer download
    csv_train_final = df_train_final.to_csv(index=False).encode("utf-8")
    csv_val_final = df_val_final.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Final Train Set",
        data=csv_train_final,
        file_name="train_final_transformed.csv",
        mime="text/csv"
    )

    st.download_button(
        label="⬇️ Download Final Validation Set",
        data=csv_val_final,
        file_name="val_final_transformed.csv",
        mime="text/csv"
    )





    # === Ready for Modeling Confirmation ===
    if st.button("🚀 Proceed to Model Selection and Training"):
        st.session_state["ready_for_modeling"] = True
        st.success("✅ You can now proceed to model training below.")
        st.rerun()




##########################################################################################################################
###########################    Machine Learning Methods for Binary Classification     ####################################
##########################################################################################################################

    # Only show model selection if user has confirmed
    if st.session_state.get("ready_for_modeling", False):

        st.markdown("### 🧠 Step 4: Select ML Models to Train")

        # Initialize model state
        if "models_confirmed" not in st.session_state:
            st.session_state["models_confirmed"] = False
        if "selected_models" not in st.session_state:
            st.session_state["selected_models"] = []

        # User model selection input
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
                "Bagging", 
                "Pasting",
                "Stacking",
                "Voting Classifier"
            ],
            default=st.session_state["selected_models"]
        )

        # Confirm button
        if st.button("✅ Confirm Model Selection"):

            if len(model_selection_input) < 4:
                st.warning("⚠️ Please select at least four models to continue.")
                st.stop()

            st.session_state["selected_models"] = model_selection_input
            st.session_state["models_confirmed"] = True
            st.rerun()

        # Require confirmation to proceed
        if not st.session_state["models_confirmed"]:
            st.info("👈 Please confirm model selection to continue.")
            st.stop()

        # ✅ Extract selected models after confirmation
        selected_models = st.session_state["selected_models"]

        # ✅ Get the final training/validation sets from transformation or PCA
        X_train_final, X_val_final = get_final_train_val_sets()

        # ✅ Get labels (after transformation section)
        y_train = st.session_state["y_train"]
        y_val = st.session_state["y_val"]

        # === Label Encoding ===
        if y_train.dtype == 'object' or y_train.dtype.name == 'category':
            label_classes = sorted(np.unique(y_train))
            label_map = {label: i for i, label in enumerate(label_classes)}
        else:
            label_classes = [0, 1]
            label_map = {0: 0, 1: 1}

        # Apply mapping
        y_train_encoded = y_train.map(label_map) if hasattr(y_train, "map") else np.vectorize(label_map.get)(y_train)

        # Save to session
        st.session_state["y_train_encoded"] = y_train_encoded
        st.session_state["label_classes_"] = label_classes
        st.session_state["label_map_"] = label_map

        # Debug info
        st.success(f"✅ {len(model_selection_input)} model(s) selected and confirmed.")
        st.write("🎯 Label encoding applied during training:")
        st.write("Classes:", label_classes)
        st.write("Mapping:", label_map)







        # === Train Models ===

        # === Ridge Logistic Regression (with CV + Tuning) ===
        if "Ridge Logistic Regression" in selected_models:
            with st.expander("🧱 Ridge Logistic Regression (L2)"):
                st.write("**Hyperparameters**")
                enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning (Grid or Random Search)?", key="ridge_tuning")

                ridge_model_ready = False

                if enable_tuning:
                    search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="ridge_search_method")
                    ridge_max_iter = st.slider("Ridge: Max Iterations", 100, 2000, 1000, step=100, key="ridge_max_iter")
                    c_range = st.slider("C Range (log scale)", 0.01, 10.0, (0.1, 5.0), step=0.1, key="ridge_c_range")
                    param_grid = {"C": np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), num=10)}
                    n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="ridge_cv_folds")

                    if st.button("🚀 Train Ridge Model with Tuning"):
                        with st.spinner("Running hyperparameter tuning..."):
                            base_model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=ridge_max_iter, random_state=42)
                            if search_method == "Grid Search":
                                ridge_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                ridge_search = RandomizedSearchCV(
                                    base_model, param_distributions=param_grid, n_iter=10,
                                    cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42
                                )

                            ridge_search.fit(X_train_final, y_train)
                            ridge_model = ridge_search.best_estimator_
                            st.write(f"Best estimator classes: {ridge_model.classes_}")  # ✅ Show model class order
                            st.session_state["ridge_model"] = ridge_model
                            st.session_state["ridge_predictions"] = ridge_model.predict(X_train_final)
                            st.session_state["ridge_probabilities"] = get_class1_proba(ridge_model, X_train_final)
                            st.success(f"Best C: {ridge_model.C:.4f}")
                            ridge_model_ready = True

                else:
                    ridge_C = st.slider("Ridge: Regularization strength (C)", 0.01, 10.0, 1.0, key="ridge_C_manual")
                    ridge_max_iter = st.slider("Ridge: Max iterations", 100, 2000, 1000, step=100, key="ridge_iter_manual")

                    if st.button("🚀 Train Ridge Model (Manual)"):
                        with st.spinner("Training Ridge model..."):
                            ridge_model = LogisticRegression(
                                penalty='l2',
                                C=ridge_C,
                                solver='liblinear',
                                max_iter=ridge_max_iter,
                                random_state=42
                            )
                            ridge_model.fit(X_train_final, y_train)
                            st.write(f"Model trained with classes: {ridge_model.classes_}")  # ✅ Show model class order
                            st.session_state["ridge_model"] = ridge_model
                            st.session_state["ridge_predictions"] = ridge_model.predict(X_train_final)
                            st.session_state["ridge_probabilities"] = get_class1_proba(ridge_model, X_train_final)
                            st.success("Model trained successfully!")
                            ridge_model_ready = True

                # === After training (common to both paths) ===
                if "ridge_model" in st.session_state:
                    ridge_model = st.session_state["ridge_model"]

                    # Optional cross-validation
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

                    # Training performance and export

                    df_ridge_train_export, ridge_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=st.session_state["ridge_model"],
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="Ridge",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="ridge_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in ridge_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download Ridge Training Set with Predictions")
                    csv_ridge_train = df_ridge_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Ridge Training Data",
                        data=csv_ridge_train,
                        file_name="ridge_training_predictions.csv",
                        mime="text/csv"
                    )






        # === Lasso Logistic Regression (with CV + Tuning) ===
        if "Lasso Logistic Regression" in selected_models:
            with st.expander("🧊 Lasso Logistic Regression (L1)"):
                st.write("**Hyperparameters**")

                enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning (Grid or Random Search)?", key="lasso_tuning")
                lasso_model_ready = False

                if enable_tuning:
                    search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="lasso_search_method")
                    lasso_max_iter = st.slider("Lasso: Max Iterations", 100, 2000, 1000, step=100, key="lasso_max_iter")
                    c_range = st.slider("C Range (log scale)", 0.01, 10.0, (0.1, 5.0), step=0.1, key="lasso_c_range")
                    param_grid = {"C": np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), num=10)}
                    n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="lasso_cv_folds")

                    if st.button("🚀 Train Lasso Model with Tuning"):
                        with st.spinner("Running hyperparameter tuning..."):
                            base_model = LogisticRegression(
                                penalty='l1', solver='liblinear', max_iter=lasso_max_iter, random_state=42
                            )

                            if search_method == "Grid Search":
                                lasso_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                lasso_search = RandomizedSearchCV(
                                    base_model, param_distributions=param_grid, n_iter=10,
                                    cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42
                                )

                            lasso_search.fit(X_train_final, y_train)
                            lasso_model = lasso_search.best_estimator_

                            st.session_state["lasso_model"] = lasso_model
                            st.session_state["lasso_predictions"] = lasso_model.predict(X_train_final)
                            st.session_state["lasso_probabilities"] = get_class1_proba(lasso_model, X_train_final)

                            st.success(f"Best C: {lasso_model.C:.4f}")
                            lasso_model_ready = True

                else:
                    lasso_C = st.slider("Lasso: Regularization strength (C)", 0.01, 10.0, 1.0, key="lasso_C_manual")
                    lasso_max_iter = st.slider("Lasso: Max iterations", 100, 2000, 1000, step=100, key="lasso_iter_manual")

                    if st.button("🚀 Train Lasso Model (Manual)"):
                        with st.spinner("Training Lasso Logistic Regression..."):
                            lasso_model = LogisticRegression(
                                penalty='l1',
                                C=lasso_C,
                                solver='liblinear',
                                max_iter=lasso_max_iter,
                                random_state=42
                            )
                            lasso_model.fit(X_train_final, y_train)

                            st.session_state["lasso_model"] = lasso_model
                            st.session_state["lasso_predictions"] = lasso_model.predict(X_train_final)
                            st.session_state["lasso_probabilities"] = get_class1_proba(lasso_model, X_train_final)

                            st.success("Model trained successfully!")
                            lasso_model_ready = True

                # === After training (common to both paths) ===
                if "lasso_model" in st.session_state:
                    lasso_model = st.session_state["lasso_model"]

                    # Optional cross-validation
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

                    # Training performance and export
                    df_lasso_train_export, lasso_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=st.session_state["lasso_model"],
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="Lasso",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="lasso_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in lasso_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download Lasso Training Set with Predictions")
                    csv_lasso_train = df_lasso_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Lasso Training Data",
                        data=csv_lasso_train,
                        file_name="lasso_training_predictions.csv",
                        mime="text/csv"
                    )





        # === Elastic Net Logistic Regression (with CV + Tuning) ===
        if "ElasticNet Logistic Regression" in selected_models:
            with st.expander("🧬 Elastic Net Logistic Regression"):
                st.write("**Hyperparameters**")

                enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning (Grid or Random Search)?", key="enet_tuning")
                enet_model_ready = False

                if enable_tuning:
                    search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="enet_search_method")
                    enet_max_iter = st.slider("Elastic Net: Max Iterations", 100, 2000, 1000, step=100, key="enet_max_iter")
                    c_range = st.slider("C Range (log scale)", 0.01, 10.0, (0.1, 5.0), step=0.1, key="enet_c_range")
                    l1_range = st.slider("L1 Ratio Range", 0.0, 1.0, (0.2, 0.8), step=0.1, key="enet_l1_range")

                    param_grid = {
                        "C": np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), num=5),
                        "l1_ratio": np.linspace(l1_range[0], l1_range[1], num=5)
                    }

                    n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="enet_cv_folds")

                    if st.button("🚀 Train Elastic Net Model with Tuning"):
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
                                enet_search = RandomizedSearchCV(
                                    base_model, param_distributions=param_grid, n_iter=10,
                                    cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42
                                )

                            enet_search.fit(X_train_final, y_train)
                            enet_model = enet_search.best_estimator_

                            st.session_state["enet_model"] = enet_model
                            st.session_state["enet_predictions"] = enet_model.predict(X_train_final)
                            st.session_state["enet_probabilities"] = get_class1_proba(enet_model, X_train_final)

                            st.success(f"Best C: {enet_model.C:.4f}, L1 Ratio: {enet_model.l1_ratio:.2f}")
                            enet_model_ready = True

                else:
                    enet_C = st.slider("Elastic Net: Regularization strength (C)", 0.01, 10.0, 1.0, key="enet_C_manual")
                    enet_max_iter = st.slider("Elastic Net: Max iterations", 100, 2000, 1000, step=100, key="enet_iter_manual")
                    enet_l1_ratio = st.slider("Elastic Net: L1 Ratio (0=L2, 1=L1)", 0.0, 1.0, 0.5, step=0.01, key="enet_l1_ratio_manual")

                    if st.button("🚀 Train Elastic Net Model (Manual)"):
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

                            st.session_state["enet_model"] = enet_model
                            st.session_state["enet_predictions"] = enet_model.predict(X_train_final)
                            st.session_state["enet_probabilities"] = get_class1_proba(enet_model, X_train_final)

                            st.success("Model trained successfully!")
                            enet_model_ready = True

                # === After training (common to both paths) ===
                if "enet_model" in st.session_state:
                    enet_model = st.session_state["enet_model"]

                    # Optional cross-validation
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

                    # Training performance and export
                    df_enet_train_export, enet_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=enet_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="ElasticNet",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="enet_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in enet_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download ElasticNet Training Set with Predictions")
                    csv_enet_train = df_enet_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download ElasticNet Training Data",
                        data=csv_enet_train,
                        file_name="elasticnet_training_predictions.csv",
                        mime="text/csv"
                    )






        # === Partial Least Squares Discriminant Analysis (PLS-DA) ===
        if "PLS-DA" in selected_models:
            from sklearn.cross_decomposition import PLSRegression

            with st.expander("🧪 Partial Least Squares Discriminant Analysis (PLS-DA)"):
                enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for n_components?", key="pls_tuning")
                pls_model_ready = False

                if enable_tuning:
                    search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="pls_search_method")

                    max_comp_limit = min(X_train_final.shape[1], 10)
                    n_components_range = st.slider(
                        "Range of Components to Search", 1, max_comp_limit, (2, max_comp_limit), key="pls_comp_range"
                    )
                    comp_grid = {"n_components": list(range(n_components_range[0], n_components_range[1] + 1))}
                    n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="pls_cv_folds")

                    if st.button("🚀 Train PLS-DA Model with Tuning"):
                        with st.spinner("Running PLS-DA hyperparameter tuning..."):
                            pls_base = PLSRegression()
                            if search_method == "Grid Search":
                                pls_search = GridSearchCV(pls_base, comp_grid, cv=n_folds, scoring='r2')
                            else:
                                pls_search = RandomizedSearchCV(pls_base, comp_grid, n_iter=5, cv=n_folds, scoring='r2', random_state=42)

                            pls_search.fit(X_train_final, y_train)
                            pls_model = pls_search.best_estimator_

                            y_scores_train_pls = pls_model.predict(X_train_final).ravel()
                            y_pred_train_pls = (y_scores_train_pls >= 0.5).astype(int)

                            st.session_state["pls_model"] = pls_model
                            st.session_state["pls_predictions"] = y_pred_train_pls
                            st.session_state["pls_probabilities"] = y_scores_train_pls

                            st.success(f"Best n_components: {pls_model.n_components}")
                            pls_model_ready = True

                else:
                    pls_n_components = st.slider(
                        "PLS-DA: Number of Components", 1, min(X_train_final.shape[1], 10), 2, key="pls_n_components"
                    )

                    if st.button("🚀 Train PLS-DA Model (Manual)"):
                        with st.spinner("Training PLS-DA..."):
                            pls_model = PLSRegression(n_components=pls_n_components)
                            pls_model.fit(X_train_final, y_train)

                            y_scores_train_pls = pls_model.predict(X_train_final).ravel()
                            y_pred_train_pls = (y_scores_train_pls >= 0.5).astype(int)

                            st.session_state["pls_model"] = pls_model
                            st.session_state["pls_predictions"] = y_pred_train_pls
                            st.session_state["pls_probabilities"] = y_scores_train_pls

                            st.success("Model trained successfully!")
                            pls_model_ready = True

                # === After training (common to both paths) ===
                if "pls_model" in st.session_state:
                    pls_model = st.session_state["pls_model"]

                    if st.checkbox("🔁 Run 10-Fold Cross-Validation for PLS-DA?", key="pls_run_cv"):
                        with st.spinner("Running cross-validation..."):
                            y_scores_cv_pls = cross_val_predict(
                                pls_model, X_train_final, y_train, cv=10, method="predict"
                            ).ravel()
                            y_pred_cv_pls = (y_scores_cv_pls >= 0.5).astype(int)

                            st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                            st.text(f"Accuracy:  {accuracy_score(y_train, y_pred_cv_pls):.4f}")
                            st.text(f"Precision: {precision_score(y_train, y_pred_cv_pls):.4f}")
                            st.text(f"Recall:    {recall_score(y_train, y_pred_cv_pls):.4f}")
                            st.text(f"F1-Score:  {f1_score(y_train, y_pred_cv_pls):.4f}")
                            st.text(f"AUC:       {roc_auc_score(y_train, y_scores_cv_pls):.4f}")

                    # === Training performance and export ===
                    from ml_utils import export_training_data_pls_da  # Make sure it's in your helpers

                    df_pls_train_export, pls_metrics = export_training_data_pls_da(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=pls_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="PLSDA",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="pls_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in pls_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download PLS-DA Training Set with Predictions")
                    csv_pls_train = df_pls_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download PLS-DA Training Data",
                        data=csv_pls_train,
                        file_name="plsda_training_predictions.csv",
                        mime="text/csv"
                    )








        # === K-Nearest Neighbors (KNN) with CV + Tuning ===
        if "K-Nearest Neighbors" in selected_models:
            from sklearn.neighbors import KNeighborsClassifier

            with st.expander("📍 K-Nearest Neighbors (KNN)"):
                st.write("**Hyperparameters**")
                enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning?", key="knn_tuning")
                knn_model_ready = False

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

                    if st.button("🚀 Train KNN Model with Tuning"):
                        with st.spinner("Running KNN hyperparameter tuning..."):
                            base_model = KNeighborsClassifier()
                            if search_method == "Grid Search":
                                knn_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                knn_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                                cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                            knn_search.fit(X_train_final, y_train)
                            knn_model = knn_search.best_estimator_

                            st.session_state["knn_model"] = knn_model
                            st.session_state["knn_predictions"] = knn_model.predict(X_train_final)
                            st.session_state["knn_probabilities"] = get_class1_proba(knn_model, X_train_final)

                            st.success(
                                f"Best Parameters: k={knn_model.n_neighbors}, "
                                f"weights={knn_model.weights}, metric={knn_model.metric}"
                            )
                            knn_model_ready = True

                else:
                    knn_n_neighbors = st.slider("KNN: Number of Neighbors (k)", min_value=1, max_value=50, value=5, key="knn_n_neighbors")
                    knn_weights = st.selectbox("KNN: Weight Function", options=["uniform", "distance"], key="knn_weights")
                    knn_metric = st.selectbox("KNN: Distance Metric", options=["minkowski", "euclidean", "manhattan"], key="knn_metric")

                    if st.button("🚀 Train KNN Model (Manual)"):
                        with st.spinner("Training KNN..."):
                            knn_model = KNeighborsClassifier(
                                n_neighbors=knn_n_neighbors,
                                weights=knn_weights,
                                metric=knn_metric
                            )
                            knn_model.fit(X_train_final, y_train)

                            st.session_state["knn_model"] = knn_model
                            st.session_state["knn_predictions"] = knn_model.predict(X_train_final)
                            st.session_state["knn_probabilities"] = get_class1_proba(knn_model, X_train_final)

                            st.success("Model trained successfully!")
                            knn_model_ready = True

                # === After training (common to both paths) ===
                if "knn_model" in st.session_state:
                    knn_model = st.session_state["knn_model"]

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

                    # === Training performance and export ===
                    df_knn_train_export, knn_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=knn_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="KNN",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="knn_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in knn_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download KNN Training Set with Predictions")
                    csv_knn_train = df_knn_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download KNN Training Data",
                        data=csv_knn_train,
                        file_name="knn_training_predictions.csv",
                        mime="text/csv"
                    )









        # === Naive Bayes (GaussianNB) with CV + Tuning ===
        if "Naive Bayes" in selected_models:
            from sklearn.naive_bayes import GaussianNB

            with st.expander("📦 Naive Bayes (GaussianNB)"):
                st.write("Naive Bayes assumes feature independence and models each feature using a normal distribution.")

                enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for Naive Bayes?", key="nb_tuning")
                nb_model_ready = False

                if enable_tuning:
                    search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="nb_search_method")

                    smoothing_range = st.slider(
                        "Variance Smoothing Range (log scale)", -12, -2, (-9, -6),
                        key="nb_smoothing_range"
                    )
                    smoothing_values = np.logspace(smoothing_range[0], smoothing_range[1], num=5)
                    param_grid = {"var_smoothing": smoothing_values}

                    n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="nb_cv_folds")

                    if st.button("🚀 Train Naive Bayes with Tuning"):
                        with st.spinner("Running hyperparameter tuning for Naive Bayes..."):
                            base_model = GaussianNB()
                            if search_method == "Grid Search":
                                nb_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                nb_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=5,
                                                            cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                            nb_search.fit(X_train_final, y_train)
                            nb_model = nb_search.best_estimator_

                            st.session_state["nb_model"] = nb_model
                            st.session_state["nb_predictions"] = nb_model.predict(X_train_final)
                            st.session_state["nb_probabilities"] = get_class1_proba(nb_model, X_train_final)

                            st.success(f"Best var_smoothing: {nb_model.var_smoothing:.1e}")
                            nb_model_ready = True

                else:
                    if st.button("🚀 Train Naive Bayes (Manual)"):
                        with st.spinner("Training Naive Bayes..."):
                            nb_model = GaussianNB()
                            nb_model.fit(X_train_final, y_train)

                            st.session_state["nb_model"] = nb_model
                            st.session_state["nb_predictions"] = nb_model.predict(X_train_final)
                            st.session_state["nb_probabilities"] = get_class1_proba(nb_model, X_train_final)

                            st.success("Model trained successfully!")
                            nb_model_ready = True

                # === After training (common to both paths) ===
                if "nb_model" in st.session_state:
                    nb_model = st.session_state["nb_model"]

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

                    # === Training performance and export ===
                    df_nb_train_export, nb_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=nb_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="NB",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="nb_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in nb_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download Naive Bayes Training Set with Predictions")
                    csv_nb_train = df_nb_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Naive Bayes Training Data",
                        data=csv_nb_train,
                        file_name="naive_bayes_training_predictions.csv",
                        mime="text/csv"
                    )








        # === Support Vector Machine (SVM) with CV + Tuning ===
        if "Support Vector Machine" in selected_models:
            from sklearn.svm import SVC

            with st.expander("🔲 Support Vector Machine (SVM)"):
                st.write("**Hyperparameters**")
                enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for SVM?", key="svm_tuning")
                svm_model_ready = False

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

                    if st.button("🚀 Train SVM with Tuning"):
                        with st.spinner("Running SVM hyperparameter tuning..."):
                            base_model = SVC(probability=True, random_state=42)
                            if search_method == "Grid Search":
                                svm_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                svm_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                                cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                            svm_search.fit(X_train_final, y_train)
                            svm_model = svm_search.best_estimator_

                            st.session_state["svm_model"] = svm_model
                            st.session_state["svm_predictions"] = svm_model.predict(X_train_final)
                            st.session_state["svm_probabilities"] = get_class1_proba(svm_model, X_train_final)

                            st.success(
                                f"Best Parameters: C={svm_model.C}, kernel={svm_model.kernel}, gamma={svm_model.gamma}"
                            )
                            svm_model_ready = True

                else:
                    svm_kernel = st.selectbox("SVM: Kernel", ['linear', 'rbf', 'poly', 'sigmoid'], index=1, key="svm_kernel")
                    svm_C = st.slider("SVM: Regularization parameter (C)", 0.01, 10.0, 1.0, key="svm_C")
                    svm_gamma = st.selectbox("SVM: Gamma", ['scale', 'auto'], key="svm_gamma")

                    if st.button("🚀 Train SVM (Manual)"):
                        with st.spinner("Training Support Vector Machine..."):
                            svm_model = SVC(
                                C=svm_C,
                                kernel=svm_kernel,
                                gamma=svm_gamma,
                                probability=True,
                                random_state=42
                            )
                            svm_model.fit(X_train_final, y_train)

                            st.session_state["svm_model"] = svm_model
                            st.session_state["svm_predictions"] = svm_model.predict(X_train_final)
                            st.session_state["svm_probabilities"] = get_class1_proba(svm_model, X_train_final)

                            st.success("Model trained successfully!")
                            svm_model_ready = True

                # === After training (common to both paths) ===
                if "svm_model" in st.session_state:
                    svm_model = st.session_state["svm_model"]

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

                    # === Training performance and export ===
                    df_svm_train_export, svm_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=svm_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="SVM",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="svm_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in svm_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download SVM Training Set with Predictions")
                    csv_svm_train = df_svm_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download SVM Training Data",
                        data=csv_svm_train,
                        file_name="svm_training_predictions.csv",
                        mime="text/csv"
                    )





        # === Decision Tree Classifier with CV + Tuning ===
        if "Decision Tree" in selected_models:
            from sklearn.tree import DecisionTreeClassifier

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

                    if st.button("🚀 Train Decision Tree with Tuning"):
                        with st.spinner("Running Decision Tree hyperparameter tuning..."):
                            base_model = DecisionTreeClassifier(random_state=42)

                            if search_method == "Grid Search":
                                tree_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                tree_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                                cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                            tree_search.fit(X_train_final, y_train)
                            tree_model = tree_search.best_estimator_

                            st.session_state["tree_model"] = tree_model
                            st.session_state["tree_predictions"] = tree_model.predict(X_train_final)
                            st.session_state["tree_probabilities"] = get_class1_proba(tree_model, X_train_final)

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

                    if st.button("🚀 Train Decision Tree (Manual)"):
                        with st.spinner("Training Decision Tree..."):
                            tree_model = DecisionTreeClassifier(
                                max_depth=tree_max_depth,
                                min_samples_split=tree_min_samples_split,
                                min_samples_leaf=tree_min_samples_leaf,
                                criterion=tree_criterion,
                                random_state=42
                            )
                            tree_model.fit(X_train_final, y_train)

                            st.session_state["tree_model"] = tree_model
                            st.session_state["tree_predictions"] = tree_model.predict(X_train_final)
                            st.session_state["tree_probabilities"] = get_class1_proba(tree_model, X_train_final)

                            st.success("Model trained successfully!")

                # === Metrics and Export ===
                if "tree_model" in st.session_state:
                    tree_model = st.session_state["tree_model"]

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

                    # === Training performance and export ===
                    df_dt_train_export, tree_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=tree_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="DT",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="tree_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in tree_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download Decision Tree Training Set with Predictions")
                    csv_dt_train = df_dt_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Decision Tree Training Data",
                        data=csv_dt_train,
                        file_name="decision_tree_training_predictions.csv",
                        mime="text/csv"
                    )






        # === Random Forest Classifier with CV + Tuning ===
        if "Random Forest" in selected_models:
            from sklearn.ensemble import RandomForestClassifier

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
                        "max_depth": list(range(depth_range[0], depth_range[1] + 1)),
                        "min_samples_leaf": list(range(leaf_range[0], leaf_range[1] + 1))
                    }

                    n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="rf_cv_folds")

                    if st.button("🚀 Train Random Forest with Tuning"):
                        with st.spinner("Running Random Forest hyperparameter tuning..."):
                            base_model = RandomForestClassifier(random_state=42)

                            if search_method == "Grid Search":
                                rf_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                rf_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                            cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                            rf_search.fit(X_train_final, y_train)
                            rf_model = rf_search.best_estimator_

                            st.session_state["rf_model"] = rf_model
                            st.session_state["rf_predictions"] = rf_model.predict(X_train_final)
                            st.session_state["rf_probabilities"] = get_class1_proba(rf_model, X_train_final)

                            st.success(
                                f"Best Parameters: n_estimators={rf_model.n_estimators}, "
                                f"max_depth={rf_model.max_depth}, min_samples_leaf={rf_model.min_samples_leaf}"
                            )

                else:
                    n_estimators = st.slider("Number of Trees", 10, 200, 100, key="rf_n_estimators")
                    max_depth = st.slider("Max Depth", 1, 20, 5, key="rf_max_depth")

                    if st.button("🚀 Train Random Forest (Manual)"):
                        with st.spinner("Training Random Forest..."):
                            rf_model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=42
                            )
                            rf_model.fit(X_train_final, y_train)

                            st.session_state["rf_model"] = rf_model
                            st.session_state["rf_predictions"] = rf_model.predict(X_train_final)
                            st.session_state["rf_probabilities"] = get_class1_proba(rf_model, X_train_final)

                            st.success("Model trained successfully!")

                # === Evaluation & Download Section ===
                if "rf_model" in st.session_state:
                    rf_model = st.session_state["rf_model"]

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

                    # === Training performance and export using helper ===
                    df_rf_train_export, rf_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=rf_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="RF",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="rf_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in rf_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download Random Forest Training Set with Predictions")
                    csv_rf_train = df_rf_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Random Forest Training Data",
                        data=csv_rf_train,
                        file_name="random_forest_training_predictions.csv",
                        mime="text/csv"
                    )







        # === Gradient Boosting Machine (GBM) with CV + Tuning ===
        if "Gradient Boosting" in selected_models:
            from sklearn.ensemble import GradientBoostingClassifier

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

                    if st.button("🚀 Train Gradient Boosting with Tuning"):
                        with st.spinner("Running GBM hyperparameter tuning..."):
                            base_model = GradientBoostingClassifier(random_state=42)

                            if search_method == "Grid Search":
                                gbm_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                gbm_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                                cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                            gbm_search.fit(X_train_final, y_train)
                            gbm_model = gbm_search.best_estimator_

                            st.session_state["gbm_model"] = gbm_model
                            st.session_state["gbm_predictions"] = gbm_model.predict(X_train_final)
                            st.session_state["gbm_probabilities"] = get_class1_proba(gbm_model, X_train_final)

                            st.success("Best Parameters Selected via Tuning.")

                else:
                    gbm_n_estimators = st.slider("GBM: Number of Estimators", 10, 500, 100, key="gbm_n_estimators")
                    gbm_learning_rate = st.slider("GBM: Learning Rate", 0.01, 1.0, 0.1, step=0.01, key="gbm_learning_rate")
                    gbm_max_depth = st.slider("GBM: Max Depth", 1, 10, 3, key="gbm_max_depth")
                    gbm_subsample = st.slider("GBM: Subsample", 0.1, 1.0, 1.0, step=0.1, key="gbm_subsample")
                    gbm_min_samples_split = st.slider("GBM: Min Samples Split", 2, 20, 2, key="gbm_min_samples_split")
                    gbm_min_samples_leaf = st.slider("GBM: Min Samples Leaf", 1, 20, 1, key="gbm_min_samples_leaf")

                    if st.button("🚀 Train Gradient Boosting (Manual)"):
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

                            st.session_state["gbm_model"] = gbm_model
                            st.session_state["gbm_predictions"] = gbm_model.predict(X_train_final)
                            st.session_state["gbm_probabilities"] = get_class1_proba(gbm_model, X_train_final)

                            st.success("Model trained successfully.")

                # === Evaluation + CV + Export via Helper ===
                if "gbm_model" in st.session_state:
                    gbm_model = st.session_state["gbm_model"]

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

                    df_gbm_train_export, gbm_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=gbm_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="GB",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="gbm_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in gbm_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download Gradient Boosting Training Set with Predictions")
                    csv_gb_train = df_gbm_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Gradient Boosting Training Data",
                        data=csv_gb_train,
                        file_name="gradient_boosting_training_predictions.csv",
                        mime="text/csv"
                    )







        # === Neural Network (MLPClassifier) with CV + Tuning ===
        if "Neural Network" in selected_models:
            from sklearn.neural_network import MLPClassifier

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

                    if st.button("🚀 Train Neural Network with Tuning"):
                        with st.spinner("Running Neural Network hyperparameter tuning..."):
                            base_model = MLPClassifier(solver='adam', max_iter=max_iter, random_state=42)

                            if search_method == "Grid Search":
                                nn_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                nn_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                            cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                            nn_search.fit(X_train_final, y_train)
                            nn_model = nn_search.best_estimator_

                            st.session_state["nn_model"] = nn_model
                            st.session_state["nn_predictions"] = nn_model.predict(X_train_final)
                            st.session_state["nn_probabilities"] = get_class1_proba(nn_model, X_train_final)

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

                    if st.button("🚀 Train Neural Network (Manual)"):
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

                            st.session_state["nn_model"] = nn_model
                            st.session_state["nn_predictions"] = nn_model.predict(X_train_final)
                            st.session_state["nn_probabilities"] = get_class1_proba(nn_model, X_train_final)

                            st.success("Model trained successfully.")

                # === Evaluation + CV + Download via Helper ===
                if "nn_model" in st.session_state:
                    nn_model = st.session_state["nn_model"]

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

                    df_nn_train_export, nn_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=nn_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="NN",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="nn_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in nn_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download Neural Network Training Set with Predictions")
                    csv_nn_train = df_nn_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Neural Network Training Data",
                        data=csv_nn_train,
                        file_name="neural_network_training_predictions.csv",
                        mime="text/csv"
                    )




        # === Bagging Classifier (Manual + Tuning) ===
        if "Bagging" in selected_models:

            with st.expander("🌿 Bagging Classifier"):
                st.write("**Hyperparameters**")
                enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for Bagging?", key="bag_tuning")

                use_base_estimator = version.parse(sklearn_version) < version.parse("1.2")

                if enable_tuning:
                    search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="bag_search_method")

                    n_estimators_range = st.slider("Number of Estimators", 10, 200, (50, 100), step=10, key="bag_n_range")
                    max_samples_range = st.slider("Max Samples (as % of training)", 0.1, 1.0, (0.5, 1.0), step=0.1, key="bag_samp_range")
                    bootstrap_choice = st.radio("Sampling Strategy", ["Bootstrap (Bagging)", "No Bootstrap (Pasting)"], key="bag_bootstrap")

                    param_grid = {
                        "n_estimators": list(range(n_estimators_range[0], n_estimators_range[1] + 1, 10)),
                        "max_samples": np.linspace(max_samples_range[0], max_samples_range[1], 5),
                        "bootstrap": [bootstrap_choice == "Bootstrap (Bagging)"]
                    }

                    n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="bag_cv_folds")

                    if st.button("🚀 Train Bagging Classifier with Tuning"):
                        with st.spinner("Running Bagging hyperparameter tuning..."):
                            base_args = {"random_state": 42}
                            if use_base_estimator:
                                base_args["base_estimator"] = DecisionTreeClassifier()
                            else:
                                base_args["estimator"] = DecisionTreeClassifier()

                            base_model = BaggingClassifier(**base_args)

                            if search_method == "Grid Search":
                                bag_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                bag_search = RandomizedSearchCV(
                                    base_model, param_distributions=param_grid, n_iter=10,
                                    cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42
                                )

                            bag_search.fit(X_train_final, y_train)
                            bag_model = bag_search.best_estimator_

                            st.session_state["bag_model"] = bag_model
                            st.session_state["bag_predictions"] = bag_model.predict(X_train_final)
                            st.session_state["bag_probabilities"] = get_class1_proba(bag_model, X_train_final)

                            st.success("✅ Best Bagging Parameters selected.")

                else:
                    bag_n_estimators = st.slider("Number of Estimators", 10, 200, 100, key="bag_n_estimators")
                    bag_max_samples = st.slider("Max Samples", 0.1, 1.0, 1.0, step=0.1, key="bag_max_samples")
                    bag_bootstrap = st.checkbox("Use Bootstrap Sampling (Bagging)?", value=True, key="bag_bootstrap_manual")

                    if st.button("🚀 Train Bagging Classifier (Manual)"):
                        with st.spinner("Training Bagging Classifier..."):
                            base_args = {
                                "n_estimators": bag_n_estimators,
                                "max_samples": bag_max_samples,
                                "bootstrap": bag_bootstrap,
                                "random_state": 42
                            }
                            if use_base_estimator:
                                base_args["base_estimator"] = DecisionTreeClassifier()
                            else:
                                base_args["estimator"] = DecisionTreeClassifier()

                            bag_model = BaggingClassifier(**base_args)
                            bag_model.fit(X_train_final, y_train)

                            st.session_state["bag_model"] = bag_model
                            st.session_state["bag_predictions"] = bag_model.predict(X_train_final)
                            st.session_state["bag_probabilities"] = get_class1_proba(bag_model, X_train_final)

                            st.success("✅ Bagging Classifier trained successfully!")

                # === Evaluation & Download Section ===
                if "bag_model" in st.session_state:
                    bag_model = st.session_state["bag_model"]

                    if st.checkbox("🔁 Run 10-Fold Cross-Validation for Bagging?", key="bag_run_cv"):
                        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                        with st.spinner("Running cross-validation..."):
                            cv_results = cross_validate(
                                bag_model, X_train_final, y_train,
                                cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                            )

                        st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                        for metric in scoring:
                            mean_score = cv_results[f'test_{metric}'].mean()
                            std_score = cv_results[f'test_{metric}'].std()
                            st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")

                    # Export and training performance
                    df_bag_train_export, bag_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=bag_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="Bagging",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="bag_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in bag_metrics.items():
                        st.text(f"{metric}: {value:.4f}" if value is not None else f"{metric}: N/A")

                    st.markdown("#### 📥 Download Bagging Classifier Training Set with Predictions")
                    csv_bag_train = df_bag_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Bagging Training Data",
                        data=csv_bag_train,
                        file_name="bagging_training_predictions.csv",
                        mime="text/csv"
                    )










        # === Pasting Classifier (Bagging with bootstrap=False) ===
        if "Pasting" in selected_models:

            with st.expander("📦 Pasting Classifier (No Replacement)"):
                st.write("**Hyperparameters**")
                enable_tuning = st.checkbox("🔍 Enable Hyperparameter Tuning for Pasting?", key="pasting_tuning")

                use_base_estimator = version.parse(sklearn_version) < version.parse("1.2")

                if enable_tuning:
                    search_method = st.radio("Search Method:", ["Grid Search", "Random Search"], key="pasting_search_method")

                    estimator_range = st.slider("Number of Base Estimators", 10, 200, (50, 150), step=10, key="pasting_n_range")
                    max_samples_range = st.slider("Max Samples (Fraction)", 0.1, 1.0, (0.5, 1.0), step=0.1, key="pasting_sample_range")

                    param_grid = {
                        "n_estimators": list(range(estimator_range[0], estimator_range[1] + 1, 10)),
                        "max_samples": np.linspace(max_samples_range[0], max_samples_range[1], 5),
                        "bootstrap": [False]  # Ensure it's always pasting
                    }

                    n_folds = st.slider("Cross-validation folds", 3, 10, 10, key="pasting_cv_folds")

                    if st.button("🚀 Train Pasting Classifier with Tuning"):
                        with st.spinner("Running Pasting hyperparameter tuning..."):
                            base_args = {"bootstrap": False, "random_state": 42}
                            if use_base_estimator:
                                base_args["base_estimator"] = DecisionTreeClassifier()
                            else:
                                base_args["estimator"] = DecisionTreeClassifier()

                            base_model = BaggingClassifier(**base_args)

                            if search_method == "Grid Search":
                                paste_search = GridSearchCV(base_model, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1)
                            else:
                                paste_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10,
                                                                cv=n_folds, scoring='roc_auc', n_jobs=-1, random_state=42)

                            paste_search.fit(X_train_final, y_train)
                            paste_model = paste_search.best_estimator_

                            st.session_state["past_model"] = paste_model
                            st.session_state["past_predictions"] = paste_model.predict(X_train_final)
                            st.session_state["past_probabilities"] = get_class1_proba(paste_model, X_train_final)

                            st.success(
                                f"Best Parameters: n_estimators={paste_model.n_estimators}, "
                                f"max_samples={paste_model.max_samples}"
                            )

                else:
                    n_estimators = st.slider("Number of Base Estimators", 10, 200, 100, key="pasting_n_estimators")
                    max_samples = st.slider("Max Samples (Fraction)", 0.1, 1.0, 1.0, step=0.1, key="pasting_max_samples")

                    if st.button("🚀 Train Pasting Classifier (Manual)"):
                        with st.spinner("Training Pasting..."):
                            base_args = {
                                "n_estimators": n_estimators,
                                "max_samples": max_samples,
                                "bootstrap": False,
                                "random_state": 42
                            }
                            if use_base_estimator:
                                base_args["base_estimator"] = DecisionTreeClassifier()
                            else:
                                base_args["estimator"] = DecisionTreeClassifier()

                            paste_model = BaggingClassifier(**base_args)
                            paste_model.fit(X_train_final, y_train)

                            st.session_state["past_model"] = paste_model
                            st.session_state["past_predictions"] = paste_model.predict(X_train_final)
                            st.session_state["past_probabilities"] = get_class1_proba(paste_model, X_train_final)

                            st.success("✅ Pasting Classifier trained successfully!")

                # === Evaluation & Export ===
                if "past_model" in st.session_state:
                    paste_model = st.session_state["past_model"]

                    if st.checkbox("🔁 Run 10-Fold Cross-Validation for Pasting?", key="pasting_run_cv"):
                        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                        with st.spinner("Running cross-validation..."):
                            cv_results = cross_validate(
                                paste_model, X_train_final, y_train,
                                cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                            )

                        st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                        for metric in scoring:
                            mean_score = cv_results[f'test_{metric}'].mean()
                            std_score = cv_results[f'test_{metric}'].std()
                            st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")

                    df_paste_train_export, paste_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=paste_model,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="Pasting",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="pasting_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in paste_metrics.items():
                        st.text(f"{metric}: {value:.4f}" if value is not None else f"{metric}: N/A")

                    st.markdown("#### 📥 Download Pasting Training Set with Predictions")
                    csv_paste_train = df_paste_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Pasting Training Data",
                        data=csv_paste_train,
                        file_name="pasting_training_predictions.csv",
                        mime="text/csv"
                    )











        # === Stacking Classifier (Stacking Ensemble) ===
        if "Stacking" in selected_models:

            with st.expander("📚 Stacking Classifier (Meta-Ensemble)"):
                st.write("Stacking uses base learners to feed predictions into a final estimator.")
                st.write("Only trained models with `predict_proba` support can be stacked.")

                model_map = {
                    "Ridge Logistic Regression": "ridge_model",
                    "Lasso Logistic Regression": "lasso_model",
                    "ElasticNet Logistic Regression": "enet_model",
                    "Random Forest": "rf_model",
                    "Decision Tree": "tree_model",
                    "Support Vector Machine": "svm_model",
                    "Gradient Boosting": "gbm_model",
                    "K-Nearest Neighbors": "knn_model",
                    "Naive Bayes": "nb_model",
                    "Neural Network": "nn_model",
                    "Bagging": "bag_model",
                    "Pasting": "past_model"
                }

                available_models = []
                model_names = []

                for name, var_name in model_map.items():
                    if name in selected_models and var_name in st.session_state:
                        available_models.append((name[:3], st.session_state[var_name]))
                        model_names.append(name)

                if len(available_models) < 2:
                    st.warning("⚠️ Please select and train at least two models to stack.")
                else:
                    meta_model_choice = st.selectbox(
                        "Meta-model (Final Estimator)",
                        ["Logistic Regression", "Random Forest", "Ridge", "Lasso", "ElasticNet"],
                        key="stacking_meta"
                    )

                    if st.button("🚀 Train Stacking Classifier"):
                        with st.spinner("Training Stacking Classifier..."):
                            # Choose final estimator
                            if meta_model_choice == "Logistic Regression":
                                final_estimator = LogisticRegression(solver="lbfgs", max_iter=1000)
                            elif meta_model_choice == "Ridge":
                                from sklearn.linear_model import RidgeClassifier
                                final_estimator = RidgeClassifier()
                            elif meta_model_choice == "Lasso":
                                from sklearn.linear_model import LogisticRegression
                                final_estimator = LogisticRegression(penalty='l1', solver='saga', max_iter=1000)
                            elif meta_model_choice == "ElasticNet":
                                from sklearn.linear_model import LogisticRegression
                                final_estimator = LogisticRegression(penalty='elasticnet', solver='saga',
                                                                    l1_ratio=0.5, max_iter=1000)
                            else:
                                final_estimator = RandomForestClassifier(n_estimators=100, random_state=42)

                            stacking_model = StackingClassifier(
                                estimators=available_models,
                                final_estimator=final_estimator,
                                passthrough=False,
                                cv=5,
                                n_jobs=-1
                            )

                            stacking_model.fit(X_train_final, y_train)

                            y_pred_stack = stacking_model.predict(X_train_final)
                            y_prob_stack = get_class1_proba(stacking_model, X_train_final)

                            st.session_state["stack_model"] = stacking_model
                            st.session_state["stack_predictions"] = y_pred_stack
                            st.session_state["stack_probabilities"] = y_prob_stack

                            st.success(f"✅ Stacking model trained with: {', '.join(model_names)} → {meta_model_choice}")

            if "stack_model" in st.session_state:
                stacking_model = st.session_state["stack_model"]

                if st.checkbox("🔁 Run 10-Fold Cross-Validation for Stacking?", key="stacking_run_cv"):
                    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                    with st.spinner("Running cross-validation..."):
                        cv_results = cross_validate(
                            stacking_model, X_train_final, y_train,
                            cv=10, scoring=scoring, return_train_score=False, n_jobs=-1
                        )

                    st.markdown("**📊 10-Fold Cross-Validation Results (Train Set)**")
                    for metric in scoring:
                        mean_score = cv_results[f'test_{metric}'].mean()
                        std_score = cv_results[f'test_{metric}'].std()
                        st.text(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")

                df_stack_train_export, stack_metrics = export_training_data_general(
                    X_train_final=X_train_final,
                    y_train_raw=y_train,
                    model=stacking_model,
                    row_ids=st.session_state.get("row_id_train"),
                    model_name="Stacking",
                    use_original_labels=True,
                    flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="stacking_flip_export"),
                    label_map=st.session_state.get("label_map_")
                )

                st.markdown("**📊 Training Set Performance**")
                for metric, value in stack_metrics.items():
                    st.text(f"{metric}: {value:.4f}" if value is not None else f"{metric}: N/A")

                st.markdown("#### 📥 Download Stacking Training Set with Predictions")
                csv_stack_train = df_stack_train_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Stacking Training Data",
                    data=csv_stack_train,
                    file_name="stacking_training_predictions.csv",
                    mime="text/csv"
                )




        # === Voting Classifier (Soft Voting Only) with CV ===
        from sklearn.ensemble import VotingClassifier

        if "Voting Classifier" in selected_models:
            with st.expander("🗳️ Voting Classifier (Soft Voting Ensemble)"):
                st.write("**Soft voting averages predicted probabilities across models.**")
                st.write("All models must support `predict_proba()`. At least 2 models are required.")
                st.write("PLS-DA and Stacking are excluded from voting ensemble due to incompatibility or nesting.")

                # All eligible models for voting
                voting_model_map = {
                    "Ridge Logistic Regression": "ridge_model",
                    "Lasso Logistic Regression": "lasso_model",
                    "ElasticNet Logistic Regression": "enet_model",
                    "Random Forest": "rf_model",
                    "Decision Tree": "tree_model",
                    "Support Vector Machine": "svm_model",
                    "Gradient Boosting": "gbm_model",
                    "K-Nearest Neighbors": "knn_model",
                    "Naive Bayes": "nb_model",
                    "Neural Network": "nn_model",
                    "Bagging": "bagging_model",
                    "Pasting": "pasting_model"
                }

                # Filter models that are trained and available
                trained_voting_models = {
                    name: st.session_state[var_name]
                    for name, var_name in voting_model_map.items()
                    if name in selected_models and var_name in st.session_state
                }

                # Let user choose which trained models to include in voting
                vote_model_choices = st.multiselect(
                    "Select models to include in the Voting Classifier:",
                    options=list(trained_voting_models.keys()),
                    default=list(trained_voting_models.keys()),
                    key="voting_model_selection"
                )

                if len(vote_model_choices) < 2:
                    st.warning("⚠️ Please select at least two trained models.")
                else:
                    selected_estimators = [
                        (name[:3], trained_voting_models[name]) for name in vote_model_choices
                    ]

                    if st.button("🚀 Train Voting Classifier", key="vote_train_btn"):
                        with st.spinner("Training Voting Classifier..."):
                            voting_clf = VotingClassifier(
                                estimators=selected_estimators,
                                voting="soft"
                            )
                            voting_clf.fit(X_train_final, y_train)

                            y_pred_vote = voting_clf.predict(X_train_final)
                            y_prob_vote = get_class1_proba(voting_clf, X_train_final)

                            st.session_state["voting_model"] = voting_clf
                            st.session_state["voting_predictions"] = y_pred_vote
                            st.session_state["voting_probabilities"] = y_prob_vote

                            st.success(f"✅ Voting Classifier trained using: {', '.join(vote_model_choices)}")

                # === Evaluation and Export ===
                if "voting_model" in st.session_state:
                    voting_clf = st.session_state["voting_model"]

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

                    df_vote_train_export, vote_metrics = export_training_data_general(
                        X_train_final=X_train_final,
                        y_train_raw=y_train,
                        model=voting_clf,
                        row_ids=st.session_state.get("row_id_train"),
                        model_name="Voting",
                        use_original_labels=True,
                        flip_outputs=st.checkbox("🔄 Flip training predictions for export?", key="vote_flip_export"),
                        label_map=st.session_state.get("label_map_")
                    )

                    st.markdown("**📊 Training Set Performance**")
                    for metric, value in vote_metrics.items():
                        if value is not None:
                            st.text(f"{metric}: {value:.4f}")
                        else:
                            st.text(f"{metric}: N/A")

                    st.markdown("#### 📥 Download Voting Classifier Training Set with Predictions")
                    csv_vote_train = df_vote_train_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Voting Classifier Training Data",
                        data=csv_vote_train,
                        file_name="voting_classifier_training_predictions.csv",
                        mime="text/csv"
                    )












##########################################################################################################################
######################################             Validation             ################################################
##########################################################################################################################

    # === Final Validation Set Evaluation ===
    if st.session_state.get("models_confirmed", False):
        st.subheader("📊 Final Validation Set Comparison (Full Metrics)")

        selected_models = st.session_state.get("selected_models", [])
        val_predictions = {}

        # Map model names to session state keys
        model_keys = {
            "Ridge Logistic Regression": "ridge_model",
            "Lasso Logistic Regression": "lasso_model",
            "ElasticNet Logistic Regression": "enet_model",
            "PLS-DA": "pls_model",
            "Random Forest": "rf_model",
            "K-Nearest Neighbors": "knn_model",
            "Naive Bayes": "nb_model",
            "Support Vector Machine": "svm_model",
            "Decision Tree": "tree_model",
            "Gradient Boosting": "gbm_model",
            "Neural Network": "nn_model",
            "Bagging": "bag_model",
            "Pasting": "past_model",
            "Stacking": "stack_model",
            "Voting Classifier": "voting_model"
        }

        # Run validation predictions
        for model_name in selected_models:
            model_key = model_keys.get(model_name)
            if model_key in st.session_state:
                model = st.session_state[model_key]
                try:
                    if model_name == "PLS-DA":
                        scores = model.predict(X_val_final).ravel()
                        preds = (scores >= 0.5).astype(int)
                        val_predictions[model_name] = (preds, scores)
                    else:
                        preds = model.predict(X_val_final)
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X_val_final)[:, 1]
                        else:
                            # Fallback: Use decision function if available (e.g., some stacking setups)
                            if hasattr(model, "decision_function"):
                                probs = model.decision_function(X_val_final)
                                probs = 1 / (1 + np.exp(-probs))  # Sigmoid to get probabilities
                            else:
                                probs = np.full_like(preds, fill_value=np.nan, dtype=np.float64)
                        val_predictions[model_name] = (preds, probs)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate validation predictions for {model_name}: {e}")

        # Compute metrics for all successfully evaluated models
        def compute_metrics(y_true, y_pred, y_prob, model_name):
            return {
                "Model": model_name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred),
                "F1-Score": f1_score(y_true, y_pred),
                "AUC": roc_auc_score(y_true, y_prob) if not np.isnan(y_prob).any() else None
            }

        metrics = [
            compute_metrics(y_val, y_pred, y_prob, model_name)
            for model_name, (y_pred, y_prob) in val_predictions.items()
        ]

        # Show comparison table
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
                df_test = pd.read_csv(filepath) if file_format == "csv" else \
                        pd.read_excel(filepath) if file_format == "xlsx" else \
                        pd.read_json(filepath)
                st.success(f"✅ Loaded sample test file: {dataset} ({file_format})")
            except Exception as e:
                st.error(f"❌ Could not load sample test file: {e}")
                st.stop()

        else:
            test_file = st.file_uploader("Upload a test dataset (same structure as training data):", type=["csv", "xlsx", "json"], key="test_file")
            if test_file is not None:
                try:
                    df_test = pd.read_csv(test_file) if test_file.name.endswith(".csv") else \
                            pd.read_excel(test_file) if test_file.name.endswith(".xlsx") else \
                            pd.read_json(test_file)
                    st.success("✅ Test file loaded successfully.")
                except Exception as e:
                    st.error(f"Error reading test file: {e}")
                    st.stop()

        # === Preview and Process Test Data ===
        if df_test is not None:
            st.dataframe(df_test.head())

            df_test_original = df_test.copy()

            # === Ensure row_id exists for tracking ===
            if "row_id" not in df_test.columns:
                df_test.insert(0, "row_id", np.arange(1, len(df_test) + 1))

            # === Preserve original row_id for final output ===
            row_ids = df_test["row_id"].copy()

            # === Columns used during training (excluding row_id) ===
            expected_columns = st.session_state.get("selected_columns", [])
            expected_columns = [col for col in expected_columns if col != "row_id"]

            # === Validate presence of expected columns ===
            missing_columns = [col for col in expected_columns if col not in df_test.columns]
            if missing_columns:
                st.error(f"❌ Your test file is missing these required columns: {missing_columns}")
                st.stop()

            # === Filter test data to expected columns ===
            df_test = df_test.reindex(columns=expected_columns, fill_value=0)

            # === Encode categorical variables ===
            df_test_encoded = pd.get_dummies(df_test, drop_first=True)

            # === Reapply missing value handling steps from training ===
            if "missing_value_steps" in st.session_state:
                for method, col, value in st.session_state["missing_value_steps"]:
                    if col not in df_test_encoded.columns:
                        continue  # Skip columns not in test set

                    if method == "drop_rows":
                        df_test_encoded = df_test_encoded[df_test_encoded[col].notna()]
                        row_ids = row_ids.loc[df_test_encoded.index].reset_index(drop=True)

                    elif method == "drop_column":
                        df_test_encoded.drop(columns=[col], inplace=True)

                    elif method == "zero":
                        df_test_encoded[col].fillna(0, inplace=True)

                    elif method == "mean":
                        df_test_encoded[col].fillna(value, inplace=True)

                    elif method == "median":
                        df_test_encoded[col].fillna(value, inplace=True)


            # === Align with training set columns ===
            training_cols = st.session_state["X_raw"].columns
            missing_in_test = set(training_cols) - set(df_test_encoded.columns)
            for col in missing_in_test:
                df_test_encoded[col] = 0
            df_test_encoded = df_test_encoded[training_cols].astype("float64")

            # === Drop rows with invalid values ===
            df_test_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
            invalid_rows = df_test_encoded.isnull().any(axis=1)
            if invalid_rows.any():
                st.warning(f"⚠️ Removed {invalid_rows.sum()} rows with NaNs/infs.")
                df_test_encoded = df_test_encoded[~invalid_rows]
                df_test = df_test[~invalid_rows]
                df_test_original = df_test_original[~invalid_rows]
                row_ids = row_ids[~invalid_rows].reset_index(drop=True)

            # === Check for target column ===
            has_target_column = st.radio(
                "Does your test set include the target column (e.g. diagnosis)?",
                ["Yes", "No"],
                index=0,
                key="has_target_in_test"
            )

            target_column = st.session_state.get("target_column", None)
            target_column_present = (
                has_target_column == "Yes" and
                target_column is not None and
                target_column in df_test.columns
            )

            if target_column_present:
                st.markdown(f"✅ Target column **`{target_column}`** detected in test set.")
                st.markdown("#### 📊 Test Set Target Value Distribution (Raw)")
                st.dataframe(df_test[target_column].value_counts())

                df_test_target = df_test[[target_column]].copy()

                label_classes = st.session_state.get("label_classes_", None)
                label_map = st.session_state.get("label_map_", None)

                if label_classes is None or label_map is None:
                    st.error("❌ Missing label mapping from training. Please re-run training.")
                    st.stop()

                test_labels = set(df_test_target[target_column].dropna().unique())
                expected_labels = set(label_classes)
                if test_labels != expected_labels:
                    st.error(f"❌ Test labels do not match training labels: {sorted(expected_labels)}. Found: {sorted(test_labels)}")
                    st.stop()

                df_test_target["encoded_target"] = df_test_target[target_column].map(label_map)
                df_test_target = df_test_target.dropna(subset=["encoded_target"]).astype({"encoded_target": "int64"})

                # Align rows across test structures
                df_test_encoded = df_test_encoded.loc[df_test_target.index].reset_index(drop=True)
                df_test_original = df_test_original.loc[df_test_target.index].reset_index(drop=True)
                row_ids = row_ids.loc[df_test_target.index].reset_index(drop=True)

                df_test_target_final = df_test_target["encoded_target"]
                st.markdown("#### ✅ Encoded Target Value Distribution")
                st.dataframe(df_test_target_final.value_counts())
            else:
                st.info("ℹ️ No target column found in test set. Predictions will be made but evaluation metrics will be skipped.")

            # === Apply all stored transformation steps ===
            df_test_transformed = df_test_encoded.copy()
            transformed_cols = []

            if "transform_steps" in st.session_state:
                for step_name, transformer, target in st.session_state["transform_steps"]:
                    if step_name.startswith("minmax") or step_name.startswith("standard"):
                        if target in df_test_transformed.columns:
                            df_test_transformed[target] = transformer.transform(df_test_transformed[[target]])
                            transformed_cols.append(target)
                        else:
                            st.warning(f"⚠️ Skipped scaling for '{target}' — column not found.")

                    elif step_name == "create_feature":
                        op = transformer["operation"]
                        col1 = transformer["col1"]
                        col2 = transformer.get("col2")
                        new_name = transformer["new_col"]

                        if col1 not in df_test_transformed.columns or (col2 and col2 not in df_test_transformed.columns):
                            st.warning(f"⚠️ Skipped creating feature '{new_name}' — missing columns.")
                            continue

                        try:
                            if op == "Add":
                                df_test_transformed[new_name] = df_test_transformed[col1] + df_test_transformed[col2]
                            elif op == "Subtract":
                                df_test_transformed[new_name] = df_test_transformed[col1] - df_test_transformed[col2]
                            elif op == "Multiply":
                                df_test_transformed[new_name] = df_test_transformed[col1] * df_test_transformed[col2]
                            elif op == "Divide":
                                df_test_transformed[new_name] = df_test_transformed[col1] / (df_test_transformed[col2] + 1e-9)
                            elif op == "Log":
                                df_test_transformed[new_name] = np.log1p(df_test_transformed[col1])
                            elif op == "Square":
                                df_test_transformed[new_name] = df_test_transformed[col1] ** 2
                            transformed_cols.append(new_name)
                        except Exception as e:
                            st.warning(f"⚠️ Error creating '{new_name}': {e}")

                for step_name, transformer, target in st.session_state["transform_steps"]:
                    if step_name == "drop_columns":
                        cols_to_drop = transformer.get("columns_dropped", [])
                        df_test_transformed.drop(columns=[col for col in cols_to_drop if col in df_test_transformed.columns], inplace=True)
                        transformed_cols = [col for col in transformed_cols if col not in cols_to_drop]


                # Save pre-PCA transformed version
                df_test_transformed_pre_pca = df_test_transformed.copy()

                # Save this list for export logic later
                st.session_state["transformed_test_columns"] = transformed_cols

                # === Apply PCA if used ===
                use_pca = st.session_state.get("use_pca", "No")
                if use_pca == "Yes" and st.session_state.get("pca_ready"):
                    scaler = st.session_state["scaler"]
                    pca = st.session_state["pca"]
                    n_components = st.session_state["n_components"]
                    pca_input_columns = st.session_state["pca_input_columns"]

                    df_test_scaled = scaler.transform(df_test_transformed[pca_input_columns])
                    df_test_transformed = pd.DataFrame(
                        pca.transform(df_test_scaled),
                        columns=[f"PC{i+1}" for i in range(n_components)]
                    )

                # === Optional: Download Transformed Test Set (before prediction) ===
                df_test_download = pd.DataFrame({"row_id": row_ids})  # ⬅️ Include row_id for tracking
                for col in df_test_transformed.columns:
                    df_test_download[col] = df_test_transformed[col].values

                csv_test = df_test_download.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Final Transformed Test Set", csv_test, "test_transformed.csv", "text/csv")

                # === Thresholds for traffic light classification ===
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



                # row id session update
                st.session_state["row_id_test"] = row_ids.reset_index(drop=True)

                # === Allow optional flipping ===
                flip_predictions = st.checkbox("🔁 Flip predictions and probabilities for all models?", value=False)




                ###########################################################
                # ✅ The transformed test set is now ready for prediction
                ###########################################################


                # === Retrieve selected models ===
                selected_models = st.session_state.get("selected_models", [])

                # === Prepare input data (exclude row_id) ===
                df_test_input = df_test_transformed.drop(columns=["row_id"], errors="ignore")

                # === Initialize results DataFrame ===
                df_results = df_test_original.copy()
                if target_column_present and 'df_test_target_final' in locals():
                    df_results[target_column] = df_test_target_final.reset_index(drop=True)

                # === Make Predictions and Add Columns Dynamically ===

                df_results = df_test_original.copy()

                # ✅ Add row_id if available
                if "row_id_test" in st.session_state:
                    df_results["row_id"] = st.session_state["row_id_test"].reset_index(drop=True)



                if "Random Forest" in selected_models:
                    test_pred_rf = rf_model.predict(df_test_input)
                    prob_pred_rf = rf_model.predict_proba(df_test_input)[:, 1]
                    test_pred_rf, prob_pred_rf = apply_flipping("Random Forest", test_pred_rf, prob_pred_rf, flip_predictions)
                    df_results["RandomForest_Prediction"] = test_pred_rf
                    df_results["RandomForest_Prob"] = prob_pred_rf
                    df_results["RandomForest_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_rf, prob_pred_rf)
                    ]

                if "Ridge Logistic Regression" in selected_models:
                    test_pred_ridge = ridge_model.predict(df_test_input)
                    prob_pred_ridge = ridge_model.predict_proba(df_test_input)[:, 1]
                    test_pred_ridge, prob_pred_ridge = apply_flipping("Ridge Logistic Regression", test_pred_ridge, prob_pred_ridge, flip_predictions)
                    df_results["Ridge_Prediction"] = test_pred_ridge
                    df_results["Ridge_Prob"] = prob_pred_ridge
                    df_results["Ridge_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_ridge, prob_pred_ridge)
                    ]

                if "Lasso Logistic Regression" in selected_models:
                    test_pred_lasso = lasso_model.predict(df_test_input)
                    prob_pred_lasso = lasso_model.predict_proba(df_test_input)[:, 1]
                    test_pred_lasso, prob_pred_lasso = apply_flipping("Lasso Logistic Regression", test_pred_lasso, prob_pred_lasso, flip_predictions)
                    df_results["Lasso_Prediction"] = test_pred_lasso
                    df_results["Lasso_Prob"] = prob_pred_lasso
                    df_results["Lasso_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_lasso, prob_pred_lasso)
                    ]

                if "ElasticNet Logistic Regression" in selected_models:
                    test_pred_enet = enet_model.predict(df_test_input)
                    prob_pred_enet = enet_model.predict_proba(df_test_input)[:, 1]
                    test_pred_enet, prob_pred_enet = apply_flipping("ElasticNet Logistic Regression", test_pred_enet, prob_pred_enet, flip_predictions)
                    df_results["ElasticNet_Prediction"] = test_pred_enet
                    df_results["ElasticNet_Prob"] = prob_pred_enet
                    df_results["ElasticNet_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_enet, prob_pred_enet)
                    ]

                if "PLS-DA" in selected_models:
                    test_scores_pls = pls_model.predict(df_test_input).ravel()
                    test_pred_pls = (test_scores_pls >= 0.5).astype(int)
                    test_pred_pls, test_scores_pls = apply_flipping("PLS-DA", test_pred_pls, test_scores_pls, flip_predictions)
                    df_results["PLSDA_Prediction"] = test_pred_pls
                    df_results["PLSDA_Test_scores"] = test_scores_pls
                    df_results["PLSDA_TrafficLight (no yellow)"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_pls, test_scores_pls)
                    ]

                if "K-Nearest Neighbors" in selected_models:
                    test_pred_knn = knn_model.predict(df_test_input)
                    prob_pred_knn = knn_model.predict_proba(df_test_input)[:, 1]
                    test_pred_knn, prob_pred_knn = apply_flipping("K-Nearest Neighbors", test_pred_knn, prob_pred_knn, flip_predictions)
                    df_results["KNN_Prediction"] = test_pred_knn
                    df_results["KNN_Prob"] = prob_pred_knn
                    df_results["KNN_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_knn, prob_pred_knn)
                    ]

                if "Naive Bayes" in selected_models:
                    test_pred_nb = nb_model.predict(df_test_input)
                    prob_pred_nb = nb_model.predict_proba(df_test_input)[:, 1]
                    test_pred_nb, prob_pred_nb = apply_flipping("Naive Bayes", test_pred_nb, prob_pred_nb, flip_predictions)
                    df_results["NB_Prediction"] = test_pred_nb
                    df_results["NB_Prob"] = prob_pred_nb
                    df_results["NB_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_nb, prob_pred_nb)
                    ]

                if "Support Vector Machine" in selected_models:
                    test_pred_svm = svm_model.predict(df_test_input)
                    prob_pred_svm = svm_model.predict_proba(df_test_input)[:, 1]
                    test_pred_svm, prob_pred_svm = apply_flipping("Support Vector Machine", test_pred_svm, prob_pred_svm, flip_predictions)
                    df_results["SVM_Prediction"] = test_pred_svm
                    df_results["SVM_Prob"] = prob_pred_svm
                    df_results["SVM_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_svm, prob_pred_svm)
                    ]

                if "Decision Tree" in selected_models:
                    test_pred_tree = tree_model.predict(df_test_input)
                    prob_pred_tree = tree_model.predict_proba(df_test_input)[:, 1]
                    test_pred_tree, prob_pred_tree = apply_flipping("Decision Tree", test_pred_tree, prob_pred_tree, flip_predictions)
                    df_results["DecisionTree_Prediction"] = test_pred_tree
                    df_results["DecisionTree_Prob"] = prob_pred_tree
                    df_results["DecisionTree_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_tree, prob_pred_tree)
                    ]

                if "Gradient Boosting" in selected_models:
                    test_pred_gbm = gbm_model.predict(df_test_input)
                    prob_pred_gbm = gbm_model.predict_proba(df_test_input)[:, 1]
                    test_pred_gbm, prob_pred_gbm = apply_flipping("Gradient Boosting", test_pred_gbm, prob_pred_gbm, flip_predictions)
                    df_results["GBM_Prediction"] = test_pred_gbm
                    df_results["GBM_Prob"] = prob_pred_gbm
                    df_results["GBM_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_gbm, prob_pred_gbm)
                    ]

                if "Neural Network" in selected_models:
                    test_pred_nn = nn_model.predict(df_test_input)
                    prob_pred_nn = nn_model.predict_proba(df_test_input)[:, 1]
                    test_pred_nn, prob_pred_nn = apply_flipping("Neural Network", test_pred_nn, prob_pred_nn, flip_predictions)
                    df_results["NN_Prediction"] = test_pred_nn
                    df_results["NN_Prob"] = prob_pred_nn
                    df_results["NN_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_nn, prob_pred_nn)
                    ]

                if "Bagging" in selected_models:
                    test_pred_bag = bag_model.predict(df_test_input)
                    prob_pred_bag = bag_model.predict_proba(df_test_input)[:, 1]
                    test_pred_bag, prob_pred_bag = apply_flipping("Bagging", test_pred_bag, prob_pred_bag, flip_predictions)
                    df_results["Bagging_Prediction"] = test_pred_bag
                    df_results["Bagging_Prob"] = prob_pred_bag
                    df_results["Bagging_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_bag, prob_pred_bag)
                    ]


                if "Pasting" in selected_models:
                    test_pred_paste = paste_model.predict(df_test_input)
                    prob_pred_paste = paste_model.predict_proba(df_test_input)[:, 1]
                    test_pred_paste, prob_pred_paste = apply_flipping("Pasting", test_pred_paste, prob_pred_paste, flip_predictions)
                    df_results["Pasting_Prediction"] = test_pred_paste
                    df_results["Pasting_Prob"] = prob_pred_paste
                    df_results["Pasting_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_paste, prob_pred_paste)
                    ]


                if "Stacking" in selected_models:
                    test_pred_stack = stack_model.predict(df_test_input)
                    prob_pred_stack = stack_model.predict_proba(df_test_input)[:, 1]
                    test_pred_stack, prob_pred_stack = apply_flipping("Stacking", test_pred_stack, prob_pred_stack, flip_predictions)
                    df_results["Stacking_Prediction"] = test_pred_stack
                    df_results["Stacking_Prob"] = prob_pred_stack
                    df_results["Stacking_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_stack, prob_pred_stack)
                    ]



                if "Voting Classifier" in selected_models:
                    test_pred_vote = voting_clf.predict(df_test_input)
                    prob_pred_vote = voting_clf.predict_proba(df_test_input)[:, 1]
                    test_pred_vote, prob_pred_vote = apply_flipping("Voting Classifier", test_pred_vote, prob_pred_vote, flip_predictions)
                    df_results["Vote_Prediction"] = test_pred_vote
                    df_results["Vote_Prob"] = prob_pred_vote
                    df_results["Vote_TrafficLight"] = [
                        get_traffic_light(pred, prob, threshold_0, threshold_1)
                        for pred, prob in zip(test_pred_vote, prob_pred_vote)
                    ]




                # Store prediction results globally
                st.session_state["df_results"] = df_results


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

                if "Bagging" in selected_models:
                    test_predictions["Bagging"] = (test_pred_bag, prob_pred_bag)

                if "Pasting" in selected_models:
                    test_predictions["Pasting"] = (test_pred_paste, prob_pred_paste)
                    
                if "Stacking" in selected_models:
                    test_predictions["Stacking"] = (test_pred_stack, prob_pred_stack)

                if "Voting Classifier" in selected_models:
                    test_predictions["Voting Classifier"] = (test_pred_vote, prob_pred_vote)


                # === If target is present, compute performance metrics ===
                if target_column_present:
                    st.markdown("### 📊 Test Set Performance Metrics")

                    def compute_metrics(y_true, y_pred, y_prob, model_name):
                        # ✅ Safeguard: skip if labels aren't binary numeric
                        if type_of_target(y_true) != "binary":
                            st.warning(f"{model_name}: Target labels are not binary numeric. Metrics may fail.")
                            return {
                                'Model': model_name,
                                'Accuracy': None,
                                'Precision': None,
                                'Recall': None,
                                'F1-Score': None,
                                'AUC': None
                            }

                        # Compute AUC to check for signal direction
                        auc_score = roc_auc_score(y_true, y_prob)
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


                    if flip_predictions:
                        st.info("🔁 Predictions and probabilities were flipped to match expected label orientation.")

                    test_summary_df = pd.DataFrame(test_metrics)
                    st.dataframe(test_summary_df.style.format({
                        "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                        "Recall": "{:.4f}", "F1-Score": "{:.4f}", "AUC": "{:.4f}"
                    }))
                else:
                    st.info("ℹ️ Target column not found in test data. Skipping performance metrics.")





            # === Show and Download ===
            try:
                st.markdown("### 📄 Predictions on Uploaded Test Data")

                # Let the user choose which columns to include
                st.markdown("#### 🧩 Select Columns to Include in Final Download")
                include_original = st.checkbox("Include original test columns", value=True)
                include_transformed = st.checkbox("Include transformed columns (scaling, standardization, etc.)", value=False)
                include_pca = st.checkbox("Include PCA components", value=False)
                include_predictions = st.checkbox("Include predictions", value=True)

                # Build final export DataFrame
                df_export = pd.DataFrame()

                # ✅ Include original raw test data
                if include_original:
                    df_export = df_test.copy()

                # ✅ Insert row_id as the first column if available AND not already present
                if "row_id_test" in st.session_state:
                    if "row_id" not in df_export.columns:
                        df_export.insert(0, "row_id", st.session_state["row_id_test"].reset_index(drop=True))

                # ✅ Include manually transformed features (like Add Radius)
                if include_transformed and "df_test_transformed_pre_pca" in locals():
                    tf_cols = st.session_state.get("transformed_test_columns", [])
                    df_trans = df_test_transformed_pre_pca[tf_cols].copy()
                    df_trans.columns = [f"TF_{col}" for col in df_trans.columns]
                    df_export = pd.concat([df_export, df_trans], axis=1)

                # ✅ Include PCA columns if selected and PCA was used
                if include_pca and use_pca == "Yes" and "df_test_transformed" in locals():
                    df_pca = df_test_transformed.copy()
                    df_pca.columns = [f"PC{i+1}" for i in range(df_pca.shape[1])]
                    df_export = pd.concat([df_export, df_pca], axis=1)

                # ✅ Include model predictions
                if include_predictions:
                    prediction_cols = []
                    for model in st.session_state.get("selected_models", []):
                        prefix = ""
                        if "Ridge" in model: prefix = "Ridge"
                        elif "Lasso" in model: prefix = "Lasso"
                        elif "ElasticNet" in model: prefix = "ElasticNet"
                        elif "Random Forest" in model: prefix = "RandomForest"
                        elif "Decision Tree" in model: prefix = "DecisionTree"
                        elif "Support Vector" in model: prefix = "SVM"
                        elif "Gradient Boosting" in model: prefix = "GradientBoosting"
                        elif "PLS-DA" in model: prefix = "PLSDA"
                        elif "K-Nearest" in model: prefix = "KNN"
                        elif "Naive Bayes" in model: prefix = "NaiveBayes"
                        elif "Neural Network" in model: prefix = "NN"
                        elif "Voting" in model: prefix = "Vote"
                        elif "Bagging" in model: prefix = "Bagging"
                        elif "Pasting" in model: prefix = "Pasting"
                        elif "Stacking" in model: prefix = "Stacking"

                        prediction_cols += [f"{prefix}_Prediction", f"{prefix}_Prob", f"{prefix}_TrafficLight"]

                    existing_cols = [col for col in prediction_cols if col in df_results.columns]
                    df_export = pd.concat([df_export, df_results[existing_cols]], axis=1)


                # === Show preview
                st.markdown("#### 📝 Preview of Download File")
                st.dataframe(df_export.head())

                # Select file format
                file_format = st.selectbox("Select file format for download:", ["CSV", "JSON"])

                # Generate downloadable data based on selection
                if file_format == "CSV":
                    file_data = df_export.to_csv(index=False).encode("utf-8")
                    file_name = "classified_results.csv"
                    mime_type = "text/csv"
                elif file_format == "JSON":
                    file_data = df_export.to_json(orient="records", indent=2).encode("utf-8")
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

