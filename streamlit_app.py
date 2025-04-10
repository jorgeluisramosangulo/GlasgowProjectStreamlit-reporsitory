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
        C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
        max_iter = st.slider("Max iterations", 100, 1000, 100)

        lr_model = LogisticRegression(C=C, max_iter=max_iter)
        lr_model.fit(X_train_final, y_train)

        y_pred_lr = lr_model.predict(X_train_final)
        y_proba_lr = lr_model.predict_proba(X_val_final)[:, 1]
        st.text("Classification Report (Training Set):")
        st.text(classification_report(y_train, y_pred_lr))

    # === Random Forest ===
    with st.expander("🌳 Random Forest"):
        n_estimators = st.slider("Number of trees", 10, 200, 100)
        max_depth = st.slider("Max depth", 1, 20, 5)

        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_train_final, y_train)

        y_pred_rf = rf_model.predict(X_train_final)
        y_proba_rf = rf_model.predict_proba(X_val_final)[:, 1]
        st.text("Classification Report (Training Set):")
        st.text(classification_report(y_train, y_pred_rf))

    # === Validation Metrics Summary ===
    st.subheader("📊 Final Validation Set Comparison")
    y_val_pred_lr = lr_model.predict(X_val_final)
    y_val_pred_rf = rf_model.predict(X_val_final)

    metrics_summary = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [accuracy_score(y_val, y_val_pred_lr), accuracy_score(y_val, y_val_pred_rf)],
        'Precision': [precision_score(y_val, y_val_pred_lr, zero_division=0), precision_score(y_val, y_val_pred_rf, zero_division=0)],
        'Recall': [recall_score(y_val, y_val_pred_lr, zero_division=0), recall_score(y_val, y_val_pred_rf, zero_division=0)],
        'F1 Score': [f1_score(y_val, y_val_pred_lr, zero_division=0), f1_score(y_val, y_val_pred_rf, zero_division=0)],
        'AUC': [roc_auc_score(y_val, y_proba_lr), roc_auc_score(y_val, y_proba_rf)]
    })

    st.dataframe(metrics_summary.style.format("{:.2f}"))

else:
    st.warning("📂 Please upload a CSV, Excel, or JSON file to proceed.")





# # === Sidebar Inputs ===
# with st.sidebar:
#     st.header('Input features')
#     island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
#     bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
#     bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
#     flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
#     body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
#     gender = st.selectbox('Gender', ('male', 'female'))

#     data = {
#         'island': island,
#         'bill_length_mm': bill_length_mm,
#         'bill_depth_mm': bill_depth_mm,
#         'flipper_length_mm': flipper_length_mm,
#         'body_mass_g': body_mass_g,
#         'sex': gender
#     }
#     input_df = pd.DataFrame(data, index=[0])
#     input_penguins = pd.concat([input_df, X_raw], axis=0)

# # === Input Preview ===
# with st.expander('Input features'):
#     st.write('**Input penguin**')
#     st.dataframe(input_df)
#     st.write('**Combined penguins data**')
#     st.dataframe(input_penguins)

# # === Encoding Data ===
# encode = ['island', 'sex']
# df_penguins = pd.get_dummies(input_penguins, columns=encode)

# X = df_penguins[1:]
# input_row = df_penguins[:1]

# target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
# def target_encode(val):
#     return target_mapper[val]

# y = y_raw.apply(target_encode)

# with st.expander('Data preparation'):
#     st.write('**Encoded X (input penguin)**')
#     st.dataframe(input_row)
#     st.write('**Encoded y**')
#     st.dataframe(y)

# # === Model Training ===
# clf = RandomForestClassifier()
# clf.fit(X, y)

# prediction = clf.predict(input_row)
# prediction_proba = clf.predict_proba(input_row)

# # === Prediction Output ===
# df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

# st.subheader('Predicted Species')
# st.dataframe(df_prediction_proba,
#              column_config={
#                  'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', width='medium', min_value=0, max_value=1),
#                  'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', width='medium', min_value=0, max_value=1),
#                  'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', width='medium', min_value=0, max_value=1)
#              },
#              hide_index=True)

# penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
# st.success(f"Predicted species: {penguins_species[prediction][0]}")
