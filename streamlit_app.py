import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("🤖 Binary Classification App")
st.info("This app builds a binary classification model!")

# === File Upload ===
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls", "json"])

# === Proceed only if file is uploaded ===
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


    # Show data overview
    st.markdown("### 📋 Dataset Overview")

    # Shape
    st.write(f"🔢 **Rows:** {df.shape[0]} &nbsp;&nbsp;&nbsp;&nbsp; 📁 **Columns:** {df.shape[1]}")

    # Data types per column
    st.markdown("#### 📌 Column Data Types")
    col_dtype_df = pd.DataFrame({'Column': df.columns, 'Data Type': df.dtypes.values})
    st.dataframe(col_dtype_df)

    # Data type frequency
    st.markdown("#### 📊 Data Type Frequency")
    dtype_counts = df.dtypes.value_counts()
    dtype_counts_df = pd.DataFrame({'Data Type': dtype_counts.index.astype(str), 'Count': dtype_counts.values})
    st.dataframe(dtype_counts_df)



    # === Target Column Selection ===
    target_column = st.selectbox(
        "Select the target (classification) column:",
        df.columns,
        help="This is the column the model will try to predict (e.g. species, outcome, label)"
    )

    # === Feature/Target Split ===
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]

    # === Train/Validation Split ===
    st.subheader("📚 Train/Validation Split")
    test_size_percent = st.slider("Select validation set size (%)", min_value=10, max_value=50, value=20, step=5)
    test_size = test_size_percent / 100.0

    # Randomize and split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=42, shuffle=True
    )

    st.write(f"🔹 Training set size: {X_train.shape[0]} rows")
    st.write(f"🔸 Validation set size: {X_val.shape[0]} rows")

    # === PCA Option ===
    use_pca = st.radio("Would you like to apply PCA?", ["No", "Yes"])

    if use_pca == "Yes":
        st.subheader("🔍 PCA Analysis")

        # Standardize the training data
        scaler = StandardScaler()

        X_train_numeric = X_train.select_dtypes(include=np.number)
        X_val_numeric = X_val.select_dtypes(include=np.number)

        X_train_scaled = scaler.fit_transform(X_train_numeric)
        X_val_scaled = scaler.transform(X_val_numeric)


        # Fit PCA on scaled training data
        pca = PCA()
        pca.fit(X_train_scaled)

        # Plot explained variance
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(cum_var)+1), cum_var, marker='o')
        ax.set_title("Cumulative Explained Variance")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance")
        ax.grid(True)
        st.pyplot(fig)

        # Select how many components to keep
        n_components = st.slider("Select number of principal components to keep", 1, X_train.shape[1], 2)

        # Apply PCA with selected components
        pca = PCA(n_components=n_components)
        X_train_final = pd.DataFrame(pca.fit_transform(X_train_scaled), columns=[f'PC{i+1}' for i in range(n_components)])
        X_val_final = pd.DataFrame(pca.transform(X_val_scaled), columns=[f'PC{i+1}' for i in range(n_components)])

        st.write("✅ PCA applied. Transformed training set:")
        st.dataframe(X_train_final)

    else:
        st.info("PCA not applied. Using original features.")
        X_train_final = X_train.copy()
        X_val_final = X_val.copy()

    # === Visualization ===
    with st.expander("📊 Data Visualization"):
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_columns) < 2:
            st.warning("❗Please upload a dataset with at least two numeric columns to create a scatter plot.")
        else:
            x_axis = st.selectbox("Select X-axis", options=numeric_columns, index=0)
            y_axis = st.selectbox("Select Y-axis", options=[col for col in numeric_columns if col != x_axis], index=0)

            use_legend = st.radio("Would you like to color by a third column (legend)?", ["No", "Yes"], index=0)

            if use_legend == "Yes":
                legend_col = st.selectbox(
                    "Select the column to use for legend (color grouping):",
                    options=[col for col in df.columns if col not in [x_axis, y_axis]]
                )
                st.scatter_chart(data=df, x=x_axis, y=y_axis, color=legend_col)
            else:
                st.scatter_chart(data=df[[x_axis, y_axis]], x=x_axis, y=y_axis)

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
