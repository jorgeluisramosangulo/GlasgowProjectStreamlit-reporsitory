import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Binary Classification App')
st.info('This app builds a binary classification model!')

# === Data Loading and Preview ===
with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv(r'C:\Users\jorger\Desktop\Studies\Glasgow\Dissertation\Dashboard for binary classification\GlasgowProjectStreamlit reporsitory\penguins_cleaned.csv')
    st.dataframe(df)

    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.write('**y**')
    y_raw = df['species']
    st.dataframe(y_raw)

# === Visualization ===
with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# === Sidebar Inputs ===
with st.sidebar:
    st.header('Input features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))

    data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

# === Input Preview ===
with st.expander('Input features'):
    st.write('**Input penguin**')
    st.dataframe(input_df)
    st.write('**Combined penguins data**')
    st.dataframe(input_penguins)

# === Encoding Data ===
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, columns=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
    st.write('**Encoded X (input penguin)**')
    st.dataframe(input_row)
    st.write('**Encoded y**')
    st.dataframe(y)

# === Model Training ===
clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# === Prediction Output ===
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
                 'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', width='medium', min_value=0, max_value=1),
                 'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', width='medium', min_value=0, max_value=1),
                 'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', width='medium', min_value=0, max_value=1)
             },
             hide_index=True)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f"Predicted species: {penguins_species[prediction][0]}")
