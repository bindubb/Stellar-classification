import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
import base64

st.title("⋆⭒˚.⋆🪐 ⋆⭒˚.⋆STELLAR CLASSIFICATION⋆⭒˚.⋆🪐 ⋆⭒˚.⋆")

# Load the model
model = load("gb_model.joblib")

# Load the dataset and fit scaler
stellar = pd.read_csv("star_classification.csv")
stellar.rename(columns = {"u" : "Ultravoilet", "g" : "Green", "r":"Red", "i" : "Near Infrared", "z": "Infrared" }, inplace = True)

if "obj_ID" in stellar.columns.values:
    stellar.drop(["obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID","MJD","fiber_ID","plate"], axis=1, inplace=True)

# Drop the target variable
X = stellar.drop(['class'], axis=1)
scaler = StandardScaler()
scaler.fit(X)


st.header("Enter Object Information")

# Input fields for each feature
input_data = []
with st.sidebar:
     for col in X.columns:
       value = st.slider(f"{col}", value=float(X[col].mean()))
       input_data.append(value)

    


input_data = np.array([input_data])


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Star Class"):
    prediction = model.predict(input_data_scaled)
    class_map = {0: " Galaxy", 1: "Quasar", 2: "Star"}  # Update with actual class mappings
    predicted_class = class_map[prediction[0]]
    st.success(f"The predicted class is: {predicted_class}")

with st.expander('**Visualization**'):
        st.write('What is the distribution of stars, galaxies, and quasars in the dataset?')
        st.image("s1.png")
        st.write(' Outlier Detection: Are there any outliers in the data?')
        st.image('s3.png')
        st.write('Outliers have been detected and handled through capping.')
        st.image('s4.png')
        st.write('Updated plots after handling outliers:')
        st.image('s5.png')
        st.write('The class distribution appears highly imbalanced.')
        st.image('s7.png')
        st.write('Spatial distribution of data:')
        st.image('s9.png')
        st.image('s10.png')
        st.write('Most features are correlated except for Redshift.')
        st.image('s14.png')
        st.write('Quasars appear spread across a wide wavelength, indicating universe expansion.')




def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://th.bing.com/th/id/OIP.qNk62O2APJHRIY3G0C4QKQHaEK?w=1920&h=1080&rs=1&pid=ImgDetMain.jpg");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local()