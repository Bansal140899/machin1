import streamlit as st
import numpy as np
import pickle
import sklearn

with open('models/scaler.pkl' , 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open('models/model.pkl' , 'rb') as model_file:
    loaded_model = pickle.load(model_file)


st.title("e_commerce predictor")

avg_seesion_length = st.number_input("average session length")
time_on_app = st.number_input("time on app")
length_of_membership = st.number_input("length of membership")

if st.button('predict'):
    data = np.array([avg_seesion_length ,time_on_app ,length_of_membership]) .reshape(1,-1)
    data_new = loaded_scaler.transform(data)
    prediction = loaded_model.predict(data_new)

    st.success(f" yearly amount spent is $ {prediction[0]}")