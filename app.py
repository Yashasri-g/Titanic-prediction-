import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('saved_model.pkl')

st.title('🚢 Titanic Survival Prediction App')

st.markdown('### Please enter passenger details:')

# Input form
pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=25)
sibsp = st.number_input('Number of Siblings/Spouses aboard (SibSp)', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children aboard (Parch)', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, value=32.0)
embarked = st.selectbox('Port of Embarkation (Embarked)', ['C', 'Q', 'S'])

if st.button('Predict'):
    # One-hot encoding manually
    sex_male = 1 if sex == 'male' else 0
    sex_female = 1 if sex == 'female' else 0

    embarked_C = 1 if embarked == 'C' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0

    # Arrange features in the same order as training
    input_data = np.array([[pclass, age, sibsp, parch, fare,
                            sex_female, sex_male,
                            embarked_C, embarked_Q, embarked_S]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success('🎯 The passenger is likely to SURVIVE!')
    else:
        st.error('⚠️ The passenger is likely to NOT survive.')
