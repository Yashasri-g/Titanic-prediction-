import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('saved_model.pkl')

# Title
st.title('🚢 Titanic Survival Prediction')

# Inputs
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare Paid', min_value=0.0, value=50.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Predict button
if st.button('Predict'):
    # Manual one-hot encoding
    sex_female = 1 if sex == 'female' else 0
    sex_male = 1 if sex == 'male' else 0
    embarked_C = 1 if embarked == 'C' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0

    # Prepare input
    input_df = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_female': [sex_female],
        'Sex_male': [sex_male],
        'Embarked_C': [embarked_C],
        'Embarked_Q': [embarked_Q],
        'Embarked_S': [embarked_S]
    })

    input_df = input_df.values  # Convert to NumPy array

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Show result
    if prediction == 1:
        st.success('🎯 The passenger is likely to SURVIVE!')
    else:
        st.error('⚠️ The passenger is likely to NOT survive.')
