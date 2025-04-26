import pandas as pd
import joblib
import streamlit as st

# Load model
model = joblib.load('saved_model.pkl')

# Columns used in training
input_features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 
                  'Sex_female', 'Sex_male', 
                  'Embarked_C', 'Embarked_Q', 'Embarked_S']

# Streamlit UI
st.title('Titanic Survival Prediction')

Pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
Age = st.slider('Age', 0, 80, 25)
Fare = st.number_input('Fare', 0, 500, 50)
SibSp = st.number_input('Siblings/Spouses Aboard', 0, 10, 0)
Parch = st.number_input('Parents/Children Aboard', 0, 10, 0)
Sex = st.selectbox('Sex', ['male', 'female'])
Embarked = st.selectbox('Embarked Port', ['C', 'Q', 'S'])

# Preprocess user input
input_data = {
    'Pclass': Pclass,
    'Age': Age,
    'Fare': Fare,
    'SibSp': SibSp,
    'Parch': Parch,
    'Sex_female': 1 if Sex == 'female' else 0,
    'Sex_male': 1 if Sex == 'male' else 0,
    'Embarked_C': 1 if Embarked == 'C' else 0,
    'Embarked_Q': 1 if Embarked == 'Q' else 0,
    'Embarked_S': 1 if Embarked == 'S' else 0
}

input_df = pd.DataFrame([input_data])

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success('🎯 Passenger Survived!')
    else:
        st.error('💀 Passenger Did Not Survive.')
