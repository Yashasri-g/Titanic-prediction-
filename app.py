import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('saved_model.pkl')

# Streamlit page configuration
st.set_page_config(page_title="Titanic Survival Prediction 🚢", layout="centered")

# Title
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter the passenger's details below to predict their survival chances.")

# Sidebar for input fields
st.sidebar.header("📝 Passenger Details")

# Input Fields
pclass = st.sidebar.selectbox('Passenger Class (Pclass)', [1, 2, 3])

sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])

age = st.sidebar.slider('Age', 0, 100, 25)

sibsp = st.sidebar.number_input('Number of Siblings/Spouses Aboard (SibSp)', min_value=0, max_value=10, value=0)

parch = st.sidebar.number_input('Number of Parents/Children Aboard (Parch)', min_value=0, max_value=10, value=0)

fare = st.sidebar.number_input('Fare Paid ($)', min_value=0.0, max_value=600.0, value=50.0)

embarked = st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Prepare input for the model
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    sex_female = 1 if sex == 'Female' else 0
    sex_male = 1 if sex == 'Male' else 0

    embarked_C = 1 if embarked == 'C' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0

    input_data = np.array([[pclass, sex_female, sex_male, age, sibsp, parch, fare, embarked_C, embarked_Q, embarked_S]])
    return input_data

# Predict button
if st.sidebar.button('Predict'):
    input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("🎯 Prediction Result")
    if prediction[0] == 1:
        st.success("🥳 The passenger **would survive**!")
    else:
        st.error("😢 The passenger **would not survive**.")

    st.subheader("📊 Prediction Probabilities")
    st.write(f"**Survival Probability:** {prediction_proba[0][1]*100:.2f}%")
    st.write(f"**Death Probability:** {prediction_proba[0][0]*100:.2f}%")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
