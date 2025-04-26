if st.button('Predict'):
    # Manual One-hot encoding
    sex_female = 1 if sex == 'female' else 0
    sex_male = 1 if sex == 'male' else 0

    embarked_C = 1 if embarked == 'C' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0

    # Create a DataFrame
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

    # ❗ Convert input_df to numpy array because model expects array
    input_df = input_df.values

    # Predict
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success('🎯 The passenger is likely to SURVIVE!')
    else:
        st.error('⚠️ The passenger is likely to NOT survive.')
