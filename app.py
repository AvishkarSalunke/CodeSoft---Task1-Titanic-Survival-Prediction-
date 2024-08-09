import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model and scaler
model = joblib.load('model.pkl')
scaler =joblib.load('scaler.pkl')

# Title of the app
st.title("Titanic Survival Prediction")

# Sidebar for user input
st.header("User Input")

# Input fields
pclass = st.selectbox("Pclass :", [1, 2, 3])
sex = st.selectbox("Sex:", ["Male", "Female"])
age = st.slider("Age:", min_value=0, max_value=120, value=30)
#age = st.slider('Age', 30,120)


# Convert 'Sex' to numeric
sex_num = 1 if sex == "Male" else 0

# Prepare input features for prediction
features = np.array([[pclass, sex_num,age]])
#features_scaled = scaler.transform(features)

# Predict survival
prediction = model.predict(features)

result = "Survived" if prediction[0] == 1 else "Did not survive"

# Display result
st.write(f"### Prediction: {result}")

# Optionally, show the input features used for prediction
st.write("### Input Features:")
st.write(f"Pclass: {pclass}")
st.write(f"Sex: {sex}")
st.write(f"Age: {age}")


