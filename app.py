import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import dill

# Load preprocessing object and model
with open('objects/preprocessor.pkl', 'rb') as f:
    ctf = dill.load(f)

model = load_model('notebooks/model.h5')

st.title('Churn Prediction App')

# User inputs
credit = st.number_input(label='CreditScore')
geo = st.selectbox(label='Geography', options=['France', 'Spain'])
gender = st.selectbox(label='Gender', options=['Female', 'Male'])
age = st.slider(label='Age', min_value=0, max_value=100)
tenure = st.number_input(label='Tenure', min_value=0, max_value=10)
balance = st.number_input('Balance')
nop = st.number_input('Number of Products')
Creditcard = st.selectbox('Do you have a Credit Card?', options=['Yes', 'No'])
active = st.selectbox('Are you an Active Member?', options=['Yes', 'No'])
estimated_salary = float(st.number_input('Salary'))

# Prepare input record
columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
record = pd.DataFrame([[credit, geo, gender, age, tenure, float(balance), nop, int(Creditcard == 'Yes'), int(active == 'Yes'), estimated_salary]], columns=columns)

# Prediction function
def func():
    transformed_record = ctf.transform(record)
    pred_prob = model.predict(transformed_record)[0][0]   # Assuming binary classification (single output neuron)
    st.write(f"Predicted probability of churn = {pred_prob:.2f}")
    if pred_prob > 0.5:
        st.error("Employee will churn")
    else:
        st.success("Employee will stay")

# Button
st.button(label='Predict', on_click=func)
