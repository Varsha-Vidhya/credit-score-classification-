import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Mapping categorical options to numerical values
payment_of_min_amount_options = {
    "Low_spent_Small_value_payments": 0,
    "High_spent_Medium_value_payments": 1,
    "Low_spent_Medium_value_payments": 2,
    "High_spent_Large_value_payments": 3,
    "High_spent_Small_value_payments": 4,
    "Low_spent_Large_value_payments": 5,
}

type_of_loan_options = {
    "Auto Loan": 0,
    "Credit-Builder Loan": 1,
    "Debt Consolidation Loan": 2,
    "Home Equity Loan": 3,
    "Mortgage Loan": 4,
    "Payday Loan": 5,
    "Personal Loan": 6,
    "Student Loan": 7,
}

credit_mix_options = {"Bad": 0, "Good": 1, "Standard": 2}

payment_behaviour_options = {"NM": 0, "NO": 1, "YES": 2}

# Define the feature names
numerical_features = [
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Bank_Accounts",
    "Num_Credit_Inquiries", "Delay_from_due_date", "Interest_Rate",
    "Credit_History_Age", "Outstanding_Debt", "Credit_Utilization_Efficiency"
]

categorical_features = [
    "Type_of_Loan", "Payment_Behaviour", "Credit_Mix", "Payment_of_Min_Amount"
]

# Map prediction values to credit scores
score_mapping = {0: "Poor", 1: "Standard", 2: "Good"}

# Streamlit UI
st.title("Credit Score Prediction App")
st.markdown("""
Enter the required details below, and click **Predict Credit Score** to get your credit score classification:
""")

# User input for numerical features
st.header("Numerical Features")
numerical_inputs = {}
for feature in numerical_features:
    numerical_inputs[feature] = st.number_input(f"{feature}:", min_value=0.0, step=0.01)

# User input for categorical features
st.header("Categorical Features")
categorical_inputs = {}

categorical_inputs["Type_of_Loan"] = st.selectbox(
    "Type_of_Loan:", options=list(type_of_loan_options.keys())
)

categorical_inputs["Payment_Behaviour"] = st.selectbox(
    "Payment_Behaviour:", options=list(payment_behaviour_options.keys())
)

categorical_inputs["Credit_Mix"] = st.selectbox(
    "Credit_Mix:", options=list(credit_mix_options.keys())
)

categorical_inputs["Payment_of_Min_Amount"] = st.selectbox(
    "Payment_of_Min_Amount:", options=list(payment_of_min_amount_options.keys())
)

# Prediction button
if st.button("Predict Credit Score"):
    try:
        # Convert user inputs into model-ready format
        input_data = [
            numerical_inputs[feature] for feature in numerical_features
        ] + [
            type_of_loan_options[categorical_inputs["Type_of_Loan"]],
            payment_behaviour_options[categorical_inputs["Payment_Behaviour"]],
            credit_mix_options[categorical_inputs["Credit_Mix"]],
            payment_of_min_amount_options[categorical_inputs["Payment_of_Min_Amount"]],
        ]

        # Reshape the input for prediction
        input_array = np.array(input_data).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(input_array)[0]

        # Display the prediction
        st.success(f"The predicted credit score is: **{score_mapping[prediction]}**")
    except Exception as e:
        st.error(f"An error occurred: {e}")


                      




