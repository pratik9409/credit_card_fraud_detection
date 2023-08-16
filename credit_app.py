import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Create a Streamlit web app
st.title("Credit Card Fraud Detection")

# Input form for user to enter transaction details
st.write("Please enter the transaction details:")

# time = st.number_input("Time (in seconds)")
v1 = st.number_input("V1")
v2 = st.number_input("V2")
v3 = st.number_input("V3")
V4 = st.number_input("V4")
V5 = st.number_input("V5")
V6 = st.number_input("V6")
V7 = st.number_input("V7")
V8 = st.number_input("V8")
V9 = st.number_input("V9")
V10 = st.number_input("V10")
V11 = st.number_input("V11")
V12 = st.number_input("V12")
V13 = st.number_input("V13")
V14 = st.number_input("V14")
V15 = st.number_input("V15")
V16 = st.number_input("V16")
V17 = st.number_input("V17")
V18 = st.number_input("V18")
V19 = st.number_input("V19")
V20 = st.number_input("V20")
V21 = st.number_input("V21")
V22 = st.number_input("V22")
V23 = st.number_input("V23")
V24 = st.number_input("V24")
V25 = st.number_input("V25")
V26 = st.number_input("V26")
V27 = st.number_input("V27")
V28 = st.number_input("V28")
normalized_amount = st.number_input("Amount")

# Predict if the transaction is fraudulent or not
if st.button("Check"):
    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        # "Time": [time],
        "V1": [v1],
        "V2": [v2],
        "V3": [v3],
        "V4": [V4],
        "V5": [V5],
        "V6": [V6],
        "V7": [V7],
        "V8": [V8],
        "V9": [V9],
        "V10": [V10],
        "V11": [V11],
        "V12": [V12],
        "V13": [V13],
        "V14": [V14],
        "V15": [V15],
        "V16": [V16],
        "V17": [V17],
        "V18": [V18],
        "V19": [V19],
        "V20": [V20],
        "V21": [V21],
        "V22": [V22],
        "V23": [V23],
        "V24": [V24],
        "V25": [V25],
        "V26": [V26],
        "V27": [V27],
        "V28": [V28],
        "NormalizedAmount": [normalized_amount]


    })

    # Make a prediction
    prediction = model.predict(input_data)[0]

    # Display the prediction
    if prediction == 0:
        st.write("This transaction is predicted as Genuine.")
    else:
        st.write("This transaction is predicted as Fraudulent.")

