# import streamlit as st
# import numpy as np
# import pickle

# # import pandas as pd
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# # # Load the dataset
# # df = pd.read_csv("./creditcard.csv")

# # # Split the data into train and test sets
# # X = df.drop("Class", axis=1)
# # y = df["Class"]
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # # Train the model
# # model = RandomForestClassifier()
# # model.fit(X_train, y_train)

# # # Make predictions on the test set
# # y_pred = model.predict(X_test)

# # # Calculate the accuracy, precision, recall, and F1 score
# # accuracy = accuracy_score(y_test, y_pred)
# # precision = precision_score(y_test, y_pred)
# # recall = recall_score(y_test, y_pred)
# # f1 = f1_score(y_test, y_pred)



# # Create a Streamlit app
# st.title("Credit Card Fraud Detection Webapp")


# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)


# # Input features
# st.sidebar.header("Enter the input features")
# amount = st.sidebar.slider("Amount", step=100)
# time = st.sidebar.slider("Time", step=100)

# # Predict the fraud
# if st.button("Predict"):
#     features = np.array([[amount, time]])
#     y_pred = model.predict(features)
#     if y_pred[0] == 1:
#         st.write("Fraudulent transaction")
#     else:
#         st.write("Legitimate transaction")

# # Print the accuracy, precision, recall, and F1 score
# # st.write("Accuracy: ", accuracy)
# # st.write("Precision: ", precision)
# # st.write("Recall: ", recall)
# # st.write("F1 score: ", f1)




# import streamlit as st
# import pickle
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # Load the trained model
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# # Create a function to preprocess input data
# def preprocess_input(data):
#     scaler = StandardScaler()
#     data["NormalizedAmount"] = scaler.fit_transform(data["Amount"].values.reshape(-1, 1))
#     data.drop(["Amount", "Time"], inplace=True, axis=1)
#     return data

# # Create the Streamlit web app
# def main():
#     st.title("Credit Card Fraud Detection")

#     st.write("This web app predicts if a credit card transaction is fraudulent or not.")

#     st.sidebar.header("User Input Features")

#     # Collect user input features
#     Amount = st.sidebar.number_input("Amount", value=0.0)
#     V1 = st.sidebar.number_input("V1", value=0.0)
#     V2 = st.sidebar.number_input("V2", value=0.0)
#     V3 = st.sidebar.number_input("V3", value=0.0)
#     V4 = st.sidebar.number_input("V4", value=0.0)
#     V5 = st.sidebar.number_input("V5", value=0.0)
#     V6 = st.sidebar.number_input("V6", value=0.0)
#     V7 = st.sidebar.number_input("V7", value=0.0)
#     V8 = st.sidebar.number_input("V8", value=0.0)
#     V9 = st.sidebar.number_input("V9", value=0.0)
#     V10 = st.sidebar.number_input("V10", value=0.0)
#     V11 = st.sidebar.number_input("V11", value=0.0)
#     V12 = st.sidebar.number_input("V12", value=0.0)
#     V13 = st.sidebar.number_input("V13", value=0.0)
#     V14 = st.sidebar.number_input("V14", value=0.0)
#     V15 = st.sidebar.number_input("V15", value=0.0)
#     V16 = st.sidebar.number_input("V16", value=0.0)
#     V17 = st.sidebar.number_input("V17", value=0.0)
#     V18 = st.sidebar.number_input("V18", value=0.0)
#     V19 = st.sidebar.number_input("V19", value=0.0)
#     V20 = st.sidebar.number_input("V20", value=0.0)
#     V21 = st.sidebar.number_input("V21", value=0.0)
#     V22 = st.sidebar.number_input("V22", value=0.0)
#     V23 = st.sidebar.number_input("V23", value=0.0)
#     V24 = st.sidebar.number_input("V24", value=0.0)
#     V25 = st.sidebar.number_input("V25", value=0.0)
#     V26 = st.sidebar.number_input("V26", value=0.0)
#     V27 = st.sidebar.number_input("V27", value=0.0)
#     V28 = st.sidebar.number_input("V28", value=0.0)

#     # Create a dictionary from user input
#     input_data = {
#         "Amount": [Amount],
#         "V1": [V1],
#         "V2": [V2],
#         "V3": [V3],
#         "V4": [V4],
#         "V5": [V5],
#         "V6": [V6],
#         "V7": [V7],
#         "V8": [V8],
#         "V9": [V9],
#         "V10": [V10],
#         "V11": [V11],
#         "V12": [V12],
#         "V13": [V13],
#         "V14": [V14],
#         "V15": [V15],
#         "V16": [V16],
#         "V17": [V17],
#         "V18": [V18],
#         "V19": [V19],
#         "V20": [V20],
#         "V21": [V21],
#         "V22": [V22],
#         "V23": [V23],
#         "V24": [V24],
#         "V25": [V25],
#         "V26": [V26],
#         "V27": [V27],
#         "V28": [V28]


#     }

#     # Convert the dictionary to a DataFrame
#     input_df = pd.DataFrame(input_data)

#     # Preprocess the input data
#     input_df = preprocess_input(input_df)

#     # Get the prediction
#     prediction = model.predict(input_df)
    
#     st.subheader("Prediction")
#     if prediction[0] == 1:
#         st.write("This transaction is **fraudulent**.")
#     else:
#         st.write("This transaction is **genuine**.")

# if __name__ == "__main__":
#     main()




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
normalized_amount = st.number_input("Normalized Amount")

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

