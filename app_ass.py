import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

st.title("📡 Telco Customer Churn Predictor")
st.write("Enter the customer details, then click **Predict**.")

st.subheader("Account")
c1, c2 = st.columns(2)
tenure = c1.slider("Tenure (months)", 0, 72, 12)
contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly = c1.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
total = c2.number_input("Total Charges ($)", 0.0, 10000.0, float(round(tenure * monthly, 2)), step=10.0)
paperless = c1.selectbox("Paperless Billing", ["Yes", "No"])
payment = c2.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", "Credit card (automatic)"])

st.subheader("Customer")
c1, c2 = st.columns(2)
gender = c1.selectbox("Gender", ["Female", "Male"])
senior = c2.selectbox("Senior Citizen", ["No", "Yes"])
partner = c1.selectbox("Partner", ["Yes", "No"])
dependents = c2.selectbox("Dependents", ["No", "Yes"])

st.subheader("Services")
c1, c2 = st.columns(2)
phone = c1.selectbox("Phone Service", ["Yes", "No"])
multiple = c2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet = c1.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

no_net = internet == "No"
opts = ["No internet service"] if no_net else ["No", "Yes"]
online_sec = c2.selectbox("Online Security", opts)
online_bkp = c1.selectbox("Online Backup", opts)
device = c2.selectbox("Device Protection", opts)
tech = c1.selectbox("Tech Support", opts)
tv = c2.selectbox("Streaming TV", opts)
movies = c1.selectbox("Streaming Movies", opts)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
        "DeviceProtection": device, "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0, 1]
    if pred == 1:
        st.error(f"This customer is at risk of leaving ⚠️  (churn probability: {prob:.1%})")
    else:
        st.success(f"This customer is likely to stay ✅  (churn probability: {prob:.1%})")
    st.progress(float(prob))
