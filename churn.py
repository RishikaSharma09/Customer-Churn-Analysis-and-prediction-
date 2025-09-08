import streamlit as st
import pandas as pd
import joblib

model = joblib.load("CHURN_ANALYSIS.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📱 Customer Churn Prediction App")
st.markdown("Upload your customer data and check who is likely to churn.")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(data.head())

    if st.button("🚀 Predict Churn"):
        try:
            predictions = model.predict(data)

            # Add predictions to DataFrame
            data["Churn_Prediction"] = ["Yes" if pred == 1 else "No" for pred in predictions]

            st.subheader("✅ Prediction Results")
            st.dataframe(data)

            # Show churn summary
            churn_counts = data["Churn_Prediction"].value_counts()
            st.bar_chart(churn_counts)

            st.success("🎉 Predictions completed successfully!")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
else:
    st.info("👆 Please upload a dataset to start predictions.")

st.markdown("""
---
Made with ❤️ using Streamlit  
""")
