import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath("."))

from src.predict import load_pipeline, predict_single_customer

st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Credit Risk & Default Prediction Dashboard")
st.markdown(
    '''
    This system calculates **Loan Default Risk** (the probability of non-payment)
    based on customer application information, using a machine learning pipeline 
    (`RandomForestClassifier`).
    '''
)
st.write("---")

@st.cache_resource
def get_cached_pipeline():
    return load_pipeline()

try:
    pipeline = get_cached_pipeline()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

st.subheader("📋 Customer Application Information")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Annual Income",
        min_value=1000,
        max_value=1000000,
        value=50000,
        step=1000,
        help="The annual gross income declared by the customer."
    )
    age = st.slider(
        "Customer Age",
        min_value=18,
        max_value=90,
        value=30,
        help="The applicant's age."
    )

with col2:
    loan = st.number_input(
        "Requested Loan Amount",
        min_value=500,
        max_value=500000,
        value=15000,
        step=500,
        help="The total principal amount the customer wishes to withdraw."
    )
    loan_to_income = loan / income if income > 0 else 0
    st.metric(
        label="Debt-to-Income Ratio",
        value=f"{loan_to_income:.2f}",
        help="The ratio of the loan amount to income. A ratio above 0.40 is generally a risk signal."
    )

st.write("---")

if st.button("🚀 Assess Credit Risk", use_container_width=True):
    with st.spinner("The model is performing an evaluation..."):
        customer_payload = {
            "Income" : float(income),
            "Age" : float(age),
            "Loan" : float(loan)
        }

        result = predict_single_customer(customer_payload, pipeline=pipeline)

        st.subheader("🎯 Evaluation Result")

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            if result["prediction"] == 0:
                st.success("✅ **Credit Approval Recommended**")
                st.write("**Status:** Safe Profile (Pays Regularly)")
            else:
                st.error("⚠️ **Credit Refusal / High Risk**")
                st.write("**Status:** High-Risk Profile (Default Expected)")

        with col_res2:
            st.metric(
                label="Probability of Default (Bankruptcy)",
                value=f"%{result['default_probability']:.1f}"
            )

        risk_ratio = result["default_probability"] / 100
        st.progress(risk_ratio)

        if result["prediction"] == 1:
            st.warning(
                "💡**Explanation:** The loan amount requested by the customer is " \
                "high relative to their income or exceeds risk thresholds. " \
                "Additional collateral or a guarantor may be required."
            )
        else:
            st.info(
                "💡 **Explanation:** The customer's financial indicators " \
                "appear safe within the credit score limits."
            )
