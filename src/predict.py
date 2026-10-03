import os
import joblib
import pandas as pd

def load_pipeline(model_path=None):

    if model_path is None:
        model_path = os.path.join("models", "credit_default_pipeline.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please run src/train.py first.")

    pipeline = joblib.load(model_path)
    return pipeline

def predict_single_customer(customer_data, pipeline=None):

    if pipeline is None:
        pipeline = load_pipeline()

    if isinstance(customer_data, dict):
        df_input = pd.DataFrame([customer_data])

    elif isinstance(customer_data, pd.DataFrame):
        df_input = customer_data.copy()

    else:
        raise ValueError("Invalid data type. 'dict' or 'pd.DataFrame' expected.")

    if "Loan to Income" not in df_input.columns:
        df_input["Loan to Income"] = df_input["Loan"] / df_input["Income"]

    prediction = pipeline.predict(df_input)[0]
    default_probability = pipeline.predict_proba(df_input)[0][1]

    return {
        "prediction" : int(prediction),
        "status": "Risky (Default)" if prediction == 1 else "Safe (Pays Regularly)",
        "default_probability" : round(float(default_probability) * 100, 2)
    }

if __name__ == "__main__":

    print("--- Loading and Testing the Saved Model ---")
    pipeline = load_pipeline()

    customers = [
        {
            "customer" : "Customer_1",
            "data" : {"Income" : 22000, "Age" : 23, "Loan" : 45000}
        },
        {
            "customer" : "Customer_2",
            "data" : {"Income" : 85000, "Age" : 42, "Loan" : 6000}
        },
        {
            "customer" : "Customer_3",
            "data" : {"Income" : 43000, "Age" : 32, "Loan" : 45000}
        }
    ]

    for customer in customers:
        result = predict_single_customer(customer_data=customer["data"], pipeline=pipeline)
        print(f"\n[{customer['customer']}]:")
        print(f"Girdiler: {customer['data']}")
        print(f"Decision: {result['status']} | Probability of Default: %{result['default_probability']}")