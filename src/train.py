import os
import joblib as jb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def build_pipeline(model, numeric_features):

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features)
    ])

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return full_pipeline

def run_training():

    data_path = os.path.join("data", "credit_risk_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. make_dataset.py must be run first.")

    df = pd.read_csv(data_path)

    feature_cols = ["Income", "Age", "Loan"]
    X = df[feature_cols].copy()
    y = df["Default"]

    X["Loan to Income"] = df["Loan"] / df["Income"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic_Regression" : LogisticRegression(class_weight="balanced", random_state=42),
        "Random_Forest" : RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    }

    best_score = 0
    best_pipeline = None
    best_model_name = ""

    for name, clf in models.items():

        print(f"{'='*20} {name} Training... {'='*20}")
        pipeline = build_pipeline(clf, numeric_features)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC Score: {auc:.4f}")
        clf_report = classification_report(y_test, y_pred)
        print("Classification Report: ")
        print(clf_report)

        if auc > best_score:
            best_score = auc
            best_pipeline = pipeline
            best_model_name = name

    os.makedirs("models", exist_ok=True)
    model_output_path = os.path.join("models", "credit_default_pipeline.pkl")
    jb.dump(best_pipeline, model_output_path)
    print(f"\nThe best-performing model ({best_model_name} - AUC: {best_score:.4f}) has been saved to disk: {model_output_path}")

if __name__ == "__main__":
    run_training()