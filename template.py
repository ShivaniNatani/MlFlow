import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pandas as pd

# Load dataset
X, y = load_iris(return_X_y=True)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)
y = pd.Series(y, name="target")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to compare
model_configs = [
    ("LogisticRegression", LogisticRegression(max_iter=200)),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42))
]

# Set MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("First Experiment")

# Train and log each model
for model_name, model in model_configs:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Generate signature & input example
    input_example = X_test.head(2)
    signature = infer_signature(X_test, model.predict(X_test))

    # Log to MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("f1_score", report["macro avg"]["f1-score"])
        mlflow.log_metric("recall", report["macro avg"]["recall"])

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        print(f"✅ Logged {model_name} to MLflow")

