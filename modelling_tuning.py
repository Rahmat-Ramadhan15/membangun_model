import mlflow
import dagshub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Inisialisasi MLflow ke localhost
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Inisialisasi DagsHub
dagshub.init(repo_owner="Rahmat-Ramadhan15", repo_name="membangun_model", mlflow=True)

# Set eksperimen
mlflow.set_experiment("Telco Customer Churn")

# Load dataset hasil preprocessing
df = pd.read_csv("data_automate_processing.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    params = {
        'n_estimators': [100, 150],
        'max_depth': [5, 10]
    }

    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, params, cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Manual logging
    mlflow.log_param("best_params", grid.best_params_)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))

    # Simpan model
    mlflow.sklearn.log_model(best_model, "model")
    print("Model dan metrik berhasil disimpan ke MLflow & DagsHub.")
