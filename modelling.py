import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Inisialisasi MLflow ke localhost
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Inisialisasi ke DagsHub
dagshub.init(repo_owner="Rahmat-Ramadhan15", repo_name="membangun_model", mlflow=True)

# Set nama eksperimen
mlflow.set_experiment("Telco Customer Churn")

# Aktifkan autologging
mlflow.sklearn.autolog()

# Load data hasil preprocessing
df = pd.read_csv("data_automate_processing.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc}")
