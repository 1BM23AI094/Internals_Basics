import pandas as pd
import numpy as np
import mlflow
import json

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("drilling_rate_m_per_hr", axis=1)
y = df["drilling_rate_m_per_hr"]

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0)
}

results = []

mlflow.set_experiment("drillsense-drilling-rate-m-per-hr")

for name, model in models.items():
    with mlflow.start_run():
        model.fit(X, y)
        preds = model.predict(X)

        rmse = np.sqrt(mean_squared_error(y, preds))
        mae = mean_absolute_error(y, preds)

        mlflow.log_param("model", name)
        if name == "Ridge":
            mlflow.log_param("alpha", 1.0)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        mlflow.set_tag("project_phase", "model_selection")

        results.append({
            "name": name,
            "rmse": rmse,
            "mae": mae
        })

# Select best model
best_model = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "drillsense-drilling-rate-m-per-hr",
    "models": results,
    "best_model": best_model["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best_model["rmse"]
}

# Save result
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 1 completed ✅")