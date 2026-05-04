import pandas as pd
import numpy as np
import mlflow
import json

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("drilling_rate_m_per_hr", axis=1)
y = df["drilling_rate_m_per_hr"]

alphas = [0.1, 1.0, 10.0, 50.0]

results = []

mlflow.set_experiment("drillsense-hyperparameter-tuning")

for alpha in alphas:
    with mlflow.start_run():
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        preds = model.predict(X)

        rmse = np.sqrt(mean_squared_error(y, preds))
        mae = mean_absolute_error(y, preds)

        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        results.append({
            "alpha": alpha,
            "rmse": rmse,
            "mae": mae
        })

# Select best alpha
best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "drillsense-hyperparameter-tuning",
    "results": results,
    "best_alpha": best["alpha"],
    "best_rmse": best["rmse"]
}

# Save JSON
with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 2 completed ✅")