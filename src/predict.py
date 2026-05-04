import pandas as pd
import json
from sklearn.linear_model import Ridge

# Load training data
df = pd.read_csv("data/training_data.csv")

X = df.drop("drilling_rate_m_per_hr", axis=1)
y = df["drilling_rate_m_per_hr"]

# Train model (use best alpha from Task 2, assume 1.0 if unsure)
model = Ridge(alpha=1.0)
model.fit(X, y)

# Example input (you can change if question gives)
input_data = [[5, 100, 2000, 1500]]

prediction = model.predict(input_data)[0]

output = {
    "input": input_data,
    "predicted_drilling_rate": float(prediction)
}

with open("results/prediction_s3.json", "w") as f:
    json.dump(output, f, indent=2)

print("Prediction saved ✅")