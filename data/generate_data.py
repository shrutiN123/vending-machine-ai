import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 1000

data = pd.DataFrame({
    "temperature": np.random.normal(25, 3, n_samples),
    "humidity": np.random.normal(50, 10, n_samples),
    "sales": np.random.normal(200, 50, n_samples),
    "stock_level": np.random.normal(500, 100, n_samples)
})

# anomalies
n_anomalies = 50
anomalies = pd.DataFrame({
    "temperature": np.random.uniform(40, 60, n_anomalies),
    "humidity": np.random.uniform(80, 100, n_anomalies),
    "sales": np.random.uniform(10, 50, n_anomalies),
    "stock_level": np.random.uniform(50, 150, n_anomalies)
})

data["label"] = 0
anomalies["label"] = 1

final_data = pd.concat([data, anomalies])
final_data.to_csv("data/vending_machine_data.csv", index=False)

print("Dataset created")
