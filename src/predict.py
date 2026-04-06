import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/vending_machine_data.csv")

X = data.drop("label", axis=1)
y = data["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = load_model("models/autoencoder.h5")

recon = model.predict(X_scaled)
mse = np.mean((X_scaled - recon) ** 2, axis=1)

threshold = np.percentile(mse, 95)

pred = (mse > threshold).astype(int)

print("Detected anomalies:", sum(pred))
