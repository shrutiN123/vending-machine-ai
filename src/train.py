import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import build_autoencoder
import matplotlib.pyplot as plt

data = pd.read_csv("data/vending_machine_data.csv")

X = data.drop("label", axis=1)
y = data["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train = X_scaled[y == 0]

model = build_autoencoder(X_train.shape[1])

history = model.fit(
    X_train, X_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

model.save("models/autoencoder.h5")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'val'])
plt.savefig("results/loss.png")
plt.show()
