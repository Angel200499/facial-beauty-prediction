import pandas as pd
from sklearn.linear_model import LinearRegression

# load dataset
data = pd.read_csv("dataset.csv")

X = data[['symmetry', 'proportion']]
y = data['score']

# buat model
model = LinearRegression()
model.fit(X, y)

# simpan model
import pickle
with open("ml_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model berhasil dilatih!")