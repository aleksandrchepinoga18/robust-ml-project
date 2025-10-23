import joblib
from sklearn.datasets import fetch_california_housing

def load_model(path="models/best_model.pkl"):
    return joblib.load(path)

def get_sample_data(n=5):
    data = fetch_california_housing()
    return data.data[:n]