import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import os


def run_eda(save_path="reports/eda.png"):
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 8))
    sns.pairplot(df.sample(500, random_state=42))  # меньше точек — быстрее
    plt.savefig(save_path)
    print(f"✅ EDA saved to {save_path}")