import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import os

def explain_model(model_path="models/best_model.pkl", plot_path="reports/shap.png"):
    model = joblib.load(model_path)
    data = fetch_california_housing()
    X = data.data[:100]  # 100 примеров для скорости

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"✅ SHAP plot saved to {plot_path}")