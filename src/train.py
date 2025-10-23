import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

def train_models():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "rf": RandomForestRegressor(n_estimators=50, random_state=42),
        "hgb": HistGradientBoostingRegressor(random_state=42),
        "huber": HuberRegressor(max_iter=1000)
    }

    results = {}
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        results[name] = mae
        joblib.dump(model, f"models/{name}_model.pkl")
        print(f"  → MAE: {mae:.4f}")

    best_model_name = min(results, key=results.get)
    print(f"\n🏆 Best model: {best_model_name} (MAE={results[best_model_name]:.4f})")
    
    # Сохраняем лучшую модель отдельно
    best_model = joblib.load(f"models/{best_model_name}_model.pkl")
    joblib.dump(best_model, "models/best_model.pkl")
    return best_model_name