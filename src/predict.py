from src.model_utils import load_model, get_sample_data

def predict():
    model = load_model()
    X = get_sample_data()
    preds = model.predict(X)
    print("🔍 Predictions for 5 samples:", [round(p, 2) for p in preds])
    return preds