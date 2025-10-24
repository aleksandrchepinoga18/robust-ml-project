# robust-ml-project
# 🛡️ Robust ML Project

Проект по созданию **устойчивой к выбросам** модели машинного обучения с полным ML-пайплайном: от EDA до REST API.

---

## 🎯 Цель проекта

Разработать и задеплоить модель регрессии, устойчивую к выбросам, на примере датасета **California Housing**. Проект включает:
- исследование данных,
- обучение нескольких робастных моделей,
- интерпретацию через SHAP,
- REST API для инференса,
- сохранение метрик и графиков на диск.

---

## 🧠 Используемые модели (устойчивые к выбросам)

| Модель | Библиотека | Особенность |
|--------|-----------|-------------|
| **HistGradientBoostingRegressor** | `sklearn` | Встроенная робастность, работает с пропусками |
| **Random Forest** | `sklearn` | Ансамбль деревьев, нечувствителен к выбросам |
| **Huber Regressor** | `sklearn` | Явно оптимизирован для устойчивости к выбросам |

✅ Лучшая модель выбирается автоматически по **MAE** (Mean Absolute Error).

---

## 📂 Структура проекта

robust-ml-project/
├── src/ # Основной код: обучение, EDA, SHAP
├── app/ # Flask API
├── models/ # Сохранённые модели (.pkl)
├── reports/ # Графики: EDA, SHAP
├── requirements.txt # Зависимости
├── README.md # Этот файл
└── .gitignore

---

## 🚀 Как запустить

1. **Установите зависимости**:
   ```bash
   
   pip install -r requirements.txt

2. **Запустите обучение:**
bash

python -c "from src.train import train_models; train_models()"

4. **Постройте интерпретацию SHAP:**
bash

python -c "from src.evaluate import explain_model; explain_model()"

6. **Запустите API:**
bash

python -m app.api

8. **Проверьте работоспособность:**
bash

curl http://localhost:5000/health

10. **Сделайте предсказание:**

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]}'

📊 Результаты

Лучшая модель: HistGradientBoostingRegressor
MAE: ~0.31
Графики сохраняются в папку reports/:
eda.png — парные графики признаков
shap.png — важность признаков

📦 Требования

Python ≥ 3.9 (рекомендуется 3.10)
Библиотеки: scikit-learn, shap, flask, joblib, matplotlib, seaborn

💡 Особенности

Все скрипты модульны и могут использоваться повторно.
API поддерживает мгновенный инференс.
Проект легко расширить новыми моделями или датасетами.

