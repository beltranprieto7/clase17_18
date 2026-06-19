import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

np.random.seed(42)

# Leer el dataset
df = pd.read_csv("datos_credito.csv")

n_muestras = df.shape[0]

# Crear la variable objetivo (1: Aceptado, 0: Rechazado) basada en reglas lógicas con ruido
score = (
    (df["ingresos_mensuales"] * 0.3)
    + (df["historial_crediticio"] * 1500)
    - (df["deuda_actual"] * 0.2)
    + (df["antiguedad_laboral_meses"] * 10)
)
df["estado_credito"] = (score > np.median(score)).astype(int)

# 2. Separar variables predictoras (X) y variable objetivo (y)
X = df.drop(columns=["estado_credito"])
y = df["estado_credito"]

# 3. Dividir el dataset en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Configurar y entrenar el modelo XGBoost
modelo = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
)

modelo.fit(X_train, y_train)

# 5. Realizar predicciones
predicciones = modelo.predict(X_test)
probabilidades = modelo.predict_proba(X_test)[:, 1]

# 6. Evaluar el rendimiento del modelo
print("--- Reporte de Clasificación ---")
print(classification_report(y_test, predicciones))

print(f"Área bajo la curva ROC (AUC-ROC): {roc_auc_score(y_test, probabilidades):.4f}")

# 7. Ver la importancia de cada variable en la decisión
importancias = pd.DataFrame(
    {"Variable": X.columns, "Importancia": modelo.feature_importances_}
).sort_values(by="Importancia", ascending=False)

print("\n--- Importancia de las Variables ---")
print(importancias.to_string(index=False))

# Exportar el modelo entrenado para su uso futuro
joblib.dump(modelo, "modelo_credito_xgboost.pkl")