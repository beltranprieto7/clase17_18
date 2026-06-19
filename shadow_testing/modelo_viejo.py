import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
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

# 4. ESCALAR LOS DATOS (Paso obligatorio para Regresión Logística)
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

# 5. Configurar y entrenar el modelo de Regresión Logística
modelo = LogisticRegression(random_state=42)
modelo.fit(X_train_escalado, y_train)

# 6. Realizar predicciones
predicciones = modelo.predict(X_test_escalado)
probabilidades = modelo.predict_proba(X_test_escalado)[:, 1]

# 7. Evaluar el rendimiento del modelo
print("--- Reporte de Clasificación (Regresión Logística) ---")
print(classification_report(y_test, predicciones))

print(f"Área bajo la curva ROC (AUC-ROC): {roc_auc_score(y_test, probabilidades):.4f}")

# 8. Ver la importancia de cada variable (Coeficientes)
coeficientes = pd.DataFrame(
    {"Variable": X.columns, "Coeficiente": modelo.coef_[0]}
).sort_values(by="Coeficiente", ascending=False)

print("\n--- Coeficientes del Modelo (Impacto en la decisión) ---")
print(coeficientes.to_string(index=False))

# Exportar el modelo entrenado y el escalador para uso futuro
joblib.dump(modelo, "modelo_credito_regresion_logistica.pkl")
joblib.dump(escalador, "escalador.pkl")