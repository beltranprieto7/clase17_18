# Shadow testing con el modelo viejo (Regresión Logística) y el modelo nuevo (XGBoost)
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

# 1. Cargar ambos modelos y el dataset de prueba

# Cargar el modelo viejo (Regresión Logística) y su escalador
modelo_viejo = joblib.load("modelo_credito_regresion_logistica.pkl")
escalador = joblib.load("escalador.pkl")

# Cargar el modelo nuevo (XGBoost)
modelo_nuevo = joblib.load("modelo_credito_xgboost.pkl")

# Leer el dataset de prueba
df_prueba = pd.read_csv("datos_credito.csv")

# Crear la variable objetivo (1: Aceptado, 0: Rechazado) basada en reglas lógicas con ruido
score = (
    (df_prueba["ingresos_mensuales"] * 0.3)
    + (df_prueba["historial_crediticio"] * 1500)
    - (df_prueba["deuda_actual"] * 0.2)
    + (df_prueba["antiguedad_laboral_meses"] * 10)
)
df_prueba["estado_credito"] = (score > np.median(score)).astype(int)

# 2. Preparar los datos para ambos modelos y realizar predicciones

# Separar variables predictoras (X) y variable objetivo (y)
X_prueba = df_prueba.drop(columns=["estado_credito"])
y_prueba = df_prueba["estado_credito"]

# Escalar los datos para el modelo viejo
X_prueba_escalado = escalador.transform(X_prueba)

# Realizar predicciones con ambos modelos
predicciones_viejo = modelo_viejo.predict(X_prueba_escalado)
predicciones_nuevo = modelo_nuevo.predict(X_prueba)

# 3. Cálculo de la Tasa de Discrepancia. 
# Es el porcentaje de casos donde las decisiones son diferentes
casos_discrepantes = predicciones_viejo != predicciones_nuevo
discrepancias = np.sum(casos_discrepantes)
tasa_discrepancia = discrepancias / len(y_prueba) * 100

# 4. Cálculo del Riesgo Expuesto Financiero
# Casos donde el nuevo modelo dice "Aprobar" (1) pero el modelo viejo dice "Rechazar" (0)
casos_riesgosos = (predicciones_nuevo == 1) & (predicciones_viejo == 0)
riesgo_expuesto = np.sum(casos_riesgosos)
# sumar columna de deuda_actual para esos casos de riesgo y así tener una idea del monto total en riesgo
monto_riesgo_expuesto = df_prueba.loc[casos_riesgosos, "deuda_actual"].sum()
# identificar casos específicos de riesgo expuesto
casos_riesgo_expuesto = df_prueba.loc[casos_riesgosos, ["ingresos_mensuales", "historial_crediticio", "deuda_actual", "antiguedad_laboral_meses"]]

# 5. Impresión de Resultados
print("=== RESULTADOS DEL SHADOW TESTING ===")
print(f"Total de solicitudes evaluadas: {len(y_prueba)}")
print(f"Casos con discrepancia entre modelos: {discrepancias}")
print(f"Tasa de Discrepancia Global: {tasa_discrepancia:.2f}%")

print("-" * 37)

print(f"Riesgo Expuesto Financiero (Nuevas Aprobaciones Potencialmente Riesgosas):")
print(f"Casos donde el nuevo modelo arriesga más que el modelo actual: {riesgo_expuesto}")
print(f"Monto total en riesgo expuesto por nuevo modelo: ${monto_riesgo_expuesto:.2f}")
print("\nCasos de Riesgo Expuesto:")
print(casos_riesgo_expuesto)