import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Configuración de datos simulados (5,000 transacciones)
np.random.seed(24)
n_transacciones = 5000

# El fraude es un evento raro en la banca (simulamos un 3% de fraude real)
realidad = np.random.choice([0, 1], size=n_transacciones, p=[0.97, 0.03])

# El modelo asigna probabilidades más altas a los fraudes reales
probabilidades = np.where(
    realidad == 1,
    np.random.beta(5, 2, size=n_transacciones),  # Probabilidades altas para fraude
    np.random.beta(1, 8, size=n_transacciones)   # Probabilidades bajas para legítimas
)

df = pd.DataFrame({'real': realidad, 'probabilidad_ml': probabilidades})
print("=== DATOS SIMULADOS DE TRANSACCIONES BANCARIAS ===")
print(df.head())

# 2. Definición de Costos Financieros del Negocio
COSTO_FALSO_POSITIVO = 20   # Molestar al cliente legítimo
COSTO_FALSO_NEGATIVO = 150  # Dinero perdido por fraude no detectado

# 3. Función para calcular el impacto económico por umbral
def calcular_costo_bancario(df, umbral):
    # Predicción según el umbral evaluado
    prediccion = (df['probabilidad_ml'] >= umbral).astype(int)
    
    # Identificar errores
    falsos_positivos = ((prediccion == 1) & (df['real'] == 0)).sum()
    falsos_negativos = ((prediccion == 0) & (df['real'] == 1)).sum()
    
    # Calcular costo financiero total
    costo_total = (falsos_positivos * COSTO_FALSO_POSITIVO) + (falsos_negativos * COSTO_FALSO_NEGATIVO)
    return costo_total, falsos_positivos, falsos_negativos

# 4. Evaluación de un rango de umbrales
umbrales_a_probar = np.arange(0.05, 0.96, 0.05)
resultados = []

for u in umbrales_a_probar:
    costo, fp, fn = calcular_costo_bancario(df, u)
    resultados.append({'Umbral': round(u, 2), 'Costo_Total_USD': costo, 'Falsos_Positivos': fp, 'Falsos_Negativos': fn})

df_resultados = pd.DataFrame(resultados)
print("\n=== RESULTADOS DE LA EVALUACIÓN DE UMBRALES ===")
print(df_resultados)

# 5. Encontrar el óptimo de negocio vs el estándar (0.50)
row_estandar = df_resultados[df_resultados['Umbral'] == 0.50].iloc[0]
row_optimo = df_resultados.sort_values(by='Costo_Total_USD').iloc[0]

print("=== OPTIMIZACIÓN DE UMBRALES DE FRAUDE ===")
print(f"Métrica Estándar (Umbral 0.50):")
print(f"  -> Costo Total: ${row_estandar['Costo_Total_USD']:,.2f} USD")
print(f"  -> Falsos Positivos: {row_estandar['Falsos_Positivos']} | Falsos Negativos: {row_estandar['Falsos_Negativos']}")

print("-" * 50)

print(f"Métrica Óptima de Negocio (Umbral {row_optimo['Umbral']}):")
print(f"  -> Costo Total: ${row_optimo['Costo_Total_USD']:,.2f} USD")
print(f"  -> Falsos Positivos: {row_optimo['Falsos_Positivos']} | Falsos Negativos: {row_optimo['Falsos_Negativos']}")
print(f"\n¡Ahorro neto para el banco!: ${row_estandar['Costo_Total_USD'] - row_optimo['Costo_Total_USD']:,.2f} USD")

#6. Visualización del impacto económico por umbral
plt.figure(figsize=(8, 4))
plt.plot(df_resultados['Umbral'], df_resultados['Costo_Total_USD'], marker='o', color='darkblue')
plt.axvline(x=row_optimo['Umbral'], color='red', linestyle='--', label=f"Óptimo ({row_optimo['Umbral']})")
plt.title("Costo Financiero Total según el Umbral del Modelo")
plt.xlabel("Umbral de Probabilidad (Threshold)")
plt.ylabel("Costo Total (USD)")
plt.legend()
plt.grid(True)
plt.savefig("costo_financiero_total_por_umbral.png", dpi=300, bbox_inches='tight')
plt.show()