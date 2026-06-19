import numpy as np
import pandas as pd

# Fijar semilla para que el resultado sea siempre el mismo
np.random.seed(42)

# 1. Datos de entrenamiento (Clientes con ingresos estables y altos)
ingresos_entrenamiento = np.random.normal(loc=3500, scale=800, size=1000)

# 2. Datos de producción actuales (Los ingresos bajaron por cambios económicos)
ingresos_produccion = np.random.normal(loc=2800, scale=700, size=1000)

# Convertir a DataFrames de Pandas
df_entrenamiento = pd.DataFrame({'ingresos': ingresos_entrenamiento})
df_produccion = pd.DataFrame({'ingresos': ingresos_produccion})

# Calcular el PSI (Population Stability Index)
# El cálculo divide los datos de entrenamiento en 10 bloques (buckets) 
# con la misma cantidad de datos y compara cuántos registros caen en esos mismos rangos 
# dentro del grupo de producción.
def calcular_psi(entrenamiento, produccion, num_buckets=10):
    # Crear los cortes (buckets) basados en los cuantiles del entrenamiento
    cortes = np.percentile(entrenamiento, np.linspace(0, 100, num_buckets + 1))
    
    # Ajustar extremos para evitar errores de límites fuera de rango
    cortes[0] = -np.inf
    cortes[-1] = np.inf
    
    # Contar cuántos datos caen en cada bucket
    conteo_entreno = pd.cut(entrenamiento, bins=cortes).value_counts(sort=False)
    conteo_prod = pd.cut(produccion, bins=cortes).value_counts(sort=False)
    
    # Calcular los porcentajes (proporciones) en cada bucket
    prop_entreno = conteo_entreno / len(entrenamiento)
    prop_prod = conteo_prod / len(produccion)
    
    # Crear un DataFrame para comparar ambos lados
    tabla_psi = pd.DataFrame({
        'Entrenamiento (%)': prop_entreno,
        'Produccion (%)': prop_prod
    })
    
    # Reemplazar ceros con un valor mínimo para evitar divisiones por cero o log(0)
    tabla_psi = tabla_psi.replace(0, 0.0001)
    
    # Fórmula del PSI: (Actual - Esperado) * ln(Actual / Esperado)
    tabla_psi['PSI_Bucket'] = (tabla_psi['Produccion (%)'] - tabla_psi['Entrenamiento (%)']) * np.log(tabla_psi['Produccion (%)'] / tabla_psi['Entrenamiento (%)'])
    
    # El PSI total es la suma de todos los buckets
    psi_total = tabla_psi['PSI_Bucket'].sum()
    
    return psi_total, tabla_psi

# Ejecutar la función con nuestros datos simulados
psi_resultado, tabla_detalle = calcular_psi(df_entrenamiento['ingresos'], df_produccion['ingresos'])

print(f"Resultado del PSI: {psi_resultado:.4f}\n")

# Las reglas estándar en la industria bancaria para interpretar el PSI son:
# PSI < 0.10: Sin cambios significativos (El modelo sigue siendo seguro).
# 0.10 ≤ PSI < 0.25: Desviación moderada (Requiere monitoreo o ajustes leves).
# PSI ≥ 0.25: Desviación severa (Concept Drift detectado, el modelo debe ser reentrenado).

# Evaluar el nivel de riesgo según los estándares bancarios
if psi_resultado < 0.10:
    print("Estado: Estable. No hay drift. El modelo es confiable.")
elif psi_resultado < 0.25:
    print("Alerta: Desviación moderada. Monitorear de cerca los ingresos de los solicitantes.")
else:
    print("ALERTA CRÍTICA: Concept Drift Detectado. El perfil de ingresos cambió drásticamente.")
    print("Acción requerida: Detener transiciones automáticas y reentrenar el modelo de riesgo.")

# Mostrar la tabla de detalle del PSI por bucket
print("\nDetalle del PSI por bucket:")
print(tabla_detalle)

# Graficar la distribución de ingresos para inspección visual
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.hist(df_entrenamiento['ingresos'], bins=30, density=True, alpha=0.5, label="Entrenamiento", color="steelblue")
plt.hist(df_produccion['ingresos'], bins=30, density=True, alpha=0.5, label="Producción", color="tomato")
plt.xlabel("Ingresos")
plt.ylabel("Densidad")
plt.title("Distribución de ingresos: Entrenamiento vs Producción")
plt.legend()
plt.tight_layout()
plt.savefig("psi_ingresos_banco.png")
plt.close()