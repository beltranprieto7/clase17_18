import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

print("Caso de estudio: comparar la distribución del score crediticio entre dos periodos.")
rng = np.random.default_rng(42)

# Datos históricos del banco (periodo base)
score_base = np.clip(rng.normal(loc=720, scale=45, size=1200), 300, 850)

# Datos actuales (periodo de monitoreo)
# Se simula un cambio en la distribución para evidenciar possible drift
score_actual = np.clip(rng.normal(loc=690, scale=55, size=1200), 300, 850)

datos = pd.DataFrame(
    {
        "score_crediticio": np.concatenate([score_base, score_actual]),
        "periodo": ["histórico"] * len(score_base) + ["actual"] * len(score_actual),
    }
)

# Prueba KS de dos muestras
estadistico, p_valor = ks_2samp(score_base, score_actual)

# Interpretación
alpha = 0.05
decision = "rechazar H0" if p_valor < alpha else "no rechazar H0"
interpretacion = (
    "Se detecta evidencia de cambio en la distribución del score crediticio."
    if p_valor < alpha
    else "No hay evidencia suficiente de cambio en la distribución."
)

print("=== Caso de estudio: detección de drift en riesgo crediticio ===")
print(f"Promedio histórico: {score_base.mean():.2f}")
print(f"Promedio actual: {score_actual.mean():.2f}")
print(f"Estadístico KS: {estadistico:.4f}")
print(f"p-valor: {p_valor:.4f}")
print(f"Nivel de significancia alpha: {alpha}")
print(f"Decisión: {decision}")
print(interpretacion)

# Guardar gráfico para inspección visual
plt.figure(figsize=(8, 5))
plt.hist(score_base, bins=30, density=True, alpha=0.5, label="Histórico", color="steelblue")
plt.hist(score_actual, bins=30, density=True, alpha=0.5, label="Actual", color="tomato")
plt.xlabel("Score crediticio")
plt.ylabel("Densidad")
plt.title("Distribución del score crediticio por periodo")
plt.legend()
plt.tight_layout()
plt.savefig("ks_credit_score_banco.png")
plt.close()

print("\nGráfico guardado como ks_credit_score_banco.png")
