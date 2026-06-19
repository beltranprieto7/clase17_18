import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

rng = np.random.default_rng(42)

def generar_datos(n, drift=False):
    """Genera datos sintéticos de solicitudes bancarias con o sin cambio de contexto."""
    ingresos = rng.normal(5200 if not drift else 3600, 900 if not drift else 1100, size=n)
    ingresos = np.clip(ingresos, 1000, 12000)

    ratio_deuda_ingreso = rng.normal(0.35 if not drift else 0.60, 0.07 if not drift else 0.10, size=n)
    ratio_deuda_ingreso = np.clip(ratio_deuda_ingreso, 0.05, 1.20)

    score_crediticio = rng.normal(740 if not drift else 640, 45 if not drift else 55, size=n)
    score_crediticio = np.clip(score_crediticio, 300, 850)

    antiguedad = rng.integers(6, 36, size=n)
    deuda = ingresos * ratio_deuda_ingreso

    riesgo = (
        3.0 * ratio_deuda_ingreso
        + 0.8 * (deuda / 1000)
        - 0.7 * (ingresos / 1000)
        - 0.002 * score_crediticio
        + 0.04 * antiguedad
    )
    p_default = 1 / (1 + np.exp(-(riesgo - 1.5)))
    default = rng.binomial(1, p_default)

    return pd.DataFrame(
        {
            "ingresos": ingresos,
            "ratio_deuda_ingreso": ratio_deuda_ingreso,
            "score_crediticio": score_crediticio,
            "antiguedad": antiguedad,
            "default": default,
        }
    )


def calcular_psi(entrenamiento, produccion, num_buckets=10):
    """Calcula el Population Stability Index entre dos distribuciones."""
    cortes = np.percentile(entrenamiento, np.linspace(0, 100, num_buckets + 1))
    cortes[0] = -np.inf
    cortes[-1] = np.inf

    conteo_entreno = pd.Series(pd.cut(entrenamiento, bins=cortes)).value_counts(sort=False)
    conteo_prod = pd.Series(pd.cut(produccion, bins=cortes)).value_counts(sort=False)

    prop_entreno = conteo_entreno / len(entrenamiento)
    prop_prod = conteo_prod / len(produccion)

    tabla_psi = pd.DataFrame(
        {
            "Entrenamiento (%)": prop_entreno,
            "Produccion (%)": prop_prod,
        }
    )
    tabla_psi = tabla_psi.replace(0, 1e-4)
    tabla_psi["PSI_Bucket"] = (
        (tabla_psi["Produccion (%)"] - tabla_psi["Entrenamiento (%)"])
        * np.log(tabla_psi["Produccion (%)"] / tabla_psi["Entrenamiento (%)"])
    )

    return tabla_psi["PSI_Bucket"].sum(), tabla_psi


print("=== Ejemplo de model drift en riesgo crediticio ===")
print("Contexto: cambio económico que afecta la calidad de las solicitudes bancarias.\n")

# Datos históricos del banco
train_df = generar_datos(2200, drift=False)
# Datos actuales de producción, con deterioro del perfil crediticio
prod_df = generar_datos(1800, drift=True)

X_train = train_df[["ingresos", "ratio_deuda_ingreso", "score_crediticio", "antiguedad"]]
y_train = train_df["default"]
X_prod = prod_df[["ingresos", "ratio_deuda_ingreso", "score_crediticio", "antiguedad"]]
y_prod = prod_df["default"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

prob_train = model.predict_proba(X_train)[:, 1]
prob_prod = model.predict_proba(X_prod)[:, 1]

pred_train = (prob_train >= 0.5).astype(int)
pred_prod = (prob_prod >= 0.5).astype(int)

metrics = {
    "accuracy_train": accuracy_score(y_train, pred_train),
    "accuracy_prod": accuracy_score(y_prod, pred_prod),
    "precision_prod": precision_score(y_prod, pred_prod, zero_division=0),
    "recall_prod": recall_score(y_prod, pred_prod, zero_division=0),
    "roc_auc_prod": roc_auc_score(y_prod, prob_prod),
}

print(f"Accuracy en entrenamiento: {metrics['accuracy_train']:.3f}")
print(f"Accuracy en producción: {metrics['accuracy_prod']:.3f}")
print(f"Precisión en producción: {metrics['precision_prod']:.3f}")
print(f"Recall en producción: {metrics['recall_prod']:.3f}")
print(f"ROC-AUC en producción: {metrics['roc_auc_prod']:.3f}")

psi, tabla_psi = calcular_psi(prob_train, prob_prod, num_buckets=10)
print(f"PSI del score del modelo: {psi:.4f}")

if psi < 0.10:
    print("Estado: estable. No se observa drift significativo.")
elif psi < 0.25:
    print("Alerta: drift moderado. Se recomienda monitoreo y revisión del modelo.")
else:
    print("ALERTA CRÍTICA: se detecta model drift. El perfil crediticio cambió y el modelo debe revisarse o reentrenarse.")

print("\nDetalle del PSI por bucket:")
print(tabla_psi)

plt.figure(figsize=(8, 5))
plt.hist(prob_train, bins=15, density=True, alpha=0.5, label="Entrenamiento", color="steelblue")
plt.hist(prob_prod, bins=15, density=True, alpha=0.5, label="Producción", color="tomato")
plt.xlabel("Probabilidad de default")
plt.ylabel("Densidad")
plt.title("Distribución del score del modelo antes y después del drift")
plt.legend()
plt.tight_layout()
plt.savefig("modelo_drift_bancario.png")
plt.close()

print("\nGráfico guardado como modelo_drift_bancario.png")
