#Zavala Arredondo Dahir Fernando
#23170375
import os
import numpy as np
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def ejecutar_california_housing():
    datos = fetch_california_housing()
    X, y = datos.data, datos.target

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.25, random_state=101
    )

    escalador = StandardScaler()
    X_ent_esc = escalador.fit_transform(X_entrenamiento)
    X_pru_esc = escalador.transform(X_prueba)

    regresor = LinearRegression()
    regresor.fit(X_ent_esc, y_entrenamiento)

    predicciones = regresor.predict(X_pru_esc)

    mae = mean_absolute_error(y_prueba, predicciones)
    rmse = np.sqrt(mean_squared_error(y_prueba, predicciones))
    r2 = r2_score(y_prueba, predicciones)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    for nombre, coef in zip(datos.feature_names, regresor.coef_):
        print(f"{nombre:<12}: {coef:+.4f}")

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))

    ejes[0].scatter(y_prueba, predicciones, alpha=0.25, color="#1f77b4", s=12)
    limites = [min(y_prueba.min(), predicciones.min()), max(y_prueba.max(), predicciones.max())]
    ejes[0].plot(limites, limites, "r--", linewidth=1.5)
    ejes[0].set_title("Valor Real vs Predicho (California Housing)")
    ejes[0].set_xlabel("Valor Real ($100k)")
    ejes[0].set_ylabel("Valor Predicho ($100k)")
    ejes[0].grid(True, alpha=0.3)

    ejes[1].barh(datos.feature_names, regresor.coef_, color="#2ca02c")
    ejes[1].set_title("Importancia de Variables (Coeficientes)")
    ejes[1].set_xlabel("Peso Estandarizado")
    ejes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("RegresionLineal/california_resultados.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    ejecutar_california_housing()