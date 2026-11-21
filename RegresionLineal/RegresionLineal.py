#Zavala Arredondo Dahir Fernando
#23170375
import os
import numpy as np
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    np.random.seed(42)
    X = 2.5 * np.random.rand(60, 1) + 1.0 
    y = 15.0 + 8.5 * X.squeeze() + np.random.randn(60) * 2.5 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== RESULTADOS REGRESIÓN LINEAL (SCIKIT-LEARN) ===")
    print(f"Pendiente (coeficiente w1): {modelo.coef_[0]:.4f}")
    print(f"Intercepto (sesgo b):       {modelo.intercept_:.4f}")
    print(f"Error Cuadrático Medio:     {mse:.4f}")
    print(f"Coeficiente R^2:            {r2:.4f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color="tab:blue", alpha=0.7, label="Datos de entrenamiento")
    plt.scatter(X_test, y_test, color="tab:orange", alpha=0.9, label="Datos de prueba")

    x_linea = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_linea = modelo.predict(x_linea)
    plt.plot(x_linea, y_linea, color="tab:red", linewidth=2, label="Recta de regresión")

    plt.title("Regresión Lineal Simple")
    plt.xlabel("X (Variable independiente)")
    plt.ylabel("Y (Variable dependiente)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    salida = "RegresionLineal/grafica_regresion.png"
    plt.savefig(salida, dpi=150)
    print(f"Gráfica guardada en: {salida}")
    plt.show()


if __name__ == "__main__":
    main()