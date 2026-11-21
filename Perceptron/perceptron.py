#Zavala Arredondo Dahir Fernando
#23170375
import os
import numpy as np
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, num_entradas, tasa_aprendizaje=0.1, epocas=20):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.pesos = np.zeros(num_entradas)
        self.sesgo = 0.0

    def funcion_activacion(self, z):
        """Función escalón unitario."""
        return 1 if z >= 0 else 0

    def predecir(self, x):
        """Calcula la suma ponderada z = w·x + b y aplica activación."""
        z = np.dot(x, self.pesos) + self.sesgo
        return self.funcion_activacion(z)

    def entrenar(self, X, y):
        """Ajusta pesos y sesgo usando la regla de aprendizaje del perceptrón."""
        for epoca in range(self.epocas):
            errores = 0
            for xi, objetivo in zip(X, y):
                prediccion = self.predecir(xi)
                error = objetivo - prediccion
                if error != 0:
                    self.pesos += self.tasa_aprendizaje * error * xi
                    self.sesgo += self.tasa_aprendizaje * error
                    errores += 1
            print(f"Época {epoca + 1:02d} | Errores: {errores} | Pesos: {self.pesos} | Sesgo: {self.sesgo:.2f}")
            if errores == 0:
                print("Convergencia alcanzada.")
                break


def graficar_frontera(X, y, perceptron, titulo, archivo_salida):
    plt.figure(figsize=(6, 5))
    for punto, etiqueta in zip(X, y):
        color = "tab:blue" if etiqueta == 1 else "tab:red"
        marcador = "o" if etiqueta == 1 else "x"
        plt.scatter(punto[0], punto[1], c=color, marker=marcador, s=100, label=f"Clase {etiqueta}")


    handles, labels = plt.gca().get_legend_handles_labels()
    por_etiqueta = dict(zip(labels, handles))
    plt.legend(por_etiqueta.values(), por_etiqueta.keys())
    if perceptron.pesos[1] != 0:
        x_vals = np.linspace(-0.2, 1.2, 100)
        y_vals = -(perceptron.pesos[0] * x_vals + perceptron.sesgo) / perceptron.pesos[1]
        plt.plot(x_vals, y_vals, "--g", label="Frontera de decisión")

    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.title(titulo)
    plt.xlabel("Entrada X1")
    plt.ylabel("Entrada X2")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(archivo_salida, dpi=150)
    print(f"Gráfica guardada en: {archivo_salida}")
    plt.show()


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    print("=== ENTRENAMIENTO COMPUERTA AND ===")
    y_and = np.array([0, 0, 0, 1])
    p_and = Perceptron(num_entradas=2, tasa_aprendizaje=0.1, epocas=20)
    p_and.entrenar(X, y_and)

    print("\nResultados AND:")
    for xi in X:
        print(f"  Entrada: {xi} -> Predicción: {p_and.predecir(xi)}")

    graficar_frontera(X, y_and, p_and, "Perceptrón - Compuerta AND", "Perceptron/frontera_and.png")
    print("\n=== ENTRENAMIENTO COMPUERTA OR ===")
    y_or = np.array([0, 1, 1, 1])
    p_or = Perceptron(num_entradas=2, tasa_aprendizaje=0.1, epocas=20)
    p_or.entrenar(X, y_or)

    print("\nResultados OR:")
    for xi in X:
        print(f"  Entrada: {xi} -> Predicción: {p_or.predecir(xi)}")

    graficar_frontera(X, y_or, p_or, "Perceptrón - Compuerta OR", "Perceptron/frontera_or.png")