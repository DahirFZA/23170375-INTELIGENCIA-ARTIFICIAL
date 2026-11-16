import numpy as np

# Datos de ejemplo: Horas de estudio (X) vs Calificación obtenida (Y)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y = np.array([55.0, 62.0, 71.0, 78.0, 88.0])

# 1. Parámetros iniciales de la recta: y_pred = m * x + b
m = 0.0  # Pendiente inicial
b = 0.0  # Intercepto inicial
tasa_aprendizaje = 0.05
epocas = 1000
n = float(len(X))

print("=== ENTENDIENDO LA REGRESIÓN LINEAL (DESCENSO DEL GRADIENTE) ===")
print(f"Fórmula del modelo: y = m*x + b\n")

# 2. Entrenamiento iterativo
for epoca in range(epocas):
    # Predicción actual
    Y_pred = m * X + b

    # Error Cuadrático Medio (MSE)
    mse = np.mean((Y - Y_pred) ** 2)

    # Derivadas parciales (gradientes) respecto a m y b
    dm = (-2.0 / n) * np.sum(X * (Y - Y_pred))
    db = (-2.0 / n) * np.sum(Y - Y_pred)

    # Actualización de parámetros en dirección opuesta al gradiente
    m -= tasa_aprendizaje * dm
    b -= tasa_aprendizaje * db

    if epoca % 200 == 0 or epoca == epocas - 1:
        print(f"Época {epoca:04d} | MSE: {mse:8.4f} | m (pendiente): {m:6.3f} | b (intercepto): {b:6.3f}")

# 3. Evaluación final
print("\n=== MODELO FINAL APRENDIDO ===")
print(f"Ecuación resultante: y = {m:.3f}x + {b:.3f}")

horas_nuevas = 6.0
prediccion = m * horas_nuevas + b
print(f"Si un alumno estudia {horas_nuevas} horas, calificación estimada: {prediccion:.2f}")