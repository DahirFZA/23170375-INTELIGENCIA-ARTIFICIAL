#Zavala Arredondo Dahir Fernando
#23170375
import numpy as np

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y = np.array([55.0, 62.0, 71.0, 78.0, 88.0])


m = 0.0 
b = 0.0 
tasa_aprendizaje = 0.05
epocas = 1000
n = float(len(X))

print("=== ENTENDIENDO LA REGRESIÓN LINEAL (DESCENSO DEL GRADIENTE) ===")
print(f"Fórmula del modelo: y = m*x + b\n")


for epoca in range(epocas):
    Y_pred = m * X + b

    mse = np.mean((Y - Y_pred) ** 2)

    dm = (-2.0 / n) * np.sum(X * (Y - Y_pred))
    db = (-2.0 / n) * np.sum(Y - Y_pred)

    m -= tasa_aprendizaje * dm
    b -= tasa_aprendizaje * db

    if epoca % 200 == 0 or epoca == epocas - 1:
        print(f"Época {epoca:04d} | MSE: {mse:8.4f} | m (pendiente): {m:6.3f} | b (intercepto): {b:6.3f}")

print("\n=== MODELO FINAL APRENDIDO ===")
print(f"Ecuación resultante: y = {m:.3f}x + {b:.3f}")

horas_nuevas = 6.0
prediccion = m * horas_nuevas + b
print(f"Si un alumno estudia {horas_nuevas} horas, calificación estimada: {prediccion:.2f}")