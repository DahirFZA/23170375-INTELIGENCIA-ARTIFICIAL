import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf


class AppDibujarNumero:
    def __init__(self, root, ruta_modelo="MLP/modelo_mnist.keras"):
        self.root = root
        self.root.title("Predictor de Dígitos - MLP MNIST")
        self.modelo = tf.keras.models.load_model(ruta_modelo)

        self.tam_canvas = 280
        self.canvas = tk.Canvas(root, width=self.tam_canvas, height=self.tam_canvas, bg="black", cursor="cross")
        self.canvas.pack(pady=10)

        # Imagen en memoria sincronizada con el canvas (fondo negro, trazo blanco)
        self.imagen = Image.new("L", (self.tam_canvas, self.tam_canvas), 0)
        self.draw = ImageDraw.Draw(self.imagen)

        self.canvas.bind("<B1-Motion>", self.dibujar)

        frame_botones = tk.Frame(root)
        frame_botones.pack(pady=5)

        tk.Button(frame_botones, text="Predecir", command=self.predecir, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(frame_botones, text="Limpiar", command=self.limpiar, width=12).pack(side=tk.LEFT, padx=5)

        self.etiqueta_resultado = tk.Label(root, text="Dibuja un número (0-9)", font=("Arial", 14, "bold"))
        self.etiqueta_resultado.pack(pady=10)

    def dibujar(self, evento):
        r = 10
        x1, y1 = evento.x - r, evento.y - r
        x2, y2 = evento.x + r, evento.y + r
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def limpiar(self):
        self.canvas.delete("all")
        self.imagen = Image.new("L", (self.tam_canvas, self.tam_canvas), 0)
        self.draw = ImageDraw.Draw(self.imagen)
        self.etiqueta_resultado.config(text="Dibuja un número (0-9)")

    def predecir(self):
        # Redimensionar a 28x28 píxeles como espera MNIST
        img_28 = self.imagen.resize((28, 28), Image.Resampling.LANCZOS)
        arreglo = np.array(img_28).astype("float32") / 255.0
        arreglo = arreglo.reshape(1, 28, 28)

        predicciones = self.modelo.predict(arreglo, verbose=0)[0]
        digito = int(np.argmax(predicciones))
        confianza = float(np.max(predicciones)) * 100

        self.etiqueta_resultado.config(text=f"Predicción: {digito} ({confianza:.1f}% confianza)")


if __name__ == "__main__":
    ventana = tk.Tk()
    app = AppDibujarNumero(ventana)
    ventana.mainloop()