import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

# Cargar el modelo usando ruta absoluta relativa a este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "red_digitos_cnn.keras")
modelo = tf.keras.models.load_model(RUTA_MODELO)

def predecir_digito(datos_lienzo):
    if datos_lienzo is None or "composite" not in datos_lienzo:
        return {}

    # Obtener la imagen dibujada en escala de grises
    imagen_rgba = Image.fromarray(datos_lienzo["composite"]).convert("L")
    
    # Invertir colores si el fondo es blanco y el trazo negro
    arreglo_temp = np.array(imagen_rgba)
    if np.mean(arreglo_temp) > 127:
        arreglo_temp = 255 - arreglo_temp
    imagen_gris = Image.fromarray(arreglo_temp)

    # Recortar y centrar el dígito igual que en MNIST (20x20 dentro de 28x28)
    caja = imagen_gris.getbbox()
    if caja is None:
        return {}

    recorte = imagen_gris.crop(caja)
    recorte.thumbnail((20, 20), Image.Resampling.LANCZOS)

    lienzo_mnist = Image.new("L", (28, 28), color=0)
    offset_x = (28 - recorte.width) // 2
    offset_y = (28 - recorte.height) // 2
    lienzo_mnist.paste(recorte, (offset_x, offset_y))

    entrada = (np.array(lienzo_mnist, dtype="float32") / 255.0).reshape(1, 28, 28, 1)
    probabilidades = modelo.predict(entrada, verbose=0)[0]

    return {str(i): float(probabilidades[i]) for i in range(10)}

interfaz = gr.Interface(
    fn=predecir_digito,
    inputs=gr.Sketchpad(
        label="Dibuja un dígito (0-9)",
        type="numpy",
        canvas_size=(280, 280),
        brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed")
    ),
    outputs=gr.Label(num_top_classes=3, label="Predicción en tiempo real"),
    title="Clasificador CNN de Dígitos (MNIST)",
    description="Dibuja un número del 0 al 9 en el recuadro y presiona Submit (o usa el modo continuo).",
    live=True
)

if __name__ == "__main__":
    interfaz.launch()