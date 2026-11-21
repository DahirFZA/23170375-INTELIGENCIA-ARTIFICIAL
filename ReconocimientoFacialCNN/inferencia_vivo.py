#Zavala Arredondo Dahir Fernando
#23170375
import json
from collections import deque
import cv2
import numpy as np
import tensorflow as tf


RUTA_MODELO = "ReconocimientoFacialCNN/mejor_modelo_facial.keras"
RUTA_CLASES = "ReconocimientoFacialCNN/etiquetas.json"
TAM_IMG = (160, 160)
UMBRAL_CONFIANZA = 0.60


def iniciar_reconocimiento():
    """Ejecuta la detección facial y clasificación en tiempo real sobre el flujo de webcam."""
    modelo = tf.keras.models.load_model(RUTA_MODELO)
    with open(RUTA_CLASES, "r", encoding="utf-8") as f:
        clases = json.load(f)

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    camara = cv2.VideoCapture(0)
    # Ventana deslizante para promediar probabilidades y estabilizar etiquetas entre fotogramas
    buffer_predicciones = deque(maxlen=6)

    while True:
        ok, fotograma = camara.read()
        if not ok:
            break

        # Detección previa de la región de interés (ROI) del rostro
        gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(gris)
        rostros = detector.detectMultiScale(gris, scaleFactor=1.15, minNeighbors=5, minSize=(75, 75))

        for (x, y, w, h) in rostros:
            # Convertir de BGR (OpenCV) a RGB (Keras) y ajustar dimensiones de entrada
            roi_bgr = fotograma[y : y + h, x : x + w]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            roi_redim = cv2.resize(roi_rgb, TAM_IMG, interpolation=cv2.INTER_AREA)
            entrada = np.expand_dims(roi_redim.astype("float32"), axis=0)

            # Inferencia y suavizado temporal de probabilidades
            probs = modelo.predict(entrada, verbose=0)[0]
            buffer_predicciones.append(probs)
            probs_suavizadas = np.mean(buffer_predicciones, axis=0)

            idx = int(np.argmax(probs_suavizadas))
            confianza = float(probs_suavizadas[idx])

            # Asignar etiqueta y color según el umbral mínimo de confianza
            if confianza >= UMBRAL_CONFIANZA:
                texto = f"{clases[idx]} ({confianza * 100:.1f}%)"
                color = (40, 210, 90)
            else:
                texto = f"Desconocido ({confianza * 100:.1f}%)"
                color = (40, 80, 230)

            # Renderizar caja delimitadora y banner superior con la predicción
            cv2.rectangle(fotograma, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(fotograma, (x, y - 32), (x + w, y), color, -1)
            cv2.putText(
                fotograma,
                texto,
                (x + 6, y - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Reconocimiento Facial CNN - En Vivo", fotograma)
        # Finalizar ejecución al presionar la tecla ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    iniciar_reconocimiento()