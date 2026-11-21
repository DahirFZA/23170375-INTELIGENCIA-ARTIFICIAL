#Zavala Arredondo Dahir Fernando
#23170375
import json
import os
import matplotlib

# Configurar backend sin ventana gráfica si se ejecuta en un entorno remoto sin pantalla
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


RUTA_DATASET = "ReconocimientoFacialCNN/dataset"
RUTA_MODELO = "ReconocimientoFacialCNN/mejor_modelo_facial.keras"
RUTA_CLASES = "ReconocimientoFacialCNN/etiquetas.json"
RUTA_GRAFICA = "ReconocimientoFacialCNN/metricas_entrenamiento.png"
TAM_IMG = (160, 160)
BATCH_SIZE = 16
EPOCAS = 35


def construir_red(num_clases):
    """Define la arquitectura CNN con preprocesamiento, bloques convolucionales y regularización."""
    # Transformaciones aleatorias aplicadas en tiempo de entrenamiento para evitar sobreajuste
    aumento_datos = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ])

    entradas = layers.Input(shape=(TAM_IMG[0], TAM_IMG[1], 3))
    # Normalizar valores de píxeles del rango [0, 255] a [0, 1]
    x = layers.Rescaling(1.0 / 255.0)(entradas)
    x = aumento_datos(x)

    # Bloques de extracción de características con profundidad creciente
    for filtros in (32, 64, 128, 256):
        x = layers.Conv2D(filtros, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filtros, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

    # Clasificador denso con reducción espacial global y salida multiclase
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(192, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.45)(x)
    salidas = layers.Dense(num_clases, activation="softmax")(x)

    modelo = models.Model(inputs=entradas, outputs=salidas)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def entrenar():
    """Carga el dataset dividido en 85% entrenamiento y 15% validación, entrena y guarda curvas."""
    ds_entrenamiento = tf.keras.utils.image_dataset_from_directory(
        RUTA_DATASET,
        validation_split=0.15,
        subset="training",
        seed=73,
        image_size=TAM_IMG,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    ds_validacion = tf.keras.utils.image_dataset_from_directory(
        RUTA_DATASET,
        validation_split=0.15,
        subset="validation",
        seed=73,
        image_size=TAM_IMG,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    # Exportar el orden exacto de las clases para mapear predicciones durante la inferencia
    nombres_clases = ds_entrenamiento.class_names
    with open(RUTA_CLASES, "w", encoding="utf-8") as f:
        json.dump(nombres_clases, f, ensure_ascii=False, indent=2)

    # Optimizar lectura de lotes en memoria y precarga asíncrona
    autotune = tf.data.AUTOTUNE
    ds_entrenamiento = ds_entrenamiento.cache().shuffle(500).prefetch(buffer_size=autotune)
    ds_validacion = ds_validacion.cache().prefetch(buffer_size=autotune)

    modelo = construir_red(len(nombres_clases))

    # Callbacks para guardar los mejores pesos, frenar estancamiento y ajustar tasa de aprendizaje
    lista_callbacks = [
        callbacks.ModelCheckpoint(
            RUTA_MODELO,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    historial = modelo.fit(
        ds_entrenamiento,
        validation_data=ds_validacion,
        epochs=EPOCAS,
        callbacks=lista_callbacks,
    )

    # Generar gráficas comparativas de precisión y pérdida
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(historial.history["accuracy"], label="Entrenamiento")
    ax1.plot(historial.history["val_accuracy"], label="Validación")
    ax1.set_title("Precisión (Accuracy)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(historial.history["loss"], label="Entrenamiento")
    ax2.plot(historial.history["val_loss"], label="Validación")
    ax2.set_title("Pérdida (Categorical Crossentropy)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RUTA_GRAFICA, dpi=150)
    plt.show()


if __name__ == "__main__":
    entrenar()