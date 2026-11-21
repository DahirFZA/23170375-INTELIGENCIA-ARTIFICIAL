import json
import os
import matplotlib

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
EPOCAS = 25


def construir_red(num_clases):
    """Construye un clasificador usando MobileNetV2 preentrenada como extractor de rasgos."""
    aumento_datos = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    base_modelo = tf.keras.applications.MobileNetV2(
        input_shape=(TAM_IMG[0], TAM_IMG[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_modelo.trainable = False

    entradas = layers.Input(shape=(TAM_IMG[0], TAM_IMG[1], 3))
    x = aumento_datos(entradas)
    # Normalizar píxeles al rango [-1, 1] requerido por MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_modelo(x, training=False)

    # Bloque de regularización y salida multiclase
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    salidas = layers.Dense(num_clases, activation="softmax")(x)

    modelo = models.Model(inputs=entradas, outputs=salidas)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def entrenar():
    """Carga el dataset dividido en 80% entrenamiento y 20% validación, entrena y guarda curvas."""
    ds_entrenamiento = tf.keras.utils.image_dataset_from_directory(
        RUTA_DATASET,
        validation_split=0.20,
        subset="training",
        seed=42,
        image_size=TAM_IMG,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    ds_validacion = tf.keras.utils.image_dataset_from_directory(
        RUTA_DATASET,
        validation_split=0.20,
        subset="validation",
        seed=42,
        image_size=TAM_IMG,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    nombres_clases = ds_entrenamiento.class_names
    with open(RUTA_CLASES, "w", encoding="utf-8") as f:
        json.dump(nombres_clases, f, ensure_ascii=False, indent=2)

    autotune = tf.data.AUTOTUNE
    ds_entrenamiento = ds_entrenamiento.cache().shuffle(500).prefetch(buffer_size=autotune)
    ds_validacion = ds_validacion.cache().prefetch(buffer_size=autotune)

    modelo = construir_red(len(nombres_clases))

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
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    historial = modelo.fit(
        ds_entrenamiento,
        validation_data=ds_validacion,
        epochs=EPOCAS,
        callbacks=lista_callbacks,
    )

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
