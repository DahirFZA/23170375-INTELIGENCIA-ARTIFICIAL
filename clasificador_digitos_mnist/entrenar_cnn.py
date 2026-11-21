#Zavala Arredondo Dahir Fernando
#23170375
import tensorflow as tf
from tensorflow.keras import layers, models

def construir_y_entrenar():
    # 1. Cargar el dataset MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 2. Preprocesamiento: normalizar valores [0, 1] y agregar dimensión de canal (28x28x1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # 3. Arquitectura Convolucional (CNN)
    modelo = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Entrenamiento
    print("Iniciando entrenamiento del modelo...")
    modelo.fit(
        x_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1
    )

    # 5. Evaluación y guardado
    perdida, precision = modelo.evaluate(x_test, y_test, verbose=0)
    print(f"\nPrecisión en datos de prueba: {precision * 100:.2f}%")

    archivo_salida = "red_digitos_cnn.keras"
    modelo.save(archivo_salida)
    print(f"Modelo guardado exitosamente como: {archivo_salida}")

if __name__ == "__main__":
    construir_y_entrenar()