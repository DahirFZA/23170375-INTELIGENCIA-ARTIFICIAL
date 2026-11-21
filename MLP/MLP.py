#Zavala Arredondo Dahir Fernando
#23170375
import tensorflow as tf
from tensorflow.keras import layers, models


def entrenar_mlp():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    modelo = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Entrenando modelo MLP con MNIST...")
    modelo.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    perdida, precision = modelo.evaluate(x_test, y_test, verbose=0)
    print(f"\nPrecisión en datos de prueba: {precision * 100:.2f}%")

    modelo.save("MLP/modelo_mnist.keras")
    print("Modelo guardado exitosamente en: MLP/modelo_mnist.keras")


if __name__ == "__main__":
    entrenar_mlp()