#Zavala Arredondo Dahir Fernando
#23170375
import os
import cv2


def asegurar_directorio(ruta):
    """Verifica la existencia del directorio destino y lo crea si no existe."""
    if not os.path.exists(ruta):
        os.makedirs(ruta)


def capturar_desde_camara(nombre_clase, carpeta_base="ReconocimientoFacialCNN/dataset", limite=80):
    """Captura muestras faciales desde la webcam y las guarda redimensionadas a 160x160."""
    destino = os.path.join(carpeta_base, nombre_clase)
    asegurar_directorio(destino)

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    camara = cv2.VideoCapture(0)
    contador = len(os.listdir(destino))

    while contador < limite:
        ok, fotograma = camara.read()
        if not ok:
            break

        # Convertir a escala de grises y ecualizar histograma para compensar cambios de luz
        gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(gris)
        rostros = detector.detectMultiScale(gris, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in rostros:
            # Añadir un margen del 12% alrededor del rostro para conservar contexto facial
            margen = int(0.12 * w)
            x1 = max(0, x - margen)
            y1 = max(0, y - margen)
            x2 = min(fotograma.shape[1], x + w + margen)
            y2 = min(fotograma.shape[0], y + h + margen)

            recorte = cv2.resize(fotograma[y1:y2, x1:x2], (160, 160), interpolation=cv2.INTER_AREA)
            ruta_img = os.path.join(destino, f"{nombre_clase}_{contador:03d}.jpg")
            cv2.imwrite(ruta_img, recorte)
            contador += 1

            # Dibujar guía visual del progreso de captura en pantalla
            cv2.rectangle(fotograma, (x1, y1), (x2, y2), (0, 220, 120), 2)
            cv2.putText(
                fotograma,
                f"{nombre_clase}: {contador}/{limite}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 120),
                2,
            )

        cv2.imshow("Recolector de Dataset Facial", fotograma)
        if cv2.waitKey(60) & 0xFF == 27 or contador >= limite:
            break

    camara.release()
    cv2.destroyAllWindows()


def extraer_desde_carpeta_fotos(carpeta_origen, nombre_clase, carpeta_base="ReconocimientoFacialCNN/dataset"):
    """Detecta y recorta rostros a partir de fotografías previamente descargadas."""
    destino = os.path.join(carpeta_base, nombre_clase)
    asegurar_directorio(destino)

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    indice = len(os.listdir(destino))

    for archivo in os.listdir(carpeta_origen):
        ruta = os.path.join(carpeta_origen, archivo)
        imagen = cv2.imread(ruta)
        if imagen is None:
            continue

        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        rostros = detector.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        for (x, y, w, h) in rostros:
            recorte = cv2.resize(imagen[y : y + h, x : x + w], (160, 160), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(destino, f"{nombre_clase}_{indice:03d}.jpg"), recorte)
            indice += 1


if __name__ == "__main__":
    print("1. Capturar desde camara web")
    print("2. Extraer rostros desde carpeta de fotos (famosos)")
    opcion = input("Selecciona opcion (1/2): ").strip()
    etiqueta = input("Nombre de la persona (sin espacios): ").strip()

    if opcion == "1":
        cantidad = int(input("Cantidad de muestras a capturar: ").strip())
        capturar_desde_camara(etiqueta, limite=cantidad)
    elif opcion == "2":
        origen = input("Ruta de carpeta con fotos originales: ").strip()
        extraer_desde_carpeta_fotos(origen, etiqueta)