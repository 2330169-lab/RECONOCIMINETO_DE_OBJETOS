import cv2
import numpy as np
import tensorflow as tf

# ============================================================
# CONFIGURACION
# ============================================================

MODELO = "mejor_detector.keras"
ARCHIVO_CLASES = "clases.txt"

IMG_SIZE = (224, 224)

# Confianza minima para mostrar deteccion
CONFIANZA_MINIMA = 0.80


# ============================================================
# CARGAR MODELO
# ============================================================

print("Cargando modelo...")

# compile=False porque para reconocer no necesitamos
# detector_loss, class_accuracy ni bbox_iou
model = tf.keras.models.load_model(
    MODELO,
    compile=False
)

print("Modelo cargado correctamente.")


# ============================================================
# CARGAR CLASES
# ============================================================

with open(
    ARCHIVO_CLASES,
    "r",
    encoding="utf-8"
) as archivo:

    class_names = [
        linea.strip()
        for linea in archivo.readlines()
        if linea.strip()
    ]


print("\nClases cargadas:")

for i, clase in enumerate(class_names):
    print(f"{i} -> {clase}")


# ============================================================
# ABRIR CAMARA
# ============================================================

camara = cv2.VideoCapture(0)

if not camara.isOpened():

    print("ERROR: No se pudo abrir la camara.")
    exit()


print("\nCamara iniciada.")
print("Presiona Q para salir.\n")


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

while True:

    ret, frame = camara.read()

    if not ret:

        print(
            "No se pudo obtener imagen de la camara."
        )

        break


    # ========================================================
    # TAMAÑO ORIGINAL DEL FRAME
    # ========================================================

    alto_frame, ancho_frame = frame.shape[:2]


    # ========================================================
    # PREPARAR IMAGEN
    # ========================================================

    # OpenCV trabaja en BGR
    # TensorFlow fue entrenado con RGB
    imagen_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Cambiar a 224x224
    imagen = cv2.resize(
        imagen_rgb,
        IMG_SIZE
    )


    # Convertir a float
    imagen = imagen.astype(
        np.float32
    )


    # Normalizar 0-255 -> 0-1
    imagen = imagen / 255.0


    # Agregar dimension batch
    #
    # (224,224,3)
    #
    # pasa a:
    #
    # (1,224,224,3)

    imagen = np.expand_dims(
        imagen,
        axis=0
    )


    # ========================================================
    # HACER PREDICCION
    # ========================================================

    prediccion = model.predict(
        imagen,
        verbose=0
    )[0]


    # ========================================================
    # COORDENADAS
    # ========================================================

    # Primeros 4 valores:
    #
    # x
    # y
    # w
    # h
    #
    # Son logits, por eso aplicamos sigmoid.

    bbox_logits = prediccion[:4]

    bbox = tf.sigmoid(
        bbox_logits
    ).numpy()


    x_centro = float(bbox[0])
    y_centro = float(bbox[1])

    ancho_bbox = float(bbox[2])
    alto_bbox = float(bbox[3])


    # ========================================================
    # CLASIFICACION
    # ========================================================

    # El resto de valores son las clases
    class_logits = prediccion[4:]


    # Convertir logits a probabilidades
    probabilidades = tf.nn.softmax(
        class_logits
    ).numpy()


    # Clase con mayor probabilidad
    indice = int(
        np.argmax(
            probabilidades
        )
    )


    confianza = float(
        probabilidades[indice]
    )


    clase = class_names[
        indice
    ]


    # ========================================================
    # CONVERTIR BBOX A PIXELES
    # ========================================================

    # Formato del modelo:
    #
    # x centro
    # y centro
    # ancho
    # alto
    #
    # valores entre 0 y 1


    centro_x_px = int(
        x_centro * ancho_frame
    )

    centro_y_px = int(
        y_centro * alto_frame
    )


    ancho_px = int(
        ancho_bbox * ancho_frame
    )

    alto_px = int(
        alto_bbox * alto_frame
    )


    # ========================================================
    # CALCULAR ESQUINAS
    # ========================================================

    x1 = int(
        centro_x_px -
        ancho_px / 2
    )

    y1 = int(
        centro_y_px -
        alto_px / 2
    )


    x2 = int(
        centro_x_px +
        ancho_px / 2
    )

    y2 = int(
        centro_y_px +
        alto_px / 2
    )


    # ========================================================
    # EVITAR QUE EL CUADRO SALGA DE LA IMAGEN
    # ========================================================

    x1 = max(
        0,
        min(
            x1,
            ancho_frame - 1
        )
    )

    y1 = max(
        0,
        min(
            y1,
            alto_frame - 1
        )
    )

    x2 = max(
        0,
        min(
            x2,
            ancho_frame - 1
        )
    )

    y2 = max(
        0,
        min(
            y2,
            alto_frame - 1
        )
    )


    # ========================================================
    # MOSTRAR DETECCION
    # ========================================================

    if confianza >= CONFIANZA_MINIMA:

        # ----------------------------------------------------
        # DIBUJAR RECUADRO
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        # ----------------------------------------------------
        # TEXTO DE ETIQUETA
        # ----------------------------------------------------

        etiqueta = (
            f"{clase} "
            f"{confianza * 100:.1f}%"
        )


        # Obtener tamaño del texto
        (
            ancho_texto,
            alto_texto
        ), baseline = cv2.getTextSize(

            etiqueta,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            2
        )


        # ====================================================
        # POSICION DEL TEXTO
        # ====================================================

        texto_y = y1 - 10


        # Si no cabe arriba
        if texto_y - alto_texto < 0:

            texto_y = (
                y1 +
                alto_texto +
                15
            )


        # ====================================================
        # FONDO DE LA ETIQUETA
        # ====================================================

        cv2.rectangle(

            frame,

            (
                x1,
                texto_y
                - alto_texto
                - 10
            ),

            (
                x1
                + ancho_texto
                + 10,

                texto_y
                + baseline
            ),

            (0, 0, 0),

            -1
        )


        # ====================================================
        # ESCRIBIR ETIQUETA
        # ====================================================

        cv2.putText(

            frame,

            etiqueta,

            (
                x1 + 5,
                texto_y - 5
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        estado = (
            f"Detectado: {clase}"
        )


    else:

        estado = (
            "Objeto no reconocido"
        )


    # ========================================================
    # MOSTRAR ESTADO GENERAL
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (ancho_frame, 55),
        (0, 0, 0),
        -1
    )


    cv2.putText(

        frame,

        estado,

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 0),

        2
    )


    # ========================================================
    # MOSTRAR PROBABILIDADES
    # ========================================================

    y_texto = 85


    for i, nombre in enumerate(
        class_names
    ):

        porcentaje = (
            probabilidades[i]
            * 100
        )


        texto = (
            f"{nombre}: "
            f"{porcentaje:.1f}%"
        )


        cv2.putText(

            frame,

            texto,

            (20, y_texto),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2
        )


        y_texto += 25


    # ========================================================
    # DEBUG DE BOUNDING BOX
    # ========================================================

    texto_bbox = (

        f"x:{x_centro:.2f} "
        f"y:{y_centro:.2f} "
        f"w:{ancho_bbox:.2f} "
        f"h:{alto_bbox:.2f}"
    )


    cv2.putText(

        frame,

        texto_bbox,

        (
            20,
            alto_frame - 20
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 0),

        1
    )


    # ========================================================
    # MOSTRAR CAMARA
    # ========================================================

    cv2.imshow(
        "Detector de objetos",
        frame
    )


    # ========================================================
    # SALIR
    # ========================================================

    tecla = cv2.waitKey(1) & 0xFF


    if tecla == ord("q"):
        break


# ============================================================
# CERRAR
# ============================================================

camara.release()

cv2.destroyAllWindows()

print("\nPrograma finalizado.")