import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ============================================================
# CONFIGURACION
# ============================================================

MODELO = "mejor_detector_objectness.keras"
ARCHIVO_CLASES = "clases.txt"

IMG_SIZE = (224, 224)

# ------------------------------------------------------------
# UMBRALES
# ------------------------------------------------------------

# Que tan seguro debe estar de que realmente existe
# uno de nuestros objetos
OBJECTNESS_MINIMA = 0.80

# Que tan seguro debe estar de que clase es
CONFIANZA_CLASE_MINIMA = 0.60

# Frames utilizados para suavizar la deteccion
HISTORIAL = 5


# ============================================================
# CARGAR MODELO
# ============================================================

print("\n============================================")
print("CARGANDO MODELO")
print("============================================")

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


print("\nClases:")

for i, clase in enumerate(class_names):
    print(f"{i} -> {clase}")


# ============================================================
# COMPROBAR SALIDA DEL MODELO
# ============================================================

numero_salidas = model.output_shape[-1]

salidas_esperadas = (
    5 + len(class_names)
)


print("\nSalidas del modelo:", numero_salidas)
print("Salidas esperadas:", salidas_esperadas)


if numero_salidas != salidas_esperadas:

    print("\nERROR:")

    print(
        "El numero de salidas del modelo "
        "no coincide con reconocer.py."
    )

    print("\nFormato esperado:")

    print(
        "[x, y, w, h, objectness, "
        + ", ".join(class_names)
        + "]"
    )

    exit()


print("\nFormato correcto:")

print(
    "[x, y, w, h, objectness, "
    + ", ".join(class_names)
    + "]"
)


# ============================================================
# HISTORIALES PARA SUAVIZADO
# ============================================================

historial_objectness = deque(
    maxlen=HISTORIAL
)

historial_probabilidades = deque(
    maxlen=HISTORIAL
)

historial_bbox = deque(
    maxlen=HISTORIAL
)


# ============================================================
# ABRIR CAMARA
# ============================================================

camara = cv2.VideoCapture(0)


if not camara.isOpened():

    print(
        "ERROR: No se pudo abrir la camara."
    )

    exit()


# Intentar resolución 1280x720
camara.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camara.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


print("\n============================================")
print("CAMARA INICIADA")
print("============================================")

print("\nPresiona Q para salir.")


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

while True:

    ret, frame = camara.read()


    if not ret:

        print(
            "No se pudo obtener imagen."
        )

        break


    # ========================================================
    # DIMENSIONES DEL FRAME
    # ========================================================

    alto_frame, ancho_frame = frame.shape[:2]


    # ========================================================
    # PREPARAR IMAGEN PARA LA RED
    # ========================================================

    # BGR -> RGB
    imagen_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Resize
    imagen = cv2.resize(
        imagen_rgb,
        IMG_SIZE
    )


    # Float32
    imagen = imagen.astype(
        np.float32
    )


    # Normalizacion
    imagen = imagen / 255.0


    # Agregar batch
    imagen = np.expand_dims(
        imagen,
        axis=0
    )


    # ========================================================
    # PREDICCION
    # ========================================================

    prediccion = model.predict(
        imagen,
        verbose=0
    )[0]


    # ========================================================
    # BOUNDING BOX
    # ========================================================
    #
    # 0 = x centro
    # 1 = y centro
    # 2 = ancho
    # 3 = alto
    #
    # Durante entrenamiento estas salidas son logits,
    # por eso utilizamos sigmoid.
    # ========================================================

    bbox_logits = prediccion[
        0:4
    ]


    bbox = tf.sigmoid(
        bbox_logits
    ).numpy()


    # ========================================================
    # OBJECTNESS
    # ========================================================

    objectness_logit = prediccion[4]


    objectness = float(
        tf.sigmoid(
            objectness_logit
        ).numpy()
    )


    # ========================================================
    # CLASIFICACION
    # ========================================================

    class_logits = prediccion[
        5:
    ]


    probabilidades = tf.nn.softmax(
        class_logits
    ).numpy()


    # ========================================================
    # GUARDAR RESULTADOS EN HISTORIAL
    # ========================================================

    historial_objectness.append(
        objectness
    )


    historial_probabilidades.append(
        probabilidades
    )


    historial_bbox.append(
        bbox
    )


    # ========================================================
    # PROMEDIAR ULTIMOS FRAMES
    # ========================================================

    objectness_promedio = float(
        np.mean(
            historial_objectness
        )
    )


    probabilidades_promedio = np.mean(
        historial_probabilidades,
        axis=0
    )


    bbox_promedio = np.mean(
        historial_bbox,
        axis=0
    )


    # ========================================================
    # OBTENER CLASE GANADORA
    # ========================================================

    indice = int(
        np.argmax(
            probabilidades_promedio
        )
    )


    clase = class_names[
        indice
    ]


    confianza = float(
        probabilidades_promedio[
            indice
        ]
    )


    # ========================================================
    # BBOX PROMEDIO
    # ========================================================

    x_centro = float(
        bbox_promedio[0]
    )

    y_centro = float(
        bbox_promedio[1]
    )

    ancho_bbox = float(
        bbox_promedio[2]
    )

    alto_bbox = float(
        bbox_promedio[3]
    )


    # ========================================================
    # CONVERTIR COORDENADAS NORMALIZADAS A PIXELES
    # ========================================================

    centro_x_px = int(
        x_centro
        *
        ancho_frame
    )


    centro_y_px = int(
        y_centro
        *
        alto_frame
    )


    ancho_px = int(
        ancho_bbox
        *
        ancho_frame
    )


    alto_px = int(
        alto_bbox
        *
        alto_frame
    )


    # ========================================================
    # ESQUINAS DEL RECTANGULO
    # ========================================================

    x1 = int(
        centro_x_px
        -
        ancho_px / 2
    )


    y1 = int(
        centro_y_px
        -
        alto_px / 2
    )


    x2 = int(
        centro_x_px
        +
        ancho_px / 2
    )


    y2 = int(
        centro_y_px
        +
        alto_px / 2
    )


    # ========================================================
    # LIMITAR RECTANGULO AL FRAME
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
    # DECISION FINAL
    # ========================================================
    #
    # Para dibujar algo se deben cumplir DOS condiciones:
    #
    # 1. La red cree que realmente hay un objeto conocido.
    #
    # 2. La red tiene suficiente confianza en la clase.
    #
    # ========================================================

    objeto_detectado = (

        objectness_promedio
        >=
        OBJECTNESS_MINIMA

        and

        confianza
        >=
        CONFIANZA_CLASE_MINIMA
    )


    # ========================================================
    # DIBUJAR DETECCION
    # ========================================================

    if objeto_detectado:

        # ----------------------------------------------------
        # RECTANGULO
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        # ----------------------------------------------------
        # ETIQUETA
        # ----------------------------------------------------

        etiqueta = (
            f"{clase} "
            f"{confianza * 100:.1f}%"
        )


        (
            ancho_texto,
            alto_texto
        ), baseline = cv2.getTextSize(

            etiqueta,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            2
        )


        # ----------------------------------------------------
        # POSICION TEXTO
        # ----------------------------------------------------

        texto_y = y1 - 10


        if (
            texto_y
            -
            alto_texto
            <
            0
        ):

            texto_y = (
                y1
                +
                alto_texto
                +
                15
            )


        # ----------------------------------------------------
        # FONDO ETIQUETA
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (
                x1,
                texto_y
                -
                alto_texto
                -
                10
            ),

            (
                x1
                +
                ancho_texto
                +
                10,

                texto_y
                +
                baseline
            ),

            (0, 0, 0),

            -1
        )


        # ----------------------------------------------------
        # TEXTO
        # ----------------------------------------------------

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
            f"DETECTADO: {clase}"
        )


    else:

        estado = (
            "NINGUN OBJETO CONOCIDO"
        )


    # ========================================================
    # PANEL SUPERIOR
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (ancho_frame, 200),
        (0, 0, 0),
        -1
    )


    # ========================================================
    # ESTADO
    # ========================================================

    color_estado = (
        (0, 255, 0)
        if objeto_detectado
        else
        (0, 0, 255)
    )


    cv2.putText(

        frame,

        estado,

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        color_estado,

        2
    )


    # ========================================================
    # OBJECTNESS
    # ========================================================

    texto_objectness = (
        f"Objectness: "
        f"{objectness_promedio * 100:.1f}% "
        f"(min {OBJECTNESS_MINIMA * 100:.0f}%)"
    )


    cv2.putText(

        frame,

        texto_objectness,

        (20, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (0, 255, 255),

        2
    )


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    y_texto = 105


    for i, nombre in enumerate(
        class_names
    ):

        porcentaje = (
            probabilidades_promedio[i]
            *
            100
        )


        texto = (
            f"{nombre}: "
            f"{porcentaje:.1f}%"
        )


        # Resaltar clase ganadora
        if i == indice:

            color = (
                0,
                255,
                0
            )

        else:

            color = (
                255,
                255,
                255
            )


        cv2.putText(

            frame,

            texto,

            (
                20,
                y_texto
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            color,

            2
        )


        y_texto += 25


    # ========================================================
    # MOSTRAR CONFIANZA MINIMA
    # ========================================================

    cv2.putText(

        frame,

        (
            f"Confianza clase minima: "
            f"{CONFIANZA_CLASE_MINIMA * 100:.0f}%"
        ),

        (
            300,
            70
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        1
    )


    # ========================================================
    # MOSTRAR BBOX PARA DEBUG
    # ========================================================

    texto_bbox = (

        f"BBOX "
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
    # MOSTRAR FRAME
    # ========================================================

    cv2.imshow(
        "Detector de objetos - Objectness",
        frame
    )


    # ========================================================
    # SALIR
    # ========================================================

    tecla = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if tecla == ord("q"):
        break


# ============================================================
# CERRAR
# ============================================================

camara.release()

cv2.destroyAllWindows()


print("\n============================================")
print("PROGRAMA FINALIZADO")
print("============================================")