#crear una red neuronal para reconocer distintos objetos, los cuales para este caso seran
#una bolsa de mujer, peluches varios, y perfumes varios.

#red neuronal convolucional
#aprendizaje automatico
#clasificacion de imagenes



import tensorflow as tf
import numpy as np
import cv2
import random

from pathlib import Path


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

DATASET_DIR = Path("dataset")

CARPETA_NEGATIVOS = "no_objeto"

# Estas son las UNICAS clases reales.
# no_objeto NO es una clase.
CLASS_NAMES = [
    "Bolsas",
    "Peluches",
    "Perfume"
]

NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------------------------------------
# SALIDA DE LA RED
# ------------------------------------------------------------
#
# x
# y
# w
# h
# objectness
# Bolsas
# Peluches
# Perfume
#
# 4 + 1 + 3 = 8
# ------------------------------------------------------------

NUM_SALIDAS = 5 + NUM_CLASSES


# ============================================================
# CONFIGURACION DE ENTRENAMIENTO
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS = 150

VALIDATION_SPLIT = 0.20

SEED = 123


# ============================================================
# LEARNING RATE
# ============================================================

LEARNING_RATE = 0.00005


# ============================================================
# PESOS DE LAS PERDIDAS
# ============================================================

# Importancia de saber si realmente hay un objeto
OBJECTNESS_LOSS_WEIGHT = 1.5

# Importancia de clasificar correctamente
CLASS_LOSS_WEIGHT = 1.0

# Importancia del bounding box
BBOX_LOSS_WEIGHT = 2.0


# ============================================================
# PESO EXTRA PARA OBJETOS POSITIVOS
# ============================================================
#
# Esto ayuda a evitar que, por tener muchas imagenes
# no_objeto, la red se vuelva demasiado conservadora.
#
# 1.5 significa que equivocarse diciendo "no hay objeto"
# cuando SI existe uno tiene un poco mas de importancia.
# ============================================================

POSITIVE_OBJECTNESS_WEIGHT = 1.0


# ============================================================
# CONTROL DE NEGATIVOS
# ============================================================
#
# Evitamos que no_objeto tenga muchas mas imagenes
# que todas las clases positivas juntas.
#
# 1.0 = como maximo tantos negativos como positivos.
# ============================================================

MAX_NEGATIVE_RATIO = 1.5


# ============================================================
# ARCHIVOS GENERADOS
# ============================================================

MEJOR_MODELO = "mejor_detector_objectness.keras"

MODELO_FINAL = "modelo_detector_objectness_final.keras"

HISTORIAL_CSV = "historial_entrenamiento_objectness.csv"


# ============================================================
# EXTENSIONES
# ============================================================

EXTENSIONES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP"
}


# ============================================================
# EMPEZAR COMPLETAMENTE DESDE CERO
# ============================================================

# Limpia cualquier modelo que pudiera existir en memoria.
# NO carga ningun archivo .keras.

tf.keras.backend.clear_session()

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


print("\n============================================")
print("ENTRENAMIENTO DESDE CERO")
print("============================================")

print("\nNO se cargara ningun modelo anterior.")
print("La red sera creada con pesos nuevos.")


# ============================================================
# CREAR clases.txt CORRECTAMENTE
# ============================================================

with open(
    "clases.txt",
    "w",
    encoding="utf-8"
) as archivo:

    for clase in CLASS_NAMES:
        archivo.write(clase + "\n")


print("\nclases.txt creado/corregido.")

print("\nClases:")

for i, clase in enumerate(CLASS_NAMES):
    print(f"{i} -> {clase}")

print("\nIMPORTANTE:")
print(
    f"'{CARPETA_NEGATIVOS}' NO es una clase."
)


# ============================================================
# COMPROBAR DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"No existe la carpeta {DATASET_DIR}"
    )


# ============================================================
# LEER ETIQUETA YOLO
# ============================================================

def leer_etiqueta(ruta_txt):

    """
    Formato esperado:

    clase x_centro y_centro ancho alto

    Ejemplo:

    2 0.500000 0.480000 0.250000 0.600000
    """

    with open(
        ruta_txt,
        "r",
        encoding="utf-8"
    ) as archivo:

        lineas = [
            linea.strip()
            for linea in archivo.readlines()
            if linea.strip()
        ]


    if len(lineas) == 0:

        raise ValueError(
            f"Etiqueta vacia: {ruta_txt}"
        )


    # Este detector esta diseñado para
    # un objeto principal por imagen.

    if len(lineas) > 1:

        raise ValueError(
            f"{ruta_txt} contiene mas de un objeto."
        )


    datos = lineas[0].split()


    if len(datos) != 5:

        raise ValueError(
            f"Formato incorrecto: {ruta_txt}"
        )


    clase_id = int(datos[0])

    x = float(datos[1])
    y = float(datos[2])
    w = float(datos[3])
    h = float(datos[4])


    # --------------------------------------------------------
    # VALIDAR CLASE
    # --------------------------------------------------------

    if clase_id < 0 or clase_id >= NUM_CLASSES:

        raise ValueError(
            f"Clase invalida {clase_id} en {ruta_txt}"
        )


    # --------------------------------------------------------
    # VALIDAR BOUNDING BOX
    # --------------------------------------------------------

    for valor in [x, y, w, h]:

        if valor < 0.0 or valor > 1.0:

            raise ValueError(
                f"Coordenada fuera de rango en {ruta_txt}"
            )


    if w <= 0 or h <= 0:

        raise ValueError(
            f"Bounding box invalido en {ruta_txt}"
        )


    # --------------------------------------------------------
    # ONE-HOT DE CLASE
    # --------------------------------------------------------

    clase_one_hot = np.zeros(
        NUM_CLASSES,
        dtype=np.float32
    )

    clase_one_hot[clase_id] = 1.0


    # --------------------------------------------------------
    # ETIQUETA FINAL
    # --------------------------------------------------------
    #
    # [x, y, w, h,
    #  objectness,
    #  clase0, clase1, clase2]
    #
    # Para una imagen positiva:
    #
    # objectness = 1
    # --------------------------------------------------------

    etiqueta = np.concatenate(
        [
            np.array(
                [
                    x,
                    y,
                    w,
                    h,
                    1.0
                ],
                dtype=np.float32
            ),

            clase_one_hot
        ]
    )


    return etiqueta, clase_id


# ============================================================
# BUSCAR IMAGENES POSITIVAS
# ============================================================

positivos_por_clase = {
    i: []
    for i in range(NUM_CLASSES)
}


errores = 0


print("\n============================================")
print("BUSCANDO OBJETOS ETIQUETADOS")
print("============================================")


for clase_id, nombre_clase in enumerate(
    CLASS_NAMES
):

    carpeta = (
        DATASET_DIR /
        nombre_clase
    )


    if not carpeta.exists():

        raise FileNotFoundError(
            f"No existe: {carpeta}"
        )


    for ruta_imagen in carpeta.iterdir():

        if ruta_imagen.suffix not in EXTENSIONES:
            continue


        ruta_txt = ruta_imagen.with_suffix(
            ".txt"
        )


        # ----------------------------------------------------
        # NECESITA TXT
        # ----------------------------------------------------

        if not ruta_txt.exists():

            print(
                f"Sin etiqueta: {ruta_imagen}"
            )

            continue


        try:

            etiqueta, clase_txt = leer_etiqueta(
                ruta_txt
            )


            if clase_txt != clase_id:

                print(
                    f"ADVERTENCIA: {ruta_imagen.name}"
                )

                print(
                    f"Carpeta: {nombre_clase}"
                )

                print(
                    f"TXT: {CLASS_NAMES[clase_txt]}"
                )


            positivos_por_clase[
                clase_txt
            ].append(
                (
                    str(ruta_imagen),
                    etiqueta
                )
            )


        except Exception as e:

            print(
                f"ERROR: {ruta_imagen.name}"
            )

            print(e)

            errores += 1


# ============================================================
# MOSTRAR POSITIVOS
# ============================================================

print("\n============================================")
print("IMAGENES POSITIVAS")
print("============================================")


total_positivos = 0


for clase_id, nombre in enumerate(
    CLASS_NAMES
):

    cantidad = len(
        positivos_por_clase[
            clase_id
        ]
    )

    total_positivos += cantidad

    print(
        f"{nombre}: {cantidad}"
    )


print(
    f"\nTotal positivos: {total_positivos}"
)


if total_positivos == 0:

    raise RuntimeError(
        "No se encontraron imagenes positivas."
    )


# ============================================================
# BUSCAR IMAGENES NEGATIVAS
# ============================================================

carpeta_negativos = (
    DATASET_DIR /
    CARPETA_NEGATIVOS
)


if not carpeta_negativos.exists():

    raise FileNotFoundError(
        f"No existe la carpeta "
        f"{carpeta_negativos}"
    )


negativos = []


print("\n============================================")
print("BUSCANDO IMAGENES NO_OBJETO")
print("============================================")


for ruta_imagen in carpeta_negativos.iterdir():

    if ruta_imagen.suffix not in EXTENSIONES:
        continue


    # --------------------------------------------------------
    # ETIQUETA NEGATIVA
    # --------------------------------------------------------
    #
    # bbox = 0,0,0,0
    # objectness = 0
    # clases = 0,0,0
    #
    # La funcion loss ignorara bbox y clases
    # cuando objectness sea 0.
    # --------------------------------------------------------

    etiqueta = np.zeros(
        NUM_SALIDAS,
        dtype=np.float32
    )


    negativos.append(
        (
            str(ruta_imagen),
            etiqueta
        )
    )


print(
    f"Imagenes no_objeto encontradas: "
    f"{len(negativos)}"
)


if len(negativos) == 0:

    raise RuntimeError(
        "La carpeta no_objeto esta vacia."
    )


# ============================================================
# LIMITAR NEGATIVOS SI HAY DEMASIADOS
# ============================================================

max_negativos = int(
    total_positivos
    *
    MAX_NEGATIVE_RATIO
)


if len(negativos) > max_negativos:

    print(
        "\nHay mas negativos de los necesarios."
    )

    print(
        f"Se utilizaran {max_negativos} "
        f"de {len(negativos)}."
    )

    random.Random(
        SEED
    ).shuffle(
        negativos
    )

    negativos = negativos[
        :max_negativos
    ]


print(
    f"Negativos utilizados: {len(negativos)}"
)


# ============================================================
# DIVIDIR TRAIN / VALIDATION
# ============================================================

train_samples = []

val_samples = []


# ============================================================
# DIVIDIR POSITIVOS POR CLASE
# ============================================================

for clase_id in range(
    NUM_CLASSES
):

    muestras = positivos_por_clase[
        clase_id
    ].copy()


    random.Random(
        SEED + clase_id
    ).shuffle(
        muestras
    )


    cantidad = len(
        muestras
    )


    if cantidad < 2:

        raise RuntimeError(
            f"La clase "
            f"{CLASS_NAMES[clase_id]} "
            f"necesita mas imagenes."
        )


    cantidad_val = max(
        1,
        int(
            round(
                cantidad
                *
                VALIDATION_SPLIT
            )
        )
    )


    cantidad_val = min(
        cantidad_val,
        cantidad - 1
    )


    val_samples.extend(
        muestras[
            :cantidad_val
        ]
    )


    train_samples.extend(
        muestras[
            cantidad_val:
        ]
    )


# ============================================================
# DIVIDIR NEGATIVOS
# ============================================================

random.Random(
    SEED + 100
).shuffle(
    negativos
)


cantidad_negativos = len(
    negativos
)


cantidad_val_neg = max(
    1,
    int(
        round(
            cantidad_negativos
            *
            VALIDATION_SPLIT
        )
    )
)


cantidad_val_neg = min(
    cantidad_val_neg,
    cantidad_negativos - 1
)


val_samples.extend(
    negativos[
        :cantidad_val_neg
    ]
)


train_samples.extend(
    negativos[
        cantidad_val_neg:
    ]
)


# ============================================================
# MEZCLAR
# ============================================================

random.Random(
    SEED
).shuffle(
    train_samples
)


random.Random(
    SEED
).shuffle(
    val_samples
)


print("\n============================================")
print("DIVISION DEL DATASET")
print("============================================")

print(
    f"Entrenamiento: {len(train_samples)}"
)

print(
    f"Validacion: {len(val_samples)}"
)


# ============================================================
# SEPARAR RUTAS / ETIQUETAS
# ============================================================

train_paths = [
    muestra[0]
    for muestra in train_samples
]


train_labels = np.array(
    [
        muestra[1]
        for muestra in train_samples
    ],
    dtype=np.float32
)


val_paths = [
    muestra[0]
    for muestra in val_samples
]


val_labels = np.array(
    [
        muestra[1]
        for muestra in val_samples
    ],
    dtype=np.float32
)


# ============================================================
# LECTOR DE IMAGEN ROBUSTO PARA WINDOWS / UNICODE
# ============================================================

def leer_imagen_numpy(
    ruta_numpy
):

    # numpy_function puede entregar numpy.bytes_
    # o un arreglo escalar.

    if hasattr(
        ruta_numpy,
        "item"
    ):

        ruta_numpy = ruta_numpy.item()


    if isinstance(
        ruta_numpy,
        bytes
    ):

        ruta = ruta_numpy.decode(
            "utf-8"
        )

    else:

        ruta = str(
            ruta_numpy
        )


    # --------------------------------------------------------
    # np.fromfile permite trabajar mejor con nombres Unicode
    # en Windows, por ejemplo:
    #
    # télécharger (5).jpg
    # --------------------------------------------------------

    datos = np.fromfile(
        ruta,
        dtype=np.uint8
    )


    imagen = cv2.imdecode(
        datos,
        cv2.IMREAD_COLOR
    )


    if imagen is None:

        raise RuntimeError(
            f"No se pudo cargar: {ruta}"
        )


    # BGR -> RGB

    imagen = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2RGB
    )


    # Redimensionar

    imagen = cv2.resize(
        imagen,
        (
            IMG_SIZE[1],
            IMG_SIZE[0]
        ),
        interpolation=cv2.INTER_AREA
    )


    imagen = imagen.astype(
        np.float32
    )


    imagen = imagen / 255.0


    return imagen


# ============================================================
# FUNCION TENSORFLOW PARA CARGAR
# ============================================================

def cargar_imagen(
    ruta,
    etiqueta
):

    imagen = tf.numpy_function(
        leer_imagen_numpy,
        [ruta],
        tf.float32
    )


    imagen.set_shape(
        [
            IMG_SIZE[0],
            IMG_SIZE[1],
            3
        ]
    )


    etiqueta.set_shape(
        [NUM_SALIDAS]
    )


    return imagen, etiqueta


# ============================================================
# DATA AUGMENTATION
# ============================================================

def aumentar_datos(
    imagen,
    etiqueta
):

    etiqueta = tf.identity(
        etiqueta
    )


    # ========================================================
    # FLIP HORIZONTAL
    # ========================================================

    hacer_flip = (
        tf.random.uniform([])
        <
        0.5
    )


    def aplicar_flip():

        imagen_flip = (
            tf.image.flip_left_right(
                imagen
            )
        )


        objectness = etiqueta[4]

        x = etiqueta[0]
        y = etiqueta[1]
        w = etiqueta[2]
        h = etiqueta[3]


        # Solo modificamos x si realmente
        # existe un objeto.

        nuevo_x = tf.where(
            objectness > 0.5,
            1.0 - x,
            x
        )


        bbox = tf.stack(
            [
                nuevo_x,
                y,
                w,
                h
            ]
        )


        nueva_etiqueta = tf.concat(
            [
                bbox,
                etiqueta[4:]
            ],
            axis=0
        )


        return (
            imagen_flip,
            nueva_etiqueta
        )


    def no_flip():

        return (
            imagen,
            etiqueta
        )


    imagen, etiqueta = tf.cond(
        hacer_flip,
        aplicar_flip,
        no_flip
    )


    # ========================================================
    # BRILLO
    # ========================================================

    imagen = tf.image.random_brightness(
        imagen,
        max_delta=0.12
    )


    # ========================================================
    # CONTRASTE
    # ========================================================

    imagen = tf.image.random_contrast(
        imagen,
        lower=0.80,
        upper=1.20
    )


    # ========================================================
    # SATURACION
    # ========================================================

    imagen = tf.image.random_saturation(
        imagen,
        lower=0.85,
        upper=1.15
    )


    # ========================================================
    # LIMITAR 0-1
    # ========================================================

    imagen = tf.clip_by_value(
        imagen,
        0.0,
        1.0
    )


    return imagen, etiqueta


# ============================================================
# CREAR DATASETS
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


train_ds = (
    tf.data.Dataset.from_tensor_slices(
        (
            train_paths,
            train_labels
        )
    )
)


train_ds = train_ds.shuffle(
    buffer_size=len(
        train_paths
    ),
    seed=SEED,
    reshuffle_each_iteration=True
)


train_ds = train_ds.map(
    cargar_imagen,
    num_parallel_calls=AUTOTUNE
)


train_ds = train_ds.map(
    aumentar_datos,
    num_parallel_calls=AUTOTUNE
)


train_ds = train_ds.batch(
    BATCH_SIZE
)


train_ds = train_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# VALIDACION
# ============================================================

val_ds = (
    tf.data.Dataset.from_tensor_slices(
        (
            val_paths,
            val_labels
        )
    )
)


val_ds = val_ds.map(
    cargar_imagen,
    num_parallel_calls=AUTOTUNE
)


val_ds = val_ds.batch(
    BATCH_SIZE
)


val_ds = val_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# IoU
# ============================================================

def calcular_iou(
    bbox_real,
    bbox_pred
):

    # --------------------------------------------------------
    # REAL
    # --------------------------------------------------------

    xr = bbox_real[:, 0]
    yr = bbox_real[:, 1]
    wr = bbox_real[:, 2]
    hr = bbox_real[:, 3]


    real_x1 = xr - wr / 2.0
    real_y1 = yr - hr / 2.0

    real_x2 = xr + wr / 2.0
    real_y2 = yr + hr / 2.0


    # --------------------------------------------------------
    # PREDICHO
    # --------------------------------------------------------

    xp = bbox_pred[:, 0]
    yp = bbox_pred[:, 1]
    wp = bbox_pred[:, 2]
    hp = bbox_pred[:, 3]


    pred_x1 = xp - wp / 2.0
    pred_y1 = yp - hp / 2.0

    pred_x2 = xp + wp / 2.0
    pred_y2 = yp + hp / 2.0


    # --------------------------------------------------------
    # INTERSECCION
    # --------------------------------------------------------

    inter_x1 = tf.maximum(
        real_x1,
        pred_x1
    )

    inter_y1 = tf.maximum(
        real_y1,
        pred_y1
    )

    inter_x2 = tf.minimum(
        real_x2,
        pred_x2
    )

    inter_y2 = tf.minimum(
        real_y2,
        pred_y2
    )


    inter_w = tf.maximum(
        0.0,
        inter_x2 - inter_x1
    )

    inter_h = tf.maximum(
        0.0,
        inter_y2 - inter_y1
    )


    inter_area = (
        inter_w
        *
        inter_h
    )


    area_real = (
        tf.maximum(
            wr,
            0.0
        )
        *
        tf.maximum(
            hr,
            0.0
        )
    )


    area_pred = (
        tf.maximum(
            wp,
            0.0
        )
        *
        tf.maximum(
            hp,
            0.0
        )
    )


    union = (
        area_real
        +
        area_pred
        -
        inter_area
    )


    return (
        inter_area
        /
        (
            union
            +
            1e-7
        )
    )


# ============================================================
# FUNCION LOSS
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="DetectorObjectness"
)
def detector_loss(
    y_true,
    y_pred
):

    # ========================================================
    # REAL
    # ========================================================

    bbox_real = y_true[
        :,
        0:4
    ]


    objectness_real = y_true[
        :,
        4
    ]


    clases_reales = y_true[
        :,
        5:
    ]


    # ========================================================
    # PREDICCION
    # ========================================================

    bbox_pred = tf.sigmoid(
        y_pred[
            :,
            0:4
        ]
    )


    objectness_logit = y_pred[
        :,
        4
    ]


    class_logits = y_pred[
        :,
        5:
    ]


    # ========================================================
    # LOSS OBJECTNESS
    # ========================================================

    loss_obj_individual = (
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=objectness_real,
            logits=objectness_logit
        )
    )


    # Dar mas importancia a los positivos

    pesos_obj = tf.where(
        objectness_real > 0.5,
        POSITIVE_OBJECTNESS_WEIGHT,
        1.0
    )


    loss_objectness = tf.reduce_mean(
        loss_obj_individual
        *
        pesos_obj
    )


    # ========================================================
    # MASCARA DE OBJETOS REALES
    # ========================================================

    mascara = tf.cast(
        objectness_real > 0.5,
        tf.float32
    )


    cantidad_positivos = (
        tf.reduce_sum(
            mascara
        )
        +
        1e-7
    )


    # ========================================================
    # LOSS CLASE
    # ========================================================

    loss_clase_individual = (
        tf.nn.softmax_cross_entropy_with_logits(
            labels=clases_reales,
            logits=class_logits
        )
    )


    loss_clase = (
        tf.reduce_sum(
            loss_clase_individual
            *
            mascara
        )
        /
        cantidad_positivos
    )


    # ========================================================
    # ERROR COORDENADAS
    # ========================================================

    error_bbox = tf.reduce_mean(
        tf.square(
            bbox_real
            -
            bbox_pred
        ),
        axis=-1
    )


    # ========================================================
    # IoU LOSS
    # ========================================================

    iou = calcular_iou(
        bbox_real,
        bbox_pred
    )


    loss_iou = (
        1.0
        -
        iou
    )


    loss_bbox_individual = (
        error_bbox
        +
        loss_iou
    )


    # --------------------------------------------------------
    # SOLO CALCULAR BBOX CUANDO HAY OBJETO
    # --------------------------------------------------------

    loss_bbox = (
        tf.reduce_sum(
            loss_bbox_individual
            *
            mascara
        )
        /
        cantidad_positivos
    )


    # ========================================================
    # LOSS TOTAL
    # ========================================================

    loss_total = (

        OBJECTNESS_LOSS_WEIGHT
        *
        loss_objectness

        +

        CLASS_LOSS_WEIGHT
        *
        loss_clase

        +

        BBOX_LOSS_WEIGHT
        *
        loss_bbox
    )


    return loss_total


# ============================================================
# METRICA OBJECTNESS ACCURACY
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="DetectorObjectness"
)
def objectness_accuracy(
    y_true,
    y_pred
):

    real = (
        y_true[:, 4]
        >
        0.5
    )


    pred = (
        tf.sigmoid(
            y_pred[:, 4]
        )
        >
        0.5
    )


    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                real,
                pred
            ),
            tf.float32
        )
    )


# ============================================================
# OBJECTNESS PRECISION
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="DetectorObjectness"
)
def objectness_precision(
    y_true,
    y_pred
):

    real = tf.cast(
        y_true[:, 4] > 0.5,
        tf.float32
    )


    pred = tf.cast(
        tf.sigmoid(
            y_pred[:, 4]
        ) > 0.5,
        tf.float32
    )


    tp = tf.reduce_sum(
        real * pred
    )


    fp = tf.reduce_sum(
        (1.0 - real)
        *
        pred
    )


    return (
        tp
        /
        (
            tp
            +
            fp
            +
            1e-7
        )
    )


# ============================================================
# OBJECTNESS RECALL
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="DetectorObjectness"
)
def objectness_recall(
    y_true,
    y_pred
):

    real = tf.cast(
        y_true[:, 4] > 0.5,
        tf.float32
    )


    pred = tf.cast(
        tf.sigmoid(
            y_pred[:, 4]
        ) > 0.5,
        tf.float32
    )


    tp = tf.reduce_sum(
        real * pred
    )


    fn = tf.reduce_sum(
        real
        *
        (1.0 - pred)
    )


    return (
        tp
        /
        (
            tp
            +
            fn
            +
            1e-7
        )
    )


# ============================================================
# CLASS ACCURACY
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="DetectorObjectness"
)
def class_accuracy(
    y_true,
    y_pred
):

    mascara = tf.cast(
        y_true[:, 4] > 0.5,
        tf.float32
    )


    clase_real = tf.argmax(
        y_true[:, 5:],
        axis=-1
    )


    clase_pred = tf.argmax(
        y_pred[:, 5:],
        axis=-1
    )


    correcto = tf.cast(
        tf.equal(
            clase_real,
            clase_pred
        ),
        tf.float32
    )


    return (
        tf.reduce_sum(
            correcto
            *
            mascara
        )
        /
        (
            tf.reduce_sum(
                mascara
            )
            +
            1e-7
        )
    )


# ============================================================
# BBOX IoU
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="DetectorObjectness"
)
def bbox_iou(
    y_true,
    y_pred
):

    mascara = tf.cast(
        y_true[:, 4] > 0.5,
        tf.float32
    )


    bbox_real = y_true[
        :,
        0:4
    ]


    bbox_pred = tf.sigmoid(
        y_pred[
            :,
            0:4
        ]
    )


    iou = calcular_iou(
        bbox_real,
        bbox_pred
    )


    return (
        tf.reduce_sum(
            iou
            *
            mascara
        )
        /
        (
            tf.reduce_sum(
                mascara
            )
            +
            1e-7
        )
    )


# ============================================================
# ============================================================
#
# CREAR RED NEURONAL DESDE CERO
#
# ============================================================
# ============================================================


print("\n============================================")
print("CREANDO RED NEURONAL NUEVA")
print("============================================")


# ============================================================
# CAPA DE ENTRADA
# ============================================================

entrada = tf.keras.layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    name="CAPA_ENTRADA"
)


x = entrada


# ============================================================
# CAPAS INTERMEDIAS 1 - 6
# 32 FILTROS
# ============================================================

# 1
x = tf.keras.layers.Conv2D(
    32,
    3,
    strides=2,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.00001
    ),
    name="capa_01_conv"
)(x)


# 2
x = tf.keras.layers.BatchNormalization(
    name="capa_02_batchnorm"
)(x)


# 3
x = tf.keras.layers.ReLU(
    name="capa_03_relu"
)(x)


# 4
x = tf.keras.layers.Conv2D(
    32,
    3,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.00001
    ),
    name="capa_04_conv"
)(x)


# 5
x = tf.keras.layers.BatchNormalization(
    name="capa_05_batchnorm"
)(x)


# 6
x = tf.keras.layers.ReLU(
    name="capa_06_relu"
)(x)


# ============================================================
# CAPAS 7 - 12
# 64 FILTROS
# ============================================================

# 7
x = tf.keras.layers.Conv2D(
    64,
    3,
    strides=2,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.00001
    ),
    name="capa_07_conv"
)(x)


# 8
x = tf.keras.layers.BatchNormalization(
    name="capa_08_batchnorm"
)(x)


# 9
x = tf.keras.layers.ReLU(
    name="capa_09_relu"
)(x)


# 10
x = tf.keras.layers.Conv2D(
    64,
    3,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.00001
    ),
    name="capa_10_conv"
)(x)


# 11
x = tf.keras.layers.BatchNormalization(
    name="capa_11_batchnorm"
)(x)


# 12
x = tf.keras.layers.ReLU(
    name="capa_12_relu"
)(x)


# ============================================================
# CAPAS 13 - 18
# 128 FILTROS
# ============================================================

# 13
x = tf.keras.layers.Conv2D(
    128,
    3,
    strides=2,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.00001
    ),
    name="capa_13_conv"
)(x)


# 14
x = tf.keras.layers.BatchNormalization(
    name="capa_14_batchnorm"
)(x)


# 15
x = tf.keras.layers.ReLU(
    name="capa_15_relu"
)(x)


# 16
x = tf.keras.layers.Conv2D(
    128,
    3,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.00001
    ),
    name="capa_16_conv"
)(x)


# 17
x = tf.keras.layers.BatchNormalization(
    name="capa_17_batchnorm"
)(x)


# 18
x = tf.keras.layers.ReLU(
    name="capa_18_relu"
)(x)


# ============================================================
# CAPAS 19 - 24
# 256 FILTROS
# ============================================================

# 19
x = tf.keras.layers.Conv2D(
    256,
    3,
    strides=2,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.00001
    ),
    name="capa_19_conv"
)(x)


# 20
x = tf.keras.layers.BatchNormalization(
    name="capa_20_batchnorm"
)(x)


# 21
x = tf.keras.layers.ReLU(
    name="capa_21_relu"
)(x)


# 22
x = tf.keras.layers.Conv2D(
    256,
    3,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.00001
    ),
    name="capa_22_conv"
)(x)


# 23
x = tf.keras.layers.BatchNormalization(
    name="capa_23_batchnorm"
)(x)


# 24
x = tf.keras.layers.ReLU(
    name="capa_24_relu"
)(x)


# ============================================================
# CAPAS 25 - 30
# 384 FILTROS
# ============================================================

# 25
x = tf.keras.layers.Conv2D(
    384,
    3,
    strides=2,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.00001
    ),
    name="capa_25_conv"
)(x)


# 26
x = tf.keras.layers.BatchNormalization(
    name="capa_26_batchnorm"
)(x)


# 27
x = tf.keras.layers.ReLU(
    name="capa_27_relu"
)(x)


# 28
x = tf.keras.layers.Conv2D(
    384,
    3,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.00001
    ),
    name="capa_28_conv"
)(x)


# 29
x = tf.keras.layers.ReLU(
    name="capa_29_relu"
)(x)


# 30
#
# Flatten conserva mejor la informacion espacial
# para poder aprender x, y, w, h.

x = tf.keras.layers.Flatten(
    name="capa_30_flatten"
)(x)


# ============================================================
# CAPA DE SALIDA
# ============================================================

salida = tf.keras.layers.Dense(
    NUM_SALIDAS,
    activation=None,
    name="CAPA_SALIDA"
)(x)


# ============================================================
# CREAR MODELO
# ============================================================

model = tf.keras.Model(
    inputs=entrada,
    outputs=salida,
    name="Detector_Objectness_30_Capas"
)


# ============================================================
# COMPROBAR NUMERO DE CAPAS
# ============================================================

model.summary()


print("\n============================================")
print("COMPROBACION DE ARQUITECTURA")
print("============================================")


print(
    f"Capas totales: {len(model.layers)}"
)


print(
    f"Capas intermedias: "
    f"{len(model.layers) - 2}"
)


print(
    f"Salidas: "
    f"{model.output_shape[-1]}"
)


print(
    f"Salidas esperadas: "
    f"{NUM_SALIDAS}"
)


if len(model.layers) != 32:

    raise RuntimeError(
        "ERROR: La red no tiene "
        "1 entrada + 30 intermedias + 1 salida."
    )


if model.output_shape[-1] != NUM_SALIDAS:

    raise RuntimeError(
        "ERROR en el numero de salidas."
    )


print("\nCORRECTO:")
print(
    "1 capa de entrada"
)

print(
    "30 capas intermedias"
)

print(
    "1 capa de salida"
)


print("\nFormato de salida:")

print(
    "[x, y, w, h, objectness, "
    +
    ", ".join(
        CLASS_NAMES
    )
    +
    "]"
)


# ============================================================
# COMPILAR
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=detector_loss,

    metrics=[
        objectness_accuracy,
        objectness_precision,
        objectness_recall,
        class_accuracy,
        bbox_iou
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # --------------------------------------------------------
    # GUARDAR MEJOR MODELO
    # --------------------------------------------------------

    tf.keras.callbacks.ModelCheckpoint(

        MEJOR_MODELO,

        monitor="val_loss",

        save_best_only=True,

        mode="min",

        verbose=1
    ),


    # --------------------------------------------------------
    # BAJAR LEARNING RATE SI SE ESTANCA
    # --------------------------------------------------------
    #
    # NO detiene las 150 epocas.
    # --------------------------------------------------------

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=8,

        min_lr=0.0000001,

        verbose=1
    ),


    # --------------------------------------------------------
    # GUARDAR HISTORIAL
    # --------------------------------------------------------

    tf.keras.callbacks.CSVLogger(
        HISTORIAL_CSV,
        append=False
    ),


    # --------------------------------------------------------
    # SOLO DETENER SI HAY NaN
    # --------------------------------------------------------

    tf.keras.callbacks.TerminateOnNaN()
]


# ============================================================
# ENTRENAR
# ============================================================

print("\n============================================")
print("INICIANDO ENTRENAMIENTO DESDE CERO")
print("============================================")

print(
    f"\nEpocas: {EPOCHS}"
)

print(
    f"Batch: {BATCH_SIZE}"
)

print(
    f"Learning rate inicial: {LEARNING_RATE}"
)

print(
    f"Entrenamiento: {len(train_samples)} imagenes"
)

print(
    f"Validacion: {len(val_samples)} imagenes"
)


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# EVALUAR
# ============================================================

print("\n============================================")
print("EVALUACION FINAL")
print("============================================")


resultados = model.evaluate(
    val_ds,
    verbose=1,
    return_dict=True
)


print("\n============================================")
print("RESULTADOS")
print("============================================")


for nombre, valor in resultados.items():

    print(
        f"{nombre}: {valor:.4f}"
    )


# ============================================================
# MOSTRAR RESULTADOS EN PORCENTAJE
# ============================================================

if "objectness_accuracy" in resultados:

    print(
        "\nObjectness accuracy: "
        f"{resultados['objectness_accuracy'] * 100:.2f}%"
    )


if "objectness_precision" in resultados:

    print(
        "Objectness precision: "
        f"{resultados['objectness_precision'] * 100:.2f}%"
    )


if "objectness_recall" in resultados:

    print(
        "Objectness recall: "
        f"{resultados['objectness_recall'] * 100:.2f}%"
    )


if "class_accuracy" in resultados:

    print(
        "Precision de clasificacion: "
        f"{resultados['class_accuracy'] * 100:.2f}%"
    )


if "bbox_iou" in resultados:

    print(
        "IoU del bounding box: "
        f"{resultados['bbox_iou'] * 100:.2f}%"
    )


print(
    f"\nLoss final: "
    f"{resultados['loss']:.4f}"
)


# ============================================================
# GUARDAR MODELO DE LA ULTIMA EPOCA
# ============================================================

model.save(
    MODELO_FINAL
)


print("\n============================================")
print("ENTRENAMIENTO TERMINADO")
print("============================================")


print(
    f"\nMejor modelo:\n{MEJOR_MODELO}"
)


print(
    f"\nModelo ultima epoca:\n{MODELO_FINAL}"
)


print(
    f"\nHistorial:\n{HISTORIAL_CSV}"
)


print("\nIMPORTANTE:")

print(
    "Para reconocer.py utiliza:"
)

print(
    MEJOR_MODELO
)