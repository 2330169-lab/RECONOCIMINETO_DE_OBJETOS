#crear una red neuronal para reconocer distintos objetos, los cuales para este caso seran
#una bolsa de mujer, peluches varios, y perfumes varios.

#red neuronal convolucional
#aprendizaje automatico
#clasificacion de imagenes



import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
import random
import numpy as np
import os

# ============================================================
# CONFIGURACION
# ============================================================

DATASET_DIR = Path("dataset")
ARCHIVO_CLASES = Path("clases.txt")

IMG_SIZE = (224, 224)

BATCH_SIZE = 16
EPOCHS = 150
SEED = 123

VALIDATION_SPLIT = 0.20

# Peso que tendra el error del bounding box
# respecto al error de clasificacion
BBOX_LOSS_WEIGHT = 5.0

# Extensiones permitidas
EXTENSIONES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}

# Fijar semillas
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# COMPROBAR DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"No se encontro la carpeta: {DATASET_DIR}"
    )

if not ARCHIVO_CLASES.exists():

    raise FileNotFoundError(
        "No se encontro clases.txt. "
        "Ejecuta primero crear_etiquetas.py"
    )


# ============================================================
# LEER CLASES
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


NUM_CLASSES = len(class_names)

NUM_SALIDAS = 4 + NUM_CLASSES


print("\n==========================================")
print("CLASES")
print("==========================================")

for i, clase in enumerate(class_names):
    print(f"{i} -> {clase}")

print("\nNumero de clases:", NUM_CLASSES)
print("Numero de salidas:", NUM_SALIDAS)

print("\nFormato de salida:")

print(
    "[x, y, w, h, "
    + ", ".join(class_names)
    + "]"
)


# ============================================================
# LEER ETIQUETA YOLO
# ============================================================

def leer_etiqueta(ruta_txt):

    """
    Espera una etiqueta con formato:

    clase x_centro y_centro ancho alto

    Ejemplo:

    2 0.500000 0.500000 0.300000 0.600000
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


    # ========================================================
    # ESTE MODELO SOLO MANEJA UN OBJETO POR IMAGEN
    # ========================================================

    if len(lineas) == 0:

        raise ValueError(
            f"Etiqueta vacia: {ruta_txt}"
        )

    if len(lineas) > 1:

        raise ValueError(
            f"{ruta_txt} tiene mas de un objeto. "
            "Este modelo esta diseñado para "
            "un objeto por imagen."
        )


    datos = lineas[0].split()

    if len(datos) != 5:

        raise ValueError(
            f"Formato incorrecto en: {ruta_txt}"
        )


    clase_id = int(datos[0])

    x = float(datos[1])
    y = float(datos[2])
    w = float(datos[3])
    h = float(datos[4])


    # ========================================================
    # COMPROBAR CLASE
    # ========================================================

    if clase_id < 0 or clase_id >= NUM_CLASSES:

        raise ValueError(
            f"Clase {clase_id} invalida en {ruta_txt}"
        )


    # ========================================================
    # COMPROBAR COORDENADAS
    # ========================================================

    valores = [x, y, w, h]

    for valor in valores:

        if valor < 0.0 or valor > 1.0:

            raise ValueError(
                f"Coordenada fuera de rango en {ruta_txt}"
            )


    if w <= 0 or h <= 0:

        raise ValueError(
            f"Bounding box invalido en {ruta_txt}"
        )


    # ========================================================
    # CREAR ONE-HOT PARA LA CLASE
    # ========================================================

    clase_one_hot = np.zeros(
        NUM_CLASSES,
        dtype=np.float32
    )

    clase_one_hot[clase_id] = 1.0


    # ========================================================
    # SALIDA
    # ========================================================

    # [x, y, w, h, clase0, clase1, clase2]

    etiqueta = np.concatenate(
        [
            np.array(
                [x, y, w, h],
                dtype=np.float32
            ),

            clase_one_hot
        ]
    )


    return etiqueta, clase_id


# ============================================================
# BUSCAR IMAGENES Y ETIQUETAS
# ============================================================

muestras_por_clase = {
    i: []
    for i in range(NUM_CLASSES)
}


errores = 0


print("\n==========================================")
print("BUSCANDO IMAGENES ETIQUETADAS")
print("==========================================\n")


for clase_id, nombre_clase in enumerate(class_names):

    carpeta = DATASET_DIR / nombre_clase

    if not carpeta.exists():

        print(
            f"ADVERTENCIA: No existe {carpeta}"
        )

        continue


    for ruta_imagen in carpeta.iterdir():

        if ruta_imagen.suffix not in EXTENSIONES:
            continue


        ruta_txt = ruta_imagen.with_suffix(".txt")


        # ====================================================
        # SOLO USAR IMAGENES CON ETIQUETA
        # ====================================================

        if not ruta_txt.exists():

            print(
                f"Sin etiqueta: {ruta_imagen.name}"
            )

            continue


        try:

            etiqueta, clase_txt = leer_etiqueta(
                ruta_txt
            )


            # =================================================
            # COMPROBAR QUE CARPETA Y TXT COINCIDAN
            # =================================================

            if clase_txt != clase_id:

                print(
                    f"ADVERTENCIA: {ruta_imagen.name}"
                )

                print(
                    f"  Carpeta indica: {nombre_clase}"
                )

                print(
                    f"  TXT indica: "
                    f"{class_names[clase_txt]}"
                )


            muestras_por_clase[
                clase_txt
            ].append(
                (
                    str(ruta_imagen),
                    etiqueta
                )
            )


        except Exception as e:

            print(
                f"ERROR en {ruta_imagen.name}: {e}"
            )

            errores += 1


# ============================================================
# MOSTRAR CANTIDAD DE IMAGENES
# ============================================================

print("\n==========================================")
print("IMAGENES VALIDAS")
print("==========================================")

total_imagenes = 0

for clase_id, nombre in enumerate(class_names):

    cantidad = len(
        muestras_por_clase[clase_id]
    )

    total_imagenes += cantidad

    print(
        f"{nombre}: {cantidad}"
    )


print("\nTotal:", total_imagenes)
print("Errores:", errores)


if total_imagenes == 0:

    raise RuntimeError(
        "No se encontraron imagenes etiquetadas."
    )


# ============================================================
# DIVIDIR TRAIN / VALIDATION
# ============================================================

# Hacemos la division por clase para evitar que
# accidentalmente una clase quede fuera de validacion.

train_samples = []
val_samples = []


for clase_id in range(NUM_CLASSES):

    muestras = muestras_por_clase[
        clase_id
    ].copy()

    random.Random(
        SEED + clase_id
    ).shuffle(muestras)


    cantidad = len(muestras)


    if cantidad < 2:

        raise RuntimeError(
            f"La clase '{class_names[clase_id]}' "
            "necesita por lo menos 2 imagenes."
        )


    cantidad_val = max(
        1,
        int(
            round(
                cantidad * VALIDATION_SPLIT
            )
        )
    )


    # Evitar mandar todas a validacion
    cantidad_val = min(
        cantidad_val,
        cantidad - 1
    )


    val_clase = muestras[
        :cantidad_val
    ]

    train_clase = muestras[
        cantidad_val:
    ]


    val_samples.extend(
        val_clase
    )

    train_samples.extend(
        train_clase
    )


# Mezclar
random.Random(SEED).shuffle(
    train_samples
)

random.Random(SEED).shuffle(
    val_samples
)


print("\n==========================================")
print("DIVISION DEL DATASET")
print("==========================================")

print(
    "Entrenamiento:",
    len(train_samples)
)

print(
    "Validacion:",
    len(val_samples)
)


# ============================================================
# SEPARAR RUTAS Y ETIQUETAS
# ============================================================

train_paths = [
    x[0]
    for x in train_samples
]

train_labels = np.array(
    [
        x[1]
        for x in train_samples
    ],
    dtype=np.float32
)


val_paths = [
    x[0]
    for x in val_samples
]

val_labels = np.array(
    [
        x[1]
        for x in val_samples
    ],
    dtype=np.float32
)


# ============================================================
# FUNCION PARA CARGAR IMAGEN
# ============================================================

def cargar_imagen(
    ruta,
    etiqueta
):

    imagen = tf.io.read_file(
        ruta
    )

    imagen = tf.io.decode_image(
        imagen,
        channels=3,
        expand_animations=False
    )

    imagen.set_shape(
        [None, None, 3]
    )


    # Convertir a 0-1
    imagen = tf.image.convert_image_dtype(
        imagen,
        tf.float32
    )


    # Resize
    imagen = tf.image.resize(
        imagen,
        IMG_SIZE
    )


    return imagen, etiqueta


# ============================================================
# DATA AUGMENTATION
# ============================================================

def aumentar_datos(
    imagen,
    etiqueta
):

    # Copiar etiqueta
    etiqueta = tf.identity(
        etiqueta
    )


    # ========================================================
    # VOLTEO HORIZONTAL
    # ========================================================

    hacer_flip = (
        tf.random.uniform([])
        < 0.5
    )


    def aplicar_flip():

        imagen_flip = (
            tf.image.flip_left_right(
                imagen
            )
        )


        # Extraer bounding box
        x = etiqueta[0]
        y = etiqueta[1]
        w = etiqueta[2]
        h = etiqueta[3]


        # Al invertir horizontalmente:
        #
        # x_nuevo = 1 - x
        #
        # y, w, h permanecen iguales.

        nuevo_bbox = tf.stack(
            [
                1.0 - x,
                y,
                w,
                h
            ]
        )


        nueva_etiqueta = tf.concat(
            [
                nuevo_bbox,
                etiqueta[4:]
            ],
            axis=0
        )


        return (
            imagen_flip,
            nueva_etiqueta
        )


    def sin_flip():

        return (
            imagen,
            etiqueta
        )


    imagen, etiqueta = tf.cond(
        hacer_flip,
        aplicar_flip,
        sin_flip
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
        lower=0.85,
        upper=1.15
    )


    # Mantener valores validos
    imagen = tf.clip_by_value(
        imagen,
        0.0,
        1.0
    )


    return imagen, etiqueta


# ============================================================
# CREAR DATASETS TF.DATA
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)


train_ds = train_ds.map(
    cargar_imagen,
    num_parallel_calls=AUTOTUNE
)


train_ds = train_ds.shuffle(
    buffer_size=len(train_paths),
    seed=SEED,
    reshuffle_each_iteration=True
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

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
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
# FUNCIONES PARA BOUNDING BOX
# ============================================================

def calcular_iou(
    bbox_real,
    bbox_pred
):

    """
    Bounding boxes en formato:

    x_centro
    y_centro
    ancho
    alto
    """


    # ========================================================
    # REAL
    # ========================================================

    xr = bbox_real[:, 0]
    yr = bbox_real[:, 1]
    wr = bbox_real[:, 2]
    hr = bbox_real[:, 3]


    real_x1 = xr - wr / 2.0
    real_y1 = yr - hr / 2.0

    real_x2 = xr + wr / 2.0
    real_y2 = yr + hr / 2.0


    # ========================================================
    # PREDICHO
    # ========================================================

    xp = bbox_pred[:, 0]
    yp = bbox_pred[:, 1]
    wp = bbox_pred[:, 2]
    hp = bbox_pred[:, 3]


    pred_x1 = xp - wp / 2.0
    pred_y1 = yp - hp / 2.0

    pred_x2 = xp + wp / 2.0
    pred_y2 = yp + hp / 2.0


    # ========================================================
    # INTERSECCION
    # ========================================================

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


    area_inter = (
        inter_w * inter_h
    )


    # ========================================================
    # AREAS
    # ========================================================

    area_real = (
        tf.maximum(wr, 0.0)
        *
        tf.maximum(hr, 0.0)
    )

    area_pred = (
        tf.maximum(wp, 0.0)
        *
        tf.maximum(hp, 0.0)
    )


    union = (
        area_real
        + area_pred
        - area_inter
    )


    iou = area_inter / (
        union + 1e-7
    )


    return iou


# ============================================================
# FUNCION DE PERDIDA
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="Detector"
)
def detector_loss(
    y_true,
    y_pred
):

    # ========================================================
    # BOUNDING BOX REAL
    # ========================================================

    bbox_real = y_true[
        :, :4
    ]


    # ========================================================
    # CLASE REAL
    # ========================================================

    clase_real = y_true[
        :, 4:
    ]


    # ========================================================
    # BOUNDING BOX PREDICHO
    # ========================================================

    # La salida de Dense es libre.
    # Sigmoid obliga las coordenadas
    # a permanecer entre 0 y 1.

    bbox_pred = tf.sigmoid(
        y_pred[:, :4]
    )


    # ========================================================
    # CLASE PREDICHA
    # ========================================================

    logits_clase = y_pred[
        :, 4:
    ]


    # ========================================================
    # LOSS DE CLASIFICACION
    # ========================================================

    loss_clase = (
        tf.keras.losses.categorical_crossentropy(
            clase_real,
            logits_clase,
            from_logits=True
        )
    )


    # ========================================================
    # LOSS DE COORDENADAS
    # ========================================================

    mse_bbox = tf.reduce_mean(
        tf.square(
            bbox_real
            -
            bbox_pred
        ),
        axis=-1
    )


    # ========================================================
    # LOSS IOU
    # ========================================================

    iou = calcular_iou(
        bbox_real,
        bbox_pred
    )

    loss_iou = (
        1.0 - iou
    )


    # ========================================================
    # LOSS TOTAL DEL RECUADRO
    # ========================================================

    loss_bbox = (
        mse_bbox
        +
        loss_iou
    )


    # ========================================================
    # LOSS TOTAL
    # ========================================================

    loss_total = (
        loss_clase
        +
        BBOX_LOSS_WEIGHT
        *
        loss_bbox
    )


    return loss_total


# ============================================================
# METRICA: PRECISION DE CLASE
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="Detector"
)
def class_accuracy(
    y_true,
    y_pred
):

    clase_real = tf.argmax(
        y_true[:, 4:],
        axis=-1
    )

    clase_pred = tf.argmax(
        y_pred[:, 4:],
        axis=-1
    )


    correcto = tf.cast(
        tf.equal(
            clase_real,
            clase_pred
        ),
        tf.float32
    )


    return tf.reduce_mean(
        correcto
    )


# ============================================================
# METRICA: IoU
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="Detector"
)
def bbox_iou(
    y_true,
    y_pred
):

    bbox_real = y_true[
        :, :4
    ]

    bbox_pred = tf.sigmoid(
        y_pred[:, :4]
    )


    valores_iou = calcular_iou(
        bbox_real,
        bbox_pred
    )


    return tf.reduce_mean(
        valores_iou
    )


# ============================================================
# CREAR RED NEURONAL
# ============================================================

# ============================================================
# CAPA DE ENTRADA
# ============================================================

entrada = layers.Input(
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
# ============================================================

# CAPA 1
x = layers.Conv2D(
    16,
    3,
    strides=2,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.0001
    ),
    name="capa_01_conv"
)(x)


# CAPA 2
x = layers.BatchNormalization(
    name="capa_02_batchnorm"
)(x)


# CAPA 3
x = layers.ReLU(
    name="capa_03_relu"
)(x)


# CAPA 4
x = layers.Conv2D(
    16,
    3,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.0001
    ),
    name="capa_04_conv"
)(x)


# CAPA 5
x = layers.BatchNormalization(
    name="capa_05_batchnorm"
)(x)


# CAPA 6
x = layers.ReLU(
    name="capa_06_relu"
)(x)


# ============================================================
# CAPAS INTERMEDIAS 7 - 12
# ============================================================

# CAPA 7
x = layers.Conv2D(
    32,
    3,
    strides=2,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.0001
    ),
    name="capa_07_conv"
)(x)


# CAPA 8
x = layers.BatchNormalization(
    name="capa_08_batchnorm"
)(x)


# CAPA 9
x = layers.ReLU(
    name="capa_09_relu"
)(x)


# CAPA 10
x = layers.Conv2D(
    32,
    3,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.0001
    ),
    name="capa_10_conv"
)(x)


# CAPA 11
x = layers.BatchNormalization(
    name="capa_11_batchnorm"
)(x)


# CAPA 12
x = layers.ReLU(
    name="capa_12_relu"
)(x)


# ============================================================
# CAPAS INTERMEDIAS 13 - 18
# ============================================================

# CAPA 13
x = layers.Conv2D(
    64,
    3,
    strides=2,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.0001
    ),
    name="capa_13_conv"
)(x)


# CAPA 14
x = layers.BatchNormalization(
    name="capa_14_batchnorm"
)(x)


# CAPA 15
x = layers.ReLU(
    name="capa_15_relu"
)(x)


# CAPA 16
x = layers.Conv2D(
    64,
    3,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.0001
    ),
    name="capa_16_conv"
)(x)


# CAPA 17
x = layers.BatchNormalization(
    name="capa_17_batchnorm"
)(x)


# CAPA 18
x = layers.ReLU(
    name="capa_18_relu"
)(x)


# ============================================================
# CAPAS INTERMEDIAS 19 - 24
# ============================================================

# CAPA 19
x = layers.Conv2D(
    128,
    3,
    strides=2,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.0001
    ),
    name="capa_19_conv"
)(x)


# CAPA 20
x = layers.BatchNormalization(
    name="capa_20_batchnorm"
)(x)


# CAPA 21
x = layers.ReLU(
    name="capa_21_relu"
)(x)


# CAPA 22
x = layers.Conv2D(
    128,
    3,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.0001
    ),
    name="capa_22_conv"
)(x)


# CAPA 23
x = layers.BatchNormalization(
    name="capa_23_batchnorm"
)(x)


# CAPA 24
x = layers.ReLU(
    name="capa_24_relu"
)(x)


# ============================================================
# CAPAS INTERMEDIAS 25 - 30
# ============================================================

# CAPA 25
x = layers.Conv2D(
    256,
    3,
    strides=2,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.0001
    ),
    name="capa_25_conv"
)(x)


# CAPA 26
x = layers.BatchNormalization(
    name="capa_26_batchnorm"
)(x)


# CAPA 27
x = layers.ReLU(
    name="capa_27_relu"
)(x)


# CAPA 28
x = layers.Conv2D(
    256,
    3,
    padding="same",
    kernel_regularizer=tf.keras.regularizers.l2(
        0.0001
    ),
    name="capa_28_conv"
)(x)


# CAPA 29
x = layers.ReLU(
    name="capa_29_relu"
)(x)


# CAPA 30
x = layers.GlobalAveragePooling2D(
    name="capa_30_global_average"
)(x)


# ============================================================
# CAPA DE SALIDA
# ============================================================

# Para 3 clases:
#
# 4 coordenadas
# +
# 3 clases
# =
# 7 salidas

salida = layers.Dense(
    NUM_SALIDAS,
    activation=None,
    name="CAPA_SALIDA"
)(x)


# ============================================================
# CREAR MODELO
# ============================================================

model = Model(
    inputs=entrada,
    outputs=salida,
    name="Detector_Objetos_30_Capas"
)


# ============================================================
# COMPROBAR ARQUITECTURA
# ============================================================

print("\n==========================================")
print("ARQUITECTURA")
print("==========================================\n")


model.summary()


numero_total = len(
    model.layers
)

numero_intermedias = (
    numero_total - 2
)


print("\n==========================================")
print("COMPROBACION DE CAPAS")
print("==========================================")

print(
    "Capas totales:",
    numero_total
)

print(
    "Capas intermedias:",
    numero_intermedias
)


if numero_total == 32:

    print("\nCORRECTO:")

    print(
        "1 entrada + "
        "30 intermedias + "
        "1 salida"
    )

else:

    print(
        "\nADVERTENCIA: "
        "El modelo no tiene 32 capas."
    )


# ============================================================
# COMPILAR
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss=detector_loss,

    metrics=[
        class_accuracy,
        bbox_iou
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # Guarda el mejor modelo encontrado
    tf.keras.callbacks.ModelCheckpoint(
        "mejor_detector.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    # Baja automáticamente el learning rate
    # cuando el modelo deja de mejorar
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=0.0000001,
        verbose=1
    ),

    # Detiene solamente si aparece NaN
    tf.keras.callbacks.TerminateOnNaN()
]
# ============================================================
# ENTRENAR
# ============================================================

print("\n==========================================")
print("INICIANDO ENTRENAMIENTO")
print("==========================================\n")


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks
)


# ============================================================
# EVALUAR MODELO
# ============================================================

print("\n==========================================")
print("EVALUACION FINAL")
print("==========================================\n")


resultados = model.evaluate(
    val_ds,
    verbose=1
)


print("\nResultados:")

for nombre, valor in zip(
    model.metrics_names,
    resultados
):

    print(
        f"{nombre}: {valor:.4f}"
    )


# ============================================================
# GUARDAR MODELO FINAL
# ============================================================

model.save(
    "modelo_detector_30_capas.keras"
)


print("\n==========================================")
print("ENTRENAMIENTO TERMINADO")
print("==========================================")

print(
    "\nModelo final:"
)

print(
    "modelo_detector_30_capas.keras"
)

print(
    "\nMejor modelo:"
)

print(
    "mejor_detector.keras"
)


# ============================================================
# MOSTRAR SIGNIFICADO DE LA SALIDA
# ============================================================

print("\nLa salida de la red es:")

print(
    "[x, y, w, h, "
    + ", ".join(class_names)
    + "]"
)

print(
    "\nLas primeras 4 salidas son "
    "logits que reconocer.py convertira "
    "con sigmoid."
)

print(
    "Las salidas restantes son logits "
    "de las clases y reconocer.py "
    "convertira con softmax."
)