import cv2
import os
from pathlib import Path

# ============================================================
# CONFIGURACION
# ============================================================

DATASET_DIR = "dataset"

# IMPORTANTE:
# Estos numeros seran los IDs de cada clase
CLASES = {
    "Bolsas": 0,
    "Peluches": 1,
    "Perfume": 2
}

# Tamaño maximo de la imagen mostrada en pantalla
MAX_ANCHO = 1200
MAX_ALTO = 800

# Extensiones permitidas
EXTENSIONES = [
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
]


# ============================================================
# CREAR ARCHIVO clases.txt
# ============================================================

with open("clases.txt", "w", encoding="utf-8") as archivo:

    clases_ordenadas = sorted(
        CLASES.items(),
        key=lambda x: x[1]
    )

    for nombre, indice in clases_ordenadas:
        archivo.write(nombre + "\n")

print("Archivo clases.txt creado.")


# ============================================================
# FUNCION PARA REDIMENSIONAR SOLO LA VISUALIZACION
# ============================================================

def preparar_visualizacion(imagen):

    alto, ancho = imagen.shape[:2]

    escala_ancho = MAX_ANCHO / ancho
    escala_alto = MAX_ALTO / alto

    escala = min(
        escala_ancho,
        escala_alto,
        1.0
    )

    nuevo_ancho = int(ancho * escala)
    nuevo_alto = int(alto * escala)

    if escala < 1.0:

        imagen_mostrada = cv2.resize(
            imagen,
            (nuevo_ancho, nuevo_alto)
        )

    else:
        imagen_mostrada = imagen.copy()

    return imagen_mostrada, escala


# ============================================================
# FUNCION PARA GUARDAR ETIQUETA YOLO
# ============================================================

def guardar_etiqueta(
    ruta_txt,
    clase_id,
    x,
    y,
    w,
    h,
    ancho_imagen,
    alto_imagen
):

    # --------------------------------------------------------
    # Convertir de:
    #
    # x esquina superior izquierda
    # y esquina superior izquierda
    # ancho
    # alto
    #
    # a formato YOLO:
    #
    # x centro
    # y centro
    # ancho
    # alto
    #
    # Todos normalizados de 0 a 1
    # --------------------------------------------------------

    x_centro = (
        x + w / 2
    ) / ancho_imagen

    y_centro = (
        y + h / 2
    ) / alto_imagen

    ancho_normalizado = (
        w / ancho_imagen
    )

    alto_normalizado = (
        h / alto_imagen
    )

    # Guardar archivo
    with open(
        ruta_txt,
        "w",
        encoding="utf-8"
    ) as archivo:

        archivo.write(
            f"{clase_id} "
            f"{x_centro:.6f} "
            f"{y_centro:.6f} "
            f"{ancho_normalizado:.6f} "
            f"{alto_normalizado:.6f}\n"
        )


# ============================================================
# BUSCAR TODAS LAS IMAGENES
# ============================================================

imagenes = []

for nombre_clase, clase_id in CLASES.items():

    carpeta = Path(DATASET_DIR) / nombre_clase

    if not carpeta.exists():

        print(
            f"ADVERTENCIA: No existe la carpeta: {carpeta}"
        )

        continue

    for archivo in carpeta.iterdir():

        if archivo.suffix in EXTENSIONES:

            imagenes.append(
                (
                    archivo,
                    nombre_clase,
                    clase_id
                )
            )


# Ordenar imágenes
imagenes.sort(
    key=lambda x: str(x[0])
)


# ============================================================
# INFORMACION
# ============================================================

print("\n============================================")
print("CREADOR DE ETIQUETAS")
print("============================================")

print(f"\nTotal de imagenes encontradas: {len(imagenes)}")

print("\nClases:")

for nombre, indice in CLASES.items():
    print(f"{indice} -> {nombre}")


print("\n============================================")
print("CONTROLES")
print("============================================")

print("""
1. Dibuja un rectangulo alrededor del objeto.
2. Presiona ENTER o ESPACIO para aceptar.
3. Presiona C para cancelar el rectangulo.

Despues de seleccionar:

    S = Guardar
    R = Repetir rectangulo
    N = Saltar imagen
    Q = Salir del programa

Las imagenes que ya tengan .txt seran omitidas.
""")


input("Presiona ENTER para comenzar...")


# ============================================================
# CONTADORES
# ============================================================

total = len(imagenes)

etiquetadas = 0
omitidas = 0


# ============================================================
# RECORRER IMAGENES
# ============================================================

for numero, datos in enumerate(
    imagenes,
    start=1
):

    ruta_imagen, nombre_clase, clase_id = datos

    ruta_txt = ruta_imagen.with_suffix(".txt")


    # ========================================================
    # OMITIR SI YA EXISTE
    # ========================================================

    if ruta_txt.exists():

        print(
            f"[{numero}/{total}] "
            f"Ya etiquetada: {ruta_imagen.name}"
        )

        omitidas += 1
        continue


    # ========================================================
    # CARGAR IMAGEN
    # ========================================================

    imagen_original = cv2.imread(
        str(ruta_imagen)
    )

    if imagen_original is None:

        print(
            f"ERROR al cargar: {ruta_imagen}"
        )

        continue


    alto_original, ancho_original = (
        imagen_original.shape[:2]
    )


    # ========================================================
    # PREPARAR IMAGEN PARA PANTALLA
    # ========================================================

    imagen_mostrada, escala = preparar_visualizacion(
        imagen_original
    )


    while True:

        copia = imagen_mostrada.copy()


        # ====================================================
        # MOSTRAR INFORMACION SOBRE LA IMAGEN
        # ====================================================

        texto = (
            f"[{numero}/{total}] "
            f"Clase: {nombre_clase}"
        )

        cv2.rectangle(
            copia,
            (0, 0),
            (copia.shape[1], 45),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            copia,
            texto,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ====================================================
        # SELECCIONAR ROI
        # ====================================================

        print("\n------------------------------------")

        print(
            f"[{numero}/{total}] "
            f"{ruta_imagen}"
        )

        print(
            f"Clase: {nombre_clase} "
            f"(ID {clase_id})"
        )

        print(
            "Dibuja un rectangulo alrededor del objeto."
        )

        roi = cv2.selectROI(
            "Etiquetar objeto",
            copia,
            showCrosshair=True,
            fromCenter=False
        )

        x, y, w, h = roi


        # ====================================================
        # ROI CANCELADO
        # ====================================================

        if w == 0 or h == 0:

            print("\nNo se selecciono ningun objeto.")

            print(
                "N = Saltar"
            )

            print(
                "R = Reintentar"
            )

            print(
                "Q = Salir"
            )

            tecla = input(
                "Opcion: "
            ).lower()

            if tecla == "q":

                cv2.destroyAllWindows()

                print(
                    "\nPrograma finalizado."
                )

                exit()

            elif tecla == "n":

                print(
                    "Imagen omitida."
                )

                break

            else:
                continue


        # ====================================================
        # CONVERTIR COORDENADAS A IMAGEN ORIGINAL
        # ====================================================

        x_original = int(
            x / escala
        )

        y_original = int(
            y / escala
        )

        w_original = int(
            w / escala
        )

        h_original = int(
            h / escala
        )


        # Limitar coordenadas
        x_original = max(
            0,
            min(
                x_original,
                ancho_original - 1
            )
        )

        y_original = max(
            0,
            min(
                y_original,
                alto_original - 1
            )
        )

        w_original = min(
            w_original,
            ancho_original - x_original
        )

        h_original = min(
            h_original,
            alto_original - y_original
        )


        # ====================================================
        # MOSTRAR RESULTADO
        # ====================================================

        preview = imagen_mostrada.copy()

        cv2.rectangle(
            preview,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )


        etiqueta = (
            f"{nombre_clase}"
        )

        cv2.rectangle(
            preview,
            (x, max(0, y - 35)),
            (x + 180, y),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            preview,
            etiqueta,
            (x + 5, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.imshow(
            "Vista previa",
            preview
        )

        cv2.waitKey(1)


        # ====================================================
        # PREGUNTAR QUE HACER
        # ====================================================

        print("\nRectangulo seleccionado.")

        print("S = Guardar")
        print("R = Repetir")
        print("N = Saltar")
        print("Q = Salir")

        opcion = input(
            "Opcion: "
        ).lower()


        cv2.destroyWindow(
            "Vista previa"
        )


        # ====================================================
        # GUARDAR
        # ====================================================

        if opcion == "s":

            guardar_etiqueta(
                ruta_txt,
                clase_id,
                x_original,
                y_original,
                w_original,
                h_original,
                ancho_original,
                alto_original
            )

            etiquetadas += 1

            print(
                f"Etiqueta guardada: {ruta_txt.name}"
            )

            break


        # ====================================================
        # REPETIR
        # ====================================================

        elif opcion == "r":

            print(
                "Selecciona nuevamente."
            )

            continue


        # ====================================================
        # SALTAR
        # ====================================================

        elif opcion == "n":

            print(
                "Imagen omitida."
            )

            break


        # ====================================================
        # SALIR
        # ====================================================

        elif opcion == "q":

            cv2.destroyAllWindows()

            print("\n====================================")
            print("PROGRAMA DETENIDO")
            print("====================================")

            print(
                f"Nuevas etiquetas: {etiquetadas}"
            )

            print(
                f"Ya existentes: {omitidas}"
            )

            exit()


        # Si escribe otra cosa, repetir
        else:

            print(
                "Opcion no reconocida. "
                "Se repetira la imagen."
            )


# ============================================================
# FINAL
# ============================================================

cv2.destroyAllWindows()

print("\n============================================")
print("ETIQUETADO TERMINADO")
print("============================================")

print(
    f"Imagenes encontradas: {total}"
)

print(
    f"Nuevas etiquetas creadas: {etiquetadas}"
)

print(
    f"Etiquetas que ya existian: {omitidas}"
)

print("\nTodos los archivos fueron procesados.")


