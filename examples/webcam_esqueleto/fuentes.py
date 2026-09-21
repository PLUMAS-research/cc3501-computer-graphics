"""De donde salen las posiciones de las articulaciones.

Las dos fuentes entregan lo mismo: un diccionario de articulacion a posicion
global en metros, con el origen entre las caderas y el eje Y hacia arriba. El
resto del ejemplo no sabe cual de las dos esta conectada.

- `demo` no necesita instalar nada: sintetiza una caminata y le agrega ruido,
  para imitar lo que entrega un estimador real.
- `camara` usa mediapipe sobre la webcam o sobre un archivo de video, y ademas
  devuelve la imagen para dibujarla en pantalla.
"""

from pathlib import Path

import numpy as np

from .esqueleto import ARTICULACIONES, HUESOS

RUTA_MODELO = Path(__file__).parent.parent.parent / "assets" / "pose_landmarker_lite.task"

MENSAJE_SIN_DEPENDENCIAS = """
Este ejemplo necesita mediapipe y opencv, que no vienen instalados por omision.

    uv sync --extra webcam

Mientras tanto, la fuente sintetica corre sin instalar nada:

    uv run python caja_de_juguetes.py webcam_esqueleto --fuente demo
"""

# mediapipe entrega x hacia la derecha de la imagen, y hacia abajo y z hacia la
# camara. La escena usa Y hacia arriba, de ahi el signo de la segunda
# componente. El de la tercera decide si la figura queda mirando a la camara o
# de espaldas; si sale al reves, se invierte aqui.
CONVERSION_DE_EJES = np.array([1.0, -1.0, -1.0])

INDICE_DE_LANDMARK = {
    "cabeza": 0,
    "hombro_izq": 11, "hombro_der": 12,
    "codo_izq": 13, "codo_der": 14,
    "muneca_izq": 15, "muneca_der": 16,
    "cadera_izq": 23, "cadera_der": 24,
    "rodilla_izq": 25, "rodilla_der": 26,
    "tobillo_izq": 27, "tobillo_der": 28,
}

# los pares de landmarks cuyo punto medio define una articulacion que mediapipe
# no entrega directamente
PUNTOS_MEDIOS = {"cadera": (23, 24), "cuello": (11, 12)}

LARGOS_DEMO = {
    ("cadera", "cuello"): 0.52,
    ("cuello", "cabeza"): 0.20,
    ("cuello", "hombro_izq"): 0.18,
    ("hombro_izq", "codo_izq"): 0.28,
    ("codo_izq", "muneca_izq"): 0.25,
    ("cuello", "hombro_der"): 0.18,
    ("hombro_der", "codo_der"): 0.28,
    ("codo_der", "muneca_der"): 0.25,
    ("cadera", "cadera_izq"): 0.12,
    ("cadera_izq", "rodilla_izq"): 0.43,
    ("rodilla_izq", "tobillo_izq"): 0.41,
    ("cadera", "cadera_der"): 0.12,
    ("cadera_der", "rodilla_der"): 0.43,
    ("rodilla_der", "tobillo_der"): 0.41,
}

RUIDO_DEMO = 0.012   # metros de desviacion estandar por articulacion


def _direccion_colgante(angulo, lateral=0.0):
    """Direccion de una extremidad que cuelga y se balancea `angulo` radianes."""
    direccion = np.array([lateral, -np.cos(angulo), np.sin(angulo)])
    return direccion / np.linalg.norm(direccion)


def nueva_fuente_demo(semilla=0):
    return {"tipo": "demo", "rng": np.random.default_rng(semilla), "ruido": RUIDO_DEMO}


def _leer_demo(fuente, tiempo):
    """Una caminata sintetica, con ruido para imitar a un estimador real."""
    paso = 2.4 * tiempo
    brazo = 0.55 * np.sin(paso)
    pierna = 0.45 * np.sin(paso + np.pi)
    codo = 0.55 + 0.35 * max(0.0, np.sin(paso))
    rodilla = 0.35 + 0.45 * max(0.0, np.sin(paso + np.pi))

    posiciones = {"cadera": np.array([0.0, 0.03 * np.sin(2 * paso), 0.0])}

    direcciones = {
        ("cadera", "cuello"): np.array([0.0, 1.0, 0.0]),
        ("cuello", "cabeza"): np.array([0.0, 1.0, 0.0]),
        ("cuello", "hombro_izq"): np.array([1.0, 0.05, 0.0]),
        ("cuello", "hombro_der"): np.array([-1.0, 0.05, 0.0]),
        ("hombro_izq", "codo_izq"): _direccion_colgante(brazo, 0.18),
        ("codo_izq", "muneca_izq"): _direccion_colgante(brazo + codo, 0.12),
        ("hombro_der", "codo_der"): _direccion_colgante(-brazo, -0.18),
        ("codo_der", "muneca_der"): _direccion_colgante(-brazo + codo, -0.12),
        ("cadera", "cadera_izq"): np.array([0.7, -0.7, 0.0]),
        ("cadera", "cadera_der"): np.array([-0.7, -0.7, 0.0]),
        ("cadera_izq", "rodilla_izq"): _direccion_colgante(pierna),
        ("rodilla_izq", "tobillo_izq"): _direccion_colgante(pierna - rodilla),
        ("cadera_der", "rodilla_der"): _direccion_colgante(-pierna),
        ("rodilla_der", "tobillo_der"): _direccion_colgante(-pierna - rodilla),
    }

    for hueso in HUESOS:
        padre, hijo = hueso
        direccion = direcciones[hueso]
        direccion = direccion / np.linalg.norm(direccion)
        posiciones[hijo] = posiciones[padre] + LARGOS_DEMO[hueso] * direccion

    # el giro del cuerpo completo, para que se note que la escena es 3D
    giro = 0.6 * np.sin(0.4 * tiempo)
    coseno, seno = np.cos(giro), np.sin(giro)
    for nombre, punto in posiciones.items():
        x, y, z = punto
        posiciones[nombre] = np.array([coseno * x + seno * z, y, -seno * x + coseno * z])

    ruido = fuente["ruido"]
    if ruido > 0:
        for nombre in posiciones:
            posiciones[nombre] = posiciones[nombre] + fuente["rng"].normal(0, ruido, 3)

    return posiciones, None, None


def nueva_fuente_camara(dispositivo, ruta_video=None):
    """Abre la camara (o un video) y el estimador de pose de mediapipe."""
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise SystemExit(f"{MENSAJE_SIN_DEPENDENCIAS}\n(detalle: {error})")

    if not RUTA_MODELO.exists():
        raise SystemExit(f"falta el modelo de pose en {RUTA_MODELO}")

    origen = str(ruta_video) if ruta_video else dispositivo
    captura = cv2.VideoCapture(origen)
    if not captura.isOpened():
        raise SystemExit(
            f"no se pudo abrir {'el video' if ruta_video else 'la camara'} ({origen}). "
            "Con --fuente demo el ejemplo corre igual."
        )

    opciones = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(RUTA_MODELO)),
        running_mode=vision.RunningMode.VIDEO,
    )

    return {
        "tipo": "camara",
        "cv2": cv2,
        "mp": mp,
        "captura": captura,
        "detector": vision.PoseLandmarker.create_from_options(opciones),
        "cuadro": 0,
        "es_video": ruta_video is not None,
    }


def _leer_camara(fuente, tiempo):
    cv2 = fuente["cv2"]
    mp = fuente["mp"]

    hay_cuadro, imagen_bgr = fuente["captura"].read()
    if not hay_cuadro:
        if fuente["es_video"]:
            fuente["captura"].set(cv2.CAP_PROP_POS_FRAMES, 0)   # el video se repite
            hay_cuadro, imagen_bgr = fuente["captura"].read()
        if not hay_cuadro:
            return None, None, None

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    fuente["cuadro"] += 1

    resultado = fuente["detector"].detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb),
        int(fuente["cuadro"] * 1000 / 30),
    )

    if not resultado.pose_world_landmarks:
        return None, imagen_rgb, None

    mundo = resultado.pose_world_landmarks[0]
    posiciones = {}
    for nombre, indice in INDICE_DE_LANDMARK.items():
        punto = mundo[indice]
        posiciones[nombre] = np.array([punto.x, punto.y, punto.z]) * CONVERSION_DE_EJES
    for nombre, (primero, segundo) in PUNTOS_MEDIOS.items():
        a, b = mundo[primero], mundo[segundo]
        medio = np.array([(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2])
        posiciones[nombre] = medio * CONVERSION_DE_EJES

    # las coordenadas normalizadas sirven para calcar los puntos sobre la imagen
    imagen_landmarks = resultado.pose_landmarks[0]
    calco = {
        nombre: (imagen_landmarks[indice].x, imagen_landmarks[indice].y)
        for nombre, indice in INDICE_DE_LANDMARK.items()
    }
    for nombre, (primero, segundo) in PUNTOS_MEDIOS.items():
        a, b = imagen_landmarks[primero], imagen_landmarks[segundo]
        calco[nombre] = ((a.x + b.x) / 2, (a.y + b.y) / 2)

    return posiciones, imagen_rgb, calco


def leer(fuente, tiempo):
    """Devuelve (posiciones, imagen, calco 2D). Cualquiera puede ser None."""
    if fuente["tipo"] == "demo":
        return _leer_demo(fuente, tiempo)
    return _leer_camara(fuente, tiempo)


def cerrar(fuente):
    if fuente["tipo"] == "camara":
        fuente["captura"].release()
        fuente["detector"].close()
