"""Microjuego de ejemplo: piruetas de un zorzal.

Arriba de la ventana aparece una secuencia de flechas. Cada flecha presionada
en orden hace que el zorzal ejecute una maniobra. Se gana al completar la
secuencia. Una flecha equivocada lo marea y lo hace caer.

El zorzal ocupa una celda de una grilla de 3 x 3 (tres alturas y tres
carriles) y la maniobra depende de la celda en la que está, de modo que una
misma flecha produce maniobras distintas según las flechas anteriores:

- Arriba sube una altura. Si ya está en la más alta, da un rizo.
- Abajo desde la altura más alta es un planeo y desde la altura media es una
  picada. Si ya está en la más baja, da un rizo invertido.
- Izquierda y derecha lo cambian de carril con un ladeo. Si ya está en el
  carril del borde, da un tonel.

El movimiento tiene tres capas:

1. La posición sigue a la celda con un resorte amortiguado, así que el zorzal
   acelera, se pasa un poco y vuelve. La orientación se calcula a partir de
   la velocidad: levanta el pico cuando sube, lo baja cuando desciende e
   inclina las alas hacia el lado al que se mueve.
2. El rizo y el tonel son una rotación completa del modelo (unidad 08), que
   se suma a la orientación anterior.
3. La pose del esqueleto (unidad 21) es una mezcla de varias animaciones del
   modelo: el aleteo, el aleteo ladeado, las alas extendidas y las alas
   plegadas. El peso de cada una depende del movimiento. La mezcla se hace
   sobre la traslación, la rotación y la escala local de cada hueso, y con
   el resultado se recalculan las matrices del esqueleto en cada cuadro.

El vertex program mueve cada vértice con hasta cuatro huesos y mezcla sus
matrices según los pesos del vértice (linear blend skinning).
"""

import random
from pathlib import Path

import numpy as np
import pyglet
import pyglet.gl as GL
from PIL import Image

import grafica.transformations as tr
from grafica.background import GradientBackground
from grafica.gltf import GltfModel
from grafica.textures import texture_2D_setup
from grafica.utils import load_pipeline

ROL = "microjuego"
INSTRUCCION = "¡Haz piruetas!"
GANA_SI_SOBREVIVE = False
DURACION = 5.0

CARPETA = Path(__file__).parent
ASSETS = CARPETA.parents[2] / "assets"
ANCHO, ALTO = 800, 600

LARGO_SECUENCIA = {1: 4, 2: 6, 3: 8}
FLECHAS = ("izquierda", "arriba", "abajo", "derecha")
GIRO_TECLA = {"arriba": 0, "derecha": 90, "abajo": 180, "izquierda": 270}  # grados, en sentido horario
# El eje de la grilla que cambia cada flecha (0 es el carril, 1 es la altura)
# y en qué sentido. El ala izquierda del zorzal está en +x.
EJES = {"izquierda": (0, 1), "derecha": (0, -1), "arriba": (1, 1), "abajo": (1, -1)}
PASO = np.array([0.5, 0.3, 0.0])  # separación entre carriles y entre alturas

# Frecuencia natural (rad/s) del resorte que lleva al zorzal a su celda. Con
# una frecuencia baja el movimiento es lento, como en el planeo.
RIGIDEZ = {"ascenso": 9.0, "planeo": 4.5, "picada": 12.0, "ladeo": 9.0}
# Con una razón de amortiguación menor que 1, el zorzal se pasa un poco de la
# celda antes de detenerse.
AMORTIGUACION = 0.7
# La rapidez con la que avanza el zorzal. La cámara avanza con él, así que
# solo se usa para calcular la orientación y para mover las estelas.
RAPIDEZ = 2.5
ALABEO_POR_RAPIDEZ = 0.15  # radianes de inclinación por cada unidad de rapidez lateral

DURACION_GIRO = {"rizo": 0.60, "tonel": 0.45}  # segundos, cuando no hay otros giros en espera
RADIO_RIZO = 0.3
RADIO_TONEL = 0.12
GRAVEDAD = 6.0

TEXTURAS = {"M_Blackbird": "zorzal.png", "M_BlackbirdEyes": "zorzal.png", "M_BlackbirdWings": "plumas.png"}
# Las animaciones del modelo que se mezclan. De las que son una postura fija
# se usa solo el primer cuadro.
CLIPS = {
    "vuelo": ("Anim_Blackbird_Fly", False),
    "izquierda": ("Anim_Blackbird_FlyLeft", False),
    "derecha": ("Anim_Blackbird_FlyRight", False),
    "caida": ("Anim_Blackbird_Fallin", False),
    "extendidas": ("Anim_Blackbird_Fallin", True),
    "plegadas": ("Anim_Blackbird_Idle", True),
}
HUESOS_ALAS = ("_Clav_", "_Humerus_", "_Ulna_", "_Aula", "_PF", "_SF")
CUADROS_POR_SEGUNDO = 60
RAPIDEZ_MEZCLA = 14.0  # los pesos de las animaciones recorren el 63% de lo que les falta en 1/14 de segundo

# El zorzal mira hacia +z, extiende las alas en el eje x y su centro queda en
# (0, 0.35, -0.1). La cámara lo ve desde atrás y desde un costado.
CENTRAR = tr.translate(0.0, -0.35, 0.1)
VISTA = tr.lookAt(np.array([-3.0, 1.5, -2.5]), np.array([0.0, 0.2, 0.3]), np.array([0.0, 1.0, 0.0]))
PROYECCION = tr.perspective(45, ANCHO / ALTO, 0.1, 50.0)
DIRECCION_LUZ = np.array([0.3, 0.8, -0.5]) / np.linalg.norm([0.3, 0.8, -0.5])

# las puntas de las alas extendidas, en las coordenadas del modelo; de ahí salen las estelas
PUNTAS = np.array([[0.8, 0.6, 0.25, 1.0], [-0.8, 0.6, 0.25, 1.0]])
LARGO_ESTELA = 24  # puntos

SONIDOS = {
    nombre: pyglet.media.load(str(CARPETA / "sonidos" / f"{nombre}.wav"), streaming=False)
    for nombre in ("silbido", "trino", "error")
}


def muestrear(modelo, animacion, tiempo):
    """La traslación (3), la rotación (4, un cuaternión) y la escala (3) local de cada nodo, en un arreglo de N x 10."""
    valores = modelo.sample_animation(animacion, tiempo)
    filas = []
    for nodo in modelo.nodes:
        valor = valores.get(nodo.index, {})
        filas.append(np.concatenate([valor.get("translation", nodo.translation),
                                     valor.get("rotation", nodo.rotation),
                                     valor.get("scale", nodo.scale)]))
    return np.array(filas, dtype=np.float32)


def mezclar(capas):
    """Promedio ponderado de varias poses. Cada peso es un número o una columna con un peso por nodo."""
    referencia = capas[0][0][:, 3:7]
    suma, total = 0.0, 0.0
    for pose, peso in capas:
        # q y -q son la misma rotación; para promediar dos cuaterniones tienen que estar en el mismo hemisferio
        mismo_lado = np.where((pose[:, 3:7] * referencia).sum(axis=1, keepdims=True) < 0.0, -1.0, 1.0)
        pose = pose.copy()
        pose[:, 3:7] *= mismo_lado
        suma = suma + peso * pose
        total = total + peso
    return suma / total


def matrices_de_piel(esqueleto, pose):
    """La matriz de cada hueso: su pose global por la inversa de su pose de reposo."""
    traslacion, rotacion, escala = pose[:, :3], pose[:, 3:7], pose[:, 7:]
    # el promedio de cuaterniones unitarios no es unitario
    x, y, z, w = (rotacion / np.linalg.norm(rotacion, axis=1, keepdims=True)).T
    locales = np.zeros((len(pose), 4, 4), dtype=np.float32)
    locales[:, 0, :3] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1)
    locales[:, 1, :3] = np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1)
    locales[:, 2, :3] = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1)
    locales[:, :3, :3] *= escala[:, None, :]
    locales[:, :3, 3] = traslacion
    locales[:, 3, 3] = 1.0

    # la pose global de un nodo es la de su padre por su pose local; el orden asegura que el padre ya esté calculado
    globales = locales.copy()
    for nodo, padre in esqueleto["orden"]:
        if padre is not None:
            globales[nodo] = globales[padre] @ locales[nodo]
    return globales[esqueleto["huesos"]] @ esqueleto["inversas"]


def preparar(ventana):
    modelo = GltfModel(ASSETS / "zorzal-animation" / "gltf" / "zorzal.gltf")
    nodo = next(nodo for nodo in modelo.nodes if nodo.skin_index is not None)
    piel = modelo.skins[nodo.skin_index]
    programa = load_pipeline(CARPETA / "vertex_program.glsl", CARPETA / "fragment_program.glsl")

    texturas, primitivas = {}, []
    for primitiva in modelo.meshes[nodo.mesh_index]:
        archivo = TEXTURAS[modelo.materials[primitiva.material_index]["name"]]
        if archivo not in texturas:
            # La imagen original mide 4096 x 4096, pero el zorzal ocupa unos
            # cientos de píxeles en la ventana. Sin mip-maps, una textura mucho
            # más grande que su tamaño en pantalla parpadea (unidad 11).
            imagen = Image.open(ASSETS / archivo).reduce(4)
            # glTF pone el origen de las coordenadas uv en la esquina superior izquierda
            texturas[archivo] = texture_2D_setup(imagen, flip_top_bottom=False)
        lista = programa.vertex_list_indexed(len(primitiva.positions), GL.GL_TRIANGLES, primitiva.indices.tolist())
        lista.position[:] = primitiva.positions.flatten()
        lista.normal[:] = primitiva.normals.flatten()
        lista.uv[:] = primitiva.uvs.flatten()
        lista.joints[:] = primitiva.joints.astype(np.float32).flatten()
        lista.weights[:] = primitiva.weights.flatten()
        primitivas.append({"lista": lista, "textura": texturas[archivo], "plumas": archivo == "plumas.png"})
    # las plumas tienen bordes semitransparentes, así que van al final (unidad 12)
    primitivas.sort(key=lambda primitiva: primitiva["plumas"])

    # los nodos del esqueleto, ordenados de modo que cada padre quede antes que sus hijos
    orden = []

    def recorrer(indice, padre):
        orden.append((indice, padre))
        for hijo in modelo.nodes[indice].children:
            recorrer(hijo, indice)

    for raiz in modelo.scene_roots:
        recorrer(raiz, None)
    esqueleto = {
        "orden": orden,
        "huesos": list(piel.joints),
        "inversas": piel.inverse_bind_matrices,
        # una columna con 1 en los huesos de las alas y 0 en el resto
        "alas": np.array([[float(any(parte in nodo.name for parte in HUESOS_ALAS))] for nodo in modelo.nodes]),
        # pyglet recibe un arreglo uniform como un arreglo de ctypes
        "Huesos": (GL.GLfloat * 16) * len(piel.joints),
    }

    # las poses locales de cada cuadro de las animaciones se muestrean una sola vez
    clips = {}
    for clave, (nombre, es_postura) in CLIPS.items():
        animacion = next(indice for indice, a in enumerate(modelo.animations) if a["name"] == nombre)
        cuadros = 1 if es_postura else round(modelo.animations[animacion]["duration"] * CUADROS_POR_SEGUNDO)
        clips[clave] = [muestrear(modelo, animacion, cuadro / CUADROS_POR_SEGUNDO) for cuadro in range(cuadros)]

    # nubes de tres círculos, que pasan a distintas rapideces según su tamaño
    nubes = pyglet.graphics.Batch()
    azar = random.Random(4)
    recursos_nubes = []
    for _ in range(7):
        escala = azar.uniform(0.5, 1.4)
        circulos = [pyglet.shapes.Circle(0, 0, radio * escala, color=(255, 255, 255, 255), batch=nubes)
                    for radio in (28, 38, 26)]
        recursos_nubes.append({"x": azar.uniform(0, ANCHO + 200), "y": azar.uniform(60, ALTO - 140),
                               "escala": escala, "circulos": circulos})

    estelas = pyglet.graphics.Batch()
    # los tramos de las estelas, que se adelgazan hacia atrás
    lineas_estelas = [[pyglet.shapes.Line(0, 0, 0, 0, 4.0 - 3.0 * tramo / LARGO_ESTELA, color=(255, 255, 255, 255),
                                          batch=estelas) for _ in range(2)]
                      for tramo in range(LARGO_ESTELA - 1)]

    teclas = pyglet.graphics.Batch()
    imagen_tecla = pyglet.image.load(str(CARPETA / "imagenes" / "tecla.png")).get_texture()
    imagen_tecla.anchor_x, imagen_tecla.anchor_y = 22, 22
    sprites = [pyglet.sprite.Sprite(imagen_tecla, batch=teclas) for _ in range(max(LARGO_SECUENCIA.values()))]

    return {
        "programa": programa,
        "primitivas": primitivas,
        "esqueleto": esqueleto,
        "clips": clips,
        "estelas": estelas,
        "lineas_estelas": lineas_estelas,
        "cielo": GradientBackground(bottom_color=(0.98, 0.85, 0.70), top_color=(0.35, 0.60, 0.92)),
        "nubes": nubes,
        "recursos_nubes": recursos_nubes,
        "teclas": teclas,
        "sprites": sprites,
    }


def iniciar(dificultad, espacio):
    # este microjuego no usa el espacio de pymunk
    return {
        "secuencia": [random.choice(FLECHAS) for _ in range(LARGO_SECUENCIA[dificultad])],
        "aciertos": 0,
        "celda": [0, 0],  # carril y altura (-1, 0 o 1) a la que va el zorzal
        "posicion": np.zeros(3),
        "velocidad": np.zeros(3),
        "rigidez": np.array([RIGIDEZ["ladeo"], RIGIDEZ["ascenso"], 0.0]),  # del resorte, por eje
        "descenso": "planeo",  # el tipo del último descenso
        "alabeo": 0.0,
        "giros": [],  # los rizos y toneles por hacer; el primero es el que está en curso
        "progreso": 0.0,  # del giro en curso, entre 0 y 1
        "pesos": {clave: float(clave == "vuelo") for clave in CLIPS},  # de cada animación en la pose
        "aleteo": 0.0,  # segundos de la animación de aleteo
        "distancia": 0.0,  # la que avanzaron las nubes
        # por cuadro, la posición de las dos puntas de las alas y la intensidad de la estela; el más reciente va primero
        "estelas": [],
        "tiempo": 0.0,
        "caida": None,  # segundos desde el error
        "teclas_antes": {flecha: False for flecha in FLECHAS},
        "resultado": None,
    }


def presionar(estado, flecha):
    """Inicia la maniobra que produce la flecha desde la celda actual."""
    eje, signo = EJES[flecha]
    if estado["celda"][eje] == signo:
        # no quedan celdas en ese sentido: gira en el lugar
        estado["giros"].append({"tipo": "tonel" if eje == 0 else "rizo", "signo": signo})
        return
    if eje == 0:
        tipo = "ladeo"
    elif signo > 0:
        tipo = "ascenso"
    else:
        tipo = estado["descenso"] = "planeo" if estado["celda"][1] == 1 else "picada"
    estado["celda"][eje] += signo
    estado["rigidez"][eje] = RIGIDEZ[tipo]


def giro_en_curso(estado):
    """Desplazamiento, ángulo del rizo, ángulo del tonel y cuánto extiende las alas el giro en curso."""
    if not estado["giros"] or estado["caida"] is not None:
        return np.zeros(3), 0.0, 0.0, 0.0
    giro, progreso = estado["giros"][0], estado["progreso"]
    signo = giro["signo"]
    if giro["tipo"] == "rizo":
        # Gira en torno al eje x, el de las alas, mientras recorre una
        # circunferencia. Entra y sale del rizo con rapidez, y la pierde en
        # la parte más alta. Aletea en la subida y extiende las alas después.
        angulo = 2.0 * np.pi * progreso + 0.35 * np.sin(2.0 * np.pi * progreso)
        desplazamiento = RADIO_RIZO * np.array([0.0, signo * (1.0 - np.cos(angulo)), np.sin(angulo)])
        alas = np.clip(5.0 * (progreso - 0.3), 0.0, 1.0) * np.clip(6.0 * (1.0 - progreso), 0.0, 1.0)
        return desplazamiento, -signo * angulo, 0.0, alas
    # El tonel gira en torno al eje z, el del cuerpo, mientras su centro
    # recorre una circunferencia pequeña. Con un ángulo negativo baja el ala
    # izquierda.
    angulo = 2.0 * np.pi * progreso - 0.5 * np.sin(2.0 * np.pi * progreso)
    desplazamiento = RADIO_TONEL * np.array([signo * np.sin(angulo), 1.0 - np.cos(angulo), 0.0])
    return desplazamiento, 0.0, -signo * angulo, min(1.0, 4.0 * progreso, 4.0 * (1.0 - progreso))


def transformacion_zorzal(estado):
    desplazamiento, rizo, tonel, _ = giro_en_curso(estado)
    velocidad = estado["velocidad"]
    # el vuelo sube y baja un poco
    vaiven = np.array([0.0, 0.05 * np.sin(3.0 * estado["tiempo"]), 0.0])
    # el pico apunta en la dirección en la que se mueve el zorzal, que además avanza con RAPIDEZ en z
    guinada = 0.5 * np.arctan2(velocidad[0], RAPIDEZ)
    cabeceo = -np.arctan2(velocidad[1], RAPIDEZ)
    alabeo = estado["alabeo"]
    if estado["caida"] is not None:
        alabeo += 5.0 * estado["caida"]  # cae girando sobre su eje
    return tr.matmul([
        tr.translate(*(estado["posicion"] + desplazamiento + vaiven)),
        tr.rotationY(guinada),
        tr.rotationX(cabeceo + rizo),
        tr.rotationZ(alabeo + tonel),
        CENTRAR,
    ])


def actualizar(estado, dt, entrada):
    estado["tiempo"] += dt
    # una flecha se acaba de presionar si está presionada ahora y no lo estaba en el cuadro anterior
    recien = [flecha for flecha in FLECHAS if entrada[flecha] and not estado["teclas_antes"][flecha]]
    estado["teclas_antes"] = {flecha: entrada[flecha] for flecha in FLECHAS}

    for flecha in recien:
        if estado["resultado"] is not None:
            break
        if flecha == estado["secuencia"][estado["aciertos"]]:
            estado["aciertos"] += 1
            presionar(estado, flecha)
            SONIDOS["silbido"].play()
            if estado["aciertos"] == len(estado["secuencia"]):
                estado["resultado"] = True
                SONIDOS["trino"].play()
        else:
            estado.update(resultado=False, caida=0.0)
            SONIDOS["error"].play()

    posicion, velocidad = estado["posicion"], estado["velocidad"]
    if estado["caida"] is not None:
        estado["caida"] += dt
        velocidad[1] -= GRAVEDAD * dt
    else:
        # Resorte amortiguado hacia la celda: la aceleración crece con la
        # distancia a la celda y disminuye con la velocidad.
        rigidez = estado["rigidez"]
        destino = PASO * (*estado["celda"], 0)
        velocidad += (rigidez**2 * (destino - posicion) - 2.0 * AMORTIGUACION * rigidez * velocidad) * dt
        if estado["giros"]:
            # si hay giros en espera, el que está en curso se apura
            estado["progreso"] += dt * len(estado["giros"]) / DURACION_GIRO[estado["giros"][0]["tipo"]]
            if estado["progreso"] >= 1.0:
                estado["giros"].pop(0)
                estado["progreso"] = 0.0
    posicion += velocidad * dt

    # las alas se inclinan hacia el lado al que se mueve; el alabeo sigue a su objetivo con retardo
    mezcla = 1.0 - np.exp(-RAPIDEZ_MEZCLA * dt)
    estado["alabeo"] += (-ALABEO_POR_RAPIDEZ * velocidad[0] - estado["alabeo"]) * mezcla

    # El peso de cada animación en la pose. Al descender deja de aletear: en
    # el planeo extiende las alas y en la picada las pliega. Al moverse de
    # lado, el aleteo se reemplaza por el aleteo ladeado.
    _, _, _, alas_giro = giro_en_curso(estado)
    baja = np.clip(-velocidad[1] / 0.8, 0.0, 1.0)
    objetivo = dict.fromkeys(CLIPS, 0.0)
    if estado["caida"] is not None:
        objetivo["caida"] = 1.0
    else:
        objetivo["extendidas"] = max(alas_giro, baja if estado["descenso"] == "planeo" else 0.0)
        objetivo["plegadas"] = 0.85 * baja * (1.0 - alas_giro) if estado["descenso"] == "picada" else 0.0
        aleteo = 1.0 - objetivo["extendidas"]
        ladeo = np.clip(estado["alabeo"] / 0.3, -1.0, 1.0)
        objetivo["izquierda"] = aleteo * max(-ladeo, 0.0)
        objetivo["derecha"] = aleteo * max(ladeo, 0.0)
        objetivo["vuelo"] = aleteo * (1.0 - abs(ladeo))
    for clave in CLIPS:
        estado["pesos"][clave] += (objetivo[clave] - estado["pesos"][clave]) * mezcla

    # aletea más rápido cuando sube
    estado["aleteo"] += dt * (1.0 + 0.6 * np.clip(velocidad[1], 0.0, 1.5))
    # las nubes pasan más rápido cuando desciende y más lento cuando sube
    rapidez = np.clip(1.0 - 0.4 * velocidad[1], 0.5, 1.8)
    estado["distancia"] += rapidez * dt

    # Las estelas quedan atrás a medida que el zorzal avanza. Las alas dejan
    # estela mientras están extendidas.
    for puntas, _ in estado["estelas"]:
        puntas[:, 2] -= RAPIDEZ * rapidez * dt
    puntas = (transformacion_zorzal(estado) @ PUNTAS.T).T[:, :3]
    estado["estelas"].insert(0, (puntas, estado["pesos"]["extendidas"]))
    del estado["estelas"][LARGO_ESTELA:]


def a_pixeles(punto):
    """La posición en la ventana de un punto del mundo."""
    x, y, _, w = PROYECCION @ VISTA @ np.array([*punto, 1.0])
    return (x / w + 1.0) / 2.0 * ANCHO, (y / w + 1.0) / 2.0 * ALTO


def dibujar(recursos, estado):
    recursos["cielo"].draw()
    for nube in recursos["recursos_nubes"]:
        # las nubes más grandes están más cerca y pasan más rápido
        x = (nube["x"] - 220 * nube["escala"] * estado["distancia"]) % (ANCHO + 200) - 100
        for circulo, (dx, dy) in zip(nube["circulos"], ((-34, -4), (0, 8), (34, -6))):
            circulo.position = (x + dx * nube["escala"], nube["y"] + dy * nube["escala"])
    recursos["nubes"].draw()

    # cada tramo de una estela une las posiciones de la punta del ala en dos cuadros seguidos
    estelas = estado["estelas"]
    for indice, lineas in enumerate(recursos["lineas_estelas"]):
        for ala, linea in enumerate(lineas):
            linea.visible = indice + 1 < len(estelas)
            if linea.visible:
                linea.x, linea.y = a_pixeles(estelas[indice][0][ala])
                linea.x2, linea.y2 = a_pixeles(estelas[indice + 1][0][ala])
                # se desvanece hacia atrás
                linea.opacity = int(220 * estelas[indice][1] * (1.0 - indice / LARGO_ESTELA))
    recursos["estelas"].draw()

    # la pose es la mezcla de un cuadro de cada animación, según su peso
    esqueleto, capas = recursos["esqueleto"], []
    for clave, cuadros in recursos["clips"].items():
        if clave == "caida":
            cuadro = min(int((estado["caida"] or 0.0) * CUADROS_POR_SEGUNDO), len(cuadros) - 1)
        else:
            cuadro = int(estado["aleteo"] * CUADROS_POR_SEGUNDO) % len(cuadros)
        peso = estado["pesos"][clave]
        # de la postura con las alas plegadas se usan solo los huesos de las alas
        capas.append((cuadros[cuadro], peso * esqueleto["alas"] if clave == "plegadas" else peso))
    # GLSL lee las matrices por columnas, así que se transponen
    huesos = matrices_de_piel(esqueleto, mezclar(capas)).transpose(0, 2, 1)

    programa = recursos["programa"]
    programa.use()
    programa["view"] = VISTA.reshape(16, 1, order="F")
    programa["projection"] = PROYECCION.reshape(16, 1, order="F")
    programa["transform"] = transformacion_zorzal(estado).reshape(16, 1, order="F")
    programa["direccion_luz"] = DIRECCION_LUZ
    programa["huesos"] = esqueleto["Huesos"].from_buffer_copy(np.ascontiguousarray(huesos, np.float32))
    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
    GL.glEnable(GL.GL_DEPTH_TEST)
    for primitiva in recursos["primitivas"]:
        # las plumas no escriben profundidad, para que no se tapen entre ellas
        GL.glDepthMask(GL.GL_FALSE if primitiva["plumas"] else GL.GL_TRUE)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, primitiva["textura"])
        primitiva["lista"].draw(GL.GL_TRIANGLES)
    GL.glDepthMask(GL.GL_TRUE)
    GL.glDisable(GL.GL_DEPTH_TEST)

    # la secuencia de flechas, centrada arriba: verdes las acertadas, roja la equivocada
    secuencia = estado["secuencia"]
    for indice, sprite in enumerate(recursos["sprites"]):
        sprite.visible = indice < len(secuencia)
        if not sprite.visible:
            continue
        sprite.position = (ANCHO / 2 + 56 * (indice - (len(secuencia) - 1) / 2), ALTO - 70, 0)
        sprite.rotation = GIRO_TECLA[secuencia[indice]]
        if indice < estado["aciertos"]:
            sprite.color, sprite.scale = (120, 210, 120), 0.9
        elif indice == estado["aciertos"] and estado["resultado"] is False:
            sprite.color, sprite.scale = (235, 90, 90), 1.15
        elif indice == estado["aciertos"]:
            sprite.color, sprite.scale = (255, 255, 255), 1.15
        else:
            sprite.color, sprite.scale = (255, 255, 255), 1.0
    recursos["teclas"].draw()
