"""
Ejemplo atomico de la unidad de animacion y esqueletos: Linear Blend Skinning
sobre un esqueleto minimo construido a mano, sin cargar ningun archivo.

Un tubo vertical esta gobernado por una cadena de tres huesos. El hueso de la
base esta fijo; los otros dos se rotan con el teclado. Cada vertice del tubo
recibe dos influencias segun su altura, con una banda de mezcla suave alrededor
de cada articulacion. La posicion deformada de un vertice es

    v' = sum_j w_j * M_j * inverse_bind(j) * v,

donde M_j es la pose global actual del hueso j e inverse_bind(j) lleva el
vertice del espacio de objeto en reposo al espacio local del hueso. Esa es la
formula del skinning lineal; aqui queda a la vista sin el ruido de un loader
glTF, de materiales ni de la comparacion con otros metodos (eso vive en el
ejemplo rico `skinning`).

El color de la malla mezcla un color por hueso con los mismos pesos del
skinning, de modo que se ve que hueso domina cada region y como las bandas de
mezcla se doblan junto al esqueleto.

Teclas:
    a / z     rota el hueso 1 (articulacion media)    - / +
    s / x     rota el hueso 2 (articulacion superior)  - / +
    w         alterna malla rellena / wireframe
    c         muestra/oculta volumenes de colision (OBB por hueso vs AABB global)
    r         vuelve a la pose de reposo
"""

from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
from pyglet.gl import GLfloat

import grafica.transformations as tr
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline


# Alturas de las tres articulaciones a lo largo del tubo (en reposo). El tubo
# se extiende de y = 0 a y = ALTURA_TUBO. Las articulaciones quedan en 0, 1, 2:
# la base esta fija y las dos de arriba se rotan. Es importante que las dos
# articulaciones rotables tengan tubo por encima de su pivote (la base de la
# punta llega a y = 3), porque rotar una articulacion sin geometria sobre su
# pivote enrosca la malla alrededor del extremo y no se condice con el hueso.
ALTURA_TUBO = 3.0
ALTURA_JUNTAS = np.array([0.0, 1.0, 2.0], dtype=np.float32)
RADIO_TUBO = 0.35

# Semi-ancho de la banda de mezcla alrededor de cada articulacion. Fuera de la
# banda el peso es 1 para un solo hueso (nucleo rigido): ese tramo se mueve
# exactamente con su hueso, igual que un miembro rigido. Asi la OBB pegada al
# hueso contiene de verdad la geometria deformada (ver overlay de colision).
ANCHO_BANDA = 0.3

# Un color por hueso. La malla se tinta con la mezcla ponderada de estos
# colores, asi cada region anuncia su hueso dominante.
COLORES_HUESO = np.array(
    [
        [0.90, 0.30, 0.30],  # hueso 0, base (rojo)
        [0.30, 0.80, 0.40],  # hueso 1, medio (verde)
        [0.35, 0.55, 0.95],  # hueso 2, superior (azul)
    ],
    dtype=np.float32,
)

PASO_ANGULO = np.radians(9.0)
ANGULO_MAXIMO = np.radians(120.0)


def poses_globales(angulo_hueso_1, angulo_hueso_2):
    """
    Poses globales de los tres huesos por cinematica directa. El hueso 0 es la
    raiz fija; cada hijo compone la traslacion a su articulacion con la rotacion
    local sobre el eje Z. Con ambos angulos en cero se recupera la pose de
    reposo (los huesos quedan alineados en el eje Y).
    """
    salto = ALTURA_JUNTAS[1] - ALTURA_JUNTAS[0]
    m0 = tr.identity()
    m1 = m0 @ tr.translate(0.0, salto, 0.0) @ tr.rotationZ(angulo_hueso_1)
    m2 = m1 @ tr.translate(0.0, salto, 0.0) @ tr.rotationZ(angulo_hueso_2)
    return [m0, m1, m2]


# inverse bind: lleva un vertice del espacio de objeto en reposo al espacio
# local de cada hueso. Es la inversa de la pose global de reposo.
POSES_REPOSO = poses_globales(0.0, 0.0)
INVERSE_BIND = [np.linalg.inv(m) for m in POSES_REPOSO]


def matrices_de_skinning(angulo_hueso_1, angulo_hueso_2):
    """Una matriz por hueso: pose global actual por su inverse bind."""
    globales = poses_globales(angulo_hueso_1, angulo_hueso_2)
    return [globales[j] @ INVERSE_BIND[j] for j in range(3)], globales


# Aristas de un cubo: pares de esquinas que difieren en un solo eje. La esquina
# i codifica en sus bits que extremo toma en x (bit 0), y (bit 1), z (bit 2).
_ARISTAS_CAJA = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _esquinas_caja(minimo, maximo):
    """Las ocho esquinas de una caja a partir de sus extremos."""
    esquinas = np.empty((8, 3), dtype=np.float32)
    for i in range(8):
        esquinas[i] = (
            maximo[0] if i & 1 else minimo[0],
            maximo[1] if i & 2 else minimo[1],
            maximo[2] if i & 4 else minimo[2],
        )
    return esquinas


def _lineas_caja(esquinas):
    """Convierte ocho esquinas en los 24 vertices de sus doce aristas."""
    return np.array(
        [esquinas[indice] for arista in _ARISTAS_CAJA for indice in arista],
        dtype=np.float32,
    )


def _influencias_por_altura(altura):
    """
    Dos huesos influyen cada vertice, elegidos por su altura. Entre dos juntas
    cada hueso domina un tramo con peso 1 (nucleo rigido) y los pesos solo se
    mezclan en una banda angosta (semi-ancho ANCHO_BANDA) alrededor de cada
    articulacion. Devuelve indices y pesos rellenados a cuatro influencias (las
    dos sobrantes con peso cero) para que el vertex shader siempre sume el mismo
    numero de terminos.
    """
    b = ANCHO_BANDA
    junta_media, junta_alta = ALTURA_JUNTAS[1], ALTURA_JUNTAS[2]
    if altura < junta_media - b:
        # nucleo rigido del hueso 0
        return (0, 1, 0, 0), (1.0, 0.0, 0.0, 0.0)
    if altura <= junta_media + b:
        # banda alrededor de la articulacion media: hueso 0 -> hueso 1
        t = (altura - (junta_media - b)) / (2.0 * b)
        return (0, 1, 0, 0), (1.0 - t, t, 0.0, 0.0)
    if altura < junta_alta - b:
        # nucleo rigido del hueso 1
        return (1, 2, 0, 0), (1.0, 0.0, 0.0, 0.0)
    if altura <= junta_alta + b:
        # banda alrededor de la articulacion superior: hueso 1 -> hueso 2
        t = (altura - (junta_alta - b)) / (2.0 * b)
        return (1, 2, 0, 0), (1.0 - t, t, 0.0, 0.0)
    # nucleo rigido del hueso 2 (gobierna hasta la punta)
    return (1, 2, 0, 0), (0.0, 1.0, 0.0, 0.0)


def construir_tubo(n_anillos=49, n_segmentos=24):
    """
    Genera un tubo de triangulos alrededor del eje Y. Devuelve posiciones,
    normales (radiales), influencias de hueso, pesos e indices. Cada vertice
    obtiene sus pesos de skinning segun su altura.
    """
    alturas = np.linspace(0.0, ALTURA_TUBO, n_anillos)
    angulos = np.linspace(0.0, 2.0 * np.pi, n_segmentos, endpoint=False)

    posiciones = []
    normales = []
    juntas = []
    pesos = []
    for altura in alturas:
        indices_junta, pesos_junta = _influencias_por_altura(altura)
        for angulo in angulos:
            direccion = np.array([np.cos(angulo), 0.0, np.sin(angulo)], dtype=np.float32)
            posiciones.append([RADIO_TUBO * direccion[0], altura, RADIO_TUBO * direccion[2]])
            normales.append(direccion)
            juntas.append(indices_junta)
            pesos.append(pesos_junta)

    indices = []
    for anillo in range(n_anillos - 1):
        for segmento in range(n_segmentos):
            siguiente = (segmento + 1) % n_segmentos
            a = anillo * n_segmentos + segmento
            b = anillo * n_segmentos + siguiente
            c = (anillo + 1) * n_segmentos + segmento
            d = (anillo + 1) * n_segmentos + siguiente
            indices.extend([a, c, b, b, c, d])

    return (
        np.array(posiciones, dtype=np.float32),
        np.array(normales, dtype=np.float32),
        np.array(juntas, dtype=np.float32),
        np.array(pesos, dtype=np.float32),
        np.array(indices, dtype=np.uint32),
    )


@click.command("esqueleto_lbs", short_help="Linear Blend Skinning sobre un esqueleto minimo de 3 huesos")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=720)
def esqueleto_lbs(width, height):
    window = pyglet.window.Window(width, height, caption="esqueleto + LBS")

    GL.glClearColor(0.09, 0.10, 0.13, 1.0)

    posiciones, normales, juntas, pesos, indices = construir_tubo()
    print(
        f"[esqueleto_lbs] tubo: vertices={len(posiciones)} triangulos={len(indices) // 3} "
        f"huesos={len(ALTURA_JUNTAS)}"
    )

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )
    malla = pipeline.vertex_list_indexed(len(posiciones), GL.GL_TRIANGLES, indices.tolist())
    malla.position[:] = posiciones.flatten()
    malla.normal[:] = normales.flatten()
    malla.joints[:] = juntas.flatten()
    malla.weights[:] = pesos.flatten()

    # esqueleto: tres segmentos de hueso (lineas) y tres articulaciones
    # (puntos). El ultimo segmento va de la articulacion 2 a la punta del tubo,
    # asi rotar el hueso 2 mueve el esqueleto y la malla a la vez. Las
    # posiciones se reescriben por cuadro; el color es fijo por hueso.
    skeleton_pipeline = load_pipeline(
        Path(__file__).parent / "skeleton_vertex_program.glsl",
        Path(__file__).parent / "skeleton_fragment_program.glsl",
    )
    huesos_gpu = skeleton_pipeline.vertex_list(6, GL.GL_LINES)
    huesos_gpu.color[:] = np.array(
        [
            COLORES_HUESO[0], COLORES_HUESO[1],  # base -> medio
            COLORES_HUESO[1], COLORES_HUESO[2],  # medio -> superior
            COLORES_HUESO[2], COLORES_HUESO[2],  # superior -> punta
        ],
        dtype=np.float32,
    ).flatten()
    juntas_gpu = skeleton_pipeline.vertex_list(3, GL.GL_POINTS)
    juntas_gpu.color[:] = COLORES_HUESO.flatten()

    # Overlay opt-in de colision (tecla c). Muestra el problema de la unidad de
    # colisiones sobre una malla deformable: una OBB pegada a cada hueso (caja
    # fija en el espacio local del hueso, se mueve con su pose y queda ajustada)
    # frente a la AABB de toda la malla recalculada cada cuadro (se infla al
    # doblar). Cada vertice se asigna a su hueso dominante para armar las OBB.
    hueso_dominante = juntas[np.arange(len(juntas)), np.argmax(pesos, axis=1)].astype(int)
    obb_local_min = []
    obb_local_max = []
    for j in range(3):
        reposo_del_hueso = posiciones[hueso_dominante == j]
        homogeneo = np.hstack([reposo_del_hueso, np.ones((len(reposo_del_hueso), 1), dtype=np.float32)])
        local = (INVERSE_BIND[j] @ homogeneo.T).T[:, :3]
        obb_local_min.append(local.min(axis=0))
        obb_local_max.append(local.max(axis=0))

    # 4 cajas (3 OBB por hueso + 1 AABB global), 12 aristas cada una
    cajas_gpu = skeleton_pipeline.vertex_list(4 * len(_ARISTAS_CAJA) * 2, GL.GL_LINES)
    color_aabb = np.array([0.85, 0.85, 0.85], dtype=np.float32)
    cajas_gpu.color[:] = np.vstack(
        [np.tile(COLORES_HUESO[j] * 0.7, (24, 1)) for j in range(3)]
        + [np.tile(color_aabb, (24, 1))]
    ).astype(np.float32).flatten()

    # camara fija: un tres cuartos que muestra el tubo entero al doblarse.
    centro = np.array([0.0, ALTURA_TUBO * 0.5, 0.0], dtype=np.float32)
    eye = np.array([3.4, ALTURA_TUBO * 0.5 + 0.4, 4.6], dtype=np.float32)
    view = tr.lookAt(eye, centro, np.array([0.0, 1.0, 0.0]))
    projection = tr.perspective(45.0, width / height, 0.01, 50.0)
    light_direction = np.array([0.4, 0.7, 0.6], dtype=np.float32)
    ambient_strength = 0.35

    # buffers ctypes para los uniformes-array (pyglet exige tipo y forma exactos)
    Mat4Array = (GLfloat * 16) * 3
    skin_buffer = Mat4Array()
    skin_view = np.frombuffer(skin_buffer, dtype=np.float32).reshape(3, 16)

    Vec3Array = (GLfloat * 3) * 3
    joint_colors_buffer = Vec3Array()
    np.frombuffer(joint_colors_buffer, dtype=np.float32).reshape(3, 3)[:] = COLORES_HUESO

    panel = (
        InfoPanel(x=14, y_top=height - 26, background=(18, 18, 22))
        .add("hueso1")
        .add("hueso2")
        .add("modo")
        .add("colision")
        .footer("a z hueso 1   s x hueso 2   w wireframe   c colisiones   r reposo")
    )

    state = {
        "angulo_hueso_1": 0.0,
        "angulo_hueso_2": 0.0,
        "wireframe": False,
        "obb": False,
    }

    def actualizar_cajas_colision(skin_matrices, globales):
        lineas = []
        # una OBB por hueso: la caja local fija, transformada por la pose actual
        # del hueso. Sigue al miembro sin recalcularse, como la OBB rigida.
        for j in range(3):
            esquinas_locales = _esquinas_caja(obb_local_min[j], obb_local_max[j])
            homogeneo = np.hstack([esquinas_locales, np.ones((8, 1), dtype=np.float32)])
            esquinas_mundo = (globales[j] @ homogeneo.T).T[:, :3]
            lineas.append(_lineas_caja(esquinas_mundo))

        # AABB de toda la malla deformada: hay que re-skinnear los vertices en la
        # CPU y tomar el min/max en mundo cada cuadro. Se infla al doblar.
        homogeneo = np.hstack([posiciones, np.ones((len(posiciones), 1), dtype=np.float32)])
        matrices = np.stack(skin_matrices)
        deformadas = np.zeros((len(posiciones), 3), dtype=np.float32)
        for influencia in range(2):  # solo las dos influencias con peso > 0
            indices = juntas[:, influencia].astype(int)
            peso = pesos[:, influencia][:, None]
            aporte = np.einsum("nij,nj->ni", matrices[indices], homogeneo)[:, :3]
            deformadas += (peso * aporte).astype(np.float32)
        esquinas = _esquinas_caja(deformadas.min(axis=0), deformadas.max(axis=0))
        lineas.append(_lineas_caja(esquinas))

        cajas_gpu.position[:] = np.concatenate(lineas).flatten()

    def apply_state():
        skin_matrices, globales = matrices_de_skinning(
            state["angulo_hueso_1"], state["angulo_hueso_2"]
        )
        # column-major para glUniformMatrix4fv (transpose = GL_FALSE)
        skin_view[:] = np.array(skin_matrices, dtype=np.float32).transpose(0, 2, 1).reshape(3, 16)

        # posiciones de las articulaciones desde la pose global de cada hueso
        origenes = np.array([m[:3, 3] for m in globales], dtype=np.float32)
        # la punta del tubo es un punto fijo en el espacio local del hueso 2
        # (una unidad por encima de el en reposo); sigue su rotacion
        altura_punta_local = ALTURA_TUBO - ALTURA_JUNTAS[2]
        punta = (globales[2] @ np.array([0.0, altura_punta_local, 0.0, 1.0], dtype=np.float32))[:3]
        huesos_gpu.position[:] = np.array(
            [origenes[0], origenes[1], origenes[1], origenes[2], origenes[2], punta],
            dtype=np.float32,
        ).flatten()
        juntas_gpu.position[:] = origenes.flatten()

        if state["obb"]:
            actualizar_cajas_colision(skin_matrices, globales)

        panel["hueso1"] = f"hueso 1 (medio):    {np.degrees(state['angulo_hueso_1']):6.1f} deg"
        panel["hueso2"] = f"hueso 2 (superior): {np.degrees(state['angulo_hueso_2']):6.1f} deg"
        panel["modo"] = f"malla:   {'wireframe' if state['wireframe'] else 'relleno'}"
        panel["colision"] = (
            "colision: OBB por hueso vs AABB global" if state["obb"] else "colision: off (c)"
        )
        print(
            f"[esqueleto_lbs] hueso1={np.degrees(state['angulo_hueso_1']):.1f} "
            f"hueso2={np.degrees(state['angulo_hueso_2']):.1f} "
            f"wireframe={state['wireframe']} obb={state['obb']}"
        )

    apply_state()

    @window.event
    def on_draw():
        GL.glEnable(GL.GL_DEPTH_TEST)
        window.clear()

        pipeline.use()
        pipeline["view"] = view.reshape(16, 1, order="F")
        pipeline["projection"] = projection.reshape(16, 1, order="F")
        pipeline["light_direction"] = light_direction.reshape(3, 1, order="F")
        pipeline["ambient_strength"] = ambient_strength
        pipeline["skin_matrices"] = skin_buffer
        pipeline["joint_colors"] = joint_colors_buffer
        # el modo wireframe solo afecta a los poligonos (el tubo). Lo dejamos
        # en relleno enseguida para que el esqueleto (lineas y puntos) y el
        # panel 2D, que es geometria rellena, no salgan tambien como aristas
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if state["wireframe"] else GL.GL_FILL)
        malla.draw(GL.GL_TRIANGLES)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        # el esqueleto se dibuja sin depth test, como una radiografia sobre la
        # malla, para que se vea que la deformacion sigue a los huesos
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(2.0)
        GL.glPointSize(9.0)
        skeleton_pipeline.use()
        skeleton_pipeline["view"] = view.reshape(16, 1, order="F")
        skeleton_pipeline["projection"] = projection.reshape(16, 1, order="F")
        huesos_gpu.draw(GL.GL_LINES)
        juntas_gpu.draw(GL.GL_POINTS)

        if state["obb"]:
            GL.glLineWidth(1.0)
            cajas_gpu.draw(GL.GL_LINES)

        with ui_overlay():
            panel.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.A:
            state["angulo_hueso_1"] = max(-ANGULO_MAXIMO, state["angulo_hueso_1"] - PASO_ANGULO)
            apply_state()
        elif symbol == pyglet.window.key.Z:
            state["angulo_hueso_1"] = min(ANGULO_MAXIMO, state["angulo_hueso_1"] + PASO_ANGULO)
            apply_state()
        elif symbol == pyglet.window.key.S:
            state["angulo_hueso_2"] = max(-ANGULO_MAXIMO, state["angulo_hueso_2"] - PASO_ANGULO)
            apply_state()
        elif symbol == pyglet.window.key.X:
            state["angulo_hueso_2"] = min(ANGULO_MAXIMO, state["angulo_hueso_2"] + PASO_ANGULO)
            apply_state()
        elif symbol == pyglet.window.key.W:
            state["wireframe"] = not state["wireframe"]
            apply_state()
        elif symbol == pyglet.window.key.C:
            state["obb"] = not state["obb"]
            apply_state()
        elif symbol == pyglet.window.key.R:
            state["angulo_hueso_1"] = 0.0
            state["angulo_hueso_2"] = 0.0
            apply_state()

    pyglet.app.run()
