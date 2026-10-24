"""Suggestive contours en tiempo real.

Reproduce el método de DeCarlo, Finkelstein, Rusinkiewicz y Santella (2003),
"Suggestive Contours for Conveying Shape", en la versión de su implementación
de referencia, rtsc (DeCarlo, Finkelstein y Rusinkiewicz 2004). Un suggestive
contour es un punto donde la curvatura radial kr vale cero y crece en la
dirección de vista: ahí aparecería un contorno con un giro pequeño de la
cámara.

Las curvaturas, sus direcciones y sus derivadas se calculan una vez al cargar
el modelo (curvature.py). En cada cuadro se calculan kr y la prueba de la
derivada por vértice, y se traza un segmento en cada cara donde kr cambia de
signo (contours.py). La superficie se dibuja blanca y solo tapa las líneas
ocultas.

Controles:
- V: campo pintado sobre la superficie (blanco, H, K o kr).
- C: suggestive contours. S: contornos. B: bordes y aristas vivas.
- T: lazos completos de kr = 0, en gris.
- F: desvanece los suggestive contours cerca del umbral.
- O / P: baja o sube el umbral de la derivada.
- , / .: menos o más iteraciones de difusión de las normales.
- 1: curvas donde la curvatura máxima vale cero. 2: donde la mínima vale cero.
- M: superficie. R: reinicia la cámara. Arrastrar rota, la rueda acerca.
"""
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import grafica.transformations as tr
from grafica.arcball import Arcball
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline, load_single_mesh

from .contours import (
    compute_view_quantities,
    extract_isolines,
    extract_silhouette_edges,
    radial_curvature,
)
from .curvature import (
    build_smoothing_operator,
    compute_corner_areas,
    compute_curvature_derivatives,
    compute_curvatures,
    compute_feature_size,
    compute_vertex_normals,
    smooth_normals,
)

CARPETA = Path(__file__).parent

# Modos de relleno de la malla: blanco (el dibujo de líneas) y tres campos de curvatura.
MODOS = ["blanco", "media (H)", "gaussiana (K)", "radial (kr)"]
MODO_BLANCO, MODO_MEDIA, MODO_GAUSSIANA, MODO_RADIAL = 0, 1, 2, 3

COLOR_TINTA = (0.0, 0.0, 0.0)
COLOR_LAZOS = (0.6, 0.6, 0.6)
COLOR_CRESTA = (0.75, 0.20, 0.15)  # rojo cálido, como el positivo del colormap
COLOR_VALLE = (0.18, 0.30, 0.65)  # azul frío, como el negativo del colormap

# Valores de rtsc. El umbral y el desvanecimiento se dividen por el cuadrado
# del tamaño de rasgo, así que valen igual para cualquier modelo. El ancho de
# las líneas está en píxeles.
UMBRAL_INICIAL = 0.01
DESVANECIMIENTO = 0.03
ANCHO_LINEA = 2.5
# rtsc parte sin difusión; el paper de 2003 suavizaba la malla antes de estimar
# las curvaturas. Con dos iteraciones el ruido del escaneo del conejo deja de
# dibujarse y las líneas grandes se conservan.
DIFUSION_INICIAL = 2
# dos caras vecinas cuyas normales se separan más que esto forman una arista
# viva; una malla con más de esta fracción de aristas vivas se trata como facetada
ANGULO_ARISTA_VIVA = 60.0
FRACCION_FACETADA = 0.02


def normalizar_simetrico(campo):
    """Escala un campo escalar con signo a [-1, 1] para el colormap divergente.

    El factor de escala es el percentil 95 del valor absoluto, robusto a los
    pocos vértices de curvatura extrema que de otro modo aplastarían el resto.
    """
    escala = np.percentile(np.abs(campo), 95)
    if escala < 1e-9:
        escala = 1.0
    return np.clip(campo / escala, -1.0, 1.0).astype(np.float32)


def _matriz(matriz):
    return np.asarray(matriz, dtype=np.float32).reshape(16, 1, order="F")


@click.command("suggestive_contours", short_help="Suggestive Contours con pyglet")
@click.argument("model")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
@click.option(
    "--subdividir",
    type=int,
    default=0,
    help="Pasos de subdivisión Loop antes de calcular curvaturas, como el botón "
    "Subdivide Mesh de rtsc. Cada paso cuadruplica las caras y suaviza la "
    "superficie, así aparecen contornos en zonas donde la malla original era "
    "demasiado gruesa.",
)
@click.option("--arriba", type=click.Choice(["auto", "y", "z"]), default="auto",
              help="Eje que apunta hacia arriba en el archivo; auto usa z para los STL e y para el resto")
def suggestive_contours(model, width, height, subdividir, arriba):
    if arriba == "auto":
        arriba = "z" if model.lower().endswith(".stl") else "y"
    print(f"Cargando {model} (eje {arriba} hacia arriba)...")
    # El cálculo de curvatura de este ejemplo es por vértice, así que todas las
    # partes del modelo (Samus tiene varias) van en una sola malla, con los
    # vértices de las costuras de textura soldados. Las componentes
    # desconectadas no interfieren entre sí.
    mesh = load_single_mesh(model)

    # Subdivisión Loop opcional: agrega aristas y suaviza la superficie límite.
    # Los contornos solo pueden caer en aristas donde kr cambia de signo, así que
    # en mallas gruesas faltan; subdividir las hace representables. Usamos Loop
    # (no midpoint) porque mueve los vértices al límite suave y mejora la
    # estimación de curvatura, no solo la densidad de aristas.
    if subdividir > 0:
        print(f"Subdividiendo {subdividir} paso(s) (Loop)...")
        try:
            mesh = mesh.subdivide_loop(iterations=subdividir)
        except ValueError as error:
            # Loop no está definido en una arista con más de dos caras, como las
            # que hay donde se tocan piezas de la armadura de Samus. Los puntos
            # medios agregan aristas, pero no suavizan la superficie
            print(f"  Loop no se puede aplicar ({error}): se subdivide por puntos medios")
            for _ in range(subdividir):
                mesh = mesh.subdivide()

    # El modelo queda dentro de la esfera unitaria centrada en su caja, y la
    # cámara, a cinco radios del centro, como en rtsc.
    print("Normalizando geometría...")
    if arriba == "z":
        # giro de -90 grados en torno a X: el eje Z del archivo queda hacia arriba
        mesh.apply_transform(tr.rotationX(-np.pi / 2))
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(1.0 / np.linalg.norm(mesh.vertices, axis=1).max())
    mesh.fix_normals()
    # En una malla facetada (los Pokémon, Suzanne) las normales promediadas en
    # los vértices no representan las caras planas, y el cero de n.v
    # interpolado se corta en partes de la silueta. Ahí el contorno es la
    # arista entre una cara que mira a la cámara y otra que no.
    vivas = np.mean(mesh.face_adjacency_angles > np.radians(ANGULO_ARISTA_VIVA))
    facetada = vivas > FRACCION_FACETADA
    print(f"Aristas vivas: {100 * vivas:.1f} %; contornos "
          f"{'por aristas de silueta (malla facetada)' if facetada else 'interpolados (malla suave)'}")
    # La malla se corta en sus aristas vivas: cada lado queda con sus propios
    # vértices, así las normales y las curvaturas no se promedian a través de
    # la arista. El contorno es una curva de la superficie suave; en un cubo
    # son las aristas las que dibujan la forma.
    mesh = tm.graph.smooth_shade(mesh, angle=np.radians(ANGULO_ARISTA_VIVA), facet_minarea=None)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    # bordes de la malla cortada: los de la malla original más las aristas vivas
    bordes = mesh.edges_unique[np.bincount(mesh.edges_unique_inverse) == 1]
    lineas_borde = (vertices[bordes], np.ones((len(bordes), 2)))
    print(f"Malla: {len(vertices)} vértices, {len(faces)} caras, {len(bordes)} aristas de borde")

    corner_areas, point_areas = compute_corner_areas(vertices, faces)
    normales_base = compute_vertex_normals(vertices, faces)
    smoothing_operator = build_smoothing_operator(mesh)

    # Curvaturas y líneas que no dependen de la cámara. Se recalculan solo
    # cuando cambia la difusión de las normales.
    curvatura = {}

    def recalcular_curvaturas():
        normales = smooth_normals(normales_base, smoothing_operator, state["difusion"])
        print(f"Calculando curvaturas ({state['difusion']} iteraciones de difusión)...")
        k1, k2, d1, d2 = compute_curvatures(vertices, faces, normales, corner_areas, point_areas)
        print("Calculando derivadas de la curvatura...")
        C = compute_curvature_derivatives(vertices, faces, k1, k2, d1, d2, corner_areas, point_areas)
        curvatura.update(normales=normales, k1=k1, k2=k2, d1=d1, d2=d2, C=C)
        # curvas donde una curvatura principal vale cero: la máxima en las
        # crestas y la mínima en los valles
        curvatura["crestas"] = extract_isolines(vertices, faces, np.maximum(k1, k2))
        curvatura["valles"] = extract_isolines(vertices, faces, np.minimum(k1, k2))
        mesh_gpu.normal[:] = normales.astype(np.float32).ravel()

    window = pyglet.window.Window(width, height, caption="suggestive contours")

    mesh_pipeline = load_pipeline(CARPETA / "mesh_vertex.glsl", CARPETA / "mesh_fragment.glsl")
    line_pipeline = pyglet.graphics.shader.ShaderProgram(*(
        pyglet.graphics.shader.Shader((CARPETA / f"linea_{etapa}.glsl").read_text(), etapa)
        for etapa in ("vertex", "geometry", "fragment")
    ))

    # Buffers GPU construidos directo desde la malla: así el atributo `value`
    # por vértice alinea con los campos de curvatura.
    mesh_gpu = mesh_pipeline.vertex_list_indexed(
        len(vertices),
        GL.GL_TRIANGLES,
        faces.astype(np.uint32).ravel(),
        position=("f", vertices.astype(np.float32).ravel()),
        normal=("f", normales_base.astype(np.float32).ravel()),
        value=("f", np.zeros(len(vertices), dtype=np.float32)),
    )

    state = {
        "modo": MODO_BLANCO,
        "sugestivos": True,
        "contornos": True,
        "bordes": True,
        "lazos": False,
        "desvanecer": True,
        "crestas": False,
        "valles": False,
        "malla": True,
        "umbral": UMBRAL_INICIAL,
        "difusion": DIFUSION_INICIAL,
    }
    # el tamaño de rasgo sale de las curvaturas de la malla sin difundir, así el
    # umbral no cambia al mover la perilla de difusión
    k1, k2, _, _ = compute_curvatures(vertices, faces, normales_base, corner_areas, point_areas)
    feature_size = compute_feature_size(k1, k2, 1.0)
    print(f"Tamaño de rasgo: {feature_size:.4f} (radio del modelo: 1)")
    recalcular_curvaturas()

    # Cámara de rtsc: el campo de visión mide 2 atan(0,35) en la diagonal.
    fovy = np.degrees(2 * np.arctan(0.35 * height / np.hypot(width, height)))
    projection = tr.perspective(fovy, width / height, 0.05, 50.0)
    view = tr.lookAt(np.array([0.0, 0.0, 5.0]), np.zeros(3), np.array([0.0, 1.0, 0.0]))
    arcball = Arcball(np.linalg.inv(view), np.array((width, height), dtype=float), 1.5, np.zeros(3))

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(25, 22, 18), background_width=720)
        .add("modo")
        .add("umbral")
        .add("difusion")
        .add("lineas")
        .footer("V campo  C sugestivos  S contornos  B bordes  T lazos  F desvanecer  O P umbral  "
                ", . difusion  1 2 curvas  M malla  R reset", font_size=9)
    )

    def apply_state():
        mesh_pipeline["fieldMode"] = state["modo"]
        k1, k2 = curvatura["k1"], curvatura["k2"]
        if state["modo"] == MODO_MEDIA:
            mesh_gpu.value[:] = normalizar_simetrico((k1 + k2) / 2.0)
        elif state["modo"] == MODO_GAUSSIANA:
            mesh_gpu.value[:] = normalizar_simetrico(k1 * k2)
        # MODO_RADIAL se actualiza por cuadro en on_draw (depende de la cámara).

        panel["modo"] = f"campo: {MODOS[state['modo']]}"
        panel["umbral"] = (
            f"umbral de la derivada: {state['umbral']:.4f}"
            f"   desvanecer: {'si' if state['desvanecer'] else 'no'}"
        )
        panel["difusion"] = f"difusion de las normales: {state['difusion']} iteraciones"
        print(
            f"[sugecon] campo={MODOS[state['modo']]} umbral={state['umbral']:.4f} "
            f"desvanecer={state['desvanecer']} difusion={state['difusion']}"
        )

    apply_state()

    def dibujar_lineas(segmentos, opacidades, color, ancho, current_view):
        if len(segmentos) == 0:
            return
        lineas_gpu = line_pipeline.vertex_list(
            2 * len(segmentos),
            GL.GL_LINES,
            position=("f", segmentos.astype(np.float32).ravel()),
            alpha=("f", opacidades.astype(np.float32).ravel()),
        )
        line_pipeline.use()
        line_pipeline["view"] = _matriz(current_view)
        line_pipeline["projection"] = _matriz(projection)
        line_pipeline["tamano_ventana"] = (float(width), float(height))
        line_pipeline["ancho_px"] = ancho
        line_pipeline["color"] = color
        lineas_gpu.draw(GL.GL_LINES)
        lineas_gpu.delete()

    @window.event
    def on_draw():
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        window.clear()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)

        current_view = np.linalg.inv(arcball.pose)
        camera_pos = arcball.pose[:3, 3]
        c = curvatura
        threshold = state["umbral"] / feature_size**2
        ndotv, kr, test_num, test_den = compute_view_quantities(
            vertices, c["normales"], c["k1"], c["k2"], c["d1"], c["d2"], c["C"], camera_pos, threshold
        )

        if state["malla"]:
            if state["modo"] == MODO_RADIAL:
                mesh_gpu.value[:] = normalizar_simetrico(radial_curvature(kr, ndotv))
            # la superficie se empuja hacia atrás para que las líneas que están
            # sobre ella pasen la prueba de profundidad (los valores de rtsc)
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(5.0, 30.0)
            mesh_pipeline.use()
            mesh_pipeline["transform"] = _matriz(np.eye(4))
            mesh_pipeline["view"] = _matriz(current_view)
            mesh_pipeline["projection"] = _matriz(projection)
            mesh_pipeline["lightDir"] = (0.0, 0.7, 0.7)
            mesh_gpu.draw(GL.GL_TRIANGLES)
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        # las líneas leen la profundidad de la superficie, pero no la escriben
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        conteo = {}
        for clave, color in (("crestas", COLOR_CRESTA), ("valles", COLOR_VALLE)):
            if state[clave]:
                dibujar_lineas(*c[clave], color, 2.0, current_view)
                conteo[clave] = len(c[clave][0])
        if state["lazos"]:
            # todos los ceros de kr en las caras que miran a la cámara, sin la
            # prueba de la derivada
            lazos = extract_isolines(vertices, faces, kr, ndotv=ndotv)
            dibujar_lineas(*lazos, COLOR_LAZOS, 1.5, current_view)
            conteo["lazos"] = len(lazos[0])
        if state["sugestivos"]:
            fade = DESVANECIMIENTO / feature_size**2 if state["desvanecer"] else 0.0
            sugestivos = extract_isolines(vertices, faces, kr, test_num, test_den, ndotv, fade)
            dibujar_lineas(*sugestivos, COLOR_TINTA, ANCHO_LINEA, current_view)
            conteo["sugestivos"] = len(sugestivos[0])
        if state["contornos"]:
            if facetada:
                contornos = extract_silhouette_edges(mesh, camera_pos)
            else:
                # ceros de n.v donde kr > 0: donde kr < 0 el contorno queda
                # tapado por la misma superficie
                contornos = extract_isolines(vertices, faces, ndotv, kr)
            dibujar_lineas(*contornos, COLOR_TINTA, ANCHO_LINEA, current_view)
            conteo["contornos"] = len(contornos[0])
        if state["bordes"]:
            dibujar_lineas(*lineas_borde, COLOR_TINTA, ANCHO_LINEA, current_view)
            conteo["bordes"] = len(bordes)

        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)
        panel["lineas"] = "segmentos: " + ("   ".join(f"{clave} {n}" for clave, n in conteo.items()) or "ninguno")
        with ui_overlay():
            panel.draw()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            arcball.set_state(Arcball.STATE_ROTATE)
        elif button == pyglet.window.mouse.RIGHT:
            arcball.set_state(Arcball.STATE_PAN)
        arcball.down((x, y))

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        arcball.drag((x, y))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        arcball.scroll(scroll_y)

    @window.event
    def on_key_press(symbol, modifiers):
        key = pyglet.window.key
        alternar = {key.C: "sugestivos", key.S: "contornos", key.B: "bordes", key.T: "lazos", key.F: "desvanecer",
                    key._1: "crestas", key._2: "valles", key.M: "malla"}
        if symbol in alternar:
            state[alternar[symbol]] = not state[alternar[symbol]]
        elif symbol == key.V:
            state["modo"] = (state["modo"] + 1) % len(MODOS)
        elif symbol == key.O:
            state["umbral"] /= 1.25
        elif symbol == key.P:
            state["umbral"] *= 1.25
        elif symbol in (key.COMMA, key.PERIOD):
            difusion = max(0, state["difusion"] + (1 if symbol == key.PERIOD else -1))
            if difusion == state["difusion"]:
                return
            state["difusion"] = difusion
            recalcular_curvaturas()
        elif symbol == key.R:
            arcball.reset()
            return
        else:
            return
        apply_state()

    print("\nControles:")
    print("  Mouse izquierdo: rotar   |   derecho: pan   |   rueda: zoom")
    print("  V: campo (blanco / H / K / kr)")
    print("  C: suggestive contours   S: contornos   B: bordes y aristas vivas")
    print("  T: lazos de kr = 0   F: desvanecer")
    print("  O P : umbral de la derivada   |   , . : difusión de las normales")
    print("  1: curvatura máxima = 0   2: curvatura mínima = 0   M: malla   R: reset cámara")

    pyglet.app.run()
