import pyglet
import pyglet.gl as GL
import trimesh as tm
import numpy as np
import os
import click
from pathlib import Path

from grafica.utils import load_pipeline
from grafica.arcball import Arcball
from grafica.ui import InfoPanel, ui_overlay
import grafica.transformations as tr

from .curvature import (
    compute_mean_curvature,
    compute_gaussian_curvature,
    compute_principal_curvatures,
    approximate_principal_directions,
    build_smoothing_operator,
    smooth_vertex_attribute,
)
from .contours import (
    compute_radial_curvature,
    extract_contour_points,
    extract_silhouette_edges,
    compute_radial_curvature_derivative,
    extract_zero_crossings,
)


# Modos de relleno de la malla: papel (NPR) y tres campos de curvatura.
MODOS = ["papel", "media (H)", "gaussiana (K)", "radial (kr)"]
MODO_PAPEL, MODO_MEDIA, MODO_GAUSSIANA, MODO_RADIAL = 0, 1, 2, 3

COLOR_TINTA = (0.15, 0.12, 0.10)
COLOR_CRESTA = (0.75, 0.20, 0.15)  # rojo cálido, como el positivo del colormap
COLOR_VALLE = (0.18, 0.30, 0.65)  # azul frío, como el negativo del colormap


def normalizar_simetrico(campo):
    """Escala un campo escalar con signo a [-1, 1] para el colormap divergente.

    El factor de escala es el percentil 95 del valor absoluto, robusto a los
    pocos vértices de curvatura extrema que de otro modo aplastarían el resto.
    """
    escala = np.percentile(np.abs(campo), 95)
    if escala < 1e-9:
        escala = 1.0
    return np.clip(campo / escala, -1.0, 1.0).astype(np.float32)


@click.command("suggestive_contours", short_help="Suggestive Contours con pyglet")
@click.argument("model")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
@click.option(
    "--subdividir",
    type=int,
    default=0,
    help="Pasos de subdivisión Loop antes de calcular curvaturas. Cada paso "
    "cuadruplica las caras y suaviza la superficie, así aparecen contornos en "
    "zonas que la malla original tenía demasiado gruesas (DeCarlo et al. 2003).",
)
def suggestive_contours(model, width, height, subdividir):
    print(f"Cargando {model}...")
    mesh = tm.load(model)

    # Un OBJ con varias partes (como assets/samus/posed.obj) carga como Scene.
    # El cálculo de curvatura de este ejemplo es por vértice, así que fusionamos
    # las partes en una sola malla y el resto del código opera igual sobre
    # vértices y caras. Las componentes desconectadas no interfieren entre sí.
    if isinstance(mesh, tm.Scene):
        print(f"OBJ con {len(mesh.geometry)} partes: fusionando en una sola malla...")
        mesh = mesh.to_geometry()

    # Subdivisión Loop opcional: agrega aristas y suaviza la superficie límite.
    # Los contornos solo pueden caer en aristas donde kr cambia de signo, así que
    # en mallas gruesas faltan; subdividir las hace representables. Usamos Loop
    # (no midpoint) porque mueve los vértices al límite suave y mejora la
    # estimación de curvatura, no solo la densidad de aristas.
    if subdividir > 0:
        print(f"Subdividiendo {subdividir} paso(s) (Loop)...")
        mesh = mesh.subdivide_loop(iterations=subdividir)

    print("Normalizando geometría...")
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(2.0 / mesh.scale)
    mesh.fix_normals()

    print(f"Malla: {len(mesh.vertices)} vértices, {len(mesh.faces)} caras")

    print("Calculando curvaturas...")
    H_base = compute_mean_curvature(mesh)
    K_base = compute_gaussian_curvature(mesh)
    k1_base, k2_base = compute_principal_curvatures(H_base, K_base)
    d1, d2 = approximate_principal_directions(mesh, k1_base, k2_base)

    # Operador de difusión sobre la malla, construido una vez (depende solo de
    # la topología). Cada perilla de suavizado lo reusa.
    print("Construyendo operador de difusión...")
    smoothing_operator = build_smoothing_operator(mesh)

    window = pyglet.window.Window(width, height)

    base_path = Path(os.path.dirname(__file__))
    mesh_pipeline = load_pipeline(
        base_path / "mesh_vertex.glsl", base_path / "mesh_fragment.glsl"
    )
    contour_pipeline = load_pipeline(
        base_path / "contour_vertex.glsl", base_path / "contour_fragment.glsl"
    )
    fondo_pipeline = load_pipeline(
        base_path / "fondo_vertex.glsl", base_path / "fondo_fragment.glsl"
    )

    # Quad de fondo a pantalla completa (papel con viñeta y grano).
    fondo_gpu = fondo_pipeline.vertex_list_indexed(
        4,
        GL.GL_TRIANGLES,
        [0, 1, 2, 0, 2, 3],
        position=("f", [-1, -1, 1, -1, 1, 1, -1, 1]),
    )

    # Buffers GPU construidos directo desde la malla (no mesh_to_vertexlist, que
    # reordena y duplica vértices): así el atributo `value` por vértice alinea
    # con los campos de curvatura, que se indexan por mesh.vertices.
    num_vertices = len(mesh.vertices)
    mesh_gpu = mesh_pipeline.vertex_list_indexed(
        num_vertices,
        GL.GL_TRIANGLES,
        mesh.faces.astype(np.uint32).flatten(),
        position=("f", mesh.vertices.astype(np.float32).flatten()),
        normal=("f", mesh.vertex_normals.astype(np.float32).flatten()),
        value=("f", np.zeros(num_vertices, dtype=np.float32)),
    )

    near_plane = 0.1
    far_plane = 10.0
    projection = tr.perspective(60, float(width) / float(height), near_plane, far_plane)
    view = tr.lookAt(np.array([0, 0, 3]), np.array([0, 0, 0]), np.array([0, 1, 0]))

    arcball = Arcball(
        np.linalg.inv(view),
        np.array((width, height), dtype=float),
        1.5,
        np.array([0.0, 0.0, 0.0]),
    )

    show_mesh = True
    show_contours = True
    show_silhouettes = True
    show_ridges = False
    show_valleys = False
    contour_count = 0

    # Crestas (k1=0) y valles (k2=0). No dependen de la cámara, así que se
    # recalculan solo cuando cambia el suavizado (en apply_state), no por cuadro.
    feature_lines = {"ridges": (np.array([]), []), "valleys": (np.array([]), [])}

    state = {"mode": MODO_PAPEL, "iters": 2, "lambda": 0.3, "threshold": 0.1}
    # Curvaturas principales suavizadas según la perilla actual. Las llena
    # apply_state(); on_draw las usa para kr cada frame.
    campos = {"k1": k1_base.copy(), "k2": k2_base.copy()}

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(25, 22, 18))
        .add("modo")
        .add("difusion")
        .add("umbral")
        .add("lineas")
        .footer("V campo  , . iter  - = lambda  O P umbral  M malla  C contornos  S siluetas  1 crestas  2 valles  R reset")
    )

    def apply_state():
        # Difusión (suavizado laplaciano) de las curvaturas principales. Más
        # iteraciones o lambda mayor regularizan el campo y, con él, dónde cruza
        # kr por cero: los contornos se vuelven más limpios y continuos.
        k1s = smooth_vertex_attribute(
            mesh, k1_base, state["iters"], state["lambda"], operator=smoothing_operator
        )
        k2s = smooth_vertex_attribute(
            mesh, k2_base, state["iters"], state["lambda"], operator=smoothing_operator
        )
        campos["k1"] = k1s
        campos["k2"] = k2s

        # Crestas y valles con las curvaturas principales ya suavizadas.
        feature_lines["ridges"] = extract_zero_crossings(mesh, k1s)
        feature_lines["valleys"] = extract_zero_crossings(mesh, k2s)

        mesh_pipeline["fieldMode"] = state["mode"]
        if state["mode"] == MODO_MEDIA:
            mesh_gpu.value[:] = normalizar_simetrico((k1s + k2s) / 2.0)
        elif state["mode"] == MODO_GAUSSIANA:
            mesh_gpu.value[:] = normalizar_simetrico(k1s * k2s)
        elif state["mode"] == MODO_PAPEL:
            mesh_gpu.value[:] = np.zeros(num_vertices, dtype=np.float32)
        # MODO_RADIAL se actualiza por frame en on_draw (depende de la cámara).

        panel["modo"] = f"campo: {MODOS[state['mode']]}"
        panel["difusion"] = (
            f"difusion en la malla: {state['iters']} iter, lambda {state['lambda']:.2f}"
        )
        print(
            f"[sugecon] modo={MODOS[state['mode']]} iters={state['iters']} "
            f"lambda={state['lambda']:.2f}"
        )

    def apply_threshold():
        # El umbral no toca las curvaturas (que sí dependen de la difusión), solo
        # cuánta derivada radial exigimos para aceptar una arista como contorno.
        # Por eso vive en su propia función ligera y no llama a apply_state().
        panel["umbral"] = f"umbral seleccion contorno: {state['threshold']:.2f}"
        print(f"[sugecon] umbral seleccion contorno={state['threshold']:.2f}")

    apply_state()
    apply_threshold()

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
        nonlocal show_mesh, show_contours, show_silhouettes, show_ridges, show_valleys

        if symbol == pyglet.window.key.M:
            show_mesh = not show_mesh

        elif symbol == pyglet.window.key.C:
            show_contours = not show_contours

        elif symbol == pyglet.window.key.S:
            show_silhouettes = not show_silhouettes

        elif symbol == pyglet.window.key._1:
            show_ridges = not show_ridges

        elif symbol == pyglet.window.key._2:
            show_valleys = not show_valleys

        elif symbol == pyglet.window.key.R:
            arcball.reset()

        elif symbol == pyglet.window.key.V:
            state["mode"] = (state["mode"] + 1) % len(MODOS)
            apply_state()

        elif symbol == pyglet.window.key.COMMA:
            state["iters"] = max(0, state["iters"] - 1)
            apply_state()

        elif symbol == pyglet.window.key.PERIOD:
            state["iters"] = min(40, state["iters"] + 1)
            apply_state()

        elif symbol == pyglet.window.key.MINUS:
            state["lambda"] = max(0.0, round(state["lambda"] - 0.05, 2))
            apply_state()

        elif symbol == pyglet.window.key.EQUAL:
            state["lambda"] = min(1.0, round(state["lambda"] + 0.05, 2))
            apply_state()

        elif symbol == pyglet.window.key.O:
            state["threshold"] = max(0.0, round(state["threshold"] - 0.05, 2))
            apply_threshold()

        elif symbol == pyglet.window.key.P:
            state["threshold"] = min(0.95, round(state["threshold"] + 0.05, 2))
            apply_threshold()

    @window.event
    def on_draw():
        nonlocal contour_count

        window.clear()

        # Fondo papel: sin depth, cubre toda la pantalla.
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        fondo_pipeline.use()
        fondo_gpu.draw(GL.GL_TRIANGLES)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)

        current_view = np.linalg.inv(arcball.pose)
        camera_pos = arcball.pose[:3, 3]

        # La curvatura radial depende de la cámara: se recalcula cada frame con
        # las curvaturas principales ya suavizadas.
        kr = compute_radial_curvature(mesh, camera_pos, campos["k1"], campos["k2"], d1, d2)

        if show_mesh:
            if state["mode"] == MODO_RADIAL:
                mesh_gpu.value[:] = normalizar_simetrico(kr)

            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            mesh_pipeline.use()
            mesh_pipeline["transform"] = np.eye(4, dtype=np.float32).reshape(16, 1, order="F")
            mesh_pipeline["view"] = current_view.astype(np.float32).reshape(16, 1, order="F")
            mesh_pipeline["projection"] = projection.reshape(16, 1, order="F")
            mesh_pipeline["lightDir"] = (0.0, 0.7, 0.7)
            mesh_pipeline["fieldMode"] = state["mode"]
            mesh_gpu.draw(GL.GL_TRIANGLES)

        if show_contours:
            # La derivada radial también depende de la cámara, así que se
            # recalcula cada frame junto con kr (vectorizada, son pocos ms). El
            # umbral de selección la compara con su propia escala robusta.
            dkr = compute_radial_curvature_derivative(
                mesh, camera_pos, campos["k1"], campos["k2"], d1, d2, kr
            )
            contour_points, contour_edges, contour_kr_values = extract_contour_points(
                mesh, kr, dkr, state["threshold"]
            )
            contour_count = len(contour_edges)

            if len(contour_points) > 0 and len(contour_edges) > 0:
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glEnable(GL.GL_LINE_SMOOTH)
                GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)

                max_kr = 0.05
                alphas = (1.0 - np.clip(contour_kr_values / max_kr, 0, 1)).astype(np.float32)
                line_indices = np.array(contour_edges, dtype=np.uint32).flatten()

                contour_gpu = contour_pipeline.vertex_list_indexed(
                    len(contour_points),
                    GL.GL_LINES,
                    line_indices,
                    position=("f", contour_points.flatten()),
                    alpha=("f", alphas),
                )

                GL.glLineWidth(2.5)
                contour_pipeline.use()
                contour_pipeline["transform"] = np.eye(4, dtype=np.float32).reshape(16, 1, order="F")
                contour_pipeline["view"] = current_view.astype(np.float32).reshape(16, 1, order="F")
                contour_pipeline["projection"] = projection.reshape(16, 1, order="F")
                contour_pipeline["color"] = COLOR_TINTA
                contour_gpu.draw(GL.GL_LINES)
                contour_gpu.delete()

                GL.glDisable(GL.GL_BLEND)
                GL.glDisable(GL.GL_LINE_SMOOTH)

        if show_silhouettes:
            silhouette_points, silhouette_edges = extract_silhouette_edges(mesh, camera_pos)

            if len(silhouette_points) > 0 and len(silhouette_edges) > 0:
                line_indices = np.array(silhouette_edges, dtype=np.uint32).flatten()
                silhouette_gpu = contour_pipeline.vertex_list_indexed(
                    len(silhouette_points), GL.GL_LINES, line_indices
                )
                silhouette_gpu.position[:] = silhouette_points.flatten()

                GL.glLineWidth(5.0)
                contour_pipeline.use()
                contour_pipeline["transform"] = np.eye(4, dtype=np.float32).reshape(16, 1, order="F")
                contour_pipeline["view"] = current_view.astype(np.float32).reshape(16, 1, order="F")
                contour_pipeline["projection"] = projection.reshape(16, 1, order="F")
                contour_pipeline["color"] = COLOR_TINTA
                silhouette_gpu.draw(GL.GL_LINES)
                silhouette_gpu.delete()

        # Crestas y valles: líneas precalculadas (no dependen de la cámara). Se
        # dibujan sin blending, como las siluetas, cada una con su color.
        for activo, clave, color in (
            (show_ridges, "ridges", COLOR_CRESTA),
            (show_valleys, "valleys", COLOR_VALLE),
        ):
            if not activo:
                continue
            puntos, aristas = feature_lines[clave]
            if len(puntos) == 0 or len(aristas) == 0:
                continue
            line_indices = np.array(aristas, dtype=np.uint32).flatten()
            feature_gpu = contour_pipeline.vertex_list_indexed(
                len(puntos), GL.GL_LINES, line_indices
            )
            feature_gpu.position[:] = puntos.flatten()

            GL.glLineWidth(2.0)
            contour_pipeline.use()
            contour_pipeline["transform"] = np.eye(4, dtype=np.float32).reshape(16, 1, order="F")
            contour_pipeline["view"] = current_view.astype(np.float32).reshape(16, 1, order="F")
            contour_pipeline["projection"] = projection.reshape(16, 1, order="F")
            contour_pipeline["color"] = color
            feature_gpu.draw(GL.GL_LINES)
            feature_gpu.delete()

        panel["lineas"] = (
            f"contornos: {contour_count}   siluetas: {'ON' if show_silhouettes else 'OFF'}"
            f"   crestas: {'ON' if show_ridges else 'OFF'}   valles: {'ON' if show_valleys else 'OFF'}"
        )
        with ui_overlay():
            panel.draw()

    print("\nControles:")
    print("  Mouse izquierdo: rotar   |   derecho: pan   |   scroll: zoom")
    print("  V: campo (papel / H / K / kr)")
    print("  , . : iteraciones de difusión   |   - = : factor lambda")
    print("  O P : umbral de selección de contornos (derivada radial)")
    print("  1: crestas (k1=0)   2: valles (k2=0)")
    print("  M: malla   C: contornos   S: siluetas   R: reset cámara")

    pyglet.app.run()
