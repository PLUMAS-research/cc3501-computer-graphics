"""Lapped textures sobre una malla segmentada en parches.

Flujo:

1. Se calculan normales estables (suavizado en una copia de la malla).
2. Se elige un conjunto de vértices semilla con farthest-point sampling.
3. Dijkstra multi-fuente sobre el grafo de aristas asigna cada vértice al
   seed más cercano geodésicamente. Las caras votan el parche al que
   pertenecen entre sus 3 vértices.
4. Cada cara contribuye 3 vértices nuevos al buffer (duplicación). Las
   UV del parche se computan proyectando (posición del vértice - posición
   del seed) sobre el marco tangente fijo del seed. Todos los parches
   comparten la misma escala de mundo para la UV, por lo que el patrón de
   achurado tiene período visual constante.

El fragment shader ya no recalcula un marco tangente por fragmento; recibe
una UV estable del parche interpolada linealmente dentro de cada cara. Las
costuras entre parches son visibles en las fronteras (el marco cambia),
pero la textura dentro de cada parche es consistente.
"""

from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import grafica.transformations as tr
from grafica.arcball import Arcball
from grafica.background import GradientBackground
from grafica.ui import ui_overlay
from grafica.utils import load_pipeline


def _smoothed_normals(mesh, iterations=5):
    """Normales por vértice calculadas sobre una copia suavizada."""
    smoothed_mesh = mesh.copy()
    try:
        tm.smoothing.filter_humphrey(
            smoothed_mesh, alpha=0.1, beta=0.5, iterations=iterations
        )
    except Exception:
        tm.smoothing.filter_laplacian(
            smoothed_mesh, lamb=0.5, iterations=iterations
        )
    return np.asarray(smoothed_mesh.vertex_normals, dtype=np.float32)


def _projected_tangent_field(normals, reference_direction=None):
    """Proyecta una dirección global sobre el plano tangente por vértice."""
    if reference_direction is None:
        reference_direction = np.array([0.0, 1.0, 0.0])
    reference_direction = reference_direction / np.linalg.norm(
        reference_direction
    )

    normals = np.asarray(normals, dtype=np.float64)
    dot_products = normals @ reference_direction
    tangent_field = (
        reference_direction[None, :] - dot_products[:, None] * normals
    )
    tangent_norms = np.linalg.norm(tangent_field, axis=1)

    singular_mask = tangent_norms < 1e-4
    if np.any(singular_mask):
        fallback_direction = np.array([1.0, 0.0, 0.0])
        singular_normals = normals[singular_mask]
        dot_fallback = singular_normals @ fallback_direction
        tangent_field[singular_mask] = (
            fallback_direction[None, :]
            - dot_fallback[:, None] * singular_normals
        )
        tangent_norms[singular_mask] = np.linalg.norm(
            tangent_field[singular_mask], axis=1
        )

    tangent_field /= tangent_norms[:, None]
    return tangent_field.astype(np.float32)


def _farthest_point_sampling(vertices, num_seeds, start_index=0):
    """Elige `num_seeds` índices de vértices maximizando la distancia
    euclidiana al conjunto ya elegido. El primero es `start_index`.
    """
    num_vertices = len(vertices)
    num_seeds = min(num_seeds, num_vertices)
    seeds = np.empty(num_seeds, dtype=np.int64)
    seeds[0] = start_index
    distances_to_set = np.full(num_vertices, np.inf)

    current = start_index
    for seed_index in range(1, num_seeds):
        delta = vertices - vertices[current]
        distance_to_current = np.linalg.norm(delta, axis=1)
        distances_to_set = np.minimum(distances_to_set, distance_to_current)
        current = int(np.argmax(distances_to_set))
        seeds[seed_index] = current
    return seeds


def _edge_graph(vertices, faces):
    """Grafo sparse con pesos = longitud de arista (Euclidiana)."""
    edges_first = faces[:, [0, 1, 2]].flatten()
    edges_second = faces[:, [1, 2, 0]].flatten()
    edge_lengths = np.linalg.norm(
        vertices[edges_first] - vertices[edges_second], axis=1
    )
    # Simétrica (la malla no tiene orientación para este efecto).
    rows = np.concatenate([edges_first, edges_second])
    cols = np.concatenate([edges_second, edges_first])
    data = np.concatenate([edge_lengths, edge_lengths])
    num_vertices = len(vertices)
    return csr_matrix(
        (data, (rows, cols)), shape=(num_vertices, num_vertices)
    )


def _partition_vertices(graph, seed_vertex_indices):
    """Asigna a cada vértice el índice de su seed más cercano y devuelve
    también la distancia geodésica al seed ganador.
    """
    distances, _, predecessor_sources = dijkstra(
        graph,
        indices=seed_vertex_indices,
        return_predecessors=True,
        min_only=True,
    )
    vertex_to_seed = {
        int(vertex_index): seed_position
        for seed_position, vertex_index in enumerate(seed_vertex_indices)
    }
    assignment = np.array(
        [vertex_to_seed[int(source)] for source in predecessor_sources],
        dtype=np.int64,
    )
    return assignment, np.asarray(distances, dtype=np.float64)


def _assign_faces_to_patches(faces, vertex_patch):
    """Cada cara va al parche mayoritario entre sus 3 vértices; si hay
    empate (los tres distintos), elige el del primer vértice.
    """
    face_votes = vertex_patch[faces]
    first_vote = face_votes[:, 0]
    second_vote = face_votes[:, 1]
    third_vote = face_votes[:, 2]
    pick_second = (
        (second_vote == third_vote)
        & (first_vote != second_vote)
        & (first_vote != third_vote)
    )
    return np.where(pick_second, second_vote, first_vote)


def _patch_color_palette(num_patches, seed=17):
    """Colores planos pseudo-aleatorios por parche, con saturación media
    para que se lean bien contra el color del papel.
    """
    rng = np.random.default_rng(seed)
    hues = rng.uniform(0.0, 1.0, num_patches)
    # Conversión HSV simple con S=0.55, V=0.92.
    saturation = 0.55
    value = 0.92
    k = np.floor(hues * 6.0).astype(np.int64)
    f = hues * 6.0 - k
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))
    palette = np.zeros((num_patches, 3), dtype=np.float32)
    for i in range(num_patches):
        sector = k[i] % 6
        if sector == 0:
            palette[i] = (value, t[i], p)
        elif sector == 1:
            palette[i] = (q[i], value, p)
        elif sector == 2:
            palette[i] = (p, value, t[i])
        elif sector == 3:
            palette[i] = (p, q[i], value)
        elif sector == 4:
            palette[i] = (t[i], p, value)
        else:
            palette[i] = (value, p, q[i])
    return palette


def _build_patch_buffers(
    vertices,
    faces,
    normals,
    tangents,
    seed_vertex_indices,
    face_patch,
    geodesic_distances,
):
    """Construye los buffers expandidos (un vértice por (cara, esquina)).

    La UV de cada vértice se computa con un log map discreto aproximado:
    el ángulo sale de proyectar (vértice - seed) sobre el marco tangente
    del seed; el radio es la distancia geodésica real del vértice al seed.
    Así el período del hatching es uniforme entre parches curvos y planos
    (de lo contrario la proyección pura estira la textura donde la malla
    se aleja del plano tangente del seed). Las normales se usan solo para
    reortogonalizar el marco del seed; no viajan a GPU porque esta unidad
    aún no cubre iluminación.
    """
    seed_positions = vertices[seed_vertex_indices]
    seed_normals = normals[seed_vertex_indices]
    seed_tangents = tangents[seed_vertex_indices]
    # Reortogonalizar el marco en el seed.
    seed_tangents = seed_tangents - (
        np.sum(seed_tangents * seed_normals, axis=1, keepdims=True)
        * seed_normals
    )
    lengths = np.linalg.norm(seed_tangents, axis=1, keepdims=True)
    safe = lengths.squeeze(-1) > 1e-4
    seed_tangents[safe] /= lengths[safe]
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if np.any(~safe):
        fallback_direction = fallback - (
            seed_normals[~safe] @ fallback
        )[:, None] * seed_normals[~safe]
        fallback_lengths = np.linalg.norm(fallback_direction, axis=1, keepdims=True)
        seed_tangents[~safe] = fallback_direction / fallback_lengths
    seed_bitangents = np.cross(seed_normals, seed_tangents)

    num_faces = len(faces)
    corner_vertex_index = faces.flatten()
    expanded_positions = vertices[corner_vertex_index]

    # Por cada cara, el marco del seed ganador, replicado a las 3 esquinas.
    face_seed_position = seed_positions[face_patch]
    face_seed_tangent = seed_tangents[face_patch]
    face_seed_bitangent = seed_bitangents[face_patch]

    expanded_seed_position = np.repeat(face_seed_position, 3, axis=0)
    expanded_seed_tangent = np.repeat(face_seed_tangent, 3, axis=0)
    expanded_seed_bitangent = np.repeat(face_seed_bitangent, 3, axis=0)

    relative = expanded_positions - expanded_seed_position
    tangent_component = np.einsum("ij,ij->i", relative, expanded_seed_tangent)
    bitangent_component = np.einsum(
        "ij,ij->i", relative, expanded_seed_bitangent
    )
    azimuth = np.arctan2(bitangent_component, tangent_component)

    # Radio = distancia geodésica real del vértice al seed de su parche
    # original. Para el subconjunto de vértices cuyo parche mayoritario
    # difiere del seed que gana en la cara, este radio es una buena
    # aproximación salvo en la frontera del parche, donde el log map pierde
    # precisión de todos modos.
    expanded_radius = geodesic_distances[corner_vertex_index]
    patch_u = expanded_radius * np.cos(azimuth)
    patch_v = expanded_radius * np.sin(azimuth)
    patch_uv = np.stack([patch_u, patch_v], axis=1).astype(np.float32)

    num_patches = len(seed_vertex_indices)
    palette = _patch_color_palette(num_patches)
    expanded_patch_id = np.repeat(face_patch, 3)
    expanded_patch_color = palette[expanded_patch_id]

    return (
        expanded_positions.astype(np.float32),
        patch_uv,
        expanded_patch_color.astype(np.float32),
        np.arange(3 * num_faces, dtype=np.uint32),
    )


@click.command(
    "lapped_hatching",
    short_help="Texturas sintéticas estilo achurado con lapped textures",
)
@click.argument("mesh_path", type=click.Path(exists=True))
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=720)
@click.option(
    "--patches", "num_patches", type=int, default=48,
    help="Cantidad de parches en que se segmenta la malla.",
)
def lapped_hatching(mesh_path, width, height, num_patches):
    print(f"Cargando {mesh_path}...")
    mesh = tm.load(mesh_path, force="mesh")
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(2.0 / mesh.scale)
    mesh.fix_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    print(f"  {len(vertices)} vértices, {len(faces)} caras")

    print("Suavizando malla (copia) para estabilizar normales...")
    stable_normals = _smoothed_normals(mesh, iterations=5)

    print("Construyendo campo tangente por proyección de Y...")
    tangent_field = _projected_tangent_field(stable_normals)

    print(f"Eligiendo {num_patches} seeds con farthest-point sampling...")
    seed_vertex_indices = _farthest_point_sampling(vertices, num_patches)

    print("Particionando la malla con Dijkstra multi-fuente...")
    graph = _edge_graph(vertices, faces)
    vertex_patch, geodesic_distances = _partition_vertices(
        graph, seed_vertex_indices
    )
    face_patch = _assign_faces_to_patches(faces, vertex_patch)

    print("Construyendo buffers expandidos por parche...")
    (
        expanded_positions,
        patch_uv,
        patch_color,
        expanded_indices,
    ) = _build_patch_buffers(
        vertices.astype(np.float32),
        faces,
        stable_normals,
        tangent_field,
        seed_vertex_indices,
        face_patch,
        geodesic_distances,
    )
    print(f"  {len(expanded_positions)} vértices expandidos")

    window = pyglet.window.Window(
        width, height, "Lapped hatching sobre malla segmentada"
    )

    asset_root = Path(__file__).parent.parent.parent / "assets"
    pyglet.font.add_file(str(asset_root / "FiraCode" / "FiraCode-Regular.ttf"))

    base_path = Path(__file__).parent
    pipeline = load_pipeline(
        base_path / "vertex_program.glsl",
        base_path / "fragment_program.glsl",
    )
    overlay_pipeline = load_pipeline(
        base_path / "overlay_vertex.glsl",
        base_path / "overlay_fragment.glsl",
    )

    overlay_size_pixels = 200
    overlay_margin_pixels = 20
    x_min_ndc = 1.0 - 2.0 * (overlay_size_pixels + overlay_margin_pixels) / width
    x_max_ndc = 1.0 - 2.0 * overlay_margin_pixels / width
    y_min_ndc = 1.0 - 2.0 * (overlay_size_pixels + overlay_margin_pixels) / height
    y_max_ndc = 1.0 - 2.0 * overlay_margin_pixels / height

    overlay_positions = np.array(
        [
            x_min_ndc, y_min_ndc,
            x_max_ndc, y_min_ndc,
            x_max_ndc, y_max_ndc,
            x_min_ndc, y_max_ndc,
        ],
        dtype=np.float32,
    )
    overlay_uvs = np.array(
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        dtype=np.float32,
    )
    overlay_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    overlay_gpu = overlay_pipeline.vertex_list_indexed(
        4,
        GL.GL_TRIANGLES,
        overlay_indices,
        ndc_position=("f", overlay_positions),
        patch_uv=("f", overlay_uvs),
    )

    mesh_gpu = pipeline.vertex_list_indexed(
        len(expanded_positions),
        GL.GL_TRIANGLES,
        expanded_indices,
        position=("f", expanded_positions.flatten()),
        patch_uv=("f", patch_uv.flatten()),
        patch_color=("f", patch_color.flatten()),
    )

    near_plane = 0.1
    far_plane = 10.0
    projection = tr.perspective(
        45, float(width) / float(height), near_plane, far_plane
    )
    view = tr.lookAt(
        np.array([0, 0, 3]), np.array([0, 0, 0]), np.array([0, 1, 0])
    )

    arcball = Arcball(
        np.linalg.inv(view),
        np.array((width, height), dtype=float),
        1.5,
        np.array([0.0, 0.0, 0.0]),
    )

    background = GradientBackground(
        bottom_color=(0.88, 0.80, 0.66),
        top_color=(0.97, 0.94, 0.86),
    )

    state = {
        "stripe_period": 0.035,
        "stripe_half_width_ratio": 0.18,
        "show_uv_gradient": False,
        "show_patches": False,
        "show_texture_preview": True,
    }

    status_label = pyglet.text.Label(
        "",
        font_name="Fira Code",
        font_size=12,
        x=10,
        y=height - 20,
        anchor_x="left",
        anchor_y="top",
        color=(40, 30, 20, 255),
    )

    hint_label = pyglet.text.Label(
        "Drag: rotar  |  V: UV de parches  |  P: colores por parche  |  T: preview  |  , .: periodo  |  R: reset",
        font_name="Fira Code",
        font_size=10,
        x=10,
        y=18,
        anchor_x="left",
        anchor_y="bottom",
        color=(40, 30, 20, 220),
    )

    def update_status():
        if state["show_uv_gradient"]:
            mode = "UV del parche"
        elif state["show_patches"]:
            mode = "colores por parche"
        else:
            mode = "achurado"
        status_label.text = (
            f"Modo: {mode}  |  parches: {num_patches}  |  "
            f"periodo: {state['stripe_period']:.3f}"
        )

    update_status()

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
        keys = pyglet.window.key
        if symbol == keys.V:
            state["show_uv_gradient"] = not state["show_uv_gradient"]
            if state["show_uv_gradient"]:
                state["show_patches"] = False
        elif symbol == keys.P:
            state["show_patches"] = not state["show_patches"]
            if state["show_patches"]:
                state["show_uv_gradient"] = False
        elif symbol == keys.T:
            state["show_texture_preview"] = not state["show_texture_preview"]
        elif symbol == keys.COMMA:
            state["stripe_period"] = max(0.01, state["stripe_period"] * 0.85)
        elif symbol == keys.PERIOD:
            state["stripe_period"] = min(0.5, state["stripe_period"] * 1.18)
        elif symbol == keys.R:
            arcball.pose = np.linalg.inv(view)
        update_status()

    @window.event
    def on_draw():
        GL.glClearColor(0.88, 0.80, 0.66, 1.0)
        window.clear()
        background.draw()
        GL.glEnable(GL.GL_DEPTH_TEST)

        current_view = np.linalg.inv(arcball.pose)

        pipeline.use()
        pipeline["transform"] = np.eye(4, dtype=np.float32).reshape(
            16, 1, order="F"
        )
        pipeline["view"] = current_view.astype(np.float32).reshape(
            16, 1, order="F"
        )
        pipeline["projection"] = projection.astype(np.float32).reshape(
            16, 1, order="F"
        )
        pipeline["stripe_period"] = state["stripe_period"]
        pipeline["stripe_half_width"] = (
            state["stripe_period"] * state["stripe_half_width_ratio"]
        )
        pipeline["show_uv_gradient"] = 1 if state["show_uv_gradient"] else 0
        pipeline["show_patches"] = 1 if state["show_patches"] else 0

        mesh_gpu.draw(GL.GL_TRIANGLES)

        if state["show_texture_preview"]:
            GL.glDisable(GL.GL_DEPTH_TEST)
            overlay_pipeline.use()
            overlay_gpu.draw(GL.GL_TRIANGLES)

        with ui_overlay():
            status_label.draw()
            hint_label.draw()

    print("\nControles:")
    print("  Drag: rotar la cámara")
    print("  V: visualizar la UV de cada parche (colores)")
    print("  P: visualizar los parches con colores planos")
    print("  T: mostrar/ocultar preview de la celda de hatching")
    print("  , / .: reducir / aumentar el período de los trazos")
    print("  R: reset de cámara")

    pyglet.app.run()
