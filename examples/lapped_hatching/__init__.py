"""Lapped textures sobre una malla segmentada en parches.

Flujo:

1. Se calculan normales estables (suavizado en una copia de la malla).
2. Se elige un conjunto de vértices semilla con farthest-point sampling.
3. Dijkstra multi-fuente sobre el grafo de aristas devuelve la matriz
   completa de distancias geodésicas seed -> vértice. Cada vértice se
   asigna al seed más cercano y cada cara hereda el parche votado por
   mayoría entre sus 3 vértices.
4. Para cada seed se construye un marco tangente (proyectando Y sobre su
   plano tangente). Despues se hace parallel transport en BFS sobre el
   grafo de parches: cada parche que se visita toma como tangente la
   proyeccion de la tangente del predecesor en su propio plano. El angulo
   de rotacion entre la tangente original y la alineada queda guardado
   por parche y se aplica en el fragment shader (tecla A alterna).
5. Cada cara contribuye 3 vértices nuevos al buffer (duplicación). La UV
   de cada esquina se computa con un log map discreto: azimut respecto al
   marco tangente del seed de la cara, radio = distancia geodésica del
   vértice a ese mismo seed (consistente con el azimut).
6. Para suavizar las costuras se construye una "falda" por parche: las
   caras vecinas hasta `skirt_rings` saltos del nucleo del parche. La
   falda se renderiza en una segunda pasada con alpha blending; el alpha
   por vertice cae linealmente con la cantidad de hops al nucleo (1 en el
   nucleo, 0 en el borde externo del anillo). Donde dos parches se
   solapan, ambos contribuyen y la transicion queda continua. Tecla S
   alterna.
7. Tonal Art Map: se genera procedural un volumen de niveles tonales
   (textura 3D, 6 slices de 256x256). Cada slice agrega trazos sobre el
   anterior conservando los previos (tonal coherence). En el fragment
   shader se calcula iluminacion difusa simple y el tono = 1 - intensidad
   sirve como tercera coordenada de muestreo. El sampler interpola
   linealmente entre slices, asi la transicion entre niveles tonales es
   suave. Tecla L alterna entre TAM con luz y el achurado procedural sin
   luz (que sigue siendo util para entender el patron base).

El fragment shader recibe una UV estable del parche, una normal por
vertice, un offset de alineamiento por parche y un alpha por vertice.
Los diagnosticos (V, P) ignoran la falda y la iluminacion; en esos modos
solo se ve el nucleo con la informacion del parche.
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
    """Devuelve la matriz completa de distancias geodésicas seed -> vértice
    (shape (K, N)) y la asignación vértice -> índice del seed más cercano.

    La matriz completa permite consultar la distancia geodésica del vértice
    al seed de cualquier parche, no solo al ganador. Esto se necesita para
    que la UV polar de una cara use el radio respecto al seed de su propio
    parche, aún cuando uno de sus vértices pertenezca al parche vecino.
    """
    distances = dijkstra(graph, indices=seed_vertex_indices)
    distances = np.asarray(distances, dtype=np.float64)
    assignment = np.argmin(distances, axis=0).astype(np.int64)
    return assignment, distances


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


def _generate_tam_volume(num_levels=6, size=256):
    """Volumen procedural estilo Tonal Art Map.

    Devuelve un array (num_levels, size, size) en [0, 1] que representa el
    "ink amount" en cada celda de cada nivel tonal. El nivel 0 es papel
    (sin tinta) y los niveles van acumulando trazos hasta el mas oscuro.
    Cada paso del schedule agrega trazos pero conserva los anteriores
    (tonal coherence: una vez que un stroke aparece, sigue presente en
    niveles mas oscuros). Asi la interpolacion lineal entre slices en la
    GPU no hace que los trazos "parpadeen".

    Los trazos son aproximadamente tileables en u y v. Los angulos no
    axiales (60 grados) introducen un pequeno offset al cruzar el borde,
    pero a 256 pixeles y con repeat el artefacto es minimo.
    """
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    u_grid, v_grid = np.meshgrid(coords, coords, indexing="xy")

    # Schedule: cada paso agrega una capa de trazos.
    # (angulo_grados, periodo_uv, ancho_uv, intensidad)
    schedule = [
        (0,    1.0 / 4.0, 0.025, 1.0),   # nivel 1: stripes 0 grados, sparse
        (0,    1.0 / 8.0, 0.025, 1.0),   # nivel 2: doble densidad mismo angulo
        (60,   1.0 / 4.0, 0.025, 1.0),   # nivel 3: agrega +60 grados, sparse
        (60,   1.0 / 8.0, 0.025, 1.0),   # nivel 4: doble densidad +60
        (-60,  1.0 / 4.0, 0.030, 1.0),   # nivel 5: agrega -60 grados
    ]

    volume = np.zeros((num_levels, size, size), dtype=np.float32)
    ink_amount = np.zeros((size, size), dtype=np.float32)

    for level_index, (angle_deg, period, width, intensity) in enumerate(schedule):
        angle = np.deg2rad(angle_deg)
        # Coordenada perpendicular a la direccion del trazo
        perp = np.sin(angle) * u_grid + np.cos(angle) * v_grid
        dist = np.abs(np.mod(perp / period + 0.5, 1.0) - 0.5) * period
        falloff_inner = width * 0.7
        falloff_outer = width
        stripe_mask = 1.0 - np.clip(
            (dist - falloff_inner) / (falloff_outer - falloff_inner),
            0.0,
            1.0,
        )
        stripe_mask = stripe_mask * intensity
        ink_amount = np.clip(
            ink_amount + (1.0 - ink_amount) * stripe_mask, 0.0, 1.0
        )
        target_level = level_index + 1  # nivel 0 queda como papel limpio
        if target_level < num_levels:
            volume[target_level] = ink_amount

    # Si el schedule no alcanza para todos los niveles, repetir el ultimo.
    for level in range(len(schedule) + 1, num_levels):
        volume[level] = volume[len(schedule)]

    return volume


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


def _seed_frames(seed_vertex_indices, vertices, normals, initial_tangents):
    """Construye el marco (posicion, normal, tangente, bitangente) por seed.

    Reortogonaliza la tangente respecto a la normal y normaliza. Si la
    tangente queda degenerada (tangente paralela a la normal), proyecta un
    fallback global como tangente.
    """
    seed_positions = vertices[seed_vertex_indices]
    seed_normals = normals[seed_vertex_indices]
    seed_tangents = initial_tangents[seed_vertex_indices].copy()

    seed_tangents = seed_tangents - (
        np.sum(seed_tangents * seed_normals, axis=1, keepdims=True)
        * seed_normals
    )
    lengths = np.linalg.norm(seed_tangents, axis=1, keepdims=True)
    safe = lengths.squeeze(-1) > 1e-4
    seed_tangents[safe] /= lengths[safe]
    if np.any(~safe):
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        fallback_direction = fallback - (
            seed_normals[~safe] @ fallback
        )[:, None] * seed_normals[~safe]
        fallback_lengths = np.linalg.norm(fallback_direction, axis=1, keepdims=True)
        seed_tangents[~safe] = fallback_direction / fallback_lengths
    seed_bitangents = np.cross(seed_normals, seed_tangents)
    return (
        seed_positions.astype(np.float32),
        seed_normals.astype(np.float32),
        seed_tangents.astype(np.float32),
        seed_bitangents.astype(np.float32),
    )


def _unique_patch_edges(face_patch, face_adjacency):
    """Aristas del grafo de adyacencia entre parches, deduplicadas y con
    orden (a < b) por fila.
    """
    pairs = face_patch[face_adjacency]
    mask = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[mask]
    sorted_pairs = np.sort(pairs, axis=1)
    return np.unique(sorted_pairs, axis=0)


def _aligned_seed_tangents(
    seed_normals, initial_tangents, unique_patch_edges, num_patches, root=0
):
    """Propaga la tangente del parche raiz a sus vecinos por parallel
    transport discreto: cada parche visitado proyecta la tangente del
    predecesor sobre su propio plano tangente.

    El orden es BFS, asi que la holonomia (diferencia segun el camino)
    es la del arbol generador BFS, no la de un camino libre. Para los
    fines de alinear achurado eso basta: parches vecinos en la malla
    terminan con tangentes similares y el patron cruza la costura con
    direccion casi continua.
    """
    from collections import deque

    aligned = initial_tangents.copy()
    visited = np.zeros(num_patches, dtype=bool)
    adjacency = [[] for _ in range(num_patches)]
    for a, b in unique_patch_edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    starts = [root] + [i for i in range(num_patches) if i != root]
    for start in starts:
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        while queue:
            current = queue.popleft()
            current_tangent = aligned[current]
            for neighbor in adjacency[current]:
                if visited[neighbor]:
                    continue
                projected = current_tangent - np.dot(
                    current_tangent, seed_normals[neighbor]
                ) * seed_normals[neighbor]
                norm = np.linalg.norm(projected)
                if norm > 1e-6:
                    aligned[neighbor] = projected / norm
                visited[neighbor] = True
                queue.append(neighbor)
    return aligned


def _alignment_offsets(initial_tangents, aligned_tangents, seed_normals):
    """Angulo (radianes) que rota la tangente inicial hasta la alineada,
    medido en el plano tangente del seed con orientacion positiva CCW
    alrededor de la normal.
    """
    initial_bitangents = np.cross(seed_normals, initial_tangents)
    cos_theta = np.einsum("ij,ij->i", initial_tangents, aligned_tangents)
    sin_theta = np.einsum("ij,ij->i", initial_bitangents, aligned_tangents)
    return np.arctan2(sin_theta, cos_theta).astype(np.float32)


def _build_patch_buffers(
    vertices,
    faces,
    vertex_normals,
    face_patch,
    seed_positions,
    seed_tangents,
    seed_bitangents,
    full_geodesic_distances,
    face_is_on_boundary,
    palette,
    alignment_offsets,
):
    """Construye los buffers expandidos (un vértice por (cara, esquina)).

    La UV de cada vértice se computa con un log map discreto: el ángulo
    sale de proyectar (vértice - seed) sobre el marco tangente del seed;
    el radio es la distancia geodésica del vértice al seed del parche al
    que pertenece la cara. Como azimut y radio comparten origen, la UV
    es consistente incluso para vértices de frontera.

    Se devuelve además una bandera por esquina (cara en frontera de
    parche), coordenadas baricéntricas (para dibujar costura solo en
    caras frontera), un alpha uniforme = 1 y el offset de alineamiento
    del parche.
    """
    num_faces = len(faces)
    corner_vertex_index = faces.flatten()
    expanded_positions = vertices[corner_vertex_index]
    expanded_normals = vertex_normals[corner_vertex_index]

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

    expanded_radius = full_geodesic_distances[
        face_patch[:, None], faces
    ].flatten().astype(np.float64)
    patch_u = expanded_radius * np.cos(azimuth)
    patch_v = expanded_radius * np.sin(azimuth)
    patch_uv = np.stack([patch_u, patch_v], axis=1).astype(np.float32)

    expanded_patch_id = np.repeat(face_patch, 3)
    expanded_patch_color = palette[expanded_patch_id]
    expanded_alignment_offset = alignment_offsets[expanded_patch_id]

    expanded_boundary_flag = np.repeat(
        face_is_on_boundary.astype(np.float32), 3
    )
    barycentric_template = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    expanded_barycentric = np.tile(barycentric_template, (num_faces, 1))

    expanded_alpha = np.ones(3 * num_faces, dtype=np.float32)

    return (
        expanded_positions.astype(np.float32),
        expanded_normals.astype(np.float32),
        patch_uv,
        expanded_patch_color.astype(np.float32),
        expanded_boundary_flag,
        expanded_barycentric,
        expanded_alpha,
        expanded_alignment_offset.astype(np.float32),
        np.arange(3 * num_faces, dtype=np.uint32),
    )


def _face_boundary_mask(face_patch, face_adjacency):
    """Marca como `True` cada cara que tiene al menos un vecino directo
    perteneciente a un parche distinto. `face_adjacency` viene de trimesh
    con shape (M, 2): cada fila son dos caras que comparten arista.
    """
    different_patch = face_patch[face_adjacency[:, 0]] != face_patch[
        face_adjacency[:, 1]
    ]
    boundary = np.zeros(len(face_patch), dtype=bool)
    boundary[face_adjacency[different_patch, 0]] = True
    boundary[face_adjacency[different_patch, 1]] = True
    return boundary


def _compute_ring_distances(graph, vertex_patch, num_patches):
    """Para cada parche P, hops del vertice mas cercano que tenga
    vertex_patch == P. Usa dijkstra unweighted (BFS) sobre el grafo de
    aristas. Shape: (num_patches, num_vertices).
    """
    num_vertices = graph.shape[0]
    ring = np.full((num_patches, num_vertices), np.inf, dtype=np.float64)
    for p in range(num_patches):
        sources = np.where(vertex_patch == p)[0]
        if len(sources) == 0:
            continue
        ring[p] = dijkstra(
            graph, indices=sources, min_only=True, unweighted=True
        )
    return ring


def _build_skirt_buffers(
    vertices,
    faces,
    vertex_normals,
    face_patch,
    seed_positions,
    seed_tangents,
    seed_bitangents,
    full_geodesic_distances,
    ring_distances,
    palette,
    alignment_offsets,
    skirt_rings,
):
    """Para cada parche P, genera vertices duplicados de las caras que
    estan fuera de P (face_patch != P) pero tienen al menos un vertice a
    menos de `skirt_rings` hops de algun vertice del nucleo de P.

    Alfa por vertice = max(0, 1 - hops / skirt_rings):
    - Vertices del nucleo de P (hops=0): alpha = 1.
    - Vertices a 1 hop: alpha = 1 - 1/skirt_rings (0 si skirt_rings=1).
    - Mas alla del anillo: alpha = 0 (no aporta).

    Devuelve los mismos atributos que _build_patch_buffers (sin
    barycentric/boundary porque las caras de falda no se visualizan en
    modo diagnostico). barycentric se pone a (1,0,0) por relleno.
    """
    num_patches = len(seed_positions)
    all_positions = []
    all_normals = []
    all_uvs = []
    all_colors = []
    all_alphas = []
    all_offsets = []

    for p in range(num_patches):
        ring_p = ring_distances[p]
        in_radius_vertex = ring_p < skirt_rings  # vertices con alpha > 0
        face_in_radius = in_radius_vertex[faces].any(axis=1)
        skirt_mask = face_in_radius & (face_patch != p)
        if not skirt_mask.any():
            continue
        skirt_face_indices = np.where(skirt_mask)[0]
        skirt_faces = faces[skirt_face_indices]  # (M, 3)
        skirt_vertices = vertices[skirt_faces]   # (M, 3, 3)
        skirt_normals = vertex_normals[skirt_faces]  # (M, 3, 3)

        relative = skirt_vertices - seed_positions[p][None, None, :]
        tangent_component = np.einsum(
            "ijk,k->ij", relative, seed_tangents[p]
        )
        bitangent_component = np.einsum(
            "ijk,k->ij", relative, seed_bitangents[p]
        )
        azimuth = np.arctan2(bitangent_component, tangent_component)
        radius = full_geodesic_distances[p, skirt_faces]
        uv = np.stack(
            [radius * np.cos(azimuth), radius * np.sin(azimuth)], axis=-1
        )

        hops = ring_p[skirt_faces]
        alpha = np.maximum(0.0, 1.0 - hops / skirt_rings)

        num_skirt_faces = skirt_face_indices.shape[0]
        color = np.broadcast_to(
            palette[p][None, None, :], (num_skirt_faces, 3, 3)
        )
        offset = np.full((num_skirt_faces, 3), alignment_offsets[p])

        all_positions.append(skirt_vertices.reshape(-1, 3))
        all_normals.append(skirt_normals.reshape(-1, 3))
        all_uvs.append(uv.reshape(-1, 2))
        all_colors.append(np.array(color).reshape(-1, 3))
        all_alphas.append(alpha.reshape(-1))
        all_offsets.append(offset.reshape(-1))

    if not all_positions:
        empty = np.zeros((0,), dtype=np.float32)
        return (
            empty.reshape(0, 3), empty.reshape(0, 3),
            empty.reshape(0, 2), empty.reshape(0, 3),
            empty, empty.reshape(0, 3),
            empty, empty,
            np.zeros((0,), dtype=np.uint32),
        )

    positions = np.concatenate(all_positions).astype(np.float32)
    normals = np.concatenate(all_normals).astype(np.float32)
    uvs = np.concatenate(all_uvs).astype(np.float32)
    colors = np.concatenate(all_colors).astype(np.float32)
    alphas = np.concatenate(all_alphas).astype(np.float32)
    offsets = np.concatenate(all_offsets).astype(np.float32)
    indices = np.arange(len(positions), dtype=np.uint32)

    # boundary_flag y barycentric se rellenan: las caras de falda no se
    # muestran en modos diagnostico.
    boundary_flag = np.zeros(len(positions), dtype=np.float32)
    barycentric_template = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    num_skirt_faces_total = len(positions) // 3
    barycentric = np.tile(barycentric_template, (num_skirt_faces_total, 1))

    return (
        positions, normals, uvs, colors, boundary_flag, barycentric,
        alphas, offsets, indices,
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
@click.option(
    "--skirt-rings", "skirt_rings", type=int, default=1,
    help="Anillos de vertices que extienden cada parche para suavizar las "
         "costuras con alpha decreciente. 1 = anillo minimo (transicion en "
         "una arista); 2 = transicion mas suave; 0 = sin falda.",
)
def lapped_hatching(mesh_path, width, height, num_patches, skirt_rings):
    if skirt_rings < 0:
        raise click.BadParameter("--skirt-rings debe ser >= 0")

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
    vertex_patch, full_geodesic_distances = _partition_vertices(
        graph, seed_vertex_indices
    )
    face_patch = _assign_faces_to_patches(faces, vertex_patch)
    face_adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    face_is_on_boundary = _face_boundary_mask(face_patch, face_adjacency)
    print(
        f"  {face_is_on_boundary.sum()} caras frontera de "
        f"{len(faces)} totales"
    )

    print("Calculando marco tangente por seed...")
    vertices_f32 = vertices.astype(np.float32)
    seed_positions, seed_normals, seed_tangents_initial, _ = _seed_frames(
        seed_vertex_indices, vertices_f32, stable_normals, tangent_field
    )

    print("Alineando tangentes entre parches (parallel transport BFS)...")
    unique_patch_edges = _unique_patch_edges(face_patch, face_adjacency)
    seed_tangents_aligned = _aligned_seed_tangents(
        seed_normals, seed_tangents_initial, unique_patch_edges, num_patches
    )
    alignment_offsets = _alignment_offsets(
        seed_tangents_initial, seed_tangents_aligned, seed_normals
    )
    seed_bitangents_initial = np.cross(seed_normals, seed_tangents_initial)

    palette = _patch_color_palette(num_patches)

    print("Construyendo buffers expandidos por parche...")
    (
        expanded_positions,
        expanded_normals,
        patch_uv,
        patch_color,
        expanded_boundary_flag,
        expanded_barycentric,
        expanded_alpha,
        expanded_alignment_offset,
        expanded_indices,
    ) = _build_patch_buffers(
        vertices_f32,
        faces,
        stable_normals,
        face_patch,
        seed_positions,
        seed_tangents_initial,
        seed_bitangents_initial,
        full_geodesic_distances,
        face_is_on_boundary,
        palette,
        alignment_offsets,
    )
    print(f"  {len(expanded_positions)} vértices expandidos (nucleo)")

    if skirt_rings > 0:
        print(f"Calculando distancias en anillos (BFS por parche)...")
        ring_distances = _compute_ring_distances(
            graph, vertex_patch, num_patches
        )
        print(f"Construyendo buffers de falda con {skirt_rings} anillo(s)...")
        (
            skirt_positions, skirt_normals, skirt_uv, skirt_color,
            skirt_boundary_flag, skirt_barycentric,
            skirt_alpha, skirt_alignment_offset, skirt_indices,
        ) = _build_skirt_buffers(
            vertices_f32,
            faces,
            stable_normals,
            face_patch,
            seed_positions,
            seed_tangents_initial,
            seed_bitangents_initial,
            full_geodesic_distances,
            ring_distances,
            palette,
            alignment_offsets,
            skirt_rings,
        )
        print(f"  {len(skirt_positions)} vértices expandidos (falda)")
    else:
        skirt_positions = np.zeros((0, 3), dtype=np.float32)

    num_tam_levels = 6
    print(f"Generando volumen TAM ({num_tam_levels} niveles, 256x256)...")
    tam_volume_array = _generate_tam_volume(
        num_levels=num_tam_levels, size=256
    )

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
        normal=("f", expanded_normals.flatten()),
        patch_uv=("f", patch_uv.flatten()),
        patch_color=("f", patch_color.flatten()),
        boundary_flag=("f", expanded_boundary_flag),
        barycentric=("f", expanded_barycentric.flatten()),
        alpha=("f", expanded_alpha),
        alignment_offset=("f", expanded_alignment_offset),
    )

    skirt_gpu = None
    if len(skirt_positions) > 0:
        skirt_gpu = pipeline.vertex_list_indexed(
            len(skirt_positions),
            GL.GL_TRIANGLES,
            skirt_indices,
            position=("f", skirt_positions.flatten()),
            normal=("f", skirt_normals.flatten()),
            patch_uv=("f", skirt_uv.flatten()),
            patch_color=("f", skirt_color.flatten()),
            boundary_flag=("f", skirt_boundary_flag),
            barycentric=("f", skirt_barycentric.flatten()),
            alpha=("f", skirt_alpha),
            alignment_offset=("f", skirt_alignment_offset),
        )

    # Sube el volumen TAM como textura 3D. R8 single channel (ink amount).
    import ctypes
    tam_texture_id = GL.GLuint(0)
    GL.glGenTextures(1, ctypes.byref(tam_texture_id))
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_3D, tam_texture_id)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    tam_bytes = np.ascontiguousarray(
        (tam_volume_array * 255.0).astype(np.uint8)
    )
    tam_depth, tam_height, tam_width = tam_bytes.shape
    GL.glTexImage3D(
        GL.GL_TEXTURE_3D, 0, GL.GL_R8,
        tam_width, tam_height, tam_depth,
        0, GL.GL_RED, GL.GL_UNSIGNED_BYTE,
        tam_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    )
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glTexParameteri(
        GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE
    )

    # Direccion de la luz en espacio de mundo (fija). Vector hacia la luz.
    light_direction_vec = np.array([0.4, 0.7, 0.6], dtype=np.float32)
    light_direction_vec /= np.linalg.norm(light_direction_vec)

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
        # Cuántas capas de achurado se acumulan: 1 (solo trazos a 0 grados),
        # 2 (cross-hatch a +60), 3 (triple cross-hatch a 0, +60, -60).
        "hatching_layers": 3,
        # Tangentes alineadas entre parches vecinos via BFS parallel
        # transport. Por defecto activo: el achurado se ve mas continuo.
        "align_patches": True,
        # Render de la falda con alpha gradient sobre las caras nativas.
        # Por defecto activo: las costuras quedan suavizadas.
        "draw_skirt": skirt_gpu is not None,
        # Modo TAM: usa una textura 3D con niveles tonales y la
        # iluminacion difusa selecciona el nivel. Si esta apagado se
        # vuelve al achurado procedural sin iluminacion.
        "use_tam": True,
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
        "Drag: rotar  |  V: UV  |  P: parches  |  T: preview  |  L: TAM/proc  "
        "|  H: capas  |  A: alineacion  |  S: falda  |  , .: periodo  |  R: reset",
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
        elif state["use_tam"]:
            mode = "TAM con luz"
        else:
            mode = "achurado procedural"
        align_str = "alineadas" if state["align_patches"] else "Y-proyectadas"
        skirt_str = (
            f"falda {skirt_rings}" if state["draw_skirt"] else "sin falda"
        )
        status_label.text = (
            f"Modo: {mode}  |  parches: {num_patches}  |  "
            f"periodo: {state['stripe_period']:.3f}  |  "
            f"tangentes: {align_str}  |  {skirt_str}"
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
        elif symbol == keys.H:
            # 1 -> 2 -> 3 -> 1: el alumno ve cuanto aporta cada capa.
            state["hatching_layers"] = (state["hatching_layers"] % 3) + 1
        elif symbol == keys.A:
            state["align_patches"] = not state["align_patches"]
        elif symbol == keys.S:
            if skirt_gpu is not None:
                state["draw_skirt"] = not state["draw_skirt"]
        elif symbol == keys.L:
            state["use_tam"] = not state["use_tam"]
        elif symbol == keys.R:
            arcball.pose = np.linalg.inv(view)
        update_status()

    @window.event
    def on_draw():
        GL.glClearColor(0.88, 0.80, 0.66, 1.0)
        window.clear()
        background.draw()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_BLEND)

        current_view = np.linalg.inv(arcball.pose)

        # La textura 3D queda atada a la unidad 0; ambos pipelines (mesh y
        # overlay) referencian esa unidad.
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, tam_texture_id)

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
        stripe_half_width = (
            state["stripe_period"] * state["stripe_half_width_ratio"]
        )
        pipeline["stripe_period"] = state["stripe_period"]
        pipeline["stripe_half_width"] = stripe_half_width
        pipeline["show_uv_gradient"] = 1 if state["show_uv_gradient"] else 0
        pipeline["show_patches"] = 1 if state["show_patches"] else 0
        pipeline["hatching_layers"] = int(state["hatching_layers"])
        pipeline["align_patches"] = 1 if state["align_patches"] else 0
        pipeline["use_tam"] = 1 if state["use_tam"] else 0
        pipeline["tam_levels"] = int(num_tam_levels)
        pipeline["tile_world_size"] = float(state["stripe_period"] * 4.0)
        pipeline["light_direction"] = tuple(light_direction_vec.tolist())
        pipeline["tam_volume"] = 0

        # Pasada 1: caras nativas (alpha=1, depth write activo, sin blend).
        mesh_gpu.draw(GL.GL_TRIANGLES)

        # Pasada 2: faldas con alpha gradient. Mismas shaders y uniforms,
        # pero blend SRC_ALPHA, depth test LEQUAL (acepta misma profundidad
        # que las nativas) y depth write apagado (las faldas comparten
        # superficie con las nativas).
        in_hatching_mode = (
            not state["show_uv_gradient"] and not state["show_patches"]
        )
        if (
            skirt_gpu is not None
            and state["draw_skirt"]
            and in_hatching_mode
        ):
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glDepthFunc(GL.GL_LEQUAL)
            skirt_gpu.draw(GL.GL_TRIANGLES)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDepthFunc(GL.GL_LESS)
            GL.glDisable(GL.GL_BLEND)

        if state["show_texture_preview"]:
            GL.glDisable(GL.GL_DEPTH_TEST)
            overlay_pipeline.use()
            # El overlay muestra el atlas TAM (2 columnas x 3 filas).
            overlay_pipeline["tam_volume"] = 0
            overlay_pipeline["tam_levels"] = int(num_tam_levels)
            overlay_gpu.draw(GL.GL_TRIANGLES)

        with ui_overlay():
            status_label.draw()
            hint_label.draw()

    print("\nControles:")
    print("  Drag: rotar la cámara")
    print("  V: visualizar la UV de cada parche (colores)")
    print("  P: visualizar los parches con colores planos y costuras")
    print("  T: mostrar/ocultar preview del atlas TAM")
    print("  L: alternar TAM con iluminacion vs achurado procedural")
    print("  H: ciclar entre 1, 2 y 3 capas de achurado (modo procedural)")
    print("  A: alternar alineacion de tangentes entre parches")
    print("  S: alternar render de la falda con alpha gradient")
    print("  , / .: reducir / aumentar el período de los trazos")
    print("  R: reset de cámara")

    pyglet.app.run()
