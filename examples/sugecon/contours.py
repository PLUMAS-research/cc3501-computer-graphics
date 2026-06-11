"""
Suggestive contours con curvatura radial.
"""
import numpy as np


def compute_radial_curvature(mesh, camera_pos, k1, k2, d1, d2):
    """
    Calcula la curvatura radial κᵣ para cada vértice.
    κᵣ = k₁·cos²(θ) + k₂·sin²(θ)
    
    donde θ es el ángulo entre el vector de vista proyectado
    y la dirección principal d1.
    
    Args:
        mesh: trimesh.Trimesh object
        camera_pos: posición de la cámara (x, y, z)
        k1, k2: curvaturas principales por vértice
        d1, d2: direcciones principales por vértice
    
    Returns:
        kr: curvatura radial por vértice (N,)
    """
    vertices = mesh.vertices
    normals = mesh.vertex_normals
    N = len(vertices)
    
    # Vector de vista por vértice (de vértice hacia cámara)
    view_vectors = camera_pos - vertices  # (N, 3)
    view_vectors = view_vectors / np.linalg.norm(view_vectors, axis=1, keepdims=True)
    
    # Proyectar vector de vista al plano tangente
    # w = v - (v·n)n
    ndotv = np.sum(normals * view_vectors, axis=1, keepdims=True)  # (N, 1)
    w = view_vectors - ndotv * normals  # (N, 3)
    
    # Normalizar w
    w_norm = np.linalg.norm(w, axis=1, keepdims=True)
    w = w / (w_norm + 1e-10)  # evitar división por cero
    
    # Calcular ángulo θ con dirección principal d1
    # cos(θ) = w · d1
    cos_theta = np.sum(w * d1, axis=1)  # (N,)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # sin²(θ) = 1 - cos²(θ)
    cos2_theta = cos_theta**2
    sin2_theta = 1 - cos2_theta
    
    # Curvatura radial
    kr = k1 * cos2_theta + k2 * sin2_theta
    
    return kr


def extract_contour_points(mesh, kr, dkr, threshold=1e-6):
    # Esta función corre cada frame (kr depende de la cámara), así que el test
    # de cruce por cero se vectoriza sobre todas las aristas a la vez en vez de
    # iterarlas en Python.
    edges = mesh.edges_unique
    v1 = edges[:, 0]
    v2 = edges[:, 1]
    kr1 = kr[v1]
    kr2 = kr[v2]

    # Aristas con cruce por cero (kr de distinto signo en los extremos) cuya
    # derivada direccional interpolada supera el umbral de visibilidad.
    crossing = kr1 * kr2 < 0
    abs1 = np.abs(kr1)
    abs2 = np.abs(kr2)
    denom = abs1 + abs2
    denom[denom == 0] = 1.0
    t = abs1 / denom
    dkr_interp = (1 - t) * dkr[v1] + t * dkr[v2]
    selected = crossing & (dkr_interp > threshold)

    if not np.any(selected):
        return np.array([]), [], np.array([])

    ts = t[selected][:, np.newaxis]
    p1 = mesh.vertices[v1[selected]]
    p2 = mesh.vertices[v2[selected]]
    contour_points = (1 - ts) * p1 + ts * p2

    # Distancia a cero del kr interpolado: alimenta la transparencia.
    kr_interp = (1 - t[selected]) * kr1[selected] + t[selected] * kr2[selected]
    contour_kr_values = np.abs(kr_interp)

    contour_edges = chain_contour_segments(contour_points, max_distance=0.1)

    return contour_points, contour_edges, contour_kr_values


def chain_contour_segments(points, max_distance=0.05):
    if len(points) < 2:
        return []
    
    from scipy.spatial import cKDTree
    
    tree = cKDTree(points)
    visited = set()
    edges = []
    
    # Para cada punto, intentar formar una cadena
    for start_idx in range(len(points)):
        if start_idx in visited:
            continue
        
        # Iniciar cadena
        chain = [start_idx]
        visited.add(start_idx)
        current = start_idx
        
        # Extender cadena hacia adelante
        while True:
            distances, neighbors = tree.query(points[current], k=4)
            found_next = False
            
            for dist, neighbor in zip(distances[1:], neighbors[1:]):
                if neighbor not in visited and dist < max_distance:
                    chain.append(neighbor)
                    visited.add(neighbor)
                    current = neighbor
                    found_next = True
                    break
            
            if not found_next:
                break
        
        # Convertir cadena a edges
        for i in range(len(chain) - 1):
            edges.append((chain[i], chain[i+1]))
    
    return edges

def extract_silhouette_edges(mesh, camera_pos):
    vertices = mesh.vertices
    faces = mesh.faces
    face_normals = mesh.face_normals
    
    # Calcular visibilidad de cada cara
    face_centers = np.mean(vertices[faces], axis=1)
    view_vectors = camera_pos - face_centers
    view_vectors = view_vectors / np.linalg.norm(view_vectors, axis=1, keepdims=True)
    
    face_dot = np.sum(face_normals * view_vectors, axis=1)
    face_visibility = face_dot > 0
    
    # Usar face_adjacency para obtener pares de caras adyacentes
    adjacency = mesh.face_adjacency
    
    # Filtrar solo pares con visibilidad opuesta
    vis_pairs = face_visibility[adjacency]
    is_silhouette = vis_pairs[:, 0] != vis_pairs[:, 1]
    silhouette_face_pairs = adjacency[is_silhouette]
    
    if len(silhouette_face_pairs) == 0:
        return np.array([]), []
    
    # Para cada par de caras adyacentes, encontrar la arista compartida
    silhouette_edges_list = []
    for f1, f2 in silhouette_face_pairs:
        # Encontrar vértices compartidos entre las dos caras
        shared = np.intersect1d(faces[f1], faces[f2])
        if len(shared) == 2:
            silhouette_edges_list.append(shared)
    
    if len(silhouette_edges_list) == 0:
        return np.array([]), []
    
    silhouette_edges_array = np.array(silhouette_edges_list)
    
    # Crear lista de puntos duplicados para GL_LINES
    silhouette_points = vertices[silhouette_edges_array.flatten()]
    num_edges = len(silhouette_edges_array)
    silhouette_edges = np.arange(num_edges * 2).reshape(-1, 2)
    
    return silhouette_points, silhouette_edges

def compute_radial_curvature_derivative(mesh, camera_pos, k1, k2, d1, d2, kr):
    """
    Calcula la derivada de kr en dirección de vista: D_w kr.
    Aproximación mediante diferencias finitas.
    """
    vertices = mesh.vertices
    normals = mesh.vertex_normals
    N = len(vertices)
    
    # Vector de vista
    view_vectors = camera_pos - vertices
    view_vectors = view_vectors / np.linalg.norm(view_vectors, axis=1, keepdims=True)
    
    # Proyectar al plano tangente
    ndotv = np.sum(normals * view_vectors, axis=1, keepdims=True)
    w = view_vectors - ndotv * normals
    w_norm = np.linalg.norm(w, axis=1, keepdims=True)
    w = w / (w_norm + 1e-10)
    
    # Aproximar derivada usando vecinos
    dkr = np.zeros(N)
    
    for i in range(N):
        neighbors = mesh.vertex_neighbors[i]
        if len(neighbors) == 0:
            continue
        
        # Calcular diferencia promedio con vecinos en dirección w
        kr_diffs = []
        for j in neighbors:
            diff_vec = vertices[j] - vertices[i]
            proj = np.dot(diff_vec, w[i])
            if abs(proj) > 1e-8:
                kr_diffs.append((kr[j] - kr[i]) / proj)
        
        if len(kr_diffs) > 0:
            dkr[i] = np.mean(kr_diffs)
    
    return dkr