"""
Cálculo de curvaturas en mallas triangulares.
Implementación simplificada usando método de cotangente (Meyer et al. 2003)
"""
import numpy as np
import scipy.sparse as sp


def compute_vertex_areas(mesh):
    """
    Calcula el área de Voronoi para cada vértice.
    """
    vertex_areas = np.zeros(len(mesh.vertices))
    
    # Distribuir 1/3 del área de cada cara a cada vértice
    face_areas_third = mesh.area_faces / 3.0
    
    # Acumular en cada vértice usando add.at (más rápido que loops)
    for i in range(3):
        np.add.at(vertex_areas, mesh.faces[:, i], face_areas_third)
    
    return vertex_areas


def compute_cotangent_weights(mesh):
    cotangent_weights = {}
    
    # Vectorizar por cara
    faces = mesh.faces
    vertices = mesh.vertices
    
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            v_opposite = face[(i + 2) % 3]
            
            edge1 = vertices[v1] - vertices[v_opposite]
            edge2 = vertices[v2] - vertices[v_opposite]
            
            dot = np.dot(edge1, edge2)
            norm_prod = np.linalg.norm(edge1) * np.linalg.norm(edge2)
            
            if norm_prod > 1e-10:
                cos_angle = np.clip(dot / norm_prod, -1.0, 1.0)
                sin_angle = np.sqrt(1 - cos_angle**2)
                cot = cos_angle / (sin_angle + 1e-10)
                
                edge = tuple(sorted([v1, v2]))
                cotangent_weights[edge] = cotangent_weights.get(edge, 0.0) + cot
    
    return cotangent_weights


def compute_mean_curvature(mesh):
    vertices = mesh.vertices
    N = len(vertices)
    
    print(f"    Calculando áreas de vértices...")
    vertex_areas = compute_vertex_areas(mesh)
    
    print(f"    Calculando pesos cotangente...")
    cotangent_weights = compute_cotangent_weights(mesh)
    
    print(f"    Computando Laplaciano por vértice...")
    H = np.zeros(N)
    
    # Construir vecindad de forma más eficiente usando edges
    print(f"    Construyendo vecindad...")
    vertex_neighbors = [[] for _ in range(N)]
    for edge, weight in cotangent_weights.items():
        v1, v2 = edge
        vertex_neighbors[v1].append((v2, weight))
        vertex_neighbors[v2].append((v1, weight))
    
    # Computar Laplaciano
    for i in range(N):
        if i % 10000 == 0 and i > 0:
            print(f"      Procesado {i}/{N} vértices...")
        
        neighbors = vertex_neighbors[i]
        if len(neighbors) == 0:
            continue
        
        # Vectorizar el cálculo del laplaciano
        neighbor_indices = [n[0] for n in neighbors]
        weights = np.array([n[1] for n in neighbors])
        
        neighbor_positions = vertices[neighbor_indices]
        diff = neighbor_positions - vertices[i]
        
        laplacian = np.sum(weights[:, np.newaxis] * diff, axis=0)
        laplacian_norm = np.linalg.norm(laplacian)
        
        if vertex_areas[i] > 1e-10:
            H[i] = laplacian_norm / (4.0 * vertex_areas[i])
    
    return H


def compute_gaussian_curvature(mesh):
    vertices = mesh.vertices
    N = len(vertices)
    K = np.zeros(N)
    
    vertex_areas = compute_vertex_areas(mesh)
    
    print(f"    Computando defecto angular por vértice...")
    
    # Pre-calcular todas las caras adyacentes
    vertex_faces_list = [[] for _ in range(N)]
    for face_idx, face in enumerate(mesh.faces):
        for v in face:
            vertex_faces_list[v].append(face_idx)
    
    for i in range(N):
        if i % 10000 == 0 and i > 0:
            print(f"      Procesado {i}/{N} vértices...")
        
        angle_sum = 0.0
        
        for face_idx in vertex_faces_list[i]:
            face = mesh.faces[face_idx]
            local_idx = np.where(face == i)[0][0]
            
            v_prev = face[(local_idx - 1) % 3]
            v_next = face[(local_idx + 1) % 3]
            
            edge1 = vertices[v_prev] - vertices[i]
            edge2 = vertices[v_next] - vertices[i]
            
            norm1 = np.linalg.norm(edge1)
            norm2 = np.linalg.norm(edge2)
            
            if norm1 > 1e-10 and norm2 > 1e-10:
                cos_angle = np.clip(np.dot(edge1, edge2) / (norm1 * norm2), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angle_sum += angle
        
        angle_defect = 2 * np.pi - angle_sum
        
        if vertex_areas[i] > 1e-10:
            K[i] = angle_defect / vertex_areas[i]
    
    return K


def compute_principal_curvatures(H, K):
    """
    Calcula curvaturas principales k1 y k2 desde H y K.
    k1 = H + sqrt(H² - K)
    k2 = H - sqrt(H² - K)
    """
    discriminant = H**2 - K
    # Evitar valores negativos por ruido numérico
    discriminant = np.maximum(discriminant, 0)
    sqrt_disc = np.sqrt(discriminant)
    
    k1 = H + sqrt_disc
    k2 = H - sqrt_disc
    
    return k1, k2


def approximate_principal_directions(mesh, k1, k2):
    """
    Aproximación de direcciones principales usando vecinos.
    Basado en gradientes de curvatura.
    """
    vertices = mesh.vertices
    normals = mesh.vertex_normals
    N = len(vertices)
    
    d1 = np.zeros((N, 3))
    d2 = np.zeros((N, 3))
    
    # Construir vecindad
    vertex_neighbors = [set() for _ in range(N)]
    for face in mesh.faces:
        for i in range(3):
            v1, v2, v3 = face[i], face[(i+1)%3], face[(i+2)%3]
            vertex_neighbors[v1].add(v2)
            vertex_neighbors[v1].add(v3)
    
    for i in range(N):
        n = normals[i]
        neighbors = list(vertex_neighbors[i])
        
        if len(neighbors) < 2:
            # Caso degenerado: usar base arbitraria
            if abs(n[0]) < 0.9:
                u = np.cross(n, [1, 0, 0])
            else:
                u = np.cross(n, [0, 1, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            d1[i] = u
            d2[i] = v
            continue
        
        # Aproximar dirección de máxima curvatura usando gradiente de k1
        grad_k1 = np.zeros(3)
        for j in neighbors:
            edge_vec = vertices[j] - vertices[i]
            # Proyectar al plano tangente
            edge_tangent = edge_vec - np.dot(edge_vec, n) * n
            norm = np.linalg.norm(edge_tangent)
            if norm > 1e-8:
                edge_tangent /= norm
                # Aproximar gradiente
                grad_k1 += (k1[j] - k1[i]) * edge_tangent
        
        grad_norm = np.linalg.norm(grad_k1)
        
        if grad_norm > 1e-8:
            # d1 alineado con gradiente (perpendicular a isolíneas)
            d1[i] = grad_k1 / grad_norm
            # d2 perpendicular a d1 en plano tangente
            d2[i] = np.cross(n, d1[i])
        else:
            # Sin gradiente claro: usar base arbitraria
            if abs(n[0]) < 0.9:
                u = np.cross(n, [1, 0, 0])
            else:
                u = np.cross(n, [0, 1, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            d1[i] = u
            d2[i] = v
    
    return d1, d2

def build_smoothing_operator(mesh):
    """
    Construye el operador de promediado de vecinos como matriz dispersa.

    Devuelve la matriz de adyacencia normalizada por filas A, donde
    (A @ phi)[i] es el promedio de phi sobre los vecinos de i. Con ella el
    suavizado laplaciano de un atributo es un producto matriz-vector, en vez
    de un doble bucle en Python. El operador depende solo de la topología de la
    malla, así que se construye una vez y se reusa en cada iteración y cada vez
    que cambian las perillas de difusión.
    """
    N = len(mesh.vertices)
    edges = mesh.edges_unique
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(N, N)
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    return sp.diags(1.0 / degree) @ adjacency


def smooth_vertex_attribute(mesh, attribute, iterations=3, lambda_factor=0.5, operator=None):
    """
    Suavizado laplaciano de un atributo por vértice.

    Es una difusión sobre la variedad: cada iteración mezcla el valor del
    vértice con el promedio de sus vecinos,

        phi_i <- (1 - lambda) * phi_i + lambda * mean_{j in N(i)} phi_j,

    que es Euler explícito de la ecuación del calor con el laplaciano de grafo
    (compara con `difusion_calor`, que resuelve la misma ecuación sobre una
    grilla 2D). El promedio de vecinos se escribe como `operator @ phi`.

    Args:
        mesh: trimesh object
        attribute: array (N,) con valores por vértice
        iterations: número de pasos de difusión
        lambda_factor: factor de difusión (0-1, mayor = más suavizado por paso)
        operator: matriz de promediado precalculada (ver build_smoothing_operator);
                  si es None se construye al vuelo

    Returns:
        attribute_smooth: array suavizado
    """
    averaging = operator if operator is not None else build_smoothing_operator(mesh)
    phi = attribute.astype(float).copy()
    for _ in range(iterations):
        phi = (1.0 - lambda_factor) * phi + lambda_factor * (averaging @ phi)
    return phi