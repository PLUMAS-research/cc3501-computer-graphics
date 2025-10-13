"""
Cálculo de curvaturas en mallas triangulares.
Implementación simplificada usando método de cotangente (Meyer et al. 2003)
"""
import numpy as np


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

def smooth_vertex_attribute(mesh, attribute, iterations=3, lambda_factor=0.5):
    """
    Suavizado Laplaciano de un atributo por vértice.
    
    Args:
        mesh: trimesh object
        attribute: array (N,) con valores por vértice
        iterations: número de iteraciones de suavizado
        lambda_factor: factor de suavizado (0-1, menor = más suave)
    
    Returns:
        attribute_smooth: array suavizado
    """
    N = len(mesh.vertices)
    attribute_smooth = attribute.copy()
    
    # Construir vecindad
    vertex_neighbors = [[] for _ in range(N)]
    for face in mesh.faces:
        for i in range(3):
            v1, v2, v3 = face[i], face[(i+1)%3], face[(i+2)%3]
            if v2 not in vertex_neighbors[v1]:
                vertex_neighbors[v1].append(v2)
            if v3 not in vertex_neighbors[v1]:
                vertex_neighbors[v1].append(v3)
    
    for _ in range(iterations):
        attribute_new = attribute_smooth.copy()
        
        for i in range(N):
            neighbors = vertex_neighbors[i]
            if len(neighbors) == 0:
                continue
            
            # Promedio de vecinos
            neighbor_avg = np.mean(attribute_smooth[neighbors])
            
            # Suavizado: mezcla entre valor actual y promedio
            attribute_new[i] = (1 - lambda_factor) * attribute_smooth[i] + lambda_factor * neighbor_avg
        
        attribute_smooth = attribute_new
    
    return attribute_smooth