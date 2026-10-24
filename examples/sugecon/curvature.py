"""
Curvaturas de una malla de triángulos y sus derivadas.

Sigue a Rusinkiewicz (2004), "Estimating Curvatures and Their Derivatives on
Triangle Meshes", que es el método de la implementación de referencia de los
suggestive contours (rtsc, sobre la biblioteca trimesh2). La idea es la misma
dos veces:

- La segunda forma fundamental II (el tensor de curvatura) mide cuánto cambia
  la normal al moverse sobre la superficie. En cada cara se ajusta, por mínimos
  cuadrados, el II que mejor explica la diferencia de las normales entre los
  extremos de cada arista.
- El tensor C de derivadas de la curvatura mide cuánto cambia II. En cada cara
  se ajusta el C que mejor explica la diferencia de II entre los extremos de
  cada arista.

Cada cara reparte su tensor entre sus tres vértices, ponderado por el área de
Voronoi de cada esquina, y cada vértice termina con sus curvaturas principales
k1 y k2 (|k1| >= |k2|), sus direcciones principales d1 y d2 y su tensor C,
expresado en la base (d1, d2).
"""
import numpy as np
import scipy.sparse as sp


def compute_vertex_normals(vertices, faces):
    """Normales por vértice con los pesos de Max (1999).

    Cada cara aporta su normal sin normalizar dividida por el producto de los
    cuadrados de las dos aristas que llegan al vértice. Para una malla que
    muestrea una esfera, estos pesos recuperan la normal exacta.
    """
    p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
    a, b, c = p0 - p1, p1 - p2, p2 - p0
    l2a, l2b, l2c = (np.einsum("ij,ij->i", e, e) for e in (a, b, c))
    validas = (l2a > 0) & (l2b > 0) & (l2c > 0)
    normal_cara = np.cross(a, b)
    normales = np.zeros_like(vertices)
    for esquina, peso in enumerate((l2a * l2c, l2b * l2a, l2c * l2b)):
        aporte = normal_cara[validas] / peso[validas, np.newaxis]
        for eje in range(3):
            normales[:, eje] += np.bincount(faces[validas, esquina], aporte[:, eje], len(vertices))
    return _normalizar(normales)


def compute_corner_areas(vertices, faces):
    """Área de Voronoi de cada esquina de cada cara, recortada al triángulo, y
    su suma por vértice.

    En un triángulo agudo, cada esquina se queda con la región más cercana a
    ella que a las otras dos, y las tres regiones se juntan en el circuncentro.
    En uno obtuso el circuncentro cae fuera del triángulo: cada esquina aguda
    se queda con el triángulo rectángulo que corta la mediatriz de su arista
    hacia la esquina obtusa, y la obtusa, con el resto.
    """
    e = [vertices[faces[:, (i + 2) % 3]] - vertices[faces[:, (i + 1) % 3]] for i in range(3)]
    area = 0.5 * np.linalg.norm(np.cross(e[0], e[1]), axis=1)
    l2 = [np.einsum("ij,ij->i", arista, arista) for arista in e]
    # pesos baricéntricos del circuncentro: uno no positivo indica un ángulo obtuso
    bcw = [l2[i] * (l2[(i + 1) % 3] + l2[(i + 2) % 3] - l2[i]) for i in range(3)]
    esquinas = np.zeros((len(faces), 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        escala = 0.5 * area / (bcw[0] + bcw[1] + bcw[2])
        for i in range(3):
            esquinas[:, i] = escala * (bcw[(i + 1) % 3] + bcw[(i + 2) % 3])
        for obtuso in range(3):
            caso = bcw[obtuso] <= 0
            for i in range(obtuso):
                caso &= bcw[i] > 0
            siguiente, anterior = (obtuso + 1) % 3, (obtuso + 2) % 3
            esquinas[caso, siguiente] = (-0.25 * l2[anterior] * area / np.einsum("ij,ij->i", e[obtuso], e[anterior]))[caso]
            esquinas[caso, anterior] = (-0.25 * l2[siguiente] * area / np.einsum("ij,ij->i", e[obtuso], e[siguiente]))[caso]
            esquinas[caso, obtuso] = area[caso] - esquinas[caso, siguiente] - esquinas[caso, anterior]
    esquinas = np.nan_to_num(esquinas)
    areas = np.bincount(faces.ravel(), esquinas.ravel(), len(vertices))
    return esquinas, areas


def _normalizar(vectores):
    largo = np.linalg.norm(vectores, axis=-1, keepdims=True)
    largo[largo == 0] = 1.0
    return vectores / largo


def _rotar_base(u, v, normal):
    """Gira la base tangente (u, v) para que quede perpendicular a `normal`.

    El giro es el mínimo que lleva la normal de la base, u x v, a `normal`: la
    componente de cada vector fuera del plano de las dos normales no cambia.
    """
    normal_vieja = np.cross(u, v)
    ndot = np.einsum("ij,ij->i", normal_vieja, normal)[:, np.newaxis]
    perpendicular = normal - ndot * normal_vieja
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (normal_vieja + normal) / (1.0 + ndot)
    nuevo_u = u - delta * np.einsum("ij,ij->i", u, perpendicular)[:, np.newaxis]
    nuevo_v = v - delta * np.einsum("ij,ij->i", v, perpendicular)[:, np.newaxis]
    opuesta = ndot[:, 0] <= -1.0
    nuevo_u[opuesta], nuevo_v[opuesta] = -u[opuesta], -v[opuesta]
    return nuevo_u, nuevo_v


def _cambio_de_base(u_viejo, v_viejo, u_nuevo, v_nuevo):
    """Cosenos (u1, v1, u2, v2) entre la base nueva, girada al plano de la
    vieja, y la base vieja."""
    u_girado, v_girado = _rotar_base(u_nuevo, v_nuevo, np.cross(u_viejo, v_viejo))
    return (np.einsum("ij,ij->i", u_girado, u_viejo), np.einsum("ij,ij->i", u_girado, v_viejo),
            np.einsum("ij,ij->i", v_girado, u_viejo), np.einsum("ij,ij->i", v_girado, v_viejo))


def _proyectar_curvatura(u_viejo, v_viejo, ku, kuv, kv, u_nuevo, v_nuevo):
    """Expresa el tensor de curvatura [[ku, kuv], [kuv, kv]] en otra base."""
    u1, v1, u2, v2 = _cambio_de_base(u_viejo, v_viejo, u_nuevo, v_nuevo)
    return (ku * u1 * u1 + kuv * 2 * u1 * v1 + kv * v1 * v1,
            ku * u1 * u2 + kuv * (u1 * v2 + u2 * v1) + kv * v1 * v2,
            ku * u2 * u2 + kuv * 2 * u2 * v2 + kv * v2 * v2)


def _proyectar_derivada(u_viejo, v_viejo, C, u_nuevo, v_nuevo):
    """Expresa el tensor simétrico C = (C111, C112, C122, C222) en otra base."""
    u1, v1, u2, v2 = _cambio_de_base(u_viejo, v_viejo, u_nuevo, v_nuevo)
    c0, c1, c2, c3 = C.T
    return np.stack([
        c0 * u1**3 + c1 * 3 * u1 * u1 * v1 + c2 * 3 * u1 * v1 * v1 + c3 * v1**3,
        c0 * u1 * u1 * u2 + c1 * (u1 * u1 * v2 + 2 * u2 * u1 * v1)
        + c2 * (u2 * v1 * v1 + 2 * u1 * v1 * v2) + c3 * v1 * v1 * v2,
        c0 * u1 * u2 * u2 + c1 * (u2 * u2 * v1 + 2 * u1 * u2 * v2)
        + c2 * (u1 * v2 * v2 + 2 * u2 * v2 * v1) + c3 * v1 * v2 * v2,
        c0 * u2**3 + c1 * 3 * u2 * u2 * v2 + c2 * 3 * u2 * v2 * v2 + c3 * v2**3,
    ], axis=1)


def _base_de_cara(vertices, faces):
    """Aristas de cada cara (la arista i es la opuesta al vértice i) y una base
    ortonormal (t, b) de su plano."""
    e = [vertices[faces[:, (i + 2) % 3]] - vertices[faces[:, (i + 1) % 3]] for i in range(3)]
    t = _normalizar(e[0])
    b = _normalizar(np.cross(np.cross(e[0], e[1]), t))
    return e, t, b


def _resolver(matrices, lados):
    """Resuelve un sistema por cara. Devuelve también qué caras lo tenían
    resoluble: una cara degenerada queda con solución cero y fuera del reparto."""
    escala = np.abs(matrices).max(axis=(1, 2))
    resolubles = np.abs(np.linalg.det(matrices)) > 1e-12 * escala ** matrices.shape[1]
    soluciones = np.zeros_like(lados)
    soluciones[resolubles] = np.linalg.solve(matrices[resolubles], lados[resolubles][..., np.newaxis])[..., 0]
    return soluciones, resolubles


def _repartir(faces, valores, pesos, n_vertices):
    """Suma en cada vértice los valores de sus esquinas, ponderados."""
    total = np.zeros((n_vertices,) + valores.shape[2:])
    for esquina in range(3):
        aporte = valores[:, esquina] * pesos[:, esquina].reshape((-1,) + (1,) * (valores.ndim - 2))
        if aporte.ndim == 1:
            total += np.bincount(faces[:, esquina], aporte, n_vertices)
        else:
            for k in range(aporte.shape[1]):
                total[:, k] += np.bincount(faces[:, esquina], aporte[:, k], n_vertices)
    return total


def compute_curvatures(vertices, faces, normals, corner_areas, point_areas):
    """Curvaturas y direcciones principales por vértice (Rusinkiewicz 2004).

    Devuelve k1, k2, d1 y d2 con |k1| >= |k2|. Con las normales hacia afuera,
    la curvatura es positiva donde la superficie es convexa.
    """
    n_vertices = len(vertices)
    # base tangente inicial de cada vértice, a partir de una arista que sale de él
    d1 = np.zeros_like(vertices)
    for i in range(3):
        d1[faces[:, i]] = vertices[faces[:, (i + 1) % 3]] - vertices[faces[:, i]]
    d1 = _normalizar(np.cross(d1, normals))
    d2 = np.cross(normals, d1)

    # ajuste de II por cara: la diferencia de normales a lo largo de cada arista
    # (dnu, dnv) debe ser II (u, v), con (u, v) la arista en la base de la cara
    e, t, b = _base_de_cara(vertices, faces)
    sistema = np.zeros((len(faces), 3, 3))
    lado = np.zeros((len(faces), 3))
    for j in range(3):
        u = np.einsum("ij,ij->i", e[j], t)
        v = np.einsum("ij,ij->i", e[j], b)
        dn = normals[faces[:, (j + 2) % 3]] - normals[faces[:, (j + 1) % 3]]
        dnu = np.einsum("ij,ij->i", dn, t)
        dnv = np.einsum("ij,ij->i", dn, b)
        sistema[:, 0, 0] += u * u
        sistema[:, 0, 1] += u * v
        sistema[:, 2, 2] += v * v
        lado[:, 0] += dnu * u
        lado[:, 1] += dnu * v + dnv * u
        lado[:, 2] += dnv * v
    sistema[:, 1, 1] = sistema[:, 0, 0] + sistema[:, 2, 2]
    sistema[:, 1, 2] = sistema[:, 0, 1]
    sistema[:, 1, 0] = sistema[:, 0, 1]
    sistema[:, 2, 1] = sistema[:, 1, 2]
    tensor_cara, validas = _resolver(sistema, lado)

    # cada esquina lleva el tensor de su cara a la base de su vértice
    esquinas = np.zeros((len(faces), 3, 3))
    for j in range(3):
        vj = faces[:, j]
        esquinas[:, j] = np.stack(_proyectar_curvatura(t, b, *tensor_cara.T, d1[vj], d2[vj]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pesos = np.nan_to_num(corner_areas / point_areas[faces]) * validas[:, np.newaxis]
    ku, kuv, kv = _repartir(faces, esquinas, pesos, n_vertices).T

    # diagonalización por un giro de Jacobi
    with np.errstate(divide="ignore", invalid="ignore"):
        h = 0.5 * (kv - ku) / kuv
        tt = np.where(h < 0, 1.0 / (h - np.sqrt(1 + h * h)), 1.0 / (h + np.sqrt(1 + h * h)))
    tt[kuv == 0] = 0.0
    c = 1.0 / np.sqrt(1 + tt * tt)
    s = tt * c
    k1 = ku - tt * kuv
    k2 = kv + tt * kuv
    cambia = np.abs(k1) < np.abs(k2)
    k1[cambia], k2[cambia] = k2[cambia], k1[cambia].copy()
    d1_nueva = np.where(cambia[:, np.newaxis], s[:, np.newaxis] * d1 + c[:, np.newaxis] * d2,
                        c[:, np.newaxis] * d1 - s[:, np.newaxis] * d2)
    return k1, k2, d1_nueva, np.cross(normals, d1_nueva)


def compute_curvature_derivatives(vertices, faces, k1, k2, d1, d2, corner_areas, point_areas):
    """Tensor C de derivadas de la curvatura por vértice, en la base (d1, d2).

    Sus cuatro componentes (C111, C112, C122, C222) dan la derivada de II en
    cualquier dirección (u, v): [[C111 u + C112 v, C112 u + C122 v],
    [C112 u + C122 v, C122 u + C222 v]].
    """
    e, t, b = _base_de_cara(vertices, faces)
    # II de cada vértice expresado en la base de la cara
    tensores = [np.stack(_proyectar_curvatura(d1[faces[:, j]], d2[faces[:, j]], k1[faces[:, j]],
                                              np.zeros(len(faces)), k2[faces[:, j]], t, b), axis=1)
                for j in range(3)]
    sistema = np.zeros((len(faces), 4, 4))
    lado = np.zeros((len(faces), 4))
    for j in range(3):
        dII = tensores[(j + 2) % 3] - tensores[(j + 1) % 3]
        u = np.einsum("ij,ij->i", e[j], t)
        v = np.einsum("ij,ij->i", e[j], b)
        sistema[:, 0, 0] += u * u
        sistema[:, 0, 1] += u * v
        sistema[:, 3, 3] += v * v
        lado[:, 0] += u * dII[:, 0]
        lado[:, 1] += v * dII[:, 0] + 2 * u * dII[:, 1]
        lado[:, 2] += 2 * v * dII[:, 1] + u * dII[:, 2]
        lado[:, 3] += v * dII[:, 2]
    sistema[:, 1, 1] = 2 * sistema[:, 0, 0] + sistema[:, 3, 3]
    sistema[:, 1, 2] = 2 * sistema[:, 0, 1]
    sistema[:, 2, 2] = sistema[:, 0, 0] + 2 * sistema[:, 3, 3]
    sistema[:, 2, 3] = sistema[:, 0, 1]
    for fila, columna in ((0, 1), (1, 2), (2, 3)):
        sistema[:, columna, fila] = sistema[:, fila, columna]
    C_cara, validas = _resolver(sistema, lado)

    esquinas = np.stack([_proyectar_derivada(t, b, C_cara, d1[faces[:, j]], d2[faces[:, j]])
                         for j in range(3)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pesos = np.nan_to_num(corner_areas / point_areas[faces]) * validas[:, np.newaxis]
    return _repartir(faces, esquinas, pesos, len(vertices))


def compute_feature_size(k1, k2, radius):
    """Tamaño de rasgo de la malla, la longitud que vuelve adimensionales los
    umbrales: una curvatura se multiplica por él y su derivada, por su cuadrado.

    Es el de rtsc: 0,01 dividido por el percentil 10 de |k1| y |k2|, con un
    tope del 5 % del radio de la esfera envolvente. rtsc estima el percentil
    con 500 vértices al azar; aquí se usan todos, así no depende del orden de
    los vértices en el archivo.
    """
    percentil = np.percentile(np.abs(np.concatenate([k1, k2])), 10)
    return min(0.01 / percentil if percentil > 0 else np.inf, 0.05 * radius)


def build_smoothing_operator(mesh):
    """
    Construye el operador de promediado de vecinos como matriz dispersa.

    Devuelve la matriz de adyacencia normalizada por filas A, donde
    (A @ phi)[i] es el promedio de phi sobre los vecinos de i. Con ella el
    suavizado laplaciano de un atributo es un producto matriz-vector, en vez
    de un doble bucle en Python. El operador depende solo de la topología de la
    malla, así que se construye una vez y se reusa en cada iteración.
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


def smooth_normals(normals, operator, iterations, lambda_factor=0.5):
    """
    Difusión de las normales sobre la malla.

    Cada iteración mezcla la normal del vértice con el promedio de sus vecinos,

        n_i <- (1 - lambda) * n_i + lambda * mean_{j in N(i)} n_j,

    y la vuelve a normalizar. Es Euler explícito de la ecuación del calor con el
    laplaciano de grafo (compara con `difusion_calor`, que resuelve la misma
    ecuación sobre una grilla 2D). Como las curvaturas se estiman de cómo
    cambian las normales, difundirlas suaviza las curvaturas y sus derivadas.
    """
    suavizadas = normals.copy()
    for _ in range(iterations):
        suavizadas = _normalizar((1.0 - lambda_factor) * suavizadas + lambda_factor * (operator @ suavizadas))
    return suavizadas
