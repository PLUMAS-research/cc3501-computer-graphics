"""
Contornos y suggestive contours de una malla, para una cámara dada.

Sigue la implementación de referencia (rtsc) de DeCarlo, Finkelstein y
Rusinkiewicz (2004). Por cada vértice se calculan n.v, la curvatura radial y
la prueba de la derivada; luego, en cada cara donde un campo cambia de signo,
se traza un segmento entre los dos cruces por cero de sus aristas.
"""
import numpy as np


def compute_view_quantities(vertices, normals, k1, k2, d1, d2, C, camera_pos, threshold):
    """Cantidades por vértice que dependen de la cámara.

    Con `vista` el vector unitario hacia la cámara y (u, v) sus componentes
    sobre las direcciones principales d1 y d2, devuelve:

    - ndotv = n.vista, que vale cero en el contorno.
    - kr = k1 u^2 + k2 v^2, que es la curvatura radial multiplicada por
      sin^2(theta), con theta el ángulo entre n y la vista. Tiene los mismos
      ceros que la curvatura radial y no se indefine cuando la cámara mira de
      frente.
    - La prueba de la derivada como fracción test_num / test_den, que es
      positiva donde D_w kr tan(theta) supera `threshold`. El numerador y el
      denominador se interpolan por separado sobre la cara, porque el cociente
      se dispara donde n.vista vale cero.
    """
    vista = camera_pos - vertices
    vista /= np.linalg.norm(vista, axis=1, keepdims=True)
    ndotv = np.einsum("ij,ij->i", vista, normals)
    u = np.einsum("ij,ij->i", vista, d1)
    v = np.einsum("ij,ij->i", vista, d2)
    u2, v2 = u * u, v * v
    kr = k1 * u2 + k2 * v2

    # D_w kr en un cero de kr (DeCarlo et al. 2004, apéndice C) es la derivada
    # de la curvatura en la dirección de vista, C(w, w, w), menos un término por
    # el giro de w sobre la superficie, que depende de la torsión geodésica tau.
    # Dividido por ndotv, test_num da D_w kr tan(theta) menos el umbral
    with np.errstate(divide="ignore", invalid="ignore"):
        sin2 = np.maximum(u2 + v2, 1e-12)
        derivada = (u2 * (u * C[:, 0] + 3 * v * C[:, 1]) + v2 * (3 * u * C[:, 2] + v * C[:, 3])) / sin2
        tau = (k2 - k1) * u * v / sin2
    test_num = derivada - 2 * ndotv * tau * tau - threshold * ndotv
    return ndotv, kr, test_num, ndotv


def radial_curvature(kr, ndotv):
    """La curvatura radial propiamente tal, para pintarla sobre la malla."""
    return kr / np.maximum(1.0 - ndotv * ndotv, 1e-6)


def extract_isolines(vertices, faces, field, test_num=None, test_den=None, ndotv=None, fade=0.0):
    """Segmentos donde `field` cruza por cero, uno por cara.

    En cada cara con valores de distinto signo, el vértice de signo distinto a
    los otros dos define las dos aristas que cruza la curva; el cruce en cada
    una se interpola linealmente. Si hay prueba, el segmento se recorta a las
    partes donde test_num / test_den es positivo (sin denominador, donde test_num
    lo es). Con `ndotv` se descartan las caras que dan la espalda a la cámara.

    Devuelve los extremos de los segmentos (S, 2, 3) y la opacidad de cada
    extremo (S, 2). Con `fade` positivo la opacidad crece desde cero en el
    borde del recorte: es x / (fade + x), con x = test_num / test_den.
    """
    valores = field[faces]
    candidatas = (valores > 0).any(1) & (valores < 0).any(1)
    if ndotv is not None:
        candidatas &= (ndotv[faces] > 0).any(1)
    if test_num is not None:
        num = test_num[faces]
        den = np.ones_like(num) if test_den is None else test_den[faces]
        # se descarta la cara si la prueba falla en sus tres vértices
        candidatas &= ~(((num <= 0) & (den >= 0)).all(1) | ((num >= 0) & (den <= 0)).all(1))
    caras = faces[candidatas]
    valores = valores[candidatas]

    # vértice de signo distinto, con la misma prioridad que rtsc ante los ceros
    distinto = np.full(len(caras), 2)
    for i in (1, 0):
        otros = valores[:, [(i + 1) % 3, (i + 2) % 3]]
        es = ((valores[:, i] < 0) & (otros >= 0).all(1)) | ((valores[:, i] > 0) & (otros <= 0).all(1))
        distinto[es] = i
    filas = np.arange(len(caras))[:, np.newaxis]
    a, b, c = caras[filas, (distinto[:, np.newaxis] + np.arange(3)) % 3].T

    # cruces por cero en las aristas a-b y a-c
    t_ab = (field[a] / (field[a] - field[b]))[:, np.newaxis]
    t_ac = (field[a] / (field[a] - field[c]))[:, np.newaxis]
    p1 = (1 - t_ab) * vertices[a] + t_ab * vertices[b]
    p2 = (1 - t_ac) * vertices[a] + t_ac * vertices[c]
    if test_num is None:
        segmentos = np.stack([p1, p2], axis=1)
        return segmentos, np.ones((len(segmentos), 2))

    # la prueba en los extremos y sus cruces por cero a lo largo del segmento
    def en_los_extremos(campo):
        return ((1 - t_ab[:, 0]) * campo[a] + t_ab[:, 0] * campo[b],
                (1 - t_ac[:, 0]) * campo[a] + t_ac[:, 0] * campo[c])

    num1, num2 = en_los_extremos(test_num)
    den1, den2 = en_los_extremos(test_den) if test_den is not None else (np.ones(len(a)),) * 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cruce_num = np.where((num1 >= 0) != (num2 >= 0), num1 / (num1 - num2), np.nan)
        cruce_den = np.where((den1 >= 0) != (den2 >= 0), den1 / (den1 - den2), np.nan)
    cortes = np.sort(np.stack([cruce_num, cruce_den], 1), axis=1)  # los nan quedan al final
    n_cortes = (~np.isnan(cortes)).sum(1)
    valido = (num1 >= 0) == (den1 >= 0)

    # tramos válidos: la validez se invierte en cada corte
    tramos = []
    inicio = np.zeros(len(a))
    for k in range(3):
        fin = cortes[:, k] if k < 2 else np.ones(len(a))
        fin = np.where(k < n_cortes, fin, 1.0)
        activo = valido & (k <= n_cortes) & (fin > inicio)
        tramos.append((activo, inicio.copy(), fin.copy()))
        valido = valido != (k < n_cortes)
        inicio = np.where(k < n_cortes, fin, 2.0)

    segmentos, opacidades = [], []
    for activo, desde, hasta in tramos:
        extremos = np.stack([desde[activo], hasta[activo]], 1)[..., np.newaxis]
        segmentos.append((1 - extremos) * p1[activo, np.newaxis] + extremos * p2[activo, np.newaxis])
        num = (1 - extremos[..., 0]) * num1[activo, np.newaxis] + extremos[..., 0] * num2[activo, np.newaxis]
        den = (1 - extremos[..., 0]) * den1[activo, np.newaxis] + extremos[..., 0] * den2[activo, np.newaxis]
        if fade > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                opacidades.append(np.nan_to_num(num / (den * fade + num), nan=1.0).clip(0, 1))
        else:
            opacidades.append(np.ones_like(num))
    return np.concatenate(segmentos), np.concatenate(opacidades)


def extract_silhouette_edges(mesh, camera_pos):
    """Aristas entre una cara que mira a la cámara y otra que no.

    Es el contorno exacto de un poliedro. Devuelve los segmentos (S, 2, 3) y su
    opacidad, como extract_isolines.
    """
    centros = mesh.triangles_center
    de_frente = np.einsum("ij,ij->i", mesh.face_normals, camera_pos - centros) > 0
    pares = mesh.face_adjacency
    silueta = de_frente[pares[:, 0]] != de_frente[pares[:, 1]]
    segmentos = mesh.vertices[mesh.face_adjacency_edges[silueta]]
    return segmentos, np.ones((len(segmentos), 2))
