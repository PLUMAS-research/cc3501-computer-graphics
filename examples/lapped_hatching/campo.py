"""Campo de direcciones de los trazos, alineado con las curvaturas principales.

Sigue la sección 6 de Praun, Hoppe, Webb y Finkelstein (2001). Las direcciones
principales de curvatura muestran la forma de la superficie, pero en una malla
son ruidosas y quedan indefinidas donde la superficie se curva igual en todas
direcciones (una esfera, una zona plana). El campo se construye en cuatro pasos:

1. Se simplifica la malla con QEM, así las curvaturas describen la forma
   general y no los detalles finos.
2. En cada cara de la malla simplificada se juntan los vértices vecinos de sus
   tres vértices y se ajusta una cuádrica z = a x^2 + b xy + c y^2 + d x + e y + f
   en el marco de la cara. Sus direcciones principales son las del operador de
   forma de esa cuádrica. La confianza de la dirección sube con la diferencia
   entre las dos curvaturas y baja con el residuo del ajuste.
3. Se suaviza el campo con una optimización global que conserva las
   direcciones de alta confianza y llena las de baja confianza con las de sus
   vecinas. Un trazo es una línea, así que las direcciones d y -d son la misma:
   cada dirección se representa con el número complejo z = exp(2 i theta), que
   no cambia al girar theta en 180 grados.
4. Se remuestrea el campo suavizado sobre las caras de la malla original.

El módulo no depende de OpenGL: se puede ejecutar y probar sin ventana.
"""

import numpy as np
import scipy.sparse as sp
import trimesh as tm
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from examples.simplificacion_qem.qem import simplificar_a_niveles
from grafica.half_edge import HalfEdgeMesh

from .malla import agrupar_vertices, es_variedad


def marcos_de_caras(vertices, caras):
    """Normal unitaria y base tangente ortonormal (e1, e2) de cada cara."""
    a, b, c = vertices[caras[:, 0]], vertices[caras[:, 1]], vertices[caras[:, 2]]
    normal = np.cross(b - a, c - a)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
    e1 = b - a
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-20)
    e2 = np.cross(normal, e1)
    return normal, e1, e2


def _direcciones_principales(vertices, caras):
    """Dirección de mayor curvatura (en valor absoluto), su confianza y esa curvatura, por cara."""
    cantidad_vertices = len(vertices)
    cantidad_caras = len(caras)
    normal, e1, e2 = marcos_de_caras(vertices, caras)
    centroides = vertices[caras].mean(axis=1)

    # vecindario de una cara: los vértices adyacentes a cualquiera de los tres
    # suyos. Se obtiene multiplicando la incidencia cara-vértice por la
    # adyacencia vértice-vértice (con la diagonal)
    aristas = np.concatenate([caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [2, 0]]])
    adyacencia = sp.coo_matrix(
        (np.ones(2 * len(aristas)), (aristas.ravel(), aristas[:, ::-1].ravel())),
        shape=(cantidad_vertices, cantidad_vertices),
    ).tocsr() + sp.identity(cantidad_vertices, format="csr")
    incidencia = sp.csr_matrix(
        (np.ones(3 * cantidad_caras), (np.repeat(np.arange(cantidad_caras), 3), caras.ravel())),
        shape=(cantidad_caras, cantidad_vertices),
    )
    vecindario = (incidencia @ adyacencia).tocsr()

    direccion = np.zeros((cantidad_caras, 3))
    confianza = np.zeros(cantidad_caras)
    curvatura = np.zeros(cantidad_caras)
    for cara in range(cantidad_caras):
        indices = vecindario.indices[vecindario.indptr[cara]:vecindario.indptr[cara + 1]]
        relativos = vertices[indices] - centroides[cara]
        x = relativos @ e1[cara]
        y = relativos @ e2[cara]
        z = relativos @ normal[cara]
        if len(indices) < 6:
            direccion[cara] = e1[cara]
            continue
        sistema = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=1)
        coeficientes = np.linalg.lstsq(sistema, z, rcond=None)[0]
        a, b, c, d, e = coeficientes[:5]

        # operador de forma S = I^-1 II de la superficie z = h(x, y) en el origen
        primera = np.array([[1 + d * d, d * e], [d * e, 1 + e * e]])
        segunda = np.array([[2 * a, b], [b, 2 * c]]) / np.sqrt(1 + d * d + e * e)
        try:
            curvaturas, vectores = np.linalg.eig(np.linalg.solve(primera, segunda))
        except np.linalg.LinAlgError:
            # vecindad degenerada (por ejemplo, puntos sobre dos planos que se
            # cortan): la cuádrica no define direcciones y la confianza queda en 0
            direccion[cara] = e1[cara]
            continue
        curvaturas = curvaturas.real
        vectores = vectores.real
        mayor = int(np.argmax(np.abs(curvaturas)))
        k_mayor = abs(curvaturas[mayor])
        k_menor = abs(curvaturas[1 - mayor])
        u, v = vectores[:, mayor]
        tangente = u * e1[cara] + v * e2[cara]
        direccion[cara] = tangente / max(np.linalg.norm(tangente), 1e-20)

        # la cuádrica explica bien la vecindad si su residuo es chico frente a
        # la variación de altura de los puntos (como el R^2 de una regresión);
        # en una zona casi plana esa variación es ruido y la calidad baja
        residuo = np.mean((sistema @ coeficientes - z) ** 2)
        calidad = max(0.0, 1.0 - residuo / max(np.var(z), 1e-20))
        anisotropia = (k_mayor - k_menor) / (k_mayor + k_menor + 1e-12)
        confianza[cara] = anisotropia * calidad
        curvatura[cara] = k_mayor
    return direccion, confianza, curvatura


def suavizar(vertices, caras, direccion, confianza, peso_datos=4.0):
    """Campo suave que respeta las direcciones de alta confianza.

    Minimiza la suma, sobre las aristas interiores (f, g), de |z_g - r_fg z_f|^2
    más la suma sobre las caras de peso_datos * confianza_f * |z_f - z0_f|^2,
    donde z0 son las direcciones de entrada y r_fg traslada una dirección de la
    cara f a la g girando alrededor de la arista común (conserva el ángulo con
    esa arista).

    Con |z| = 1, cada término de las aristas vale 2 - 2 cos(2 (theta_g -
    theta_f - rho_fg)): es la energía de Hertzmann y Zorin (2000) con simetría
    de 180 grados. Sin la restricción |z| = 1 el problema queda lineal y
    disperso; al final se normaliza z.
    """
    _, e1, e2 = marcos_de_caras(vertices, caras)
    angulo = np.arctan2(np.einsum("ij,ij->i", direccion, e2), np.einsum("ij,ij->i", direccion, e1))
    z0 = np.exp(2j * angulo)

    malla = tm.Trimesh(vertices, caras, process=False)
    f, g = malla.face_adjacency.T
    v0, v1 = malla.face_adjacency_edges.T
    arista = vertices[v1] - vertices[v0]
    angulo_en_f = np.arctan2(np.einsum("ij,ij->i", arista, e2[f]), np.einsum("ij,ij->i", arista, e1[f]))
    angulo_en_g = np.arctan2(np.einsum("ij,ij->i", arista, e2[g]), np.einsum("ij,ij->i", arista, e1[g]))
    traslado = np.exp(2j * (angulo_en_g - angulo_en_f))

    cantidad = len(caras)
    datos = peso_datos * confianza + 1e-6
    filas = np.concatenate([f, g, g, f, np.arange(cantidad)])
    columnas = np.concatenate([f, g, f, g, np.arange(cantidad)])
    valores = np.concatenate([
        np.ones(len(f)), np.ones(len(f)), -traslado, -np.conj(traslado), datos
    ])
    sistema = sp.csc_matrix((valores, (filas, columnas)), shape=(cantidad, cantidad))
    z = spsolve(sistema, datos * z0)

    angulo = 0.5 * np.angle(z)
    return np.cos(angulo)[:, None] * e1 + np.sin(angulo)[:, None] * e2


def _remuestrear(vertices_gruesos, caras_gruesas, direccion_gruesa, curvatura_gruesa,
                 vertices, caras, vecinos=4):
    """Lleva el campo y la curvatura de la malla simplificada a la original.

    Cada cara fina promedia las direcciones de las caras gruesas más cercanas,
    proyectadas sobre su plano. El promedio se hace sobre exp(2 i theta) para
    que d y -d cuenten igual. Una cara gruesa cuya normal apunta en contra no
    aporta: está al otro lado de una parte delgada, como una oreja.
    """
    normal_gruesa = marcos_de_caras(vertices_gruesos, caras_gruesas)[0]
    normal, e1, e2 = marcos_de_caras(vertices, caras)
    arbol = cKDTree(vertices_gruesos[caras_gruesas].mean(axis=1))
    distancia, cercanas = arbol.query(vertices[caras].mean(axis=1), k=vecinos)

    proyectada = direccion_gruesa[cercanas]
    proyectada -= np.einsum("fkj,fj->fk", proyectada, normal)[:, :, None] * normal[:, None, :]
    alineacion = np.einsum("fkj,fj->fk", normal_gruesa[cercanas], normal)
    peso = np.maximum(alineacion, 0.0) / (distancia + 1e-9)
    angulo = np.arctan2(
        np.einsum("fkj,fj->fk", proyectada, e2), np.einsum("fkj,fj->fk", proyectada, e1)
    )
    curvatura = (peso * curvatura_gruesa[cercanas]).sum(axis=1) / np.maximum(peso.sum(axis=1), 1e-20)
    # la longitud de la proyección también pesa: una dirección casi normal a la
    # cara fina no dice nada sobre su plano
    peso *= np.einsum("fkj,fkj->fk", proyectada, proyectada)
    z = (peso * np.exp(2j * angulo)).sum(axis=1)
    angulo = 0.5 * np.angle(z)
    return np.cos(angulo)[:, None] * e1 + np.sin(angulo)[:, None] * e2, curvatura


def campo_de_direcciones(vertices, caras, direccion_principal="minima", base=None, reporte=print):
    """Dirección unitaria del trazo en cada cara de la malla, tangente a ella.

    `direccion_principal` elige qué dirección de curvatura siguen los trazos:
    "maxima" los pone alrededor de las partes cilíndricas (como anillos) y
    "minima" a lo largo de ellas, como el pelo de una oreja. Las dos son
    perpendiculares, así que la segunda es la primera girada en 90 grados
    dentro de cada cara.

    `base` es la malla (vértices, caras) que se simplifica; si la malla que se
    dibuja es una subdivisión de otra, conviene simplificar la original, que
    tiene menos caras y describe la misma forma. Si no es una variedad, QEM no
    se puede aplicar y se simplifica agrupando vértices.

    Devuelve también la mayor curvatura (en valor absoluto) de cada cara, que
    sirve para achicar los parches donde la superficie se curva mucho.
    """
    vertices_base, caras_base = base if base is not None else (vertices, caras)
    objetivo = int(np.clip(0.2 * len(caras_base), 300, 1500))
    gruesa = None
    if es_variedad(caras_base):
        if reporte:
            reporte(f"  simplificando la malla a {objetivo} caras (QEM)...")
        try:
            gruesa, = simplificar_a_niveles(HalfEdgeMesh.from_faces(vertices_base, caras_base), [objetivo])
        except (KeyError, IndexError, ValueError) as error:
            if reporte:
                reporte(f"  QEM falló ({error!r})")
    if gruesa is None:
        if reporte:
            reporte(f"  simplificando la malla a unas {objetivo} caras (agrupando vértices)...")
        gruesa = agrupar_vertices(vertices_base, caras_base, objetivo)[:2]
    vertices_gruesos, caras_gruesas = gruesa
    direccion, confianza, curvatura_gruesa = _direcciones_principales(vertices_gruesos, caras_gruesas)
    if reporte:
        reporte(
            f"  {len(caras_gruesas)} caras gruesas, confianza media {confianza.mean():.2f}, "
            f"{np.mean(confianza > 0.5) * 100:.0f} % sobre 0,5"
        )
    suave = suavizar(vertices_gruesos, caras_gruesas, direccion, confianza)
    fina, curvatura_fina = _remuestrear(
        vertices_gruesos, caras_gruesas, suave, curvatura_gruesa, vertices, caras
    )
    if direccion_principal == "minima":
        fina = np.cross(marcos_de_caras(vertices, caras)[0], fina)
    return fina, curvatura_fina
