"""Preparación de la malla para el achurado.

1. Se cargan sus partes con sus materiales: cada parte trae coordenadas de
   textura y una imagen, o solo un color.
2. Se funden los vértices repetidos (las costuras de las coordenadas de
   textura), se quitan las caras degeneradas o repetidas y se orientan las
   caras de forma coherente.
3. Se escala al cubo [-1, 1], con el eje elegido hacia arriba.
4. Se tapan los agujeros chicos.
5. Se subdivide hasta llegar a CARAS_MINIMAS caras.

Las coordenadas de textura viven en las esquinas de las caras, porque en una
costura un mismo vértice tiene dos coordenadas. Por eso cada paso que cambia
las caras arrastra también las coordenadas de sus esquinas.

El módulo no depende de OpenGL: se puede ejecutar y probar sin ventana.
"""

import itertools
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh as tm
from scipy.sparse.csgraph import connected_components

# la subdivisión automática llega al menos a esta cantidad de caras: un parche
# tiene que abarcar varias caras
CARAS_MINIMAS = 8000

# dos caras vecinas forman un pliegue (una arista viva) si sus normales se
# separan más que esto
ANGULO_PLIEGUE_GRADOS = 60.0

# con más aristas vivas que esta fracción, la malla no se suaviza al subdividir
FRACCION_PLIEGUES_LOOP = 0.02

# una malla con más caras que esto se simplifica a CARAS_SIMPLIFICADAS antes de
# todo lo demás: los modelos del paper tienen entre 7.500 y 15.000 caras, y el
# crecimiento de parches en Python se vuelve lento con cientos de miles
CARAS_MAXIMAS = 40000
CARAS_SIMPLIFICADAS = 20000

# lado máximo de las imágenes de textura en la GPU
LADO_MAXIMO_TEXTURA = 2048

# al subdividir, cada cara se parte en cuatro (en el orden de trimesh); estas
# son las coordenadas baricéntricas de las esquinas de cada hija en su madre
_HIJAS = np.array([
    [[1, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]],
    [[0.5, 0.5, 0], [0, 1, 0], [0, 0.5, 0.5]],
    [[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]],
    [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]],
])


def es_variedad(caras):
    """La malla es una variedad orientada.

    Cada arista está en una o dos caras, que la recorren en sentidos opuestos,
    y alrededor de cada vértice hay tantas aristas como caras (abanico
    cerrado) o una más (abanico abierto, en el borde). La estructura half-edge
    lo supone, y con ella la simplificación QEM y la subdivisión de Loop.
    """
    aristas = caras[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    no_dirigidas, caras_por_arista = np.unique(np.sort(aristas, axis=1), axis=0, return_counts=True)
    _, veces_por_sentido = np.unique(aristas, axis=0, return_counts=True)
    cantidad = caras.max() + 1
    exceso = np.bincount(no_dirigidas.ravel(), minlength=cantidad) - np.bincount(caras.ravel(), minlength=cantidad)
    return caras_por_arista.max() <= 2 and veces_por_sentido.max() == 1 and exceso.max() <= 1


def agrupar_vertices(vertices, caras, objetivo):
    """Simplificación por agrupamiento: los vértices que caen en una misma
    celda de una grilla se funden en su promedio.

    Sirve para cualquier malla, aunque no sea una variedad. Una malla de
    triángulos tiene cerca del doble de caras que de vértices, así que con
    celdas de lado sqrt(2 A / objetivo) sobre una superficie de área A quedan
    unas `objetivo` caras. Devuelve los vértices, las caras y, por cada cara,
    el índice de la cara original de la que viene.
    """
    a, b, c = vertices[caras[:, 0]], vertices[caras[:, 1]], vertices[caras[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    lado = np.sqrt(2.0 * area / objetivo)
    _, grupo = np.unique(np.floor((vertices - vertices.min(axis=0)) / lado), axis=0, return_inverse=True)
    grupo = grupo.ravel()
    cuantos = np.bincount(grupo)
    agrupados = np.stack([np.bincount(grupo, weights=vertices[:, k]) for k in range(3)], axis=1) / cuantos[:, None]
    nuevas = grupo[caras]
    distintas = np.flatnonzero(
        (nuevas[:, 0] != nuevas[:, 1]) & (nuevas[:, 1] != nuevas[:, 2]) & (nuevas[:, 2] != nuevas[:, 0])
    )
    # dos caras que quedan sobre los mismos tres vértices cuentan una vez
    _, primeras = np.unique(np.sort(nuevas[distintas], axis=1), axis=0, return_index=True)
    quedan = distintas[np.sort(primeras)]
    return agrupados, nuevas[quedan], quedan


def _partes(ruta):
    """Partes del modelo, cada una con sus caras, coordenadas y material.

    Se carga como escena: al fundirlo en una sola malla, trimesh arma un atlas
    con todas las texturas, que con imágenes de 4096 x 4096 ocupa gigabytes.

    Si un OBJ nombra un archivo de materiales que no está, trimesh inventa un
    material gris; en ese caso el modelo se queda sin color propio.
    """
    con_materiales = True
    if ruta.lower().endswith(".obj"):
        bibliotecas = re.findall(r"^mtllib\s+(.+?)\s*$", Path(ruta).read_text(errors="ignore"), re.MULTILINE)
        con_materiales = any((Path(ruta).parent / nombre).exists() for nombre in bibliotecas)
    partes = []
    for malla in tm.load(ruta, force="scene").dump():
        visual = malla.visual
        material = getattr(visual, "material", None) if con_materiales else None
        uv = getattr(visual, "uv", None)
        imagen = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
        if imagen is None or uv is None or len(uv) != len(malla.vertices):
            imagen = None
        # un material sin imagen aporta su color difuso; sin material, papel blanco
        color = np.asarray(material.main_color, dtype=np.uint8) if material is not None else None
        caras = np.asarray(malla.faces, dtype=np.int64)
        partes.append({
            "vertices": np.asarray(malla.vertices, dtype=np.float64),
            "caras": caras,
            "uv": np.asarray(uv, dtype=np.float64)[caras] if imagen is not None else np.zeros((len(caras), 3, 2)),
            "imagen": imagen,
            "color": color,
        })
    return partes


def _capas_de_textura(partes):
    """Imágenes de textura de las partes como capas de un mismo lado.

    Devuelve un arreglo (N, lado, lado, 4) con las capas y, por parte, el
    índice de su capa. Las partes que comparten imagen comparten capa; una
    parte sin imagen tiene una capa de su color. La fila 0 de cada capa es la
    de abajo de la imagen, porque en un OBJ la coordenada v crece hacia arriba.
    """
    imagenes = [p["imagen"] for p in partes if p["imagen"] is not None]
    lado = max([max(im.size) for im in imagenes], default=1)
    lado = int(min(LADO_MAXIMO_TEXTURA, 2 ** np.ceil(np.log2(lado))))
    capas = []
    indice_por_imagen = {}
    capa_por_parte = []
    for parte in partes:
        if parte["imagen"] is not None:
            clave = id(parte["imagen"])
            if clave not in indice_por_imagen:
                imagen = parte["imagen"].convert("RGBA").resize((lado, lado))
                indice_por_imagen[clave] = len(capas)
                capas.append(np.asarray(imagen)[::-1])
            capa_por_parte.append(indice_por_imagen[clave])
        else:
            color = parte["color"] if parte["color"] is not None else np.full(4, 255, dtype=np.uint8)
            capa_por_parte.append(len(capas))
            capas.append(np.broadcast_to(color, (lado, lado, 4)))
    return np.stack(capas).astype(np.uint8), capa_por_parte


def _tapar_agujeros(vertices, caras, fraccion_maxima=0.5):
    """Cierra cada agujero chico de la malla con un abanico de triángulos.

    Un agujero es un ciclo de aristas de borde (las que tienen una sola cara).
    Se agrega un vértice en el centro del ciclo y un triángulo por arista,
    recorrida al revés que en su cara para mantener la orientación. En el
    borde de un agujero los parches no pueden crecer, así que sus caras no
    alcanzarían a quedar cubiertas.

    Solo se tapan los agujeros más chicos que `fraccion_maxima` del tamaño de
    su pieza (la diagonal de su caja). Así se cierran las perforaciones de un
    escaneo, como la base del conejo, pero no el contorno de una pieza que es
    una hoja abierta (una pluma) ni la boca de una habitación.

    Devuelve también, por cada triángulo nuevo, la arista de borde que lo
    origina como índice 3 f + k (la arista k de la cara f).
    """
    aristas = caras[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    de_borde = tm.grouping.group_rows(np.sort(aristas, axis=1), require_count=1)
    # pieza de cada vértice y tamaño de cada pieza
    cantidad = len(vertices)
    grafo = sp.coo_matrix((np.ones(len(aristas)), (aristas[:, 0], aristas[:, 1])), shape=(cantidad, cantidad))
    _, pieza = connected_components(grafo, directed=False)
    minimos = np.full((pieza.max() + 1, 3), np.inf)
    maximos = np.full((pieza.max() + 1, 3), -np.inf)
    np.minimum.at(minimos, pieza, vertices)
    np.maximum.at(maximos, pieza, vertices)
    tamano_pieza = np.linalg.norm(maximos - minimos, axis=1)

    siguiente = {int(aristas[i, 0]): (int(aristas[i, 1]), int(i)) for i in de_borde}
    vertices = list(vertices)
    nuevas = []
    origen = []
    while siguiente:
        inicio, (destino, arista) = siguiente.popitem()
        ciclo = [inicio, destino]
        recorridas = [arista]
        while ciclo[-1] in siguiente:
            destino, arista = siguiente.pop(ciclo[-1])
            ciclo.append(destino)
            recorridas.append(arista)
        # el ciclo vuelve al inicio: su último vértice ya es el primero
        if ciclo[-1] != ciclo[0]:
            continue
        borde = np.array([vertices[v] for v in ciclo[:-1]])
        diametro = np.linalg.norm(borde[:, None] - borde[None], axis=2).max()
        if diametro > fraccion_maxima * tamano_pieza[pieza[ciclo[0]]]:
            continue
        centro = len(vertices)
        vertices.append(np.mean([vertices[v] for v in ciclo[:-1]], axis=0))
        nuevas += [(b, a, centro) for a, b in itertools.pairwise(ciclo)]
        origen += recorridas
    if nuevas:
        caras = np.concatenate([caras, np.array(nuevas, dtype=np.int64)])
    return np.array(vertices), caras, np.array(origen, dtype=np.int64)


def cargar(ruta, subdivisiones=None, arriba="y"):
    """Lee el modelo y lo deja listo para el achurado.

    `arriba` es el eje que apunta hacia arriba en el archivo ("y" o "z"); la
    escena usa Y. Con `subdivisiones` igual a None se subdivide hasta llegar a
    CARAS_MINIMAS, con un máximo de tres pasos. Una superficie suave de una
    sola pieza se subdivide con Loop: cada triángulo se divide en cuatro y los
    vértices se acercan a una superficie límite suave. Una malla con aristas
    vivas (un cubo, un modelo de pocos polígonos), de varias piezas o que no
    es una variedad se subdivide por puntos medios, que no cambia su forma.

    Devuelve un dict con la malla base y la subdividida ("base", "vertices",
    "caras"), las coordenadas de textura de cada esquina ("uv", (F, 3, 2)), la
    capa de textura de cada cara ("capa"), las capas ("capas") y si el modelo
    trae colores propios ("con_textura"), y una descripción de la subdivisión.
    """
    partes = _partes(ruta)
    capas, capa_por_parte = _capas_de_textura(partes)
    inicio = np.cumsum([0] + [len(p["vertices"]) for p in partes])
    vertices = np.concatenate([p["vertices"] for p in partes])
    caras = np.concatenate([p["caras"] + desde for p, desde in zip(partes, inicio)])
    uv = np.concatenate([p["uv"] for p in partes])
    capa = np.concatenate([np.full(len(p["caras"]), c) for p, c in zip(partes, capa_por_parte)])
    # el modelo trae color propio si alguna parte tiene imagen o un color que no es blanco
    con_textura = any(
        p["imagen"] is not None or (p["color"] is not None and (p["color"][:3] < 255).any()) for p in partes
    )

    # los vértices repetidos en las costuras se funden: la superficie es una
    # sola aunque su textura tenga cortes. Fundir no cambia el orden de las caras
    malla = tm.Trimesh(vertices, caras, process=False)
    malla.merge_vertices()
    # sin caras degeneradas (área nula) ni repetidas, que no tienen plano tangente
    quedan = np.flatnonzero(malla.nondegenerate_faces())
    malla.update_faces(quedan)
    unicas = np.flatnonzero(malla.unique_faces())
    malla.update_faces(unicas)
    quedan = quedan[unicas]
    malla.remove_unreferenced_vertices()
    # orientar las caras de forma coherente puede invertir algunas: la esquina
    # k de cada cara puede pasar a otra posición
    antes = malla.faces.copy()
    malla.fix_normals()
    posicion = np.argmax(antes[:, None, :] == malla.faces[:, :, None], axis=2)
    uv = uv[quedan][np.arange(len(quedan))[:, None], posicion]
    capa = capa[quedan]

    vertices = np.asarray(malla.vertices, dtype=np.float64)
    caras = np.asarray(malla.faces, dtype=np.int64)
    simplificada = ""
    if len(caras) > CARAS_MAXIMAS:
        vertices, caras, quedan = agrupar_vertices(vertices, caras, CARAS_SIMPLIFICADAS)
        uv, capa = uv[quedan], capa[quedan]
        simplificada = f"simplificada a {len(caras)} caras agrupando vértices, "
    if arriba == "z":
        # giro de -90 grados en torno a X: el eje Z del archivo queda hacia arriba
        vertices = np.stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]], axis=1)
    minimo, maximo = vertices.min(axis=0), vertices.max(axis=0)
    vertices = (vertices - (minimo + maximo) / 2) * (2.0 / (maximo - minimo).max())

    # cada triángulo que tapa un agujero toma la textura de la arista de borde
    # que lo origina; su centro, la del punto medio de esa arista
    caras_previas = len(caras)
    vertices, caras, aristas_origen = _tapar_agujeros(vertices, caras)
    if len(aristas_origen):
        cara, k = aristas_origen // 3, aristas_origen % 3
        uv_a, uv_b = uv[cara, k], uv[cara, (k + 1) % 3]
        uv = np.concatenate([uv, np.stack([uv_b, uv_a, 0.5 * (uv_a + uv_b)], axis=1)])
        capa = np.concatenate([capa, capa[cara]])
    base = (vertices, caras)
    malla = tm.Trimesh(vertices, caras, process=False)

    if subdivisiones is None:
        subdivisiones = 0
        while caras_previas * 4**subdivisiones < CARAS_MINIMAS and subdivisiones < 3:
            subdivisiones += 1
    angulos = malla.face_adjacency_angles
    pliegues = np.mean(angulos > np.radians(ANGULO_PLIEGUE_GRADOS)) if len(angulos) else 0.0
    if subdivisiones == 0:
        descripcion = simplificada + "sin subdividir"
    elif es_variedad(caras) and pliegues <= FRACCION_PLIEGUES_LOOP and malla.body_count == 1:
        malla = malla.subdivide_loop(iterations=subdivisiones)
        descripcion = simplificada + f"{subdivisiones} subdivisiones de Loop"
    else:
        for _ in range(subdivisiones):
            malla = malla.subdivide()
        descripcion = simplificada + (
            f"{subdivisiones} subdivisiones por puntos medios ({pliegues * 100:.0f} % de aristas vivas)"
        )

    # las hijas de la cara f son las caras 4f a 4f + 3, con las coordenadas
    # baricéntricas de _HIJAS; tras n pasos, se componen n veces
    baricentricas = np.broadcast_to(np.eye(3), (len(caras), 3, 3))
    for _ in range(subdivisiones):
        baricentricas = np.einsum("hij,fjk->fhik", _HIJAS, baricentricas).reshape(-1, 3, 3)
    madre = np.arange(len(baricentricas)) // 4**subdivisiones
    return {
        "base": base,
        "vertices": np.asarray(malla.vertices, dtype=np.float64),
        "caras": np.asarray(malla.faces, dtype=np.int64),
        "uv": np.einsum("fij,fjd->fid", baricentricas, uv[madre]),
        "capa": capa[madre],
        "capas": capas,
        "con_textura": con_textura,
        "descripcion": descripcion,
    }


def normales_por_esquina(vertices, caras):
    """Normal de cada esquina de cada cara, de forma (F, 3, 3).

    Es el promedio, pesado por área, de las normales de las caras que tocan el
    vértice y no forman un pliegue con la cara de la esquina. En una superficie
    suave es la normal por vértice de siempre; en una arista viva cada lado
    conserva su normal y el tono no se corre de un lado al otro.
    """
    a, b, c = vertices[caras[:, 0]], vertices[caras[:, 1]], vertices[caras[:, 2]]
    ponderada = np.cross(b - a, c - a)
    unitaria = ponderada / np.maximum(np.linalg.norm(ponderada, axis=1, keepdims=True), 1e-20)

    # cada esquina se empareja con todas las esquinas de su mismo vértice. El
    # número de pares es la suma de los cuadrados de las valencias, así que un
    # vértice de valencia alta (el centro de un agujero tapado) solo agrega los
    # pares propios
    plano = caras.ravel()
    orden = np.argsort(plano, kind="stable")
    cuantas = np.bincount(plano, minlength=len(vertices))
    inicio = np.concatenate([[0], np.cumsum(cuantas)[:-1]])
    repeticiones = cuantas[plano]
    esquina = np.repeat(np.arange(len(plano)), repeticiones)
    posicion = np.arange(len(esquina)) - np.repeat(np.cumsum(repeticiones) - repeticiones, repeticiones)
    vecina = orden[inicio[plano[esquina]] + posicion] // 3

    coseno = np.einsum("ij,ij->i", unitaria[esquina // 3], unitaria[vecina])
    incluir = coseno > np.cos(np.radians(ANGULO_PLIEGUE_GRADOS))
    normal = np.stack([
        np.bincount(esquina[incluir], weights=ponderada[vecina[incluir], k], minlength=len(plano))
        for k in range(3)
    ], axis=1).reshape(-1, 3, 3)
    return normal / np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-20)
