"""Trazado de rayos por CPU sobre un grafo de escena (`Scenegraph`).

A diferencia del ejemplo atomico (`raytracing_basico`, primitivas analiticas),
aqui la geometria son mallas de triangulos que viven en el mismo `Scenegraph`
que se rasteriza con OpenGL. La idea pedagogica: la misma escena se puede
dibujar por rasterizacion (object-order) o por trazado de rayos (image-order).

El trazador recolecta cada malla del grafo y la lleva a coordenadas de mundo.
La interseccion es matematica propia y vectorizada, sin libreria externa: por
cada objeto, una fase ancha descarta los rayos que no cruzan su caja contenedora
(AABB, metodo de los slabs) y solo los rayos candidatos se prueban contra los
triangulos con Moller-Trumbore. Asi el costo cae de "todo rayo contra toda malla"
a "cada rayo solo contra las mallas que su caja podria tocar". La iluminacion es
Phong local con rayos de sombra. Las cajas AABB tambien se pueden dibujar en la
vista rasterizada (tecla B en el ejemplo).
"""

import numpy as np
import trimesh as tm


def _normalizar_filas(vectores):
    return vectores / np.linalg.norm(vectores, axis=1, keepdims=True)


class GeometriaEscena:
    """Una malla del grafo en coordenadas de mundo, con su material y su AABB.

    Precalcula lo que el trazador necesita en el bucle caliente: los triangulos
    (T, 3, 3), las normales de cara (T, 3) y la caja contenedora (caja_min,
    caja_max). Asi no se vuelve a tocar trimesh durante el render.
    """

    def __init__(self, nombre, mesh_mundo, color, reflectividad, es_piso):
        self.nombre = nombre
        self.triangulos = np.asarray(mesh_mundo.triangles, dtype=float)
        self.normales_cara = np.asarray(mesh_mundo.face_normals, dtype=float)
        self.caja_min, self.caja_max = np.asarray(mesh_mundo.bounds, dtype=float)
        self.color = np.asarray(color, dtype=float)
        self.reflectividad = reflectividad
        self.es_piso = es_piso


def recolectar_geometria(graph, reflectividades=None, nodos_piso=None):
    """Extrae las mallas del grafo como geometria de mundo para el trazador.

    reflectividades -- dict {nombre_objeto: reflectividad} para los rebotes de
        reflexion (0 si el objeto no esta listado).
    nodos_piso -- conjunto de nombres de objeto que deben sombrearse como piso
        (tablero de ajedrez en vez de color plano).

    Los nombres se derivan del nombre del objeto pasado a `add_object`, antes
    del sufijo interno `_mesh` / `_child_N`.
    """
    reflectividades = reflectividades or {}
    nodos_piso = nodos_piso or set()

    graph.calculate_global_transforms()

    geometrias = []
    for clave, nodo in graph.nodes.items():
        objeto = nodo.get("object")
        if not isinstance(objeto, tm.Trimesh):
            continue

        transformacion_global = graph.global_transforms[clave]
        mesh_mundo = objeto.copy()
        mesh_mundo.apply_transform(transformacion_global)

        nombre_base = clave.split("_mesh")[0]
        color = nodo.get("instance_attributes", {}).get("color", np.array([0.8, 0.8, 0.8]))
        geometrias.append(
            GeometriaEscena(
                clave,
                mesh_mundo,
                color,
                reflectividades.get(nombre_base, 0.0),
                nombre_base in nodos_piso,
            )
        )
    return geometrias


def _color_tablero(puntos, color_a, color_b, escala):
    celda = np.floor(puntos[:, [0, 2]] * escala).astype(int)
    par = (celda[:, 0] + celda[:, 1]) % 2 == 0
    return np.where(par[:, None], color_a, color_b)


def _rayos_tocan_aabb(origenes, direcciones, caja_min, caja_max):
    """Fase ancha: marca que rayos cruzan la caja AABB (metodo de los slabs).

    Para cada eje, el rayo entra y sale de la franja (slab) de la caja en dos
    distancias; el rayo cruza la caja si el mayor de los "entra" es menor o
    igual que el menor de los "sale". Vectorizado sobre rayos. Las direcciones
    con componente nula dan division infinita, que min/max manejan sin caso
    especial. Si un rayo no toca la caja de un objeto, no hace falta probarlo
    contra sus triangulos.
    """
    with np.errstate(divide="ignore"):
        inverso = 1.0 / direcciones
    distancia_a = (caja_min - origenes) * inverso
    distancia_b = (caja_max - origenes) * inverso
    entra = np.minimum(distancia_a, distancia_b).max(axis=1)
    sale = np.maximum(distancia_a, distancia_b).min(axis=1)
    return sale >= np.maximum(entra, 0.0)


def _interseccion_triangulos(origenes, direcciones, triangulos, bloque=2048):
    """Interseccion rayo-triangulo (Moller-Trumbore) vectorizada sobre rayos Y triangulos.

    Cada par (rayo, triangulo) se resuelve con coordenadas baricentricas (u, v)
    por broadcasting: los arreglos intermedios tienen forma (rayos, triangulos, 3).
    Eso evita el bucle por triangulo en Python, que domina cuando hay muchos
    triangulos y pocos rayos (las franjas del render progresivo). Los rayos se
    procesan en bloques para acotar la memoria de los arreglos (rayos, triangulos).
    Devuelve, por rayo, la distancia al triangulo mas cercano (inf si ninguno) y su indice.
    """
    n_rayos = origenes.shape[0]
    mejor_t = np.full(n_rayos, np.inf)
    indice = np.full(n_rayos, -1, dtype=int)

    vertice_0 = triangulos[:, 0]                 # (T, 3)
    arista_1 = triangulos[:, 1] - vertice_0      # (T, 3)
    arista_2 = triangulos[:, 2] - vertice_0      # (T, 3)

    for inicio in range(0, n_rayos, bloque):
        fin = min(inicio + bloque, n_rayos)
        origen = origenes[inicio:fin, None, :]    # (R, 1, 3)
        direccion = direcciones[inicio:fin, None, :]

        p = np.cross(direccion, arista_2)                       # (R, T, 3)
        determinante = np.einsum("rtk,tk->rt", p, arista_1)     # (R, T)
        paralelo = np.abs(determinante) < 1e-9
        inverso = np.where(paralelo, 0.0, 1.0 / np.where(paralelo, 1.0, determinante))

        desde_v0 = origen - vertice_0                           # (R, T, 3)
        u = np.einsum("rtk,rtk->rt", desde_v0, p) * inverso
        q = np.cross(desde_v0, arista_1)                        # (R, T, 3)
        v = np.einsum("rtk,rtk->rt", direccion, q) * inverso
        t = np.einsum("tk,rtk->rt", arista_2, q) * inverso

        dentro = (
            ~paralelo
            & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-4)
        )
        t = np.where(dentro, t, np.inf)

        triangulo_cercano = np.argmin(t, axis=1)                # (R,)
        fila = np.arange(fin - inicio)
        distancia = t[fila, triangulo_cercano]
        impacto = distancia < np.inf

        mejor_t[inicio:fin][impacto] = distancia[impacto]
        indice[inicio:fin][impacto] = triangulo_cercano[impacto]

    return mejor_t, indice


def _impactos_mas_cercanos(geometrias, origenes, direcciones):
    """Para cada rayo, devuelve la malla mas cercana que toca.

    Retorna (mejor_t, normal, indice_geometria, impacto), todos indexados por
    rayo. indice_geometria vale -1 donde no hubo impacto. Cada objeto se prueba
    solo con los rayos que cruzan su caja (fase ancha).
    """
    n_rayos = origenes.shape[0]
    mejor_t = np.full(n_rayos, np.inf)
    normal = np.zeros((n_rayos, 3))
    indice_geometria = np.full(n_rayos, -1, dtype=int)

    for indice, geo in enumerate(geometrias):
        candidato = _rayos_tocan_aabb(origenes, direcciones, geo.caja_min, geo.caja_max)
        rayos = np.where(candidato)[0]
        if rayos.size == 0:
            continue

        distancia, triangulo = _interseccion_triangulos(
            origenes[rayos], direcciones[rayos], geo.triangulos
        )
        mejora = distancia < mejor_t[rayos]
        rayos_mejorados = rayos[mejora]

        mejor_t[rayos_mejorados] = distancia[mejora]
        normal[rayos_mejorados] = geo.normales_cara[triangulo[mejora]]
        indice_geometria[rayos_mejorados] = indice

    return mejor_t, normal, indice_geometria, indice_geometria >= 0


def _en_sombra(geometrias, origenes_sombra, direcciones_sombra, distancia_luz):
    """Marca que rayos de sombra encuentran un bloqueador antes de la luz."""
    sombra = np.zeros(origenes_sombra.shape[0], dtype=bool)
    for geo in geometrias:
        candidato = _rayos_tocan_aabb(
            origenes_sombra, direcciones_sombra, geo.caja_min, geo.caja_max
        )
        rayos = np.where(candidato)[0]
        if rayos.size == 0:
            continue
        distancia, _ = _interseccion_triangulos(
            origenes_sombra[rayos], direcciones_sombra[rayos], geo.triangulos
        )
        bloquea = distancia < distancia_luz[rayos]
        sombra[rayos[bloquea]] = True
    return sombra


def _sombrear_phong(
    geometrias, puntos, normales, albedo, posicion_camara, luz,
    ambiente, difuso, especular, brillo, usar_sombras,
):
    """Iluminacion local Phong (ambiente + difuso + especular) con sombras.

    No recurre: la reflexion la maneja el bucle de rebotes de `trazar_grafo`.
    """
    hacia_luz = _normalizar_filas(luz["posicion"] - puntos)
    hacia_camara = _normalizar_filas(posicion_camara - puntos)

    color = ambiente * albedo

    iluminado = np.ones(puntos.shape[0], dtype=bool)
    if usar_sombras:
        distancia_luz = np.linalg.norm(luz["posicion"] - puntos, axis=1)
        sombra = _en_sombra(geometrias, puntos + normales * 1e-3, hacia_luz, distancia_luz)
        iluminado = ~sombra

    coseno_difuso = np.clip(np.sum(normales * hacia_luz, axis=1), 0.0, None)
    aporte_difuso = difuso * coseno_difuso[:, None] * albedo * luz["color"]
    media = _normalizar_filas(hacia_luz + hacia_camara)
    coseno_especular = np.clip(np.sum(normales * media, axis=1), 0.0, None)
    aporte_especular = especular * (coseno_especular[:, None] ** brillo) * luz["color"]

    color = color + iluminado[:, None] * (aporte_difuso + aporte_especular)
    return color


def trazar_grafo(
    geometrias,
    camara,
    luz,
    ancho,
    alto,
    ambiente=0.12,
    difuso=0.9,
    especular=0.5,
    brillo=40.0,
    color_cielo_arriba=(0.55, 0.70, 0.95),
    color_cielo_abajo=(0.85, 0.90, 0.98),
    escala_tablero=1.2,
    color_piso_a=(0.95, 0.95, 0.95),
    color_piso_b=(0.20, 0.20, 0.20),
    usar_sombras=True,
    rebotes=1,
    fila_inicio=0,
    fila_fin=None,
):
    """Traza la geometria recolectada y devuelve una imagen (n_filas, ancho, 3).

    La fila 0 de la imagen es la superior. Con fila_inicio / fila_fin se traza
    solo una banda de filas (para render progresivo); por defecto, la imagen
    completa. `rebotes` es la cantidad de rebotes de reflexion: 0 deja solo
    iluminacion local, 1 agrega el primer espejo, etc.

    Se traza de forma iterativa, no recursiva: en cada nivel se sombrea el
    impacto local y, para las superficies con reflectividad, se prepara el rayo
    espejado del siguiente nivel. `peso` acumula la atenuacion por reflectividad
    a lo largo de la cadena, asi que el aporte de cada rebote ya entra escalado.
    """
    if fila_fin is None:
        fila_fin = alto

    color_arriba = np.asarray(color_cielo_arriba)
    color_abajo = np.asarray(color_cielo_abajo)
    piso_a = np.asarray(color_piso_a)
    piso_b = np.asarray(color_piso_b)

    origenes, direcciones = camara.rayos(ancho, alto, fila_inicio, fila_fin)
    n_rayos = origenes.shape[0]

    color = np.zeros((n_rayos, 3))
    peso = np.ones((n_rayos, 3))  # atenuacion acumulada por reflexiones
    activo = np.ones(n_rayos, dtype=bool)

    for nivel in range(rebotes + 1):
        rayos_activos = np.where(activo)[0]
        if rayos_activos.size == 0:
            break

        origen = origenes[rayos_activos]
        direccion = direcciones[rayos_activos]
        mejor_t, normal, indice_geometria, impacto = _impactos_mas_cercanos(
            geometrias, origen, direccion
        )

        # rayos que no tocaron nada: toman el color del cielo y se apagan
        fallo = ~impacto
        if np.any(fallo):
            mezcla = 0.5 * (direccion[fallo, 1] + 1.0)
            fondo = (1.0 - mezcla[:, None]) * color_abajo + mezcla[:, None] * color_arriba
            indices_fallo = rayos_activos[fallo]
            color[indices_fallo] += peso[indices_fallo] * fondo
            activo[indices_fallo] = False

        if not np.any(impacto):
            continue

        indices_impacto = rayos_activos[impacto]
        direccion_impacto = direccion[impacto]
        puntos = origen[impacto] + direccion_impacto * mejor_t[impacto, None]

        normales = normal[impacto]
        # las normales de cara pueden apuntar hacia atras; las orientamos al rayo
        cara_invertida = np.sum(normales * direccion_impacto, axis=1) > 0.0
        normales[cara_invertida] *= -1.0

        # albedo y reflectividad por impacto (el piso usa tablero por posicion)
        albedo = np.zeros((puntos.shape[0], 3))
        reflectividad = np.zeros(puntos.shape[0])
        for indice, geo in enumerate(geometrias):
            seleccion = indice_geometria[impacto] == indice
            if not np.any(seleccion):
                continue
            if geo.es_piso:
                albedo[seleccion] = _color_tablero(puntos[seleccion], piso_a, piso_b, escala_tablero)
            else:
                albedo[seleccion] = geo.color
            reflectividad[seleccion] = geo.reflectividad

        color_local = _sombrear_phong(
            geometrias, puntos, normales, albedo, camara.posicion, luz,
            ambiente, difuso, especular, brillo, usar_sombras,
        )

        # las superficies reflectivas, mientras queden rebotes, reparten su color
        # entre lo local (1 - reflectividad) y el reflejo (reflectividad)
        va_a_reflejar = (reflectividad > 0.0) & (nivel < rebotes)
        factor_local = np.where(va_a_reflejar, 1.0 - reflectividad, 1.0)
        color[indices_impacto] += peso[indices_impacto] * factor_local[:, None] * color_local

        # preparar el rayo reflejado para el siguiente nivel
        no_refleja = indices_impacto[~va_a_reflejar]
        activo[no_refleja] = False

        if np.any(va_a_reflejar):
            indices_reflejo = indices_impacto[va_a_reflejar]
            normales_reflejo = normales[va_a_reflejar]
            direccion_reflejo = direccion_impacto[va_a_reflejar]
            direccion_espejada = (
                direccion_reflejo
                - 2.0 * np.sum(direccion_reflejo * normales_reflejo, axis=1)[:, None] * normales_reflejo
            )
            origenes[indices_reflejo] = puntos[va_a_reflejar] + normales_reflejo * 1e-3
            direcciones[indices_reflejo] = _normalizar_filas(direccion_espejada)
            peso[indices_reflejo] *= reflectividad[va_a_reflejar][:, None]

    return np.clip(color.reshape(fila_fin - fila_inicio, ancho, 3), 0.0, 1.0)
