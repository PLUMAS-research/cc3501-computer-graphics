"""Tonal art map (TAM): imágenes de achurado por tono y por nivel de mipmap.

Sigue la sección 4 de Praun, Hoppe, Webb y Finkelstein (2001), "Real-time
hatching". Un TAM es una grilla de imágenes: cada columna es un tono (de claro a
oscuro) y cada fila un nivel de mipmap (de grueso a fino). Los trazos cumplen la
propiedad de anidamiento: un trazo de la imagen (nivel, tono) aparece en todas
las imágenes más finas del mismo tono y en todas las de tonos más oscuros. Así,
al mezclar tonos vecinos o niveles vecinos solo cambian unos pocos píxeles y los
trazos no "nadan" al acercar el objeto o al cambiar la luz.

Construcción, columna por columna (de claro a oscuro):

1. La columna parte como copia de la anterior (o en blanco si es la primera).
2. Sus imágenes se llenan de la más gruesa a la más fina. A cada una se le
   agregan trazos hasta que su tinta media alcanza el tono de la columna. Un
   trazo agregado a un nivel también se agrega a todos los niveles más finos.
3. Cada trazo se elige entre muchos candidatos al azar (en la figura 2 del
   paper, 1000 en los tonos claros y 100 en los oscuros). La bondad de un
   candidato es la tinta que agregaría a todas las imágenes que lo reciben,
   como fracción de la tinta que dejaría sobre papel blanco. La tinta se mide
   también en una pirámide de cada imagen, dibujada a menor resolución con el
   mismo ancho de trazo en píxeles: dos trazos vecinos que no se tocan en la
   imagen fina se solapan en los niveles gruesos de su pirámide, así que el
   candidato que llena un hueco agrega más tinta que el que cae junto a otro
   trazo. Eso reparte los trazos de forma pareja.

Los trazos tienen el mismo ancho en píxeles en todos los niveles: al alejar el
objeto, el mipmap elige un nivel más grueso, con menos trazos pero igual de
anchos en pantalla, y el tono se mantiene. Las imágenes son toroidales (un
trazo que sale por un borde entra por el opuesto) para poder repetirlas.

El módulo no depende de OpenGL: se puede ejecutar y probar sin ventana.
"""

import hashlib
from pathlib import Path

import numpy as np

# niveles de mipmap que se construyen con trazos, del más grueso al más fino
RESOLUCIONES = (32, 64, 128, 256)

# tinta media de cada columna. El papel blanco es un séptimo tono implícito: al
# dibujar, recibe el peso que les falta a los seis tonos para sumar 1
TONOS = tuple(k / 7 for k in range(1, 7))


def _perfil_cometa(u):
    """Cabeza redonda en u = 0 y cola que se afina hasta u = 1 (figura 2)."""
    return np.sqrt(np.clip(1.0 - u, 0.0, 1.0))


def _perfil_huso(u):
    """Delgado en los extremos y más grueso cerca del primer tercio."""
    seno = np.maximum(np.sin(np.pi * np.clip(u, 0.0, 1.0)), 0.0)
    return np.clip(seno**0.7 * (1.15 - 0.3 * u), 0.0, 1.0)


# Estilos de TAM. Lo que cambia entre ellos es lo que el paper menciona como
# fuente de variación: el ángulo de los trazos, el calendario de achurado
# cruzado, el rango de largos y la forma del trazo.
#   largos: fracción del ancho de la imagen en que se agrega el trazo
#   desvio_grados: perturbación del ángulo respecto de la horizontal o vertical
#   semiancho: mitad del ancho máximo del trazo, en píxeles de cada nivel
#   direcciones: por cada tono, "h" (horizontal), "v" (vertical) o "hv" (ambas)
#   candidatos: cuántos trazos al azar se comparan en el tono más claro y en
#   el más oscuro (en los intermedios se interpola)
ESTILOS = {
    # El TAM de la figura 2 del paper: trazos casi horizontales en los tres
    # tonos claros, verticales en el cuarto y quinto, y ambos en el sexto.
    "figura2": {
        "largos": (0.3, 1.0),
        "desvio_grados": 1.0,
        "semiancho": 2.4,
        "perfil": _perfil_cometa,
        "direcciones": ("h", "h", "h", "v", "v", "hv"),
        "candidatos": (1000, 100),
    },
    # Trazos delgados que se afinan en las dos puntas, con más variación de
    # ángulo, parecidos a los del conejo de la figura 5.
    "conejo": {
        "largos": (0.3, 0.8),
        "desvio_grados": 6.0,
        "semiancho": 0.6,
        "perfil": _perfil_huso,
        "direcciones": ("h", "h", "h", "v", "v", "hv"),
        "candidatos": (400, 60),
    },
}

ANGULOS_BASE = {"h": (0.0,), "v": (np.pi / 2,), "hv": (0.0, np.pi / 2)}

# los candidatos se evalúan en bloques para acotar la memoria del broadcasting
TAMANO_BLOQUE = 200


def _rasterizar(cx, cy, largo, angulo, resolucion, semiancho, perfil):
    """Cobertura de N trazos dibujados en una imagen toroidal de lado `resolucion`.

    Cada trazo va desde su cabeza P0 hasta su cola P1, centrado en (cx, cy) en
    [0, 1)^2, con largo como fracción del ancho y ángulo en radianes. El ancho
    del trazo en píxeles no depende de la resolución.

    Se recorren las columnas que cruza el trazo y, en cada una, unas pocas filas
    alrededor de la línea central. Los trazos más verticales que horizontales
    se reflejan respecto de la diagonal (x <-> y) para recorrerlos igual, y sus
    índices se vuelven a reflejar al final.

    Devuelve (indices, cobertura), ambos de forma (N, M): índices planos de
    píxel (fila * resolucion + columna) y cobertura en [0, 1].
    """
    r = resolucion
    dx = np.cos(angulo)
    dy = np.sin(angulo)
    vertical = np.abs(dy) > np.abs(dx)

    # en el marco reflejado, el trazo vertical queda horizontal
    px = np.where(vertical, cy, cx) * r
    py = np.where(vertical, cx, cy) * r
    dx, dy = np.where(vertical, dy, dx), np.where(vertical, dx, dy)

    largo_px = largo * r
    p0x = px - 0.5 * largo_px * dx
    p0y = py - 0.5 * largo_px * dy

    margen = semiancho + 2.0
    x_inicio = np.floor(np.minimum(p0x, p0x + largo_px * dx) - margen)
    cantidad_columnas = int(np.ceil((largo_px * np.abs(dx)).max() + 2.0 * margen))
    pendiente = dy / dx
    # filas que puede tocar el trazo en una columna: su ancho vertical más la
    # inclinación dentro de la columna
    extension = semiancho / np.abs(dx).min() + 0.5 * np.abs(pendiente).max() + 0.5
    cantidad_filas = int(np.ceil(2.0 * extension)) + 1

    columnas = x_inicio[:, None] + np.arange(cantidad_columnas)
    y_linea = p0y[:, None] + (columnas + 0.5 - p0x[:, None]) * pendiente[:, None]
    primera_fila = np.floor(y_linea - extension)

    def por_trazo(valores):
        return valores.astype(np.float32)[:, None, None]

    # coordenadas del centro de cada píxel respecto de la cabeza P0, a lo largo
    # del trazo y transversal a él
    qx = (columnas + 0.5 - p0x[:, None]).astype(np.float32)[:, :, None]
    qy = (primera_fila + 0.5 - p0y[:, None]).astype(np.float32)[:, :, None] + np.arange(
        cantidad_filas, dtype=np.float32
    )
    a_lo_largo = qx * por_trazo(dx) + qy * por_trazo(dy)
    transversal = qx * por_trazo(dy) - qy * por_trazo(dx)
    largo_trazo = por_trazo(largo_px)

    # cuerpo: cobertura de una banda de semiancho h sobre un píxel de lado 1
    h = semiancho * perfil(a_lo_largo / largo_trazo)
    cuerpo = np.minimum(h, transversal + 0.5) - np.maximum(-h, transversal - 0.5)

    # extremos redondeados, con el radio que tiene el perfil en cada punta
    fuera = np.maximum(-a_lo_largo, a_lo_largo - largo_trazo)
    radio_punta = np.where(
        a_lo_largo < 0.0, semiancho * perfil(np.zeros(1)), semiancho * perfil(np.ones(1))
    )
    punta = radio_punta - np.hypot(fuera, transversal) + 0.5
    cobertura = np.clip(np.where(fuera > 0.0, punta, cuerpo), 0.0, 1.0).astype(np.float32)

    columna_pixel = np.mod(columnas.astype(np.int32), r)[:, :, None]
    fila_pixel = np.mod(
        primera_fila.astype(np.int32)[:, :, None] + np.arange(cantidad_filas, dtype=np.int32), r
    )
    indices = np.where(
        vertical[:, None, None],
        columna_pixel * r + fila_pixel,  # se deshace la reflexión
        fila_pixel * r + columna_pixel,
    )
    n = len(cx)
    return indices.reshape(n, -1), cobertura.reshape(n, -1)


def _imagen_de_trazo(cx, cy, largo, angulo, resolucion, semiancho, perfil):
    """Cobertura de un solo trazo como imagen plana de resolucion^2 píxeles."""
    indices, cobertura = _rasterizar(
        np.array([cx]), np.array([cy]), np.array([largo]), np.array([angulo]),
        resolucion, semiancho, perfil,
    )
    imagen = np.zeros(resolucion * resolucion, dtype=np.float32)
    # un trazo de largo 1 puede dar la vuelta y pisar sus propios píxeles
    np.maximum.at(imagen, indices[0], cobertura[0])
    return imagen


def _resoluciones_de_piramide(resolucion):
    """Niveles de la pirámide de una imagen: su resolución y las menores."""
    return [r for r in RESOLUCIONES if r <= resolucion]


def _bondad(cx, cy, largo, angulo, imagenes_por_resolucion, semiancho, perfil):
    """Fracción de su tinta que cada candidato agregaría a las imágenes receptoras.

    `imagenes_por_resolucion` agrupa por resolución la blancura de todas las
    imágenes (y niveles de pirámide) que recibirían el trazo, así cada
    candidato se rasteriza una sola vez por resolución. La tinta nueva en un
    píxel es la cobertura del trazo por la blancura previa, y la tinta media de
    una imagen de lado r cambia en esa suma dividida por r^2.

    El paper divide la tinta agregada por el largo del trazo, para no favorecer
    a los largos. Aquí se divide por la tinta que el trazo dejaría sobre papel
    blanco: al dividir por el largo, la tinta de las puntas pesa más en un
    trazo corto y los cortos salen favorecidos.
    """
    cantidad = len(cx)
    tinta_agregada = np.zeros(cantidad)
    tinta_en_blanco = np.zeros(cantidad)
    for r, imagenes in imagenes_por_resolucion.items():
        blancura_sumada = np.sum(imagenes, axis=0)
        for inicio in range(0, cantidad, TAMANO_BLOQUE):
            bloque = slice(inicio, inicio + TAMANO_BLOQUE)
            indices, cobertura = _rasterizar(
                cx[bloque], cy[bloque], largo[bloque], angulo[bloque], r, semiancho, perfil
            )
            tinta_agregada[bloque] += (cobertura * blancura_sumada[indices]).sum(axis=1) / (r * r)
            tinta_en_blanco[bloque] += cobertura.sum(axis=1) * len(imagenes) / (r * r)
    return tinta_agregada / tinta_en_blanco


def generar_tam(estilo="conejo", semilla=7, reporte=print):
    """Construye el TAM y devuelve sus niveles de mipmap.

    El resultado es una lista de arreglos de forma (6, lado, lado) con la tinta
    de cada tono en [0, 1], del nivel más fino (256) al más grueso (1). Los
    niveles de 32 hacia arriba se construyen con trazos; los menores son
    promedios de 2x2 del nivel de 32, para que el mipmap esté completo.
    """
    parametros = ESTILOS[estilo]
    semiancho = parametros["semiancho"]
    perfil = parametros["perfil"]
    largo_minimo, largo_maximo = parametros["largos"]
    desvio = np.radians(parametros["desvio_grados"])
    rng = np.random.default_rng(semilla)

    # blancura (1 - tinta) de cada imagen de la columna en construcción y de su
    # pirámide: blancura[nivel][resolucion] es un arreglo plano
    blancura = [
        {r: np.ones(r * r, dtype=np.float32) for r in _resoluciones_de_piramide(lado)}
        for lado in RESOLUCIONES
    ]
    columnas = []
    largos_elegidos = []

    candidatos_claro, candidatos_oscuro = parametros["candidatos"]
    for indice_tono, tono in enumerate(TONOS):
        fraccion = indice_tono / (len(TONOS) - 1)
        cantidad = round(candidatos_claro + fraccion * (candidatos_oscuro - candidatos_claro))
        angulos_base = ANGULOS_BASE[parametros["direcciones"][indice_tono]]

        # la columna parte como copia de la anterior: los arreglos de blancura
        # se siguen oscureciendo en el lugar
        for nivel, lado in enumerate(RESOLUCIONES):
            receptores = blancura[nivel:]
            imagenes_por_resolucion = {}
            for piramide in receptores:
                for r, imagen in piramide.items():
                    imagenes_por_resolucion.setdefault(r, []).append(imagen)

            trazos_nuevos = 0
            while 1.0 - blancura[nivel][lado].mean() < tono:
                cx = rng.random(cantidad)
                cy = rng.random(cantidad)
                largo = rng.uniform(largo_minimo, largo_maximo, cantidad)
                angulo = rng.choice(angulos_base, cantidad) + rng.normal(0.0, desvio, cantidad)

                elegido = int(np.argmax(
                    _bondad(cx, cy, largo, angulo, imagenes_por_resolucion, semiancho, perfil)
                ))
                largos_elegidos.append(largo[elegido])
                # el trazo elegido entra a esta imagen, a las más finas de la
                # columna y a las pirámides de todas ellas
                for r, imagenes in imagenes_por_resolucion.items():
                    trazo = _imagen_de_trazo(
                        cx[elegido], cy[elegido], largo[elegido], angulo[elegido],
                        r, semiancho, perfil,
                    )
                    for imagen in imagenes:
                        imagen *= 1.0 - trazo
                trazos_nuevos += 1
            if reporte:
                reporte(
                    f"  tono {indice_tono + 1}/6, nivel {lado}x{lado}: "
                    f"{trazos_nuevos} trazos nuevos, tinta media "
                    f"{1.0 - blancura[nivel][lado].mean():.3f} (meta {tono:.3f})"
                )
        columnas.append([
            1.0 - blancura[nivel][lado].reshape(lado, lado)
            for nivel, lado in enumerate(RESOLUCIONES)
        ])

    if reporte:
        reporte(
            f"  {len(largos_elegidos)} trazos en total, largo medio "
            f"{np.mean(largos_elegidos):.2f} del ancho de la imagen"
        )

    niveles = [
        np.stack([columna[nivel] for columna in columnas]).astype(np.float32)
        for nivel in reversed(range(len(RESOLUCIONES)))
    ]
    # niveles menores que 32: promedio de bloques de 2x2
    while niveles[-1].shape[-1] > 1:
        anterior = niveles[-1]
        lado = anterior.shape[-1] // 2
        niveles.append(anterior.reshape(len(TONOS), lado, 2, lado, 2).mean(axis=(2, 4)))
    return niveles


def cargar_o_generar(estilo, directorio, semilla=7, reporte=print):
    """Lee el TAM del directorio de caché o lo genera y lo guarda ahí.

    El nombre del archivo incluye una huella de este módulo: si se cambia un
    parámetro o el algoritmo, el TAM se vuelve a generar.
    """
    huella = hashlib.sha1(Path(__file__).read_bytes() + f"{estilo}-{semilla}".encode()).hexdigest()[:12]
    directorio = Path(directorio)
    archivo = directorio / f"tam_{estilo}_{huella}.npz"
    if archivo.exists():
        datos = np.load(archivo)
        return [datos[f"nivel_{k}"] for k in range(len(datos.files))]
    if reporte:
        reporte(f"Generando el TAM '{estilo}' (se guarda en {archivo})...")
    niveles = generar_tam(estilo, semilla, reporte)
    directorio.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archivo, **{f"nivel_{k}": nivel for k, nivel in enumerate(niveles)})
    return niveles
