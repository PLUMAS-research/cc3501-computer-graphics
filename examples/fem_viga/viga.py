"""Una viga en voladizo simulada con dos modelos sobre la misma malla.

La viga es un rectangulo de largo L y alto H empotrado por su borde izquierdo,
que cuelga bajo su propio peso. Los dos modelos comparten la malla de
triangulos, la masa de cada vertice y el integrador. Lo que cambia es como se
calculan las fuerzas internas:

- masa-resorte: cada arista de la malla es un resorte de rigidez k.
- elementos finitos (FEM): cada triangulo es un pedazo de material con modulo
  de Young E y coeficiente de Poisson nu, y la fuerza sale de la energia que
  acumula al deformarse.

Las unidades son del sistema internacional (metros, kilogramos, segundos). La
viga se trata como una lamina (tension plana): su espesor multiplica por igual
las fuerzas y las masas, asi que no cambia el movimiento.
"""

import numpy as np

IDENTIDAD = np.eye(2)


def _acumular(indices, valores, cantidad):
    """Suma los vectores `valores` (M, 2) en las filas `indices` de un (cantidad, 2)."""
    return np.stack(
        [np.bincount(indices, valores[:, eje], minlength=cantidad) for eje in range(2)],
        axis=1,
    )


def crear_malla(largo, alto, columnas, filas, densidad):
    """Triangula el rectangulo [0, largo] x [-alto, 0] y precalcula el reposo."""
    xs = np.linspace(0.0, largo, columnas + 1)
    ys = np.linspace(0.0, -alto, filas + 1)
    posiciones = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

    triangulos = []
    for fila in range(filas):
        for columna in range(columnas):
            # esquinas de la celda: a y b arriba, c y d abajo.
            a = fila * (columnas + 1) + columna
            b, c, d = a + 1, a + columnas + 1, a + columnas + 2
            # la diagonal alterna como en un tablero, asi la malla no tiene
            # una direccion preferida.
            if (fila + columna) % 2 == 0:
                triangulos += [(a, c, d), (a, d, b)]
            else:
                triangulos += [(a, c, b), (b, c, d)]
    triangulos = np.array(triangulos)

    # aristas de cada triangulo en reposo, como columnas de una matriz 2x2.
    p0, p1, p2 = (posiciones[triangulos[:, vertice]] for vertice in range(3))
    aristas_reposo = np.stack([p1 - p0, p2 - p0], axis=2)
    areas = 0.5 * np.linalg.det(aristas_reposo)

    # masa de cada vertice: un tercio de la masa de cada triangulo que lo toca.
    masas = np.bincount(
        triangulos.ravel(), np.repeat(densidad * areas / 3.0, 3), minlength=len(posiciones)
    )

    # cada lado de un triangulo es una arista de la malla; las interiores
    # aparecen en dos triangulos y las del borde en uno.
    lados = np.sort(triangulos[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    aristas, arista_de_lado, triangulos_por_arista = np.unique(
        lados, axis=0, return_inverse=True, return_counts=True
    )

    return {
        "reposo": posiciones,
        "triangulos": triangulos,
        "inversas_reposo": np.linalg.inv(aristas_reposo),
        "areas": areas,
        "masas": masas,
        "aristas": aristas,
        "largos_reposo": np.linalg.norm(
            posiciones[aristas[:, 1]] - posiciones[aristas[:, 0]], axis=1
        ),
        "arista_de_lado": arista_de_lado.reshape(-1, 3),
        "triangulos_por_arista": triangulos_por_arista,
        # el borde izquierdo esta empotrado.
        "libres": posiciones[:, 0] > 0.0,
    }


def fuerzas_resortes(malla, posiciones, rigidez):
    """Fuerzas de Hooke de las aristas y energia acumulada en cada triangulo."""
    aristas = malla["aristas"]
    delta = posiciones[aristas[:, 1]] - posiciones[aristas[:, 0]]
    largos = np.linalg.norm(delta, axis=1)
    estiramiento = largos - malla["largos_reposo"]
    # fuerza sobre el primer extremo de cada arista; el segundo recibe la opuesta.
    fuerza = (rigidez * estiramiento / largos)[:, None] * delta

    cantidad = len(posiciones)
    fuerzas = _acumular(aristas[:, 0], fuerza, cantidad) - _acumular(
        aristas[:, 1], fuerza, cantidad
    )

    # la energia 1/2 k (L - L0)^2 de cada resorte se reparte entre los
    # triangulos que lo comparten.
    energia_arista = 0.5 * rigidez * estiramiento**2 / malla["triangulos_por_arista"]
    return fuerzas, energia_arista[malla["arista_de_lado"]].sum(axis=1)


def fuerzas_fem(malla, posiciones, young, poisson):
    """Fuerzas internas y energia de cada triangulo con elementos finitos.

    El gradiente de deformacion F = Ds Dm^-1 lleva las aristas en reposo (Dm)
    a las actuales (Ds). F mezcla un giro R con un estiramiento S (F = R S), y
    solo el estiramiento acumula energia. Por eso se gira el triangulo de
    vuelta, se mide su deformacion eps = S - I y se le aplica la ley de Hooke
    del material, con energia por unidad de area

        psi = mu eps:eps + (lambda / 2) traza(eps)^2.

    Los parametros de Lame mu y lambda se calculan desde E y nu. La tension
    resultante se vuelve a girar con R para obtener la fuerza sobre cada vertice.
    """
    lame_mu = young / (2.0 * (1.0 + poisson))
    lame_lambda = young * poisson / (1.0 - poisson**2)

    triangulos = malla["triangulos"]
    inversas_reposo = malla["inversas_reposo"]
    p0, p1, p2 = (posiciones[triangulos[:, vertice]] for vertice in range(3))
    gradiente = np.stack([p1 - p0, p2 - p0], axis=2) @ inversas_reposo

    # en 2D el giro de F tiene una formula cerrada.
    angulo = np.arctan2(
        gradiente[:, 1, 0] - gradiente[:, 0, 1], gradiente[:, 0, 0] + gradiente[:, 1, 1]
    )
    coseno, seno = np.cos(angulo), np.sin(angulo)
    giro = np.stack([np.stack([coseno, -seno], axis=1), np.stack([seno, coseno], axis=1)], axis=1)
    deformacion = np.swapaxes(giro, 1, 2) @ gradiente - IDENTIDAD
    traza = deformacion[:, 0, 0] + deformacion[:, 1, 1]
    tension = 2.0 * lame_mu * deformacion + lame_lambda * traza[:, None, None] * IDENTIDAD

    # las columnas de -area * R tension Dm^-T son las fuerzas sobre p1 y p2; la
    # de p0 es la suma de ambas con signo opuesto.
    fuerza_lados = (
        -malla["areas"][:, None, None]
        * (giro @ tension)
        @ np.swapaxes(inversas_reposo, 1, 2)
    )
    fuerza_p1, fuerza_p2 = fuerza_lados[:, :, 0], fuerza_lados[:, :, 1]
    por_vertice = np.stack([-fuerza_p1 - fuerza_p2, fuerza_p1, fuerza_p2], axis=1)
    fuerzas = _acumular(triangulos.ravel(), por_vertice.reshape(-1, 2), len(posiciones))

    densidad_energia = lame_mu * np.sum(deformacion**2, axis=(1, 2)) + 0.5 * lame_lambda * traza**2
    return fuerzas, malla["areas"] * densidad_energia


def paso_maximo(fuerza, malla, epsilon=1e-6):
    """Mayor paso estable del Euler semi-implicito, dt < 2 / omega_max.

    omega_max^2 es el mayor valor propio de M^-1 K, donde K = -df/dx es la
    matriz de rigidez en reposo. K se arma columna por columna con diferencias
    finitas: se mueve un grado de libertad libre y se mide como cambian las
    fuerzas.
    """
    reposo = malla["reposo"].ravel()
    libres = np.repeat(malla["libres"], 2)
    columnas = []
    for grado in np.flatnonzero(libres):
        movida = reposo.copy()
        movida[grado] += epsilon
        fuerzas, _ = fuerza(movida.reshape(-1, 2))
        columnas.append(-fuerzas.ravel()[libres] / epsilon)
    rigidez = np.array(columnas).T

    # M^-1/2 K M^-1/2 tiene los mismos valores propios que M^-1 K y es simetrica.
    raiz_masas = np.sqrt(np.repeat(malla["masas"], 2)[libres])
    valores_propios = np.linalg.eigvalsh(rigidez / np.outer(raiz_masas, raiz_masas))
    return 2.0 / np.sqrt(valores_propios.max())


def nuevo_estado(malla):
    """Viga en reposo: posiciones, velocidades y energia de cada triangulo."""
    return {
        "posiciones": malla["reposo"].copy(),
        "velocidades": np.zeros_like(malla["reposo"]),
        "energia": np.zeros(len(malla["triangulos"])),
        "exploto": False,
    }


def paso(estado, malla, fuerza, dt, substeps, gravedad, amortiguacion, fijos=()):
    """Avanza dt segundos con Euler semi-implicito en pasos de dt / substeps.

    `fuerza(posiciones)` devuelve las fuerzas internas y la energia de cada
    triangulo. `fijos` son vertices que no se mueven en este cuadro (el que
    arrastra el mouse), ademas de los del borde empotrado.
    """
    if estado["exploto"]:
        return

    quietos = ~malla["libres"]
    quietos[list(fijos)] = True
    sub_dt = dt / substeps
    for _ in range(substeps):
        fuerzas, estado["energia"] = fuerza(estado["posiciones"])
        aceleracion = fuerzas / malla["masas"][:, None] - amortiguacion * estado["velocidades"]
        aceleracion[:, 1] -= gravedad
        estado["velocidades"] += sub_dt * aceleracion
        estado["velocidades"][quietos] = 0.0
        estado["posiciones"] += sub_dt * estado["velocidades"]

    posiciones = estado["posiciones"]
    if not np.all(np.isfinite(posiciones)) or np.max(np.abs(posiciones)) > 100.0:
        estado["exploto"] = True
