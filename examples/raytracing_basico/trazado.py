"""Trazador de rayos minimalista sobre primitivas analiticas.

Este modulo aisla el nucleo del ray tracing: por cada pixel se lanza un rayo,
se busca la interseccion mas cercana con la escena y se calcula el color con un
modelo de iluminacion local (Phong) mas rayos de sombra y reflexion. No depende
de OpenGL ni de pyglet, asi que se puede ejecutar y probar sin ventana.

La matematica de interseccion rayo-esfera y rayo-plano sigue la del apunte
(unidad de trazado de rayos). Las primitivas son analiticas a proposito: la
ecuacion de interseccion queda a la vista, sin mallas de por medio.
"""

import numpy as np


def normalizar(vector):
    """Devuelve el vector unitario sin modificar el original."""
    return vector / np.linalg.norm(vector)


class Esfera:
    def __init__(self, centro, radio, color, reflectividad=0.3):
        self.centro = np.asarray(centro, dtype=float)
        self.radio = float(radio)
        self.color_base = np.asarray(color, dtype=float)
        self.reflectividad = reflectividad

    def intersecta(self, origen, direccion):
        """Distancia desde origen hasta la esfera, o np.inf si no la toca.

        Resuelve la cuadratica ||origen + t*direccion - centro||^2 = radio^2.
        Asume direccion unitaria, asi que el coeficiente cuadratico es 1.
        """
        centro_a_origen = origen - self.centro
        b = 2.0 * np.dot(direccion, centro_a_origen)
        c = np.dot(centro_a_origen, centro_a_origen) - self.radio * self.radio
        discriminante = b * b - 4.0 * c
        if discriminante < 0.0:
            return np.inf
        raiz = np.sqrt(discriminante)
        t0 = (-b - raiz) / 2.0
        t1 = (-b + raiz) / 2.0
        if t0 > 1e-4:
            return t0
        if t1 > 1e-4:
            return t1
        return np.inf

    def normal(self, punto):
        return normalizar(punto - self.centro)

    def color(self, punto):
        return self.color_base


class Plano:
    def __init__(self, punto, normal, color_a, color_b, reflectividad=0.2, escala_tablero=1.0):
        self.punto = np.asarray(punto, dtype=float)
        self.normal_plano = normalizar(np.asarray(normal, dtype=float))
        self.color_a = np.asarray(color_a, dtype=float)
        self.color_b = np.asarray(color_b, dtype=float)
        self.reflectividad = reflectividad
        self.escala_tablero = escala_tablero

    def intersecta(self, origen, direccion):
        denominador = np.dot(direccion, self.normal_plano)
        if abs(denominador) < 1e-6:
            return np.inf
        t = np.dot(self.punto - origen, self.normal_plano) / denominador
        return t if t > 1e-4 else np.inf

    def normal(self, punto):
        return self.normal_plano

    def color(self, punto):
        """Tablero de ajedrez segun la posicion sobre el plano."""
        celda_x = int(np.floor(punto[0] * self.escala_tablero))
        celda_z = int(np.floor(punto[2] * self.escala_tablero))
        return self.color_a if (celda_x + celda_z) % 2 == 0 else self.color_b


class Camara:
    """Camara en perspectiva que genera un rayo primario por pixel.

    Sigue la construccion del apunte: el origen es fijo (eye) y la direccion
    varia segun el pixel. forward/right/up forman el sistema local de la camara.
    """

    def __init__(self, posicion, objetivo, arriba_mundo, fov_grados, aspecto):
        self.posicion = np.asarray(posicion, dtype=float)
        self.forward = normalizar(np.asarray(objetivo, dtype=float) - self.posicion)
        self.right = normalizar(np.cross(self.forward, np.asarray(arriba_mundo, dtype=float)))
        self.up = np.cross(self.right, self.forward)

        fov_radianes = np.radians(fov_grados)
        self.half_height = np.tan(fov_radianes / 2.0)
        self.half_width = self.half_height * aspecto

    def rayo(self, columna, fila, ancho, alto):
        """Direccion del rayo para el pixel (columna, fila).

        fila 0 es la parte superior de la imagen (v = +1).
        """
        u = 2.0 * (columna + 0.5) / ancho - 1.0
        v = 1.0 - 2.0 * (fila + 0.5) / alto
        punto_plano = (
            self.posicion
            + self.forward
            + u * self.half_width * self.right
            + v * self.half_height * self.up
        )
        return normalizar(punto_plano - self.posicion)

    def rayos(self, ancho, alto, fila_inicio=0, fila_fin=None):
        """Genera los rayos primarios de una banda de filas (version vectorizada).

        Devuelve (origenes, direcciones), ambos de forma (n_filas*ancho, 3) en
        orden por filas. La fila 0 es la superior de la imagen. Con fila_inicio /
        fila_fin se traza solo una franja (para render progresivo); el valor de v
        usa el alto completo, asi la franja calza con la imagen total. Util para
        trazar mallas con trimesh, que acepta arreglos de rayos en una llamada.
        """
        if fila_fin is None:
            fila_fin = alto
        u = 2.0 * (np.arange(ancho) + 0.5) / ancho - 1.0
        v = 1.0 - 2.0 * (np.arange(fila_inicio, fila_fin) + 0.5) / alto
        malla_u, malla_v = np.meshgrid(u, v)

        direcciones = (
            self.forward
            + malla_u[..., None] * self.half_width * self.right
            + malla_v[..., None] * self.half_height * self.up
        ).reshape(-1, 3)
        direcciones /= np.linalg.norm(direcciones, axis=1, keepdims=True)

        origenes = np.tile(self.posicion, (direcciones.shape[0], 1))
        return origenes, direcciones


def interseccion_mas_cercana(escena, origen, direccion):
    """Objeto y distancia del primer impacto del rayo, o (None, inf)."""
    objeto_cercano = None
    distancia_cercana = np.inf
    for objeto in escena:
        distancia = objeto.intersecta(origen, direccion)
        if distancia < distancia_cercana:
            distancia_cercana = distancia
            objeto_cercano = objeto
    return objeto_cercano, distancia_cercana


def en_sombra(escena, punto, hacia_luz, distancia_luz):
    """True si algun objeto bloquea la linea entre el punto y la luz."""
    for objeto in escena:
        distancia = objeto.intersecta(punto, hacia_luz)
        if distancia < distancia_luz:
            return True
    return False


def color_rayo(
    escena,
    origen,
    direccion,
    luz,
    posicion_camara,
    ambiente,
    difuso,
    especular,
    brillo,
    color_fondo,
    usar_sombras,
    rebotes_restantes,
):
    """Color que devuelve un rayo, con sombras y reflexion recursiva."""
    objeto, distancia = interseccion_mas_cercana(escena, origen, direccion)
    if objeto is None:
        return color_fondo(direccion)

    punto = origen + direccion * distancia
    normal = objeto.normal(punto)
    albedo = objeto.color(punto)

    hacia_luz = normalizar(luz["posicion"] - punto)
    hacia_camara = normalizar(posicion_camara - punto)

    # componente ambiental: siempre presente
    color = ambiente * albedo

    punto_desplazado = punto + normal * 1e-4
    distancia_luz = np.linalg.norm(luz["posicion"] - punto)
    iluminado = not (usar_sombras and en_sombra(escena, punto_desplazado, hacia_luz, distancia_luz))

    if iluminado:
        # componente difusa (Lambert)
        coseno_difuso = max(np.dot(normal, hacia_luz), 0.0)
        color = color + difuso * coseno_difuso * albedo * luz["color"]
        # componente especular (Blinn-Phong)
        media = normalizar(hacia_luz + hacia_camara)
        coseno_especular = max(np.dot(normal, media), 0.0)
        color = color + especular * (coseno_especular ** brillo) * luz["color"]

    # reflexion: lanzamos un rayo espejado y mezclamos segun la reflectividad
    if rebotes_restantes > 0 and objeto.reflectividad > 0.0:
        direccion_reflejada = direccion - 2.0 * np.dot(direccion, normal) * normal
        color_reflejo = color_rayo(
            escena,
            punto_desplazado,
            normalizar(direccion_reflejada),
            luz,
            posicion_camara,
            ambiente,
            difuso,
            especular,
            brillo,
            color_fondo,
            usar_sombras,
            rebotes_restantes - 1,
        )
        color = (1.0 - objeto.reflectividad) * color + objeto.reflectividad * color_reflejo

    return color


def trazar(
    escena,
    camara,
    luz,
    ancho,
    alto,
    ambiente=0.10,
    difuso=0.9,
    especular=0.6,
    brillo=50.0,
    color_cielo_arriba=(0.55, 0.70, 0.95),
    color_cielo_abajo=(0.85, 0.90, 0.98),
    usar_sombras=True,
    rebotes=2,
    fila_inicio=0,
    fila_fin=None,
):
    """Traza una banda de filas y devuelve una imagen (n_filas, ancho, 3) en [0,1].

    La fila 0 de la imagen corresponde a la parte superior. Con fila_inicio /
    fila_fin se traza solo una franja (para render progresivo); por defecto se
    traza la imagen completa.
    """
    if fila_fin is None:
        fila_fin = alto

    color_arriba = np.asarray(color_cielo_arriba, dtype=float)
    color_abajo = np.asarray(color_cielo_abajo, dtype=float)

    def color_fondo(direccion):
        # gradiente vertical simple segun la altura del rayo
        t = 0.5 * (direccion[1] + 1.0)
        return (1.0 - t) * color_abajo + t * color_arriba

    imagen = np.zeros((fila_fin - fila_inicio, ancho, 3), dtype=float)
    for indice_fila, fila in enumerate(range(fila_inicio, fila_fin)):
        for columna in range(ancho):
            direccion = camara.rayo(columna, fila, ancho, alto)
            color = color_rayo(
                escena,
                camara.posicion,
                direccion,
                luz,
                camara.posicion,
                ambiente,
                difuso,
                especular,
                brillo,
                color_fondo,
                usar_sombras,
                rebotes,
            )
            imagen[indice_fila, columna] = np.clip(color, 0.0, 1.0)
    return imagen
