"""La jerarquia de huesos y el paso de posiciones globales a locales.

Un estimador de pose entrega una posicion global por articulacion, sin ninguna
jerarquia. Un grafo de escena necesita lo contrario: una transformacion local
por nodo, relativa a su padre. La conversion entre ambas es la regla de
composicion de la unidad leida al reves,

    M_n = inv(G_padre) @ G_n,

que es la misma linea con que el ejemplo `katamari` reparenta un objeto sin que
salte de lugar.
"""

import numpy as np

import grafica.transformations as tr

# las articulaciones que usamos. Los nombres izq/der son los de la persona,
# que en una imagen sin espejo aparecen cambiados de lado.
ARTICULACIONES = (
    "cadera", "cuello", "cabeza",
    "hombro_izq", "codo_izq", "muneca_izq",
    "hombro_der", "codo_der", "muneca_der",
    "cadera_izq", "rodilla_izq", "tobillo_izq",
    "cadera_der", "rodilla_der", "tobillo_der",
)

# cada hueso va de una articulacion a la siguiente. El orden importa: un hueso
# aparece despues del que termina donde el empieza, asi el recorrido de arriba
# hacia abajo siempre encuentra al padre ya resuelto.
HUESOS = (
    ("cadera", "cuello"),
    ("cuello", "cabeza"),
    ("cuello", "hombro_izq"),
    ("hombro_izq", "codo_izq"),
    ("codo_izq", "muneca_izq"),
    ("cuello", "hombro_der"),
    ("hombro_der", "codo_der"),
    ("codo_der", "muneca_der"),
    ("cadera", "cadera_izq"),
    ("cadera_izq", "rodilla_izq"),
    ("rodilla_izq", "tobillo_izq"),
    ("cadera", "cadera_der"),
    ("cadera_der", "rodilla_der"),
    ("rodilla_der", "tobillo_der"),
)

GRUPO_DE_HUESO = {
    "izq": ("hombro_izq", "codo_izq", "muneca_izq",
            "cadera_izq", "rodilla_izq", "tobillo_izq"),
    "der": ("hombro_der", "codo_der", "muneca_der",
            "cadera_der", "rodilla_der", "tobillo_der"),
}

EJE_DEL_HUESO = np.array([0.0, 1.0, 0.0])


def nombre_de_hueso(padre, hijo):
    return f"{padre}__{hijo}"


def padre_de_cada_hueso():
    """Para cada hueso, el hueso que termina donde el empieza (o None)."""
    termina_en = {hijo: (padre, hijo) for padre, hijo in HUESOS}
    padres = {}
    for padre, hijo in HUESOS:
        anterior = termina_en.get(padre)
        padres[(padre, hijo)] = anterior
    return padres


def grupo(hueso):
    """A que mitad del cuerpo pertenece un hueso, para pintarlo."""
    _, hijo = hueso
    for nombre, articulaciones in GRUPO_DE_HUESO.items():
        if hijo in articulaciones:
            return nombre
    return "centro"


def orientacion_hacia(direccion):
    """Rotacion que lleva el eje +Y del hueso a la direccion medida.

    Es lo unico que se puede recuperar de una cadena de puntos: el giro del
    hueso sobre su propio eje no mueve ninguna articulacion, asi que no deja
    rastro en los datos y aqui queda en cero.
    """
    largo = np.linalg.norm(direccion)
    if largo < 1e-9:
        return tr.identity()

    destino = direccion / largo
    eje = np.cross(EJE_DEL_HUESO, destino)
    seno = np.linalg.norm(eje)
    coseno = float(np.dot(EJE_DEL_HUESO, destino))

    if seno < 1e-9:
        # paralelos: o ya coincide, o apunta justo al reves
        return tr.identity() if coseno > 0 else tr.rotationZ(np.pi)

    return tr.rotationA(float(np.arctan2(seno, coseno)), eje / seno)


def largos_medidos(posiciones):
    """El largo de cada hueso segun las posiciones de este cuadro."""
    return {
        hueso: float(np.linalg.norm(posiciones[hueso[1]] - posiciones[hueso[0]]))
        for hueso in HUESOS
    }


def transformaciones_globales(posiciones, largos):
    """Arma la pose recorriendo la jerarquia desde la cadera.

    De cada articulacion medida se usa solo la direccion; el largo del hueso lo
    pone `largos`. Con los largos calibrados el esqueleto queda rigido y la pose
    es una aproximacion; con los largos de este cuadro las articulaciones caen
    justo sobre los datos y los huesos se estiran.
    """
    globales = {}
    posicion_de = {"cadera": np.asarray(posiciones["cadera"], dtype=np.float64)}

    for hueso in HUESOS:
        padre, hijo = hueso
        direccion = np.asarray(posiciones[hijo] - posiciones[padre], dtype=np.float64)
        norma = np.linalg.norm(direccion)
        if norma < 1e-9:
            direccion = EJE_DEL_HUESO.copy()
            norma = 1.0

        origen = posicion_de[padre]
        globales[hueso] = tr.translate(*origen) @ orientacion_hacia(direccion)
        posicion_de[hijo] = origen + largos[hueso] * (direccion / norma)

    return globales, posicion_de


def transformaciones_locales(globales):
    """M_n = inv(G_padre) @ G_n para cada hueso; la regla de composicion al reves."""
    padres = padre_de_cada_hueso()
    locales = {}
    for hueso, global_del_hueso in globales.items():
        padre = padres[hueso]
        if padre is None:
            locales[hueso] = global_del_hueso
        else:
            locales[hueso] = np.linalg.inv(globales[padre]) @ global_del_hueso
    return locales
