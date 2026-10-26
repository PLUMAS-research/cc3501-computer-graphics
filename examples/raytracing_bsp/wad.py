"""Lectura de un mapa de Doom desde un archivo WAD.

Un WAD es una lista de bloques ("lumps") con nombre. Cada mapa es un bloque
vacío con su nombre (E1M1) seguido de sus datos: THINGS (objetos, entre ellos
el jugador), LINEDEFS (las líneas del mapa), SIDEDEFS (el lado de cada línea y
su sector), VERTEXES (los extremos de las líneas) y SECTORS (las regiones del
piso, cada una con la altura de su piso y de su techo). El resto (SEGS,
SSECTORS, NODES) es el BSP que Doom calculó al armar el mapa.

Doom II y Freedoom guardan sus mapas en el mismo formato, así que el ejemplo
también los lee; en Doom II los mapas se llaman MAP01 a MAP32. Si no se le
indica un WAD, usa el de la versión shareware de Doom 1.9, que trae los nueve
mapas del primer episodio.
"""

import struct
from pathlib import Path

import numpy as np

SIN_LADO = 0xFFFF


def cargar_mapa(ruta, nombre="E1M1"):
    """Las líneas, los sectores y el inicio del jugador de un mapa.

    Devuelve un dict con:
    - vertices (V, 2) y lineas (L, 2), con los índices de los extremos de cada
      línea. Su lado derecho, yendo del primer extremo al segundo, es el frente.
    - sector_derecho y sector_izquierdo (L,): el sector a cada lado, o -1 si la
      línea tiene un solo lado. Una línea de un solo lado es una pared sólida;
      una de dos lados separa dos sectores y se ve a través de ella.
    - especial (L,): el número de la acción de cada línea, o 0 si no tiene; la
      1, por ejemplo, abre la puerta que está detrás de la línea.
    - piso, techo y luz (S,) de cada sector, con la luz de 0 a 255, y cielo (S,),
      si su techo es el cielo.
    - jugador: posición (x, y) y ángulo en radianes.
    """
    datos = Path(ruta).read_bytes()
    cantidad, inicio_directorio = struct.unpack_from("<ii", datos, 4)
    directorio = [struct.unpack_from("<ii8s", datos, inicio_directorio + 16 * i) for i in range(cantidad)]
    nombres = [entrada[2].rstrip(b"\0").decode("ascii") for entrada in directorio]
    mapas = [nombres[i] for i in range(len(nombres) - 1) if nombres[i + 1] == "THINGS"]
    if nombre not in mapas:
        raise ValueError(f"{ruta} no tiene el mapa {nombre}; tiene {', '.join(mapas)}")
    inicio_mapa = nombres.index(nombre)
    bloques = {nombres[i]: directorio[i] for i in range(inicio_mapa + 1, inicio_mapa + 11)}

    def leer(bloque, formato):
        posicion, tamano, _ = bloques[bloque]
        return list(struct.iter_unpack(formato, datos[posicion:posicion + tamano]))

    vertices = np.array(leer("VERTEXES", "<hh"), dtype=np.float64)
    lineas = leer("LINEDEFS", "<HHhhhHH")
    lados = leer("SIDEDEFS", "<hh8s8s8sH")
    sectores = leer("SECTORS", "<hh8s8shhh")
    jugador = next(cosa for cosa in leer("THINGS", "<hhhhh") if cosa[3] == 1)

    def sector(lado):
        return -1 if lado == SIN_LADO else lados[lado][5]

    return {
        "vertices": vertices,
        "lineas": np.array([(linea[0], linea[1]) for linea in lineas]),
        "sector_derecho": np.array([sector(linea[5]) for linea in lineas]),
        "sector_izquierdo": np.array([sector(linea[6]) for linea in lineas]),
        "especial": np.array([linea[3] for linea in lineas]),
        "piso": np.array([s[0] for s in sectores], dtype=np.float64),
        "techo": np.array([s[1] for s in sectores], dtype=np.float64),
        "luz": np.array([s[4] for s in sectores]),
        "cielo": np.array([s[3].rstrip(b"\0") == b"F_SKY1" for s in sectores]),
        "jugador": (np.array(jugador[:2], dtype=np.float64), np.radians(jugador[2])),
    }
