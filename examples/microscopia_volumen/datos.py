"""Lectura de un hyperstack de ImageJ para el visor de microscopia.

ImageJ guarda un hyperstack como un TIFF con una pagina por imagen 2D y los ejes
en orden TZCYX: tiempo, profundidad, canal, filas y columnas. tifffile lo abre sin
copiarlo a memoria (memmap), porque ImageJ escribe los pixeles de forma contigua.

Un volumen de microscopia suele ser grande: el de la clase pesa 583 MB por tiempo,
demasiado para subirlo entero a la GPU de un computador cualquiera. Por eso cada
tiempo se reduce promediando bloques de voxeles, y el resultado se guarda en
cache/ para que la siguiente ejecucion lo lea directo.

El modulo no depende de OpenGL, asi que se puede probar sin abrir una ventana.
"""

from pathlib import Path

import numpy as np

CARPETA_CACHE = Path(__file__).resolve().parents[2] / "cache" / "microscopia_volumen"
CANALES = 2


def reducir(volumen, factor):
    """Promedia bloques de factor x factor x factor voxeles de un arreglo (Z, Y, X).

    Promediar antes de bajar la resolucion es un filtro: tomar un voxel de cada
    bloque produciria aliasing, igual que muestrear un rayo con un paso largo.
    """
    if factor == 1:
        return np.asarray(volumen)
    z, y, x = (lado // factor * factor for lado in volumen.shape)
    bloques = np.asarray(volumen[:z, :y, :x]).reshape(
        z // factor, factor, y // factor, factor, x // factor, factor)
    return bloques.mean(axis=(1, 3, 5), dtype=np.float32).round().astype(np.uint8)


def cargar_hiperstack(ruta, factor=2):
    """Devuelve una lista con un arreglo (Z, Y, X, 2) uint8 por tiempo.

    El eje Y queda invertido respecto del archivo: en una imagen la fila 0 es la
    de arriba, mientras que en la textura 3D la coordenada t crece hacia arriba.
    """
    import tifffile

    ruta = Path(ruta)
    marca = ruta.stat()
    archivo_cache = CARPETA_CACHE / f"{ruta.stem}-{marca.st_size}-{int(marca.st_mtime)}-r{factor}.npz"
    if archivo_cache.exists():
        print(f"[microscopia_volumen] leyendo la version reducida de {archivo_cache}")
        with np.load(archivo_cache) as guardado:
            return [guardado[nombre] for nombre in sorted(guardado.files)]

    hiperstack = tifffile.memmap(ruta)
    if hiperstack.ndim != 5 or hiperstack.shape[2] != CANALES:
        raise ValueError(f"se esperaba un hyperstack TZCYX con {CANALES} canales, "
                         f"y el archivo tiene forma {hiperstack.shape}")
    print(f"[microscopia_volumen] {ruta.name}: {hiperstack.shape} (TZCYX), "
          f"{hiperstack.nbytes / 1e9:.2f} GB; reduciendo por {factor} en cada eje")

    tiempos = []
    for indice_tiempo in range(hiperstack.shape[0]):
        canales = [reducir(hiperstack[indice_tiempo, :, canal], factor) for canal in range(CANALES)]
        volumen = np.stack(canales, axis=-1)[:, ::-1]
        tiempos.append(np.ascontiguousarray(volumen))
        print(f"[microscopia_volumen] tiempo {indice_tiempo}: {volumen.shape}")

    CARPETA_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(archivo_cache, **{f"tiempo_{i:03d}": v for i, v in enumerate(tiempos)})
    print(f"[microscopia_volumen] version reducida guardada en {archivo_cache}")
    return tiempos


def ventana_por_canal(volumen, percentil_inferior=95.0, percentil_superior=99.9):
    """Intensidades minima y maxima de cada canal, en [0, 1].

    Una tomografia tiene una escala fija (las HU); la fluorescencia no, porque
    depende del marcador, del laser y de la ganancia del detector. Por eso la
    ventana de cada canal sale de los datos. El minimo es un percentil de todos
    los voxeles, y deja fuera el ruido de fondo, que de otro modo se acumula en
    cada rayo como una niebla. El maximo es un percentil alto de los voxeles con
    senal, para que unos pocos voxeles saturados no oscurezcan el resto.
    """
    minimos, maximos = [], []
    for canal in range(volumen.shape[-1]):
        valores = volumen[..., canal]
        con_senal = valores[valores > 0]
        minimo = float(np.percentile(valores, percentil_inferior))
        maximo = float(np.percentile(con_senal, percentil_superior)) if con_senal.size else 255.0
        minimos.append(minimo / 255.0)
        maximos.append(max(maximo, minimo + 1.0) / 255.0)
    return tuple(minimos), tuple(maximos)


def ajustar_esfera(volumen, minimos, maximos, box_half, umbral=0.5, maximo_puntos=200_000):
    """Centro y radio de la esfera que mejor pasa por los voxeles brillantes.

    Las celulas de la capa exterior del embrion forman una cascara esferica. Un
    punto p esta en la esfera de centro c y radio r si |p|^2 = 2 c.p + (r^2 - |c|^2),
    que es lineal en c y en d = r^2 - |c|^2. Con todos los voxeles brillantes eso es
    un sistema sobredeterminado, que se resuelve por minimos cuadrados.

    Las posiciones van en las coordenadas de la caja [-box_half, box_half], las
    mismas del shader. Se usan los voxeles con brillo sobre `umbral` en algun canal.
    """
    minimos, maximos = np.float32(minimos), np.float32(maximos)
    brillo = (volumen.astype(np.float32) / 255.0 - minimos) / (maximos - minimos)
    profundidad_idx, alto_idx, ancho_idx = np.nonzero((brillo > umbral).any(axis=-1))
    indices = np.stack([ancho_idx, alto_idx, profundidad_idx], axis=1).astype(np.float64)
    if len(indices) > maximo_puntos:
        elegidos = np.random.default_rng(0).choice(len(indices), maximo_puntos, replace=False)
        indices = indices[elegidos]

    tamano = np.array(volumen.shape[2::-1], dtype=np.float64)  # (ancho, alto, profundidad)
    puntos = (2.0 * (indices + 0.5) / tamano - 1.0) * box_half
    sistema = np.c_[2.0 * puntos, np.ones(len(puntos))]
    solucion, *_ = np.linalg.lstsq(sistema, (puntos ** 2).sum(axis=1), rcond=None)
    centro = solucion[:3]
    radio = float(np.sqrt(solucion[3] + centro @ centro))
    return centro.astype(np.float32), radio
