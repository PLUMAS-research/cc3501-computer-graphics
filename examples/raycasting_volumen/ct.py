"""Carga tomografias computarizadas reales y las entrega en unidades Hounsfield.

Las tomografias vienen de pyvista (se descargan una vez y quedan cacheadas en
disco). Cada dataset guarda los valores en su propia escala; aqui se restan al
offset que deja el aire cerca de -1000 HU y el agua cerca de 0 HU, asi la misma
funcion de transferencia del fantoma (aire / tejido / hueso) clasifica bien.
"""

import numpy as np

# Por dataset: funcion de descarga de pyvista y offset que lleva los valores
# almacenados a HU (HU = valor - offset). full_head guarda HU + 1024, asi que el
# aire de fondo (0) queda en -1024 y el peak de tejido blando (~1024) en ~0.
DATASETS = {
    "full_head": ("download_full_head", 1024.0),
}


def cargar_ct(nombre="full_head"):
    """Devuelve (volumen_hu, medios_lados).

    volumen_hu es float32 de forma (D, H, W) en unidades Hounsfield. medios_lados
    son los medios lados de la caja del volumen, normalizados para que el eje mas
    largo valga 1.0; preservan las proporciones fisicas del estudio (el espaciado
    entre cortes suele ser mayor que el espaciado dentro de un corte).
    """
    from pyvista import examples

    nombre_descarga, offset = DATASETS[nombre]
    grid = getattr(examples, nombre_descarga)()

    dimensiones = np.array(grid.dimensions)            # (nx, ny, nz)
    espaciado = np.array(grid.spacing)
    valores = np.asarray(grid.active_scalars).astype(np.float32)
    volumen = valores.reshape(dimensiones[::-1]) - offset  # (z, y, x) crudo, en HU

    # Reorienta a la convencion de mundo del visor: world x = izquierda-derecha,
    # world y = superior-inferior (arriba = craneo), world z = anterior-posterior.
    # En full_head el eje de cortes (axis0, z) es superior-inferior con indice 0 en
    # el craneo; axis1 (y) es anterior-posterior; axis2 (x) izquierda-derecha. Se
    # lleva el eje superior-inferior a la posicion de world y (axis1 de la textura,
    # que el shader mapea a la vertical) y se invierte para que el craneo quede
    # arriba. Sin esto la cabeza sale acostada y de cabeza.
    volumen = np.transpose(volumen, (1, 0, 2))[:, ::-1, :]  # -> (AP, SI, LR)

    extension = dimensiones * espaciado                 # (LR, AP, SI)
    extension_mundo = np.array([extension[0], extension[2], extension[1]])  # (LR, SI, AP)
    medios_lados = (extension_mundo / extension_mundo.max()).astype(np.float32)
    return volumen, medios_lados
