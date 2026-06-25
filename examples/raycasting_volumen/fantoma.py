"""Fantoma de Shepp-Logan en 3D, con valores en unidades Hounsfield.

El fantoma de Shepp-Logan es el volumen de prueba canonico para tomografia: un
conjunto de elipsoides que imitan un corte de cabeza (craneo, cerebro,
ventriculos, masas). Aqui se construye la version 3D pintando cada elipsoide
sobre el anterior, con valores en unidades Hounsfield (HU) para que la funcion
de transferencia del ejemplo distinga aire, tejido blando y hueso.

El modulo no depende de OpenGL: genera un arreglo de numpy y se puede inspeccionar
o probar sin abrir una ventana.
"""

import numpy as np

# Cada elipsoide: valor en HU, semiejes (a, b, c), centro (x0, y0, z0) y un
# angulo de rotacion phi (grados) en el plano xy. La geometria sigue al fantoma
# de Shepp-Logan; los valores en HU se eligieron para imitar una tomografia:
# aire -1000, tejido blando ~40, hueso cortical ~1000.
ELIPSOIDES = [
    # hu,     a,    b,    c,    x0,    y0,    z0,    phi
    (1000.0, 0.69, 0.90, 0.90, 0.00,  0.000, 0.00,   0.0),  # craneo (hueso)
    (  40.0, 0.66, 0.86, 0.86, 0.00, -0.018, 0.00,   0.0),  # cerebro (tejido blando)
    ( 220.0, 0.11, 0.31, 0.22, 0.22,  0.000, 0.00, -18.0),  # masa lateral derecha
    ( 220.0, 0.16, 0.41, 0.28,-0.22,  0.000, 0.00,  18.0),  # masa lateral izquierda
    ( 320.0, 0.21, 0.25, 0.30, 0.00,  0.350,-0.15,   0.0),  # estructura densa
    (-150.0, 0.046,0.046,0.046,0.00,  0.100, 0.25,   0.0),  # ventriculo (oscuro)
    (-150.0, 0.046,0.046,0.046,0.00, -0.100, 0.25,   0.0),  # ventriculo (oscuro)
    ( 600.0, 0.046,0.023,0.050,-0.08,-0.605, 0.00,   0.0),  # nodulo denso
]


def generar_fantoma(resolucion=128):
    """Construye el volumen (resolucion^3) pintando los elipsoides en orden.

    Devuelve un arreglo float32 de forma (D, H, W) con valores en HU. El fondo es
    aire (-1000 HU). Cada elipsoide sobrescribe al anterior dentro de su region,
    asi el craneo grande queda con un reborde de hueso y el interior de tejido.
    """
    n = resolucion
    eje = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    # indexing="ij": el primer eje (profundidad) es z, el segundo y, el tercero x.
    z, y, x = np.meshgrid(eje, eje, eje, indexing="ij")

    volumen = np.full((n, n, n), -1000.0, dtype=np.float32)
    for hu, a, b, c, x0, y0, z0, phi in ELIPSOIDES:
        radianes = np.radians(phi)
        cos_phi, sin_phi = np.cos(radianes), np.sin(radianes)
        x_rotado = (x - x0) * cos_phi + (y - y0) * sin_phi
        y_rotado = -(x - x0) * sin_phi + (y - y0) * cos_phi
        z_centrado = z - z0
        dentro = (
            (x_rotado / a) ** 2 + (y_rotado / b) ** 2 + (z_centrado / c) ** 2
        ) <= 1.0
        volumen[dentro] = hu

    return volumen


if __name__ == "__main__":
    volumen = generar_fantoma(128)
    print(f"forma: {volumen.shape}")
    print(f"rango HU: [{volumen.min():.0f}, {volumen.max():.0f}]")
    valores, conteos = np.unique(volumen, return_counts=True)
    for valor, conteo in zip(valores, conteos):
        print(f"  HU={valor:8.0f}  voxeles={conteo}")
