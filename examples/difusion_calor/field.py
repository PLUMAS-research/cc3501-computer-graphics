"""Difusion de calor 2D sobre una grilla fija (enfoque euleriano).

La grilla no se mueve: el campo escalar T (temperatura) evoluciona sobre ella
segun la ecuacion del calor

    dT/dt = alpha * laplaciano(T).

Discretizamos el laplaciano con el stencil de 5 puntos y avanzamos en el tiempo
con Euler explicito:

    T_new[i,j] = T[i,j] + r * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4 T[i,j]),

donde r = alpha * dt / dx^2 es el numero de difusion (usamos dx = 1).

El metodo explicito es estable solo si r <= 1/4 en 2D (condicion CFL). Si se
sube alpha o dt por encima de ese limite, el campo desarrolla oscilaciones tipo
tablero de ajedrez que crecen sin control. Subdividir el paso temporal baja r y
recupera la estabilidad. Es el mismo fenomeno que en el ejemplo masa_resorte,
pero discretizando el espacio en vez del material.
"""

import numpy as np

BORDES = ("dirichlet", "neumann")
CFL_MAXIMO = 0.25  # limite de estabilidad del metodo explicito en 2D.


class CampoCalor:
    """Campo de temperatura T sobre una grilla n x n."""

    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.T = np.zeros((self.n, self.n), dtype=np.float32)
        self.exploto = False

    def numero_difusion(self, alpha, dt, substeps):
        """r por substep. dx = 1, asi que r = alpha * (dt / substeps)."""
        return alpha * (dt / substeps)

    def inyectar(self, columna, fila, radio, valor=1.0):
        """Fija a `valor` un disco de la grilla (pincel de calor)."""
        ys, xs = np.ogrid[: self.n, : self.n]
        mascara = (xs - columna) ** 2 + (ys - fila) ** 2 <= radio**2
        self.T[mascara] = valor

    def paso(self, dt, alpha, substeps, borde):
        """Avanza dt segundos resolviendo la ecuacion del calor.

        `borde` dirichlet: los bordes quedan fijos en 0 (el calor escapa).
        `borde` neumann: bordes aislados, flujo cero (el calor se conserva).
        """
        if self.exploto:
            return

        numero_difusion = self.numero_difusion(alpha, dt, substeps)
        # mode='edge' replica el borde: el vecino exterior es igual al del borde,
        # asi el gradiente normal es cero (Neumann). Para Dirichlet rellenamos 0.
        modo_pad = "edge" if borde == "neumann" else "constant"

        for _ in range(substeps):
            padded = np.pad(self.T, 1, mode=modo_pad)
            laplaciano = (
                padded[2:, 1:-1]
                + padded[:-2, 1:-1]
                + padded[1:-1, 2:]
                + padded[1:-1, :-2]
                - 4.0 * self.T
            )
            self.T = self.T + numero_difusion * laplaciano

            if borde == "dirichlet":
                # el valor del borde permanece fijo en cero.
                self.T[0, :] = 0.0
                self.T[-1, :] = 0.0
                self.T[:, 0] = 0.0
                self.T[:, -1] = 0.0

        if not np.all(np.isfinite(self.T)) or np.max(np.abs(self.T)) > 1e3:
            self.exploto = True
