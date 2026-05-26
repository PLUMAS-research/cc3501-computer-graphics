"""Cadena masa-resorte 1D (una hebra colgante) integrada a mano.

La hebra es una serie de N masas puntuales unidas por resortes lineales.
El nodo 0 esta anclado. Cada resorte aplica la fuerza de Hooke

    F_ij = -k (|x_i - x_j| - L0) * (x_i - x_j) / |x_i - x_j|

mas una amortiguacion a lo largo del resorte

    F_damp = -c ((v_i - v_j) . n) n,   n = (x_i - x_j) / |x_i - x_j|

y la gravedad m*g sobre cada masa libre.

El punto del ejemplo es comparar tres integradores explicitos sobre el mismo
sistema. Con resortes rigidos (k alto) el Euler explicito gana energia y
explota, mientras que el semi-implicito y Verlet se mantienen acotados.
"""

import numpy as np

METODOS = ("euler_explicito", "euler_semi_implicito", "verlet")


def _fuerzas(positions, velocities, rest_lengths, k, c, masa, gravedad):
    """Suma de fuerzas sobre cada nodo: resortes + amortiguacion + gravedad.

    positions, velocities: arreglos (N, 2).
    rest_lengths: arreglo (N-1,) con la longitud de reposo L0 de cada resorte.
    Devuelve un arreglo (N, 2) de fuerzas.
    """
    fuerzas = np.zeros_like(positions)

    # gravedad: actua hacia abajo (-y) sobre todas las masas.
    fuerzas[:, 1] -= masa * gravedad

    # cada resorte conecta el nodo i con el i+1.
    delta = positions[1:] - positions[:-1]
    distancias = np.linalg.norm(delta, axis=1)
    # evitamos dividir por cero si dos nodos coinciden.
    distancias = np.maximum(distancias, 1e-6)
    normales = delta / distancias[:, None]

    estiramiento = distancias - rest_lengths
    fuerza_resorte = (k * estiramiento)[:, None] * normales

    velocidad_relativa = velocities[1:] - velocities[:-1]
    rapidez_a_lo_largo = np.sum(velocidad_relativa * normales, axis=1)
    fuerza_damp = (c * rapidez_a_lo_largo)[:, None] * normales

    fuerza_total_resorte = fuerza_resorte + fuerza_damp
    # el nodo i (extremo bajo del resorte) recibe +fuerza, el i+1 recibe -fuerza.
    np.add.at(fuerzas, np.arange(len(rest_lengths)), fuerza_total_resorte)
    np.add.at(fuerzas, np.arange(len(rest_lengths)) + 1, -fuerza_total_resorte)

    return fuerzas


class HebraMasaResorte:
    """Estado de la hebra y su avance temporal.

    El indice 0 esta siempre anclado. Ademas se puede fijar otro nodo
    (el que arrastra el mouse) pasandolo en `fijos` al llamar a `paso`.
    """

    def __init__(self, n_nodos, ancla, separacion, masa=1.0, gravedad=400.0):
        self.n_nodos = n_nodos
        self.ancla = np.array(ancla, dtype=np.float64)
        self.separacion = separacion
        self.masa = masa
        self.gravedad = gravedad
        self.rest_lengths = np.full(n_nodos - 1, separacion, dtype=np.float64)
        self.reset()

    def reset(self):
        """Coloca la hebra colgando recta desde el ancla, en reposo."""
        offsets = np.arange(self.n_nodos) * self.separacion
        self.positions = np.zeros((self.n_nodos, 2), dtype=np.float64)
        self.positions[:, 0] = self.ancla[0]
        self.positions[:, 1] = self.ancla[1] - offsets
        self.velocities = np.zeros_like(self.positions)
        # Verlet necesita la posicion anterior; en reposo es igual a la actual.
        self.previous_positions = self.positions.copy()
        self.exploto = False

    def energia_cinetica(self):
        return 0.5 * self.masa * float(np.sum(self.velocities**2))

    def estiramiento_relativo(self):
        """(|x_i - x_{i+1}| - L0) / L0 por resorte. Util para colorear."""
        delta = self.positions[1:] - self.positions[:-1]
        distancias = np.linalg.norm(delta, axis=1)
        return (distancias - self.rest_lengths) / self.rest_lengths

    def paso(self, dt, metodo, k, c, substeps, fijos=()):
        """Avanza dt segundos con el integrador elegido.

        `substeps` divide dt en pasos pequenos dt/substeps; subir este valor
        relaja la condicion de estabilidad dt < 2/omega, omega = sqrt(k/m).
        `fijos` es una lista de indices que permanecen quietos este frame.
        """
        if self.exploto:
            return

        indices_fijos = np.array((0,) + tuple(fijos), dtype=int)
        posiciones_fijas = self.positions[indices_fijos].copy()
        sub_dt = dt / substeps

        for _ in range(substeps):
            fuerzas = _fuerzas(
                self.positions,
                self.velocities,
                self.rest_lengths,
                k,
                c,
                self.masa,
                self.gravedad,
            )
            aceleracion = fuerzas / self.masa

            if metodo == "euler_explicito":
                # la posicion usa la velocidad VIEJA (origen de la inestabilidad).
                nuevas_pos = self.positions + sub_dt * self.velocities
                nuevas_vel = self.velocities + sub_dt * aceleracion
                self.previous_positions = self.positions
                self.positions = nuevas_pos
                self.velocities = nuevas_vel

            elif metodo == "euler_semi_implicito":
                # la posicion usa la velocidad NUEVA (Symplectic Euler).
                self.velocities = self.velocities + sub_dt * aceleracion
                self.previous_positions = self.positions
                self.positions = self.positions + sub_dt * self.velocities

            elif metodo == "verlet":
                # Verlet de posicion: x' = 2x - x_prev + a dt^2.
                nuevas_pos = (
                    2.0 * self.positions
                    - self.previous_positions
                    + aceleracion * sub_dt * sub_dt
                )
                self.previous_positions = self.positions
                self.positions = nuevas_pos
                # velocidad estimada para la amortiguacion del siguiente substep.
                self.velocities = (self.positions - self.previous_positions) / sub_dt

            else:
                raise ValueError(f"metodo desconocido: {metodo}")

            # reimponemos los nodos fijos: vuelven a su posicion y quedan quietos.
            self.positions[indices_fijos] = posiciones_fijas
            self.velocities[indices_fijos] = 0.0
            self.previous_positions[indices_fijos] = posiciones_fijas

        # deteccion de inestabilidad: posiciones no finitas o disparadas.
        if not np.all(np.isfinite(self.positions)) or np.max(
            np.abs(self.positions)
        ) > 1e6:
            self.exploto = True
