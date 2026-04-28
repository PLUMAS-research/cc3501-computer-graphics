import numpy as np
from scipy import spatial


PALETA_PECES = np.array(
    [
        [1.00, 0.55, 0.25],
        [1.00, 0.85, 0.30],
        [0.95, 0.35, 0.40],
        [0.40, 0.75, 0.95],
        [0.55, 0.85, 0.45],
        [1.00, 0.70, 0.85],
        [0.85, 0.55, 0.95],
    ],
    dtype=np.float32,
)


class Pecera:
    """Simulación de boids 3D dentro de una caja.

    Cada paso construye un KD-tree con las posiciones de los peces para encontrar
    vecinos. Sobre cada pez se aplican las tres reglas clásicas de boids más una
    fuerza que lo aleja de las paredes de la caja.
    """

    def __init__(
        self,
        n_peces,
        bounds_min,
        bounds_max,
        speed=0.018,
        vision=0.55,
        separation_distance=0.18,
        cohere_factor=0.005,
        separation_factor=0.06,
        match_factor=0.04,
        border_margin=0.30,
        border_factor=0.0015,
        seed=42,
    ):
        rng = np.random.default_rng(seed)

        self.bounds_min = np.asarray(bounds_min, dtype=np.float32)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float32)
        size = self.bounds_max - self.bounds_min
        margen = size * 0.1

        self.positions = (
            self.bounds_min
            + margen
            + rng.random((n_peces, 3)).astype(np.float32) * (size - 2 * margen)
        )
        random_dirs = rng.random((n_peces, 3)).astype(np.float32) * 2.0 - 1.0
        random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
        self.velocities = random_dirs * speed

        self.colors = PALETA_PECES[rng.integers(0, len(PALETA_PECES), size=n_peces)]

        self.n = n_peces
        self.speed = speed
        self.vision = vision
        self.separation_distance = separation_distance
        self.cohere_factor = cohere_factor
        self.separation_factor = separation_factor
        self.match_factor = match_factor
        self.border_margin = border_margin
        self.border_factor = border_factor

    def step(self):
        tree = spatial.cKDTree(self.positions)
        nuevas_velocidades = self.velocities.copy()

        for i in range(self.n):
            indices = tree.query_ball_point(self.positions[i], self.vision)
            indices = [j for j in indices if j != i]

            if indices:
                pos_vecinos = self.positions[indices]
                vel_vecinos = self.velocities[indices]

                cohesion = pos_vecinos.mean(axis=0) - self.positions[i]
                alineamiento = vel_vecinos.mean(axis=0)

                deltas = self.positions[i] - pos_vecinos
                distancias = np.linalg.norm(deltas, axis=1)
                cercanos = distancias < self.separation_distance
                if cercanos.any():
                    deltas_cerca = deltas[cercanos]
                    distancias_cerca = np.maximum(
                        distancias[cercanos].reshape(-1, 1), 1e-5
                    )
                    separacion = (deltas_cerca / distancias_cerca).sum(axis=0)
                else:
                    separacion = np.zeros(3, dtype=np.float32)

                nuevas_velocidades[i] += (
                    cohesion * self.cohere_factor
                    + alineamiento * self.match_factor
                    + separacion * self.separation_factor
                )

            fuerza_borde = np.zeros(3, dtype=np.float32)
            for eje in range(3):
                bajo = self.bounds_min[eje] + self.border_margin
                alto = self.bounds_max[eje] - self.border_margin
                if self.positions[i, eje] < bajo:
                    fuerza_borde[eje] += self.border_factor
                elif self.positions[i, eje] > alto:
                    fuerza_borde[eje] -= self.border_factor
            nuevas_velocidades[i] += fuerza_borde

        normas = np.maximum(
            np.linalg.norm(nuevas_velocidades, axis=1, keepdims=True), 1e-8
        )
        self.velocities = (nuevas_velocidades / normas) * self.speed
        self.positions = self.positions + self.velocities

        for eje in range(3):
            debajo = self.positions[:, eje] < self.bounds_min[eje]
            arriba = self.positions[:, eje] > self.bounds_max[eje]
            self.positions[debajo, eje] = self.bounds_min[eje]
            self.positions[arriba, eje] = self.bounds_max[eje]
            self.velocities[debajo, eje] = abs(self.velocities[debajo, eje])
            self.velocities[arriba, eje] = -abs(self.velocities[arriba, eje])

    def fish_triangles(self, size=0.07):
        """Triángulos 3D que apuntan en la dirección de la velocidad.

        Devuelve dos arreglos planos listos para subir como atributos position y
        color a un vertex_list de pyglet.
        """
        forwards = self.velocities / np.maximum(
            np.linalg.norm(self.velocities, axis=1, keepdims=True), 1e-8
        )
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        rights = np.cross(forwards, world_up)
        right_norms = np.linalg.norm(rights, axis=1, keepdims=True)
        fallback_right = np.tile(
            np.array([1.0, 0.0, 0.0], dtype=np.float32), (self.n, 1)
        )
        rights = np.where(
            right_norms < 1e-4,
            fallback_right,
            rights / np.maximum(right_norms, 1e-8),
        )

        nariz = self.positions + forwards * size
        cola_izq = self.positions - forwards * (size * 0.5) + rights * (size * 0.5)
        cola_der = self.positions - forwards * (size * 0.5) - rights * (size * 0.5)

        triangulos = np.empty((self.n * 3, 3), dtype=np.float32)
        triangulos[0::3] = nariz
        triangulos[1::3] = cola_izq
        triangulos[2::3] = cola_der

        colors = np.repeat(self.colors, 3, axis=0)
        return triangulos.flatten(), colors.flatten()
