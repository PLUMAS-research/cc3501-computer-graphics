"""Cubo deformable como reticulo 3D de masas unidas por resortes.

Enfoque Lagrangiano: se discretiza el material en una grilla de n x n x n masas.
Cada masa se une a sus vecinas con resortes de tres tipos, que cumplen roles
distintos para que el cubo se comporte como gelatina y no como un acordeon:

- estructurales: vecinos a lo largo de los ejes (mantienen las distancias).
- de corte (shear): vecinos en diagonal (resisten que el cubo se "tuerza").
- de flexion (bend): vecinos a dos pasos en cada eje (resisten el pandeo).

La integracion es de Verlet con relajacion de restricciones (igual que cloth):
se integran las posiciones y luego se itera sobre los resortes empujando sus
extremos hacia la distancia de reposo. Es estable con resortes rigidos, donde
el Euler explicito explotaria (la misma leccion de masa_resorte).

Todo esta vectorizado con numpy: los resortes son arreglos de indices y la
correccion se reparte con np.add.at.
"""

import numpy as np

GRAVEDAD = np.array([0.0, -9.8, 0.0])


class CuboGelatina:
    def __init__(self, n=5, lado=2.0, centro_y=0.6, piso_y=-1.6, dt=1.0 / 300.0):
        self.n = n
        self.piso_y = piso_y
        self.dt = dt
        self.amortiguacion = 0.012
        self.rigidez = 0.15       # fraccion de la correccion aplicada por iteracion.
        self.iteraciones = 2      # rigidez baja + pocas iteraciones = gelatina blanda.
        self.restitucion = 0.35   # rebote contra el piso.
        self.friccion = 0.86      # roce horizontal en el contacto.

        separacion = lado / (n - 1)
        ejes = np.linspace(-lado / 2, lado / 2, n)
        grilla = np.stack(np.meshgrid(ejes, ejes, ejes, indexing="ij"), axis=-1)
        # cubo centrado en el origen y luego elevado, asi flota sobre el piso y
        # cae al iniciar la simulacion.
        self.reposo = grilla.reshape(-1, 3).astype(np.float64)
        self.reposo[:, 1] += centro_y

        self.posiciones = self.reposo.copy()
        self.previas = self.reposo.copy()

        indice_a, indice_b, descanso = self._construir_resortes(n, separacion)
        self.indice_a = indice_a
        self.indice_b = indice_b
        self.descanso = descanso
        # cuantos resortes toca cada masa: se promedia la correccion por este
        # numero para que sumar muchos resortes no sobre-corrija y diverja.
        self.conteo = np.bincount(np.concatenate([indice_a, indice_b]),
                                  minlength=len(self.posiciones)).astype(np.float64)
        self.conteo[self.conteo == 0] = 1.0

        self.superficie = self._indices_superficie(n)

    def _construir_resortes(self, n, separacion):
        """Genera los pares de resortes (estructurales, de corte y de flexion)."""
        def indice(i, j, k):
            return (i * n + j) * n + k

        # direcciones "hacia adelante" para no duplicar cada resorte.
        estructurales_corte = [d for d in _direcciones_pm1() if d > (0, 0, 0)]
        flexion = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
        direcciones = estructurales_corte + flexion

        a, b, descanso = [], [], []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for di, dj, dk in direcciones:
                        ni, nj, nk = i + di, j + dj, k + dk
                        # el limite inferior importa: un indice negativo se
                        # envolveria por la formula y conectaria masas opuestas.
                        if 0 <= ni < n and 0 <= nj < n and 0 <= nk < n:
                            a.append(indice(i, j, k))
                            b.append(indice(ni, nj, nk))
                            descanso.append(separacion * np.sqrt(di * di + dj * dj + dk * dk))
        return (np.array(a, dtype=np.int64),
                np.array(b, dtype=np.int64),
                np.array(descanso, dtype=np.float64))

    def _indices_superficie(self, n):
        """Triangulos de las seis caras externas del reticulo (cascara)."""
        def indice(i, j, k):
            return (i * n + j) * n + k

        triangulos = []
        # cada cara fija un eje en 0 o n-1 y teselа la grilla (n-1)x(n-1).
        caras = [
            (lambda u, v: indice(0, u, v)),
            (lambda u, v: indice(n - 1, u, v)),
            (lambda u, v: indice(u, 0, v)),
            (lambda u, v: indice(u, n - 1, v)),
            (lambda u, v: indice(u, v, 0)),
            (lambda u, v: indice(u, v, n - 1)),
        ]
        for cara in caras:
            for u in range(n - 1):
                for v in range(n - 1):
                    a, b = cara(u, v), cara(u + 1, v)
                    c, d = cara(u + 1, v + 1), cara(u, v + 1)
                    triangulos.append((a, b, c))
                    triangulos.append((a, c, d))
        # la orientacion de los triangulos no importa: las normales se reorientan
        # hacia afuera por geometria al calcularlas (ver normales_superficie).
        return np.array(triangulos, dtype=np.uint32)

    def paso(self, dt):
        """Un paso de simulacion: Verlet + restricciones + colision con el piso."""
        velocidad = (self.posiciones - self.previas) * (1.0 - self.amortiguacion)
        nuevas = self.posiciones + velocidad + GRAVEDAD * dt * dt
        self.previas = self.posiciones.copy()
        self.posiciones = nuevas
        for _ in range(self.iteraciones):
            self._satisfacer_resortes()
        self._colision_piso()

    def _satisfacer_resortes(self):
        delta = self.posiciones[self.indice_b] - self.posiciones[self.indice_a]
        largo = np.linalg.norm(delta, axis=1)
        largo = np.where(largo < 1e-9, 1e-9, largo)
        factor = (self.rigidez * (largo - self.descanso) / largo)[:, None]
        correccion = delta * factor
        desplazamiento = np.zeros_like(self.posiciones)
        np.add.at(desplazamiento, self.indice_a, correccion)
        np.add.at(desplazamiento, self.indice_b, -correccion)
        self.posiciones += desplazamiento / self.conteo[:, None]

    def _colision_piso(self):
        bajo = self.posiciones[:, 1] < self.piso_y
        if not np.any(bajo):
            return
        velocidad = self.posiciones[bajo] - self.previas[bajo]
        self.posiciones[bajo, 1] = self.piso_y
        # se invierte la velocidad vertical (rebote) y se frena la horizontal (roce).
        self.previas[bajo, 1] = self.piso_y + velocidad[:, 1] * self.restitucion
        self.previas[bajo, 0] = self.posiciones[bajo, 0] - velocidad[:, 0] * self.friccion
        self.previas[bajo, 2] = self.posiciones[bajo, 2] - velocidad[:, 2] * self.friccion

    def lanzar(self, impulso_vertical=6.0):
        """Empuja el cubo hacia arriba con un giro aleatorio para verlo temblar."""
        centro = self.posiciones.mean(axis=0)
        giro = np.random.uniform(-3.0, 3.0, size=3)
        velocidad = np.cross(giro, self.posiciones - centro)
        velocidad += np.array([0.0, impulso_vertical, 0.0])
        self.previas = self.posiciones - velocidad * self.dt

    def reiniciar(self):
        self.posiciones = self.reposo.copy()
        self.previas = self.reposo.copy()

    def normales_superficie(self):
        """Normales por vertice de la cascara, reorientadas hacia afuera.

        Acumula la normal de cada triangulo en sus vertices y luego voltea cada
        normal para que apunte lejos del centro del cubo. Asi la orientacion no
        depende del sentido de giro de los triangulos.
        """
        tri = self.superficie
        p0 = self.posiciones[tri[:, 0]]
        p1 = self.posiciones[tri[:, 1]]
        p2 = self.posiciones[tri[:, 2]]
        normal_cara = np.cross(p1 - p0, p2 - p0)

        normales = np.zeros_like(self.posiciones)
        np.add.at(normales, tri[:, 0], normal_cara)
        np.add.at(normales, tri[:, 1], normal_cara)
        np.add.at(normales, tri[:, 2], normal_cara)

        centro = self.posiciones.mean(axis=0)
        hacia_afuera = self.posiciones - centro
        signo = np.sign(np.sum(normales * hacia_afuera, axis=1))
        signo[signo == 0] = 1.0
        normales *= signo[:, None]

        largo = np.linalg.norm(normales, axis=1, keepdims=True)
        largo[largo < 1e-9] = 1.0
        return (normales / largo).astype(np.float32)


def _direcciones_pm1():
    """Las 26 direcciones de un cubo 3x3x3 sin el centro."""
    return [(i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)]
