import numpy as np


# Clase Particle genérica que solo maneja las propiedades físicas básicas
class Particle(object):
    def __init__(self, position, velocity=None, acceleration=None, mass=1.0, ttl=1.0):
        """
        Inicializa una partícula con sus propiedades físicas básicas.

        Args:
            position (array-like): Posición inicial [x, y]
            velocity (array-like, optional): Velocidad inicial [vx, vy]
            acceleration (array-like, optional): Aceleración inicial [ax, ay]
            mass (float, optional): Masa de la partícula
            ttl (float, optional): Tiempo de vida inicial ("time to live")
        """
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity if velocity is not None else [0, 0], dtype=np.float32)
        self.acceleration = np.array(acceleration if acceleration is not None else [0, 0], dtype=np.float32)
        self.mass = mass
        self.ttl = ttl
        # Propiedades adicionales que pueden ser útiles
        self.age = 0.0  # Edad de la partícula
        self.alive = True  # Estado de la partícula

    def apply_force(self, force):
        """Aplica una fuerza a la partícula, afectando su aceleración."""
        self.acceleration += force / self.mass

    def reset_acceleration(self):
        """Reinicia la aceleración de la partícula a cero."""
        self.acceleration[:] = 0

    def update(self, dt, force_func=None):
        """
        Actualiza el estado de la partícula.

        Args:
            dt (float): Delta de tiempo
            force_func (callable, optional): Función que aplica fuerzas a la partícula
        """
        # Actualizar tiempo de vida y edad
        self.ttl -= dt
        self.age += dt

        # Verificar si sigue viva
        if self.ttl <= 0:
            self.alive = False
            return

        # Aplicar fuerzas externas si se proporciona una función
        if force_func:
            self.reset_acceleration()
            force_func(self)

        # Método de integración: Velocity Verlet
        # 1. Actualizar posición con velocidad actual y media aceleración
        self.position += dt * self.velocity + 0.5 * dt * dt * self.acceleration

        # 2. Guardar aceleración actual
        old_acceleration = self.acceleration.copy()

        # 3. Calcular nueva aceleración (solo si hay una función de fuerzas)
        if force_func:
            self.reset_acceleration()
            force_func(self)

        # 4. Actualizar velocidad con aceleración promedio
        self.velocity += 0.5 * dt * (old_acceleration + self.acceleration)


class ParticleSystem:
    """Sistema de partículas vectorizado con Velocity Verlet.

    Almacena todas las partículas en arrays NumPy contiguos (struct of arrays)
    para evitar loops de Python tanto en la integración como en la copia a GPU.

    El sistema preasigna memoria para ``max_particles`` partículas.
    Un índice ``n`` indica cuántas están activas (las primeras ``n`` en cada
    array). Las partículas muertas se compactan al final de cada update.
    """

    def __init__(self, max_particles, dims=2):
        self.max = max_particles
        self.dims = dims
        self.n = 0  # partículas activas

        # Arrays preasignados
        self.position = np.zeros((max_particles, dims), dtype=np.float32)
        self.velocity = np.zeros((max_particles, dims), dtype=np.float32)
        self.acceleration = np.zeros((max_particles, dims), dtype=np.float32)
        self.mass = np.ones(max_particles, dtype=np.float32)
        self.ttl = np.zeros(max_particles, dtype=np.float32)
        self.age = np.zeros(max_particles, dtype=np.float32)

    # ── Emisión ──

    def emit(self, positions, velocities, masses, ttls):
        """Agrega partículas al sistema.

        Cada argumento es un array de forma (k, ...) donde k es la cantidad
        de partículas a emitir. Si no caben todas, se emiten las que quepan.
        """
        k = positions.shape[0]
        available = self.max - self.n
        if k > available:
            k = available
        if k == 0:
            return

        s = slice(self.n, self.n + k)
        self.position[s] = positions[:k]
        self.velocity[s] = velocities[:k]
        self.acceleration[s] = 0
        self.mass[s] = masses[:k]
        self.ttl[s] = ttls[:k]
        self.age[s] = 0
        self.n += k

    # ── Integración ──

    def _compact(self):
        """Elimina partículas muertas, compactando los arrays."""
        alive = self.ttl[:self.n] > 0
        n_alive = int(alive.sum())
        if n_alive < self.n:
            self.position[:n_alive] = self.position[:self.n][alive]
            self.velocity[:n_alive] = self.velocity[:self.n][alive]
            self.acceleration[:n_alive] = self.acceleration[:self.n][alive]
            self.mass[:n_alive] = self.mass[:self.n][alive]
            self.ttl[:n_alive] = self.ttl[:self.n][alive]
            self.age[:n_alive] = self.age[:self.n][alive]
            self.n = n_alive

    def update(self, dt, force_func):
        """Velocity Verlet vectorizado + eliminación de partículas muertas.

        ``force_func(system, alive_slice)`` debe escribir las aceleraciones
        directamente en ``system.acceleration[alive_slice]``.

        Requiere dos evaluaciones de fuerzas por paso: una en la posición
        actual y otra en la nueva posición. Esto lo hace preciso (error
        local O(h^4)), pero más costoso que Euler simpléctico.
        """
        if self.n == 0:
            return

        s = slice(0, self.n)

        # TTL y edad
        self.ttl[s] -= dt
        self.age[s] += dt

        # Fuerzas → aceleración (primera evaluación)
        self.acceleration[s] = 0
        force_func(self, s)

        # Velocity Verlet paso 1: posición
        self.position[s] += (
            dt * self.velocity[s] + 0.5 * dt * dt * self.acceleration[s]
        )

        # Guardar aceleración vieja
        old_acc = self.acceleration[s].copy()

        # Fuerzas en la nueva posición (segunda evaluación)
        self.acceleration[s] = 0
        force_func(self, s)

        # Velocity Verlet paso 2: velocidad
        self.velocity[s] += 0.5 * dt * (old_acc + self.acceleration[s])

        self._compact()

    def update_euler(self, dt, force_func):
        """Euler simpléctico vectorizado + eliminación de partículas muertas.

        ``force_func(system, alive_slice)`` debe escribir las aceleraciones
        directamente en ``system.acceleration[alive_slice]``.

        Evalúa fuerzas una sola vez por paso. El orden de actualización
        (primero velocidad, luego posición) lo hace simpléctico: conserva
        mejor la energía que el Euler explícito convencional, con el mismo
        costo. El error local es O(h^2), suficiente para partículas de
        vida corta donde la precisión de la trayectoria individual importa
        menos que el comportamiento colectivo.
        """
        if self.n == 0:
            return

        s = slice(0, self.n)

        # TTL y edad
        self.ttl[s] -= dt
        self.age[s] += dt

        # Una sola evaluación de fuerzas
        self.acceleration[s] = 0
        force_func(self, s)

        # Euler simpléctico: v primero, luego r con la v nueva
        self.velocity[s] += dt * self.acceleration[s]
        self.position[s] += dt * self.velocity[s]

        self._compact()

    # ── Acceso a datos para GPU ──

    def positions_flat(self):
        """Posiciones como array plano (n*dims,) listo para enviar a GPU."""
        return self.position[:self.n].ravel()

    def ttls_flat(self):
        """TTL como array plano (n,) listo para enviar a GPU."""
        return self.ttl[:self.n]

