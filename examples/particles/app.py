import os
from pathlib import Path

import numpy as np
import OpenGL.GL as GL
import pyglet

import click

from grafica.utils import load_pipeline
from grafica.particle import ParticleSystem


# %% Comando principal

@click.command("particles", short_help="Partículas simples con comportamiento basado en fuerzas")
@click.option("--width", type=int, default=900)
@click.option("--height", type=int, default=600)
@click.option("--max_ttl", type=int, default=3)
@click.option("--emission_rate", type=int, default=3, help="Partículas emitidas por frame")
@click.option("--max_particles", type=int, default=500)
@click.option(
    "--integrator",
    type=click.Choice(["verlet", "euler"]),
    default="euler",
    help="Método de integración: verlet (preciso, 2 eval/paso) o euler simpléctico (rápido, 1 eval/paso)",
)
def particulas(width, height, max_ttl, emission_rate, max_particles, integrator):
    win = pyglet.window.Window(width, height)

    pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "point_vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "point_fragment_program.glsl",
    )

    pipeline.use()
    pipeline["max_ttl"] = max_ttl
    pipeline["resolution"] = (width, height)

    # %% Sistema de partículas vectorizado

    system = ParticleSystem(max_particles)

    # Selección de integrador
    step = system.update if integrator == "verlet" else system.update_euler

    # Tiempo global y posición del mouse
    time = 0.0
    last_mouse_pos = np.array([width // 2, height // 2], dtype=np.float32)

    # %% Buffer de GPU preasignado
    # Los slots inactivos tienen ttl=0, por lo que el shader los hace
    # invisibles (gl_PointSize=0, alpha=0). Esto evita recrear buffers.

    particle_data = pipeline.vertex_list(
        max_particles, pyglet.gl.GL_POINTS, position="f", ttl="f"
    )

    # Array de zeros preasignado para limpiar slots inactivos en GPU
    _zero_ttl = np.zeros(max_particles, dtype=np.float32)

    # %% Scratch arrays preasignados para evitar allocaciones por frame
    # Se usan dentro de apply_forces y emit_batch.

    _scratch_repulsion = np.zeros((max_particles, 2), dtype=np.float32)
    _scratch_turbulence = np.zeros((max_particles, 2), dtype=np.float32)
    _scratch_mass_col = np.zeros((max_particles, 1), dtype=np.float32)

    # Generador de números aleatorios con soporte para out=
    _rng = np.random.default_rng()

    # %% Emisión de partículas

    def emit_batch(n, center):
        """Emite n partículas centradas en ``center`` con jitter."""
        if n <= 0:
            return
        positions = center + np.random.uniform(-15, 15, (n, 2)).astype(np.float32)

        angles = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
        speeds = np.random.uniform(10, 80, n).astype(np.float32)
        velocities = np.column_stack([
            speeds * np.cos(angles),
            speeds * np.sin(angles) - 30,
        ]).astype(np.float32)

        masses = np.random.uniform(0.8, 1.2, n).astype(np.float32)
        ttls = (max_ttl * np.random.uniform(0.7, 1.3, n)).astype(np.float32)

        system.emit(positions, velocities, masses, ttls)

    # %% Fuerzas vectorizadas (sin allocaciones temporales)

    def apply_forces(sys, s):
        """Aplica todas las fuerzas sobre las partículas activas.

        Escribe directamente en sys.acceleration[s]. Los arrays temporales
        están preasignados para evitar allocaciones en cada frame.
        """
        n = sys.n
        pos = sys.position[s]

        # Masa como columna para broadcast: (n, 1)
        mass_col = _scratch_mass_col[:n]
        mass_col[:, 0] = sys.mass[:n]

        # 1. Gravedad
        sys.acceleration[:n, 1] += -98.0

        # 2. Viento oscilante
        sys.acceleration[:n, 0] += 20 * np.sin(time * 0.5)

        # 3. Turbulencia aleatoria (in-place en scratch)
        turb = _scratch_turbulence[:n]
        _rng.random(size=(n, 2), dtype=np.float32, out=turb)
        turb *= 20.0   # escalar [0,1) -> [0,20)
        turb -= 10.0   # desplazar -> [-10, 10)
        turb /= mass_col
        sys.acceleration[s] += turb

        # 4. Repulsión de bordes
        repulsion = _scratch_repulsion[:n]
        repulsion[:] = 0

        dist_left = pos[:, 0]
        dist_right = width - pos[:, 0]
        dist_bottom = pos[:, 1]
        dist_top = height - pos[:, 1]

        edge_margin = 50.0

        mask = dist_left < edge_margin
        repulsion[mask, 0] += 5 * (edge_margin - dist_left[mask])
        mask = dist_right < edge_margin
        repulsion[mask, 0] -= 5 * (edge_margin - dist_right[mask])
        mask = dist_bottom < edge_margin
        repulsion[mask, 1] += 5 * (edge_margin - dist_bottom[mask])
        mask = dist_top < edge_margin
        repulsion[mask, 1] -= 5 * (edge_margin - dist_top[mask])

        repulsion /= mass_col
        sys.acceleration[s] += repulsion

    # %% Colisiones con los bordes

    def handle_boundary_collisions():
        n = system.n
        if n == 0:
            return

        pos = system.position[:n]
        vel = system.velocity[:n]

        # X
        mask = pos[:, 0] < 0
        pos[mask, 0] = 0
        vel[mask, 0] *= -0.7

        mask = pos[:, 0] > width
        pos[mask, 0] = width
        vel[mask, 0] *= -0.7

        # Y
        mask = pos[:, 1] < 0
        pos[mask, 1] = 0
        vel[mask, 1] *= -0.6

        mask = pos[:, 1] > height
        pos[mask, 1] = height
        vel[mask, 1] *= -0.7

    # %% Eventos

    @win.event
    def on_draw():
        win.clear()
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        pipeline.use()

        particle_data.draw(pyglet.gl.GL_POINTS)

    @win.event
    def on_mouse_motion(x, y, dx, dy):
        nonlocal last_mouse_pos
        last_mouse_pos = np.array([x, y], dtype=np.float32)
        emit_batch(2, last_mouse_pos)

    def emit_particles(dt, win):
        emit_batch(emission_rate, last_mouse_pos)

    def update_particle_system(dt, win):
        nonlocal time
        time += dt

        # Integración vectorizada + compactación
        step(dt, apply_forces)
        handle_boundary_collisions()

        # Enviar datos a GPU (sin recrear buffers)
        n = system.n
        if n > 0:
            particle_data.position[:n * 2] = system.positions_flat()
            particle_data.ttl[:n] = system.ttls_flat()

        # Los slots restantes quedan con ttl <= 0, así el shader los oculta
        if n < max_particles:
            particle_data.ttl[n:] = _zero_ttl[:max_particles - n]

    # %% Inicio

    pyglet.clock.schedule(emit_particles, win)
    pyglet.clock.schedule(update_particle_system, win)

    print(f"Partículas (máx {max_particles}, integrador: {integrator})")
    print("Mueve el mouse para emitir partículas")

    pyglet.app.run()
