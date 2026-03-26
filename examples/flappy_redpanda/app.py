import os
from pathlib import Path
import random
import math

import numpy as np
import pyglet
from OpenGL import GL
import click

from grafica.utils import load_pipeline
from grafica.particle import Particle


# %% Comando principal

@click.command("flappy_redpanda", short_help="Flappy Red Panda: EDO con Velocity Verlet")
@click.option("--width", type=int, default=600)
@click.option("--height", type=int, default=400)
@click.option("--gravity", type=float, default=-600.0, help="Gravedad (px/s^2)")
@click.option("--flap_impulse", type=float, default=250.0, help="Impulso vertical al aletear (px/s)")
def flappy_redpanda(width, height, gravity, flap_impulse):
    """Flappy Red Panda: un juego que demuestra la resolución de EDOs.

    La posición vertical del panda rojo está gobernada por la ecuación
    diferencial y'' = g, resuelta con el método Velocity Verlet del
    módulo grafica.particle. Al presionar ESPACIO se aplica un impulso
    instantáneo que modifica la velocidad.
    """

    win = pyglet.window.Window(width, height, caption="Flappy Red Panda")

    pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    # %% Sonidos

    assets_dir = Path(os.path.dirname(__file__)) / "assets"
    snd_flap = pyglet.media.load(str(assets_dir / "flap.mp3"), streaming=False)
    snd_point = pyglet.media.load(str(assets_dir / "point.mp3"), streaming=False)
    snd_hit = pyglet.media.load(str(assets_dir / "flappy-bird-hit-sound.mp3"), streaming=False)
    snd_die = pyglet.media.load(str(assets_dir / "die.mp3"), streaming=False)
    snd_swoosh = pyglet.media.load(str(assets_dir / "swoosh.mp3"), streaming=False)

    # %% Parámetros del juego

    PIPE_WIDTH = 50
    PIPE_GAP = 130
    PIPE_SPEED = 120.0
    PIPE_INTERVAL = 2.0
    PANDA_X = width // 4
    PANDA_RADIUS = 18
    GROUND_HEIGHT = 20

    # %% El panda como partícula
    # La EDO que gobierna su movimiento es:
    #   dy/dt = v
    #   dv/dt = g
    # El integrador Velocity Verlet la resuelve en cada frame.

    panda = Particle(
        position=[PANDA_X, height / 2],
        velocity=[0, 0],
        mass=1.0,
        ttl=float("inf"),
    )

    # %% Estado del juego

    game_state = {
        "running": False,
        "game_over": False,
        "score": 0,
        "time": 0.0,
    }
    pipes = []
    pipe_timer = [0.0]
    gpu_data = {"vertex_list": None}

    # %% Generación de geometría

    def circle_vertices(cx, cy, r, n_segments=16):
        """Genera vértices para un círculo como lista de triángulos."""
        verts = []
        for i in range(n_segments):
            a0 = 2 * math.pi * i / n_segments
            a1 = 2 * math.pi * (i + 1) / n_segments
            verts.extend([cx, cy])
            verts.extend([cx + r * math.cos(a0), cy + r * math.sin(a0)])
            verts.extend([cx + r * math.cos(a1), cy + r * math.sin(a1)])
        return verts

    def rect_vertices(x, y, w, h):
        """Genera 6 vértices (2 triángulos) para un rectángulo."""
        return [
            x, y, x + w, y, x + w, y + h,
            x, y, x + w, y + h, x, y + h,
        ]

    def build_panda(cx, cy):
        """Construye la geometría del panda rojo."""
        r = PANDA_RADIUS
        verts = []
        colors = []

        # Cara principal (naranja rojizo)
        face = circle_vertices(cx, cy, r, 20)
        verts.extend(face)
        colors.extend([0.85, 0.35, 0.1] * (len(face) // 2))

        # Orejas (triángulos oscuros)
        for sign in [-1, 1]:
            verts.extend([
                cx + sign * r * 0.6, cy + r * 0.6,
                cx + sign * r * 0.3, cy + r * 1.3,
                cx + sign * r * 0.9, cy + r * 1.1,
            ])
            colors.extend([0.3, 0.15, 0.05] * 3)

        # Ojos (blancos)
        for sign in [-1, 1]:
            eye = circle_vertices(cx + sign * r * 0.35, cy + r * 0.15, r * 0.2, 8)
            verts.extend(eye)
            colors.extend([1.0, 1.0, 1.0] * (len(eye) // 2))

        # Pupilas
        for sign in [-1, 1]:
            pupil = circle_vertices(cx + sign * r * 0.35, cy + r * 0.15, r * 0.1, 8)
            verts.extend(pupil)
            colors.extend([0.1, 0.1, 0.1] * (len(pupil) // 2))

        # Nariz
        nose = circle_vertices(cx, cy - r * 0.15, r * 0.12, 8)
        verts.extend(nose)
        colors.extend([0.15, 0.08, 0.02] * (len(nose) // 2))

        return verts, colors

    def build_pipe(pipe):
        """Genera la geometría de un par de tubos (superior e inferior)."""
        x = pipe["x"]
        gap_y = pipe["gap_y"]
        half_gap = PIPE_GAP / 2

        verts = []
        colors = []

        # Tubo inferior
        bottom_h = gap_y - half_gap
        if bottom_h > 0:
            v = rect_vertices(x, 0, PIPE_WIDTH, bottom_h)
            verts.extend(v)
            colors.extend([0.2, 0.65, 0.2] * (len(v) // 2))

        # Tubo superior
        top_y = gap_y + half_gap
        if top_y < height:
            v = rect_vertices(x, top_y, PIPE_WIDTH, height - top_y)
            verts.extend(v)
            colors.extend([0.2, 0.65, 0.2] * (len(v) // 2))

        # Labios de los tubos (bordes más anchos)
        lip_extra = 6
        lip_h = 16

        # Labio del tubo inferior
        if bottom_h > 0:
            v = rect_vertices(
                x - lip_extra, bottom_h - lip_h,
                PIPE_WIDTH + 2 * lip_extra, lip_h,
            )
            verts.extend(v)
            colors.extend([0.15, 0.55, 0.15] * (len(v) // 2))

        # Labio del tubo superior
        if top_y < height:
            v = rect_vertices(
                x - lip_extra, top_y,
                PIPE_WIDTH + 2 * lip_extra, lip_h,
            )
            verts.extend(v)
            colors.extend([0.15, 0.55, 0.15] * (len(v) // 2))

        return verts, colors

    def build_ground():
        """Genera el suelo."""
        v = rect_vertices(0, 0, width, GROUND_HEIGHT)
        c = [0.55, 0.35, 0.17] * (len(v) // 2)
        return v, c

    # %% Función de fuerzas para la EDO

    def gravity_force(particle):
        """Define la fuerza sobre el panda: F = m * g.

        Esto produce la EDO:
            dy/dt = v
            dv/dt = g
        que Velocity Verlet resuelve numéricamente.
        """
        particle.apply_force(np.array([0, gravity], dtype=np.float32))

    # %% Detección de colisiones

    def check_collision():
        px, py = panda.position
        r = PANDA_RADIUS

        # Suelo y techo
        if py - r < GROUND_HEIGHT or py + r > height:
            return True

        # Tubos
        for pipe in pipes:
            pipe_x = pipe["x"]
            gap_y = pipe["gap_y"]
            half_gap = PIPE_GAP / 2

            if px + r > pipe_x and px - r < pipe_x + PIPE_WIDTH:
                if py - r < gap_y - half_gap or py + r > gap_y + half_gap:
                    return True

        return False

    # %% Reconstrucción de la escena

    def rebuild_scene():
        if gpu_data["vertex_list"] is not None:
            gpu_data["vertex_list"].delete()
            gpu_data["vertex_list"] = None

        all_verts = []
        all_colors = []

        # Suelo
        v, c = build_ground()
        all_verts.extend(v)
        all_colors.extend(c)

        # Tubos
        for pipe in pipes:
            v, c = build_pipe(pipe)
            all_verts.extend(v)
            all_colors.extend(c)

        # Panda (se dibuja encima de todo)
        v, c = build_panda(panda.position[0], panda.position[1])
        all_verts.extend(v)
        all_colors.extend(c)

        n_verts = len(all_verts) // 2
        if n_verts > 0:
            gpu_data["vertex_list"] = pipeline.vertex_list(
                n_verts,
                pyglet.gl.GL_TRIANGLES,
                position=("f", all_verts),
                color=("f", all_colors),
            )

    # %% Labels

    score_label = pyglet.text.Label(
        "0",
        font_size=28,
        x=width // 2,
        y=height - 40,
        anchor_x="center",
        anchor_y="center",
        color=(255, 255, 255, 255),
    )

    info_label = pyglet.text.Label(
        "ESPACIO para comenzar",
        font_size=14,
        x=width // 2,
        y=height // 2 - 50,
        anchor_x="center",
        anchor_y="center",
        color=(255, 255, 255, 200),
    )

    method_label = pyglet.text.Label(
        "Velocity Verlet  |  y'' = g",
        font_size=10,
        x=10,
        y=height - 15,
        anchor_x="left",
        anchor_y="center",
        color=(200, 200, 200, 180),
    )

    # %% Eventos

    @win.event
    def on_draw():
        GL.glClearColor(0.4, 0.7, 0.9, 1.0)
        win.clear()

        pipeline.use()
        pipeline["resolution"] = np.array([width, height], dtype=np.float32)

        if gpu_data["vertex_list"] is not None:
            gpu_data["vertex_list"].draw(pyglet.gl.GL_TRIANGLES)

        score_label.draw()
        method_label.draw()

        if not game_state["running"]:
            info_label.draw()

    @win.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            if game_state["game_over"]:
                snd_swoosh.play()
                reset_game()
            elif not game_state["running"]:
                game_state["running"] = True
                info_label.text = ""
                panda.velocity[1] = flap_impulse
                snd_flap.play()
            else:
                # Impulso instantáneo: modifica v directamente
                panda.velocity[1] = flap_impulse
                snd_flap.play()

        elif symbol == pyglet.window.key.ESCAPE:
            win.close()

    def reset_game():
        panda.position[:] = [PANDA_X, height / 2]
        panda.velocity[:] = [0, 0]
        panda.acceleration[:] = [0, 0]
        pipes.clear()
        pipe_timer[0] = 0.0
        game_state["running"] = False
        game_state["game_over"] = False
        game_state["score"] = 0
        game_state["time"] = 0.0
        score_label.text = "0"
        info_label.text = "ESPACIO para comenzar"

    # %% Loop de actualización

    def update(dt, win):
        if not game_state["running"]:
            rebuild_scene()
            return

        # Limitar dt para evitar saltos grandes
        dt = min(dt, 1 / 30.0)
        game_state["time"] += dt

        # Integración de la EDO con Velocity Verlet
        panda.update(dt, gravity_force)

        # Generar tubos nuevos
        pipe_timer[0] += dt
        if pipe_timer[0] >= PIPE_INTERVAL:
            pipe_timer[0] -= PIPE_INTERVAL
            gap_y = random.uniform(height * 0.25, height * 0.75)
            pipes.append({"x": float(width), "gap_y": gap_y, "scored": False})

        # Mover tubos
        for pipe in pipes:
            pipe["x"] -= PIPE_SPEED * dt

        # Eliminar tubos fuera de pantalla
        while pipes and pipes[0]["x"] + PIPE_WIDTH < 0:
            pipes.pop(0)

        # Puntaje
        for pipe in pipes:
            if not pipe["scored"] and pipe["x"] + PIPE_WIDTH < PANDA_X:
                pipe["scored"] = True
                game_state["score"] += 1
                score_label.text = str(game_state["score"])
                snd_point.play()

        # Colisiones
        if check_collision():
            game_state["running"] = False
            game_state["game_over"] = True
            info_label.text = f"Puntaje: {game_state['score']}. ESPACIO para reiniciar"
            snd_hit.play()
            snd_die.play()

        rebuild_scene()

    # %% Inicio

    pyglet.clock.schedule(update, win)

    print("Flappy Red Panda")
    print("La posición del panda está gobernada por la EDO: y'' = g")
    print("Integración numérica: Velocity Verlet")
    print("Controles: ESPACIO para aletear, ESC para salir")

    pyglet.app.run()
