import numpy as np
import pyglet
import OpenGL.GL as GL

from .world import World
from pathlib import Path
from grafica.utils import load_pipeline
import click


# variables del estado del programa
program_state = {
    "paused": False,
    "show_quadtree": False,
    "show_vision": False,
    "selected_boid": None,
}

# variables del mundo que simularemos
world_parameters = {
    "vision": {"min": 60, "max": 200, "default": 100},
    "cohere_factor": {"min": 0.0001, "max": 0.001, "default": 0.00075},
    "separation_factor": {"min": 0.0001, "max": 0.01, "default": 0.0075},
    "match_factor": {"min": 0.0001, "max": 0.01, "default": 0.0075},
    "distance": {"min": 20, "max": 60, "default": 25},
    "speed": {"min": 0.01, "max": 1.0, "default": 0.75},
}


@click.command(
    "boids_particles",
    short_help="Simulador de vuelo de pajaritos usando ABM (version particulas)",
)
@click.option("--n_pajaritos", type=int, default=60)
@click.option("--width", type=int, default=1024)
@click.option("--height", type=int, default=768)
@click.option(
    "--spatial",
    type=click.Choice(["quadtree", "kdtree"]),
    default="quadtree",
    help="Estructura espacial para consulta de vecinos",
)
def boids_particles(n_pajaritos, width, height, spatial):
    window = pyglet.window.Window(width=width, height=height)
    window.set_caption("BOIDS - P: pausa, T: quadtree, V: vision, +/-: velocidad")

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    pipeline.use()
    pipeline["resolution"] = (width, height)

    # este es el mundo a simular.
    flock = World(
        n_pajaritos,
        width=width,
        height=height,
        speed=world_parameters["speed"]["default"],
        vision=world_parameters["vision"]["default"],
        distance=world_parameters["distance"]["default"],
        cohere_factor=world_parameters["cohere_factor"]["default"],
        separation_factor=world_parameters["separation_factor"]["default"],
        match_factor=world_parameters["match_factor"]["default"],
        spatial_method=spatial,
    )

    # esta funcion ejecutara un paso de la simulacion
    def tick(time):
        if not program_state["paused"]:
            flock.step()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.P:
            program_state["paused"] = not program_state["paused"]
        elif symbol == pyglet.window.key.T:
            program_state["show_quadtree"] = not program_state["show_quadtree"]
        elif symbol == pyglet.window.key.V:
            program_state["show_vision"] = not program_state["show_vision"]
        elif symbol == pyglet.window.key.PLUS or symbol == pyglet.window.key.EQUAL:
            flock.speed = min(world_parameters["speed"]["max"], flock.speed + 0.05)
            for b in flock.id_to_agent.values():
                b.speed = flock.speed
        elif symbol == pyglet.window.key.MINUS:
            flock.speed = max(world_parameters["speed"]["min"], flock.speed - 0.05)
            for b in flock.id_to_agent.values():
                b.speed = flock.speed
        elif symbol == pyglet.window.key.UP:
            flock.vision = min(world_parameters["vision"]["max"], flock.vision + 10)
            for b in flock.id_to_agent.values():
                b.vision = flock.vision
        elif symbol == pyglet.window.key.DOWN:
            flock.vision = max(world_parameters["vision"]["min"], flock.vision - 10)
            for b in flock.id_to_agent.values():
                b.vision = flock.vision

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            # seleccionar el boid mas cercano
            best = None
            best_dist = float("inf")
            for idx, boid in flock.id_to_agent.items():
                d = (boid.pos[0] - x) ** 2 + (boid.pos[1] - y) ** 2
                if d < best_dist:
                    best_dist = d
                    best = idx
            if best is not None and best_dist < 900:
                program_state["selected_boid"] = best
            else:
                program_state["selected_boid"] = None

    @window.event
    def on_draw():
        GL.glClearColor(0.92, 0.92, 0.90, 1.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        window.clear()

        pipeline.use()

        # dibujar cuadrantes del quadtree
        if program_state["show_quadtree"] and flock.qt is not None:
            draw_quadtree()

        # dibujar radio de vision del boid seleccionado
        if program_state["show_vision"] and program_state["selected_boid"] is not None:
            draw_vision()

        # dibujar boids
        draw_boids()

        # HUD con metricas
        draw_hud()

    def make_rect_lines(cx, cy, hw, hh):
        x0, x1 = cx - hw, cx + hw
        y0, y1 = cy - hh, cy + hh
        return [
            x0, y0, x1, y0,
            x1, y0, x1, y1,
            x1, y1, x0, y1,
            x0, y1, x0, y0,
        ]

    def make_circle_lines(cx, cy, r, segments=32):
        verts = []
        for i in range(segments):
            a0 = 2 * np.pi * i / segments
            a1 = 2 * np.pi * (i + 1) / segments
            verts.extend([
                cx + r * np.cos(a0), cy + r * np.sin(a0),
                cx + r * np.cos(a1), cy + r * np.sin(a1),
            ])
        return verts

    def draw_quadtree():
        rects = flock.qt.get_rectangles()
        positions = []
        colors = []
        for cx, cy, hw, hh, depth in rects:
            positions.extend(make_rect_lines(cx, cy, hw, hh))
            t = min(1.0, depth / 6)
            c = (0.55 + 0.2 * t, 0.65 + 0.1 * t, 0.75)
            colors.extend(c * 8)

        if not positions:
            return

        n_verts = len(positions) // 2
        vl = pipeline.vertex_list(n_verts, pyglet.gl.GL_LINES, position="f", color="f")
        vl.position[:] = np.array(positions, dtype=np.float32)
        vl.color[:] = np.array(colors, dtype=np.float32)
        vl.draw(pyglet.gl.GL_LINES)
        vl.delete()

    def draw_vision():
        idx = program_state["selected_boid"]
        if idx not in flock.id_to_agent:
            return
        boid = flock.id_to_agent[idx]

        # circulo de vision
        circle_pos = make_circle_lines(boid.pos[0], boid.pos[1], boid.vision)
        n_circle = len(circle_pos) // 2
        circle_color = [0.3, 0.6, 0.9] * n_circle

        vl = pipeline.vertex_list(n_circle, pyglet.gl.GL_LINES, position="f", color="f")
        vl.position[:] = np.array(circle_pos, dtype=np.float32)
        vl.color[:] = np.array(circle_color, dtype=np.float32)
        vl.draw(pyglet.gl.GL_LINES)
        vl.delete()

        # lineas a vecinos
        neighbors = flock.query_area(boid.pos, boid.vision)
        neighbors = [n for n in neighbors if n is not boid]
        if neighbors:
            line_pos = []
            line_col = []
            for n in neighbors:
                line_pos.extend([boid.pos[0], boid.pos[1], n.pos[0], n.pos[1]])
                line_col.extend([0.3, 0.6, 0.9] * 2)

            n_verts = len(line_pos) // 2
            vl = pipeline.vertex_list(
                n_verts, pyglet.gl.GL_LINES, position="f", color="f"
            )
            vl.position[:] = np.array(line_pos, dtype=np.float32)
            vl.color[:] = np.array(line_col, dtype=np.float32)
            vl.draw(pyglet.gl.GL_LINES)
            vl.delete()

    def draw_boids():
        positions = np.zeros(n_pajaritos * 3 * 2, dtype=np.float32)
        colors = np.zeros(n_pajaritos * 3 * 3, dtype=np.float32)

        selected = program_state["selected_boid"]
        polarization = flock.compute_polarization()

        for i, boid in enumerate(flock.iter_agents()):
            angle = np.arctan2(boid.velocity[1], boid.velocity[0])

            size = 10.0

            v1 = (
                boid.pos[0] + size * np.cos(angle),
                boid.pos[1] + size * np.sin(angle),
            )
            v2 = (
                boid.pos[0] + size * 0.5 * np.cos(angle + np.pi * 2 / 3),
                boid.pos[1] + size * 0.5 * np.sin(angle + np.pi * 2 / 3),
            )
            v3 = (
                boid.pos[0] + size * 0.5 * np.cos(angle - np.pi * 2 / 3),
                boid.pos[1] + size * 0.5 * np.sin(angle - np.pi * 2 / 3),
            )

            positions[i * 6 : i * 6 + 2] = v1
            positions[i * 6 + 2 : i * 6 + 4] = v2
            positions[i * 6 + 4 : i * 6 + 6] = v3

            if selected is not None and i == selected:
                r, g, b = 0.9, 0.2, 0.2
            else:
                # color basado en velocidad relativa
                speed_ratio = min(
                    1.0, boid.current_speed / world_parameters["speed"]["max"]
                )
                r = 0.2 + 0.6 * speed_ratio
                g = 0.3 + 0.4 * (1.0 - speed_ratio)
                b = 0.4

            for j in range(3):
                colors[i * 9 + j * 3 : i * 9 + j * 3 + 3] = (r, g, b)

        vl = pipeline.vertex_list(
            n_pajaritos * 3, pyglet.gl.GL_TRIANGLES, position="f", color="f"
        )
        vl.position[:] = positions
        vl.color[:] = colors
        vl.draw(pyglet.gl.GL_TRIANGLES)
        vl.delete()

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )
    hud_label = pyglet.text.Label(
        "",
        font_name="FiraCode",
        font_size=11,
        x=10,
        y=height - 20,
        color=(60, 60, 60, 200),
        multiline=True,
        width=width - 20,
    )

    def draw_hud():
        pol = flock.compute_polarization()
        disp = flock.compute_dispersion()
        method = flock.spatial_method
        qt_info = ""
        if flock.qt is not None:
            qt_info = f"  Nodos: {len(flock.qt.get_rectangles())}"

        hud_label.text = (
            f"Boids: {n_pajaritos}  |  Pol: {pol:.2f}  |  Disp: {disp:.0f}  |  "
            f"Vision: {flock.vision:.0f}  |  Speed: {flock.speed:.2f}  |  "
            f"{method}{qt_info}\n"
            f"[P] pausa  [T] quadtree  [V] vision  "
            f"[+/-] velocidad  [Up/Down] vision  [Click] seleccionar"
        )
        hud_label.draw()

    pyglet.clock.schedule_interval(tick, 1 / 60)
    pyglet.app.run()


if __name__ == "__main__":
    boids_particles()
