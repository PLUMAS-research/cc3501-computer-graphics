import numpy as np
import pyglet
import OpenGL.GL as GL

from pathlib import Path
from grafica.utils import load_pipeline
from grafica.quadtree import QuadTree, Rectangle
import click


@click.command(
    "quadtree_demo",
    short_help="Demostración interactiva de quadtree",
)
@click.option("--width", type=int, default=900)
@click.option("--height", type=int, default=700)
@click.option("--capacity", type=int, default=4, help="Capacidad maxima por nodo")
@click.option("--query_radius", type=float, default=80.0, help="Radio de consulta")
def quadtree_demo(width, height, capacity, query_radius):
    window = pyglet.window.Window(width=width, height=height)
    window.set_caption("Quadtree - Click para agregar puntos, rueda para cambiar radio")

    pipeline = load_pipeline(
        Path(__file__).parent / "vertex_program.glsl",
        Path(__file__).parent / "fragment_program.glsl",
    )

    pipeline.use()
    pipeline["resolution"] = (width, height)

    state = {
        "mouse_x": width / 2,
        "mouse_y": height / 2,
        "query_radius": query_radius,
        "show_query": True,
        "show_grid": True,
    }

    points = []

    def build_quadtree():
        boundary = Rectangle(width / 2, height / 2, width / 2, height / 2)
        qt = QuadTree(boundary, capacity=capacity, max_depth=8)
        for i, (x, y) in enumerate(points):
            qt.insert(x, y, i)
        return qt

    qt = build_quadtree()

    # -- funciones de dibujo --

    def make_rect_lines(cx, cy, hw, hh):
        """Genera 8 vertices (4 lineas) para un rectangulo."""
        x0, x1 = cx - hw, cx + hw
        y0, y1 = cy - hh, cy + hh
        return [
            x0, y0, x1, y0,
            x1, y0, x1, y1,
            x1, y1, x0, y1,
            x0, y1, x0, y0,
        ]

    def make_circle_lines(cx, cy, r, segments=48):
        """Genera vertices para dibujar un circulo como lineas."""
        verts = []
        for i in range(segments):
            a0 = 2 * np.pi * i / segments
            a1 = 2 * np.pi * (i + 1) / segments
            verts.extend([
                cx + r * np.cos(a0), cy + r * np.sin(a0),
                cx + r * np.cos(a1), cy + r * np.sin(a1),
            ])
        return verts

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            points.append((x, y))
            nonlocal qt
            qt = build_quadtree()

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        state["mouse_x"] = x
        state["mouse_y"] = y

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.RIGHT:
            state["mouse_x"] = x
            state["mouse_y"] = y

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        state["query_radius"] = max(10, state["query_radius"] + scroll_y * 10)

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            state["show_query"] = not state["show_query"]
        elif symbol == pyglet.window.key.G:
            state["show_grid"] = not state["show_grid"]
        elif symbol == pyglet.window.key.C:
            points.clear()
            nonlocal qt
            qt = build_quadtree()
        elif symbol == pyglet.window.key.R:
            # agregar puntos aleatorios
            for _ in range(20):
                px = np.random.uniform(20, width - 20)
                py = np.random.uniform(20, height - 20)
                points.append((px, py))
            qt = build_quadtree()

    @window.event
    def on_draw():
        GL.glClearColor(0.95, 0.95, 0.92, 1.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        window.clear()

        pipeline.use()

        # dibujar cuadrantes del quadtree
        if state["show_grid"]:
            draw_quadtree_grid()

        # dibujar consulta de rango
        if state["show_query"] and points:
            draw_query()

        # dibujar puntos
        if points:
            draw_points()

        # dibujar texto informativo
        draw_hud()

    def draw_quadtree_grid():
        mx, my = state["mouse_x"], state["mouse_y"]
        r = state["query_radius"]

        if state["show_query"] and points:
            # mostrar cuadrantes visitados con color diferenciado
            visited = qt.get_visited_rectangles(mx, my, r)
            positions = []
            colors = []
            for cx, cy, hw, hh, depth, intersects in visited:
                positions.extend(make_rect_lines(cx, cy, hw, hh))
                if intersects:
                    c = (0.2, 0.5, 0.8)
                else:
                    c = (0.75, 0.75, 0.75)
                colors.extend(c * 8)
        else:
            rects = qt.get_rectangles()
            positions = []
            colors = []
            for cx, cy, hw, hh, depth in rects:
                positions.extend(make_rect_lines(cx, cy, hw, hh))
                t = min(1.0, depth / 6)
                c = (0.4 + 0.3 * t, 0.4 + 0.1 * t, 0.4 - 0.2 * t)
                colors.extend(c * 8)

        if not positions:
            return

        n_verts = len(positions) // 2
        vl = pipeline.vertex_list(n_verts, pyglet.gl.GL_LINES, position="f", color="f")
        vl.position[:] = np.array(positions, dtype=np.float32)
        vl.color[:] = np.array(colors, dtype=np.float32)
        vl.draw(pyglet.gl.GL_LINES)
        vl.delete()

    def draw_query():
        mx, my = state["mouse_x"], state["mouse_y"]
        r = state["query_radius"]

        # circulo de consulta
        circle_pos = make_circle_lines(mx, my, r)
        n_circle = len(circle_pos) // 2
        circle_color = [0.2, 0.5, 0.8] * n_circle

        vl = pipeline.vertex_list(n_circle, pyglet.gl.GL_LINES, position="f", color="f")
        vl.position[:] = np.array(circle_pos, dtype=np.float32)
        vl.color[:] = np.array(circle_color, dtype=np.float32)
        vl.draw(pyglet.gl.GL_LINES)
        vl.delete()

        # puntos encontrados en la consulta: resaltarlos
        found = qt.query_circle(mx, my, r)
        if found:
            positions = []
            colors = []
            size = 6.0
            for (px, py), _ in found:
                positions.extend([
                    px, py + size,
                    px - size * 0.866, py - size * 0.5,
                    px + size * 0.866, py - size * 0.5,
                ])
                colors.extend([0.2, 0.5, 0.8] * 3)

            n_verts = len(positions) // 2
            vl = pipeline.vertex_list(
                n_verts, pyglet.gl.GL_TRIANGLES, position="f", color="f"
            )
            vl.position[:] = np.array(positions, dtype=np.float32)
            vl.color[:] = np.array(colors, dtype=np.float32)
            vl.draw(pyglet.gl.GL_TRIANGLES)
            vl.delete()

    def draw_points():
        positions = []
        colors = []
        size = 4.0

        for x, y in points:
            positions.extend([
                x, y + size,
                x - size * 0.866, y - size * 0.5,
                x + size * 0.866, y - size * 0.5,
            ])
            colors.extend([0.3, 0.3, 0.3] * 3)

        n_verts = len(positions) // 2
        vl = pipeline.vertex_list(
            n_verts, pyglet.gl.GL_TRIANGLES, position="f", color="f"
        )
        vl.position[:] = np.array(positions, dtype=np.float32)
        vl.color[:] = np.array(colors, dtype=np.float32)
        vl.draw(pyglet.gl.GL_TRIANGLES)
        vl.delete()

    pyglet.font.add_file(
        str(Path(__file__).parent.parent.parent / "assets" / "FiraCode" / "FiraCode-Regular.ttf")
    )
    hud_label = pyglet.text.Label(
        "", font_name="Fira Code", font_size=11,
        x=10, y=height - 20,
        color=(80, 80, 80, 200),
        multiline=True, width=width - 20,
    )

    def draw_hud():
        mx, my = state["mouse_x"], state["mouse_y"]
        r = state["query_radius"]
        n_found = len(qt.query_circle(mx, my, r)) if points else 0
        rects = qt.get_rectangles()

        hud_label.text = (
            f"Puntos: {len(points)}  |  Nodos: {len(rects)}  |  "
            f"Radio: {r:.0f}  |  En rango: {n_found}\n"
            f"[Click] agregar  [R] +20 aleatorios  [C] limpiar  "
            f"[G] grilla  [Q] consulta  [Rueda] radio"
        )
        hud_label.draw()

    pyglet.app.run()
