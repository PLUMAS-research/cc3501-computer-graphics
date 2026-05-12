"""Pinball 2D simulado con pymunk y dibujado en 3D con pyglet.

Pone en accion los conceptos de la unidad de cuerpos rigidos: cuerpos dinamicos
(la bola), estaticos (paredes y bumpers), articulaciones (los flippers, con
PivotJoint y RotaryLimitJoint) y callbacks de colision (cualquier objeto que
toca la bola se enciende en amarillo; los bumpers ademas suman puntos y
empujan la bola). El mundo fisico es 2D, pero cada cuerpo se dibuja como un
cubo o esfera en una escena 3D.

Controles:
- Shift izquierdo: flipper izquierdo.
- Shift derecho:   flipper derecho.
- Espacio:         lanza una bola desde el canal del plunger.
- R:               reinicia el puntaje.
- P:               alterna proyeccion perspectiva / ortografica 2D.
- ESC:             cierra la ventana.
"""

import os
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import pymunk
import trimesh as tm

import grafica.transformations as tr


# tipos de colision para los pares en callbacks.
COLLISION_TYPE_DEFAULT = 0  # paredes y flippers
COLLISION_TYPE_BALL = 1
COLLISION_TYPE_BUMPER = 2
COLLISION_TYPE_DRAIN = 3

# colores.
COLOR_WALL = (0.7, 0.7, 0.75)
COLOR_FLIPPER = (0.95, 0.65, 0.2)
COLOR_BUMPER = (0.4, 0.7, 1.0)
COLOR_BALL = (1.0, 0.45, 0.45)
COLOR_FLASH = (1.0, 1.0, 0.1)  # amarillo brillante para "ball touched me"

# dimensiones del cabinete (unidades arbitrarias).
CAB_W = 7.0
CAB_H = 14.0

# canal del plunger: lo suficientemente ancho para que la bola (diametro 0.9)
# quepa con margen entre la pared interior y el borde derecho del cabinete.
CHANNEL_INNER_X = 5.3
CHANNEL_TOP_Y = 8.0  # donde termina la pared vertical del canal

# bola.
BALL_RADIUS = 0.45
BALL_LAUNCH_VELOCITY = 38.0

# rampas y flippers: las rampas inferiores son simetricas y los pivotes de los
# flippers quedan SOBRE la rampa, en x=+-FLIPPER_PIVOT_DX. el flipper en reposo
# es colineal con la rampa (lo que evita huecos donde la bola se atasque).
# como rampa y flipper se intersectan en el pivote, las pongo en el mismo
# ShapeFilter.group para que no se colisionen entre si (la bola si colisiona
# con ambos porque tiene group=0).
DRAIN_HALF = 1.4
FLIPPER_PIVOT_DX = 3.0
RAMP_LEFT_START = (-CAB_W, -10.0)
RAMP_LEFT_END = (-DRAIN_HALF, -CAB_H + 0.4)
RAMP_RIGHT_END = (DRAIN_HALF, -CAB_H + 0.4)


def _line_y_at(x, p1, p2):
    return p1[1] + (p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0])


def _line_angle(p1, p2):
    import math
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


FLIPPER_Y = _line_y_at(-FLIPPER_PIVOT_DX, RAMP_LEFT_START, RAMP_LEFT_END)
LEFT_RAMP_ANGLE = _line_angle(RAMP_LEFT_START, RAMP_LEFT_END)
# rampa derecha simetrica a la izquierda: misma pendiente absoluta.
RAMP_RIGHT_START = (CHANNEL_INNER_X, RAMP_RIGHT_END[1] - LEFT_RAMP_ANGLE * (CHANNEL_INNER_X - DRAIN_HALF))
# que sea exactamente simetrica respecto a x=0
RAMP_RIGHT_START = (CHANNEL_INNER_X, _line_y_at(CHANNEL_INNER_X, (-RAMP_LEFT_END[0], RAMP_LEFT_END[1]), (-RAMP_LEFT_START[0], RAMP_LEFT_START[1])))

FLIPPER_LENGTH = 1.9
FLIPPER_THICKNESS = 0.32
FLIPPER_REST_ANGLE_LEFT = LEFT_RAMP_ANGLE  # colineal con la rampa
FLIPPER_REST_ANGLE_RIGHT = -LEFT_RAMP_ANGLE
FLIPPER_UP_ANGLE_LEFT = 0.45
FLIPPER_UP_ANGLE_RIGHT = -0.45
FLIPPER_GROUP = 1

# escenario.
WALL_RADIUS = 0.18
SUBSTEPS = 4  # subdivision de cada frame para evitar tunneling


@click.command("pymunk_pinball", short_help="Pinball 2D con joints y callbacks, render 3D")
@click.option("--width", type=int, default=720)
@click.option("--height", type=int, default=1080)
def pymunk_pinball(width, height):
    window = pyglet.window.Window(width, height, caption="pymunk pinball")

    # mallas: cubo para paredes, bumpers y flippers; esfera para la bola.
    cube = tm.load("assets/cube.off")
    cube.apply_translation(-cube.centroid)
    cube.apply_scale(np.sqrt(3) / cube.scale)

    sphere = tm.load("assets/sphere.off")
    sphere.apply_translation(-sphere.centroid)
    sphere.apply_scale(np.sqrt(3) / sphere.scale)

    with open(Path(os.path.dirname(__file__)) / "vertex_program.glsl") as f:
        vertex_source_code = f.read()
    with open(Path(os.path.dirname(__file__)) / "fragment_program.glsl") as f:
        fragment_source_code = f.read()

    vert_shader = pyglet.graphics.shader.Shader(vertex_source_code, "vertex")
    frag_shader = pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    pipeline = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)

    def make_gpu(mesh):
        vl = tm.rendering.mesh_to_vertexlist(mesh)
        gpu = pipeline.vertex_list_indexed(len(vl[4][1]) // 3, GL.GL_TRIANGLES, vl[3])
        gpu.position[:] = vl[4][1]
        return gpu

    cube_gpu = make_gpu(cube)
    sphere_gpu = make_gpu(sphere)

    # ---- mundo fisico ----
    world = pymunk.Space()
    world.gravity = (0.0, -28.0)
    world.damping = 0.992

    # estado del juego.
    state = {
        "balls": [],  # (body, shape)
        "score": 0,
        "best": 0,
        "flash": {},  # id(shape) -> ttl en segundos
        "use_perspective": True,
        "left_pressed": False,
        "right_pressed": False,
    }

    walls = []  # (p1, p2, shape) para dibujar

    def add_wall(p1, p2, group=0):
        seg = pymunk.Segment(world.static_body, p1, p2, WALL_RADIUS)
        seg.elasticity = 0.55
        seg.friction = 0.35
        seg.collision_type = COLLISION_TYPE_DEFAULT
        if group != 0:
            seg.filter = pymunk.ShapeFilter(group=group)
        world.add(seg)
        walls.append((p1, p2, seg))

    # cabinete exterior.
    add_wall((-CAB_W, -CAB_H), (-CAB_W, CAB_H))  # izquierda
    add_wall((CAB_W, -CAB_H), (CAB_W, CAB_H))    # derecha
    add_wall((-CAB_W, CAB_H), (CAB_W, CAB_H))    # techo

    # rampas inferiores: van desde las paredes hasta los bordes del drenaje.
    # los flippers pivotean sobre ellas, asi que las marcamos con group=FLIPPER_GROUP
    # para que no colisionen con los flippers (que tambien estan en ese group).
    add_wall(RAMP_LEFT_START, RAMP_LEFT_END, group=FLIPPER_GROUP)
    add_wall(RAMP_RIGHT_START, RAMP_RIGHT_END, group=FLIPPER_GROUP)

    # canal del plunger: pared interior vertical + piso + rampa de salida.
    add_wall((CHANNEL_INNER_X, -CAB_H + 0.5), (CHANNEL_INNER_X, CHANNEL_TOP_Y))
    add_wall((CHANNEL_INNER_X, -CAB_H + 0.5), (CAB_W, -CAB_H + 0.5))
    # rampa de salida: dentro del canal, en el techo, redirige la bola que
    # sube hacia la izquierda (al campo de juego).
    add_wall((CAB_W, 9.0), (CHANNEL_INNER_X, 11.0))

    # ---- drenaje ----
    drain = pymunk.Segment(
        world.static_body,
        (-DRAIN_HALF, -CAB_H + 0.4),
        (DRAIN_HALF, -CAB_H + 0.4),
        0.1,
    )
    drain.sensor = True
    drain.collision_type = COLLISION_TYPE_DRAIN
    world.add(drain)

    # ---- bumpers ----
    bumpers = []  # dicts con offset, radio, shape

    def add_bumper(x, y, radius=0.85):
        shape = pymunk.Circle(world.static_body, radius, offset=(x, y))
        # elasticity 1.0 evita ganancias de energia que dejan a la bola
        # atrapada rebotando dentro del bumper.
        shape.elasticity = 1.0
        shape.friction = 0.0
        shape.collision_type = COLLISION_TYPE_BUMPER
        world.add(shape)
        bumpers.append({"offset": (x, y), "radius": radius, "shape": shape})

    add_bumper(0.0, 5.5, 1.0)
    add_bumper(-2.8, 2.5)
    add_bumper(2.0, 2.5)
    add_bumper(-3.5, 8.5, 0.7)
    add_bumper(3.5, 7.0, 0.7)

    # ---- flippers ----
    def make_flipper(pivot_world, side):
        mass = 1.5
        if side == "left":
            verts = [
                (0.0, -FLIPPER_THICKNESS / 2),
                (FLIPPER_LENGTH, -FLIPPER_THICKNESS / 2),
                (FLIPPER_LENGTH, FLIPPER_THICKNESS / 2),
                (0.0, FLIPPER_THICKNESS / 2),
            ]
            rest_angle = FLIPPER_REST_ANGLE_LEFT
            up_angle = FLIPPER_UP_ANGLE_LEFT
        else:
            verts = [
                (-FLIPPER_LENGTH, -FLIPPER_THICKNESS / 2),
                (0.0, -FLIPPER_THICKNESS / 2),
                (0.0, FLIPPER_THICKNESS / 2),
                (-FLIPPER_LENGTH, FLIPPER_THICKNESS / 2),
            ]
            rest_angle = FLIPPER_REST_ANGLE_RIGHT
            up_angle = FLIPPER_UP_ANGLE_RIGHT

        moment = pymunk.moment_for_poly(mass, verts)
        body = pymunk.Body(mass, moment)
        body.position = pivot_world
        body.angle = rest_angle
        shape = pymunk.Poly(body, verts)
        shape.elasticity = 0.5
        shape.friction = 0.6
        shape.collision_type = COLLISION_TYPE_DEFAULT
        # ShapeFilter group=FLIPPER_GROUP: el flipper no choca con la rampa
        # bajo el pivote (donde se solapan), pero si con la bola.
        shape.filter = pymunk.ShapeFilter(group=FLIPPER_GROUP)
        world.add(body, shape)

        # PivotJoint al cuerpo estatico en el extremo del flipper.
        world.add(pymunk.PivotJoint(world.static_body, body, pivot_world))
        # RotaryLimitJoint limita el rango angular.
        lo, hi = sorted([rest_angle, up_angle])
        world.add(pymunk.RotaryLimitJoint(world.static_body, body, lo, hi))

        return {
            "body": body,
            "shape": shape,
            "pivot_world": pivot_world,
            "rest_angle": rest_angle,
            "up_angle": up_angle,
            "side": side,
        }

    flipper_left = make_flipper((-FLIPPER_PIVOT_DX, FLIPPER_Y), "left")
    flipper_right = make_flipper((FLIPPER_PIVOT_DX, FLIPPER_Y), "right")

    # ---- bola ----
    def spawn_ball():
        if len(state["balls"]) >= 3:
            return
        body = pymunk.Body(
            0.6, pymunk.moment_for_circle(0.6, 0, BALL_RADIUS)
        )
        # spawn al centro del canal del plunger, un poco sobre el piso.
        body.position = ((CHANNEL_INNER_X + CAB_W) / 2.0, -CAB_H + 1.3)
        body.velocity = (0.0, BALL_LAUNCH_VELOCITY)
        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.elasticity = 0.8
        shape.friction = 0.35
        shape.collision_type = COLLISION_TYPE_BALL
        world.add(body, shape)
        state["balls"].append((body, shape))

    # ---- callbacks de colision ----
    def flash(shape, ttl=0.4):
        state["flash"][id(shape)] = ttl

    def on_ball_vs_bumper(arbiter, space, data):
        # solo los bumpers se iluminan. usamos begin (un disparo por contacto)
        # para evitar acumulacion de score o energia mientras dura el contacto.
        for s in arbiter.shapes:
            if s.collision_type == COLLISION_TYPE_BUMPER:
                flash(s)
                state["score"] += 100
        return True  # dejar que pymunk procese el rebote elastico normal

    def on_ball_vs_drain(arbiter, space, data):
        ball = next(
            (s for s in arbiter.shapes if s.collision_type == COLLISION_TYPE_BALL),
            None,
        )
        if ball is not None and ball.body in world.bodies:
            world.remove(ball.body, ball)
            state["balls"] = [
                (b, sh) for (b, sh) in state["balls"] if sh is not ball
            ]
            state["best"] = max(state["best"], state["score"])
            state["score"] = 0
            print("drain. la bola se perdio.")
        return False  # el shape es sensor; nada que cancelar

    world.on_collision(
        COLLISION_TYPE_BALL, COLLISION_TYPE_BUMPER, begin=on_ball_vs_bumper
    )
    world.on_collision(
        COLLISION_TYPE_BALL, COLLISION_TYPE_DRAIN, begin=on_ball_vs_drain
    )

    # ---- input ----
    def update_flipper_drive():
        # estilo cinematico: forzamos angular_velocity cada frame. cuando la
        # tecla esta presionada, alto positivo (hacia arriba); cuando se
        # suelta, negativo moderado (vuelve al reposo). el RotaryLimitJoint
        # impide pasarse de los topes.
        flipper_left["body"].angular_velocity = 28.0 if state["left_pressed"] else -12.0
        flipper_right["body"].angular_velocity = -28.0 if state["right_pressed"] else 12.0

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.LSHIFT:
            state["left_pressed"] = True
        elif symbol == pyglet.window.key.RSHIFT:
            state["right_pressed"] = True
        elif symbol == pyglet.window.key.SPACE:
            spawn_ball()
        elif symbol == pyglet.window.key.R:
            state["score"] = 0
        elif symbol == pyglet.window.key.P:
            state["use_perspective"] = not state["use_perspective"]
            update_projection()
        elif symbol == pyglet.window.key.ESCAPE:
            window.close()

    @window.event
    def on_key_release(symbol, modifiers):
        if symbol == pyglet.window.key.LSHIFT:
            state["left_pressed"] = False
        elif symbol == pyglet.window.key.RSHIFT:
            state["right_pressed"] = False

    # ---- proyeccion y vista ----
    def update_projection():
        aspect = width / height
        if state["use_perspective"]:
            state["projection"] = tr.perspective(45, aspect, 0.1, 200.0)
            state["view"] = tr.lookAt(
                np.array([0.0, -2.0, 38.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )
        else:
            ortho_size = CAB_H + 2.0
            state["projection"] = tr.ortho(
                -ortho_size * aspect,
                ortho_size * aspect,
                -ortho_size,
                ortho_size,
                0.1,
                200.0,
            )
            state["view"] = tr.lookAt(
                np.array([0.0, 0.0, 30.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )

    update_projection()

    # ---- simulacion ----
    dt = 1.0 / 60.0

    def step(_):
        update_flipper_drive()
        # subdivision: varios pasos pequenos por frame para que las bolas
        # rapidas no atraviesen las paredes.
        sub_dt = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            world.step(sub_dt)
        # decay de los flashes.
        for k in list(state["flash"].keys()):
            state["flash"][k] -= dt
            if state["flash"][k] <= 0:
                del state["flash"][k]
        # bolas escapadas (salvavidas).
        survivors = []
        for body, shape in state["balls"]:
            if abs(body.position.x) > 30 or abs(body.position.y) > 40:
                if body in world.bodies:
                    world.remove(body, shape)
            else:
                survivors.append((body, shape))
        state["balls"] = survivors

    pyglet.clock.schedule_interval(step, dt)

    # ---- dibujo ----
    def color_for(shape, base):
        return COLOR_FLASH if id(shape) in state["flash"] else base

    def draw_with(gpu, transform, color):
        pipeline["transform"] = transform.reshape(16, 1, order="F")
        pipeline["instance_color"] = color
        gpu.draw(GL.GL_TRIANGLES)

    def draw_box(center, angle, size_x, size_y, color, size_z=0.5):
        transform = (
            tr.translate(center[0], center[1], 0.0)
            @ tr.rotationZ(angle)
            @ tr.scale(size_x, size_y, size_z)
        )
        draw_with(cube_gpu, transform, color)

    def draw_segment(p1, p2, color):
        cx = (p1[0] + p2[0]) / 2.0
        cy = (p1[1] + p2[1]) / 2.0
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(np.sqrt(dx * dx + dy * dy), 1e-6)
        angle = np.arctan2(dy, dx)
        draw_box((cx, cy), angle, length, 2 * WALL_RADIUS, color)

    def draw_flipper(flipper):
        body = flipper["body"]
        sign = 1.0 if flipper["side"] == "left" else -1.0
        transform = (
            tr.translate(body.position[0], body.position[1], 0.0)
            @ tr.rotationZ(body.angle)
            @ tr.translate(sign * FLIPPER_LENGTH / 2.0, 0.0, 0.0)
            @ tr.scale(FLIPPER_LENGTH, FLIPPER_THICKNESS, 0.5)
        )
        draw_with(cube_gpu, transform, color_for(flipper["shape"], COLOR_FLIPPER))

    def draw_ball(body, shape):
        transform = (
            tr.translate(body.position[0], body.position[1], 0.0)
            @ tr.uniformScale(2 * BALL_RADIUS)
        )
        draw_with(sphere_gpu, transform, color_for(shape, COLOR_BALL))

    @window.event
    def on_draw():
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        window.clear()

        pipeline.use()
        pipeline["view"] = state["view"].reshape(16, 1, order="F")
        pipeline["projection"] = state["projection"].reshape(16, 1, order="F")

        # paredes.
        for p1, p2, shape in walls:
            draw_segment(p1, p2, color_for(shape, COLOR_WALL))

        # bumpers (como cilindros chatos: cubos cuadrados escalados al diametro).
        for b in bumpers:
            draw_box(
                b["offset"],
                0.0,
                2 * b["radius"],
                2 * b["radius"],
                color_for(b["shape"], COLOR_BUMPER),
                size_z=1.0,
            )

        # flippers.
        draw_flipper(flipper_left)
        draw_flipper(flipper_right)

        # bolas (esferas).
        for body, shape in state["balls"]:
            draw_ball(body, shape)

        # HUD.
        info = f"score: {state['score']}   best: {state['best']}   bolas: {len(state['balls'])}/3"
        pyglet.text.Label(
            info,
            font_name="sans-serif",
            font_size=14,
            x=20,
            y=height - 30,
            color=(240, 240, 240, 255),
        ).draw()
        pyglet.text.Label(
            "shift izq/der: flippers   espacio: nueva bola   P: 2D/3D   R: reset",
            font_name="sans-serif",
            font_size=10,
            x=20,
            y=20,
            color=(200, 200, 200, 255),
        ).draw()

    print("Controles:")
    print("- Shift izquierdo: flipper izquierdo")
    print("- Shift derecho:   flipper derecho")
    print("- Espacio:         lanza una bola desde el canal")
    print("- R:               reinicia el puntaje")
    print("- P:               alterna proyeccion 3D / 2D")

    pyglet.app.run()


if __name__ == "__main__":
    pymunk_pinball()
