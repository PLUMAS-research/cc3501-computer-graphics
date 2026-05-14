"""Pinball 2D simulado con pymunk y dibujado en 3D con Scenegraph.

Pone en accion los conceptos de la unidad de cuerpos rigidos: cuerpos dinamicos
(la bola), estaticos (paredes y bumpers), articulaciones (los flippers, con
PivotJoint y RotaryLimitJoint) y callbacks de colision (los bumpers se
encienden en amarillo cuando la pelota los toca, ademas suman puntos y empujan
la pelota). El mundo fisico es 2D, pero cada cuerpo se dibuja dentro de un
Scenegraph 3D siguiendo una regla simple: las primitivas pymunk.Poly y
pymunk.Segment se renderizan como cubos escalados, y las pymunk.Circle se
renderizan como esferas.

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
from grafica.scenegraph import Scenegraph
from grafica.scenegraph_nodes import _node_from_mesh
from grafica.ui import ui_overlay


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
DRAIN_HALF = 0.7
FLIPPER_PIVOT_DX = 2.5
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
RAMP_RIGHT_START = (CHANNEL_INNER_X, _line_y_at(CHANNEL_INNER_X, (-RAMP_LEFT_END[0], RAMP_LEFT_END[1]), (-RAMP_LEFT_START[0], RAMP_LEFT_START[1])))

FLIPPER_LENGTH = 2.5
FLIPPER_THICKNESS = 0.32
# en reposo los flippers quedan levantados apuntando al centro (forma /\).
# al apretar shift bajan a la posicion colineal con la rampa (forma V).
FLIPPER_REST_ANGLE_LEFT = 0.55
FLIPPER_REST_ANGLE_RIGHT = -0.55
FLIPPER_PRESSED_ANGLE_LEFT = LEFT_RAMP_ANGLE  # colineal con la rampa
FLIPPER_PRESSED_ANGLE_RIGHT = -LEFT_RAMP_ANGLE
# parametros del resorte que devuelve el flipper a su angulo objetivo. la
# tecla shift cambia el angulo objetivo entre reposo y presionado; el resorte
# se encarga del movimiento. asi la pelota puede empujar al flipper y la
# entrada en el limite no es brusca. la rigidez controla con cuanta fuerza
# el flipper golpea a la pelota: mas rigido = swing mas rapido = impacto mas
# fuerte.
FLIPPER_STIFFNESS = 16000.0
FLIPPER_DAMPING = 320.0
FLIPPER_GROUP = 1

# escenario.
WALL_RADIUS = 0.18
SUBSTEPS = 4  # subdivision de cada frame para evitar tunneling


@click.command("pymunk_pinball", short_help="Pinball 2D con joints y callbacks, render 3D")
@click.option("--width", type=int, default=720)
@click.option("--height", type=int, default=1080)
def pymunk_pinball(width, height):
    window = pyglet.window.Window(width, height, caption="pymunk pinball")

    pyglet.font.add_file(
        str(
            Path(__file__).parent.parent.parent
            / "assets"
            / "FiraCode"
            / "FiraCode-Regular.ttf"
        )
    )

    # ---- grafo de escena ----
    # registramos un cubo y una esfera unitarios. cada cuerpo pymunk se va a
    # dibujar como una instancia de una de las dos mallas con su transform
    # propio; el render queda a cargo del Scenegraph.
    cube = tm.load("assets/cube.off")
    cube.apply_translation(-cube.centroid)
    cube.apply_scale(np.sqrt(3) / cube.scale)

    sphere = tm.load("assets/sphere.off")
    sphere.apply_translation(-sphere.centroid)
    sphere.apply_scale(np.sqrt(3) / sphere.scale)

    graph = Scenegraph("root")
    graph.register_mesh("cube", _node_from_mesh(cube))
    graph.register_mesh("sphere", _node_from_mesh(sphere))
    graph.load_and_register_pipeline(
        "default",
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    # ---- mundo fisico ----
    world = pymunk.Space()
    world.gravity = (0.0, -28.0)
    world.damping = 0.992

    # estado del juego.
    state = {
        "balls": [],  # (body, shape, node_name)
        "ball_counter": 0,
        "score": 0,
        "best": 0,
        "flash": {},  # id(shape) -> ttl en segundos
        "use_perspective": True,
        "left_pressed": False,
        "right_pressed": False,
    }

    walls = []  # (p1, p2, shape) — el render lo maneja el grafo

    def add_wall(p1, p2, group=0):
        seg = pymunk.Segment(world.static_body, p1, p2, WALL_RADIUS)
        seg.elasticity = 0.55
        seg.friction = 0.35
        seg.collision_type = COLLISION_TYPE_DEFAULT
        if group != 0:
            seg.filter = pymunk.ShapeFilter(group=group)
        world.add(seg)
        walls.append((p1, p2, seg))

        # nodo del grafo: un cubo escalado al largo del segmento y rotado al
        # angulo de la recta. el transform es fijo porque las paredes son
        # estaticas.
        cx = (p1[0] + p2[0]) / 2.0
        cy = (p1[1] + p2[1]) / 2.0
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(np.sqrt(dx * dx + dy * dy), 1e-6)
        angle = np.arctan2(dy, dx)
        graph.add_object(
            f"wall_{len(walls) - 1}", "cube", "default", parent="root",
            transform=(
                tr.translate(cx, cy, 0.0)
                @ tr.rotationZ(angle)
                @ tr.scale(length, 2 * WALL_RADIUS, 0.5)
            ),
            instance_color=np.array(COLOR_WALL),
        )

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
    # sensor: dispara on_ball_vs_drain pero no rebota. no se dibuja.
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
    bumpers = []  # cada bumper guarda su shape y el nombre del nodo malla

    def add_bumper(x, y, radius=0.85):
        shape = pymunk.Circle(world.static_body, radius, offset=(x, y))
        # elasticity 1.0 evita ganancias de energia que dejan a la bola
        # atrapada rebotando dentro del bumper.
        shape.elasticity = 1.0
        shape.friction = 0.0
        shape.collision_type = COLLISION_TYPE_BUMPER
        world.add(shape)

        node_name = f"bumper_{len(bumpers)}"
        bumpers.append({
            "offset": (x, y),
            "radius": radius,
            "shape": shape,
            "mesh_node": f"{node_name}_mesh",
        })
        # bumper como esfera: pymunk.Circle se renderiza con la malla esfera.
        graph.add_object(
            node_name, "sphere", "default", parent="root",
            transform=(
                tr.translate(x, y, 0.0)
                @ tr.uniformScale(2 * radius)
            ),
            instance_color=np.array(COLOR_BUMPER),
        )

    add_bumper(0.0, 5.5, 1.0)
    add_bumper(-2.8, 2.5)
    add_bumper(2.0, 2.5)
    add_bumper(-3.5, 8.5, 0.7)
    add_bumper(3.5, 7.0, 0.7)

    # ---- helpers de transform para cuerpos dinamicos ----
    def flipper_transform(flipper):
        body = flipper["body"]
        sign = 1.0 if flipper["side"] == "left" else -1.0
        # z=0.3 lo levanta sobre la superficie del cabinete (paredes y rampas
        # estan en z=0). asi el flipper no pelea por profundidad con la rampa
        # con la que comparte el pivote.
        return (
            tr.translate(body.position[0], body.position[1], 0.3)
            @ tr.rotationZ(body.angle)
            @ tr.translate(sign * FLIPPER_LENGTH / 2.0, 0.0, 0.0)
            @ tr.scale(FLIPPER_LENGTH, FLIPPER_THICKNESS, 0.5)
        )

    def ball_transform(body):
        return (
            tr.translate(body.position[0], body.position[1], 0.0)
            @ tr.uniformScale(2 * BALL_RADIUS)
        )

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
            pressed_angle = FLIPPER_PRESSED_ANGLE_LEFT
        else:
            verts = [
                (-FLIPPER_LENGTH, -FLIPPER_THICKNESS / 2),
                (0.0, -FLIPPER_THICKNESS / 2),
                (0.0, FLIPPER_THICKNESS / 2),
                (-FLIPPER_LENGTH, FLIPPER_THICKNESS / 2),
            ]
            rest_angle = FLIPPER_REST_ANGLE_RIGHT
            pressed_angle = FLIPPER_PRESSED_ANGLE_RIGHT

        moment = pymunk.moment_for_poly(mass, verts)
        body = pymunk.Body(mass, moment)
        body.position = pivot_world
        body.angle = rest_angle
        shape = pymunk.Poly(body, verts)
        shape.elasticity = 0.7
        shape.friction = 0.6
        shape.collision_type = COLLISION_TYPE_DEFAULT
        # ShapeFilter group=FLIPPER_GROUP: el flipper no choca con la rampa
        # bajo el pivote (donde se solapan), pero si con la bola.
        shape.filter = pymunk.ShapeFilter(group=FLIPPER_GROUP)
        world.add(body, shape)

        # PivotJoint al cuerpo estatico en el extremo del flipper.
        world.add(pymunk.PivotJoint(world.static_body, body, pivot_world))
        # RotaryLimitJoint limita el rango angular.
        lo, hi = sorted([rest_angle, pressed_angle])
        world.add(pymunk.RotaryLimitJoint(world.static_body, body, lo, hi))

        # DampedRotarySpring: tira al flipper hacia rest_angle (en reposo) o
        # hacia pressed_angle (mientras shift esta presionado). a diferencia
        # de fijar angular_velocity a mano, esto deja que la pelota empuje al
        # flipper y suaviza el choque contra los limites.
        spring = pymunk.DampedRotarySpring(
            world.static_body, body,
            rest_angle, FLIPPER_STIFFNESS, FLIPPER_DAMPING,
        )
        world.add(spring)

        return {
            "body": body,
            "shape": shape,
            "spring": spring,
            "pivot_world": pivot_world,
            "rest_angle": rest_angle,
            "pressed_angle": pressed_angle,
            "side": side,
        }

    flipper_left = make_flipper((-FLIPPER_PIVOT_DX, FLIPPER_Y), "left")
    flipper_right = make_flipper((FLIPPER_PIVOT_DX, FLIPPER_Y), "right")

    for flipper, name in [(flipper_left, "flipper_left"), (flipper_right, "flipper_right")]:
        flipper["node_name"] = name
        graph.add_object(
            name, "cube", "default", parent="root",
            transform=flipper_transform(flipper),
            instance_color=np.array(COLOR_FLIPPER),
        )

    # ---- bola ----
    def remove_ball_from_graph(name):
        if name in graph.nodes:
            graph.remove_nodes_from([name, f"{name}_mesh"])

    def spawn_ball():
        if len(state["balls"]) >= 3:
            return
        body = pymunk.Body(
            0.6, pymunk.moment_for_circle(0.6, 0, BALL_RADIUS)
        )
        # spawn al centro del canal del plunger, un poco sobre el piso.
        # se agrega una pequena perturbacion en posicion, velocidad vertical
        # y giro inicial para que dos lanzamientos consecutivos no sigan la
        # misma trayectoria. los rangos son chicos: la pelota entra al campo
        # desde el canal igual, pero rebota distinto contra la rampa de
        # salida.
        spawn_x = (CHANNEL_INNER_X + CAB_W) / 2.0 + np.random.uniform(-0.12, 0.12)
        body.position = (spawn_x, -CAB_H + 1.3)
        body.velocity = (
            np.random.uniform(-0.4, 0.4),
            BALL_LAUNCH_VELOCITY * np.random.uniform(0.95, 1.05),
        )
        body.angular_velocity = np.random.uniform(-2.0, 2.0)
        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.elasticity = 0.8
        shape.friction = 0.35
        shape.collision_type = COLLISION_TYPE_BALL
        world.add(body, shape)

        state["ball_counter"] += 1
        node_name = f"ball_{state['ball_counter']}"
        graph.add_object(
            node_name, "sphere", "default", parent="root",
            transform=ball_transform(body),
            instance_color=np.array(COLOR_BALL),
        )
        state["balls"].append((body, shape, node_name))

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
        if ball is None or ball.body not in world.bodies:
            return False
        for body, shape, node_name in state["balls"]:
            if shape is ball:
                world.remove(body, shape)
                remove_ball_from_graph(node_name)
                break
        state["balls"] = [
            entry for entry in state["balls"] if entry[1] is not ball
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
    def drive_flipper(flipper, pressed):
        flipper["spring"].rest_angle = (
            flipper["pressed_angle"] if pressed else flipper["rest_angle"]
        )

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.LSHIFT:
            state["left_pressed"] = True
            drive_flipper(flipper_left, True)
        elif symbol == pyglet.window.key.RSHIFT:
            state["right_pressed"] = True
            drive_flipper(flipper_right, True)
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
            drive_flipper(flipper_left, False)
        elif symbol == pyglet.window.key.RSHIFT:
            state["right_pressed"] = False
            drive_flipper(flipper_right, False)

    # ---- proyeccion y vista ----
    def update_projection():
        aspect = width / height
        if state["use_perspective"]:
            projection = tr.perspective(50, aspect, 0.1, 200.0)
            # camara delante del cabinete (y muy negativo) y elevada (z alto),
            # apuntando al centro de la mesa. la linea de vision queda a 45
            # grados respecto al plano del cabinete, parecido a la perspectiva
            # de un pinball real visto por un jugador. con target=(0,0,0) y
            # eye=(0,-25,25) entran tanto el drenaje (y=-14) como el techo
            # (y=14) dentro del FOV de 50 grados.
            view = tr.lookAt(
                np.array([0.0, -25.0, 25.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )
        else:
            ortho_size = CAB_H + 2.0
            projection = tr.ortho(
                -ortho_size * aspect,
                ortho_size * aspect,
                -ortho_size,
                ortho_size,
                0.1,
                200.0,
            )
            view = tr.lookAt(
                np.array([0.0, 0.0, 30.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )
        graph.register_view_transform(view)
        graph.set_global_attributes(projection=projection)

    update_projection()

    # ---- simulacion ----
    dt = 1.0 / 60.0

    def step(_):
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
        for body, shape, node_name in state["balls"]:
            if abs(body.position.x) > 30 or abs(body.position.y) > 40:
                if body in world.bodies:
                    world.remove(body, shape)
                remove_ball_from_graph(node_name)
            else:
                survivors.append((body, shape, node_name))
        state["balls"] = survivors

    pyglet.clock.schedule_interval(step, dt)

    # ---- render ----
    def update_dynamic_state():
        # transforms de cuerpos que se mueven cada frame.
        for flipper in (flipper_left, flipper_right):
            graph.nodes[flipper["node_name"]]["transform"] = flipper_transform(flipper)
        for body, shape, node_name in state["balls"]:
            graph.nodes[node_name]["transform"] = ball_transform(body)
        # los bumpers cambian de color cuando se prenden tras un impacto.
        for b in bumpers:
            color = COLOR_FLASH if id(b["shape"]) in state["flash"] else COLOR_BUMPER
            graph.apply_instance_attributes(b["mesh_node"], instance_color=np.array(color))

    @window.event
    def on_draw():
        GL.glClearColor(0.08, 0.08, 0.10, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        window.clear()

        update_dynamic_state()
        graph.render()

        # HUD 2D sobre la escena 3D.
        info = f"score: {state['score']}   best: {state['best']}   bolas: {len(state['balls'])}/3"
        with ui_overlay():
            pyglet.text.Label(
                info,
                font_name="Fira Code",
                font_size=14,
                x=20,
                y=height - 30,
                color=(240, 240, 240, 255),
            ).draw()
            pyglet.text.Label(
                "shift izq/der: flippers   espacio: nueva bola   P: 2D/3D   R: reset",
                font_name="Fira Code",
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
