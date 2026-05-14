"""Ejemplo intermedio de pymunk + pyglet con render 3D.

Toma la simulacion 2D de cuerpos rigidos y la dibuja como un mundo 3D usando
shaders propios. Sobre el ejemplo basico agrega:

- **Restricciones**: un pendulo en cadena (PinJoint) y un par de engranajes
  acoplados (GearJoint + SimpleMotor). Demuestran que pymunk puede construir
  sistemas articulados a partir de cuerpos rigidos.
- **Arrastre con el raton**: en vista 2D ortografica, click izquierdo sobre un
  cuerpo lo "agarra" usando un PivotJoint contra un body cinematico. Soltar
  remueve el joint. En vista 3D el click solo agrega cuerpos.
- **Callback de colision**: cuando un cuerpo dinamico golpea el suelo con
  suficiente impulso, su color se enciende por unos cuadros. Se usa el sistema
  collision_type + on_collision de pymunk.

Controles:
- Click izquierdo: en 2D, agarra el cuerpo bajo el puntero; si no hay cuerpo,
  agrega una caja en posicion aleatoria. En 3D, agrega una caja.
- Click derecho: agrega una caja con velocidad inicial hacia el centro.
- Tecla V: muestra/oculta vectores de velocidad.
- Tecla P: alterna proyeccion perspectiva / ortografica 2D.
- Tecla C: limpia los cuerpos dinamicos (deja el pendulo y los engranajes).
"""

import os
from itertools import chain
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import pymunk
import trimesh as tm

import grafica.transformations as tr
from grafica.ui import ui_overlay


# tipos de colision para identificar pares cuerpo-suelo en los callbacks.
COLLISION_TYPE_DYNAMIC = 1
COLLISION_TYPE_STATIC = 2

# colores reutilizados.
COLOR_BASE = (1.0, 1.0, 1.0)
COLOR_FLASH = (1.0, 0.3, 0.3)
COLOR_CHAIN = (0.6, 0.85, 1.0)
COLOR_GEAR_A = (1.0, 0.85, 0.4)
COLOR_GEAR_B = (0.4, 1.0, 0.6)
COLOR_GRABBED = (1.0, 1.0, 0.2)


@click.command(
    "falling_boxes",
    short_help="pymunk + pyglet con joints, arrastre y callbacks (render 3D)",
)
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=960)
def falling_boxes(width, height):
    window = pyglet.window.Window(width, height)

    pyglet.font.add_file(
        str(
            Path(__file__).parent.parent.parent
            / "assets"
            / "FiraCode"
            / "FiraCode-Regular.ttf"
        )
    )

    # cubo y esfera unitarios para dibujar cuerpos. usamos el cubo para
    # cuerpos pymunk.Poly (cajas, eslabones del pendulo) y la esfera para
    # cuerpos pymunk.Circle (engranajes), asi la malla refleja la primitiva
    # fisica.
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
        gpu = pipeline.vertex_list_indexed(
            len(vl[4][1]) // 3, GL.GL_TRIANGLES, vl[3]
        )
        gpu.position[:] = vl[4][1]
        return gpu

    cube_gpu = make_gpu(cube)
    sphere_gpu = make_gpu(sphere)

    # grilla del suelo (igual que la version original).
    grid_resolution = 100
    xv, yv = np.meshgrid(
        np.linspace(-1, 1, grid_resolution),
        np.linspace(-1, 1, grid_resolution),
        indexing="xy",
    )
    grid_vertices = np.vstack(
        (
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            np.zeros(shape=(1, grid_resolution**2)),
        )
    ).T
    grid_indices = [
        [
            (grid_resolution * row + i, grid_resolution * row + i + 1)
            for i in range(grid_resolution - 1)
        ]
        for row in range(grid_resolution)
    ]
    grid_indices.extend(
        [
            [
                (
                    grid_resolution * column + i,
                    grid_resolution * column + i + grid_resolution,
                )
                for i in range(grid_resolution)
            ]
            for column in range(grid_resolution - 1)
        ]
    )
    grid_indices = list(chain(*chain(*grid_indices)))
    grid_gpu = pipeline.vertex_list_indexed(
        grid_resolution**2, GL.GL_LINES, grid_indices
    )
    grid_gpu.position[:] = grid_vertices.reshape(-1, 1, order="C")

    # mundo fisico.
    world = pymunk.Space()
    world.gravity = (0, -9.81)

    # suelo: segmento estatico. registramos collision_type para el callback.
    groundBody = pymunk.Segment(world.static_body, (-50, -0.05), (50, -0.05), 0.1)
    groundBody.friction = 1.0
    groundBody.collision_type = COLLISION_TYPE_STATIC
    world.add(groundBody)

    # estado del programa: cuerpos dinamicos, formas de joints visibles, etc.
    state = {
        "dynamic_bodies": [],  # cuerpos que el usuario crea con clicks
        "pendulum_bodies": [],  # cuerpos del pendulo (no se borran con C)
        "gear_bodies": [],  # cuerpos de engranajes (no se borran con C)
        "pin_joints": [],  # lista de PinJoint para dibujar
        "flash_ttl": {},  # shape -> segundos de flash restantes
        "drag_joint": None,  # PivotJoint mientras el usuario arrastra
        "total_time": 0.0,
        "show_velocity": True,
        "velocity_gpu": None,
        "use_perspective": True,
        "grid_transform": tr.translate(0, 0, 0)
        @ tr.rotationX(np.pi / 2.0)
        @ tr.uniformScale(100),
    }

    # body cinematico que sigue al raton para implementar el arrastre.
    mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    label_proj = pyglet.text.Label(
        "", font_name="Fira Code", font_size=12, x=20, y=height - 25,
        color=(230, 230, 230, 255),
    )
    label_vel = pyglet.text.Label(
        "", font_name="Fira Code", font_size=12, x=20, y=height - 45,
        color=(230, 230, 230, 255),
    )
    label_bodies = pyglet.text.Label(
        "", font_name="Fira Code", font_size=12, x=20, y=height - 65,
        color=(230, 230, 230, 255),
    )
    label_help = pyglet.text.Label(
        "click izq: agarrar/agregar  click der: agregar  V: vectores  P: 2D/3D  C: limpiar",
        font_name="Fira Code", font_size=10, x=20, y=20,
        color=(200, 200, 200, 255),
    )

    def apply_hud():
        proj_text = "perspectiva" if state["use_perspective"] else "ortografica 2D"
        label_proj.text = f"proyeccion: {proj_text}"
        label_vel.text = f"vectores velocidad: {'on' if state['show_velocity'] else 'off'}"
        n = len(state["dynamic_bodies"])
        label_bodies.text = f"cajas dinamicas: {n}"

    def make_box(mass, size, position, color=COLOR_BASE, collision_type=COLLISION_TYPE_DYNAMIC):
        points = [(-size, -size), (-size, size), (size, size), (size, -size)]
        body = pymunk.Body(mass, pymunk.moment_for_poly(mass, points, (0, 0)))
        body.position = position
        shape = pymunk.Poly(body, points)
        shape.friction = 1.0
        shape.elasticity = 0.2
        shape.collision_type = collision_type
        shape.color_base = color
        shape.box_size = size  # guardamos para dibujar
        world.add(body, shape)
        return body, shape

    def make_circle(mass, radius, position, color=COLOR_BASE):
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.friction = 1.0
        shape.elasticity = 0.4
        shape.collision_type = COLLISION_TYPE_DYNAMIC
        shape.color_base = color
        shape.box_size = radius  # usamos el cubo escalado al radio
        world.add(body, shape)
        return body, shape

    # pendulo en cadena: anclaje estatico arriba y eslabones unidos por PinJoint.
    def build_pendulum(origin=(-7.0, 12.0), n=5, link_size=0.3, gap=0.1):
        prev_body = world.static_body
        prev_anchor_world = origin
        for i in range(n):
            cx = origin[0] + (i + 1) * (2 * link_size + gap)
            cy = origin[1]
            body, shape = make_box(0.5, link_size, (cx, cy), color=COLOR_CHAIN)
            # PinJoint con anchors locales: para el primer eslabon el anchor
            # del cuerpo estatico es la posicion mundial; para los demas, el
            # borde derecho del cuerpo anterior.
            if prev_body is world.static_body:
                anchor_a = origin
            else:
                anchor_a = (link_size, 0)
            anchor_b = (-link_size, 0)
            pj = pymunk.PinJoint(prev_body, body, anchor_a, anchor_b)
            world.add(pj)
            state["pin_joints"].append(pj)
            state["pendulum_bodies"].append((body, shape))
            prev_body = body

    # par de engranajes: dos cuerpos circulares pivoteados en su centro,
    # acoplados con GearJoint (ratio -1 = giran en direcciones opuestas).
    # un SimpleMotor anclado al body estatico mueve al primero a velocidad fija.
    def build_gears(center_a=(6.0, 4.0), center_b=(8.6, 4.0), radius=1.2):
        body_a, shape_a = make_circle(2.0, radius, center_a, color=COLOR_GEAR_A)
        body_b, shape_b = make_circle(2.0, radius, center_b, color=COLOR_GEAR_B)
        # quitamos el collision_type dinamico para que no disparen el callback
        # de impacto contra el suelo (no van a tocarlo, pero por claridad).
        shape_a.collision_type = 0
        shape_b.collision_type = 0
        # los engranajes no chocan entre si (mismo group).
        f = pymunk.ShapeFilter(group=1)
        shape_a.filter = f
        shape_b.filter = f
        # pivots al body estatico para mantenerlos en su lugar pero libres
        # de rotar.
        world.add(pymunk.PivotJoint(world.static_body, body_a, center_a))
        world.add(pymunk.PivotJoint(world.static_body, body_b, center_b))
        # GearJoint acopla las rotaciones. phase=0, ratio=-1.
        world.add(pymunk.GearJoint(body_a, body_b, 0.0, -1.0))
        # SimpleMotor: rotacion fija en rad/s sobre el engranaje A.
        world.add(pymunk.SimpleMotor(world.static_body, body_a, 1.5))
        state["gear_bodies"].extend([(body_a, shape_a), (body_b, shape_b)])

    build_pendulum()
    build_gears()

    # callback de colision: cuerpos dinamicos contra suelo. encendemos un flash
    # en el shape dinamico si el impulso total supera un umbral.
    def on_impact(arbiter, space, data):
        if arbiter.total_impulse.length < 1.5:
            return
        for s in arbiter.shapes:
            if s.body.body_type == pymunk.Body.DYNAMIC:
                state["flash_ttl"][s] = 0.25
                print(f"impacto: shape={s} impulso={arbiter.total_impulse.length:.2f}")

    world.on_collision(
        COLLISION_TYPE_DYNAMIC,
        COLLISION_TYPE_STATIC,
        post_solve=on_impact,
    )

    # mapeos pantalla <-> mundo. solo en 2D ortografica.
    ORTHO_SIZE = 15.0
    CAMERA_TARGET_Y_2D = 8.0

    def screen_to_world_2d(x, y):
        aspect = width / height
        wx = (x / width - 0.5) * 2 * ORTHO_SIZE * aspect
        wy = CAMERA_TARGET_Y_2D + (y / height - 0.5) * 2 * ORTHO_SIZE
        return (wx, wy)

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT and not state["use_perspective"]:
            wx, wy = screen_to_world_2d(x, y)
            info = world.point_query_nearest(
                (wx, wy), 0.0, pymunk.ShapeFilter()
            )
            if info and info.shape.body.body_type == pymunk.Body.DYNAMIC:
                target = info.shape.body
                mouse_body.position = (wx, wy)
                local = target.world_to_local((wx, wy))
                joint = pymunk.PivotJoint(mouse_body, target, (0, 0), local)
                joint.max_force = 50000.0
                joint.error_bias = (1.0 - 0.15) ** 60
                world.add(joint)
                state["drag_joint"] = (joint, info.shape)
                return

        # comportamiento original: agregar caja. spawn dentro del area visible
        # de la camara 3D para que el alumno vea la caja caer.
        mass = 1.0
        size = 0.5
        body, shape = make_box(
            mass, size, (-8 + 16 * np.random.random(), 13), color=COLOR_BASE
        )
        if button == pyglet.window.mouse.RIGHT:
            center_x, center_y = width // 2, height // 2
            body.velocity = (
                (center_x - x) * 0.02,
                (center_y - y) * 0.02,
            )
        state["dynamic_bodies"].append((body, shape))

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if state["drag_joint"] is not None and not state["use_perspective"]:
            mouse_body.position = screen_to_world_2d(x, y)

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        if state["drag_joint"] is not None:
            joint, _ = state["drag_joint"]
            world.remove(joint)
            state["drag_joint"] = None

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.V:
            state["show_velocity"] = not state["show_velocity"]
            print(f"vectores velocidad: {state['show_velocity']}")
            apply_hud()
        elif symbol == pyglet.window.key.C:
            for body, shape in state["dynamic_bodies"]:
                if body in world.bodies:
                    world.remove(body, *body.shapes)
            state["dynamic_bodies"].clear()
            state["flash_ttl"].clear()
            print("cajas dinamicas eliminadas (el pendulo y los engranajes se conservan)")
            apply_hud()
        elif symbol == pyglet.window.key.P:
            state["use_perspective"] = not state["use_perspective"]
            update_projection()
            print(f"proyeccion: {'perspectiva' if state['use_perspective'] else 'ortografica 2D'}")
            apply_hud()

    def create_velocity_vectors():
        vertices = []
        for body, _ in state["dynamic_bodies"] + state["pendulum_bodies"]:
            pos_x, pos_y = body.position
            vel_x, vel_y = body.velocity
            speed = np.sqrt(vel_x**2 + vel_y**2)
            if speed < 0.1:
                continue
            scale = 0.5
            end_x = pos_x + vel_x * scale
            end_y = pos_y + vel_y * scale
            vertices.extend([pos_x, pos_y, 0.0, end_x, end_y, 0.0])
        if not vertices:
            return None
        gpu = pipeline.vertex_list(len(vertices) // 3, GL.GL_LINES)
        gpu.position[:] = vertices
        return gpu

    def create_joint_lines():
        """Linea entre los anchors mundiales de cada PinJoint, para visualizarlo."""
        vertices = []
        for pj in state["pin_joints"]:
            a = pj.a.local_to_world(pj.anchor_a)
            b = pj.b.local_to_world(pj.anchor_b)
            vertices.extend([a.x, a.y, 0.0, b.x, b.y, 0.0])
        if not vertices:
            return None
        gpu = pipeline.vertex_list(len(vertices) // 3, GL.GL_LINES)
        gpu.position[:] = vertices
        return gpu

    time_step = 1.0 / 60.0

    def update_projection():
        if state["use_perspective"]:
            state["projection"] = tr.perspective(50, width / height, 0.1, 200.0)
            state["view"] = tr.lookAt(
                np.array([0.0, 7.0, 24.0]),
                np.array([0.0, 5.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )
        else:
            aspect = width / height
            state["projection"] = tr.ortho(
                -ORTHO_SIZE * aspect,
                ORTHO_SIZE * aspect,
                -ORTHO_SIZE,
                ORTHO_SIZE,
                0.001,
                100.0,
            )
            state["view"] = tr.lookAt(
                np.array([0.0, CAMERA_TARGET_Y_2D, 25]),
                np.array([0, CAMERA_TARGET_Y_2D, 0]),
                np.array([0.0, 1.0, 0]),
            )

    def update_world(dt, window):
        state["total_time"] += dt
        world.step(dt)
        # remover cuerpos que cayeron muy bajo (solo dinamicos creados por click).
        survivors = []
        for body, shape in state["dynamic_bodies"]:
            if body.position.y < -20 and body in world.bodies:
                world.remove(body, *body.shapes)
                state["flash_ttl"].pop(shape, None)
            else:
                survivors.append((body, shape))
        state["dynamic_bodies"] = survivors
        # decay del flash.
        for s in list(state["flash_ttl"].keys()):
            state["flash_ttl"][s] -= dt
            if state["flash_ttl"][s] <= 0:
                del state["flash_ttl"][s]
        state["velocity_gpu"] = create_velocity_vectors()

    update_projection()
    pyglet.clock.schedule_interval(update_world, time_step, window)

    def color_for(shape):
        if state["drag_joint"] is not None and state["drag_joint"][1] is shape:
            return COLOR_GRABBED
        if shape in state["flash_ttl"]:
            return COLOR_FLASH
        return shape.color_base

    def draw_body(body, shape):
        size = shape.box_size
        transform = (
            tr.translate(body.position[0], body.position[1], 0.0)
            @ tr.rotationZ(body.angle)
            @ tr.uniformScale(size * 2)
        )
        pipeline["transform"] = transform.reshape(16, 1, order="F")
        pipeline["instance_color"] = color_for(shape)
        gpu = sphere_gpu if isinstance(shape, pymunk.Circle) else cube_gpu
        gpu.draw(GL.GL_TRIANGLES)

    @window.event
    def on_draw():
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glLineWidth(1.0)
        window.clear()

        pipeline.use()
        pipeline["view"] = state["view"].reshape(16, 1, order="F")
        pipeline["projection"] = state["projection"].reshape(16, 1, order="F")

        # cuerpos del pendulo, engranajes y dinamicos.
        for body, shape in state["pendulum_bodies"]:
            draw_body(body, shape)
        for body, shape in state["gear_bodies"]:
            draw_body(body, shape)
        for body, shape in state["dynamic_bodies"]:
            draw_body(body, shape)

        # lineas de los joints.
        joint_gpu = create_joint_lines()
        if joint_gpu is not None:
            GL.glLineWidth(2.0)
            pipeline["transform"] = tr.identity().reshape(16, 1, order="F")
            pipeline["instance_color"] = (1.0, 1.0, 0.4)
            joint_gpu.draw(GL.GL_LINES)
            GL.glLineWidth(1.0)

        # vectores de velocidad.
        if state["show_velocity"] and state["velocity_gpu"] is not None:
            GL.glLineWidth(3.0)
            pipeline["transform"] = tr.identity().reshape(16, 1, order="F")
            pipeline["instance_color"] = (0.2, 1.0, 1.0)
            state["velocity_gpu"].draw(GL.GL_LINES)
            GL.glLineWidth(1.0)

        # grilla del piso.
        pipeline["transform"] = state["grid_transform"].reshape(16, 1, order="F")
        pipeline["instance_color"] = (0.8, 0.8, 0.8)
        grid_gpu.draw(GL.GL_LINES)

        # HUD 2D sobre la escena 3D. El render 3D quedo en wireframe
        # (GL_LINE), restauramos GL_FILL para que los glifos se rasterizen
        # como triangulos solidos.
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        apply_hud()
        with ui_overlay():
            label_proj.draw()
            label_vel.draw()
            label_bodies.draw()
            label_help.draw()

    print("Controles:")
    print("- Click izquierdo: en 2D agarra un cuerpo bajo el puntero; si no hay, agrega caja.")
    print("- Click derecho: agrega caja con velocidad inicial hacia el centro.")
    print("- Tecla V: muestra/oculta vectores de velocidad.")
    print("- Tecla P: alterna entre 3D (perspectiva) y 2D (ortografica).")
    print("- Tecla C: limpia las cajas (conserva pendulo y engranajes).")

    pyglet.app.run()
