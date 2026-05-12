"""Ejemplo basico de pymunk + pyglet.

Cubre los elementos minimos de una simulacion 2D de cuerpos rigidos:
- Espacio fisico con gravedad.
- Cuerpos estaticos (suelo y paredes).
- Cuerpos dinamicos (circulos y cajas) con masa y momento de inercia.
- Friccion y elasticidad (coeficiente de restitucion).
- Dibujo de cada cuerpo leyendo position y angle desde pymunk.

Controles:
- Click izquierdo: agrega un circulo en la posicion del puntero.
- Click derecho:  agrega una caja en la posicion del puntero.
- Tecla R:        reinicia la simulacion.
- Tecla G:        cambia el signo de la gravedad.
- Tecla ESC:      cierra la ventana.
"""

import math
import random

import click
import pyglet
import pymunk


def color_aleatorio():
    """Devuelve una tupla RGB pasable a pyglet.shapes."""
    return (
        random.randint(80, 255),
        random.randint(80, 255),
        random.randint(80, 255),
    )


def crear_circulo(space, x, y, radio=20.0, masa=1.0):
    """Crea un cuerpo dinamico circular y lo agrega al espacio."""
    # moment_for_circle(masa, radio_interior, radio_exterior)
    # un disco solido tiene radio interior 0.
    inercia = pymunk.moment_for_circle(masa, 0.0, radio)
    cuerpo = pymunk.Body(masa, inercia)
    cuerpo.position = (x, y)

    forma = pymunk.Circle(cuerpo, radio)
    forma.friction = 0.7
    forma.elasticity = 0.6
    forma.color_render = color_aleatorio()  # atributo para uso del programa
    space.add(cuerpo, forma)
    return cuerpo, forma


def crear_caja(space, x, y, lado=40.0, masa=1.0):
    """Crea un cuerpo dinamico con forma de caja y lo agrega al espacio."""
    inercia = pymunk.moment_for_box(masa, (lado, lado))
    cuerpo = pymunk.Body(masa, inercia)
    cuerpo.position = (x, y)
    # un pequeno angulo inicial aleatorio hace mas visible la rotacion.
    cuerpo.angle = random.uniform(-0.3, 0.3)

    forma = pymunk.Poly.create_box(cuerpo, (lado, lado))
    forma.friction = 0.7
    forma.elasticity = 0.4
    forma.color_render = color_aleatorio()
    space.add(cuerpo, forma)
    return cuerpo, forma


def crear_paredes(space, ancho, alto, grosor=10.0):
    """Crea suelo y paredes laterales como segmentos estaticos."""
    estaticos = [
        # suelo
        pymunk.Segment(space.static_body, (0, grosor), (ancho, grosor), grosor),
        # pared izquierda
        pymunk.Segment(space.static_body, (grosor, 0), (grosor, alto), grosor),
        # pared derecha
        pymunk.Segment(
            space.static_body, (ancho - grosor, 0), (ancho - grosor, alto), grosor
        ),
    ]
    for seg in estaticos:
        seg.friction = 0.9
        seg.elasticity = 0.5
    space.add(*estaticos)
    return estaticos


@click.command("pymunk_basico", short_help="Ejemplo introductorio de pymunk + pyglet")
@click.option("--width", type=int, default=900)
@click.option("--height", type=int, default=700)
def pymunk_basico(width, height):
    window = pyglet.window.Window(width, height, caption="pymunk basico")

    # 1) creamos el espacio fisico.
    #    la gravedad esta en pixeles por segundo al cuadrado.
    #    valores tipicos: 500 a 1200 para escenas que se sientan "terrenales".
    space = pymunk.Space()
    space.gravity = (0.0, -900.0)

    # 2) suelo y paredes estaticos.
    crear_paredes(space, width, height)

    # 3) listas de cuerpos dinamicos. guardamos ambos (cuerpo y forma) porque
    #    sus posiciones y angulos se actualizan en cada paso.
    cuerpos_circulo = []
    cuerpos_caja = []

    def reset():
        for cuerpo, forma in cuerpos_circulo + cuerpos_caja:
            space.remove(cuerpo, forma)
        cuerpos_circulo.clear()
        cuerpos_caja.clear()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            cuerpos_circulo.append(crear_circulo(space, x, y))
        elif button == pyglet.window.mouse.RIGHT:
            cuerpos_caja.append(crear_caja(space, x, y))

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.R:
            reset()
        elif symbol == pyglet.window.key.G:
            gx, gy = space.gravity
            space.gravity = (gx, -gy)
            # despertamos a los cuerpos para que reaccionen a la nueva gravedad.
            for cuerpo, _ in cuerpos_circulo + cuerpos_caja:
                cuerpo.activate()
        elif symbol == pyglet.window.key.ESCAPE:
            window.close()

    # 4) integracion: paso fijo de 1/60 s.
    #    usar un dt fijo es importante para que la simulacion sea estable y
    #    reproducible. el motor usa semi-implicit Euler internamente.
    dt = 1.0 / 60.0

    def actualizar(_):
        space.step(dt)

    pyglet.clock.schedule_interval(actualizar, dt)

    # 5) dibujo. usamos pyglet.shapes (sin shaders propios).
    #    creamos un batch que agrupa las primitivas y las dibuja juntas.
    @window.event
    def on_draw():
        window.clear()
        batch = pyglet.graphics.Batch()
        # guardamos las primitivas vivas para que el batch las dibuje.
        primitivas = []

        # paredes y suelo: las pintamos como rectangulos grises.
        gris = (90, 90, 90)
        primitivas.append(
            pyglet.shapes.Rectangle(0, 0, width, 20, color=gris, batch=batch)
        )
        primitivas.append(
            pyglet.shapes.Rectangle(0, 0, 20, height, color=gris, batch=batch)
        )
        primitivas.append(
            pyglet.shapes.Rectangle(
                width - 20, 0, 20, height, color=gris, batch=batch
            )
        )

        # circulos: leemos posicion de pymunk. para mostrar la rotacion
        # dibujamos un radio desde el centro hasta el borde.
        for cuerpo, forma in cuerpos_circulo:
            cx, cy = cuerpo.position
            r = forma.radius
            primitivas.append(
                pyglet.shapes.Circle(
                    cx, cy, r, color=forma.color_render, batch=batch
                )
            )
            ex = cx + r * math.cos(cuerpo.angle)
            ey = cy + r * math.sin(cuerpo.angle)
            primitivas.append(
                pyglet.shapes.Line(cx, cy, ex, ey, thickness=2, color=(20, 20, 20), batch=batch)
            )

        # cajas: pyglet.shapes.Rectangle soporta rotacion en grados.
        # el anchor se fija al centro para que la rotacion sea natural.
        for cuerpo, forma in cuerpos_caja:
            cx, cy = cuerpo.position
            # el lado se recupera de los vertices locales.
            vs = forma.get_vertices()
            lado = (vs[1] - vs[0]).length
            rect = pyglet.shapes.Rectangle(
                cx, cy, lado, lado, color=forma.color_render, batch=batch
            )
            rect.anchor_x = lado / 2
            rect.anchor_y = lado / 2
            rect.rotation = -math.degrees(cuerpo.angle)
            primitivas.append(rect)

        # informacion en pantalla.
        info = (
            "click izq: circulo  |  click der: caja  |  "
            "R: reset  |  G: invertir gravedad  |  "
            f"cuerpos: {len(cuerpos_circulo) + len(cuerpos_caja)}"
        )
        label = pyglet.text.Label(
            info,
            font_name="sans-serif",
            font_size=11,
            x=30,
            y=height - 25,
            color=(220, 220, 220, 255),
            batch=batch,
        )
        primitivas.append(label)

        batch.draw()

    pyglet.app.run()


if __name__ == "__main__":
    pymunk_basico()
