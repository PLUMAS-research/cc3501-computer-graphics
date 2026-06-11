import click
import numpy as np
import pyglet
import pyglet.gl as GL

from grafica.ui import InfoPanel, ui_overlay
from .trazado import Esfera, Plano, Camara, trazar
from . import render_progresivo


# Anchos disponibles para el render por CPU (las teclas , . recorren la lista).
# A mayor resolucion se aprecia mejor el detalle, pero el trazado por CPU es
# lento, por eso se calcula de forma progresiva (ver RenderProgresivo).
RESOLUCIONES = [160, 240, 360, 500, 700, 900]


def construir_escena():
    """Tres esferas de colores sobre un piso a cuadros."""
    return [
        Esfera([-1.1, 0.0, 4.2], 1.0, [0.85, 0.20, 0.20], reflectividad=0.35),
        Esfera([1.1, -0.2, 3.4], 0.8, [0.20, 0.45, 0.90], reflectividad=0.35),
        Esfera([0.1, -0.5, 2.4], 0.5, [0.30, 0.80, 0.35], reflectividad=0.5),
        Plano(
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            color_a=[0.95, 0.95, 0.95],
            color_b=[0.15, 0.15, 0.15],
            reflectividad=0.15,
            escala_tablero=0.6,
        ),
    ]


@click.command("raytracing_basico", short_help="Ray tracing atomico sobre esferas y un plano")
@click.option("--width", type=int, default=900)
@click.option("--height", type=int, default=675)
@click.option("--resolucion", type=int, default=200, help="Ancho en pixeles del render por CPU")
def raytracing_basico(width, height, resolucion):
    window = pyglet.window.Window(width, height, caption="Ray tracing basico")

    aspecto = width / height

    escena = construir_escena()
    camara = Camara(
        posicion=[0.0, 0.6, -1.5],
        objetivo=[0.0, 0.0, 3.5],
        arriba_mundo=[0.0, 1.0, 0.0],
        fov_grados=45.0,
        aspecto=aspecto,
    )
    luz = {
        "posicion": np.array([5.0, 6.0, -3.0]),
        "color": np.array([1.0, 1.0, 1.0]),
    }

    state = {"sombras": True, "rebotes": 2}

    def trazar_banda(ancho, alto, fila_inicio, fila_fin):
        return trazar(
            escena, camara, luz, ancho, alto,
            usar_sombras=state["sombras"], rebotes=state["rebotes"],
            fila_inicio=fila_inicio, fila_fin=fila_fin,
        )

    estado_render = render_progresivo.nuevo_estado(
        trazar_banda, RESOLUCIONES, aspecto, resolucion_inicial=resolucion
    )

    def perillas():
        return (state["sombras"], state["rebotes"])

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(15, 15, 20))
        .add("resolucion")
        .add("sombras")
        .add("rebotes")
        .add("tiempo")
        .footer("S sombras   R rebotes   , . resolucion   espacio recalcular")
    )

    def actualizar_panel():
        ancho_render, alto_render = render_progresivo.resolucion_actual(estado_render)
        panel["resolucion"] = f"resolucion: {ancho_render} x {alto_render}"
        panel["sombras"] = f"sombras: {'ON' if state['sombras'] else 'off'}"
        panel["rebotes"] = f"rebotes de reflexion: {state['rebotes']}"
        if estado_render["activo"]:
            panel["tiempo"] = f"calculando... {render_progresivo.porcentaje(estado_render)}%"
        else:
            panel["tiempo"] = f"tiempo de calculo: {estado_render['segundos']:.2f} s"

    def avanzar_render(dt):
        if not estado_render["activo"]:
            return
        termino = render_progresivo.avanzar(estado_render, dt)
        actualizar_panel()
        if termino:
            print(
                f"[raytracing] {estado_render['ancho']}x{estado_render['alto']} "
                f"sombras={state['sombras']} rebotes={state['rebotes']} "
                f"tiempo={estado_render['segundos']:.2f}s"
            )

    render_progresivo.iniciar(estado_render, perillas())
    actualizar_panel()

    @window.event
    def on_draw():
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        window.clear()
        if estado_render["textura"] is not None:
            estado_render["textura"].blit(0, 0, width=width, height=height)
        with ui_overlay():
            panel.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.S:
            state["sombras"] = not state["sombras"]
            render_progresivo.mostrar(estado_render, perillas())
        elif symbol == pyglet.window.key.R:
            state["rebotes"] = (state["rebotes"] + 1) % 4
            render_progresivo.mostrar(estado_render, perillas())
        elif symbol == pyglet.window.key.COMMA:
            render_progresivo.menos_resolucion(estado_render)
            render_progresivo.mostrar(estado_render, perillas())
        elif symbol == pyglet.window.key.PERIOD:
            render_progresivo.mas_resolucion(estado_render)
            render_progresivo.mostrar(estado_render, perillas())
        elif symbol == pyglet.window.key.SPACE:
            # espacio fuerza un recalculo aunque la imagen este cacheada
            render_progresivo.iniciar(estado_render, perillas())
        actualizar_panel()

    pyglet.clock.schedule_interval(avanzar_render, 1 / 60.0)
    pyglet.app.run()
