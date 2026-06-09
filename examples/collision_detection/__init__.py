"""Dino runner: detección de colisiones AABB en 2D, con las cajas a la vista.

El ejemplo atómico de la unidad de colisiones. Un dinosaurio corre hacia la
derecha (en pantalla, son los obstáculos los que se desplazan hacia la
izquierda) y salta cactus. La colisión se decide con la prueba AABB
(*Axis-Aligned Bounding Box*): dos cajas alineadas a los ejes se intersectan
si y solo si sus intervalos se solapan en cada eje.

La prueba queda a la vista: cada entidad dibuja su caja contenedora y el
panel muestra el solape por eje contra el obstáculo más cercano. Cuando hay
solape en un solo eje la caja del obstáculo se pinta amarilla; cuando hay
solape en ambos ejes hay colisión (rojo) y el juego termina.

Controles:
- Espacio: saltar.
- P: pausar (útil para inspeccionar las cajas).
- R: reiniciar.
- ESC: cerrar.
"""

import random
from pathlib import Path

import click
import pyglet
import pyglet.gl as GL

from grafica.ui import InfoPanel

# mundo en píxeles.
NIVEL_SUELO = 52.0             # altura de la superficie donde corre el dinosaurio
GRAVEDAD = 2600.0
VELOCIDAD_SALTO = 950.0
VELOCIDAD_INICIAL = 320.0      # rapidez de los obstáculos hacia la izquierda
ACELERACION = 12.0             # la rapidez crece con el tiempo jugado
DINO_X = 150.0                 # el dinosaurio no se mueve en pantalla
DINO_SEMIALTO = 38.0

# brecha entre obstáculos: lo suficiente para que el salto siempre alcance.
# el tiempo en el aire es 2 * VELOCIDAD_SALTO / GRAVEDAD; la brecha mínima
# debe superar la distancia que recorre un obstáculo en ese tiempo.
TIEMPO_EN_EL_AIRE = 2 * VELOCIDAD_SALTO / GRAVEDAD

COLOR_CAJA = (70, 70, 90, 255)
COLOR_SOLAPE_PARCIAL = (240, 210, 60, 255)
COLOR_COLISION = (240, 70, 70, 255)
COLOR_CACTUS = (90, 170, 90)
COLOR_SUELO = (110, 100, 90)


def intervalos_se_solapan(minimo_a, maximo_a, minimo_b, maximo_b):
    """Dos intervalos 1D se solapan si cada uno empieza antes de que el otro termine."""
    return minimo_a <= maximo_b and minimo_b <= maximo_a


def prueba_aabb(caja_a, caja_b):
    """Prueba AABB por eje: hay colisión si los intervalos se solapan en x Y en y.

    Cada caja es un dict con centro (x, y), semiancho y semialto. Devuelve el
    solape de cada eje por separado: basta que un eje esté separado para
    descartar la colisión (la versión más simple del teorema del eje
    separador).
    """
    solape_x = intervalos_se_solapan(
        caja_a["x"] - caja_a["semiancho"], caja_a["x"] + caja_a["semiancho"],
        caja_b["x"] - caja_b["semiancho"], caja_b["x"] + caja_b["semiancho"],
    )
    solape_y = intervalos_se_solapan(
        caja_a["y"] - caja_a["semialto"], caja_a["y"] + caja_a["semialto"],
        caja_b["y"] - caja_b["semialto"], caja_b["y"] + caja_b["semialto"],
    )
    return solape_x, solape_y


@click.command("dino_runner", short_help="Detección de colisiones AABB en 2D")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=540)
def dino_runner(width, height):
    window = pyglet.window.Window(width, height, caption="dino runner")

    # el dinosaurio es un sprite; su caja contenedora es un poco más chica
    # que el dibujo, porque los píxeles transparentes del borde no deberían
    # contar como cuerpo del personaje.
    imagen_dino = pyglet.image.load(
        str(Path(__file__).parent.parent.parent / "assets" / "dinosaur.png")
    )
    sprite_dino = pyglet.sprite.Sprite(imagen_dino)
    sprite_dino.scale = 2 * DINO_SEMIALTO / imagen_dino.height

    dino = {
        "x": DINO_X,
        "y": NIVEL_SUELO + DINO_SEMIALTO,  # centro de la caja
        "semiancho": sprite_dino.width * 0.38,
        "semialto": DINO_SEMIALTO * 0.92,
        "velocidad_y": 0.0,
        "en_el_aire": False,
    }

    state = {
        "tiempo": 0.0,
        "velocidad": VELOCIDAD_INICIAL,
        "pausado": False,
        "terminado": False,
        "obstaculos": [],  # cada uno: caja AABB + su rectángulo de pyglet
        "mejor": 0.0,
    }

    batch_suelo = pyglet.graphics.Batch()
    suelo = pyglet.shapes.Rectangle(
        0, 0, width, NIVEL_SUELO, color=COLOR_SUELO, batch=batch_suelo
    )

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(20, 20, 20), background_width=460)
        .add("puntaje").add("velocidad").add("prueba")
        .footer("espacio saltar   P pausa   R reiniciar")
    )

    def crear_obstaculo(x):
        """Un cactus: rectángulo de tamaño aleatorio apoyado en el suelo."""
        ancho = random.uniform(20.0, 45.0)
        alto = random.uniform(35.0, 70.0)
        caja = {
            "x": x,
            "y": NIVEL_SUELO + alto / 2,
            "semiancho": ancho / 2,
            "semialto": alto / 2,
            "solape_x": False,
            "solape_y": False,
        }
        caja["figura"] = pyglet.shapes.Rectangle(
            x - ancho / 2, NIVEL_SUELO, ancho, alto, color=COLOR_CACTUS
        )
        return caja

    def brecha_minima():
        # distancia que recorre un obstáculo durante un salto completo, con margen.
        return state["velocidad"] * TIEMPO_EN_EL_AIRE + 120.0

    def reset():
        state["mejor"] = max(state["mejor"], state["tiempo"])
        state["tiempo"] = 0.0
        state["velocidad"] = VELOCIDAD_INICIAL
        state["terminado"] = False
        state["pausado"] = False
        state["obstaculos"].clear()
        dino["y"] = NIVEL_SUELO + DINO_SEMIALTO
        dino["velocidad_y"] = 0.0
        dino["en_el_aire"] = False
        print(f"[dino] nueva partida (mejor tiempo: {state['mejor']:.1f} s)")

    def actualizar(dt):
        if state["pausado"] or state["terminado"]:
            return
        state["tiempo"] += dt
        state["velocidad"] = VELOCIDAD_INICIAL + ACELERACION * state["tiempo"]

        # salto con caída por gravedad.
        if dino["en_el_aire"]:
            dino["velocidad_y"] -= GRAVEDAD * dt
            dino["y"] += dino["velocidad_y"] * dt
            if dino["y"] <= NIVEL_SUELO + DINO_SEMIALTO:
                dino["y"] = NIVEL_SUELO + DINO_SEMIALTO
                dino["velocidad_y"] = 0.0
                dino["en_el_aire"] = False

        # los obstáculos avanzan hacia la izquierda y desaparecen al salir.
        for obstaculo in state["obstaculos"]:
            obstaculo["x"] -= state["velocidad"] * dt
            obstaculo["figura"].x = obstaculo["x"] - obstaculo["semiancho"]
        state["obstaculos"] = [
            obstaculo for obstaculo in state["obstaculos"] if obstaculo["x"] > -100.0
        ]

        # se agrega un obstáculo nuevo cuando el último ya dejó la brecha libre.
        ultimo_x = max(
            (obstaculo["x"] for obstaculo in state["obstaculos"]), default=-1e9
        )
        if ultimo_x < width - brecha_minima():
            state["obstaculos"].append(
                crear_obstaculo(width + random.uniform(50.0, 250.0))
            )

        # prueba de colisión contra cada obstáculo. el resultado por eje se
        # guarda para colorear las cajas en pantalla.
        for obstaculo in state["obstaculos"]:
            solape_x, solape_y = prueba_aabb(dino, obstaculo)
            obstaculo["solape_x"] = solape_x
            obstaculo["solape_y"] = solape_y
            if solape_x and solape_y:
                state["terminado"] = True
                print(
                    f"[dino] colisión tras {state['tiempo']:.1f} s: "
                    "solape en x y en y a la vez"
                )

    pyglet.clock.schedule_interval(actualizar, 1.0 / 60.0)

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.SPACE and not dino["en_el_aire"]:
            dino["velocidad_y"] = VELOCIDAD_SALTO
            dino["en_el_aire"] = True
        elif symbol == pyglet.window.key.P:
            state["pausado"] = not state["pausado"]
            print(f"[dino] pausa: {state['pausado']}")
        elif symbol == pyglet.window.key.R:
            reset()
        elif symbol == pyglet.window.key.ESCAPE:
            window.close()

    def contorno_de_caja(caja, color):
        """Contorno de la caja AABB de una entidad."""
        return pyglet.shapes.Box(
            caja["x"] - caja["semiancho"],
            caja["y"] - caja["semialto"],
            2 * caja["semiancho"],
            2 * caja["semialto"],
            thickness=2,
            color=color,
        )

    def actualizar_panel():
        panel["puntaje"] = (
            f"tiempo: {state['tiempo']:5.1f} s   mejor: {state['mejor']:5.1f} s"
        )
        panel["velocidad"] = f"velocidad: {state['velocidad']:.0f} px/s"
        # la prueba AABB contra el obstáculo más cercano por delante.
        proximo = min(
            (
                obstaculo for obstaculo in state["obstaculos"]
                if obstaculo["x"] + obstaculo["semiancho"]
                >= dino["x"] - dino["semiancho"]
            ),
            key=lambda obstaculo: obstaculo["x"],
            default=None,
        )
        if state["terminado"]:
            panel["prueba"] = "¡colisión! solape en x y en y (R reinicia)"
            panel.color("prueba", COLOR_COLISION)
        elif proximo is not None:
            solape_x, solape_y = prueba_aabb(dino, proximo)
            panel["prueba"] = (
                f"próximo obstáculo: solape x {'sí' if solape_x else 'no'}, "
                f"solape y {'sí' if solape_y else 'no'}"
            )
            panel.color(
                "prueba",
                COLOR_SOLAPE_PARCIAL if solape_x else InfoPanel.TEXTO,
            )
        else:
            panel["prueba"] = "sin obstáculos a la vista"
            panel.color("prueba", InfoPanel.TEXTO)

    @window.event
    def on_draw():
        # fondo claro: el dinosaurio del asset es dibujo de línea negra.
        GL.glClearColor(0.9, 0.92, 0.95, 1.0)
        window.clear()
        batch_suelo.draw()

        # el sprite se posiciona desde la caja (su ancla es la esquina
        # inferior izquierda de la imagen).
        sprite_dino.x = dino["x"] - sprite_dino.width / 2
        sprite_dino.y = dino["y"] - DINO_SEMIALTO
        sprite_dino.draw()

        # cajas contenedoras: la de cada obstáculo coloreada según el
        # resultado de la prueba por eje, y la del dinosaurio.
        colision = False
        for obstaculo in state["obstaculos"]:
            obstaculo["figura"].draw()
            # mientras el dinosaurio corre por el suelo casi siempre hay
            # solape en y, así que el color destaca el eje x: amarillo es
            # "solapados en x pero separados en y" (saltando por encima).
            if obstaculo["solape_x"] and obstaculo["solape_y"]:
                color = COLOR_COLISION
                colision = True
            elif obstaculo["solape_x"]:
                color = COLOR_SOLAPE_PARCIAL
            else:
                color = COLOR_CAJA
            contorno_de_caja(obstaculo, color).draw()
        contorno_de_caja(dino, COLOR_COLISION if colision else COLOR_CAJA).draw()

        actualizar_panel()
        panel.draw()

    print("[dino] espacio salta, P pausa, R reinicia")
    pyglet.app.run()


if __name__ == "__main__":
    dino_runner()
