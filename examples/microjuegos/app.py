"""Anfitrión del proyecto colectivo de microjuegos.

Un microjuego dura unos pocos segundos, abre con una instrucción corta y se
gana o se pierde. El anfitrión los encadena: lleva las vidas, el puntaje, la
velocidad y la dificultad, y entre un microjuego y otro muestra un interludio
y una transición. Cada pieza es un paquete de Python que declara su ROL y
cumple el contrato de ese rol, que está descrito en el enunciado de la tarea 3.

El anfitrión es dueño de la ventana, del reloj y de la entrada. Las piezas no
registran eventos ni programan funciones en el reloj: reciben el tiempo y la
entrada como argumentos.

Un microjuego o un jefe solo describe su juego. El anfitrión crea el espacio de
pymunk de cada ronda y lo avanza, y dibuja la ventana con el fragment program
del juego y los valores que entrega su función uniformes(). Un juego que
necesita otra cosa, como una escena 3D, define su propia función dibujar().
"""

import importlib
import math
import random
import re
import traceback
from pathlib import Path

import click
import pyglet
import pyglet.gl as GL
import pymunk
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4
from pyglet.window import key, mouse

from grafica.ui import load_firacode, ui_overlay

ANCHO, ALTO = 800, 600
VIDAS = 4

# física de los juegos: un espacio de pymunk por ronda, en píxeles de la ventana
GRAVEDAD = (0.0, -900.0)
# el paso de cada cuadro se divide en subpasos de a lo más este largo, porque
# un paso largo hace que los cuerpos rápidos atraviesen a los demás
PASO_FISICA = 1 / 240
DURACION_MICROJUEGO = 4.0  # si el microjuego no define DURACION

# duraciones en segundos de juego; la velocidad las acorta en tiempo real
DURACION_INTERLUDIO = 1.2
DURACION_INSTRUCCION = 0.8
DURACION_MAXIMA_JEFE = 60.0
# la transición dura lo mismo a cualquier velocidad
DURACION_TRANSICION = 0.4

RONDAS_POR_ACELERACION = 4
FACTOR_ACELERACION = 1.15
RONDAS_POR_JEFE = 12

CONTRATO = {
    "microjuego": ("INSTRUCCION", "GANA_SI_SOBREVIVE", "iniciar", "actualizar"),
    "jefe": ("INSTRUCCION", "iniciar", "actualizar"),
    "interludio": ("preparar", "iniciar", "actualizar", "dibujar"),
    "transicion": ("preparar", "dibujar"),
}
JUEGOS = ("microjuego", "jefe")

# Para los juegos que dibujan con uniformes(): un rectángulo que cubre la
# ventana y le entrega a cada píxel su posición en píxeles de la ventana, las
# mismas unidades del mouse y del espacio de pymunk.
VERTEX_PANTALLA = """
#version 330
in vec2 position;
uniform vec2 resolucion;
out vec2 pixel;
void main()
{
    pixel = (position + 1.0) / 2.0 * resolucion;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# se usan cuando ninguna contribución cubre ese rol, o cuando todas fallaron
EJEMPLOS = {
    "microjuego": ["examples.microjuegos.esquiva", "examples.microjuegos.suelta", "examples.microjuegos.despierta",
                   "examples.microjuegos.reacciona", "examples.microjuegos.piruetas"],
    "interludio": ["examples.microjuegos.interludio_simple"],
    "transicion": ["examples.microjuegos.giro"],
}

PATRON_RUT = re.compile(r"^\d{7,8}-[\dkK]$")
PATRON_ROL = re.compile(r"^ROL\s*=", re.MULTILINE)


# Las piezas de los estudiantes van en esta carpeta de la raíz del repositorio,
# cada una en una subcarpeta con su RUT, para no mezclarse con las carpetas de
# las otras tareas, que también llevan el RUT como nombre.
CARPETA_PIEZAS = "microjuegos"


def buscar_contribuciones(carpeta=CARPETA_PIEZAS):
    """Paquetes microjuegos.RUT cuyo __init__.py declara un ROL."""
    if not Path(carpeta).is_dir():
        return []
    nombres = []
    for ruta in sorted(Path(carpeta).iterdir()):
        inicio = ruta / "__init__.py"
        if PATRON_RUT.match(ruta.name) and inicio.exists():
            if PATRON_ROL.search(inicio.read_text(encoding="utf-8")):
                nombres.append(f"{carpeta}.{ruta.name}")
    return nombres


def nombre_de_paquete(nombre):
    """Un RUT suelto se busca en la carpeta de las piezas; lo demás es un paquete o una ruta."""
    nombre = nombre.rstrip("/\\")
    if PATRON_RUT.match(nombre):
        return f"{CARPETA_PIEZAS}.{nombre}"
    return nombre.replace("/", ".").replace("\\", ".")


def preparar_pantalla(modulo):
    """Programa con el fragment_program.glsl del juego y el rectángulo que cubre la ventana."""
    fuente = (Path(modulo.__file__).parent / "fragment_program.glsl").read_text(encoding="utf-8")
    programa = ShaderProgram(Shader(VERTEX_PANTALLA, "vertex"), Shader(fuente, "fragment"))
    rectangulo = programa.vertex_list(4, GL.GL_TRIANGLE_STRIP, position=("f", (-1, -1, 1, -1, -1, 1, 1, 1)))
    return {"programa": programa, "rectangulo": rectangulo, "avisados": set()}


def valor_uniform(valor, tamano):
    """Convierte lo que entrega uniformes() al formato que espera pyglet.

    Un arreglo de NumPy pasa a lista. Un arreglo más corto que el del shader se
    completa con ceros, así el juego no tiene que rellenarlo cuando tiene menos
    objetos que el máximo.
    """
    if hasattr(valor, "tolist"):
        valor = valor.tolist()
    if tamano > 1 and isinstance(valor, (list, tuple)) and len(valor) < tamano:
        if not valor:
            return None  # un arreglo vacío no cambia el uniform
        cero = tuple(0.0 for _ in valor[0]) if isinstance(valor[0], (list, tuple)) else 0
        valor = list(valor) + [cero] * (tamano - len(valor))
    return valor


def cargar(nombre, ventana):
    """Importa una pieza, revisa su contrato y prepara lo que dibuja."""
    nombre = nombre.rstrip("/\\")
    try:
        modulo = importlib.import_module(nombre)
    except Exception:
        traceback.print_exc()
        print(f"[microjuegos] {nombre}: no se pudo importar y queda fuera")
        return None

    rol = getattr(modulo, "ROL", None)
    if rol not in CONTRATO:
        print(f"[microjuegos] {nombre}: ROL = {rol!r} no es uno de {list(CONTRATO)}")
        return None
    faltan = [atributo for atributo in CONTRATO[rol] if not hasattr(modulo, atributo)]
    if rol in JUEGOS and not (hasattr(modulo, "uniformes") or hasattr(modulo, "dibujar")):
        faltan.append("uniformes (o dibujar)")
    if faltan:
        print(f"[microjuegos] {nombre}: al {rol} le falta {', '.join(faltan)}")
        return None
    duracion = getattr(modulo, "DURACION", DURACION_MICROJUEGO)
    if rol == "microjuego" and not (isinstance(duracion, (int, float)) and 3 <= duracion <= 6):
        print(f"[microjuegos] {nombre}: DURACION = {duracion}, debe estar entre 3 y 6")
        return None

    pieza = {"nombre": nombre, "rol": rol, "modulo": modulo, "duracion": duracion, "recursos": None, "pantalla": None}
    try:
        if hasattr(modulo, "preparar"):
            pieza["recursos"] = modulo.preparar(ventana)
        if rol in JUEGOS and not hasattr(modulo, "dibujar"):
            pieza["pantalla"] = preparar_pantalla(modulo)
    except Exception:
        traceback.print_exc()
        print(f"[microjuegos] {nombre}: no se pudo preparar y queda fuera")
        return None

    print(f"[microjuegos] {nombre}: {rol} listo")
    return pieza


@click.command("microjuegos", short_help="Anfitrión del proyecto colectivo de microjuegos")
@click.argument("contribuciones", nargs=-1)
@click.option("--dificultad", type=click.IntRange(1, 3), default=None, help="Fija la dificultad (1 a 3)")
@click.option("--velocidad", type=float, default=None, help="Fija la velocidad (1 es la inicial)")
def microjuegos(contribuciones, dificultad, velocidad):
    """Encadena microjuegos.

    CONTRIBUCIONES son RUT de carpetas dentro de microjuegos/ o nombres de
    paquetes. Sin argumentos, usa todas las carpetas con RUT de microjuegos/ que
    declaren un ROL, y completa con los ejemplos los roles que nadie cubrió.
    """
    ventana = pyglet.window.Window(ANCHO, ALTO, caption="Microjuegos CC3501")
    proyeccion_base = ventana.projection
    ancho_fb, alto_fb = ventana.get_framebuffer_size()

    nombres = [nombre_de_paquete(nombre) for nombre in contribuciones] or buscar_contribuciones()
    print(f"[microjuegos] contribuciones: {nombres if nombres else 'ninguna, solo ejemplos'}")
    piezas = [pieza for pieza in (cargar(nombre, ventana) for nombre in nombres) if pieza]
    reparto = {rol: [pieza for pieza in piezas if pieza["rol"] == rol] for rol in CONTRATO}
    respaldo = {}

    def disponibles(rol):
        """Las piezas de un rol; si no queda ninguna, los ejemplos de ese rol."""
        if reparto[rol] or rol not in EJEMPLOS:
            return reparto[rol]
        if rol not in respaldo:
            respaldo[rol] = [pieza for pieza in (cargar(nombre, ventana) for nombre in EJEMPLOS[rol]) if pieza]
        return respaldo[rol]

    def retirar(pieza, motivo):
        print(f"[microjuegos] {pieza['nombre']} {motivo} y sale de la rotación")
        for lista in (reparto[pieza["rol"]], respaldo.get(pieza["rol"], [])):
            if pieza in lista:
                lista.remove(pieza)

    def llamar(pieza, funcion, *argumentos):
        """Llama a una función de la pieza. Si falla, la saca de la rotación."""
        try:
            return getattr(pieza["modulo"], funcion)(*argumentos), True
        except Exception:
            traceback.print_exc()
            retirar(pieza, f"falló en {funcion}()")
            return None, False

    # %% Estado de la partida

    partida = {"escena": None, "pieza": None, "estado": None, "tiempo": 0.0, "destino": None}

    def reiniciar():
        partida.update(
            vidas=VIDAS,
            puntaje=0,
            ronda=0,
            velocidad=velocidad or 1.0,
            dificultad=dificultad or 1,
            ultimo_juego=None,
        )

    def es_ronda_de_jefe(ronda):
        return bool(disponibles("jefe")) and (ronda + 1) % RONDAS_POR_JEFE == 0

    def abrir_interludio(evento, resultado=None):
        pieza = random.choice(disponibles("interludio"))
        situacion = {
            "evento": evento,
            "resultado": resultado,
            "vidas": partida["vidas"],
            "vidas_iniciales": VIDAS,
            "puntaje": partida["puntaje"],
            "ronda": partida["ronda"],
            "velocidad": partida["velocidad"],
            "dificultad": partida["dificultad"],
        }
        estado, correcto = llamar(pieza, "iniciar", pieza["recursos"], situacion)
        if not correcto:
            return abrir_interludio(evento, resultado)
        partida.update(escena="interludio", pieza=pieza, estado=estado, tiempo=0.0, evento=evento)
        print(f"[microjuegos] interludio {pieza['nombre']}: {evento}")

    def abrir_juego():
        if es_ronda_de_jefe(partida["ronda"]):
            candidatos = disponibles("jefe")
        else:
            candidatos = disponibles("microjuego")
            # no repetir el anterior si hay otro
            candidatos = [pieza for pieza in candidatos if pieza is not partida["ultimo_juego"]] or candidatos
        pieza = random.choice(candidatos)
        espacio = pymunk.Space()
        espacio.gravity = GRAVEDAD
        estado, correcto = llamar(pieza, "iniciar", partida["dificultad"], espacio)
        if not correcto:
            return abrir_juego()
        if not isinstance(estado, dict):
            retirar(pieza, "no devolvió un diccionario en iniciar()")
            return abrir_juego()
        estado.setdefault("resultado", None)
        partida.update(escena="juego", pieza=pieza, estado=estado, espacio=espacio, tiempo=0.0, ultimo_juego=pieza)
        hud["instruccion"].text = hud["sombra"].text = pieza["modulo"].INSTRUCCION
        print(
            f"[microjuegos] ronda {partida['ronda'] + 1}: {pieza['nombre']} "
            f"(dificultad {partida['dificultad']}, velocidad {partida['velocidad']:.2f})"
        )

    def cerrar_juego():
        pieza = partida["pieza"]
        resultado = partida["estado"].get("resultado")
        if resultado is None:
            resultado = pieza["rol"] == "microjuego" and pieza["modulo"].GANA_SI_SOBREVIVE
        resultado = bool(resultado)
        partida["puntaje"] += resultado
        partida["vidas"] -= not resultado
        partida["ronda"] += 1
        print(f"[microjuegos] {'ganó' if resultado else 'perdió'}; vidas {partida['vidas']}, puntaje {partida['puntaje']}")

        # tras un jefe (o cada RONDAS_POR_JEFE rondas si no hay jefes) sube la
        # dificultad y la velocidad vuelve a 1
        ciclo_completo = pieza["rol"] == "jefe" or (
            not disponibles("jefe") and partida["ronda"] % RONDAS_POR_JEFE == 0
        )
        evento = "normal"
        if partida["vidas"] == 0:
            evento = "fin"
        elif ciclo_completo:
            if velocidad is None:
                partida["velocidad"] = 1.0
            if dificultad is None and partida["dificultad"] < 3:
                partida["dificultad"] += 1
                evento = "dificultad"
        elif es_ronda_de_jefe(partida["ronda"]):
            evento = "jefe"
        elif partida["ronda"] % RONDAS_POR_ACELERACION == 0 and velocidad is None:
            partida["velocidad"] *= FACTOR_ACELERACION
            evento = "acelera"
        pedir_cambio(lambda: abrir_interludio(evento, resultado))

    def pedir_cambio(destino):
        """El cambio se hace en on_draw, después de capturar el último cuadro."""
        partida["destino"] = destino

    def terminar_transicion():
        destino = partida["destino"]
        partida["destino"] = None
        destino()

    # %% Entrada: el anfitrión la traduce a un diccionario por cuadro

    presionadas = set()
    pulsos = {"espacio": False, "clic": False}
    raton = {"posicion": (0, 0), "presionado": False}

    @ventana.event
    def on_key_press(simbolo, modificadores):
        presionadas.add(simbolo)
        if simbolo == key.SPACE:
            pulsos["espacio"] = True

    @ventana.event
    def on_key_release(simbolo, modificadores):
        presionadas.discard(simbolo)

    @ventana.event
    def on_mouse_press(x, y, boton, modificadores):
        if boton == mouse.LEFT:
            raton.update(posicion=(x, y), presionado=True)
            pulsos["clic"] = True

    @ventana.event
    def on_mouse_release(x, y, boton, modificadores):
        if boton == mouse.LEFT:
            raton.update(posicion=(x, y), presionado=False)

    @ventana.event
    def on_mouse_motion(x, y, dx, dy):
        raton["posicion"] = (x, y)

    @ventana.event
    def on_mouse_drag(x, y, dx, dy, botones, modificadores):
        raton["posicion"] = (x, y)

    def leer_entrada():
        return {
            "izquierda": key.LEFT in presionadas,
            "derecha": key.RIGHT in presionadas,
            "arriba": key.UP in presionadas,
            "abajo": key.DOWN in presionadas,
            "espacio": key.SPACE in presionadas,
            "espacio_presionado": pulsos["espacio"],
            "raton": raton["posicion"],
            "raton_presionado": raton["presionado"],
            "clic": pulsos["clic"],
        }

    # %% Ciclo de actualización

    def actualizar(dt):
        if partida["destino"] is not None and partida["escena"] != "transicion":
            return  # espera a que on_draw capture el cuadro

        escena = partida["escena"]
        pieza, estado = partida["pieza"], partida["estado"]
        entrada = leer_entrada()
        pulsos.update(espacio=False, clic=False)

        if escena == "transicion":
            partida["tiempo"] += dt
            if partida["tiempo"] >= DURACION_TRANSICION:
                terminar_transicion()

        elif escena == "interludio":
            dt_juego = dt * partida["velocidad"]
            partida["tiempo"] += dt_juego
            _, correcto = llamar(pieza, "actualizar", estado, dt_juego)
            if not correcto:
                abrir_interludio(partida["evento"])
            elif partida["evento"] in ("inicio", "fin"):
                if entrada["espacio_presionado"]:
                    if partida["evento"] == "fin":
                        reiniciar()
                    pedir_cambio(abrir_juego)
            elif partida["tiempo"] >= DURACION_INTERLUDIO:
                pedir_cambio(abrir_juego)

        elif escena == "juego":
            dt_juego = dt * partida["velocidad"]
            partida["tiempo"] += dt_juego
            _, correcto = llamar(pieza, "actualizar", estado, dt_juego, entrada)
            if correcto:
                correcto = avanzar_fisica(pieza, dt_juego)
            if not correcto:
                pedir_cambio(lambda: abrir_interludio("normal"))
                return
            if pieza["rol"] == "jefe":
                if estado.get("resultado") is not None or partida["tiempo"] >= DURACION_MAXIMA_JEFE:
                    cerrar_juego()
            elif partida["tiempo"] >= pieza["duracion"]:
                cerrar_juego()

    def avanzar_fisica(pieza, dt):
        """Avanza el espacio de la ronda con el mismo dt del juego, en subpasos."""
        pasos = max(1, math.ceil(dt / PASO_FISICA))
        try:
            for _ in range(pasos):
                partida["espacio"].step(dt / pasos)
        except Exception:
            traceback.print_exc()
            retirar(pieza, "falló al avanzar su espacio de pymunk")
            return False
        return True

    # %% Dibujo

    load_firacode()
    lote = pyglet.graphics.Batch()
    fondo, frente = pyglet.graphics.Group(order=0), pyglet.graphics.Group(order=1)
    hud = {
        "barra_fondo": pyglet.shapes.Rectangle(0, 0, ANCHO, 12, color=(10, 14, 80), batch=lote, group=fondo),
        "barra": pyglet.shapes.Rectangle(0, 0, ANCHO, 12, color=(207, 56, 137), batch=lote, group=frente),
    }
    texto = {"font_name": "Fira Code", "font_size": 46, "anchor_x": "center", "anchor_y": "center"}
    hud["sombra"] = pyglet.text.Label("", x=ANCHO // 2 + 3, y=ALTO // 2 - 3, color=(10, 14, 80, 255), **texto)
    hud["instruccion"] = pyglet.text.Label("", x=ANCHO // 2, y=ALTO // 2, color=(255, 255, 255, 255), **texto)

    # el último cuadro de cada escena, que recibe la transición
    cuadro = pyglet.image.Texture.create(ancho_fb, alto_fb)

    def estado_gl_base(limpiar):
        """Deja OpenGL como lo espera cada pieza, aunque la anterior lo haya cambiado."""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, ancho_fb, alto_fb)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        ventana.projection = proyeccion_base
        ventana.view = Mat4()
        if limpiar:
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            ventana.clear()

    def dibujar_pantalla(pieza, estado):
        """Dibuja la ventana con el fragment program del juego y los valores de uniformes()."""
        valores, correcto = llamar(pieza, "uniformes", estado)
        if not correcto:
            return False
        pantalla = pieza["pantalla"]
        programa = pantalla["programa"]
        programa.use()
        valores = {"resolucion": (ANCHO, ALTO), "tiempo": partida["tiempo"], **(valores or {})}
        for nombre, valor in valores.items():
            if nombre not in programa.uniforms:
                # el compilador de GLSL elimina los uniforms que el shader no usa
                if nombre not in ("resolucion", "tiempo") and nombre not in pantalla["avisados"]:
                    pantalla["avisados"].add(nombre)
                    print(f"[microjuegos] {pieza['nombre']}: el fragment program no usa el uniform {nombre}")
                continue
            valor = valor_uniform(valor, programa.uniforms[nombre]["size"])
            if valor is None:
                continue
            try:
                programa[nombre] = valor
            except Exception:
                traceback.print_exc()
                retirar(pieza, f"entregó un valor que no calza con el uniform {nombre}")
                return False
        pantalla["rectangulo"].draw(GL.GL_TRIANGLE_STRIP)
        return True

    def dibujar_hud():
        pieza = partida["pieza"]
        if pieza["rol"] == "microjuego":
            restante = max(0.0, 1.0 - partida["tiempo"] / pieza["duracion"])
            hud["barra"].width = ANCHO * restante
        else:
            hud["barra"].width = 0
        with ui_overlay():
            lote.draw()
            if partida["tiempo"] < DURACION_INSTRUCCION:
                hud["sombra"].draw()
                hud["instruccion"].draw()

    @ventana.event
    def on_draw():
        estado_gl_base(limpiar=True)
        escena, pieza = partida["escena"], partida["pieza"]

        if escena == "transicion":
            progreso = min(1.0, partida["tiempo"] / DURACION_TRANSICION)
            _, correcto = llamar(pieza, "dibujar", pieza["recursos"], cuadro, progreso)
            if not correcto:
                terminar_transicion()
            return

        if escena == "juego" and pieza["pantalla"] is not None:
            correcto = dibujar_pantalla(pieza, partida["estado"])
        elif escena == "juego":
            _, correcto = llamar(pieza, "dibujar", pieza["recursos"], partida["estado"])
        else:
            _, correcto = llamar(pieza, "dibujar", partida["estado"])
        estado_gl_base(limpiar=False)
        if not correcto:
            if escena == "interludio":
                abrir_interludio(partida["evento"])
            elif partida["destino"] is None:
                pedir_cambio(lambda: abrir_interludio("normal"))
        elif escena == "juego":
            dibujar_hud()

        if partida["destino"] is not None:
            GL.glBindTexture(cuadro.target, cuadro.id)
            GL.glCopyTexSubImage2D(cuadro.target, 0, 0, 0, 0, 0, ancho_fb, alto_fb)
            pieza = random.choice(disponibles("transicion"))
            partida.update(escena="transicion", pieza=pieza, estado=None, tiempo=0.0)

    reiniciar()
    abrir_interludio("inicio")
    pyglet.clock.schedule_interval(actualizar, 1 / 60)
    pyglet.app.run()
