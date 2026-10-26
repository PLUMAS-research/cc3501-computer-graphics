"""Particiones binarias del espacio (BSP y kd-tree) para lanzar rayos sobre el mapa E1M1 de Doom.

A la izquierda, el mapa visto desde arriba con los colores del automapa de
Doom: rojo para las paredes sólidas, café donde cambia la altura del piso y
amarillo donde cambia la del techo. Encima van las rectas de corte del árbol y
el recorrido del rayo central: las regiones que visita, numeradas en orden, y
las paredes que prueba. A la derecha, la vista en primera persona: un rayo por
columna, cada uno recorriendo el mismo árbol (primera_persona.py).

El BSP usa las rectas de las paredes, como el de Doom; el kd-tree corta en la
mediana con rectas verticales u horizontales (arboles.py). Los dos encuentran
los mismos impactos que la fuerza bruta, que prueba todas las paredes con cada
rayo; el panel compara cuántas paredes prueba cada uno.

El mapa se lee de un WAD (wad.py): por omisión, el de la versión shareware de
Doom, en assets/doom; con --wad y --mapa se elige otro.

Controles:
- flechas: avanzar, retroceder y girar.
- espacio: abre o cierra la puerta que está enfrente, como la tecla de usar de
  Doom. Abrirla sube el techo de su sector, sin cambiar los árboles (puertas.py).
- K: BSP o kd-tree.
- O: colorea las paredes del BSP en el orden de dibujo de Doom.
- T: muestra u oculta el recorrido del rayo central.
- Z: mapa completo o acercado alrededor del jugador.
- D: detalle alto (320 columnas) o bajo (160), como la tecla F5 de Doom.
- R: vuelve al inicio del mapa, con las puertas como al empezar.
"""
import time
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
from pyglet.math import Mat4, Vec3

from grafica.ui import InfoPanel, load_firacode

from . import arboles, primera_persona, puertas
from .wad import cargar_mapa

RAIZ = Path(__file__).resolve().parents[2]
ANCHO_VENTANA, ALTO_VENTANA = 1280, 720
ANCHO_MAPA = 640  # la mitad izquierda de la ventana
VISTA = (640, 240, 640, 480)  # x, y, ancho y alto; 320 x 200 estirado a 4:3, como en Doom
ACERCAMIENTO = 3.0
VELOCIDAD = 320.0  # unidades por segundo
GIRO = 2.6  # radianes por segundo

COLOR_SOLIDA = (230, 40, 40)
COLOR_CAMBIO_PISO = (190, 120, 60)
COLOR_CAMBIO_TECHO = (230, 210, 60)
COLOR_SIN_CAMBIO = (80, 80, 80)
COLOR_CORTE = {"bsp": (40, 150, 190), "kd": (170, 90, 210)}
COLOR_REGION = (60, 200, 110)
COLOR_PROBADA = (255, 160, 40)
COLOR_JUGADOR = (255, 255, 255)
SENTIDO = {1: "sube", 0: "espera", -1: "baja"}

CAPA_REGIONES = pyglet.graphics.Group(order=0)
CAPA_CORTES = pyglet.graphics.Group(order=1)
CAPA_PAREDES = pyglet.graphics.Group(order=2)
CAPA_RAYO = pyglet.graphics.Group(order=3)


@click.command("raytracing_bsp", short_help="BSP y kd-tree para lanzar rayos en el mapa E1M1 de Doom")
@click.option("--wad", "ruta_wad", type=click.Path(exists=True, dir_okay=False),
              default=str(RAIZ / "assets" / "doom" / "doom1.wad"), help="WAD con mapas en el formato de Doom")
@click.option("--mapa", "nombre_mapa", default="E1M1", help="nombre del mapa en el WAD (E1M1, MAP01, etc.)")
def raytracing_bsp(ruta_wad, nombre_mapa):
    mapa = cargar_mapa(ruta_wad, nombre_mapa)
    vertices, lineas = mapa["vertices"], mapa["lineas"]
    a, b = vertices[lineas[:, 0]], vertices[lineas[:, 1]]
    solida = mapa["sector_izquierdo"] < 0
    minimo, maximo = vertices.min(axis=0) - 64, vertices.max(axis=0) + 64
    region = np.array([minimo, [maximo[0], minimo[1]], maximo, [minimo[0], maximo[1]]])
    print(f"[bsp] {nombre_mapa}: {len(lineas)} líneas, {solida.sum()} sólidas, {len(mapa['piso'])} sectores")

    arbol = {}
    for clave, construir in (("bsp", arboles.construir_bsp), ("kd", arboles.construir_kd)):
        inicio = time.perf_counter()
        arbol[clave] = construir(a, b, np.arange(len(lineas)), region)
        todos = arboles.nodos(arbol[clave])
        arbol[clave]["internos"] = sum("normal" in nodo for nodo, _ in todos)
        arbol[clave]["profundidad"] = max(profundidad for _, profundidad in todos)
        arbol[clave]["referencias"] = sum(len(nodo["paredes"]) for nodo, _ in todos)
        print(f"[bsp] {clave}: {arbol[clave]['internos']} nodos internos, profundidad "
              f"{arbol[clave]['profundidad']}, {len(arbol[clave]['a'])} paredes y "
              f"{arbol[clave]['referencias']} referencias, en {time.perf_counter() - inicio:.2f} s")

    window = pyglet.window.Window(ANCHO_VENTANA, ALTO_VENTANA, caption=f"BSP y kd-tree en {nombre_mapa}")
    load_firacode()

    posicion_inicial, angulo_inicial = mapa["jugador"]
    techo_inicial = mapa["techo"].copy()
    state = {
        "estructura": "bsp",
        "posicion": posicion_inicial.copy(),
        "angulo": angulo_inicial,
        "orden": False,
        "recorrido": True,
        "acercamiento": 1.0,
        "columnas": 320,
        "puertas": {},  # las puertas en marcha, por sector
    }

    # el mapa completo cabe en la mitad izquierda con esta escala; acercado, el
    # jugador queda al centro de esa mitad
    escala = min((ANCHO_MAPA - 20) / (maximo[0] - minimo[0]), (ALTO_VENTANA - 20) / (maximo[1] - minimo[1]))
    centro = np.array([ANCHO_MAPA, ALTO_VENTANA]) / 2
    desplazamiento = centro - (minimo + maximo) / 2 * escala

    def en_pantalla(punto):
        if state["acercamiento"] == 1.0:
            return tuple(np.asarray(punto) * escala + desplazamiento)
        return tuple(centro + (np.asarray(punto) - state["posicion"]) * escala * state["acercamiento"])

    def armar_mapa():
        """Las figuras fijas del mapa, en la escala del mapa completo; al dibujar,
        la matriz de vista las acerca. El grosor se divide por el acercamiento
        para que en pantalla no cambie."""
        grosor = 1.5 / state["acercamiento"]
        # las figuras se guardan junto a sus lotes: si nada las referencia, el
        # recolector de basura las borra y desaparecen del lote
        lotes = {"paredes": pyglet.graphics.Batch(), "orden": pyglet.graphics.Batch(), "figuras": []}

        def linea(p, q, color, lote, grupo, ancho=grosor):
            figura = pyglet.shapes.Line(*(p * escala + desplazamiento), *(q * escala + desplazamiento),
                                        thickness=ancho, color=color, batch=lote, group=grupo)
            lotes["figuras"].append(figura)
            return figura

        # paredes con los colores del automapa
        for indice, (inicio, fin) in enumerate(zip(a, b)):
            derecho, izquierdo = mapa["sector_derecho"][indice], mapa["sector_izquierdo"][indice]
            if izquierdo < 0:
                color = COLOR_SOLIDA
            elif mapa["piso"][derecho] != mapa["piso"][izquierdo]:
                color = COLOR_CAMBIO_PISO
            elif mapa["techo"][derecho] != mapa["techo"][izquierdo]:
                color = COLOR_CAMBIO_TECHO
            else:
                color = COLOR_SIN_CAMBIO
            linea(inicio, fin, color, lotes["paredes"], CAPA_PAREDES)
        # rectas de corte de cada árbol, recortadas a la región de su nodo
        for clave, estructura in arbol.items():
            lotes[clave] = pyglet.graphics.Batch()
            for nodo, _ in arboles.nodos(estructura):
                extremos = arboles.cuerda(nodo) if "normal" in nodo else []
                if len(extremos) == 2:
                    linea(*extremos, COLOR_CORTE[clave], lotes[clave], CAPA_CORTES, grosor / 1.5)
        # los trozos de pared del BSP, que se colorean según el orden de dibujo
        lotes["trozos"] = [linea(p, q, (0, 0, 0), lotes["orden"], CAPA_PAREDES, grosor * 1.3)
                           for p, q in zip(arbol["bsp"]["a"], arbol["bsp"]["b"])]
        return lotes

    lotes = armar_mapa()

    # la imagen en primera persona se recalcula solo cuando cambia algo
    vista = {"clave": None, "textura": None}

    panel = (
        InfoPanel(x=VISTA[0] + 14, y_top=VISTA[1] - 26)
        .add("estructura", size=14)
        .add("arbol")
        .add("rayo")
        .add("cuadro")
        .add("puerta")
        .add("orden")
        .footer("flechas mover   K árbol   O orden   T recorrido   Z acercar   D detalle   R inicio", font_size=9)
    )

    def apply_state():
        actual = arbol[state["estructura"]]
        if state["estructura"] == "bsp":
            panel["estructura"] = "BSP: rectas de las paredes, como en Doom"
            panel["arbol"] = (f"{actual['internos']} nodos, profundidad {actual['profundidad']}; "
                              f"{len(actual['a'])} paredes ({len(actual['a']) - len(lineas)} partidas)")
        else:
            panel["estructura"] = "kd-tree: cortes en la mediana, en x o en y"
            panel["arbol"] = (f"{actual['internos']} nodos, profundidad {actual['profundidad']}; "
                              f"{actual['referencias']} referencias a {len(lineas)} paredes")
        panel["orden"] = "orden de dibujo: ninguna pared tapa a otra más clara" if state["orden"] else ""
        print(f"[bsp] estructura={state['estructura']} columnas={state['columnas']} orden={state['orden']} "
              f"recorrido={state['recorrido']} acercamiento={state['acercamiento']:.0f}")

    def actualizar_vista():
        clave = (tuple(state["posicion"]), state["angulo"], state["estructura"], state["columnas"],
                 mapa["techo"].tobytes())
        if clave == vista["clave"]:
            return
        inicio = time.perf_counter()
        sector = primera_persona.sector_en(mapa, arbol["bsp"], solida, state["posicion"])
        ojo = mapa["piso"][sector] + primera_persona.ALTURA_OJO
        imagen, cuenta = primera_persona.dibujar(mapa, arbol[state["estructura"]], solida, state["posicion"],
                                                 state["angulo"], ojo, columnas=state["columnas"])
        milisegundos = 1000 * (time.perf_counter() - inicio)
        # pyglet espera las filas de abajo hacia arriba
        datos = (np.flipud(imagen) * 255).astype(np.uint8)
        vista["textura"] = pyglet.image.ImageData(datos.shape[1], datos.shape[0], "RGB", datos.tobytes()).get_texture()
        # se conservan los píxeles gruesos de 320 x 200
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        vista["clave"] = clave
        fuerza_bruta = len(arbol[state["estructura"]]["a"]) * state["columnas"]
        panel["cuadro"] = (f"{state['columnas']} columnas: {cuenta['paredes']} paredes probadas "
                           f"de {fuerza_bruta}, {milisegundos:.0f} ms")
        en_marcha = [f"puerta del sector {sector}: {SENTIDO[puerta['direccion']]}, abierta "
                     f"{mapa['techo'][sector] - mapa['piso'][sector]:.0f} de "
                     f"{puerta['tope'] - mapa['piso'][sector]:.0f} unidades"
                     for sector, puerta in state["puertas"].items()]
        panel["puerta"] = "   ".join(en_marcha) or "espacio abre la puerta que está enfrente"

    def dibujar_rayo_central():
        """El rayo del centro de la vista, con las regiones que visita y las paredes que prueba."""
        actual = arbol[state["estructura"]]
        direccion = np.array([np.cos(state["angulo"]), np.sin(state["angulo"])])
        cuenta = {"nodos": 0, "paredes": 0, "visitados": [], "probadas": []}
        cruces = arboles.impactos(actual, state["posicion"], direccion, solida, cuenta)
        panel["rayo"] = (f"rayo central: {cuenta['nodos']} nodos, {cuenta['paredes']} paredes "
                         f"probadas de {len(actual['a'])}")
        lote, temporales = pyglet.graphics.Batch(), []
        if state["recorrido"]:
            hojas = [nodo for nodo in cuenta["visitados"] if "normal" not in nodo and len(nodo["region"]) >= 3]
            for orden, hoja in enumerate(hojas, start=1):
                region = hoja["region"]
                temporales.append(pyglet.shapes.Polygon(*[en_pantalla(p) for p in region], color=COLOR_REGION,
                                                        batch=lote, group=CAPA_REGIONES))
                temporales[-1].opacity = 70
                x, y = en_pantalla(region.mean(axis=0))
                temporales.append(pyglet.text.Label(str(orden), font_name="Fira Code", font_size=9, x=x, y=y,
                                                    anchor_x="center", anchor_y="center", batch=lote, group=CAPA_RAYO))
            for pared in set(cuenta["probadas"]):
                temporales.append(pyglet.shapes.Line(*en_pantalla(actual["a"][pared]), *en_pantalla(actual["b"][pared]),
                                                     thickness=3, color=COLOR_PROBADA, batch=lote, group=CAPA_RAYO))
        fin = state["posicion"] + direccion * (cruces[-1][0] if cruces else 4000)
        temporales.append(pyglet.shapes.Line(*en_pantalla(state["posicion"]), *en_pantalla(fin), thickness=1.5,
                                             color=COLOR_JUGADOR, batch=lote, group=CAPA_RAYO))
        for t, _ in cruces:
            temporales.append(pyglet.shapes.Circle(*en_pantalla(state["posicion"] + t * direccion), 3,
                                                   color=COLOR_JUGADOR, batch=lote, group=CAPA_RAYO))
        # el jugador y los bordes de su campo de visión de 90 grados
        for desvio in (-np.pi / 4, np.pi / 4):
            borde = state["posicion"] + 400 * np.array([np.cos(state["angulo"] + desvio), np.sin(state["angulo"] + desvio)])
            temporales.append(pyglet.shapes.Line(*en_pantalla(state["posicion"]), *en_pantalla(borde), thickness=1,
                                                 color=(150, 150, 150), batch=lote, group=CAPA_RAYO))
        temporales.append(pyglet.shapes.Circle(*en_pantalla(state["posicion"]), 4, color=COLOR_JUGADOR,
                                               batch=lote, group=CAPA_RAYO))
        lote.draw()

    def colorear_por_orden():
        # el recorrido de adelante hacia atrás sin rayo: una pared nunca tapa a
        # otra que va antes, aunque esa otra esté más lejos
        orden = arboles.orden_de_dibujo(arbol["bsp"], state["posicion"])
        for rango, pared in enumerate(orden):
            gris = int(255 * (1 - rango / len(orden)) ** 2)
            lotes["trozos"][pared].color = (gris, gris, gris)

    @window.event
    def on_draw():
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        window.clear()
        actualizar_vista()
        x, y, ancho, alto = VISTA
        vista["textura"].blit(x, y, width=ancho, height=alto)

        # el mapa queda en la mitad izquierda; acercado, la matriz de vista lo
        # escala y lo traslada para dejar al jugador al centro
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(0, 0, ANCHO_MAPA, ALTO_VENTANA)
        if state["acercamiento"] != 1.0:
            jugador = state["posicion"] * escala + desplazamiento
            window.view = (Mat4.from_translation(Vec3(*centro, 0)) @ Mat4.from_scale(Vec3(state["acercamiento"],
                           state["acercamiento"], 1)) @ Mat4.from_translation(Vec3(-jugador[0], -jugador[1], 0)))
        lotes[state["estructura"]].draw()
        if state["orden"] and state["estructura"] == "bsp":
            colorear_por_orden()
            lotes["orden"].draw()
        else:
            lotes["paredes"].draw()
        window.view = Mat4()
        dibujar_rayo_central()
        GL.glDisable(GL.GL_SCISSOR_TEST)
        panel.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal lotes
        key = pyglet.window.key
        if symbol == key.K:
            state["estructura"] = "kd" if state["estructura"] == "bsp" else "bsp"
        elif symbol == key.O:
            state["orden"] = not state["orden"]
        elif symbol == key.T:
            state["recorrido"] = not state["recorrido"]
        elif symbol == key.Z:
            state["acercamiento"] = ACERCAMIENTO if state["acercamiento"] == 1.0 else 1.0
            lotes = armar_mapa()
        elif symbol == key.D:
            state["columnas"] = 160 if state["columnas"] == 320 else 320
        elif symbol == key.SPACE:
            puerta = puertas.usar(mapa, arbol["bsp"], solida, state["puertas"], state["posicion"], state["angulo"])
            print(f"[bsp] usar: puerta del sector {puerta}" if puerta >= 0 else "[bsp] usar: no hay una puerta enfrente")
            return
        elif symbol == key.R:
            state["posicion"], state["angulo"] = posicion_inicial.copy(), angulo_inicial
            mapa["techo"][:] = techo_inicial
            state["puertas"].clear()
        else:
            return
        apply_state()

    # el estado de las flechas se lee en cada paso; se empuja después de los
    # manejadores decorados, que escriben en el tope de la pila de eventos
    teclas = pyglet.window.key.KeyStateHandler()
    window.push_handlers(teclas)

    def avanzar(dt):
        key = pyglet.window.key
        if state["puertas"]:
            jugador = primera_persona.sector_en(mapa, arbol["bsp"], solida, state["posicion"])
            puertas.mover(mapa, state["puertas"], dt, jugador)
        state["angulo"] += GIRO * dt * (teclas[key.LEFT] - teclas[key.RIGHT])
        paso = VELOCIDAD * dt * (teclas[key.UP] - teclas[key.DOWN])
        if paso == 0:
            return
        desplazamiento_jugador = paso * np.array([np.cos(state["angulo"]), np.sin(state["angulo"])])
        # si una pared bloquea el paso, se intenta deslizar a lo largo de x o de y, como en Doom
        for intento in (desplazamiento_jugador, desplazamiento_jugador * [1, 0], desplazamiento_jugador * [0, 1]):
            if primera_persona.puede_avanzar(mapa, arbol["bsp"], solida, state["posicion"], intento):
                state["posicion"] = state["posicion"] + intento
                return

    pyglet.clock.schedule_interval(avanzar, 1 / 60)
    apply_state()
    pyglet.app.run()
