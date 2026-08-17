"""Generacion automatica de screenshots de los ejemplos, sin abrir ventanas.

Reusa el patron de testing headless documentado en CLAUDE.md ("Testing headless
de ejemplos"): se parchea pyglet para capturar la ventana y los ticks agendados,
`pyglet.app.run` queda en no-op, se arma el ejemplo llamando su `@click.command`,
se avanza el reloj a mano (para animaciones y para el render progresivo del ray
tracing), se disparan teclas para capturar un estado concreto, y se guarda el
framebuffer a PNG.

Uso:
    uv run python tools/screenshots.py galeria               # todos, a screenshots/
    uv run python tools/screenshots.py galeria --solo bosque
    uv run python tools/screenshots.py uno raytracing_cpu
    uv run python tools/screenshots.py shot bosque --salida screenshots/bosque.png

`galeria` y `uno` lanzan un subproceso por captura (un ejemplo que falle no tumba
el resto, y cada uno parte con estado de pyglet limpio). `shot` es el worker que
captura una sola imagen en el proceso actual.
"""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click

RAIZ = Path(__file__).resolve().parent.parent
DIRECTORIO_POR_DEFECTO = "screenshots"

# el script vive en tools/, asi que agregamos la raiz del repo al path para
# poder importar caja_de_juguetes (el registro de comandos) y los ejemplos
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


@dataclass
class Toma:
    """Una captura: un estado del ejemplo (parametros, teclas, cuadros a avanzar)."""

    sufijo: str = ""           # se agrega al nombre del archivo (para varios estados)
    args: list = field(default_factory=list)   # argumentos CLI del comando (ej. un asset)
    teclas: list = field(default_factory=list)  # teclas a disparar antes de capturar
    cuadros: int = 20          # ticks de reloj a avanzar (animacion / render progresivo)


# Recetas por ejemplo. Los que no aparecen usan una toma por defecto (Toma()).
# Los que requieren un argumento obligatorio (mesh/imagen) deben tener receta.
RECETAS = {
    "arcball_example": [Toma(args=["assets/bunny.obj"], cuadros=2)],
    "lapped_hatching": [Toma(args=["assets/bunny.obj"], cuadros=2)],
    "chroma_key": [Toma(args=["assets/boo.png"], cuadros=2)],
    "image_pixel": [Toma(args=["assets/santiago.png"], cuadros=2)],
    "image_texture": [Toma(args=["assets/bricks.jpg"], cuadros=2)],
    "texture_viewer": [Toma(args=["assets/dice_cube.obj"], cuadros=2)],
    # suggestive contours: relleno papel y los campos de curvatura
    "suggestive_contours": [
        Toma(sufijo="papel", args=["assets/samus/posed.obj"], cuadros=2),
        Toma(sufijo="radial", args=["assets/samus/posed.obj"], teclas=["V", "V", "V"], cuadros=2),
    ],
    # ray tracing: varios estados en la misma escena
    "raytracing_cpu": [
        Toma(sufijo="raster", cuadros=2),
        Toma(sufijo="cajas", teclas=["B"], cuadros=2),
        Toma(sufijo="rt", teclas=["T"], cuadros=600),  # deja completar el render progresivo
    ],
    "raytracing_basico": [Toma(cuadros=600)],  # render progresivo por CPU
    "esqueleto_lbs": [
        Toma(sufijo="reposo", cuadros=2),
        Toma(sufijo="doblado", teclas=["Z"] * 8 + ["X"] * 6, cuadros=2),
        Toma(sufijo="wireframe", teclas=["W"] + ["Z"] * 8 + ["X"] * 6, cuadros=2),
        Toma(sufijo="colision", teclas=["C"] + ["Z"] * 8 + ["X"] * 6, cuadros=2),
    ],
    "animacion_esqueletica": [Toma(teclas=["PERIOD"] * 8, cuadros=2)],  # avanza la animacion
    "lsystem": [
        Toma(sufijo="arbol", cuadros=2),
        Toma(sufijo="monopodial", args=["--especie", "monopodial"], cuadros=2),
        Toma(sufijo="arbusto", args=["--especie", "arbusto"], cuadros=2),
        Toma(sufijo="binario", args=["--especie", "binario"], cuadros=2),
        Toma(sufijo="desorden", teclas=["D"], cuadros=2),
    ],
    # rendering de volumen: los tres modos de composicion sobre el mismo fantoma
    "raycasting_volumen": [
        Toma(sufijo="mip", cuadros=2),
        Toma(sufijo="promedio", teclas=["M"], cuadros=2),
        Toma(sufijo="compositing", teclas=["M", "M"], cuadros=2),
    ],
    "skinning": [
        Toma(sufijo="lbs", teclas=["PERIOD"] * 6, cuadros=2),
        Toma(sufijo="dqs", teclas=["M"] + ["PERIOD"] * 6, cuadros=2),
        Toma(sufijo="esqueleto", teclas=["E", "PERIOD", "PERIOD"], cuadros=2),
    ],
}


def _receta(nombre):
    return RECETAS.get(nombre, [Toma()])


# Ejemplos que no se pueden capturar con este harness (no son pyglet de una
# ventana, o bloquean). Se reportan como "no aplica", no como falla.
EXCLUIDOS = {
    "edo_case_studies": "usa matplotlib (plt.show), no es una escena pyglet",
    "pyvista_orbital": "usa pyvista/VTK, con su propio loop de render",
    "terrain_generation": "menu interactivo por consola (input), bloquea sin stdin",
}


# --------------------------------------------------------------------------
# Worker: captura una sola toma en el proceso actual (parchea pyglet)
# --------------------------------------------------------------------------

def _capturar(nombre, indice, salida):
    import pyglet

    # Evitar que el GC borre objetos GL cuya unica referencia viva era una
    # variable local del callback. Con la ventana real corriendo el loop, esas
    # locales siguen vivas mientras `app.run()` no retorna; aca `app.run` es
    # no-op, asi que `main()` retorna y un objeto referenciado solo por su id
    # entero (ej. el depth texture de un FBO en shadow_mapping) se recolecta y
    # su textura GL se borra, rompiendo el bind posterior. El worker es un
    # subproceso efimero, asi que no borrar nada es inocuo.
    pyglet.image.Texture.__del__ = lambda self: None
    for clase in ("Framebuffer", "Renderbuffer"):
        if hasattr(pyglet.image, clase):
            setattr(getattr(pyglet.image, clase), "__del__", lambda self: None)

    ventanas = []
    ticks = []

    clase_original = pyglet.window.Window

    class VentanaCapturada(clase_original):
        def __init__(self, *args, **kwargs):
            kwargs["visible"] = False
            super().__init__(*args, **kwargs)
            ventanas.append(self)

    pyglet.window.Window = VentanaCapturada
    pyglet.app.run = lambda *a, **k: None
    pyglet.clock.schedule_interval = lambda f, intervalo=0, *a, **k: ticks.append((f, a))
    pyglet.clock.schedule_interval_soft = lambda f, intervalo=0, *a, **k: ticks.append((f, a))
    pyglet.clock.schedule = lambda f, *a, **k: ticks.append((f, a))

    import caja_de_juguetes

    comando = caja_de_juguetes.grafica_cli.commands[nombre]
    toma = _receta(nombre)[indice]

    # armar el ejemplo (crea ventana + escena; app.run no hace nada y retorna)
    comando.main(args=list(toma.args), standalone_mode=False)

    if not ventanas:
        raise RuntimeError(f"'{nombre}' no creo ninguna ventana")
    ventana = ventanas[0]
    ventana.switch_to()

    _disparar_teclas(ventana, toma.teclas)
    _avanzar(ticks, toma.cuadros)

    dibujar = _handler(ventana, "on_draw")
    if dibujar is None:
        raise RuntimeError(f"'{nombre}' no tiene handler on_draw")
    dibujar()

    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    pyglet.image.get_buffer_manager().get_color_buffer().save(salida)


def _handler(ventana, nombre_evento):
    """Busca un handler de evento en el stack de la ventana (@window.event o push_handlers)."""
    for marco in ventana._event_stack:
        if nombre_evento in marco:
            return marco[nombre_evento]
    return None


def _disparar_teclas(ventana, teclas):
    import pyglet
    on_key = _handler(ventana, "on_key_press")
    if on_key is None:
        return
    for tecla in teclas:
        # pyglet encola dispatch_event hasta que corre el loop, asi que llamamos directo
        on_key(getattr(pyglet.window.key, tecla), 0)


def _avanzar(ticks, cuadros, dt=1 / 60.0):
    """Avanza el reloj llamando a mano las funciones agendadas (animacion, render progresivo)."""
    for _ in range(cuadros):
        for funcion, extra in ticks:
            funcion(dt, *extra)


# --------------------------------------------------------------------------
# Orquestador: lanza un subproceso por toma (aislamiento + robustez)
# --------------------------------------------------------------------------

def _nombre_archivo(nombre, toma):
    return f"{nombre}_{toma.sufijo}.png" if toma.sufijo else f"{nombre}.png"


def _generar(nombres, directorio):
    destino = Path(directorio)
    destino.mkdir(parents=True, exist_ok=True)

    ok, fallos, omitidos = 0, [], []
    for nombre in nombres:
        if nombre in EXCLUIDOS:
            omitidos.append(nombre)
            print(f"[no aplica] {nombre}: {EXCLUIDOS[nombre]}")
            continue
        if _requiere_argumento(nombre) and nombre not in RECETAS:
            omitidos.append(nombre)
            print(f"[omitido] {nombre}: requiere un argumento y no tiene receta")
            continue

        for indice, toma in enumerate(_receta(nombre)):
            salida = destino / _nombre_archivo(nombre, toma)
            try:
                resultado = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "shot",
                     nombre, "--indice", str(indice), "--salida", str(salida)],
                    cwd=str(RAIZ), capture_output=True, text=True, timeout=180,
                )
            except subprocess.TimeoutExpired:
                fallos.append(nombre)
                print(f"[falla] {nombre}: timeout")
                continue
            if resultado.returncode == 0 and salida.exists():
                ok += 1
                print(f"[ok] {salida.name}")
            else:
                fallos.append(nombre)
                error = (resultado.stderr.strip().splitlines() or ["(sin stderr)"])[-1]
                print(f"[falla] {nombre}: {error}")

    print(f"\nListo: {ok} imagenes, {len(fallos)} fallas, {len(omitidos)} omitidos en {destino}/")


def _contacto(directorio, salida, columnas=6, celda=300, margen=8):
    """Arma una grilla con miniaturas de todos los PNG de un directorio."""
    from math import ceil
    from PIL import Image, ImageDraw, ImageFont

    destino = Path(directorio)
    salida = Path(salida)
    archivos = sorted(p for p in destino.glob("*.png") if p.resolve() != salida.resolve())
    if not archivos:
        print(f"No hay PNG en {destino}/")
        return

    etiqueta_alto = 22
    ancho_celda, alto_celda = celda, celda + etiqueta_alto
    filas = ceil(len(archivos) / columnas)
    ancho = columnas * ancho_celda + margen * (columnas + 1)
    alto = filas * alto_celda + margen * (filas + 1)

    hoja = Image.new("RGB", (ancho, alto), (20, 20, 24))
    dibujo = ImageDraw.Draw(hoja)
    fuente = ImageFont.truetype(
        str(RAIZ / "assets" / "FiraCode" / "FiraCode-Regular.ttf"), 13
    )

    for i, archivo in enumerate(archivos):
        fila, columna = divmod(i, columnas)
        x = margen + columna * (ancho_celda + margen)
        y = margen + fila * (alto_celda + margen)

        miniatura = Image.open(archivo).convert("RGB")
        miniatura.thumbnail((celda, celda))
        hoja.paste(
            miniatura,
            (x + (celda - miniatura.width) // 2, y + (celda - miniatura.height) // 2),
        )
        dibujo.text((x + 2, y + celda + 3), archivo.stem, fill=(210, 210, 215), font=fuente)

    salida.parent.mkdir(parents=True, exist_ok=True)
    hoja.save(salida)
    print(f"Contact sheet: {salida} ({len(archivos)} miniaturas, {columnas}x{filas})")


def _requiere_argumento(nombre):
    import caja_de_juguetes
    comando = caja_de_juguetes.grafica_cli.commands[nombre]
    return any(
        isinstance(p, click.Argument) and p.required for p in comando.params
    )


def _todos_los_nombres():
    import caja_de_juguetes
    return sorted(caja_de_juguetes.grafica_cli.commands)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

@click.group()
def cli():
    pass


@cli.command("shot", short_help="Worker: captura una toma en el proceso actual")
@click.argument("nombre")
@click.option("--indice", type=int, default=0)
@click.option("--salida", required=True)
def shot(nombre, indice, salida):
    _capturar(nombre, indice, salida)


@cli.command("uno", short_help="Screenshots de un ejemplo (todas sus tomas)")
@click.argument("nombre")
@click.option("--dir", "directorio", default=DIRECTORIO_POR_DEFECTO)
def uno(nombre, directorio):
    _generar([nombre], directorio)


@cli.command("galeria", short_help="Screenshots de todos los ejemplos")
@click.option("--dir", "directorio", default=DIRECTORIO_POR_DEFECTO)
@click.option("--solo", default=None, help="Filtra por subcadena del nombre")
@click.option("--contacto", is_flag=True, help="Arma el contact sheet al terminar")
def galeria(directorio, solo, contacto):
    nombres = _todos_los_nombres()
    if solo:
        nombres = [n for n in nombres if solo in n]
    _generar(nombres, directorio)
    if contacto:
        _contacto(directorio, Path(directorio) / "_contacto.png")


@cli.command("contacto", short_help="Arma un contact sheet con las imagenes ya generadas")
@click.option("--dir", "directorio", default=DIRECTORIO_POR_DEFECTO)
@click.option("--salida", default=None, help="PNG de salida (default <dir>/_contacto.png)")
@click.option("--columnas", type=int, default=6)
def contacto(directorio, salida, columnas):
    _contacto(directorio, salida or (Path(directorio) / "_contacto.png"), columnas=columnas)


if __name__ == "__main__":
    cli()
