"""Achurado en tiempo real con tonal art maps y lapped textures.

Reproduce el método de Praun, Hoppe, Webb y Finkelstein (2001), "Real-time
hatching". La superficie se dibuja con trazos de pluma cuya densidad sigue la
iluminación, cuya dirección sigue la forma y que no parpadean al mover la
cámara o la luz.

Preproceso, al iniciar (el resultado queda en la carpeta de caché, así la
segunda ejecución parte de inmediato):

1. Tonal art map (tam.py): seis imágenes de achurado, de claro a oscuro, cada
   una con sus niveles de mipmap. Los trazos de un tono aparecen en todos los
   más oscuros y los de un nivel en todos los más finos.
2. Campo de direcciones (campo.py): la dirección de los trazos en cada cara,
   alineada con las curvaturas principales de una versión simplificada de la
   malla y suavizada.
3. Lapped textures (parches.py): la malla se cubre con parches superpuestos,
   cada uno aplanado de modo que el eje horizontal de la textura siga el campo.

Cada cuadro:

1. Pre-pasada de profundidad: la malla se dibuja una vez solo en el buffer de
   profundidad, así luego pasan solo los píxeles de la superficie visible.
2. Parches, en el orden en que se crearon, sobre una capa de fondo opaca. En
   cada vértice la luz da un tono continuo entre 0 (papel) y 6; los dos tonos
   del TAM que lo encierran reciben pesos y la GPU interpola esos pesos sobre
   la cara, así que un píxel mezcla hasta seis imágenes del TAM. El alfa del
   contorno del parche lo funde con los parches anteriores.
3. Siluetas: las aristas entre una cara que mira a la cámara y otra que no,
   dibujadas como líneas en un geometry shader.

Controles:
- arrastrar: rota la cámara (arcball). Rueda: zoom; la densidad de trazos en
  pantalla se mantiene porque el mipmap elige un nivel con menos trazos.
- flechas: mueven la luz (azimut y elevación, relativos a la cámara).
- - =: exposición. A Z: luz ambiente. Juntas forman la función de
  transferencia que lleva la luz al tono.
- M: esquema de mezcla de tonos (6 vías, o un tono por vértice o por cara).
- U: umbral de tono (trazos negros que se alargan en vez de grises).
- P: un tono al azar por parche, para ver los parches de la lapped texture.
- C: campo de direcciones. S: siluetas. T: panel del TAM.
- E: estilo del TAM. , .: tamaño de los trazos sobre la superficie.
- O: dónde va el color de un modelo que trae texturas o colores, como Samus:
  en el papel, bajo la tinta negra, o en los trazos, sobre papel blanco.
- R: reinicia la cámara y las perillas.
"""

import ctypes
import hashlib
from pathlib import Path

import click
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import grafica.transformations as tr
from grafica.arcball import Arcball
from grafica.ui import InfoPanel, ui_overlay
from grafica.utils import load_pipeline

from . import malla
from .campo import campo_de_direcciones, marcos_de_caras
from .parches import cubrir, generar_contorno
from .tam import ESTILOS, cargar_o_generar

CARPETA = Path(__file__).parent
RAIZ = CARPETA.parent.parent

ESQUEMAS = (
    "6 vías: tono continuo por vértice",
    "tono redondeado por vértice",
    "tono continuo por cara",
    "tono redondeado por cara",
)

# dónde va el color del modelo: en ninguna parte, en el papel o en los trazos
USOS_TEXTURA = ("sobre papel blanco", "sobre la textura", "del color de la textura")

# filtrado anisotrópico: con la superficie de canto, el mipmap elige el nivel
# según el eje menos comprimido y los trazos no se emborronan
ANISOTROPIA = 16.0

# unidades de textura de cada muestreador
UNIDAD_CLAROS, UNIDAD_OSCUROS, UNIDAD_CONTORNO, UNIDAD_MODELO = 0, 1, 2, 3

# vista de tres cuartos, desde arriba y algo de frente
OJO_INICIAL = np.array([-1.17, 0.92, 4.55])
OBJETIVO = np.array([0.0, 0.07, 0.0])

# perillas: la luz viene de arriba y de detrás del conejo, y la función de
# transferencia deja en blanco solo las caras más iluminadas
PERILLAS = {
    "esquema": 0,
    "umbral": False,
    "parches_al_azar": False,
    "siluetas": True,
    "campo": False,
    "panel_tam": False,
    "azimut": 40.0,
    "elevacion": 35.0,
    "ambiente": 0.30,
    "exposicion": 0.82,
}


def _subir_textura(niveles, formato_interno, formato, repetir):
    """Textura 2D con sus niveles de mipmap dados uno por uno.

    Los niveles no se generan con glGenerateMipmap: en el TAM cada nivel tiene
    sus propios trazos, del mismo ancho en píxeles.
    """
    textura = GL.GLuint()
    GL.glGenTextures(1, ctypes.byref(textura))
    GL.glBindTexture(GL.GL_TEXTURE_2D, textura)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    for nivel, imagen in enumerate(niveles):
        datos = np.ascontiguousarray(np.round(np.clip(imagen, 0.0, 1.0) * 255).astype(np.uint8))
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, nivel, formato_interno, imagen.shape[1], imagen.shape[0], 0,
            formato, GL.GL_UNSIGNED_BYTE, datos.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, len(niveles) - 1)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    envoltura = GL.GL_REPEAT if repetir else GL.GL_CLAMP_TO_EDGE
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, envoltura)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, envoltura)
    if ANISOTROPIA > 1 and pyglet.gl.gl_info.have_extension("GL_EXT_texture_filter_anisotropic"):
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_ANISOTROPY, ANISOTROPIA)
    return textura


def _texturas_tam(niveles):
    """Los tonos 1 a 3 en los canales RGB de una textura y los 4 a 6 en otra."""
    claros = [np.moveaxis(nivel[0:3], 0, -1) for nivel in niveles]
    oscuros = [np.moveaxis(nivel[3:6], 0, -1) for nivel in niveles]
    return (
        _subir_textura(claros, GL.GL_RGB8, GL.GL_RGB, repetir=True),
        _subir_textura(oscuros, GL.GL_RGB8, GL.GL_RGB, repetir=True),
    )


def _textura_modelo(capas):
    """Las imágenes y colores del modelo como capas de una textura 2D array.

    Todas las capas tienen el mismo lado, así una cara elige la suya con un
    índice, sin cambiar de textura entre caras.
    """
    textura = GL.GLuint()
    GL.glGenTextures(1, ctypes.byref(textura))
    GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, textura)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    cantidad, alto, ancho, _ = capas.shape
    datos = np.ascontiguousarray(capas)
    GL.glTexImage3D(
        GL.GL_TEXTURE_2D_ARRAY, 0, GL.GL_RGBA8, ancho, alto, cantidad, 0,
        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, datos.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    )
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D_ARRAY)
    GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    return textura


def _textura_contorno(alfa):
    """Alfa del contorno con mipmaps que conservan un borde transparente.

    Fuera del cuadrado [0, 1]^2 la textura repite su borde, así que el borde de
    cada nivel tiene que ser transparente; si no, el parche se extendería por
    toda la superficie que crece más allá del contorno.
    """
    niveles = [alfa]
    while niveles[-1].shape[0] > 1:
        lado = niveles[-1].shape[0] // 2
        siguiente = niveles[-1].reshape(lado, 2, lado, 2).mean(axis=(1, 3))
        if lado >= 4:
            siguiente[[0, -1], :] = 0.0
            siguiente[:, [0, -1]] = 0.0
        niveles.append(siguiente)
    return _subir_textura(niveles, GL.GL_R8, GL.GL_RED, repetir=False)


def _vertices_de_capa(modelo, normales_esquina, lapped, caras_capa, uv, margen, parche):
    """Arreglos por vértice para dibujar una lista de caras, cara por cara.

    Cada esquina de cada cara es un vértice propio: lleva su normal, la normal
    de la cara (para los esquemas de un tono por cara), la coordenada de la
    textura original del modelo, la coordenada del contorno de su parche y su
    margen y, del parche, el lado del contorno, el corrimiento del TAM y el
    tono al azar.
    """
    vertices, caras = modelo["vertices"], modelo["caras"]
    normal_cara = marcos_de_caras(vertices, caras)[0]
    esquinas = caras[caras_capa].reshape(-1)
    parche_por_esquina = np.repeat(parche, 3)
    columnas = {
        "position": vertices[esquinas],
        "normal": normales_esquina[caras_capa].reshape(-1, 3),
        "normal_cara": np.repeat(normal_cara[caras_capa], 3, axis=0),
        "uv_modelo": modelo["uv"][caras_capa].reshape(-1, 2),
        "capa_modelo": np.repeat(modelo["capa"][caras_capa], 3),
        "uv_contorno": uv.reshape(-1, 2),
        "margen": margen.reshape(-1),
        "escala_parche": lapped["escala"][parche_por_esquina],
        "desplazamiento": lapped["desplazamiento"][parche_por_esquina],
        "tono_parche": lapped["tono"][parche_por_esquina],
    }
    return {nombre: np.asarray(valores, dtype=np.float32) for nombre, valores in columnas.items()}


def _aristas_con_opuestos(vertices, caras):
    """Posiciones (a, p0, p1, b) de cada arista para el geometry shader.

    La cara A = (p0, p1, a) recorre la arista de p0 a p1 y la cara B la recorre
    al revés. En las aristas de borde no hay cara B y b se repite igual a a.
    """
    malla = tm.Trimesh(vertices, caras, process=False)
    cara_a = malla.face_adjacency[:, 0]
    v0, v1 = malla.face_adjacency_edges.T
    opuesto_a, opuesto_b = malla.face_adjacency_unshared.T
    # orientar la arista como la recorre la cara A
    posicion_v0 = np.argmax(caras[cara_a] == v0[:, None], axis=1)
    directa = caras[cara_a, (posicion_v0 + 1) % 3] == v1
    p0 = np.where(directa, v0, v1)
    p1 = np.where(directa, v1, v0)
    interiores = np.stack([opuesto_a, p0, p1, opuesto_b], axis=1)

    # aristas de borde: las que aparecen en una sola cara
    aristas = caras[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    unicas = tm.grouping.group_rows(np.sort(aristas, axis=1), require_count=1)
    cara = unicas // 3
    opuesto = caras[cara, (unicas % 3 + 2) % 3]
    borde = np.stack([opuesto, aristas[unicas, 0], aristas[unicas, 1], opuesto], axis=1)

    indices = np.concatenate([interiores, borde]).reshape(-1)
    return vertices[indices].astype(np.float32)


def _segmentos_del_campo(vertices, caras, campo):
    """Un segmento corto por cara, centrado y a lo largo del campo."""
    normal = marcos_de_caras(vertices, caras)[0]
    centro = vertices[caras].mean(axis=1) + 0.004 * normal
    largo = 0.35 * np.linalg.norm(vertices[caras[:, 1]] - vertices[caras[:, 0]], axis=1).mean()
    extremos = np.stack([centro - largo * campo, centro + largo * campo], axis=1)
    return extremos.reshape(-1).astype(np.float32)


def _campo_y_parches(base, vertices, caras, direccion_principal, parche, semilla, directorio):
    """Campo de direcciones y lapped texture de la malla, de la caché o calculados.

    La huella del archivo incluye la malla, los parámetros y el código de los
    módulos que los calculan: si cambia cualquiera, se vuelven a calcular.
    """
    huella = hashlib.sha1(f"{direccion_principal}-{parche}-{semilla}".encode())
    for arreglo in (*base, vertices, caras):
        huella.update(np.ascontiguousarray(arreglo).tobytes())
    for fuente in (CARPETA / "campo.py", CARPETA / "parches.py", CARPETA / "malla.py",
                   RAIZ / "examples" / "simplificacion_qem" / "qem.py", RAIZ / "grafica" / "half_edge.py"):
        huella.update(fuente.read_bytes())
    archivo = Path(directorio) / f"parches_{huella.hexdigest()[:12]}.npz"

    if archivo.exists():
        print(f"Campo de direcciones y parches leídos de {archivo}")
        datos = dict(np.load(archivo))
        return datos.pop("campo"), datos

    print("Campo de direcciones:")
    campo, curvatura_caras = campo_de_direcciones(vertices, caras, direccion_principal, base)
    # el modelo mide 2 unidades en su lado más largo
    escala_parche = 2.0 * parche
    print(f"Lapped textures (contorno de hasta {escala_parche:.2f} unidades):")
    lapped = cubrir(vertices, caras, campo, curvatura_caras, escala_parche, generar_contorno(), semilla)
    Path(directorio).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archivo, campo=campo, **lapped)
    return campo, lapped


def _direccion_luz(azimut, elevacion):
    """Vector hacia la luz en el espacio de la cámara (x derecha, y arriba, z hacia el ojo)."""
    azimut, elevacion = np.radians(azimut), np.radians(elevacion)
    return np.array([
        np.cos(elevacion) * np.sin(azimut),
        np.sin(elevacion),
        np.cos(elevacion) * np.cos(azimut),
    ], dtype=np.float32)


def _matriz(m):
    return m.astype(np.float32).reshape(16, 1, order="F")


@click.command("lapped_hatching", short_help="Achurado en tiempo real con tonal art maps y lapped textures")
@click.argument("mesh_path", type=click.Path(exists=True), default="assets/bunny.obj")
@click.option("--width", type=int, default=960)
@click.option("--height", type=int, default=720)
@click.option("--tam", "estilo", type=click.Choice(sorted(ESTILOS)), default="conejo",
              help="Estilo de los trazos del tonal art map")
@click.option("--curvatura", type=click.Choice(["maxima", "minima"]), default="minima",
              help="Dirección principal de curvatura que siguen los trazos")
@click.option("--parche", type=float, default=0.25,
              help="Lado máximo del contorno de un parche, como fracción del tamaño del modelo")
@click.option("--trazos", type=float, default=0.13,
              help="Lado de una repetición del TAM, como fracción del tamaño del modelo")
@click.option("--subdividir", type=int, default=None,
              help=f"Pasos de subdivisión de la malla; por omisión, los que falten para {malla.CARAS_MINIMAS} caras")
@click.option("--arriba", type=click.Choice(["auto", "y", "z"]), default="auto",
              help="Eje que apunta hacia arriba en el archivo; auto usa z para los STL e y para el resto")
@click.option("--semilla", type=int, default=11, help="Semilla de la ubicación de los parches")
@click.option("--cache", "directorio_cache", type=click.Path(), default=str(RAIZ / "cache" / "lapped_hatching"),
              help="Carpeta donde se guardan los TAM y los parches calculados")
def lapped_hatching(mesh_path, width, height, estilo, curvatura, parche, trazos, subdividir, arriba, semilla,
                    directorio_cache):
    if arriba == "auto":
        arriba = "z" if mesh_path.lower().endswith(".stl") else "y"
    print(f"Cargando {mesh_path} (eje {arriba} hacia arriba)...")
    modelo = malla.cargar(mesh_path, subdividir, arriba)
    base, vertices, caras = modelo["base"], modelo["vertices"], modelo["caras"]
    normales = malla.normales_por_esquina(vertices, caras)
    print(f"  {len(base[1])} caras; {len(caras)} caras, {modelo['descripcion']}")
    if modelo["con_textura"]:
        print(f"  textura propia: {len(modelo['capas'])} capas de {modelo['capas'].shape[1]} píxeles")

    print("Tonal art maps:")
    niveles_tam = {nombre: cargar_o_generar(nombre, directorio_cache) for nombre in sorted(ESTILOS)}
    print(f"  estilos: {', '.join(niveles_tam)}")

    campo, lapped = _campo_y_parches(base, vertices, caras, curvatura, parche, semilla, directorio_cache)
    datos_parches = _vertices_de_capa(
        modelo, normales, lapped, lapped["caras"], lapped["uv"], lapped["margen"], lapped["parche"]
    )
    datos_fondo = _vertices_de_capa(
        modelo, normales, lapped, np.arange(len(caras)), lapped["fondo_uv"],
        np.ones((len(caras), 3)), lapped["fondo_parche"],
    )
    cantidad_parches = len(lapped["escala"]) - 1
    veces_por_cara = len(lapped["caras"]) / len(caras)

    window = pyglet.window.Window(width, height, "Achurado en tiempo real")

    achurado = load_pipeline(CARPETA / "achurado_vertex.glsl", CARPETA / "achurado_fragment.glsl")
    profundidad = load_pipeline(CARPETA / "profundidad_vertex.glsl", CARPETA / "profundidad_fragment.glsl")
    lineas = load_pipeline(CARPETA / "lineas_vertex.glsl", CARPETA / "lineas_fragment.glsl")
    panel_tam = load_pipeline(CARPETA / "tam_vertex.glsl", CARPETA / "tam_fragment.glsl")
    silueta = pyglet.graphics.shader.ShaderProgram(*(
        pyglet.graphics.shader.Shader((CARPETA / f"silueta_{etapa}.glsl").read_text(), etapa)
        for etapa in ("vertex", "geometry", "fragment")
    ))

    parches_gpu, fondo_gpu = (
        achurado.vertex_list(
            len(datos["position"]), GL.GL_TRIANGLES,
            **{nombre: ("f", valores.reshape(-1)) for nombre, valores in datos.items()},
        )
        for datos in (datos_parches, datos_fondo)
    )
    malla_gpu = profundidad.vertex_list(
        3 * len(caras), GL.GL_TRIANGLES, position=("f", vertices[caras].reshape(-1).astype(np.float32)),
    )
    aristas = _aristas_con_opuestos(vertices, caras)
    aristas_gpu = silueta.vertex_list(len(aristas), GL.GL_LINES_ADJACENCY, position=("f", aristas.reshape(-1)))
    segmentos = _segmentos_del_campo(vertices, caras, campo)
    campo_gpu = lineas.vertex_list(len(segmentos) // 3, GL.GL_LINES, position=("f", segmentos))

    # panel del TAM, centrado sobre el pie de la ventana: seis columnas de 128
    # píxeles y tres filas (niveles de 32, 64 y 128 texeles) con separaciones
    ancho_panel, alto_panel = 6 * 128 + 7 * 8, 32 + 64 + 128 + 4 * 8
    origen_panel = ((width - ancho_panel) // 2, 30)
    x0, y0 = 2.0 * origen_panel[0] / width - 1.0, 2.0 * origen_panel[1] / height - 1.0
    x1, y1 = x0 + 2.0 * ancho_panel / width, y0 + 2.0 * alto_panel / height
    panel_tam_gpu = panel_tam.vertex_list(
        6, GL.GL_TRIANGLES, position=("f", [x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1]),
    )

    texturas_tam = {nombre: _texturas_tam(niveles) for nombre, niveles in niveles_tam.items()}
    textura_contorno = _textura_contorno(generar_contorno())
    textura_modelo = _textura_modelo(modelo["capas"])

    projection = tr.perspective(35, width / height, 0.1, 20.0)
    vista_inicial = tr.lookAt(OJO_INICIAL, OBJETIVO, np.array([0.0, 1.0, 0.0]))
    arcball = Arcball(np.linalg.inv(vista_inicial), np.array((width, height), dtype=float), 1.5, OBJETIVO)

    valores_iniciales = dict(PERILLAS, estilo=estilo, trazos=trazos, textura=int(modelo["con_textura"]))
    estado = dict(valores_iniciales)

    panel = (
        InfoPanel(x=14, y_top=height - 22, background=(25, 22, 18), background_width=560)
        .add("tam")
        .add("mezcla")
        .add("luz")
        .add("brillo")
        .add("parches")
        .footer("M mezcla  U umbral  P parches  C campo  S siluetas  T TAM  E estilo  O textura  "
                ", . trazos  R reset")
    )

    def apply_state():
        achurado.use()
        achurado["direccion_luz"] = tuple(_direccion_luz(estado["azimut"], estado["elevacion"]))
        achurado["exposicion"] = float(estado["exposicion"])
        achurado["ambiente"] = float(estado["ambiente"])
        # el modelo mide 2 unidades en su lado más largo
        achurado["lado_tam"] = float(2.0 * estado["trazos"])
        achurado["tono_por_parche"] = int(estado["parches_al_azar"])
        achurado["esquema_mezcla"] = estado["esquema"]
        achurado["umbral"] = int(estado["umbral"])
        achurado["usar_textura"] = estado["textura"]
        achurado["tam_claros"] = UNIDAD_CLAROS
        achurado["tam_oscuros"] = UNIDAD_OSCUROS
        achurado["contorno"] = UNIDAD_CONTORNO
        achurado["textura_modelo"] = UNIDAD_MODELO

        panel["tam"] = f"TAM {estado['estilo']} {USOS_TEXTURA[estado['textura']]}; trazos de {estado['trazos']:.3f}"
        panel["mezcla"] = "mezcla: " + ESQUEMAS[estado["esquema"]] + (", con umbral" if estado["umbral"] else "")
        panel["luz"] = f"luz (flechas): azimut {estado['azimut']:.0f}, elevación {estado['elevacion']:.0f}"
        if estado["parches_al_azar"]:
            panel["brillo"] = "tono: uno al azar por parche"
        else:
            panel["brillo"] = f"brillo = {estado['ambiente']:.2f} (A Z) + {estado['exposicion']:.2f} (- =) N.L"
        panel["parches"] = f"{cantidad_parches} parches; cada cara se dibuja {veces_por_cara:.2f} veces"
        print(
            f"[achurado] tam={estado['estilo']} trazos={estado['trazos']:.3f} mezcla={estado['esquema']} "
            f"umbral={estado['umbral']} parches_al_azar={estado['parches_al_azar']} textura={estado['textura']} "
            f"azimut={estado['azimut']:.0f} elevacion={estado['elevacion']:.0f} "
            f"ambiente={estado['ambiente']:.2f} exposicion={estado['exposicion']:.2f}"
        )

    apply_state()

    @window.event
    def on_draw():
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        window.clear()
        vista = np.linalg.inv(arcball.pose)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        # las dos pasadas de la malla se empujan un poco hacia atrás en
        # profundidad, así las siluetas quedan delante de la superficie
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)

        # 1. solo profundidad
        GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
        profundidad.use()
        profundidad["view"] = _matriz(vista)
        profundidad["projection"] = _matriz(projection)
        malla_gpu.draw(GL.GL_TRIANGLES)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)

        # 2. la capa de fondo, opaca, y encima los parches en orden: cada uno
        # se mezcla sobre los anteriores con el alfa de su contorno. Solo pasan
        # los píxeles de la superficie visible
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthMask(GL.GL_FALSE)
        claros, oscuros = texturas_tam[estado["estilo"]]
        for unidad, textura in ((UNIDAD_CLAROS, claros), (UNIDAD_OSCUROS, oscuros), (UNIDAD_CONTORNO, textura_contorno)):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unidad)
            GL.glBindTexture(GL.GL_TEXTURE_2D, textura)
        GL.glActiveTexture(GL.GL_TEXTURE0 + UNIDAD_MODELO)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, textura_modelo)
        achurado.use()
        achurado["view"] = _matriz(vista)
        achurado["projection"] = _matriz(projection)
        fondo_gpu.draw(GL.GL_TRIANGLES)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        parches_gpu.draw(GL.GL_TRIANGLES)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        # 3. siluetas y, si se pide, el campo de direcciones
        if estado["siluetas"]:
            silueta.use()
            silueta["view"] = _matriz(vista)
            silueta["projection"] = _matriz(projection)
            silueta["posicion_camara"] = tuple(arcball.pose[:3, 3].astype(np.float32))
            silueta["tamano_ventana"] = (float(width), float(height))
            silueta["ancho_px"] = 1.6
            aristas_gpu.draw(GL.GL_LINES_ADJACENCY)
        if estado["campo"]:
            lineas.use()
            lineas["view"] = _matriz(vista)
            lineas["projection"] = _matriz(projection)
            lineas["color"] = (0.75, 0.1, 0.1)
            campo_gpu.draw(GL.GL_LINES)

        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_BLEND)

        if estado["panel_tam"]:
            GL.glDisable(GL.GL_DEPTH_TEST)
            panel_tam.use()
            panel_tam["tam_claros"] = UNIDAD_CLAROS
            panel_tam["tam_oscuros"] = UNIDAD_OSCUROS
            panel_tam["origen"] = (float(origen_panel[0]), float(origen_panel[1]))
            panel_tam_gpu.draw(GL.GL_TRIANGLES)

        with ui_overlay():
            panel.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        teclas = pyglet.window.key
        alternar = {teclas.U: "umbral", teclas.P: "parches_al_azar", teclas.C: "campo",
                    teclas.S: "siluetas", teclas.T: "panel_tam"}
        if symbol in alternar:
            estado[alternar[symbol]] = not estado[alternar[symbol]]
        elif symbol == teclas.LEFT:
            estado["azimut"] -= 10.0
        elif symbol == teclas.RIGHT:
            estado["azimut"] += 10.0
        elif symbol == teclas.UP:
            estado["elevacion"] = min(estado["elevacion"] + 10.0, 90.0)
        elif symbol == teclas.DOWN:
            estado["elevacion"] = max(estado["elevacion"] - 10.0, -90.0)
        elif symbol == teclas.MINUS:
            estado["exposicion"] = max(estado["exposicion"] - 0.05, 0.0)
        elif symbol == teclas.EQUAL:
            estado["exposicion"] += 0.05
        elif symbol == teclas.A:
            estado["ambiente"] = min(estado["ambiente"] + 0.04, 1.0)
        elif symbol == teclas.Z:
            estado["ambiente"] = max(estado["ambiente"] - 0.04, 0.0)
        elif symbol == teclas.M:
            estado["esquema"] = (estado["esquema"] + 1) % len(ESQUEMAS)
        elif symbol == teclas.O and modelo["con_textura"]:
            estado["textura"] = (estado["textura"] + 1) % len(USOS_TEXTURA)
        elif symbol == teclas.E:
            nombres = sorted(ESTILOS)
            estado["estilo"] = nombres[(nombres.index(estado["estilo"]) + 1) % len(nombres)]
        elif symbol == teclas.COMMA:
            estado["trazos"] /= 1.25
        elif symbol == teclas.PERIOD:
            estado["trazos"] *= 1.25
        elif symbol == teclas.R:
            estado.update(valores_iniciales)
            arcball.reset()
        else:
            return
        apply_state()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            arcball.set_state(Arcball.STATE_ROTATE)
        elif button == pyglet.window.mouse.RIGHT:
            arcball.set_state(Arcball.STATE_PAN)
        arcball.down((x, y))

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        arcball.drag((x, y))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        arcball.scroll(scroll_y)

    pyglet.app.run()
