"""Lapped textures: cubrir una malla con parches de textura superpuestos.

Sigue a Praun, Finkelstein y Hoppe (2000), "Lapped textures", con la variante
de Praun et al. (2001) para achurado: el contorno del parche (una mancha
irregular con el borde difuminado) se guarda aparte de los trazos, así que el
TAM se repite dentro del parche con su propia escala.

Una superficie de topología arbitraria no tiene una parametrización global sin
cortes, pero sí se puede cubrir con parches pequeños, cada uno aplanado sobre
el plano con poca distorsión. El algoritmo repite estos pasos hasta cubrir la
malla:

1. Elegir una cara que aún no esté cubierta y un punto descubierto dentro de
   ella. Mientras falta cubrir la mitad de la malla se prefieren las zonas
   difíciles de parametrizar (mucha curvatura o un campo de direcciones que
   cambia brusco), para que los parches posteriores, dibujados encima, tapen
   su distorsión. El tamaño del contorno se ajusta a la curvatura de la zona.
2. Hacer crecer un parche de superficie alrededor de ese punto, cara por cara,
   en orden de distancia. Una cara entra si el parche sigue siendo un disco
   topológico, si la arista por la que entra todavía toca el contorno en el
   espacio de textura y no es un pliegue, si no se dobla demasiado respecto de
   la semilla y si no entra desde una cara mal parametrizada.
3. Parametrizar el parche: buscar coordenadas (u, v) por vértice tales que, en
   cada cara, los vectores del campo (S, T) se lleven a los ejes de la
   textura. Es un problema de mínimos cuadrados lineal y disperso. Si la nueva
   parametrización deja partes del contorno sin superficie, el parche sigue
   creciendo.
4. Marcar como cubiertas las caras bien parametrizadas que caen enteras en la
   parte opaca del contorno.

Al dibujar, cada parche usa sus coordenadas de textura y el alfa del contorno
se mezcla sobre los parches anteriores. Como los trazos de achurado no tienen
componentes de baja frecuencia, las costuras entre parches no se notan.

El módulo no depende de OpenGL: se puede ejecutar y probar sin ventana.
"""

import heapq

import numpy as np
import scipy.sparse as sp
import trimesh as tm
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .campo import marcos_de_caras
from .malla import ANGULO_PLIEGUE_GRADOS

# puntos de muestra por cara (coordenadas baricéntricas) para medir cobertura:
# vértices un poco hacia adentro, puntos medios de aristas, centro y otros
# interiores. Una cara está cubierta cuando todas sus muestras lo están
_PESOS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
          (4, 1, 1), (1, 4, 1), (1, 1, 4), (3, 3, 1), (1, 3, 3), (3, 1, 3)]
MUESTRAS = np.array(_PESOS, dtype=np.float64) + 0.02
MUESTRAS /= MUESTRAS.sum(axis=1, keepdims=True)

# una cara se dobla demasiado si su normal se aleja más de esto de la semilla
ANGULO_MAXIMO_GRADOS = 85.0

# un parche no cruza un pliegue (ver malla.ANGULO_PLIEGUE_GRADOS): los trazos de
# cada lado se parametrizan aparte, como en un dibujo, y el corte coincide con
# la arista viva

# vueltas de crecer y reparametrizar por parche
VUELTAS_MAXIMAS = 4

# parches que se intentan desde una misma cara antes de darla por cubierta
INTENTOS_POR_CARA = 1

# deformación máxima de la textura en una cara antes de descartarla
ESTIRAMIENTO_MAXIMO = 2.0

# giro de la normal, en radianes, que se admite a lo ancho de un parche: fija
# el lado del contorno según la curvatura de la zona
RADIANES_POR_PARCHE = 2.0


def generar_contorno(lado=256, semilla=3, difuminado_px=3.0):
    """Alfa de una mancha irregular (el "splotch" del paper) en [0, 1]^2.

    El borde es una curva polar r(phi) con algunos lóbulos al azar; el alfa cae
    de 1 a 0 en `difuminado_px` píxeles alrededor de esa curva. La mancha no
    toca los bordes de la imagen, así el alfa es 0 fuera del cuadrado.
    """
    rng = np.random.default_rng(semilla)
    armonicos = np.arange(1, 9)
    fases = rng.uniform(0.0, 2.0 * np.pi, len(armonicos))
    amplitudes = rng.uniform(0.5, 1.0, len(armonicos)) / armonicos**0.9
    amplitudes[0] *= 0.3

    angulos = np.linspace(0.0, 2.0 * np.pi, 2048)
    borde = (amplitudes[:, None] * np.cos(armonicos[:, None] * angulos + fases[:, None])).sum(axis=0)
    borde = 0.22 + 0.24 * (borde - borde.min()) / (borde.max() - borde.min())

    coordenadas = (np.arange(lado) + 0.5) / lado - 0.5
    x, y = np.meshgrid(coordenadas, coordenadas)
    distancia = np.hypot(x, y)
    angulo = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    radio = np.interp(angulo, angulos, borde)
    return np.clip((radio - distancia) * lado / difuminado_px + 0.5, 0.0, 1.0).astype(np.float32)


def _muestrear(imagen, uv):
    """Valor bilineal de la imagen en coordenadas uv (N, 2); 0 fuera de [0, 1]^2.

    La fila 0 de la imagen corresponde a v = 0, igual que al subirla a OpenGL.
    """
    lado = imagen.shape[0]
    x = uv[:, 0] * lado - 0.5
    y = uv[:, 1] * lado - 0.5
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    relleno = np.pad(imagen, 1)
    x0 = np.clip(x0 + 1, 0, lado)
    y0 = np.clip(y0 + 1, 0, lado)
    valor = (
        relleno[y0, x0] * (1 - fx) * (1 - fy) + relleno[y0, x0 + 1] * fx * (1 - fy)
        + relleno[y0 + 1, x0] * (1 - fx) * fy + relleno[y0 + 1, x0 + 1] * fx * fy
    )
    afuera = (uv[:, 0] < 0) | (uv[:, 0] > 1) | (uv[:, 1] < 0) | (uv[:, 1] > 1)
    return np.where(afuera, 0.0, valor)


class _Malla:
    """Datos de la malla que el crecimiento consulta muchas veces.

    El crecimiento agrega las caras de a una, así que sus datos se guardan
    también como listas de Python: indexar una lista es mucho más rápido que
    indexar un arreglo de numpy elemento por elemento.
    """

    def __init__(self, vertices, caras, campo):
        self.vertices = vertices
        self.caras = caras
        self.campo = campo
        self.normal = marcos_de_caras(vertices, caras)[0]
        self.centroide = vertices[caras].mean(axis=1)

        # vecina[f, k]: cara al otro lado de la arista k = (caras[f, k], caras[f, k+1]),
        # o -1 si la arista es de borde. La arista k de la cara f es la fila 3f + k
        aristas = caras[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        pares = tm.grouping.group_rows(np.sort(aristas, axis=1), require_count=2)
        vecina = np.full(len(aristas), -1, dtype=np.int64)
        vecina[pares[:, 0]] = pares[:, 1] // 3
        vecina[pares[:, 1]] = pares[:, 0] // 3
        self.vecina = vecina.reshape(-1, 3)

        # pliegue[f, k]: la arista k de la cara f es una arista viva
        valida = self.vecina >= 0
        coseno = np.einsum("fkj,fj->fk", self.normal[np.maximum(self.vecina, 0)], self.normal)
        self.pliegue = valida & (coseno < np.cos(np.radians(ANGULO_PLIEGUE_GRADOS)))

        # forma de la cara vista desde cada arista k = (a, b): el vértice opuesto
        # c en coordenadas a lo largo y perpendicular a la arista, medidas en
        # largos de arista. Sirve para ubicar c en el espacio de textura con un
        # triángulo semejante al de la cara
        forma = np.zeros((len(caras), 3, 2))
        for k in range(3):
            a = vertices[caras[:, k]]
            b = vertices[caras[:, (k + 1) % 3]]
            c = vertices[caras[:, (k + 2) % 3]]
            eje = b - a
            largo2 = np.einsum("ij,ij->i", eje, eje)
            x = np.einsum("ij,ij->i", c - a, eje) / largo2
            perpendicular = c - a - x[:, None] * eje
            forma[:, k, 0] = x
            forma[:, k, 1] = np.linalg.norm(perpendicular, axis=1) / np.sqrt(largo2)

        self.caras_lista = caras.tolist()
        self.vecina_lista = self.vecina.tolist()
        self.pliegue_lista = self.pliegue.tolist()
        self.forma_lista = forma.tolist()
        self.centroide_lista = self.centroide.tolist()
        self.normal_lista = self.normal.tolist()
        self.campo_lista = campo.tolist()

        # dificultad de parametrizar cada cara: cuánto cambia el campo respecto
        # de sus vecinas (con simetría de 180 grados) y cuánto se dobla la
        # superficie
        dificultad = np.zeros(len(caras))
        for k in range(3):
            otra = self.vecina[:, k]
            valida = otra >= 0
            coseno_campo = np.einsum("ij,ij->i", campo[valida], campo[otra[valida]])
            coseno_normal = np.einsum("ij,ij->i", self.normal[valida], self.normal[otra[valida]])
            dificultad[valida] += (1.0 - coseno_campo**2) + 10.0 * (1.0 - coseno_normal)
        self.dificultad = dificultad / max(dificultad.max(), 1e-12)


def _coeficientes(vertices_cara, vector):
    """(alfa, beta, gamma) con vector = alfa A + beta B + gamma C y suma 0.

    El vector está en el plano de la cara (A, B, C). Como la parametrización
    es lineal en la cara, la imagen del vector en el espacio de textura es
    alfa uv(A) + beta uv(B) + gamma uv(C).
    """
    a, b, c = vertices_cara[:, 0], vertices_cara[:, 1], vertices_cara[:, 2]
    e1 = b - a
    e2 = c - a
    g11 = np.einsum("ij,ij->i", e1, e1)
    g12 = np.einsum("ij,ij->i", e1, e2)
    g22 = np.einsum("ij,ij->i", e2, e2)
    x1 = np.einsum("ij,ij->i", vector, e1)
    x2 = np.einsum("ij,ij->i", vector, e2)
    determinante = g11 * g22 - g12 * g12
    beta = (x1 * g22 - x2 * g12) / determinante
    gamma = (x2 * g11 - x1 * g12) / determinante
    return np.stack([-beta - gamma, beta, gamma], axis=1)


def _calidad(malla, caras, uv, escala):
    """Caras con la parametrización bien orientada y, de ellas, las poco deformadas.

    Se mide el jacobiano de la parametrización en cada cara, de coordenadas
    del plano de la cara (en unidades del modelo) a (u, v), multiplicado por la
    escala del parche: vale la identidad si los trazos tienen su tamaño. Su
    determinante es negativo si la cara quedó volteada (la textura se ve en
    espejo) y sus valores singulares dicen cuánto se estira la textura en cada
    dirección.
    """
    posiciones = malla.vertices[malla.caras[caras]]
    borde_1 = posiciones[:, 1] - posiciones[:, 0]
    borde_2 = posiciones[:, 2] - posiciones[:, 0]
    eje_x = borde_1 / np.linalg.norm(borde_1, axis=1, keepdims=True)
    eje_y = np.cross(malla.normal[caras], eje_x)
    local = np.stack([
        np.stack([np.einsum("ij,ij->i", borde_1, eje_x), np.einsum("ij,ij->i", borde_1, eje_y)], axis=1),
        np.stack([np.einsum("ij,ij->i", borde_2, eje_x), np.einsum("ij,ij->i", borde_2, eje_y)], axis=1),
    ], axis=1)
    imagen = np.stack([uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]], axis=1)
    jacobiano = np.linalg.solve(local, imagen) * escala
    # una cara casi degenerada deja el jacobiano con valores no finitos
    finitas = np.isfinite(jacobiano).all(axis=(1, 2))
    jacobiano[~finitas] = 0.0
    valores_singulares = np.linalg.svd(jacobiano, compute_uv=False)
    orientadas = finitas & (np.linalg.det(jacobiano) > 0)
    buenas = (
        orientadas
        & (valores_singulares[:, 0] < ESTIRAMIENTO_MAXIMO)
        & (valores_singulares[:, 1] > 1.0 / ESTIRAMIENTO_MAXIMO)
    )
    return orientadas, buenas


class _Parche:
    """Un parche en crecimiento: sus caras, sus vértices con (u, v) y el signo
    del campo en cada cara (d y -d son la misma dirección del trazo, pero dentro
    de un parche hay que elegir una de las dos de forma coherente)."""

    def __init__(self, malla, cara, punto, escala):
        self.malla = malla
        self.escala = escala
        self.cara_semilla = cara
        self.punto = punto
        self.caras = [cara]
        self.en_parche = {cara}
        self.signo = {cara: 1.0}
        # caras mal parametrizadas en la última vuelta: el parche no crece desde ellas
        self.bloqueadas = set()
        self.normal_semilla = malla.normal_lista[cara]

        # la semilla se aplana con su propio marco: el punto va al centro del
        # contorno y el campo (S, T) a los ejes (u, v)
        s = malla.campo[cara]
        t = np.cross(malla.normal[cara], s)
        self.uv = {}
        for vertice in malla.caras_lista[cara]:
            relativo = malla.vertices[vertice] - punto
            self.uv[vertice] = [0.5 + relativo @ s / escala, 0.5 + relativo @ t / escala]

    def crecer(self, casco):
        """Agrega caras vecinas, de la más cercana a la semilla a la más lejana.

        Las condiciones son las del paso 2 del algoritmo, más un alcance máximo
        desde el punto semilla. Devuelve cuántas caras agregó.
        """
        malla = self.malla
        caras = malla.caras_lista
        vecinas = malla.vecina_lista
        pliegues = malla.pliegue_lista
        forma = malla.forma_lista
        normales = malla.normal_lista
        centroides = malla.centroide_lista
        campo = malla.campo_lista
        uv = self.uv
        en_parche = self.en_parche
        px, py, pz = self.punto
        nx, ny, nz = self.normal_semilla
        sx, sy, sz = campo[self.cara_semilla]
        coseno_minimo = np.cos(np.radians(ANGULO_MAXIMO_GRADOS))
        alcance2 = (0.75 * self.escala) ** 2
        lado = len(casco)

        def toca_contorno(a, b):
            ua, va = uv[a]
            ub, vb = uv[b]
            for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                u = ua + t * (ub - ua)
                v = va + t * (vb - va)
                if 0.0 <= u < 1.0 and 0.0 <= v < 1.0 and casco[int(v * lado)][int(u * lado)]:
                    return True
            return False

        candidatas = []

        def encolar_vecinas(cara):
            for otra in vecinas[cara]:
                if otra >= 0 and otra not in en_parche:
                    cx, cy, cz = centroides[otra]
                    heapq.heappush(candidatas, ((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2, otra))

        for cara in self.caras:
            if cara not in self.bloqueadas:
                encolar_vecinas(cara)

        agregadas = 0
        while candidatas:
            _, cara = heapq.heappop(candidatas)
            if cara in en_parche:
                continue
            vertices = caras[cara]
            junto = [k for k in range(3) if vecinas[cara][k] in en_parche]
            # disco topológico: la cara se pega por una arista y trae un vértice
            # nuevo, o se pega por dos aristas y cierra una muesca
            if len(junto) + 1 != sum(1 for v in vertices if v in uv):
                continue
            entradas = [
                k for k in junto if vecinas[cara][k] not in self.bloqueadas and not pliegues[cara][k]
            ]
            if not entradas:
                continue
            mx, my, mz = normales[cara]
            if mx * nx + my * ny + mz * nz < coseno_minimo:
                continue
            # el contorno no llega más lejos que esto del punto semilla; el
            # límite evita que un error en las (u, v) predichas haga crecer el
            # parche sobre otra parte de la superficie
            cx, cy, cz = centroides[cara]
            if (cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2 > alcance2:
                continue
            if not any(toca_contorno(vertices[k], vertices[(k + 1) % 3]) for k in entradas):
                continue

            # sentido del campo en la cara: el que forma un ángulo agudo con
            # la dirección de la semilla
            dx, dy, dz = campo[cara]
            self.signo[cara] = 1.0 if dx * sx + dy * sy + dz * sz >= 0.0 else -1.0

            if len(junto) == 1:
                # el vértice nuevo se ubica formando un triángulo semejante al
                # de la cara en 3D, apoyado en la arista ya parametrizada
                k = junto[0]
                a, b, c = vertices[k], vertices[(k + 1) % 3], vertices[(k + 2) % 3]
                ua, va = uv[a]
                eu, ev = uv[b][0] - ua, uv[b][1] - va
                x, y = forma[cara][k]
                uv[c] = [ua + x * eu - y * ev, va + x * ev + y * eu]

            self.caras.append(cara)
            en_parche.add(cara)
            agregadas += 1
            encolar_vecinas(cara)
        return agregadas

    def _uv_por_esquina(self):
        vertices = self.malla.caras[np.array(self.caras)]
        unicos, locales = np.unique(vertices, return_inverse=True)
        tabla = np.array([self.uv[v] for v in unicos.tolist()])
        return tabla[locales.reshape(-1, 3)], unicos, locales.reshape(-1, 3)

    def parametrizar(self):
        """Mínimos cuadrados: en cada cara, (S, T) se lleva a los ejes (u, v).

        Para la cara f = (A, B, C) se escriben S y T como combinación de sus
        vértices; la imagen de S por la parametrización es entonces
        alfa uv(A) + beta uv(B) + gamma uv(C), y se pide que valga (1, 0). Lo
        mismo con T y (0, 1). S tiene el largo de la escala del parche, así que
        avanzar una escala sobre la superficie avanza 1 en el contorno. La
        solución es única salvo traslación; una fila extra fija el punto
        semilla en el centro (0,5, 0,5).
        """
        malla = self.malla
        caras = np.array(self.caras)
        _, unicos, locales = self._uv_por_esquina()
        posiciones = malla.vertices[malla.caras[caras]]
        signo = np.array([self.signo[c] for c in self.caras])[:, None]
        s = self.escala * signo * malla.campo[caras]
        t = np.cross(malla.normal[caras], s)

        cantidad = len(caras)
        # filas: primero la de S de cada cara y luego la de T de cada cara
        filas = np.repeat(np.arange(2 * cantidad), 3)
        columnas = np.concatenate([locales.reshape(-1), locales.reshape(-1)])
        valores = np.concatenate([_coeficientes(posiciones, s).reshape(-1), _coeficientes(posiciones, t).reshape(-1)])

        # fila extra: el punto semilla, en baricéntricas de su cara, va al centro
        a, b, c = malla.vertices[malla.caras[self.cara_semilla]]
        normal = np.cross(b - a, c - a)
        baricentricas = np.array([
            np.cross(c - b, self.punto - b) @ normal,
            np.cross(a - c, self.punto - c) @ normal,
            np.cross(b - a, self.punto - a) @ normal,
        ]) / (normal @ normal)
        peso = 10.0
        filas = np.concatenate([filas, np.full(3, 2 * cantidad)])
        columnas = np.concatenate([columnas, locales[0]])
        valores = np.concatenate([valores, peso * baricentricas])

        sistema = sp.csr_matrix((valores, (filas, columnas)), shape=(2 * cantidad + 1, len(unicos)))
        # columna 0: objetivos de u; columna 1: objetivos de v
        objetivos = np.zeros((2 * cantidad + 1, 2))
        objetivos[:cantidad, 0] = 1.0
        objetivos[cantidad:2 * cantidad, 1] = 1.0
        objetivos[-1] = peso * 0.5
        solucion = spsolve((sistema.T @ sistema).tocsc(), sistema.T @ objetivos)
        for vertice, (ui, vi) in zip(unicos.tolist(), solucion.tolist()):
            self.uv[vertice] = [ui, vi]

        uv = solucion[locales]
        _, buenas = _calidad(malla, caras, uv, self.escala)
        self.bloqueadas = set(caras[~buenas].tolist())

    def terminar(self, contorno):
        """Caras del parche, sus (u, v) y su margen por esquina, y cuáles cubren.

        Se descartan las caras volteadas en el espacio de textura, que
        mostrarían los trazos en espejo. Las caras con los trazos estirados o
        encogidos más de ESTIRAMIENTO_MAXIMO veces se dibujan, pero no cuentan
        como cubiertas: así otro parche, dibujado después, las tapa.

        El margen vale 0 en los vértices del borde del parche por donde el
        contorno todavía no se desvanece (el crecimiento se detuvo antes, por
        ejemplo porque la superficie se dobla) y 1 en el resto. Multiplica el
        alfa, así el parche se funde con los demás en vez de cortarse en seco.
        """
        malla = self.malla
        uv, _, _ = self._uv_por_esquina()
        caras = np.array(self.caras)
        orientadas, buenas = _calidad(malla, caras, uv, self.escala)
        caras = caras[orientadas]
        uv = uv[orientadas]
        buenas = buenas[orientadas]

        # las aristas del borde de la malla no cuentan, porque más allá no hay
        # superficie que mostrar, y tampoco los pliegues, donde el corte entre
        # parches coincide con la arista viva
        en_parche = np.zeros(len(malla.caras), dtype=bool)
        en_parche[caras] = True
        vecinas = malla.vecina[caras]
        borde = (vecinas >= 0) & ~en_parche[np.maximum(vecinas, 0)] & ~malla.pliegue[caras]
        filas, aristas = np.nonzero(borde)
        t = np.linspace(0.0, 1.0, 7)[None, :, None]
        puntos = (1 - t) * uv[filas, aristas][:, None] + t * uv[filas, (aristas + 1) % 3][:, None]
        alfa = _muestrear(contorno, puntos.reshape(-1, 2)).reshape(len(filas), 7)
        duras = (alfa > 0.01).any(axis=1)
        vertices_duros = np.concatenate([
            malla.caras[caras[filas[duras]], aristas[duras]],
            malla.caras[caras[filas[duras]], (aristas[duras] + 1) % 3],
        ])
        margen = 1.0 - np.isin(malla.caras[caras], vertices_duros).astype(np.float64)

        # las caras que caen enteras fuera del contorno no aportan nada al dibujo
        esquinas_y_muestras = np.concatenate([np.eye(3), MUESTRAS])
        puntos = np.einsum("sk,fkd->fsd", esquinas_y_muestras, uv).reshape(-1, 2)
        visibles = _muestrear(contorno, puntos).reshape(len(uv), len(esquinas_y_muestras)).max(axis=1) > 0.0
        return caras[visibles], uv[visibles], margen[visibles], buenas[visibles]


def cubrir(vertices, caras, campo, curvatura, escala_maxima, contorno, semilla=11, reporte=print):
    """Cubre la malla con parches y devuelve todo lo necesario para dibujarlos.

    El resultado es un dict de arreglos. Las caras de todos los parches van
    seguidas, en orden de dibujo:

    - "caras" (N,): índice de la cara en la malla; "parche" (N,): su parche.
    - "uv" (N, 3, 2): coordenadas del contorno en cada esquina.
    - "margen" (N, 3): margen de cada esquina (ver `_Parche.terminar`).

    Por parche: "escala" (lado del contorno en unidades del modelo),
    "desplazamiento" (corrimiento al azar del TAM dentro del parche, para que
    dos parches no repitan los mismos trazos) y "tono" (un tono al azar, para
    ver los parches).

    "fondo_uv" (F, 3, 2) y "fondo_parche" (F,) dan, para cada cara de la malla,
    las coordenadas del parche que mejor la cubre. Con ellas se dibuja una capa
    opaca debajo de los parches, que tapa los pocos píxeles donde la suma de
    los alfas no alcanza a 1. Una cara que no quedó en ningún parche usa un
    parche extra, el último, aplanado con el marco del campo en su cara.

    Las caras que quedan tapadas por completo por parches posteriores no se
    dibujan, así cada cara se dibuja pocas veces.

    El lado del contorno se ajusta a la curvatura de la zona: los parches son
    más chicos donde la superficie se curva mucho y llegan a `escala_maxima`
    en las zonas planas. Como el TAM tiene su propia escala dentro del parche,
    los trazos miden lo mismo en todos los parches. El lado mínimo abarca unas
    cuantas caras, para que el parche alcance a cubrir alguna.
    """
    rng = np.random.default_rng(semilla)
    malla = _Malla(vertices, caras, campo)
    cantidad_caras = len(caras)
    arbol = cKDTree(malla.centroide)
    largo_arista = np.linalg.norm(vertices[caras[:, 1]] - vertices[caras[:, 0]], axis=1).mean()
    escala_minima = min(escala_maxima, max(escala_maxima / 3.0, 6.0 * largo_arista))

    # casco: el contorno ampliado unos píxeles, para decidir si crecer
    casco = contorno > 0
    for _ in range(3):
        casco = casco | np.roll(casco, 1, 0) | np.roll(casco, -1, 0) | np.roll(casco, 1, 1) | np.roll(casco, -1, 1)
    casco = casco.tolist()

    cubiertas = np.zeros((cantidad_caras, len(MUESTRAS)), dtype=bool)
    intentos = np.zeros(cantidad_caras, dtype=int)
    fondo_uv = np.zeros((cantidad_caras, 3, 2))
    fondo_parche = np.full(cantidad_caras, -1)
    fondo_puntaje = np.full(cantidad_caras, -1.0)
    parches = []
    while True:
        pendientes = np.flatnonzero(~cubiertas.all(axis=1))
        if len(pendientes) == 0:
            break
        # mientras falte cubrir la mitad de las caras se prefieren las difíciles
        prioridad = np.ones(len(pendientes))
        if len(pendientes) > cantidad_caras / 2:
            prioridad += 4.0 * malla.dificultad[pendientes]
        cara = int(rng.choice(pendientes, p=prioridad / prioridad.sum()))
        intentos[cara] += 1
        if intentos[cara] > INTENTOS_POR_CARA:
            # la cara no se logra cubrir (por ejemplo, porque la superficie se
            # dobla mucho a su alrededor): se da por cubierta para no quedar en
            # un ciclo; la capa de fondo la dibuja con su mejor parche
            cubiertas[cara] = True
            continue
        muestra = int(rng.choice(np.flatnonzero(~cubiertas[cara])))
        punto = MUESTRAS[muestra] @ vertices[caras[cara]]

        # con curvatura k, recorrer una distancia d gira la normal en k d
        # radianes; el contorno llega a 0,46 de su lado desde el centro
        cercanas = arbol.query_ball_point(punto, 0.5 * escala_maxima)
        curvatura_local = curvatura[cercanas].max() if cercanas else curvatura[cara]
        escala = float(np.clip(RADIANES_POR_PARCHE / max(curvatura_local, 1e-9), escala_minima, escala_maxima))

        parche = _Parche(malla, cara, punto, escala)
        for _ in range(VUELTAS_MAXIMAS):
            if parche.crecer(casco) == 0 and len(parche.caras) > 1:
                break
            parche.parametrizar()

        caras_parche, uv, margen, buenas = parche.terminar(contorno)
        if len(caras_parche) == 0:
            continue
        uv_muestras = np.einsum("sk,fkd->fsd", MUESTRAS, uv).reshape(-1, 2)
        alfa = _muestrear(contorno, uv_muestras).reshape(len(uv), len(MUESTRAS)) * (margen @ MUESTRAS.T)
        opaca = (alfa >= 0.999) & buenas[:, None]
        cubiertas[caras_parche] |= opaca

        # para la capa de fondo, cada cara recuerda el parche que mejor la cubre
        puntaje = alfa.min(axis=1) * np.where(buenas, 1.0, 0.5)
        mejora = puntaje > fondo_puntaje[caras_parche]
        fondo_puntaje[caras_parche[mejora]] = puntaje[mejora]
        fondo_uv[caras_parche[mejora]] = uv[mejora]
        fondo_parche[caras_parche[mejora]] = len(parches)

        parches.append({
            "caras": caras_parche, "uv": uv, "margen": margen, "opaco": opaca, "escala": escala,
            "desplazamiento": rng.random(2), "tono": rng.uniform(0.5, 6.0),
        })
        if reporte and len(parches) % 100 == 0:
            reporte(f"  {len(parches)} parches, {cubiertas.all(axis=1).mean() * 100:.0f} % de las caras cubiertas")

    # de atrás hacia adelante: una cara de un parche no se dibuja si los
    # parches posteriores ya la cubren entera con su parte opaca
    tapadas = np.zeros((cantidad_caras, len(MUESTRAS)), dtype=bool)
    for parche in reversed(parches):
        visibles = ~tapadas[parche["caras"]].all(axis=1)
        tapadas[parche["caras"]] |= parche["opaco"]
        for clave in ("caras", "uv", "margen"):
            parche[clave] = parche[clave][visibles]

    # parche extra para las caras que no quedaron en ninguno
    sueltas = np.flatnonzero(fondo_parche < 0)
    if len(sueltas):
        s = campo[sueltas]
        t = np.cross(malla.normal[sueltas], s)
        relativos = vertices[caras[sueltas]] - malla.centroide[sueltas][:, None]
        fondo_uv[sueltas] = 0.5 + np.stack([
            np.einsum("fkj,fj->fk", relativos, s), np.einsum("fkj,fj->fk", relativos, t)
        ], axis=-1) / escala_maxima
        fondo_parche[sueltas] = len(parches)
    parches.append({"caras": np.zeros(0, dtype=np.int64), "uv": np.zeros((0, 3, 2)), "margen": np.zeros((0, 3)),
                    "escala": escala_maxima, "desplazamiento": rng.random(2), "tono": rng.uniform(0.5, 6.0)})

    if reporte:
        dibujadas = sum(len(p["caras"]) for p in parches)
        reporte(
            f"  {len(parches) - 1} parches; cada cara se dibuja {dibujadas / cantidad_caras:.2f} "
            f"veces en promedio, más la capa de fondo ({(intentos > INTENTOS_POR_CARA).sum()} caras sin cubrir del todo)"
        )
    return {
        "caras": np.concatenate([p["caras"] for p in parches]),
        "parche": np.concatenate([np.full(len(p["caras"]), i) for i, p in enumerate(parches)]),
        "uv": np.concatenate([p["uv"] for p in parches]),
        "margen": np.concatenate([p["margen"] for p in parches]),
        "escala": np.array([p["escala"] for p in parches]),
        "desplazamiento": np.array([p["desplazamiento"] for p in parches]),
        "tono": np.array([p["tono"] for p in parches]),
        "fondo_uv": fondo_uv,
        "fondo_parche": fondo_parche,
    }
