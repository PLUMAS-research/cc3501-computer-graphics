"""Particiones binarias del espacio para lanzar rayos sobre un mapa 2D.

Un nodo divide su región con una recta: lo que queda a un lado va al hijo de
frente y lo que queda al otro, al hijo de atrás. Las dos estructuras del
ejemplo comparten el nodo y el recorrido, y difieren en cómo eligen la recta:

- BSP, como el de Doom: la recta es la de una pared del mapa. Las paredes que
  cruza se parten en dos y las que están sobre ella se guardan en el nodo, así
  que las hojas quedan vacías: son regiones convexas sin paredes adentro.
- kd-tree: la recta es vertical u horizontal y pasa por la mediana de los
  puntos medios de las paredes. Las paredes que cruza quedan en los dos hijos,
  y cada hoja guarda las paredes que la tocan.

Un rayo recorre el árbol de adelante hacia atrás: en cada nodo visita primero
el hijo del lado donde está su origen, y el otro solo si el rayo cruza la recta
antes de chocar con una pared sólida. Así los impactos salen ordenados por
distancia y el recorrido termina en la primera pared sólida.

Un árbol es un dict con "raiz" y las paredes que usa ("a", "b" y "linea", la
línea del mapa de la que sale cada pared). Un nodo interno tiene "punto" y
"normal" (su recta), "frente" y "atras"; todos los nodos tienen "region", el
polígono convexo que cubren, y "paredes".
"""
import numpy as np

EPS = 1e-6


def _cruz(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def recortar(region, punto, normal):
    """La parte de un polígono convexo que queda del lado hacia donde apunta la normal."""
    lado = (region - punto) @ normal
    resultado = []
    for i in range(len(region)):
        j = (i + 1) % len(region)
        if lado[i] >= 0:
            resultado.append(region[i])
        if (lado[i] >= 0) != (lado[j] >= 0):
            resultado.append(region[i] + (region[j] - region[i]) * lado[i] / (lado[i] - lado[j]))
    return np.array(resultado).reshape(-1, 2)


def construir_bsp(a, b, linea, region):
    """BSP con las rectas de las paredes: elige en cada nodo la que parte menos
    paredes y deja los dos lados más parejos."""
    paredes = {"a": list(a), "b": list(b), "linea": list(linea)}
    raiz = _nodo_bsp(paredes, list(range(len(a))), region)
    return {"raiz": raiz, "a": np.array(paredes["a"]), "b": np.array(paredes["b"]),
            "linea": np.array(paredes["linea"])}


def _nodo_bsp(paredes, indices, region):
    if not indices:
        return {"region": region, "paredes": []}
    a = np.array([paredes["a"][i] for i in indices])
    b = np.array([paredes["b"][i] for i in indices])
    # cada pared candidata clasifica a todas las del nodo; su normal apunta a
    # la derecha de la pared, que en Doom es el frente
    direccion = b - a
    normales = np.stack([direccion[:, 1], -direccion[:, 0]], axis=1)
    normales /= np.linalg.norm(normales, axis=1, keepdims=True)
    lado_a = np.einsum("cmk,ck->cm", a[np.newaxis] - a[:, np.newaxis], normales)
    lado_b = np.einsum("cmk,ck->cm", b[np.newaxis] - a[:, np.newaxis], normales)
    sobre = (np.abs(lado_a) < EPS) & (np.abs(lado_b) < EPS)
    frente = ~sobre & (lado_a > -EPS) & (lado_b > -EPS)
    atras = ~sobre & (lado_a < EPS) & (lado_b < EPS)
    cortadas = ~(sobre | frente | atras)
    costo = 8 * cortadas.sum(axis=1) + np.abs(frente.sum(axis=1) - atras.sum(axis=1))
    elegida = int(np.argmin(costo))
    punto, normal = a[elegida], normales[elegida]

    en_la_recta, de_frente, de_atras = [], [], []
    for j, indice in enumerate(indices):
        if sobre[elegida, j]:
            en_la_recta.append(indice)
        elif frente[elegida, j]:
            de_frente.append(indice)
        elif atras[elegida, j]:
            de_atras.append(indice)
        else:
            # la recta parte la pared: el trozo hasta el corte queda en el
            # índice original y el resto es una pared nueva
            la, lb = lado_a[elegida, j], lado_b[elegida, j]
            corte = paredes["a"][indice] + (paredes["b"][indice] - paredes["a"][indice]) * la / (la - lb)
            nueva = len(paredes["a"])
            paredes["a"].append(corte)
            paredes["b"].append(paredes["b"][indice])
            paredes["linea"].append(paredes["linea"][indice])
            paredes["b"][indice] = corte
            (de_frente if la > 0 else de_atras).append(indice)
            (de_frente if lb > 0 else de_atras).append(nueva)
    return {
        "punto": punto, "normal": normal, "region": region, "paredes": en_la_recta,
        "frente": _nodo_bsp(paredes, de_frente, recortar(region, punto, normal)),
        "atras": _nodo_bsp(paredes, de_atras, recortar(region, punto, -normal)),
    }


def construir_kd(a, b, linea, region, por_hoja=4, profundidad_maxima=24):
    """kd-tree: corta el lado más largo de cada región en la mediana de los
    puntos medios, hasta que una hoja tiene a lo más `por_hoja` paredes."""
    a, b = np.asarray(a), np.asarray(b)
    raiz = _nodo_kd(a, b, list(range(len(a))), region, profundidad_maxima, por_hoja)
    return {"raiz": raiz, "a": a, "b": b, "linea": np.asarray(linea)}


def _nodo_kd(a, b, indices, region, profundidad, por_hoja):
    if len(indices) <= por_hoja or profundidad == 0:
        return {"region": region, "paredes": indices}
    eje = int(np.argmax(np.ptp(region, axis=0)))
    corte = float(np.median((a[indices, eje] + b[indices, eje]) / 2))
    menor = np.minimum(a[indices, eje], b[indices, eje])
    mayor = np.maximum(a[indices, eje], b[indices, eje])
    de_atras = [i for i, m in zip(indices, menor) if m <= corte]
    de_frente = [i for i, m in zip(indices, mayor) if m >= corte]
    # si un lado se queda con todas las paredes, cortar no ayuda: donde muchas
    # paredes comparten un vértice, la mediana cae siempre sobre ellas
    if len(indices) in (len(de_atras), len(de_frente)):
        return {"region": region, "paredes": indices}
    normal = np.eye(2)[eje]
    punto = corte * normal
    return {
        "punto": punto, "normal": normal, "region": region, "paredes": [],
        "frente": _nodo_kd(a, b, de_frente, recortar(region, punto, normal), profundidad - 1, por_hoja),
        "atras": _nodo_kd(a, b, de_atras, recortar(region, punto, -normal), profundidad - 1, por_hoja),
    }


def intersecciones(a, b, origen, direccion):
    """Parámetro t del rayo en el cruce con cada pared, o infinito si no la cruza."""
    e = b - a
    denominador = _cruz(direccion, e)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cruz(a - origen, e) / denominador
        s = _cruz(a - origen, direccion) / denominador
    cruza = (denominador != 0) & (t > EPS) & (s >= -EPS) & (s <= 1 + EPS)
    return np.where(cruza, t, np.inf)


def impactos(arbol, origen, direccion, solida, cuenta):
    """Impactos (t, pared) ordenados por distancia, hasta la primera pared sólida.

    `solida` dice, por línea del mapa, si detiene el rayo. `cuenta` acumula los
    nodos visitados y las paredes probadas; si trae listas "visitados" y
    "probadas", guarda también cuáles fueron.
    """
    resultado = []
    _recorrer(arbol, arbol["raiz"], origen, direccion, 0.0, np.inf, solida, resultado, cuenta)
    return resultado


def _recorrer(arbol, nodo, origen, direccion, t0, t1, solida, resultado, cuenta):
    """Visita el nodo en el tramo [t0, t1) del rayo; devuelve True al chocar con una pared sólida."""
    cuenta["nodos"] += 1
    if "visitados" in cuenta:
        cuenta["visitados"].append(nodo)
    if "normal" not in nodo:
        return _probar(arbol, nodo["paredes"], origen, direccion, t0, t1, solida, resultado, cuenta)

    lado = (origen - nodo["punto"]) @ nodo["normal"]
    avance = direccion @ nodo["normal"]
    # el hijo cercano es el del lado del origen; si el origen está sobre la
    # recta, el cercano es el lado del que el rayo se aleja
    de_frente = lado > 0 or (lado == 0 and avance < 0)
    cerca, lejos = (nodo["frente"], nodo["atras"]) if de_frente else (nodo["atras"], nodo["frente"])
    t_recta = -lado / avance if avance != 0 else np.inf
    if t_recta < 0 or t_recta >= t1:
        return _recorrer(arbol, cerca, origen, direccion, t0, t1, solida, resultado, cuenta)
    if t_recta < t0:
        return _recorrer(arbol, lejos, origen, direccion, t0, t1, solida, resultado, cuenta)
    if _recorrer(arbol, cerca, origen, direccion, t0, t_recta, solida, resultado, cuenta):
        return True
    # paredes del BSP que están sobre la recta: el rayo la cruza en t_recta
    if nodo["paredes"] and t_recta > EPS:
        cuenta["paredes"] += len(nodo["paredes"])
        if "probadas" in cuenta:
            cuenta["probadas"].extend(nodo["paredes"])
        cruce = origen + t_recta * direccion
        for pared in nodo["paredes"]:
            e = arbol["b"][pared] - arbol["a"][pared]
            s = (cruce - arbol["a"][pared]) @ e / (e @ e)
            if -EPS <= s <= 1 + EPS:
                resultado.append((t_recta, pared))
                if solida[arbol["linea"][pared]]:
                    return True
    return _recorrer(arbol, lejos, origen, direccion, t_recta, t1, solida, resultado, cuenta)


def _probar(arbol, paredes, origen, direccion, t0, t1, solida, resultado, cuenta):
    """Prueba las paredes de una hoja del kd-tree y agrega las que cruza en [t0, t1)."""
    if not paredes:
        return False
    cuenta["paredes"] += len(paredes)
    if "probadas" in cuenta:
        cuenta["probadas"].extend(paredes)
    paredes = np.asarray(paredes)
    t = intersecciones(arbol["a"][paredes], arbol["b"][paredes], origen, direccion)
    # una pared que toca varias hojas se cuenta solo en la hoja donde la cruza el rayo
    dentro = np.nonzero((t >= t0) & (t < t1))[0]
    for i in dentro[np.argsort(t[dentro])]:
        resultado.append((t[i], int(paredes[i])))
        if solida[arbol["linea"][paredes[i]]]:
            return True
    return False


def impactos_fuerza_bruta(arbol, origen, direccion, solida):
    """Los mismos impactos que `impactos`, probando todas las paredes."""
    t = intersecciones(arbol["a"], arbol["b"], origen, direccion)
    resultado = []
    for pared in np.argsort(t)[: np.isfinite(t).sum()]:
        resultado.append((t[pared], int(pared)))
        if solida[arbol["linea"][pared]]:
            break
    return resultado


def orden_de_dibujo(arbol, punto):
    """Las paredes del BSP en el orden en que Doom las dibuja vistas desde `punto`.

    Es el recorrido de adelante hacia atrás sin un rayo: en cada nodo, primero
    el lado del punto, luego las paredes sobre la recta y al final el otro
    lado. En este orden una pared nunca tapa a otra que va antes, aunque esa
    otra esté más lejos, así que se puede dibujar de adelante hacia atrás sin
    ordenar por distancia.
    """
    orden = []

    def visitar(nodo):
        if "normal" not in nodo:
            return
        de_frente = (punto - nodo["punto"]) @ nodo["normal"] >= 0
        visitar(nodo["frente"] if de_frente else nodo["atras"])
        orden.extend(nodo["paredes"])
        visitar(nodo["atras"] if de_frente else nodo["frente"])

    visitar(arbol["raiz"])
    return orden


def cuerda(nodo):
    """El trozo de la recta de un nodo que queda dentro de su región, para dibujarlo."""
    region = nodo["region"]
    lado = (region - nodo["punto"]) @ nodo["normal"]
    puntos = [region[i] + (region[j] - region[i]) * lado[i] / (lado[i] - lado[j])
              for i, j in zip(range(len(region)), np.roll(range(len(region)), -1))
              if (lado[i] >= 0) != (lado[j] >= 0)]
    return np.array(puntos[:2])


def nodos(arbol):
    """Todos los nodos, con su profundidad."""
    pila, todos = [(arbol["raiz"], 1)], []
    while pila:
        nodo, profundidad = pila.pop()
        todos.append((nodo, profundidad))
        if "normal" in nodo:
            pila += [(nodo["frente"], profundidad + 1), (nodo["atras"], profundidad + 1)]
    return todos
