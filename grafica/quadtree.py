import numpy as np


class Rectangle:
    """Rectangulo definido por centro y mitad de ancho/alto."""

    __slots__ = ("cx", "cy", "hw", "hh")

    def __init__(self, cx, cy, hw, hh):
        self.cx = cx
        self.cy = cy
        self.hw = hw
        self.hh = hh

    def contains(self, x, y):
        return (
            self.cx - self.hw <= x < self.cx + self.hw
            and self.cy - self.hh <= y < self.cy + self.hh
        )

    def intersects_circle(self, cx, cy, r):
        # distancia del centro del circulo al punto mas cercano del rectangulo
        dx = max(0, abs(cx - self.cx) - self.hw)
        dy = max(0, abs(cy - self.cy) - self.hh)
        return dx * dx + dy * dy <= r * r


class QuadTree:
    """Quadtree para puntos 2D.

    Parametros
    ----------
    boundary : Rectangle
        Region que cubre este nodo.
    capacity : int
        Cantidad maxima de puntos antes de subdividir.
    max_depth : int
        Profundidad maxima del arbol.
    """

    __slots__ = (
        "boundary",
        "capacity",
        "max_depth",
        "depth",
        "points",
        "data",
        "divided",
        "nw",
        "ne",
        "sw",
        "se",
    )

    def __init__(self, boundary, capacity=4, max_depth=8, depth=0):
        self.boundary = boundary
        self.capacity = capacity
        self.max_depth = max_depth
        self.depth = depth
        self.points = []
        self.data = []
        self.divided = False
        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None

    def insert(self, x, y, datum=None):
        """Inserta un punto. Retorna True si fue insertado."""
        if not self.boundary.contains(x, y):
            return False

        if len(self.points) < self.capacity or self.depth >= self.max_depth:
            self.points.append((x, y))
            self.data.append(datum)
            return True

        if not self.divided:
            self._subdivide()

        return (
            self.nw.insert(x, y, datum)
            or self.ne.insert(x, y, datum)
            or self.sw.insert(x, y, datum)
            or self.se.insert(x, y, datum)
        )

    def query_circle(self, cx, cy, radius, found=None):
        """Retorna todos los puntos (y sus datos) dentro del circulo."""
        if found is None:
            found = []

        if not self.boundary.intersects_circle(cx, cy, radius):
            return found

        r2 = radius * radius
        for (px, py), d in zip(self.points, self.data):
            dx = px - cx
            dy = py - cy
            if dx * dx + dy * dy <= r2:
                found.append(((px, py), d))

        if self.divided:
            self.nw.query_circle(cx, cy, radius, found)
            self.ne.query_circle(cx, cy, radius, found)
            self.sw.query_circle(cx, cy, radius, found)
            self.se.query_circle(cx, cy, radius, found)

        return found

    def query_circle_points(self, cx, cy, radius):
        """Retorna solo los datos asociados a puntos dentro del circulo."""
        return [d for _, d in self.query_circle(cx, cy, radius)]

    def get_rectangles(self, rects=None):
        """Retorna la lista de rectangulos (para visualizacion).

        Cada elemento es (cx, cy, hw, hh, depth).
        """
        if rects is None:
            rects = []

        b = self.boundary
        rects.append((b.cx, b.cy, b.hw, b.hh, self.depth))

        if self.divided:
            self.nw.get_rectangles(rects)
            self.ne.get_rectangles(rects)
            self.sw.get_rectangles(rects)
            self.se.get_rectangles(rects)

        return rects

    def get_visited_rectangles(self, cx, cy, radius, rects=None):
        """Retorna rectangulos visitados durante una consulta de rango.

        Cada elemento es (cx, cy, hw, hh, depth, has_intersection).
        """
        if rects is None:
            rects = []

        b = self.boundary
        intersects = b.intersects_circle(cx, cy, radius)
        rects.append((b.cx, b.cy, b.hw, b.hh, self.depth, intersects))

        if self.divided and intersects:
            self.nw.get_visited_rectangles(cx, cy, radius, rects)
            self.ne.get_visited_rectangles(cx, cy, radius, rects)
            self.sw.get_visited_rectangles(cx, cy, radius, rects)
            self.se.get_visited_rectangles(cx, cy, radius, rects)

        return rects

    def clear(self):
        """Vacia el arbol manteniendo la misma estructura de boundary."""
        self.points.clear()
        self.data.clear()
        self.divided = False
        self.nw = self.ne = self.sw = self.se = None

    def _subdivide(self):
        b = self.boundary
        hw = b.hw / 2
        hh = b.hh / 2
        d = self.depth + 1

        self.nw = QuadTree(
            Rectangle(b.cx - hw, b.cy + hh, hw, hh),
            self.capacity, self.max_depth, d,
        )
        self.ne = QuadTree(
            Rectangle(b.cx + hw, b.cy + hh, hw, hh),
            self.capacity, self.max_depth, d,
        )
        self.sw = QuadTree(
            Rectangle(b.cx - hw, b.cy - hh, hw, hh),
            self.capacity, self.max_depth, d,
        )
        self.se = QuadTree(
            Rectangle(b.cx + hw, b.cy - hh, hw, hh),
            self.capacity, self.max_depth, d,
        )

        # redistribuir puntos existentes
        for (px, py), datum in zip(self.points, self.data):
            self.nw.insert(px, py, datum) or \
            self.ne.insert(px, py, datum) or \
            self.sw.insert(px, py, datum) or \
            self.se.insert(px, py, datum)

        self.points.clear()
        self.data.clear()
        self.divided = True
