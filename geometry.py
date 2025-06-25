def gen_edges(points):
    edges = []
    for i in range(-1, len(points)):
        edges.append(Edge(points[i], points[i+1]))
    return edges


class Polygon:
    def __init__(self, points):
        self.points = points

    def to_triangles(self):
        triangles = []
        monotones = []

    def __getitem__(self, item):
        return self.points[item % len(self.points)]

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Point(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other, self.z / other)

class Edge:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

class Face:
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.edges = gen_edges([p1, p2, p3])