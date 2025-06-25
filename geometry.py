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

    def double_res(self):
        new_points = []
        for i in range(len(self.points)):
            p1 = self[i]
            p2 = self[(i + 1) % len(self.points)]
            new_points.append(p1)
            new_points.append(p1.get_mid(p2))
        return Polygon(new_points)

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other:int|float):
        return Point(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other:int|float):
        return Point(self.x / other, self.y / other, self.z / other)

    def get_mid(self, other):
        return Point((self.x + other.x) / 2, (self.y + other.y) / 2, (self.z + other.z) / 2)

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

    def double_res(self):
        return Polygon([self.p1, self.p1.get_mid(self.p2), self.p2, self.p2.get_mid(self.p3), self.p3, self.p3.get_mid(self.p1)])