import math

def gen_edges(points):
    edges = []
    for i in range(-1, len(points)):
        edges.append(Edge(points[i], points[i+1]))
    return edges

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0, 0)
        return Vector(self.x / mag, self.y / mag, self.z / mag)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: int | float):
        return Point(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: int | float):
        return Point(self.x / other, self.y / other, self.z / other)

    def __neg__(self):
        return Point(-self.x, -self.y, -self.z)

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

    def __neg__(self):
        return Point(-self.x, -self.y, -self.z)

    @staticmethod
    def avg(points):
        x = sum(p.x for p in points) / len(points)
        y = sum(p.y for p in points) / len(points)
        z = sum(p.z for p in points) / len(points)
        return Point(x, y, z)

    def to(self, other):
        x = other.x - self.x
        y = other.y - self.y
        z = other.z - self.z
        return Vector(x, y, z)

class Edge:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def magnitude(self):
        return math.sqrt((self.p2.x - self.p1.x) ** 2 + (self.p2.y - self.p1.y) ** 2 + (self.p2.z - self.p1.z) ** 2)

    def get_mid(self):
        return Point.avg([self.p1, self.p2])

class Face:
    def __init__(self, p1, p2, p3, normal):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.normal = normal

    def double_res(self):
        return Polygon([self.p1, self.p1.get_mid(self.p2), self.p2, self.p2.get_mid(self.p3), self.p3, self.p3.get_mid(self.p1)])

    def points(self):
        return [self.p1, self.p2, self.p3]

    def get_center(self):
        return Point.avg([self.p1, self.p2, self.p3])

    def is_inside(self, point):
        n = self.normal
        c = self.get_center()
        # normals of the edges
        v1 = n.cross(self.p3.to(self.p1))
        v2 = n.cross(self.p1.to(self.p2))
        v3 = n.cross(self.p2.to(self.p3))
        if v1.dot(self.p1.to(c)) < 0:
            v1 = -v1
        if v2.dot(self.p2.to(c)) < 0:
            v2 = -v2
        if v3.dot(self.p3.to(c)) < 0:
            v3 = -v3
        return v1.dot(self.p1.to(point)) < 0 and v2.dot(self.p2.to(point)) < 0 and v3.dot(self.p3.to(point)) < 0