import numpy as np

class Part:
    def __init__(self, faces):
        self.points = ()
        self.edges = ()
        self.faces = faces


    def connect(self, other):


class Megalodon:
    class Tooth:
        def __init__(self, position, size, wear, chipping):
            self.position = position
            self.size = size
            self.wear = wear
            self.chipping = chipping