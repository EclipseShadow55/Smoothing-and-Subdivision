import numpy as np
import trimesh
import geometry as geo

mesh = trimesh.load_mesh('meshes/pyramid/pyramid.obj')
geo.ExtendedTrimesh.sharp_tri_catmull_clark(mesh, alpha=0.9, iterations=1)
mesh.export("meshes/pyramid/smoothed_pyramid.obj")