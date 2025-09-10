from vedo import Mesh, load

# Load the mesh
mesh = load('meshes/ribs/ribs.obj')

# Perform one iteration of Butterfly subdivision
subdivided_mesh = mesh.subdivide(method=3, n=5)

# Save the subdivided mesh
subdivided_mesh.write('meshes/ribs/vedo_smoothed_ribs.obj')