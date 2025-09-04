import trimesh
import numpy as np

mesh = trimesh.load_mesh('meshes/pyramid/pyramid.obj')
edges = mesh.edges_unique
edges_sorted = np.sort(edges, axis=1)
replaces = np.array([[0, 5, 14, 23, 32, 1],
                     [0, 6, 15, 24, 33, 2],
                     [0, 7, 16, 25, 34, 3],
                     [1, 8, 17, 26, 35, 3],
                     [2, 9, 18, 27, 36, 3],
                     [0, 10, 19, 28, 37, 4],
                     [1, 11, 20, 29, 38, 4],
                     [2, 12, 21, 30, 39, 4],
                     [3, 13, 22, 31, 40, 4]])
faces = mesh.faces

edges1 = faces[:, [0, 1]]
edges1_sort = np.argsort(edges1, axis=1)
edges1_sorted = np.take_along_axis(edges1, edges1_sort, axis=1)
edges2 = faces[:, [1, 2]]
edges2_sort = np.argsort(edges2, axis=1)
edges2_sorted = np.take_along_axis(edges2, edges2_sort, axis=1)
edges3 = faces[:, [2, 0]]
edges3_sort = np.argsort(edges3, axis=1)
edges3_sorted = np.take_along_axis(edges3, edges3_sort, axis=1)

matches1 = (edges_sorted[:, 0] == edges1_sorted[:, None, 0]) & (edges[:, 1] == edges1_sorted[:, None, 1])
matches2 = (edges_sorted[:, 0] == edges2_sorted[:, None, 0]) & (edges[:, 1] == edges2_sorted[:, None, 1])
matches3 = (edges_sorted[:, 0] == edges3_sorted[:, None, 0]) & (edges[:, 1] == edges3_sorted[:, None, 1])
edges1_idx = np.where(matches1)[1]
edges2_idx = np.where(matches2)[1]
edges3_idx = np.where(matches3)[1]

edges1_rep = replaces[edges1_idx]
edges2_rep = replaces[edges2_idx]
edges3_rep = replaces[edges3_idx]

flip_arr = np.array([1, 0])
flip_mask1 = (edges1_sort[:, 0] == flip_arr[None, 0]) & (edges1_sort[:, 1] == flip_arr[None, 1])
flip_mask2 = (edges2_sort[:, 0] == flip_arr[None, 0]) & (edges2_sort[:, 1] == flip_arr[None, 1])
flip_mask3 = (edges3_sort[:, 0] == flip_arr[None, 0]) & (edges3_sort[:, 1] == flip_arr[None, 1])
edges1_rep[flip_mask1] = edges1_rep[flip_mask1][:, ::-1]
edges2_rep[flip_mask2] = edges2_rep[flip_mask2][:, ::-1]
edges3_rep[flip_mask3] = edges3_rep[flip_mask3][:, ::-1]

new_faces = np.hstack((edges1_rep[:, :-1], edges2_rep[:, :-1], edges3_rep[:, :-1]))
face1 = new_faces[:, [14, 0, 1]]
face2 = new_faces[:, [4, 5, 6]]
faces3 = new_faces[:, [9, 10, 11]]
faces4 = new_faces[:, [14, 1, 2]]
faces5 = new_faces[:, [3, 4, 6]]
faces6 = new_faces[:, [8, 9, 11]]
faces7 = new_faces[:, [13, 14, 2]]
faces8 = new_faces[:, [3, 6, 7]]
faces9 = new_faces[:, [8, 11, 12]]
faces10 = new_faces[:, [7, 8, 12]]
faces11 = new_faces[:, [12, 13, 2]]
faces12 = new_faces[:, [2, 3, 7]]
faces13 = new_faces[:, [2, 7, 12]]
all_faces = np.vstack((face1, face2, faces3, faces4, faces5, faces6, faces7, faces8, faces9, faces10, faces11, faces12, faces13))

col2 = (mesh.vertices[edges[:, 0]] + mesh.vertices[edges[:, 1]]) / 2
col1 = (mesh.vertices[edges[:, 0]] + col2) / 2
col3 = (col2 + mesh.vertices[edges[:, 1]]) / 2
col4 = (col3 + mesh.vertices[edges[:, 1]]) / 2
all_vertices = np.vstack((mesh.vertices, col1, col2, col3, col4))

new_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=True)
new_vertices = new_mesh.vertices + np.random.normal(scale=0.01, size=new_mesh.vertices.shape)
new_mesh.vertices = new_vertices
new_mesh.export("test.obj")