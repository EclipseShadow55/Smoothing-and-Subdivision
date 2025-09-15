import numpy as np
import os
import pyvista as pv
from coloraide import Color
import networkx as nx
from pyvista import PolyData

dir = os.path.dirname(os.path.abspath(__file__))

def debug_changes():
    mesh = pv.read(dir + "/debug_save/init_trimesh_model.obj")
    loaded = np.load(dir + "/debug_save/debug_changes.npz", allow_pickle=True)
    init_points = loaded['init_points']
    final_points = loaded['final_points']
    arrow_dirs = loaded['arrow_dirs']
    arrow_starts = loaded['arrow_starts']

    for i in range(arrow_dirs.shape[0]):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='black')
        plotter.add_points(pv.PolyData(init_points[i]), color='red', point_size=10, name='Initial Points')
        plotter.add_points(pv.PolyData(final_points[i]), color='green', point_size=10, name='Final Points')
        plotter.add_arrows(arrow_starts[i], arrow_dirs[i], color='black', mag=1, name='Change Directions')
        plotter.view_isometric()
        plotter.show()

def debug_point_sorting():
    loaded = np.load(dir + "/debug_save/debug_point_sorting.npz", allow_pickle=True)
    mesh = pv.read(dir + "/debug_save/init_trimesh_model.obj")
    verts_by_edge = loaded['verts_by_edge']
    for i in range(verts_by_edge.shape[0]):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='black')
        edge_verts = verts_by_edge[i]
        edge_points = pv.PolyData(edge_verts)
        plotter.add_points(edge_points, color='red', point_size=20, name=f'Edge {i}')
        plotter.view_isometric()
        plotter.show()

def debug_point_dirs():
    mesh = pv.read(dir + "/debug_save/init_trimesh_model.obj")
    loaded = np.load(dir + "/debug_save/debug_point_dirs.npz", allow_pickle=True)
    starts = loaded['starts']
    dirs = loaded['edge_dirs']
    verts_by_edge = loaded['verts_by_edge']

    for i in range(dirs.shape[0]):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='black')
        plotter.add_points(pv.PolyData(starts[i]), color='red', point_size=10, name='Starts')
        plotter.add_points(pv.PolyData(verts_by_edge[i, 0]), color='orange', point_size=10, name='Firsts')
        plotter.add_points(pv.PolyData(verts_by_edge[i, 1]), color='green', point_size=10, name='Seconds')
        plotter.add_points(pv.PolyData(verts_by_edge[i, 2]), color='blue', point_size=10, name='Thirds')
        plotter.add_points(pv.PolyData(verts_by_edge[i, 3]), color='purple', point_size=10, name='Fourths')
        plotter.add_arrows(starts[i], dirs[i], color='black', mag=0.15, name='Edge Directions')
        plotter.view_isometric()
        plotter.show()

def debug_polygon_faces():
    mesh = pv.read(dir + "/debug_save/final_polymesh.obj")
    faces = mesh.regular_faces
    verts = mesh.points

    for i in range(len(faces)):
        new_mesh = pv.PolyData.from_regular_faces(verts, faces[i][None, :])
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='red', opacity=0.2, show_edges=True)
        plotter.add_mesh(new_mesh, color='blue', opacity=0.5, show_edges=True)
        plotter.view_isometric()
        plotter.show()

def debug_polygon_dirs():
    mesh = pv.read(dir + "/debug_save/final_polymesh.obj")
    loaded = np.load(dir + "/debug_save/debug_vertex_changes.npz", allow_pickle=True)
    polygons = loaded['polygons']
    vertices = loaded['new_verts']
    color_set = Color.interpolate(["#FF0000", "#FF9000", "#FFF200", "#00FF08", "#00D0FF", "#000DFF", "#7B00FF", "#FF00D9"], space="srgb", method="natural")

    for i in range(polygons.shape[0]):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='red', opacity=0.2, show_edges=True)
        poly_verts = vertices[polygons[i]]
        for j in range(polygons.shape[1]):
            color = color_set(j / polygons.shape[1]).to_string(hex=True, upper=True)
            plotter.add_points(PolyData(poly_verts[j]), color=color, point_size=10, name=f'Polygon {i}')
            plotter.add_arrows(poly_verts[j], np.roll(poly_verts, 1, axis=0)[j] - poly_verts[j], color=color)
        plotter.view_isometric()
        plotter.show()

def debug_triangles():
    mesh = pv.read(dir + "/debug_save/init_trimesh_model.obj")
    loaded = np.load(dir + "/debug_save/debug_triangles.npz", allow_pickle=True)
    vertices = loaded['vertices']
    triangles = loaded['triangles']
    color_set = Color.interpolate(["#FF0000", "#FF9000", "#FFF200", "#00FF08", "#00D0FF", "#000DFF", "#7B00FF", "#FF00D9"], space="srgb", method="natural")

    for i in range(triangles.shape[0]):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, style="wireframe", show_edges=True, edge_color='black')
        for j in range(triangles[i].shape[0]):
            tri_verts = vertices[triangles[i][j]]
            tri = pv.PolyData(tri_verts, faces=[3, 0, 1, 2])
            color = color_set(j / triangles[i].shape[0]).to_string(hex=True, upper=True)
            plotter.add_points(tri, color=color, point_size=10, name=f'Triangle {j}')
            plotter.add_mesh(tri, color=color, opacity=0.75, show_edges=True, edge_color='black')
        plotter.view_isometric()
        plotter.show()

def debug_opposites():
    mesh = pv.read(dir + "/debug_save/init_trimesh_model.obj")
    loaded = np.load(dir + "/debug_save/debug_opposites.npz", allow_pickle=True)
    edges = loaded['edges']
    opps = loaded['opposites']

    for i in range(edges.shape[0]):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='black')
        plotter.add_points(pv.PolyData(edges[i, 0]), color='red', point_size=20, name='Edge 1')
        plotter.add_points(pv.PolyData(edges[i, 1]), color='green', point_size=20, name='Edge 2')
        plotter.add_points(pv.PolyData(opps[i]), color='blue', point_size=20, name='Opposite Points')
        plotter.view_isometric()
        plotter.show()

def debug_init_intersections():
    mesh = pv.read(dir + "/debug_save/init_trimesh_model.obj")
    loaded = np.load(dir + "/debug_save/debug_init_intersections.npz", allow_pickle=True)
    init_points = loaded['init_points']
    intersections = loaded['intersections']

    for i in range(intersections.shape[0]):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='black')
        plotter.add_points(pv.PolyData(init_points[i]), color='red', point_size=10, name='Initial Points')
        plotter.add_points(pv.PolyData(intersections[i]), color='blue', point_size=20, name='Intersections')
        plotter.view_isometric()
        plotter.show()

def debug_changes_to_opposites():
    mesh = pv.read(dir + "/debug_save/init_trimesh_model.obj")
    loaded = np.load(dir + "/debug_save/debug_opposites.npz", allow_pickle=True)
    edges = loaded['edges']
    opps = loaded['opposites']
    loaded = np.load(dir + "/debug_save/debug_changes.npz", allow_pickle=True)
    final_points = loaded['final_points']
    arrow_dirs = loaded['arrow_dirs']
    arrow_starts = loaded['arrow_starts']
    for i in range(edges.shape[0]):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='black')
        plotter.add_points(pv.PolyData(edges[i, 0]), color='red', point_size=20, name='Edge 1')
        plotter.add_points(pv.PolyData(edges[i, 1]), color='red', point_size=20, name='Edge 2')
        plotter.add_points(pv.PolyData(opps[i]), color='blue', point_size=20, name='Opposite Points')
        plotter.add_points(pv.PolyData(final_points[i]), color='green', point_size=10, name='Final Points')
        plotter.add_arrows(arrow_starts[i], arrow_dirs[i], color='black', mag=0.5, name='Change Directions')
        plotter.view_isometric()
        plotter.show()

def debug_edge_changes():
    mesh = pv.read(dir + "/debug_save/middle_trimesh_model.obj")
    loaded = np.load(dir + "/debug_save/debug_edge_changes.npz", allow_pickle=True)
    vertices = loaded['vertices']
    edges = loaded['edges']
    changes = loaded['changes']
    combined = loaded['combined']
    norms = loaded['norms']
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, style="wireframe", line_width=2, edge_color='black')
    plotter.add_points(pv.PolyData(vertices), color='red', point_size=5)
    plotter.add_points(pv.PolyData(edges[:, 0]), color='green', point_size=10)
    plotter.add_points(pv.PolyData(edges[:, 1]), color='orange', point_size=10)
    #plotter.add_arrows(edges[:, 0], changes[:, 0], color="green")
    #plotter.add_arrows(edges[:, 1], changes[:, 1], color="orange")
    plotter.add_arrows(edges[:, 0], norms, color="purple")
    plotter.add_arrows(edges[:, 1], norms, color="purple")
    plotter.add_arrows(vertices, combined, color="black")
    plotter.view_isometric()
    plotter.show()

def debug_vertex_changes():
    mesh = pv.read(dir + "/debug_save/init_trimesh_model.obj")
    loaded = np.load(dir + "/debug_save/debug_vertex_changes.npz", allow_pickle=True)
    vertices = loaded['vertices']
    edges = loaded['edges']
    changes = loaded['changes']
    edge_moves = loaded['edge_moves']
    edge_changes = loaded['edge_changes']
    norms = loaded['norms']
    new_verts = loaded['new_verts']
    polygons = loaded['polygons']
    new_mesh = pv.PolyData.from_regular_faces(new_verts, polygons)
    new_mesh.save(dir + "/debug_save/final_polymesh.obj")
    for i in range(edges.shape[0]):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, style="wireframe", line_width=2, color='orange')
        plotter.add_mesh(new_mesh, color='red', opacity=0.5, show_edges=True, edge_color='black', line_width=5)
        plotter.add_points(pv.PolyData(vertices[i]), color='red', point_size=5)
        plotter.add_points(pv.PolyData(edges[i]), color='green', point_size=10)
        plotter.add_arrows(vertices[i], changes[i], color="black", mag=1)
        plotter.add_arrows(edges[i], edge_changes[i], color="orange")
        plotter.add_arrows(edges[i], edge_moves[i], color="blue", mag=1)
        plotter.add_points(new_verts, color='purple', point_size=5)
        """
        for j in range(vertices[i].shape[0]):
            plotter.add_arrows(vertices[i][j], norms[i], color="purple")
        for j in range(edges[i].shape[0]):
            plotter.add_arrows(edges[i][j], norms[i], color="purple")
        """
        plotter.view_isometric()
        plotter.show()

def debug_neighbors():
    important = [1, 2, 3, 4, 5]
    mesh = pv.read(dir + "/debug_save/final_polymesh.obj")
    vertices = mesh.points
    dtype = np.dtype([('x', vertices.dtype), ('y', vertices.dtype), ('z', vertices.dtype)])
    verts_view = vertices.view(dtype)

    edge_set = mesh.extract_all_edges()
    edge_vertices = edge_set.points.astype(vertices.dtype)
    edge_verts_view = edge_vertices.view(dtype)

    line_set = np.array(edge_set.lines)
    line_inds = np.arange(len(line_set))
    lines_arr = line_set[line_inds % 3 != 0]
    lines = np.array(np.split(lines_arr, len(lines_arr) / 2))
    unique_lines = np.unique(np.sort(lines[lines.T[0] != lines.T[1]], axis=1), axis=0)
    graph = nx.Graph()
    graph.add_edges_from(unique_lines)
    for i in important:
        point = verts_view[i]
        point_ind = np.where(np.isclose(edge_verts_view['x'], point['x']) & np.isclose(edge_verts_view['y'], point['y']) & np.isclose(edge_verts_view['z'], point['z']))[0][0]
        neighbors = list(graph.neighbors(point_ind))
        points = edge_vertices[neighbors]
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='red', opacity=0.3)
        plotter.add_points(pv.PolyData(points), color='green', point_size=10)
        plotter.add_points(pv.PolyData(vertices[i]), color='blue', point_size=10)
        plotter.add_points(pv.PolyData(mesh.points), color='red', point_size=5)
        plotter.view_isometric()
        plotter.show()

def debug_final():
    final = pv.read(dir + "/debug_save/final_trimesh_model.obj")
    plotter = pv.Plotter()
    plotter.add_mesh(final, color='lightblue', show_edges=True, edge_color='black')
    plotter.view_isometric()
    plotter.show()

"""
def debug_change_calc():
    mesh = pv.read(dir + "/debug_save/init_trimesh_model.obj")
    loaded = np.load(dir + "/debug_save/debug_changes_calc.npz", allow_pickle=True)
    init_points = loaded['init_points']
    changes = loaded['changes']
    norms = loaded['norms']
    loaded = np.load(dir + "/debug_save/debug_init_intersections.npz", allow_pickle=True)
    intersections = loaded['intersections']

    for i in range(changes.shape[0]):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, style="wireframe")
        plotter.add_points(pv.PolyData(init_points[i]), color='red', point_size=20, name='Initial Points')
        plotter.add_points(pv.PolyData(intersections[i]), color='blue', point_size=10, name='Intersections')
        plotter.add_arrows(init_points[i], changes[i], color='black', mag=1, name='Calculated Changes')
        plotter.add_arrows(init_points[i], norms[i], color='green', mag=1, name='Vertex Normals')
        plotter.view_isometric()
        plotter.show()
"""

if __name__ == "__main__":
    debug_polygon_faces()