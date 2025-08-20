import numpy as np
import trimesh
import networkx
from copy import deepcopy
from colorama import Fore


class ExtendedTrimesh:
    def __init__(self):
        raise NotImplementedError("This class is not meant to be instantiated directly.")

    @staticmethod
    def edges_in_mesh(mesh: trimesh.Trimesh, edges: np.ndarray):
        """
        Check which items in an ndarray of edges are present in a mesh. Public method that validates input types and dimensions.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :param edges: An ndarray of shape (n, 2) of edges or pairs of vertex indices, where n is the number of edges.
        :type edges: np.ndarray
        :return: ndarray of shape (n) boolean values indicating if each edge is present in the mesh, where n is the number of edges.
        :rtype: np.ndarray
        :raises TypeError: If mesh is not a trimesh.Trimesh object or edges is not a numpy ndarray.
        :raises ValueError: If edges is not a 2D array with shape (n, 2), if edges is empty, if mesh is empty, or if mesh is not watertight.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if not isinstance(edges, np.ndarray):
            raise TypeError("Edges must be a numpy ndarray.")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Edges must be a 2D array with shape (n, 2).")
        if edges.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        if mesh.is_empty:
            return np.zeros(0, dtype=bool)

        return ExtendedTrimesh._edges_in_mesh(mesh, edges)

    @staticmethod
    def _edges_in_mesh(mesh: trimesh.Trimesh, edges: np.ndarray):
        """
        Check which items in an ndarray of edges are present in a mesh. Private method that performs the actual check with no input validation.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :param edges: An ndarray of shape (n, 2) of edges or pairs of vertex indices, where n is the number of edges.
        :type edges: np.ndarray
        :return: ndarray of shape (n) boolean values indicating if each edge is present in the mesh, where n is the number of edges.
        :rtype: np.ndarray
        """
        sorted_edges = np.sort(edges, axis=1)
        mesh_edges = np.sort(mesh.face_adjacency_edges, axis=1)
        # View as structured arrays for row-wise comparison
        dtype = [('v0', sorted_edges.dtype), ('v1', sorted_edges.dtype)]
        sorted_edges_view = sorted_edges.view(dtype)
        mesh_edges_view = mesh_edges.view(dtype)
        # Use np.isin for fast membership check
        mask = np.isin(sorted_edges_view, mesh_edges_view)

        return mask

    @staticmethod
    def get_row_inds(arr1: np.ndarray, arr2: np.ndarray):
        """
        Get the row indices of arr1 that are present in arr2. Public method that validates input types and dimensions.

        :param arr1: The first array to check.
        :type arr1: np.ndarray
        :param arr2: The second array to check against.
        :type arr2: np.ndarray
        :return: ndarray of indices of rows in arr1 that are present in arr2.
        :rtype: np.ndarray
        :raises TypeError: If arr1 or arr2 are not numpy ndarrays.
        :raises ValueError: If arr1 or arr2 are not 2D arrays.
        """
        if not isinstance(arr1, np.ndarray):
            raise TypeError("arr1 must be a numpy ndarray.")
        if not isinstance(arr2, np.ndarray):
            raise TypeError("arr2 must be a numpy ndarray.")
        if arr1.ndim != 2:
            raise ValueError("arr1 must be a 2D array.")
        if arr2.ndim != 2:
            raise ValueError("arr2 must be a 2D array.")

        return ExtendedTrimesh._get_row_inds(arr1, arr2)

    @staticmethod
    def _get_row_inds(arr1: np.ndarray, arr2: np.ndarray):
        """
        Get the row indices of arr1 that are present in arr2. Private method that performs the actual check with no input validation.

        :param arr1: The first array to check.
        :type arr1: np.ndarray
        :param arr2: The second array to check against.
        :type arr2: np.ndarray
        :return: ndarray of indices of rows in arr1 that are present in arr2.
        :rtype: np.ndarray
        :raises TypeError: If arr1 or arr2 are not numpy ndarrays.
        :raises ValueError: If arr1 or arr2 are not 2D arrays.
        """

        dtype = np.dtype((np.void, arr1.dtype.itemsize * arr1.shape[2]))
        A_view = arr1.view(dtype).ravel()
        B_view = arr2.view(dtype).ravel()

        sort_idx = np.argsort(B_view)
        B_sorted = B_view[sort_idx]
        found_idx = np.searchsorted(B_sorted, A_view)
        mask = (found_idx < len(B_sorted))
        valid_idx = np.where(mask)[0]
        mask_valid = (B_sorted[found_idx[valid_idx]] == A_view[valid_idx])
        mask[valid_idx] = mask_valid
        indices = np.full(arr1.shape[0] * arr1.shape[1], -1)
        indices[mask] = sort_idx[found_idx[mask]]

        return indices

    @staticmethod
    def get_edge_normals(mesh: trimesh.Trimesh):
        """
        Get the normals for each edge in a mesh. Public method that validates input types and dimensions.
        Note: This method only works for watertight meshes and when every provided edge is present in the mesh.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :return: ndarray of normals for each edge of shape (n, 3) where n is the number of edges.
        :rtype: np.ndarray
        :raises TypeError: If mesh is not a trimesh.Trimesh object.
        :raises ValueError: If mesh is empty, if mesh is not watertight
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if mesh.is_empty:
            return np.zeros((0, 3), dtype=float)
        if not mesh.is_watertight:
            raise ValueError("Mesh must be watertight to compute edge normals.")

        return ExtendedTrimesh._get_edge_normals(mesh)

    @staticmethod
    def _get_edge_normals(mesh: trimesh.Trimesh):
        """
        Get the normals for each edge in a mesh. Private method that performs the actual computation with no input validation.
        Note: This method only works for watertight meshes and when every provided edge is present in the mesh.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :return: ndarray of normals for each edge of shape (n, 3) where n is the number of edges.
        :rtype: np.ndarray
        """
        neighbors = mesh.face_adjacency
        normals = mesh.face_normals
        normals = normals[neighbors.T]
        normals1 = normals[0]
        normals2 = normals[1]
        e_normals = (normals1 + normals2) / 2

        return e_normals

    @staticmethod
    def get_opposite(faces: np.ndarray, edges: np.ndarray):
        """
        Get the opposite vertex indices for each edge in a set of faces.

        :param faces: An ndarray of shape (n, 3) representing the faces of the mesh.
        :type faces: np.ndarray
        :param edges: An ndarray of shape (m, 2) representing the edges of the mesh.
        :type edges: np.ndarray
        :return: An ndarray of shape (w) containing the opposite vertex indices for each edge, where w is the number of edges.
        :rtype: np.ndarray
        :raises TypeError: If faces or edges are not numpy ndarrays.
        :raises ValueError: If faces is not a 2D array with shape (n, 3) or if edges is not a 2D array with shape (m, 2).
        """
        if not isinstance(faces, np.ndarray):
            raise TypeError("Faces must be a numpy ndarray.")
        if not isinstance(edges, np.ndarray):
            raise TypeError("Edges must be a numpy ndarray.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Faces must be a 2D array with shape (n, 3).")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Edges must be a 2D array with shape (m, 2).")

        return ExtendedTrimesh._get_opposite(faces, edges)

    @staticmethod
    def _get_opposite(faces: np.ndarray, edges: np.ndarray):
        """
        Get the opposite vertex indices for each edge in a set of faces. Private method that performs the actual computation with no input validation.

        :param faces: An ndarray of shape (n, 3) representing the faces of the mesh.
        :type faces: np.ndarray
        :param edges: An ndarray of shape (m, 2) representing the edges of the mesh.
        :type edges: np.ndarray
        :return: An ndarray of shape (w) containing the opposite vertex indices for each edge, where w is the number of edges.
        :rtype: np.ndarray
        """
        mask = (faces[:, :, None] == edges[:, None, :])
        mask_any = np.any(mask, axis=2)
        opposite = faces[~mask_any]

        return opposite

    @staticmethod
    def get_signed_angles(mesh: trimesh.Trimesh):
        """
        Get the signed angles between adjacent faces at each edge of a mesh. Public method that validates input types and dimensions.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :return: An ndarray of shape (n,) containing the signed angles in radians.
        :rtype: np.ndarray
        :raises TypeError: If mesh is not a trimesh.Trimesh object
        :raises ValueError: If mesh is not watertight.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if mesh.is_empty:
            return np.zeros(0, dtype=float)
        if not mesh.is_watertight:
            raise ValueError("Mesh must be watertight to compute signed angles.")

        return ExtendedTrimesh._get_signed_angles(mesh)

    @staticmethod
    def _get_signed_angles(mesh: trimesh.Trimesh):
        """
        Get the signed angles between adjacent faces at each edge of a mesh. Private method that performs the actual computation with no input validation.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :return: An ndarray of shape (n,) containing the signed angles in radians.
        :rtype: np.ndarray
        """
        neighbors = mesh.face_adjacency
        normals = mesh.face_normals[neighbors.T]
        angles = np.arccos(np.clip(np.einsum('ij,ij->i', normals[0], normals[1]), -1.0, 1.0))
        return angles

    @staticmethod
    def sharp_tri_catmull_clark(mesh: trimesh.Trimesh, alpha: float, beta: float, iterations: int = 1, proximity_digits: int = 6):
        """
        Smooth a mesh in place using intelligent edge and vertex addition.

        :param mesh: The mesh to smooth.
        :type mesh: trimesh.Trimesh
        :param alpha: A scaling factor for the amount of change applied to each vertex and edge.
        :type alpha: float
        :param beta: A scaling factor for where new vertices are placed relative to existing vertices.
        :type beta: float
        :param iterations: Number of smoothing iterations to perform. Default is 1.
        :type iterations: int
        :param proximity_digits: Number of digits to round vertex positions to for proximity checks. Default is 6.
        :type proximity_digits: int
        :return: None. The mesh is modified in place.
        :raises TypeError: If mesh is not a trimesh.Trimesh object, iterations is not an integer, or alpha is not a number.
        :raises ValueError: If iterations is less than 1, alpha is not in the range (0, 1), if beta is not in the range [0, 1), if change_tol is negative, if the mesh is not watertight, or if e_mode is not "memory_efficient" or "time_efficient."
        """
        if not isinstance(iterations, int):
            raise TypeError("Iterations must be an integer.")
        if not isinstance(alpha, (int, float)):
            raise TypeError("Alpha must be a number.")
        if not isinstance(beta, (int, float)):
            raise TypeError("Beta must be a number.")
        if not isinstance(proximity_digits, int):
            raise TypeError("Proximity tolerance must be an int.")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if iterations < 1:
            raise ValueError("Iterations must be at least 1.")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Alpha must be in the range (0, 1).")
        if beta < 0 or beta >= 1:
            raise ValueError("Beta must be in the range [0, 1).")
        if proximity_digits < 1:
            raise ValueError("Proximity tolerance must be non-negative.")
        if mesh.is_empty:
            return
        if not mesh.is_watertight:
            raise ValueError("Mesh must be watertight to perform smoothing.")
        for _ in range(iterations):
            ExtendedTrimesh._sharp_tri_catmull_clark(mesh, alpha, beta, proximity_digits)

    @staticmethod
    def _sharp_tri_catmull_clark(mesh: trimesh.Trimesh, alpha: float, beta: float, proximity_digits: int = 6):
        """
        Smooth a mesh in place using intelligent edge and vertex addition. Private method that performs the actual smoothing with no input validation.

        :param mesh: The mesh to smooth.
        :type mesh: trimesh.Trimesh
        :param alpha: A scaling factor for where new vertices are placed relative to existing vertices.
        :type alpha: float
        :param beta: A scaling factor for the amount of change applied to each existing vertex and edge.
        :type beta: float
        :param proximity_digits: Number of digits to round vertex positions to for proximity checks. Default is 6.
        :type proximity_digits: int
        :return: None. The mesh is modified in place.
        """
        # Get base values
        edges = mesh.face_adjacency_edges
        neighbors = mesh.face_adjacency
        angles = ExtendedTrimesh._get_signed_angles(mesh)
        new_factors = np.sin(angles) * alpha
        edge_norms = ExtendedTrimesh._get_edge_normals(mesh)
        i_edge_norms = -edge_norms
        old_factors = np.sin(angles) * beta * 0.5

        # Calculate intermediate values
        opposites1 = ExtendedTrimesh._get_opposite(mesh.faces[neighbors.T[0]], edges)
        opposites2 = ExtendedTrimesh._get_opposite(mesh.faces[neighbors.T[1]], edges)
        face1_edge1_int = mesh.vertices[edges[:, 0]] * new_factors[:, None] + mesh.vertices[opposites1] * (1 - new_factors)[:, None]
        face1_edge2_int = mesh.vertices[edges[:, 1]] * new_factors[:, None] + mesh.vertices[opposites1] * (1 - new_factors)[:, None]
        face2_edge1_int = mesh.vertices[edges[:, 0]] * new_factors[:, None] + mesh.vertices[opposites2] * (1 - new_factors)[:, None]
        face2_edge2_int = mesh.vertices[edges[:, 1]] * new_factors[:, None] + mesh.vertices[opposites2] * (1 - new_factors)[:, None]
        face1_change1 = face1_edge1_int - mesh.vertices[edges[:, 0]]
        face1_change2 = face1_edge2_int - mesh.vertices[edges[:, 1]]
        face2_change1 = face2_edge1_int - mesh.vertices[edges[:, 0]]
        face2_change2 = face2_edge2_int - mesh.vertices[edges[:, 1]]
        dist1 = np.einsum('ij,ij->i', face1_change1, i_edge_norms)
        dist2 = np.einsum('ij,ij->i', face2_change1, i_edge_norms)
        dists = np.min(np.column_stack([dist1, dist2]), axis=1)
        changes = dists[:, None] * i_edge_norms

        # Find actual intersection points
        face1_int1 = mesh.vertices[edges[:, 0]] + face1_change1 / np.linalg.norm(face1_change1, axis=1)[:, None] * dists[:, None]
        face1_int2 = mesh.vertices[edges[:, 1]] + face1_change2 / np.linalg.norm(face1_change2, axis=1)[:, None] * dists[:, None]
        face2_int1 = mesh.vertices[edges[:, 0]] + face2_change1 / np.linalg.norm(face2_change1, axis=1)[:, None] * dists[:, None]
        face2_int2 = mesh.vertices[edges[:, 1]] + face2_change2 / np.linalg.norm(face2_change2, axis=1)[:, None] * dists[:, None]
        change_set_norm = np.vstack((face1_change1 / np.linalg.norm(face1_change1, axis=1)[:, None] * dists[:, None],
                                           face1_change2 / np.linalg.norm(face1_change2, axis=1)[:, None] * dists[:, None],
                                           face2_change1 / np.linalg.norm(face2_change1, axis=1)[:, None] * dists[:, None],
                                           face2_change2 / np.linalg.norm(face2_change2, axis=1)[:, None] * dists[:, None]))
        cent_set = np.vstack([mesh.vertices[edges[:, 0]], mesh.vertices[edges[:, 1]], mesh.vertices[edges[:, 0]], mesh.vertices[edges[:, 1]]])


        # Add vertices to the mesh
        new_vertices = np.vstack([face1_int1, face1_int2, face2_int1, face2_int2])
        old_verts = mesh.vertices
        new_vert_inds = np.arange(old_verts.shape[0], old_verts.shape[0] + new_vertices.shape[0])
        mesh.vertices = np.vstack([mesh.vertices, new_vertices])

        # Add vertices to the graph
        graph = trimesh.graph.vertex_adjacency_graph(mesh)
        face1_int1_edges = np.sort(np.column_stack([edges[:, 0], opposites1]), axis=1)
        face1_int2_edges = np.sort(np.column_stack([edges[:, 1], opposites1]), axis=1)
        face2_int1_edges = np.sort(np.column_stack([edges[:, 0], opposites2]), axis=1)
        face2_int2_edges = np.sort(np.column_stack([edges[:, 1], opposites2]), axis=1)
        new_vert_edges = np.column_stack([face1_int1_edges[:, None, :], face1_int2_edges[:, None, :], face2_int1_edges[:, None, :], face2_int2_edges[:, None, :]])
        edges = mesh.edges_unique
        new_vert_edge_inds = ExtendedTrimesh._get_row_inds(new_vert_edges, edges)
        sort_idx = np.argsort(new_vert_edge_inds)
        sorted_vert_edge_inds = new_vert_edge_inds[sort_idx]
        sorted_verts = new_vert_inds[sort_idx]
        unique_edge_inds, group_starts = np.unique(sorted_vert_edge_inds, return_index=True)
        verts_by_edge = np.array(np.split(sorted_verts, group_starts[1:]))
        dist_from_start = np.linalg.norm(mesh.vertices[verts_by_edge] - mesh.vertices[edges[:, 0]][:, None, :], axis=2)
        edge_dirs = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
        np.savetxt("debug/debug_save/starts.xyz", mesh.vertices[edges[:, 0]], fmt='%.6f')
        np.savetxt("debug/debug_save/firsts.xyz", mesh.vertices[verts_by_edge.T[0]], fmt='%.6f')
        np.savetxt("debug/debug_save/seconds.xyz", mesh.vertices[verts_by_edge.T[1]], fmt='%.6f')
        np.savetxt("debug/debug_save/thirds.xyz", mesh.vertices[verts_by_edge.T[2]], fmt='%.6f')
        np.savetxt("debug/debug_save/fourths.xyz", mesh.vertices[verts_by_edge.T[3]], fmt='%.6f')
        np.savetxt("debug/debug_save/edge_dirs.xyz", edge_dirs, fmt='%.6f')
        order = np.argsort(dist_from_start, axis=1)
        verts_by_edge = np.take_along_axis(verts_by_edge, order, axis=1)
        verts_to_insert = np.transpose(verts_by_edge, (1, 0, 2))
        new_edge_1 = np.column_stack([edges[:, 0], verts_to_insert[:, 0]])
        new_edge_2 = np.column_stack([verts_to_insert[:, 0], verts_to_insert[:, 1]])
        new_edge_3 = np.column_stack([verts_to_insert[:, 1], verts_to_insert[:, 2]])
        new_edge_4 = np.column_stack([verts_to_insert[:, 2], verts_to_insert[:, 3]])
        new_edge_5 = np.column_stack([verts_to_insert[:, 3], edges[:, 1]])
        new_edges = np.vstack([new_edge_1, new_edge_2, new_edge_3, new_edge_4, new_edge_5])
        graph.remove_edges_from(edges)
        graph.add_edges_from(new_edges)

        # Get faces, triangulate graph
        cycles = networkx.cycle_basis(graph)
        valid_polygons = [cycle for cycle in cycles if len(cycle) >= 3]
        new_edges, new_triangles = ExtendedTrimesh._edges_to_triangulate_polygons(valid_polygons)
        new_triangles = np.array(new_triangles, dtype=int) #(n, 13, 3)

        # Add new faces to the mesh and process it
        mesh.faces = new_triangles
        mesh.process()
        trimesh.repair.fix_winding(mesh)
        mesh.remove_duplicate_faces()
        ExtendedTrimesh.remove_degen_faces(mesh)
        trimesh.repair.fill_holes(mesh)
        mesh.update_faces(np.arange(mesh.faces.shape[0]))
        mesh.remove_unreferenced_vertices()
        mesh.process()
        mesh.rezero()

        # Move old edges towards new edges
        vert_moves = np.zeros_like(old_verts)
        edge_moves = changes * old_factors * 0.5
        np.add.at(vert_moves, edges[:, 0], edge_moves)
        np.add.at(vert_moves, edges[:, 1], edge_moves)
        mesh.vertices += vert_moves

        # Cleanup
        mesh.merge_vertices(digits_vertex=proximity_digits)
        mesh.process()
        ExtendedTrimesh.remove_degen_faces(mesh)
        mesh.remove_duplicate_faces()
        trimesh.repair.fill_holes(mesh)
        mesh.update_faces(np.arange(mesh.faces.shape[0]))
        mesh.remove_unreferenced_vertices()
        mesh.process()

    @staticmethod
    def edges_to_triangulate_polygons(polygons: list[list[int]]):
        """
        Returns a list of edges to create to triangulate every polygon. Public method that validates input types and dimensions.

        :param polygons: A list of polygons, where each polygon is a list of vertex indices.
        :type polygons: list[list[int]]
        :return: A list of new edges to add that would triangulate all the polygons and the resulting triangles.
        :rtype: list[tuple[int]], list[list[tuple[int]]]
        """
        if not isinstance(polygons, list):
            raise TypeError("Polygons must be a list.")
        if not all(isinstance(polygon, list) for polygon in polygons):
            raise TypeError("Each polygon must be a list of vertex indices.")
        if not all(len(polygon) >= 3 for polygon in polygons):
            raise ValueError("Each polygon must have at least 3 vertices.")

        return ExtendedTrimesh._edges_to_triangulate_polygons(polygons)

    @staticmethod
    def _edges_to_triangulate_polygons(polygons: list[list[int]]):
        """
        Returns a list of edges to create to triangulate every polygon. Private method that performs the actual triangulation with no input validation.

        :param polygons: A list of polygons, where each polygon is a list of vertex indices.
        :type polygons: list[list[int]]
        :return: A list of new edges to add that would triangulate all the polygons and the resulting triangles.
        :rtype: list[tuple[int]], list[list[tuple[int]]]
        """
        new_edges = []
        triangles = []
        for i, poly in enumerate(deepcopy(polygons)):
            triangles.append([])
            while len(poly) > 3:
                # Always clip the first ear (for simplicity)
                i0, i1, i2 = poly[0], poly[1], poly[2]
                new_edges.append((i0, i2))  # Add diagonal
                triangles[i].append((i0, i1, i2))
                del poly[1]  # Remove ear tip
            triangles.append(tuple(poly))
            # The last triangle is (poly[0], poly[1], poly[2])
        return new_edges

    @staticmethod
    def remove_degen_faces(mesh: trimesh.Trimesh):
        """
        Remove degenerate faces from a mesh by removing faces with zero area. Public method that validates input types and dimensions.

        :param mesh: The mesh to process.
        :type mesh: trimesh.Trimesh
        :return: None. The mesh is modified in place.
        :raises TypeError: If mesh is not a trimesh.Trimesh object.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if mesh.is_empty:
            return

        ExtendedTrimesh._remove_degen_faces(mesh)

    @staticmethod
    def _remove_degen_faces(mesh: trimesh.Trimesh):
        """
        Remove degenerate faces from a mesh by removing faces with zero area. Private method that performs the actual removal with no input validation.

        :param mesh: The mesh to process.
        :type mesh: trimesh.Trimesh
        :return: None. The mesh is modified in place.
        """
        areas = mesh.area_faces
        if areas.size == 0:
            return
        mask = areas > 0
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()

    @staticmethod
    def closest_vertices(mesh: trimesh.Trimesh):
        """
        Find the closest vertices in a mesh to each other. Public method that validates input types and dimensions.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :return: 2 ndarrays of shape (n, 1) containing pairs of vertex indices that are closest to each other.
        :rtype: tuple(np.ndarray)
        :raises TypeError: If mesh is not a trimesh.Trimesh object.
        :raises ValueError: If mesh is empty.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if mesh.is_empty:
            return np.zeros((0, 2), dtype=int)

        return ExtendedTrimesh._closest_vertices(mesh)

    @staticmethod
    def _closest_vertices(mesh: trimesh.Trimesh):
        """
        Find the closest vertices in a mesh to each other. Private method that performs the actual computation with no input validation.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :return: 2 ndarrays of shape (n, 1) containing pairs of vertex indices that are closest to each other.
        :rtype: tuple(np.ndarray)
        """
        dists = np.linalg.norm(mesh.vertices[:, None, :] - mesh.vertices[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        closest_indices = np.argmin(dists, axis=1)
        closest_distances = dists[np.arange(len(mesh.vertices)), closest_indices]
        return closest_indices, closest_distances

if __name__ == "__main__":
    # Example usage
    mesh = trimesh.load_mesh("meshes/pyramid/pyramid.obj")
    ExtendedTrimesh.sharp_tri_catmull_clark(mesh, alpha=0.5, beta=0.5, iterations=2, proximity_digits=6)
    mesh.export("meshes/pyramid/pyramid_catmull_clark.obj")