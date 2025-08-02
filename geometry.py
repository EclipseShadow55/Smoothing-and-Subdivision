import numpy as np
import trimesh

class ExtendedTrimesh:
    @staticmethod
    def edges_in_mesh(mesh: trimesh.Trimesh, edges: np.ndarray) -> np.ndarray:
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
    def _edges_in_mesh(mesh: trimesh.Trimesh, edges: np.ndarray) -> np.ndarray:
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
    def get_edge_normals(mesh: trimesh.Trimesh) -> np.ndarray:
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
    def _get_edge_normals(mesh: trimesh.Trimesh) -> np.ndarray:
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
    def get_opposite(faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
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
    def _get_opposite(faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
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
    def sharp_tri_catmull_clark(mesh: trimesh.Trimesh, alpha: float, iterations: int = 1):
        """
        Smooth a mesh in place using intelligent edge and vertex addition.

        :param mesh: The mesh to smooth.
        :type mesh: trimesh.Trimesh
        :param alpha: A scaling factor for the amount of change applied to each vertex and edge.
        :type alpha: float
        :param iterations: Number of smoothing iterations to perform.
        :type iterations: int
        :return: None. The mesh is modified in place.
        :raises TypeError: If mesh is not a trimesh.Trimesh object, iterations is not an integer, or alpha is not a number.
        :raises ValueError: If iterations is less than 1, alpha is not in the range (0, 1), or if the mesh is not watertight.
        """
        if not isinstance(iterations, int):
            raise TypeError("Iterations must be an integer.")
        if not isinstance(alpha, (int, float)):
            raise TypeError("Alpha must be a number.")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if iterations < 1:
            raise ValueError("Iterations must be at least 1.")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Alpha must be in the range (0, 1).")
        if mesh.is_empty:
            return
        if not mesh.is_watertight:
            raise ValueError("Mesh must be watertight to perform smoothing.")
        for _ in range(iterations):
            ExtendedTrimesh._sharp_catmull_clark(mesh, alpha)

    @staticmethod
    def _sharp_tri_catmull_clark(mesh: trimesh.Trimesh, alpha: float, change_tol=1e-6):
        """
        Smooth a mesh in place using intelligent edge and vertex addition. Private method that performs the actual smoothing with no input validation.

        :param mesh: The mesh to smooth.
        :type mesh: trimesh.Trimesh
        :param alpha: A scaling factor for the amount of change applied to each vertex and edge.
        :type alpha: float
        :param change_tol: Tolerance for change in vertex positions to ignore an edge for smoothing.
        :type change_tol: float
        :return: None. The mesh is modified in place.
        """
        # Get base values
        edges = mesh.face_adjacency_edges
        neighbors = mesh.face_adjacency
        angles = mesh.face_adjacency_angles
        factors = np.abs(np.pi - angles) / np.pi * alpha
        edge_norms = ExtendedTrimesh._get_edge_normals(mesh)
        i_edge_norms = -edge_norms

        # Calculate intermediate values
        opposites1 = ExtendedTrimesh._get_opposite(mesh.faces[neighbors.T[0]], edges)
        opposites2 = ExtendedTrimesh._get_opposite(mesh.faces[neighbors.T[1]], edges)
        face1_edge1s = np.column_stack([edges[:, 0], opposites1])
        face1_edge2s = np.column_stack([edges[:, 1], opposites1])
        face2_edge1s = np.column_stack([edges[:, 0], opposites2])
        face2_edge2s = np.column_stack([edges[:, 1], opposites2])
        face1_edge1s_mid = mesh.vertices[face1_edge1s].mean(axis=1)
        face1_edge2s_mid = mesh.vertices[face1_edge2s].mean(axis=1)
        face2_edge1s_mid = mesh.vertices[face2_edge1s].mean(axis=1)
        face2_edge2s_mid = mesh.vertices[face2_edge2s].mean(axis=1)
        face1_edge1_int = mesh.vertices[edges[:, 0]] * (1 - factors)[:, None] + face1_edge1s_mid * factors[:, None]
        face1_edge2_int = mesh.vertices[edges[:, 1]] * (1 - factors)[:, None] + face1_edge2s_mid * factors[:, None]
        face2_edge1_int = mesh.vertices[edges[:, 0]] * (1 - factors)[:, None] + face2_edge1s_mid * factors[:, None]
        face2_edge2_int = mesh.vertices[edges[:, 1]] * (1 - factors)[:, None] + face2_edge2s_mid * factors[:, None]
        change1_1 = face1_edge1_int - mesh.vertices[edges[:, 0]]
        change1_2 = face1_edge2_int - mesh.vertices[edges[:, 1]]
        change2_1 = face2_edge1_int - mesh.vertices[edges[:, 0]]
        change2_2 = face2_edge2_int - mesh.vertices[edges[:, 1]]
        dist1 = np.sum(change1_1 * i_edge_norms, axis=1)
        dist2 = np.sum(change1_2 * i_edge_norms, axis=1)
        dists = np.min(np.column_stack([dist1, dist2]), axis=1)
        changes = dists[:, None] * i_edge_norms

        # Filter out edges with small changes
        change_mask = np.linalg.norm(changes, axis=1) > change_tol
        changes = changes[change_mask]
        change1_1 = change1_1[change_mask]
        change1_2 = change1_2[change_mask]
        change2_1 = change2_1[change_mask]
        change2_2 = change2_2[change_mask]
        edges = edges[change_mask]
        opposites1 = opposites1[change_mask]
        opposites2 = opposites2[change_mask]
        neighbors = neighbors[change_mask]
        edge_norms = edge_norms[change_mask]

        # Calculate final changes
        change1_1 = (np.sum(changes * change1_1, axis=1) / np.sum(change1_1 * change1_1, axis=1))[:, None] * change1_1
        change1_2 = (np.sum(changes * change1_2, axis=1) / np.sum(change1_2 * change1_2, axis=1))[:, None] * change1_2
        change2_1 = (np.sum(changes * change2_1, axis=1) / np.sum(change2_1 * change2_1, axis=1))[:, None] * change2_1
        change2_2 = (np.sum(changes * change2_2, axis=1) / np.sum(change2_2 * change2_2, axis=1))[:, None] * change2_2
        face1_edge1_int = mesh.vertices[edges[:, 0]] + change1_1
        face1_edge2_int = mesh.vertices[edges[:, 1]] + change1_2
        face2_edge1_int = mesh.vertices[edges[:, 0]] + change2_1
        face2_edge2_int = mesh.vertices[edges[:, 1]] + change2_2

        # Get face direction (for winding)
        norms = mesh.face_normals[neighbors]
        edge_vecs = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
        right = np.cross(edge_norms, edge_vecs)
        dirs = (np.sign(np.einsum('nij,nj->ni', norms, right)) <= 0)
        opp_inds = np.zeros(shape=dirs.shape, dtype=int)
        opp_inds[dirs] = 1
        opps = np.take_along_axis(np.column_stack([opposites1, opposites2]), opp_inds).T

        # Create new faces and vertices
        edge1_int1_inds = len(mesh.vertices) + np.arange(len(face1_edge1_int))
        edge2_int1_inds = edge1_int1_inds + len(face1_edge2_int)
        edge1_int2_inds = edge2_int1_inds + len(face2_edge1_int)
        edge2_int2_inds = edge1_int2_inds + len(face2_edge2_int)
        faces1 = np.column_stack([opps[0], edge1_int1_inds, edge2_int1_inds])
        faces1p = np.column_stack([edge1_int1_inds, edges[:, 0], edge2_int1_inds])
        faces1q = np.column_stack([edge2_int1_inds, edges[:, 0], edges[:, 1]])
        faces2 = np.column_stack([opps[1], edge2_int2_inds, edge1_int2_inds])
        faces2p = np.column_stack([edge2_int2_inds, edges[:, 1], edge1_int2_inds])
        faces2q = np.column_stack([edge1_int2_inds, edges[:, 1], edges[:, 0]])
        new_faces = np.vstack([faces1, faces1p, faces1q, faces2, faces2p, faces2q])
        new_verts = np.vstack([face1_edge1_int, face1_edge2_int, face2_edge1_int, face2_edge2_int])

        # Update the mesh
        faces_to_rem = np.unique(neighbors.flatten())
        mask = np.ones(len(mesh.faces), dtype=bool)
        edges = mesh.vertices[edges]
        mask[faces_to_rem] = False
        mesh.update_faces(mask)
        mesh.vertices = np.vstack([mesh.vertices, new_verts])
        mesh.faces = np.vstack([mesh.faces, new_faces])
        mesh._cache.clear()
        mesh.process(validate=True)

        # Clean up the mesh
        mesh.remove_unreferenced_vertices()
        ExtendedTrimesh._remove_degen_faces(mesh)
        ExtendedTrimesh._fix_overlapping_faces(mesh)

        # Update edges with new vertex indices
        edges_flat = edges.reshape(-1, 3)
        vertices = mesh.vertices
        matches = np.all(edges_flat[:, None, :] == vertices[None, :, :], axis=2)
        inds = np.argmax(matches, axis=1)
        edges = inds.reshape(edges.shape[:2])
        # 2 TODO: Implement algorthm to shift old edges towards new edges to lower curvature

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
    def fix_overlapping_faces(mesh):
        """
        Fix overlapping faces in a mesh by removing duplicate faces and unreferenced vertices. Public method that validates input types and dimensions.

        :param mesh: The mesh to fix.
        :type mesh: trimesh.Trimesh
        :return: None. The mesh is modified in place.
        :raises TypeError: If mesh is not a trimesh.Trimesh object.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if mesh.is_empty:
            return

        ExtendedTrimesh._fix_overlapping_faces(mesh)

    @staticmethod
    def _fix_overlapping_faces(mesh: trimesh.Trimesh):
        """
        Fix overlapping faces in a mesh by removing duplicate faces and unreferenced vertices. Private method that performs the actual fix with no input validation.

        :param mesh: The mesh to fix.
        :type mesh: trimesh.Trimesh
        :return: None. The mesh is modified in place.
        """
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        # 1 TODO: Implement algorithm to fix partially overlapping faces
