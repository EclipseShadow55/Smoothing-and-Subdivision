import numpy as np
import trimesh


class TrimeshGeometry:
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

        return TrimeshGeometry._edges_in_mesh(mesh, edges)

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
        mesh_edges = np.sort(mesh.edges_unique(), axis=1)
        # View as structured arrays for row-wise comparison
        dtype = [('v0', sorted_edges.dtype), ('v1', sorted_edges.dtype)]
        sorted_edges_view = sorted_edges.view(dtype)
        mesh_edges_view = mesh_edges.view(dtype)
        # Use np.isin for fast membership check
        mask = np.isin(sorted_edges_view, mesh_edges_view)

        return mask

    @staticmethod
    def get_neighbors(mesh: trimesh.Trimesh, edges) -> np.ndarray:
        """
        Get the neighboring faces for each edge in an ndarray of edges. Public method that validates input types and dimensions.
        Note: This method only works for watertight meshes and when every provided edge is present in the mesh.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :param edges: An ndarray of shape (n, 2) of edges or pairs of vertex indices, where n is the number of edges.
        :type edges: np.ndarray
        :return: ndarray of shape (n, 2) of face indices, where n is the number of edges
        :rtype: np.ndarray
        :raises TypeError: If mesh is not a trimesh.Trimesh object or edges is not a numpy ndarray.
        :raises ValueError: If edges is not a 2D array with shape (n, 2), if edges is empty, if mesh is empty, if mesh is not watertight, or if some edges are not present in the mesh.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if not isinstance(edges, np.ndarray):
            raise TypeError("Edges must be a numpy ndarray.")
        if not (edges.ndim == 2 and edges.shape[1] == 2):
            raise ValueError("Edges must be a 2D array with shape (n, 2).")
        if edges.shape[0] == 0:
            return np.empty((0, 2), dtype=int)
        if mesh.is_empty:
            return np.empty((0, 2), dtype=int)
        if not mesh.is_watertight:
            raise ValueError("Mesh must be watertight to compute neighbors.")
        if not np.all(TrimeshGeometry._edges_in_mesh(mesh, edges)):
            raise ValueError("Some edges are not present in the mesh.")

        return TrimeshGeometry._get_neighbors(mesh, edges)

    @staticmethod
    def _get_neighbors(mesh: trimesh.Trimesh, edges: np.ndarray) -> np.ndarray:
        """
        Get the neighboring faces for each edge in an ndarray of edges. Private method that performs the actual computation with no input validation.
        Note: This method only works for watertight meshes and when every provided edge is present in the mesh.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :param edges: An ndarray of shape (n, 2) of edges or pairs of vertex indices, where n is the number of edges.
        :type edges: np.ndarray
        :return: ndarray of shape (n, 2) of face indices, where n is the number of edges
        :rtype: np.ndarray
        """
        adj_ind_lookup = np.sort(mesh.face_adjacency_edges, axis=1)
        faces_lookup = mesh.face_adjacency
        dtype = [('v0', adj_ind_lookup.dtype), ('v1', adj_ind_lookup.dtype)]
        edges_view = edges.view(dtype)
        adj_edges_view = adj_ind_lookup.view(dtype)
        matches = np.isin(edges_view, adj_edges_view)
        indices = np.argmax(matches, axis=1)
        neighbors = faces_lookup[indices]

        return neighbors

    @staticmethod
    def edge_angles(mesh, edges: np.ndarray, neighbors: np.ndarray | None = None) -> np.ndarray:
        """
        Calculate the angle of each edge in a mesh. Public method that validates input types and dimensions.
        Note: This method only works for watertight meshes and when every provided edge is present in the mesh.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :param edges: An ndarray of shape (n, 2) of edges or pairs of vertex indices, where n is the number of edges.
        :type edges: np.ndarray
        :param neighbors: An ndarray of shape (n, 2) of precomputed neighbors of edges, if available.
        :type neighbors: np.ndarray, optional
        :return: ndarray of angles in radians for each edge of shape (n) where n is the number of edges.
        :rtype: np.ndarray
        :raises TypeError: If mesh is not a trimesh.Trimesh object or edges is not a numpy ndarray.
        :raises ValueError: If edges is not a 2D array with shape (n, 2), if edges is empty, if mesh is empty, if mesh is not watertight, or if some edges are not present in the mesh.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if not isinstance(edges, np.ndarray):
            raise TypeError("Edges must be a numpy ndarray.")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Edges must be a 2D array with shape (n, 2).")
        if edges.shape[0] == 0:
            return np.zeros(0, dtype=float)
        if mesh.is_empty:
            return np.zeros(0, dtype=float)
        if not mesh.is_watertight:
            raise ValueError("Mesh must be watertight to compute edge angles.")
        if not np.all(TrimeshGeometry._edges_in_mesh(mesh, edges)):
            raise ValueError("Some edges are not present in the mesh.")
        if not isinstance(neighbors, (np.ndarray, type(None))):
            raise TypeError("Neighbors must be a numpy ndarray or None.")
        if neighbors is not None and (neighbors.ndim != 2 or neighbors.shape[1] != 2):
            raise ValueError("Neighbors must be a 2D array with shape (n, 2) or None.")

        return TrimeshGeometry._edge_angles(mesh, edges, neighbors)

    @staticmethod
    def _edge_angles(mesh: trimesh.Trimesh, edges: np.ndarray, neighbors: np.ndarray | None = None) -> np.ndarray:
        """
        Calculate the angle of each edge in a mesh. Public method that validates input types and dimensions.
        Note: This method only works for watertight meshes and when every provided edge is present in the mesh.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :param edges: An ndarray of shape (n, 2) of edges or pairs of vertex indices, where n is the number of edges.
        :type edges: np.ndarray
        :param neighbors: An ndarray of shape (n, 2) of precomputed neighbors of edges, if available.
        :type neighbors: np.ndarray, optional
        :return: ndarray of angles in radians for each edge of shape (n) where n is the number of edges.
        :rtype: np.ndarray
        """
        if neighbors is None:
            neighbors = TrimeshGeometry._get_neighbors(mesh, edges)
        normals1 = mesh.face_normals[neighbors.T[0]]
        normals2 = mesh.face_normals[neighbors.T[1]]
        cos = np.dot(normals1, normals2)
        angles = np.arccos(cos)

        return angles

    @staticmethod
    def get_edge_normals(mesh: trimesh.Trimesh, edges: np.ndarray, neighbors: np.ndarray | None = None) -> np.ndarray:
        """
        Get the normals for each edge in a mesh. Public method that validates input types and dimensions.
        Note: This method only works for watertight meshes and when every provided edge is present in the mesh.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :param edges: An ndarray of shape (n, 2) of edges or pairs of vertex indices, where n is the number of edges.
        :type edges: np.ndarray
        :param neighbors: An ndarray of shape (n, 2) of precomputed neighbors of edges, if available.
        :type neighbors: np.ndarray, optional
        :return: ndarray of normals for each edge of shape (n, 3) where n is the number of edges.
        :rtype: np.ndarray
        :raises TypeError: If mesh is not a trimesh.Trimesh object or edges is not a numpy ndarray.
        :raises ValueError: If edges is not a 2D array with shape (n, 2), if edges is empty, if mesh is empty, if mesh is not watertight, or if some edges are not present in the mesh.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Mesh must be a trimesh.Trimesh object.")
        if not isinstance(edges, np.ndarray):
            raise TypeError("Edges must be a numpy ndarray.")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Edges must be a 2D array with shape (n, 2).")
        if edges.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        if mesh.is_empty:
            return np.zeros((0, 3), dtype=float)
        if not mesh.is_watertight:
            raise ValueError("Mesh must be watertight to compute edge normals.")
        if not np.all(TrimeshGeometry._edges_in_mesh(mesh, edges)):
            raise ValueError("Some edges are not present in the mesh.")
        if not isinstance(neighbors, (np.ndarray, type(None))):
            raise TypeError("Neighbors must be a numpy ndarray or None.")

        return TrimeshGeometry._get_edge_normals(mesh, edges, neighbors)

    @staticmethod
    def _get_edge_normals(mesh: trimesh.Trimesh, edges: np.ndarray, neighbors: np.ndarray | None = None) -> np.ndarray:
        """
        Get the normals for each edge in a mesh. Private method that performs the actual computation with no input validation.
        Note: This method only works for watertight meshes and when every provided edge is present in the mesh.

        :param mesh: The mesh to analyze.
        :type mesh: trimesh.Trimesh
        :param edges: An ndarray of shape (n, 2) of edges or pairs of vertex indices, where n is the number of edges.
        :type edges: np.ndarray
        :param neighbors: An ndarray of shape (n, 2) of precomputed neighbors of edges, if available.
        :type neighbors: np.ndarray, optional
        :return: ndarray of normals for each edge of shape (n, 3) where n is the number of edges.
        :rtype: np.ndarray
        """
        if neighbors is None:
            neighbors = TrimeshGeometry._get_neighbors(mesh, edges)
        normals = mesh.face_normals.reshape(-1, 3)
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

        return TrimeshGeometry._get_opposite(faces, edges)

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
    def smooth(mesh: trimesh.Trimesh, alpha: float, iterations: int = 1):
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
        TrimeshGeometry._smooth(mesh, alpha, iterations)

    @staticmethod
    def _smooth(mesh: trimesh.Trimesh, alpha: float, iterations: int = 1):
        """
        Smooth a mesh in place using intelligent edge and vertex addition. Private method that performs the actual smoothing with no input validation.

        :param mesh: The mesh to smooth.
        :type mesh: trimesh.Trimesh
        :param alpha: A scaling factor for the amount of change applied to each vertex and edge.
        :type alpha: float
        :param iterations: Number of smoothing iterations to perform.
        :type iterations: int
        :return: None. The mesh is modified in place.
        """
        for _ in range(iterations):
            edges = mesh.edges()
            neighbors = TrimeshGeometry._get_neighbors(mesh, edges)
            angles = TrimeshGeometry._edge_angles(mesh, edges, neighbors)
            factors = np.abs((np.pi - angles) / np.pi) * alpha
            edge_normals = TrimeshGeometry._get_edge_normals(mesh, edges, neighbors)
            ie_normals = -edge_normals
            midpoints = mesh.vertices[edges].mean(axis=1)
            opposites1 = TrimeshGeometry._get_opposite(neighbors.T[0], edges)
            opposites2 = TrimeshGeometry._get_opposite(neighbors.T[1], edges)
            vectors1 = midpoints - mesh.vertices[opposites1]
            vectors2 = midpoints - mesh.vertices[opposites2]
