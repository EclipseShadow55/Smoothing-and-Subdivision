import numpy as np
from numpy import float16, float32, float64, int16, int32, int64
from numpy.typing import NDArray
import json

PI2 = 2 * np.pi

class Object:
    def __init__(self, form: str, points: NDArray[int16 | int32 | int64 | float16 | float32 | float64], faces: NDArray[int16 | int32 | int64 | float16 | float32 | float64]):
        self.points = points
        self.faces = faces
        self.form = form

    def to_dict(self) -> dict:
        """
        Convert the object to a dictionary representation.
        :return: Dictionary with keys 'points', 'faces', and 'format'.
        """
        return {
            'points': self.points.tolist(),
            'faces': self.faces.tolist(),
            'format': self.form
        }

def convert_verts_to_rad(verts) -> NDArray[float16 | float32 | float64]:
    """
    Convert an array of vertices from Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, elevation, radius).
    :param verts: Array of shape (n, 3) where n is the number of vertices. Each vertex is defined by its x, y, z coordinates.
    :return: Array of shape (n, 3) where each vertex is defined by its azimuth, elevation, and radius.
    """
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    azimuth = np.atan2(z, x) # atan2 returns values in the range [-pi, pi]
    azimuth = np.where(azimuth < 0, azimuth + PI2, azimuth) % PI2 / PI2 # Normalize to [0, 1)
    hmag = np.sqrt(x**2 + z**2)
    elevation = np.arctan(y, hmag) * 2 / np.pi # Normalize to [-1, 1]
    radius = np.sqrt(x**2 + y**2 + z**2)
    return np.column_stack((azimuth, elevation, radius))

def convert_verts_to_xyz(verts) -> NDArray[float16 | float32 | float64]:
    """
    Convert an array of vertices from spherical coordinates (azimuth, elevation, radius) to Cartesian coordinates (x, y, z).
    :param verts: Array of shape (n, 3) where n is the number of vertices. Each vertex is defined by its azimuth, elevation, and radius.
    :return: Array of shape (n, 3) where each vertex is defined by its x, y, z coordinates.
    """
    azimuth = verts[:, 0]
    elevation = verts[:, 1]
    radius = verts[:, 2]
    x = radius * np.cos(azimuth * PI2) * np.cos(elevation * np.pi / 2)
    y = radius * np.sin(elevation * np.pi / 2)
    z = radius * np.sin(azimuth * PI2) * np.cos(elevation * np.pi / 2)
    return np.column_stack((x, y, z))

def to_radptf(obj:Object, radptf_file_path: str) -> None:
    """
    Convert points and faces to a RADPTF file format.
    :param obj: Object containing points and faces.
    :param radptf_file_path: Path to save the RADPTF file.
    :return: None
    """
    points = obj.points
    faces = obj.faces
    if obj.form == 'xyz':
        points = convert_verts_to_rad(points)
    elif obj.form != 'radptf':
        raise ValueError(f"Unsupported format: {obj.form}. Supported formats are 'radptf' and 'xyz'.")
    with open(radptf_file_path, 'w') as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

def to_json(obj:Object, json_file_path: str) -> None:
    """
    Convert points and faces to a JSON file format.
    :param obj: Object containing points and faces.
    :param json_file_path: Path to save the JSON file.
    :return: None
    """
    with open(json_file_path, 'w') as f:
        json.dump(obj.to_dict(), f, indent=4)

def to_obj(obj:Object, obj_file_path: str)-> None:
    """
    Convert points and faces to an OBJ file format.
    :param obj: Object containing points and faces.
    :param obj_file_path: Path to save the OBJ file.
    :return: None
    """
    points = obj.points
    faces = obj.faces
    if obj.form == 'radptf':
        points = convert_verts_to_xyz(points)
    elif obj.form != 'xyz':
        raise ValueError(f"Unsupported format: {obj.form}. Supported formats are 'radptf' and 'xyz'.")
    with open(obj_file_path, 'w') as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

def from_radptf(radptf_file_path: str) -> Object:
    """
    Load points and faces from a RADPTF file format.
    :param radptf_file_path: Path to the RADPTF file.
    :return: Object containing points and faces.
    """
    points = []
    faces = []
    with open(radptf_file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                point = list(map(float, line.split()[1:]))
                points.append(point)
            elif line.startswith('f '):
                face = list(map(int, line.split()[1:]))
                faces.append(face)
    return Object(form='radptf', points=np.array(points), faces=np.array(faces))

def from_json(json_file_path: str) -> Object:
    """
    Load points and faces from a JSON file format.
    :param json_file_path: Path to the JSON file.
    :return: Object containing points and faces.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return Object(form=data['format'], points=np.array(data['points']), faces=np.array(data['faces']))

def from_obj(obj_file_path: str) -> Object:
    """
    Load points and faces from an OBJ file format.
    :param obj_file_path: Path to the OBJ file.
    :return: Object containing points and faces.
    """
    points = []
    faces = []
    with open(obj_file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                point = list(map(float, line.split()[1:]))
                points.append(point)
            elif line.startswith('f '):
                face = list(map(int, line.split()[1:]))
                faces.append(face)
    return Object(form='xyz', points=np.array(points), faces=np.array(faces))