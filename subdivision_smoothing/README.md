# Subdivision and Smoothing
This library contains a class `SubdivisionSmoothing` that provides methods for performing subdivision and smoothing operations on 3D meshes. The class includes methods for Simple Subdivsion, Laplacian Smoothing, and a custom Subdivision Method that takes into account vertex position and edge sharpness. It also includes some helpful methods for making a subdivision or smoothing method yourself.

## Installation
To use this library, you need to have Python installed.
1. Create and navigate to your project directory.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the package using pip:
   For normal installation:
   ```bash
   pip install subdivision-smoothing
   ```
    For development installation:
    ```bash
    pip install subdivision-smoothing[dev]
    ```
4. For development, clone the `subdivision_smoothing\debug` dir from the repository. You can use the `debug.py` script to visualize the intermediate steps of the algorithms.
