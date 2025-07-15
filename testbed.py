import numpy as np
import trimesh

x = np.array([[1,2, 3], [4, 5, 6], [7, 8, 9]]) #shape (n, 3)
y = np.array([[0, 2], [1, 2], [0, 1]]) #shape (n, 2)

z = x[y.T]
z0 = z[0]
z1 = z[1]

print((z0+z1)/2)