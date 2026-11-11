"""Print the SciPy values that test/run.goose pins for qz and ordqz.
Run it with a Python that holds NumPy and SciPy 1.18.1."""
import numpy as np
from scipy.linalg import qz, ordqz

A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]], float)
B = np.array([[0, 6, 0, 0], [5, 0, 2, 1], [5, 2, 6, 6], [4, 7, 7, 7]], float)
np.set_printoptions(precision=17, linewidth=140)


def show(name, a, b):
    aa, bb, q, z = qz(a, b)
    print(name)
    print(" diag AA", np.diag(aa))
    print(" diag BB", np.diag(bb))
    print(" reconstruction", abs(q @ aa @ z.T - a).max())


show("pinned", A, B)
aa, bb, q, z = qz(A, B)
print("Q", q)
for rule in ("lhp", "rhp", "iuc", "ouc"):
    r = ordqz(A, B, sort=rule)
    print(rule, "alpha/beta", r[2] / r[3])

Bs = B.copy()
Bs[:, 1] = 0.0
show("singular B", A, Bs)
As = A.copy()
As[:, 1] = 0.0
show("singular A", As, B)
show("zeros", np.zeros((3, 3)), np.zeros((3, 3)))
show("rotation", np.array([[0, 1.0], [-1, 0]]), np.eye(2))
r = qz(np.array([[0, 1.0], [-1, 0]]), np.eye(2))
print("rotation AA", r[0], "BB", r[1])
print("side 1", qz(np.array([[3.0]]), np.array([[-2.0]])))
