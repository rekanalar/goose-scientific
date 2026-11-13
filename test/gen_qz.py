"""Print the SciPy values that test/suite/qz_tests.goose pins for qz and ordqz.
The same pins hold test/suite/ordqz_tests.goose and test/suite/qz_c_tests.goose.
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

# The complex pair. The eigenvalues, the diagonal of BB and the first
# group of each rule.
CA = np.array([[2 + 1j, 5 - 1j, 8, 7 + 2j], [5, 2 + 3j, 2, 8 - 1j],
               [7 - 2j, 5, 6 + 1j, 6], [5, 4 + 1j, 4, 8 - 2j]])
CB = np.array([[1j, 6, 0, 0], [5, 1, 2 - 1j, 1],
               [5 + 1j, 2, 6, 6 - 1j], [4, 7, 7j, 7]])
caa, cbb, cq, cz = qz(CA, CB)
ev = np.diag(caa) / np.diag(cbb).real
print("complex eigenvalues", sorted(ev, key=lambda w: (w.real, w.imag)))
print("complex diag BB", np.diag(cbb))
print("complex reconstruction", abs(cq @ caa @ cz.conj().T - CA).max())
for rule in ("lhp", "rhp", "iuc", "ouc"):
    r = ordqz(CA, CB, sort=rule)
    print("complex", rule, "alpha/beta", r[2] / r[3])

# The relative reconstruction error of SciPy at four scales.
for s in (1e-300, 1e-150, 1e150, 1e300):
    aa, bb, q, z = qz(A * s, B * s)
    ea = abs(q @ aa @ z.T - A * s).max() / abs(A * s).max()
    print("scale", s, "relative error", ea)

# A callable sort: the eigenvalues of modulus below 1 first.
r = ordqz(A, B, sort=lambda al, be: abs(al) < abs(be))
print("callable abs(alpha)<beta alpha/beta", r[2] / r[3])
