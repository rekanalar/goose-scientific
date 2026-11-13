"""Print the SciPy values that test/suite/dare_singular_tests.goose pins for solve_discrete_are
with a singular r. Run it with a Python that holds SciPy 1.18.1."""
import numpy as np
from scipy.linalg import solve_discrete_are as dare

A = np.array([[0.9, 0.5], [0, 0.7]])
B = np.eye(2)
Q = np.eye(2)
v = np.array([0.1, 0.7])
for name, R in (("ones", np.array([[1.0, 1], [1, 1]])),
                ("diag10", np.diag([1.0, 0])),
                ("zero", np.zeros((2, 2))),
                ("rank1", np.outer(v, v))):
    print(name, dare(A, B, Q, R).tolist())

A3 = np.array([[0.9, 0.5, 0.1], [0, 0.7, 0.2], [0.1, 0, 0.6]])
B3 = np.array([[1.0, 0], [0, 1], [1, 1]])
Q3 = np.diag([1.0, 2, 3])
print("three", dare(A3, B3, Q3, np.array([[1.0, 1], [1, 1]])).tolist())

E = np.array([[1.0, 0.2], [0, 1.5]])
S = np.array([[0.1, 0.0], [0.0, 0.2]])
Rs = np.array([[1.0, 1], [1, 1]])
print("es", dare(A, B, Q, Rs, e=E, s=S).tolist())
