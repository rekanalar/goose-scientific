# Prints the NumPy values that test/suite/ceig_tests.goose and test/suite/ceigh_hostile_tests.goose pins for the eigenvalues and the
# eigenvectors of complex matrices. Run it with NumPy 2.5.4:
#   python3 test/gen_ceig.py
import numpy as np

np.set_printoptions(precision=17, linewidth=200)

def flat(z):
    z = np.asarray(z).reshape(-1)
    return [float(v) for v in z.real], [float(v) for v in z.imag]

def phase_fix(v):
    # The rule of the package: the largest component of each column is real
    # and above zero. NumPy fixes no phase for the Hermitian case.
    v = v.copy()
    for j in range(v.shape[1]):
        k = int(np.argmax(np.abs(v[:, j])))
        v[:, j] *= np.conj(v[k, j]) / abs(v[k, j])
    return v

general = {
    "G1": [[1 + 1j, 2], [3 - 2j, 4 + 1j]],
    "G2": [[1, 2 - 1j, 0], [1j, 3, 1 + 1j], [2, -1j, 1 - 2j]],
    "G3": [[2, 1 + 1j, 3], [0, 1j, 2], [0, 0, -1]],
    "G4": [[3 - 4j]],
    "G5": [[1, 1], [0, 1]],
    "G6": [[0, 0], [0, 0]],
}
for name, rows in general.items():
    a = np.array(rows, dtype=complex)
    w, v = np.linalg.eig(a)
    order = sorted(range(len(w)), key=lambda i: (round(w[i].real, 9), round(w[i].imag, 9)))
    w = w[order]
    v = v[:, order]
    print(name, "a", flat(a))
    print(name, "w", flat(w))
    print(name, "v", flat(v))

herm = {
    "H1": [[2, 1 - 1j], [1 + 1j, 3]],
    "H2": [[2, 1 - 1j, 0], [1 + 1j, 3, 1j], [0, -1j, 1]],
    "H3": [[2, 0], [0, 2]],
    "H4": [[5]],
    "H5": [[1, 5, 5], [2 + 1j, 3, 5], [4, 1 - 2j, 2]],
}
for name, rows in herm.items():
    a = np.array(rows, dtype=complex)
    w, v = np.linalg.eigh(a)
    print(name, "a", flat(a))
    print(name, "w", [float(x) for x in w])
    print(name, "v", flat(phase_fix(v)))
    wu, vu = np.linalg.eigh(a, UPLO="U")
    print(name, "wu", [float(x) for x in wu])
