# Prints the SciPy values that test/run.goose pins for the generalized
# eigenproblem of a pair of matrices. Run it with SciPy:
#   python3 test/gen_ceig_pair.py
import numpy as np
import scipy.linalg as sl

np.set_printoptions(precision=17, linewidth=200)

def flat(z):
    z = np.asarray(z).reshape(-1)
    return [float(v) for v in z.real], [float(v) for v in z.imag]

def rule(v):
    # The rule of the package: length one, largest component real and above zero.
    v = v.copy()
    for j in range(v.shape[1]):
        k = int(np.argmax(np.abs(v[:, j])))
        v[:, j] *= np.conj(v[k, j]) / abs(v[k, j])
        v[:, j] /= np.linalg.norm(v[:, j])
    return v

general = {
    "P1": ([[1 + 1j, 2], [3 - 2j, 4 + 1j]], [[2, 1j], [0, 1 - 1j]]),
    "P2": ([[1, 2 - 1j, 0], [1j, 3, 1 + 1j], [2, -1j, 1 - 2j]],
           [[2, 0, 1j], [1, 1, 0], [0, -1j, 3]]),
    "P3": ([[1, 2], [3, 4]], [[1, 0], [0, 0]]),
}
for name, (ra, rb) in general.items():
    a = np.array(ra, dtype=complex)
    b = np.array(rb, dtype=complex)
    w, v = sl.eig(a, b)
    key = [(1e300 if not np.isfinite(x) else x.real, 0 if not np.isfinite(x) else x.imag) for x in w]
    order = sorted(range(len(w)), key=lambda i: key[i])
    w = w[order]
    v = rule(v[:, order])
    print(name, "a", flat(a))
    print(name, "b", flat(b))
    print(name, "w", flat(w))
    print(name, "v", flat(v))

herm = {
    "D1": ([[2, 1 - 1j], [1 + 1j, 3]], [[4, 1j], [-1j, 2]]),
    "D2": ([[2, 1 - 1j, 0], [1 + 1j, 3, 1j], [0, -1j, 1]],
           [[3, 1, 0], [1, 2, 1j], [0, -1j, 4]]),
}
for name, (ra, rb) in herm.items():
    a = np.array(ra, dtype=complex)
    b = np.array(rb, dtype=complex)
    w, v = sl.eigh(a, b)
    v = v.copy()
    for j in range(v.shape[1]):
        k = int(np.argmax(np.abs(v[:, j])))
        v[:, j] *= np.conj(v[k, j]) / abs(v[k, j])
    print(name, "a", flat(a))
    print(name, "b", flat(b))
    print(name, "w", [float(x) for x in w])
    print(name, "v", flat(v))
