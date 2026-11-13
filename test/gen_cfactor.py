"""Print the NumPy values that test/suite/cfactor_tests.goose pins for the complex factors.

The script prints the two factors of the QR decomposition and the singular
values of the pinned matrices. It also prints twenty random complex matrices
with their factors and their singular values.

Run it with a Python that holds NumPy 2.5.4."""
import numpy as np

rng = np.random.default_rng(29601)


def show(name, x):
    print(name, np.asarray(x).tolist())


def show_pair(name, x):
    x = np.asarray(x)
    print(name + " real", np.real(x).tolist())
    print(name + " imag", np.imag(x).tolist())


A1 = np.array([[1 + 1j, 2 + 0j], [3 - 2j, 4 + 1j]])
A2 = np.array([[1 + 0j, 2 - 1j], [0 + 3j, 1 + 1j], [2 + 0j, -1 + 2j]])
A3 = np.array([[1 + 1j, 2 + 0j, 0 - 1j], [3 - 2j, 4 + 1j, 1 + 0j]])
A4 = np.array([[1 + 1j, 2 + 2j], [2 + 2j, 4 + 4j]])
A5 = np.zeros((2, 2), complex)
A6 = np.array([[3 - 4j]])

for name, a in [("A1", A1), ("A2", A2), ("A3", A3), ("A4", A4), ("A5", A5), ("A6", A6)]:
    q, r = np.linalg.qr(a)
    u, s, vh = np.linalg.svd(a)
    show(name, a)
    show_pair(name + " q", q)
    show_pair(name + " r", r)
    show(name + " s", s)
    print(name + " shapes", q.shape, r.shape, u.shape, s.shape, vh.shape)
    print(name + " diag r", np.diag(r).tolist())

print("random")
for k in range(20):
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(m)
    u, s, vh = np.linalg.svd(m)
    show("  m %d" % k, m)
    show_pair("  q %d" % k, q)
    show_pair("  r %d" % k, r)
    show("  s %d" % k, s)
