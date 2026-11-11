"""Print the NumPy values that test/run.goose pins for the complex calls.

The script prints the products, the Cholesky factors and the transforms
of the pinned arrays. It also prints random Hermitian matrices with
their factors and random complex arrays with their transforms.

Run it with a Python that holds NumPy 2.5.4 and SciPy 1.18.1."""
import numpy as np
import scipy.linalg as sl

rng = np.random.default_rng(20261111)


def show(name, x):
    print(name, np.asarray(x).tolist())


a = np.array([1 + 2j, 3 - 1j])
b = np.array([2 - 1j, -1 + 0.5j])
A = np.array([[2 + 0j, 1 + 1j], [1 - 1j, 3 + 0j]])
v = np.array([1 + 0j, 2 + 0j])

show("dot a b", np.dot(a, b))
show("dot A A", np.dot(A, A))
show("dot A v", np.dot(A, v))
show("dot v A", np.dot(v, A))
show("dot one", np.dot(np.array([1 + 2j]), np.array([3 - 1j])))
show("vdot one", np.vdot(np.array([1 + 2j]), np.array([3 - 1j])))
show("dot 1x1", np.dot(np.array([[2 + 1j]]), np.array([[2 + 1j]])))
show("chol 1x1", np.linalg.cholesky(np.array([[4 + 0j]])))
show("fftn 1x1", np.fft.fftn(np.array([[1 + 2j]])))
show("fftn zeros", np.fft.fftn(np.zeros((2, 2), complex)))
show("vdot a b", np.vdot(a, b))
show("vdot b a", np.vdot(b, a))

show("chol A", np.linalg.cholesky(A))
B = np.array([[4 + 0j, 9 + 9j], [1 + 1j, 2 + 0j]])
show("chol B", np.linalg.cholesky(B))
C = np.array([[4 + 1j, 0 + 0j], [0 + 0j, 2 + 0j]])
show("chol C", np.linalg.cholesky(C))

D = np.array([[1 + 1j, 2 + 0j], [3 - 1j, 4 + 2j]])
show("fftn D", np.fft.fftn(D))
show("ifftn D", np.fft.ifftn(D))
E = np.array([[1 + 2j, 2 + 0j, -1j], [3 - 4j, 0 + 0j, 1 + 1j]])
show("fftn E", np.fft.fftn(E))
show("ifftn E", np.fft.ifftn(E))

print("chol random")
for k in range(20):
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = m + m.conj().T + 8.0 * np.eye(4)
    show("  m %d" % k, h)
    show("  l %d" % k, np.linalg.cholesky(h))

print("fft random")
for k in range(20):
    ndim = 1 + (k % 3)
    shape = (3,) * ndim
    x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    show("  x %d" % k, x)
    show("  f %d" % k, np.fft.fftn(x))
    show("  g %d" % k, np.fft.ifftn(x))

print("scipy chol B default")
try:
    show("scipy chol B", sl.cholesky(B))
except sl.LinAlgError as err:
    print("scipy chol B error", err)
show("scipy chol B lower", sl.cholesky(B, lower=True))
