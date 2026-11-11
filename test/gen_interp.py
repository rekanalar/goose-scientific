# Prints the SciPy values that test/run.goose pins for the interpolator of
# scattered data. Run it with SciPy:
#   python3 test/gen_interp.py
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator

def flat(a):
    return [float(x) for x in np.asarray(a).reshape(-1)]

p = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], float)
v = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], float)
xi = np.array([[0.25, 0.25], [0.75, 0.5], [2, 2], [0.5, 0.1]], float)
print("A linear", flat(griddata(p, v, xi, method="linear", fill_value=-1)))
print("A nearest", flat(griddata(p, v, xi, method="nearest")))
print("A object", flat(LinearNDInterpolator(p, v[:, 0], fill_value=-1)(xi)))
print("A rescale", flat(griddata(p, v, xi, method="linear", rescale=True)))

rng = np.random.default_rng(5)
p2 = rng.random((12, 2)) * np.array([4.0, 1.0])
v2 = rng.random((12, 2))
q2 = rng.random((6, 2)) * np.array([4.0, 1.0])
print("R points", flat(p2))
print("R values", flat(v2))
print("R query", flat(q2))
print("R linear", flat(griddata(p2, v2, q2, method="linear", fill_value=-1)))
print("R rescale", flat(griddata(p2, v2, q2, method="linear", fill_value=-1, rescale=True)))
print("R nearest", flat(griddata(p2, v2, q2, method="nearest")))

p1 = np.array([3.0, 1.0, 2.0, 5.0])
v1 = np.array([[30, 3], [10, 1], [20, 2], [50, 5.5]], float)
q1 = np.array([1.5, 2.5, 4.0, 6.0, 0.5])
print("L linear", flat(griddata(p1, v1, q1, method="linear", fill_value=-1)))
print("L nearest", flat(griddata(p1, v1, q1, method="nearest")))

p3 = rng.random((8, 3))
v3 = rng.random(8)
q3 = rng.random((4, 3))
print("N points", flat(p3))
print("N values", flat(v3))
print("N query", flat(q3))
print("N nearest", flat(NearestNDInterpolator(p3, v3)(q3)))
