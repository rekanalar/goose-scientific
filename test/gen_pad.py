"""Print the NumPy values that test/suite/random_tests.goose pins for pad_axes.
Run it with a Python that holds NumPy 2.5.4."""
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]], float)
b = np.array([[1, 2, 3, 4], [2, -1, 0, 5], [0, 3, 1, -2]], float)


def show(name, x, width, mode, **kw):
    print(name, np.pad(x, width, mode, **kw).tolist())


w = ((1, 1), (2, 1))
for mode in ("maximum", "mean", "minimum", "median"):
    show("a " + mode, a, w, mode, stat_length=((1, 2), (2, 3)))
show("a linear_ramp", a, w, "linear_ramp", end_values=((5, 6), (7, 8)))
show("a constant", a, ((1, 1), (1, 2)), "constant", constant_values=((1, 2), (3, 4)))
w = ((2, 1), (1, 3))
for mode in ("maximum", "mean", "minimum", "median"):
    show("b " + mode, b, w, mode, stat_length=((1, 2), (2, 3)))
show("b linear_ramp", b, w, "linear_ramp", end_values=((1, 2), (3, 4)))
show("b constant", b, w, "constant", constant_values=((1, 2), (3, 4)))
show("a nan constant", a, 1, "constant", constant_values=np.nan)
