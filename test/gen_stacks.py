#!/usr/bin/env python3
"""Print the SciPy values that the stack tests pin.

The script needs SciPy 1.18.1. Run it from the directory of the
package. It prints the buffers of the operands and of the results. The
test file holds the same numbers.
"""

import numpy as np
from scipy.linalg import rsf2csf, cdf2rdf, bandwidth


def fmt(values):
    """One flat list of the values, with the complex parts side by side."""
    flat = []
    for value in np.asarray(values).ravel():
        if np.iscomplexobj(value) or isinstance(value, complex):
            flat.append(float(np.real(value)))
            flat.append(float(np.imag(value)))
        else:
            flat.append(float(value))
    return "[" + ", ".join(repr(v) for v in flat) + "]"


def show(name, value):
    print(name, "=", fmt(value))


def main():
    import scipy
    print("scipy", scipy.__version__)

    # A real Schur form of three rows and its change of the basis.
    t0 = np.array([
        [2.6589670819169946, 1.4244045756721777, -1.9293343948770278],
        [0.0, -0.32948354095849702, -0.49063703822841376],
        [0.0, 1.3117892147840797, -0.32948354095849702],
    ])
    z0 = np.array([
        [0.72711590900990863, -0.60156188058076243, 0.33079564491817354],
        [0.52839427726760835, 0.79801891979128614, 0.28976765072380067],
        [0.43829435613014156, 0.035904143037891308, -0.89811410739236175],
    ])
    # A second real Schur form. The block of two rows holds a complex
    # pair.
    t1 = np.array([
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 3.0],
        [0.0, -1.0, 4.0],
    ])
    z1 = np.eye(3)

    t = np.stack([t0, t1])
    z = np.stack([z0, z1])
    tc, zc = rsf2csf(t, z)
    print("rsf2csf stack shape", tc.shape)
    show("t", t)
    show("z", z)
    show("tc", tc)
    show("zc", zc)

    # One change of the basis over a stack of two forms.
    tb = np.stack([t0, t1])
    tc2, zc2 = rsf2csf(tb, z0)
    print("rsf2csf broadcast shape", tc2.shape)
    show("tc2", tc2)
    show("zc2", zc2)

    # A stack of two changes of the basis over one form.
    tc3, zc3 = rsf2csf(t0, np.stack([z0, z1]))
    print("rsf2csf broadcast reverse shape", tc3.shape)
    show("tc3", tc3)
    show("zc3", zc3)

    # The complex eigenvalues and eigenvectors of one set.
    w0 = np.array([1.0, 4.0 + 5.0j, 4.0 - 5.0j])
    v0 = np.array([
        [1.0 + 0.0j, -0.019055398349167637 - 0.4001633653325205j,
         -0.019055398349167637 + 0.4001633653325205j],
        [0.0 + 0.0j, 0.0 - 0.64788354387169989j, 0.0 + 0.64788354387169989j],
        [0.0 + 0.0j, 0.6478835438717 + 0.0j, 0.6478835438717 + 0.0j],
    ])
    # A second set.
    w1 = np.array([2.0 + 3.0j, 2.0 - 3.0j, 7.0 + 0.0j])
    v1 = np.array([
        [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j],
        [4.0 + 0.0j, 0.0 + 5.0j, 6.0 + 0.0j],
        [7.0 + 0.0j, 8.0 + 0.0j, 9.0 + 1.0j],
    ])

    w = np.stack([w0, w1])
    v = np.stack([v0, v1])
    wr, vr = cdf2rdf(w, v)
    print("cdf2rdf stack shape", wr.shape, vr.shape)
    show("w", w)
    show("v", v)
    show("wr", wr)
    show("vr", vr)

    # One set of eigenvalues over two sets of eigenvectors.
    wr2, vr2 = cdf2rdf(w0[None, :], v)
    print("cdf2rdf broadcast shape", wr2.shape, vr2.shape)
    show("wr2", wr2)
    show("vr2", vr2)

    # A real value between two complex values.
    w2 = np.array([1.0 + 1.0j, 2.0 + 0.0j, 1.0 - 1.0j])
    wr3, vr3 = cdf2rdf(w2, v1)
    print("cdf2rdf pair shape", wr3.shape, vr3.shape)
    show("w2", w2)
    show("wr3", wr3)
    show("vr3", vr3)

    # The interleaved case. SciPy pairs the complex values in the order
    # they appear.
    w3 = np.array([1.0 + 1.0j, 2.0 + 1.0j, 1.0 - 1.0j, 2.0 - 1.0j])
    v3 = np.array([
        [1.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 1.0j, 2.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        [3.0 + 0.0j, 0.0 + 0.0j, 4.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 2.0j, 0.0 + 0.0j, 5.0 + 0.0j],
    ])
    wr4, vr4 = cdf2rdf(w3, v3)
    print("cdf2rdf interleaved shape", wr4.shape, vr4.shape)
    show("w3", w3)
    show("v3", v3)
    show("wr4", wr4)
    show("vr4", vr4)

    # The counts of the diagonals of a stack.
    a = np.array([
        [[1.0, 0.0, 2.0, 0.0],
         [0.0, 3.0, 0.0, 0.0],
         [4.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 2.0, 0.0, 3.0]],
    ])
    lower, upper = bandwidth(a)
    print("bandwidth stack", lower, upper)
    show("a", a)

    # The counts of a matrix of zeros and of a matrix without a batch.
    zeros = np.zeros((2, 3, 3))
    print("bandwidth zeros", bandwidth(zeros))
    print("bandwidth matrix", bandwidth(a[0]))


main()
