#!/usr/bin/env python3
"""Write the pins of the tests of the four methods of minimization.

The script needs Python 3, NumPy 2.5.4 and SciPy 1.18.1. It runs from
the maintainer. The user of the package installs nothing. Run it from
the package directory:

    /home/ubuntu/works/goose/tmp/venv/bin/python -I test/gen_optimize_fmin.py

The file it writes holds numbers only. Each row holds seven numbers:
the code of the problem, the two values of the start point, the two
values of the answer of SciPy, the value of the function at the answer
and the tolerance of the comparison.

The problems are:

1 the sphere x0^2 + 2 x1^2
2 the function of Rosenbrock
3 the square to the point (1, 2) with the constraint x0 + x1 - 2
4 the square to the point (-0.5, 0.25) under bounds
6 the function of Rosenbrock with the equality x1 - x0^3 - 1
7 the square to the point (2, 1) under bounds
8 the square with a cross term x0^2 + x0 x1 + x1^2

The hand-written test file holds the same problems and the same codes.
"""
import numpy as np
from scipy.optimize import fmin_powell, fmin_cobyla, fmin_slsqp, fmin_tnc

OUT = "test/suite/optimize_fmin_pins.goose"


def sphere(x):
    return x[0] ** 2 + 2.0 * x[1] ** 2


def sphere_grad(x):
    return np.array([2.0 * x[0], 4.0 * x[1]])


def rosen(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    return np.array([-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
                     200.0 * (x[1] - x[0] ** 2)])


def quad(x):
    return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2


def quad_grad(x):
    return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)])


def quad_bound(x):
    return (x[0] + 0.5) ** 2 + (x[1] - 0.25) ** 2


def quad_bound_grad(x):
    return np.array([2.0 * (x[0] + 0.5), 2.0 * (x[1] - 0.25)])


def far_quad(x):
    return (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2


def far_quad_grad(x):
    return np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)])


def cross_quad(x):
    return x[0] * x[0] + x[0] * x[1] + x[1] * x[1]


def cross_quad_grad(x):
    return np.array([2.0 * x[0] + x[1], x[0] + 2.0 * x[1]])


def fmt(x):
    s = repr(float(x))
    if "e" in s:
        m, _, e = s.partition("e")
        if "." not in m:
            m = m + ".0"
        s = m + "e" + e
    return s


def rows(name, cases):
    """Write the list of the rows of one method."""
    out = []
    for case, x0, ans, fval, tol in cases:
        out.append([case, x0[0], x0[1], ans[0], ans[1], fval, tol])
    body = ", ".join(fmt(v) for r in out for v in r)
    return "const %s: f64[>..] = [%s];\n" % (name, body)


def main():
    powell = []
    # The sphere from the point (2, 3).
    x0 = np.array([2.0, 3.0])
    a = fmin_powell(sphere, x0, xtol=1.0e-4, ftol=1.0e-4)
    powell.append((1, x0, a, sphere(a), 1.0e-4))
    # The function of Rosenbrock from the point (-1.2, 1).
    x0 = np.array([-1.2, 1.0])
    a = fmin_powell(rosen, x0, xtol=1.0e-4, ftol=1.0e-4)
    powell.append((2, x0, a, rosen(a), 1.0e-4))

    cobyla = []
    # The square with the constraint from the point (0, 0).
    x0 = np.array([0.0, 0.0])
    a = fmin_cobyla(quad, x0, [lambda x: x[0] + x[1] - 2.0],
                    rhobeg=1.0, rhoend=1.0e-8, maxfun=5000, catol=1.0e-8)
    cobyla.append((3, x0, a, quad(a), 1.0e-4))
    # The sphere without a constraint from the point (2, 3).
    x0 = np.array([2.0, 3.0])
    a = fmin_cobyla(sphere, x0, [], rhobeg=1.0, rhoend=1.0e-8,
                    maxfun=5000, catol=1.0e-8)
    cobyla.append((1, x0, a, sphere(a), 1.0e-4))

    slsqp = []
    # The square with the constraint from the point (0, 0).
    x0 = np.array([0.0, 0.0])
    a = fmin_slsqp(quad, x0, eqcons=[], ieqcons=[lambda x: x[0] + x[1] - 2.0],
                   acc=1.0e-10, iter=100)
    slsqp.append((3, x0, a, quad(a), 1.0e-6))
    # The function of Rosenbrock with the equality from the point
    # (0.5, 2).
    x0 = np.array([0.5, 2.0])
    a = fmin_slsqp(rosen, x0, eqcons=[lambda x: x[1] - x[0] ** 3 - 1.0],
                   acc=1.0e-10, iter=100)
    slsqp.append((6, x0, a, rosen(a), 1.0e-6))
    # The square under bounds from the point (0, 0).
    x0 = np.array([0.0, 0.0])
    a = fmin_slsqp(quad_bound, x0, eqcons=[], ieqcons=[],
                   bounds=[(-1.0, 0.3), (-1.0, 0.1)], acc=1.0e-10, iter=100)
    slsqp.append((4, x0, a, quad_bound(a), 1.0e-6))

    tnc = []
    # The square to the point (2, 1) under bounds from the point (0, 0).
    x0 = np.array([0.0, 0.0])
    res = fmin_tnc(far_quad, x0, fprime=far_quad_grad,
                   bounds=[(0.0, 1.0), (-5.0, 5.0)], disp=0)
    tnc.append((7, x0, res[0], far_quad(res[0]), 1.0e-5))
    # The sphere from the point (2, 3).
    x0 = np.array([2.0, 3.0])
    res = fmin_tnc(sphere, x0, fprime=sphere_grad, disp=0)
    tnc.append((1, x0, res[0], sphere(res[0]), 1.0e-5))
    # The square with a cross term from the point (1, 1).
    x0 = np.array([1.0, 1.0])
    res = fmin_tnc(cross_quad, x0, fprime=cross_quad_grad, disp=0)
    tnc.append((8, x0, res[0], cross_quad(res[0]), 1.0e-5))

    text = ""
    text += "// Generated by test/gen_optimize_fmin.py. Do not edit by hand.\n"
    text += "\n"
    text += "// The pins of the tests of the four methods of minimization. They\n"
    text += "// come from SciPy 1.18.1. Each row holds a code for the problem,\n"
    text += "// the start point, the answer, the value of the function there and\n"
    text += "// the tolerance of the comparison.\n"
    text += "namespace optimize_fmin_pins;\n"
    text += "\n"
    text += rows("POWELL", powell)
    text += "\n"
    text += rows("COBYLA", cobyla)
    text += "\n"
    text += rows("SLSQP", slsqp)
    text += "\n"
    text += rows("TNC", tnc)
    with open(OUT, "w") as fh:
        fh.write(text)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
