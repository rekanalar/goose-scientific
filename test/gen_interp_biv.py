#!/usr/bin/env python3
"""Write the pins of the bivariate splines and the smoothing splines.

The pins go to test/suite/interp_biv_pins.goose. They hold numbers
only. The test reads the numbers and compares them with the values of
the package.

Needs: Python 3, NumPy 2.5.4, SciPy 1.18.1.
Run:  /home/ubuntu/works/goose/tmp/venv/bin/python -I test/gen_interp_biv.py
"""

import numpy as np
import scipy.interpolate as si
import scipy.interpolate._bsplines as bsmod


def num(v):
    return repr(float(v))


def block(name, vals):
    body = ", ".join(num(v) for v in vals)
    return f"const {name}: f64[>..] = [{body}];"


def lsq_case():
    # The scattered least squares fit with given knots and weights.
    n = 12
    m = n * n
    ii = np.repeat(np.arange(n), n)
    jj = np.tile(np.arange(n), n)
    x = 0.05 + 0.9 * ii / (n - 1) + 0.01 * np.sin(3 * ii + 5 * jj)
    y = 0.1 + 0.8 * jj / (n - 1) + 0.01 * np.cos(2 * ii + 7 * jj)
    z = np.sin(3.0 * x + 2.0 * y)
    w = 1.0 + 0.1 * np.sin(np.arange(m))
    tx = np.linspace(0.1, 0.9, 6)
    ty = np.linspace(0.15, 0.85, 5)
    sp = si.LSQBivariateSpline(x, y, z, tx, ty, w=w, kx=3, ky=3)
    t1, t2, c = sp.tck
    gx = np.linspace(0.1, 0.9, 4)
    gy = np.linspace(0.15, 0.85, 5)
    grid = sp(gx, gy)
    pts = sp.ev(np.array([0.2, 0.5, 0.8]), np.array([0.3, 0.5, 0.7]))
    dxg = sp(gx, gy, dx=1)
    integ = sp.integral(0.15, 0.7, 0.2, 0.6)
    out = []
    out.append(block("LSQ_X", x))
    out.append(block("LSQ_Y", y))
    out.append(block("LSQ_Z", z))
    out.append(block("LSQ_W", w))
    out.append(block("LSQ_TX", tx))
    out.append(block("LSQ_TY", ty))
    out.append(block("LSQ_T1", t1))
    out.append(block("LSQ_T2", t2))
    out.append(block("LSQ_C", c))
    out.append(block("LSQ_GX", gx))
    out.append(block("LSQ_GY", gy))
    out.append(block("LSQ_GRID", grid.ravel()))
    out.append(block("LSQ_PTS", pts))
    out.append(block("LSQ_DX", dxg.ravel()))
    out.append(block("LSQ_INT", [integ]))
    # A second fit of degree one with no weights.
    sp2 = si.LSQBivariateSpline(x, y, z, tx, ty, kx=1, ky=2)
    t1b, t2b, cb = sp2.tck
    out.append(block("LSQ2_T1", t1b))
    out.append(block("LSQ2_T2", t2b))
    out.append(block("LSQ2_C", cb))
    out.append(block("LSQ2_GRID", sp2(gx, gy).ravel()))
    return out


def smooth_case():
    # The smoothing spline with a factor and with the factor of GCV.
    n = 30
    x = np.linspace(0.0, 3.0, n)
    y = np.sin(x) + 0.2 * np.cos(2.0 * x)
    lam = 0.5
    spl = si.make_smoothing_spline(x, y, lam=lam)
    out = []
    out.append(block("SM_X", x))
    out.append(block("SM_Y", y))
    out.append(block("SM_LAM", [lam]))
    out.append(block("SM_T", spl.t))
    out.append(block("SM_C", spl.c))
    out.append(block("SM_V", spl(np.linspace(0.2, 2.8, 7))))
    # The weights form.
    w = 1.0 + 0.5 * np.sin(np.arange(n))
    splw = si.make_smoothing_spline(x, y, w=w, lam=lam)
    out.append(block("SMW_W", w))
    out.append(block("SMW_C", splw.c))
    # The GCV form.
    yg = np.sin(x) + 0.3 * np.sin(4.0 * x)
    splg = si.make_smoothing_spline(x, yg)
    t = np.r_[[x[0]] * 3, x, [x[-1]] * 3]
    w = np.ones(n)
    wE = np.zeros((5, n))
    wE[2:, 0] = bsmod._coeff_of_divided_diff(x[:3]) / w[:3]
    wE[1:, 1] = bsmod._coeff_of_divided_diff(x[:4]) / w[:4]
    for j in range(2, n - 2):
        wE[:, j] = (x[j + 2] - x[j - 2]) * bsmod._coeff_of_divided_diff(x[j - 2:j + 3]) / w[j - 2:j + 3]
    wE[:-1, -2] = -bsmod._coeff_of_divided_diff(x[-4:]) / w[-4:]
    wE[:-2, -1] = bsmod._coeff_of_divided_diff(x[-3:]) / w[-3:]
    wE *= 6
    from scipy.interpolate import BSpline
    Xs = BSpline.design_matrix(x, t, 3)
    X = np.zeros((5, n))
    for i in range(1, 4):
        X[i, 2: -2] = Xs[i: i - 4, 3: -3][np.diag_indices(n - 4)]
    X[1, 1] = Xs[0, 0]
    X[2, :2] = ((x[2] + x[1] - 2 * x[0]) * Xs[0, 0], Xs[1, 1] + Xs[1, 2])
    X[3, :2] = ((x[2] - x[0]) * Xs[1, 1], Xs[2, 2])
    X[1, -2:] = (Xs[-3, -3], (x[-1] - x[-3]) * Xs[-2, -2])
    X[2, -2:] = (Xs[-2, -3] + Xs[-2, -2], (2 * x[-1] - x[-2] - x[-3]) * Xs[-1, -1])
    X[3, -2] = Xs[-1, -1]
    lamg = bsmod._compute_optimal_gcv_parameter(X, wE, yg, w)
    out.append(block("SMG_Y", yg))
    out.append(block("SMG_LAM", [lamg]))
    out.append(block("SMG_C", splg.c))
    out.append(block("SMG_V", splg(np.linspace(0.2, 2.8, 7))))
    return out


def knots_case():
    # The knots that the smoothing fit of Dierckx accepts.
    out = []
    x = np.linspace(-2.0, 2.0, 40)
    y = np.exp(-x * x) + 0.15 * np.sin(3.0 * x)
    cases = [(3, 0.5), (1, 1.0), (5, 0.05), (3, 20.0)]
    for i, (k, s) in enumerate(cases):
        kn = list(si.generate_knots(x, y, k=k, s=s))[-1]
        out.append(block(f"GK{i}_T", kn))
    out.append(block("GK_X", x))
    out.append(block("GK_Y", y))
    out.append(block("GK_CASES", [v for c in cases for v in c]))
    return out


def bpoly_case():
    # The piecewise polynomial in the basis of Bernstein.
    xb = np.array([0.0, 1.0, 2.5, 4.0])
    m = len(xb) - 1
    k = 3
    c = np.zeros((k + 1, m))
    for i in range(m):
        for a in range(k + 1):
            c[a, i] = np.sin(0.7 * i + 1.1 * a) + 0.2 * a - 0.3 * i
    bp = si.BPoly(c, xb)
    q = np.array([-0.5, 0.0, 0.5, 1.2, 2.5, 3.0, 3.9, 4.5])
    out = []
    out.append(block("BP_X", xb))
    out.append(block("BP_C", c.ravel(order="F")))
    out.append(block("BP_Q", q))
    out.append(block("BP_V", bp(q)))
    out.append(block("BP_D1", bp(q, nu=1)))
    out.append(block("BP_D2", bp(q, nu=2)))
    out.append(block("BP_INT", [bp.integrate(0.5, 3.2)]))
    d1 = bp.derivative()
    out.append(block("BP_DC", d1.c.ravel(order="F")))
    anti = bp.antiderivative()
    out.append(block("BP_AC", anti.c.ravel(order="F")))
    return out


def nd_case():
    # The spline and the polynomial of two axes over a grid.
    t1 = np.array([0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0])
    t2 = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 0.9, 1.2, 1.2, 1.2, 1.2])
    k = (3, 3)
    n1, n2 = len(t1) - 4, len(t2) - 4
    rng = np.arange(n1 * n2, dtype=float)
    c = (np.sin(rng * 0.7) + 0.05 * rng).reshape(n1, n2)
    spl = si.NdBSpline((t1, t2), c, k)
    pts = np.array([[0.2, 0.3], [0.5, 0.7], [0.9, 1.0], [0.05, 1.15]])
    out = []
    out.append(block("ND_T1", t1))
    out.append(block("ND_T2", t2))
    out.append(block("ND_C", c.ravel()))
    out.append(block("ND_PTS", pts.ravel()))
    out.append(block("ND_V", spl(pts)))
    out.append(block("ND_D", spl(pts, nu=(1, 0))))
    out.append(block("ND_D2", spl(pts, nu=(1, 2))))
    # The piecewise polynomial of two axes.
    x1 = np.array([0.0, 0.5, 1.0])
    x2 = np.array([0.0, 0.4, 0.8, 1.2])
    m1, m2 = len(x1) - 1, len(x2) - 1
    k1, k2 = 2, 1
    cp = np.zeros((k1 + 1, k2 + 1, m1, m2))
    for a in range(k1 + 1):
        for b in range(k2 + 1):
            for i in range(m1):
                for j in range(m2):
                    cp[a, b, i, j] = 0.3 * a - 0.2 * b + 0.1 * i + 0.05 * j
    pp = si.NdPPoly(cp, (x1, x2))
    q = np.array([[0.1, 0.2], [0.7, 0.9], [0.95, 1.1]])
    out.append(block("NP_X1", x1))
    out.append(block("NP_X2", x2))
    out.append(block("NP_C", cp.ravel()))
    out.append(block("NP_Q", q.ravel()))
    out.append(block("NP_V", pp(q)))
    out.append(block("NP_D", pp(q, nu=(1, 1))))
    return out


def main():
    parts = []
    parts.append("// Generated by test/gen_interp_biv.py. Do not edit by hand.")
    parts.append("")
    parts.append("// The pins of the tests of the bivariate splines, the")
    parts.append("// smoothing splines and the splines of many dimensions.")
    parts.append("// They come from SciPy 1.18.1 and NumPy 2.5.4.")
    parts.append("namespace interp_biv_pins;")
    parts.append("")
    for fn in (lsq_case, smooth_case, knots_case, bpoly_case, nd_case):
        parts.extend(fn())
        parts.append("")
    with open("test/suite/interp_biv_pins.goose", "w") as f:
        f.write("\n".join(parts))
    print("wrote test/suite/interp_biv_pins.goose")


if __name__ == "__main__":
    main()
