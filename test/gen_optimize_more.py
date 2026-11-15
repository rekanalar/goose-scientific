#!/usr/bin/env python3
"""Write the pins of the tests of the plain calls of the optimization
and of the responses of the systems.

The script needs NumPy 2.5.4 and SciPy 1.18.1. It runs from the
maintainer. The user of the package installs nothing.

The file it writes holds numbers only. Every family of cases holds the
inputs and the expected results of one call. A ragged list is two
buffers: the values joined end to end, and the length of each part.
The values of the Hessian updates come from NumPy, because the rule of
the update holds no call of SciPy.
"""
import warnings

import numpy as np
import scipy.optimize as opt
import scipy.signal as sg

warnings.simplefilter("ignore")

OUT = "test/suite/optimize_more_pins.goose"


def emit(lines, name, values):
    lines.append("const %s: f64[>..] = [%s];" %
                 (name, ", ".join(repr(float(v)) for v in values)))


def ragged(lines, name, parts):
    """Write a ragged list as the values and the length of each part."""
    emit(lines, name + "_D", [v for p in parts for v in p])
    emit(lines, name + "_L", [len(p) for p in parts])


def rosen_cases():
    """The values, the gradient, the Hessian and the product of the
    function of Rosenbrock."""
    lines = []
    x0 = np.array([-1.2, 1.0])
    x1 = np.array([1.0, 2.0, 3.0])
    lines.append("// rosen: the points and the values.")
    ragged(lines, "ROSEN_X", [x0, x1])
    emit(lines, "ROSEN_F", [opt.rosen(x0), opt.rosen(x1)])
    lines.append("// rosen_der: the point and the gradient.")
    emit(lines, "RD_X", [1.0, 2.0, 3.0])
    emit(lines, "RD_G", opt.rosen_der(x1))
    lines.append("// rosen_hess: the point and the Hessian.")
    emit(lines, "RH_X", [0.1, 0.2, 0.3])
    emit(lines, "RH_H", opt.rosen_hess(np.array([0.1, 0.2, 0.3])).ravel())
    lines.append("// rosen_hess_prod: the point, the direction and the product.")
    emit(lines, "RHP_X", [1.0, 2.0, 3.0])
    emit(lines, "RHP_P", [0.5, -1.0, 2.0])
    emit(lines, "RHP_H", opt.rosen_hess_prod(x1, np.array([0.5, -1.0, 2.0])))
    return lines


def rosen_val(x):
    """The value of the function of Rosenbrock."""
    s = 0.0
    for i in range(len(x) - 1):
        d = x[i + 1] - x[i] * x[i]
        s += 100.0 * d * d + (1.0 - x[i]) ** 2
    return s


def rosen_grad(x):
    """The gradient of the function of Rosenbrock."""
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        d = x[i + 1] - x[i] * x[i]
        g[i] += -400.0 * x[i] * d - 2.0 * (1.0 - x[i])
        g[i + 1] += 200.0 * d
    return g


def check_grad_cases():
    """The distance between the gradient and its difference."""
    lines = ["// check_grad: the points and the distances."]
    xs = [np.array([0.5, -1.0]), np.array([1.5, -1.5]),
          np.array([0.1, 2.2, 4.3])]
    ragged(lines, "CG_X", xs)
    emit(lines, "CG_D", [opt.check_grad(rosen_val, rosen_grad, x) for x in xs])
    return lines


def fixed_point_cases():
    """The point that the function does not move."""
    lines = ["// fixed_point: the starts and the fixed points."]
    starts = [np.array([1.0, 2.0]), np.array([0.5, 1.0])]
    ends = [opt.fixed_point(np.sqrt, starts[0]),
            opt.fixed_point(lambda x: 1.0 / (1.0 + x), starts[1])]
    ragged(lines, "FP_X", starts)
    ragged(lines, "FP_P", ends)
    return lines


def line_search_cases():
    """The step of the search of the line."""
    lines = ["// line_search: the step of the two runs."]
    f = lambda x: (x[0] - 3.0) ** 2
    g = lambda x: np.array([2.0 * (x[0] - 3.0)])
    a1 = opt.line_search(f, g, np.array([0.0]), np.array([1.0]))[0]
    # The quartic x^4 - 2 x^2 at the point of one half.
    f2 = lambda x: x[0] ** 4 - 2.0 * x[0] ** 2
    g2 = lambda x: np.array([4.0 * x[0] ** 3 - 4.0 * x[0]])
    a2 = opt.line_search(f2, g2, np.array([0.5]), np.array([1.0]))[0]
    emit(lines, "LS_A", [a1, a2])
    return lines


def bracket_cases():
    """The three points that hold the minimum."""
    lines = ["// bracket: the starts and the points and values of the bracket."]
    cases = [
        (lambda x: (x - 1.0) ** 2, 0.1, 1.0),
        (lambda x: x * x * x - 2.0 * x - 2.0, -1.0, 0.0),
        (np.sin, 1.0, 4.0),
    ]
    pts = []
    vals = []
    for f, a, b in cases:
        xa, xb, xc, fa, fb, fc, _ = opt.bracket(f, a, b)
        pts.append([xa, xb, xc])
        vals.append([fa, fb, fc])
    ragged(lines, "BR_P", pts)
    ragged(lines, "BR_V", vals)
    return lines


def brute_cases():
    """The search on a grid."""
    lines = ["// brute: the bounds, the counts, the points and the values."]
    f = lambda v: (v[0] - 1.0) ** 2 + (v[1] + 2.0) ** 2
    p1, v1 = opt.brute(f, [(-2.0, 2.0), (-4.0, 0.0)], Ns=5, full_output=True)[:2]
    f2 = lambda v: (v[0] - 0.7) ** 2
    p2, v2 = opt.brute(f2, [(0.0, 1.0)], Ns=11, full_output=True)[:2]
    emit(lines, "BT_LO", [-2.0, -4.0, 0.0])
    emit(lines, "BT_HI", [2.0, 0.0, 1.0])
    emit(lines, "BT_NS", [5.0, 11.0])
    ragged(lines, "BT_P", [p1, p2])
    emit(lines, "BT_V", [v1, v2])
    return lines


def isotonic_cases():
    """The fit that must not fall."""
    lines = ["// isotonic_regression: the values, the weights of the cases,",
             "// the fitted values and the weights of the blocks."]
    ys = [np.array([3.0, 1.0, 2.0, 2.0, 5.0]),
          np.array([1.0, 0.0, 4.0, 3.0, 5.0, 6.0, 3.0, 2.0, 7.0])]
    ragged(lines, "ISO_Y", ys)
    r0 = opt.isotonic_regression(ys[0])
    r1 = opt.isotonic_regression(ys[1])
    ragged(lines, "ISO_X", [r0.x, r1.x])
    ragged(lines, "ISO_W", [r0.weights, r1.weights])
    return lines


def hessian_cases():
    """The updates of the Hessian. The rule holds no call of SciPy, so
    NumPy gives the values."""
    lines = ["// bfgs_hessian_update: the matrix, the steps and the update.",
             "// B holds the side, the row and the column."]
    B = np.diag([2.0, 1.0])
    s = np.array([1.0, -1.0])
    y = np.array([3.0, 1.0])
    Bs = B @ s
    ys = s @ y
    B2 = B + np.outer(y, y) / ys - np.outer(Bs, Bs) / (s @ Bs)
    emit(lines, "BFGS_B", B.ravel())
    emit(lines, "BFGS_S", s)
    emit(lines, "BFGS_Y", y)
    emit(lines, "BFGS_OUT", B2.ravel())
    lines.append("// sr1_hessian_update: the matrix, the steps, the skip")
    lines.append("// flag, the denominator and the update.")
    B3 = B + np.outer(y - Bs, y - Bs) / (s @ (y - Bs))
    emit(lines, "SR1_B", B.ravel())
    emit(lines, "SR1_S", s)
    emit(lines, "SR1_Y", y)
    emit(lines, "SR1_DEN", [s @ (y - Bs)])
    emit(lines, "SR1_OUT", B3.ravel())
    return lines


def impulse_step_cases():
    """The answer of one system to the hit and to the held input. The
    system is 1 / (s + 1)^2 in the state space form of the package."""
    lines = ["// impulse: the first values and the last time of the run of",
             "// the default span, and the values of a run of three points."]
    A = np.array([[-2.0, -1.0], [1.0, 0.0]])
    B = np.array([[1.0], [0.0]])
    C = np.array([[0.0, 1.0]])
    sys = sg.lti(A, B, C, 0.0)
    t, h = sg.impulse(sys)
    emit(lines, "IMP_T3", t[:3])
    emit(lines, "IMP_H3", h[:3])
    emit(lines, "IMP_TLAST", [t[-1]])
    emit(lines, "IMP_HN", [len(t)])
    _, h2 = sg.impulse(sys, T=np.array([0.0, 0.5, 1.0]))
    emit(lines, "IMP_HT", h2)
    lines.append("// step: the first values, the last value and the values")
    lines.append("// of a run of three points.")
    _, s2 = sg.step(sys)
    emit(lines, "STEP_S3", s2[:3])
    emit(lines, "STEP_SLAST", [s2[-1]])
    _, s3 = sg.step(sys, T=np.array([0.0, 0.5, 1.0]))
    emit(lines, "STEP_ST", s3)
    return lines


def lsim_cases():
    """The walk of an input through the system, with and without the
    interpolation, and from a start of two states."""
    lines = ["// lsim: the times, the input, the start, the outputs of the",
             "// interpolated run, of the held run and of the run from the",
             "// start, and the first states of the run from the start."]
    A = np.array([[-2.0, -1.0], [1.0, 0.0]])
    B = np.array([[1.0], [0.0]])
    C = np.array([[0.0, 1.0]])
    sys = sg.lti(A, B, C, 0.0)
    tt = np.linspace(0.0, 2.0, 10)
    u = np.cos(tt)
    T, y, x = sg.lsim(sys, u, tt)
    T, yz, xz = sg.lsim(sys, u, tt, interp=False)
    T, y0, x0 = sg.lsim(sys, u, tt, X0=np.array([1.0, 0.5]))
    emit(lines, "LSIM_T", tt)
    emit(lines, "LSIM_U", u)
    emit(lines, "LSIM_X0", [1.0, 0.5])
    emit(lines, "LSIM_Y", y)
    emit(lines, "LSIM_YZ", yz)
    emit(lines, "LSIM_Y0", y0)
    ragged(lines, "LSIM_XX", [x0[0, :], x0[1, :]])
    return lines


def freq_cases():
    """The answer of 1 / (s + 1)^2 over the frequencies, with and
    without the grid of the caller."""
    lines = ["// freqresp: the grid and the parts of the answer.",
             "// bode: the grid of the caller and the size and the phase,",
             "// and the first values of the run of the default grid."]
    w = np.array([0.1, 1.0, 10.0])
    _, h = sg.freqresp(([1.0], [1.0, 2.0, 1.0]), w=w)
    emit(lines, "FR_W", w)
    emit(lines, "FR_HR", h.real)
    emit(lines, "FR_HI", h.imag)
    _, m, p = sg.bode(([1.0], [1.0, 2.0, 1.0]), w=w)
    emit(lines, "BODE_M", m)
    emit(lines, "BODE_P", p)
    w2, m2, p2 = sg.bode(([1.0], [1.0, 2.0, 1.0]))
    emit(lines, "BODE_WFIRST", [w2[0], w2[-1]])
    emit(lines, "BODE_N", [len(w2)])
    emit(lines, "BODE_M2", m2[:2])
    emit(lines, "BODE_P2", p2[:2])
    return lines


def discrete_cases():
    """The answers of the discrete system 1 / (z + 1)^2 and of the
    sizes and the phases of the answers."""
    lines = ["// dstep and dimpulse: the samples and the values."]
    sysd = sg.dlti(np.array([1.0]), np.array([1.0, 2.0, 1.0]))
    t, y = sg.dstep(sysd, n=4)
    emit(lines, "DSTEP_T", t)
    emit(lines, "DSTEP_Y", np.squeeze(y))
    _, y2 = sg.dimpulse(sysd, n=4)
    emit(lines, "DIMP_Y", np.squeeze(y2))
    sysd2 = sg.dlti(np.array([1.0]), np.array([1.0, 2.0, 1.0]), dt=0.5)
    t3, _ = sg.dstep(sysd2, n=4)
    emit(lines, "DSTEP_T05", t3)
    lines.append("// dfreqresp: the frequencies and the parts of the answers")
    lines.append("// of the two systems.")
    _, h = sg.dfreqresp(sysd, w=np.array([0.5]))
    emit(lines, "DFR_WR", [0.5])
    emit(lines, "DFR_HR", h.real)
    emit(lines, "DFR_HI", h.imag)
    _, h2 = sg.dfreqresp(sg.dlti(np.array([1.0]), np.array([1.0, -0.5])),
                         w=np.array([0.0, 0.5, np.pi / 2]))
    ragged(lines, "DFR2_HR", [h2.real])
    ragged(lines, "DFR2_HI", [h2.imag])
    emit(lines, "DFR2_W", [0.0, 0.5, np.pi / 2])
    lines.append("// dbode: the size and the phase at the frequency one and")
    lines.append("// at the step of the sample of one half.")
    _, m, p = sg.dbode(sysd, w=np.array([1.0]))
    emit(lines, "DB_M", m)
    emit(lines, "DB_P", p)
    _, m2, p2 = sg.dbode(sysd2, w=np.array([1.0]))
    emit(lines, "DB_W05", [2.0])
    emit(lines, "DB_M05", m2)
    emit(lines, "DB_P05", p2)
    _, m3, p3 = sg.dbode(sysd, w=np.array([0.5, 1.0]))
    emit(lines, "DB_M2", m3)
    emit(lines, "DB_P2", p3)
    return lines


def state_cases():
    """The zeros, the poles and the gain of the identity system, and
    the counts of the checked forms."""
    lines = ["// ss2zpk: the zeros, the poles and the gain of the system with",
             "// the matrix of the identity of two states, a direct part of",
             "// one and no input and no output."]
    z, p, k = sg.ss2zpk(np.eye(2), np.zeros((2, 1)), np.zeros((1, 2)),
                        np.array([[1.0]]))
    emit(lines, "SS_Z", z.real)
    emit(lines, "SS_P", p.real)
    emit(lines, "SS_K", [k])
    lines.append("// abcd_normalize: the counts of the four parts.")
    a, b, c, d = sg.abcd_normalize(A=[[1, 2], [3, 4]], B=[[-1], [5]], D=2.5)
    emit(lines, "AB_N", [a.shape[0], a.shape[1], b.shape[1], c.shape[0], d.shape[1]])
    emit(lines, "AB_C", c.ravel())
    return lines


def main():
    lines = [
        "// Generated by test/gen_optimize_more.py. Do not edit by hand.",
        "",
        "// The pins of the tests of the plain calls of the optimization",
        "// and of the responses of the systems.",
        "// They come from SciPy 1.18.1 and NumPy 2.5.4.",
        "namespace optimize_more_pins;",
        "",
    ]
    parts = [
        rosen_cases, check_grad_cases, fixed_point_cases,
        line_search_cases, bracket_cases, brute_cases, isotonic_cases,
        hessian_cases, impulse_step_cases, lsim_cases, freq_cases,
        discrete_cases, state_cases,
    ]
    for part in parts:
        body = part()
        lines.extend(body)
        lines.append("")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
