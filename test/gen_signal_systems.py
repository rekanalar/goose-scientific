#!/usr/bin/env python3
"""Write the pins of the tests of the systems of a signal.

The script needs Python 3 with NumPy 2.5.4 and SciPy 1.18.1. It runs
from the maintainer. The user of the package installs nothing.

The file it writes holds numbers only. Every family of cases holds the
inputs and the expected results of one call. A ragged list is two
buffers: the values joined end to end, and the length of each part.
The roots of a filter come as the real parts and the imaginary parts.
The poles of a closed loop come in the order of SciPy.
"""
import warnings

import numpy as np
import scipy.signal as sg

warnings.simplefilter("ignore")

OUT = "test/suite/signal_systems_pins.goose"


def emit(lines, name, values):
    lines.append("const %s: f64[>..] = [%s];" %
                 (name, ", ".join(repr(float(v)) for v in values)))


def ragged(lines, name, parts):
    """Write a ragged list as the values and the length of each part."""
    emit(lines, name + "_D", [v for p in parts for v in p])
    emit(lines, name + "_L", [len(p) for p in parts])


def lti_cases():
    """The numerators and the denominators of a transfer function."""
    return [
        ([1.0, 3.0], [1.0, 2.0]),
        ([0.0, 3.0, 6.0], [2.0, -5.0, 4.0]),
        ([5.0], [1.0]),
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]),
        ([0.0, 1.0], [0.0, 0.0, 1.0]),
        ([1.0e-15, 1.0, 2.0], [1.0, 3.0, 2.0]),
        ([2.0, 0.0], [4.0, 0.0, 0.0]),
        ([1.0], [2.0, 3.0]),
    ]


def dlsim_tf_cases():
    """The discrete simulation of a transfer function."""
    return [
        ([1.0], [1.0, -1.0], 1.0, [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]),
        ([1.0], [1.0, -0.5, 0.25], 0.5, [1.0, 2.0, 3.0, 4.0, 5.0],
         [0.0, 0.5, 1.0, 1.5, 2.0]),
        ([1.0, 2.0], [1.0, -1.0], 0.1, [0.5, -0.5, 0.25, 0.75, 1.0],
         [0.0, 0.1, 0.2, 0.3, 0.4]),
        ([1.0], [1.0], 1.0, [1.0, 2.0, 3.0], [0.0, 1.0, 2.0]),
        ([1.0], [1.0, -1.0], 0.25, [1.0, 1.0, 1.0, 1.0, 1.0], []),
        ([0.0, 1.0], [1.0, -0.9], 1.0, [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]),
    ]


def dlsim_ss_cases():
    """The discrete simulation of a state space form."""
    return [
        ([[-1.0]], [[1.0]], [[1.0]], [[0.0]], 0.5,
         [0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.0, 1.5], []),
        ([[0.0, 1.0], [-2.0, -3.0]], [[0.0], [1.0]], [[1.0, 0.0]], [[0.0]], 0.1,
         [1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.1, 0.2, 0.3, 0.4], [0.5, -0.5]),
        ([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -2.0, -3.0]],
         [[0.0], [0.0], [1.0]], [[1.0, 0.0, 0.0]], [[0.0]], 0.2,
         [1.0, 2.0, 3.0], [], [0.25, -0.25, 0.5]),
        ([[-0.5, 0.2], [0.1, -0.3]], [[1.0], [0.0]], [[0.0, 1.0]], [[0.25]], 1.0,
         [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0], []),
    ]


def place_cases():
    """The placement of the poles: the matrices, the poles and the
    method."""
    rng = np.random.default_rng(3)
    a3 = rng.normal(size=(3, 3))
    b32 = rng.normal(size=(3, 2))
    knv_a = np.array([[1.380, -0.2077, 6.715, -5.676],
                      [-0.5814, -4.290, 0.0, 0.6750],
                      [1.067, 4.273, -6.654, 5.893],
                      [0.0480, 4.273, 1.343, -2.104]])
    knv_b = np.array([[0.0, 5.679], [1.136, 1.136], [0.0, 0.0], [-3.146, 0.0]])
    doc_a = np.array([[0.0, 7.0 / 3.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 7.0 / 9.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
    doc_b = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return [
        (np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2), [-1.0, -2.0], "YT"),
        (np.array([[0.0, 1.0], [-4.0, -4.0]]), np.eye(2),
         [-1.0 + 1.0j, -1.0 - 1.0j], "YT"),
        (np.array([[1.0, 2.0], [3.0, 4.0]]),
         np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]), [-1.0, -2.0], "YT"),
        (np.array([[-1.0, 1.0], [0.0, -2.0]]), np.array([[0.0], [1.0]]),
         [-0.5, -0.8], "YT"),
        (np.array([[-1.0, 1.0], [0.0, -2.0]]), np.array([[0.0], [1.0]]),
         [-1.0 + 0.5j, -1.0 - 0.5j], "YT"),
        (a3, b32, [-0.5, -0.7, -0.9], "YT"),
        (a3, np.array([[0.5], [1.0], [0.25]]), [-0.5, -0.7, -0.9], "YT"),
        (knv_a, knv_b, [-0.2, -0.5, -5.0566, -8.6659], "YT"),
        (knv_a, knv_b, [-0.2, -0.5, -5.0566, -8.6659], "KNV0"),
        (doc_a, doc_b, np.array([-3.0, -1.0, -2.0 - 1.0j, -2.0 + 1.0j]) / 3.0,
         "YT"),
    ]


def sos_cases():
    """The cascades of second order sections."""
    return [
        sg.butter(4, 0.3, output="sos"),
        sg.ellip(3, 1.0, 50.0, 0.2, output="sos"),
        sg.cheby1(4, 1.0, 0.4, output="sos"),
        sg.bessel(3, 0.3, output="sos"),
        np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 1.0, -0.5, 0.25]]),
        np.array([[1.0, 2.0, 0.0, 1.0, 0.0, 0.0]]),
        np.array([[2.0, -1.0, 0.5, 1.0, -0.9, 0.2],
                  [0.5, 0.0, 0.0, 1.0, 0.3, 0.0]]),
    ]


def firls_cases():
    """The design of a filter by the least squares of the error."""
    return [
        (11, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [], 2.0),
        (21, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
         [], 2.0),
        (15, [0.1, 0.3, 0.5, 0.7], [1.0, 0.0, 0.0, 1.0], [1.0, 2.0], 2.0),
        (73, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
         [], 10.0),
        (9, [0.0, 0.5], [1.0, 1.0], [], 2.0),
        (31, [0.0, 0.25, 0.35, 0.65, 0.75, 1.0],
         [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 3.0], 2.0),
    ]


def impulse_cases():
    """The unit impulse: the shape, the place and the values."""
    return [
        ([8], [], 0),
        ([7], [2], 0),
        ([3, 3], [], 1),
        ([4, 4], [2], 0),
        ([2, 3], [1, 2], 0),
        ([5, 6, 7], [], 1),
        ([3], [0], 0),
    ]


def main():
    lines = ["// Generated by test/gen_signal_systems.py. Do not edit by hand.",
             "",
             "// The pins of the tests of the systems of a signal.",
             "// They come from SciPy 1.18.1.",
             "namespace signal_systems_pins;",
             ""]

    # The transfer function of a system.
    lc = lti_cases()
    ragged(lines, "LT_NUM", [c[0] for c in lc])
    ragged(lines, "LT_DEN", [c[1] for c in lc])
    enums, edens = [], []
    for num, den in lc:
        sys = sg.lti(num, den)
        enums.append([float(v) for v in np.atleast_1d(sys.num)])
        edens.append([float(v) for v in np.atleast_1d(sys.den)])
    ragged(lines, "LT_ENUM", enums)
    ragged(lines, "LT_EDEN", edens)

    # The discrete simulation of a transfer function.
    tc = dlsim_tf_cases()
    ragged(lines, "TF_NUM", [c[0] for c in tc])
    ragged(lines, "TF_DEN", [c[1] for c in tc])
    emit(lines, "TF_DT", [c[2] for c in tc])
    ragged(lines, "TF_U", [c[3] for c in tc])
    ragged(lines, "TF_T", [c[4] for c in tc])
    ttout, tyout = [], []
    for num, den, dt, u, t in tc:
        if t:
            tout, yout = sg.dlsim((num, den, dt), u, t=t)
        else:
            tout, yout = sg.dlsim((num, den, dt), u)
        ttout.append([float(v) for v in np.atleast_1d(tout)])
        tyout.append([float(v) for v in np.ravel(yout)])
    ragged(lines, "TF_TOUT", ttout)
    ragged(lines, "TF_YOUT", tyout)

    # The discrete simulation of a state space form.
    sc = dlsim_ss_cases()
    emit(lines, "SS_N", [np.array(c[0]).shape[0] for c in sc])
    ragged(lines, "SS_A", [list(np.ravel(c[0])) for c in sc])
    ragged(lines, "SS_B", [list(np.ravel(c[1])) for c in sc])
    ragged(lines, "SS_C", [list(np.ravel(c[2])) for c in sc])
    emit(lines, "SS_D", [float(np.ravel(c[3])[0]) for c in sc])
    emit(lines, "SS_DT", [c[4] for c in sc])
    ragged(lines, "SS_U", [c[5] for c in sc])
    ragged(lines, "SS_T", [c[6] for c in sc])
    ragged(lines, "SS_X0", [c[7] for c in sc])
    stout, syout, sxout = [], [], []
    for a, b, c, d, dt, u, t, x0 in sc:
        args = (a, b, c, d, dt)
        kw = {}
        if t:
            kw["t"] = t
        if x0:
            kw["x0"] = x0
        tout, yout, xout = sg.dlsim(args, u, **kw)
        stout.append([float(v) for v in np.atleast_1d(tout)])
        syout.append([float(v) for v in np.ravel(yout)])
        sxout.append([float(v) for v in np.ravel(xout)])
    ragged(lines, "SS_TOUT", stout)
    ragged(lines, "SS_YOUT", syout)
    ragged(lines, "SS_XOUT", sxout)

    # The placement of the poles.
    pc = place_cases()
    emit(lines, "PP_N", [c[0].shape[0] for c in pc])
    ragged(lines, "PP_A", [list(np.ravel(c[0])) for c in pc])
    ragged(lines, "PP_B", [list(np.ravel(c[1])) for c in pc])
    emit(lines, "PP_M", [c[1].shape[1] for c in pc])
    ragged(lines, "PP_PR", [[p.real for p in c[2]] for c in pc])
    ragged(lines, "PP_PI", [[p.imag for p in c[2]] for c in pc])
    emit(lines, "PP_METHOD", [1.0 if c[3] == "KNV0" else 0.0 for c in pc])
    cpr, cpi, rpr, rpi, gains, fast = [], [], [], [], [], []
    for a, b, poles, method in pc:
        ref = sg.place_poles(a, b, poles, method=method)
        cpr.append([float(p.real) for p in ref.computed_poles])
        cpi.append([float(p.imag) for p in ref.computed_poles])
        rpr.append([float(p.real) for p in ref.requested_poles])
        rpi.append([float(p.imag) for p in ref.requested_poles])
        is_fast = 1.0 if np.linalg.matrix_rank(b) == a.shape[0] else 0.0
        fast.append(is_fast)
        gains.append(list(np.ravel(ref.gain_matrix)))
    ragged(lines, "PP_CPR", cpr)
    ragged(lines, "PP_CPI", cpi)
    ragged(lines, "PP_RPR", rpr)
    ragged(lines, "PP_RPI", rpi)
    emit(lines, "PP_FAST", fast)
    ragged(lines, "PP_GAIN", gains)

    # The forms of a cascade of second order sections.
    oc = sos_cases()
    ragged(lines, "SO_SOS", [list(np.ravel(c)) for c in oc])
    ob, oa, ozr, ozi, opr, opi, ok = [], [], [], [], [], [], []
    for sos in oc:
        b, a = sg.sos2tf(sos)
        z, p, k = sg.sos2zpk(sos)
        ob.append([float(v) for v in np.atleast_1d(b)])
        oa.append([float(v) for v in np.atleast_1d(a)])
        ozr.append([float(v.real) for v in z])
        ozi.append([float(v.imag) for v in z])
        opr.append([float(v.real) for v in p])
        opi.append([float(v.imag) for v in p])
        ok.append(float(k))
    ragged(lines, "SO_B", ob)
    ragged(lines, "SO_A", oa)
    ragged(lines, "SO_ZR", ozr)
    ragged(lines, "SO_ZI", ozi)
    ragged(lines, "SO_PR", opr)
    ragged(lines, "SO_PI", opi)
    emit(lines, "SO_K", ok)

    # The design by the least squares of the error.
    fc = firls_cases()
    emit(lines, "FI_TAPS", [c[0] for c in fc])
    ragged(lines, "FI_BANDS", [c[1] for c in fc])
    ragged(lines, "FI_DES", [c[2] for c in fc])
    ragged(lines, "FI_W", [c[3] for c in fc])
    emit(lines, "FI_FS", [c[4] for c in fc])
    fco = []
    for numtaps, bands, desired, weight, fs in fc:
        if weight:
            coeffs = sg.firls(numtaps, bands, desired, weight=weight, fs=fs)
        else:
            coeffs = sg.firls(numtaps, bands, desired, fs=fs)
        fco.append([float(v) for v in np.atleast_1d(coeffs)])
    ragged(lines, "FI_COEF", fco)

    # The unit impulse.
    ic = impulse_cases()
    ragged(lines, "UI_SHAPE", [[float(v) for v in c[0]] for c in ic])
    ragged(lines, "UI_IDX", [[float(v) for v in c[1]] for c in ic])
    emit(lines, "UI_MID", [c[2] for c in ic])
    uvals = []
    for shape, idx, mid in ic:
        if mid:
            out = sg.unit_impulse(tuple(int(v) for v in shape), "mid")
        elif len(idx) == 0:
            out = sg.unit_impulse(tuple(int(v) for v in shape))
        elif len(idx) == 1:
            out = sg.unit_impulse(tuple(int(v) for v in shape), int(idx[0]))
        else:
            out = sg.unit_impulse(tuple(int(v) for v in shape),
                                  tuple(int(v) for v in idx))
        uvals.append(list(np.ravel(out)))
    ragged(lines, "UI_VAL", uvals)

    open(OUT, "w").write("\n".join(lines) + "\n")
    print("wrote %s" % OUT)
    print("lti %d, dlsim tf %d, dlsim ss %d, place %d, sos %d, firls %d, "
          "impulse %d" % (len(lc), len(tc), len(sc), len(pc), len(oc),
                          len(fc), len(ic)))


if __name__ == "__main__":
    main()
