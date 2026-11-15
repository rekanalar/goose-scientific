#!/usr/bin/env python3
"""Write the pins of the tests of the spline filters, the fast
convolution and the short time transform.

The script needs Python 3 with NumPy 2.5.4 and SciPy 1.18.1. It runs
from the directory of the package. Run it with:

  /path/to/venv/bin/python -I test/gen_signal_spline.py

The file it writes holds numbers only. The user of the package
installs nothing. Every family holds the inputs and the expected
results of one call. A ragged list is two buffers: the values joined
end to end, and the length of each part. A code names a mode:
0 full, 1 same, 2 valid for the overlap; 0 constant, 1 symmetric,
2 reflect, 3 edge, 4 wrap, 5 smooth, 6 antisymmetric, 7 antireflect,
8 line for the upsample filter; 0 onesided, 1 twosided, 2 centered
for the transform, and 0 none, 1 spectrum, 2 psd for its scale. The
phase shift of minus one means that SciPy applies no shift.
"""
import math
import warnings

import numpy as np
import scipy.signal as sg
from scipy.signal._spline_filters import compute_root_from_lambda

warnings.simplefilter("ignore")

OUT = "test/suite/signal_spline_pins.goose"


def emit(lines, name, values):
    lines.append("const %s: f64[>..] = [%s];" %
                 (name, ", ".join(repr(float(v)) for v in values)))


def ragged_out(lines, name, parts):
    emit(lines, name + "_D", [v for p in parts for v in p])
    emit(lines, name + "_L", [len(p) for p in parts])


def sig(n, seed):
    r = np.random.default_rng(seed)
    return r.standard_normal(n)


def spline_cases():
    cs_in, cs_lamb = [], []
    cs_out = []
    qs_in = []
    qs_out = []
    r = np.random.default_rng(101)
    for n, lamb in [(20, 0.0), (1, 0.0), (24, 2.0), (2, 0.0), (30, 1.0)]:
        x = r.standard_normal(n)
        cs_in.append(x)
        cs_lamb.append(lamb)
        cs_out.append(sg.cspline1d(x, lamb))
    for n in [20, 1, 2]:
        x = r.standard_normal(n)
        qs_in.append(x)
        qs_out.append(sg.qspline1d(x))
    return cs_in, cs_lamb, cs_out, qs_in, qs_out


def eval_cases(cs_in, qs_in):
    """The places, the steps and the origins of the two eval calls."""
    cases = []
    r = np.random.default_rng(102)
    new_a = [-3.5, -0.5, 0.0, 4.2, 9.0, 19.0, 20.5, 25.0]
    new_b = list(r.uniform(-2.0, 24.0, 12))
    for cj, quad in [(cs_in[0], 0), (cs_in[0], 0), (qs_in[0], 1), (qs_in[0], 1)]:
        cases.append((cj, new_a, 1.0, 0.0, quad))
        cases.append((cj, new_b, 0.5, 2.0, quad))
    return cases


def cspline2d_cases():
    r = np.random.default_rng(103)
    cases = []
    for rows, cols, lamb in [(12, 13, 0.0), (14, 16, 0.0), (20, 24, 0.0), (20, 24, 0.1)]:
        img = r.standard_normal((rows, cols))
        cases.append((img, lamb))
    return cases


def qspline2d_cases():
    r = np.random.default_rng(104)
    cases = []
    for rows, cols in [(9, 9), (10, 12)]:
        cases.append(r.standard_normal((rows, cols)))
    return cases


def symiir1_cases():
    r = np.random.default_rng(105)
    rt = math.sqrt(3.0)
    cases = []
    for c0, z1 in [(6.0 - 2.0 * rt, -2.0 + rt), (0.5, 0.3), (0.25, 0.5),
                   (-1.0, -0.5)]:
        cases.append((r.standard_normal(50), c0, z1))
    return cases


def symiir2_cases():
    r = np.random.default_rng(106)
    rr, om = compute_root_from_lambda(0.1)
    cases = []
    for r0, o0 in [(0.3, math.pi / 3.0), (0.5, 2.0), (0.4, 0.05), (rr, om)]:
        cases.append((r.standard_normal(40), r0, o0))
    return cases


def spline_filter_cases():
    r = np.random.default_rng(107)
    cases = []
    for rows, cols, lm in [(12, 13, 0.1), (12, 13, 0.0), (20, 24, 0.1), (30, 40, 5.0)]:
        img = r.standard_normal((rows, cols))
        cases.append((img, lm))
    return cases


def oa_cases():
    r = np.random.default_rng(108)
    cases = []
    a = r.standard_normal(100)
    b = r.standard_normal(7)
    for mode in ["full", "same", "valid"]:
        cases.append((a, b, mode))
    c = r.standard_normal(37)
    cases.append((c, r.standard_normal(37), "full"))
    cases.append((r.standard_normal(10), r.standard_normal(20), "valid"))
    cases.append((r.standard_normal(200), r.standard_normal(5), "same"))
    cases.append((r.standard_normal(33), r.standard_normal(64), "full"))
    return cases


UP_MODES = ["constant", "symmetric", "reflect", "edge", "wrap", "smooth",
            "antisymmetric", "antireflect", "line"]


def upfirdn_cases():
    r = np.random.default_rng(109)
    cases = []
    cases.append(([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1, 1, "constant", 0.0))
    cases.append((r.standard_normal(7), r.standard_normal(11), 3, 2,
                  "symmetric", 0.0))
    cases.append((r.standard_normal(5), r.standard_normal(13), 2, 3,
                  "reflect", 0.0))
    cases.append((r.standard_normal(6), r.standard_normal(9), 1, 4,
                  "constant", 0.0))
    cases.append((r.standard_normal(4), r.standard_normal(8), 2, 2,
                  "edge", 0.0))
    cases.append((r.standard_normal(4), r.standard_normal(8), 2, 2,
                  "constant", 0.5))
    cases.append((r.standard_normal(5), r.standard_normal(10), 3, 1,
                  "smooth", 0.0))
    cases.append((r.standard_normal(5), r.standard_normal(10), 3, 1,
                  "line", 0.0))
    cases.append((r.standard_normal(5), r.standard_normal(10), 2, 2,
                  "wrap", 0.0))
    cases.append((r.standard_normal(5), r.standard_normal(10), 2, 2,
                  "antisymmetric", 0.0))
    cases.append((r.standard_normal(5), r.standard_normal(10), 2, 2,
                  "antireflect", 0.0))
    cases.append((r.standard_normal(3), r.standard_normal(12), 5, 7,
                  "symmetric", 0.0))
    return cases


def stft_cases():
    r = np.random.default_rng(110)
    win = sg.windows.hann(24, sym=False).astype(float)
    x = r.standard_normal(120)
    cases = [
        (win, 6, 10.0, "onesided", "", 0, 0.0),
        (win, 6, 10.0, "onesided", "psd", 0, 0.0),
        (win, 6, 10.0, "onesided", "magnitude", 0, 0.0),
        (win, 6, 10.0, "onesided", "", 32, 0.0),
        (win, 6, 10.0, "twosided", "", 0, 0.0),
        (win, 6, 10.0, "centered", "", 0, 0.0),
        (win, 6, 10.0, "onesided", "", 0, 4.0),
        (win, 6, 10.0, "onesided", "", 0, -1.0),
    ]
    return [(w, h, f, m, s, mf, p, x) for (w, h, f, m, s, mf, p) in cases]


MODE_OA = {"full": 0.0, "same": 1.0, "valid": 2.0}
MODE_UP = {m: float(i) for i, m in enumerate(UP_MODES)}
MODE_FM = {"onesided": 0.0, "twosided": 1.0, "centered": 2.0}
MODE_SC = {"": 0.0, "spectrum": 1.0, "magnitude": 1.0, "psd": 2.0}


def main():
    lines = []
    lines.append("// Generated by test/gen_signal_spline.py. Do not edit by hand.")
    lines.append("")
    lines.append("// The pins of the tests of the spline filters, the fast")
    lines.append("// convolution and the short time transform.")
    lines.append("// They come from SciPy 1.18.1.")
    lines.append("namespace signal_spline_pins;")
    lines.append("")

    cs_in, cs_lamb, cs_out, qs_in, qs_out = spline_cases()
    ragged_out(lines, "CS1_IN", cs_in)
    emit(lines, "CS1_L", [len(p) for p in cs_in])
    emit(lines, "CS1_LAMB", cs_lamb)
    ragged_out(lines, "CS1_OUT", cs_out)
    ragged_out(lines, "QS1_IN", qs_in)
    ragged_out(lines, "QS1_OUT", qs_out)

    cases = eval_cases(cs_in, qs_in)
    ragged_out(lines, "EV_C", [c[0] for c in cases])
    ragged_out(lines, "EV_N", [c[1] for c in cases])
    emit(lines, "EV_DX", [c[2] for c in cases])
    emit(lines, "EV_X0", [c[3] for c in cases])
    emit(lines, "EV_Q", [c[4] for c in cases])
    ragged_out(lines, "EV_OUT", [sg.cspline1d_eval(c[0], c[1], dx=c[2], x0=c[3])
                                 if c[4] == 0 else
                                 sg.qspline1d_eval(c[0], c[1], dx=c[2], x0=c[3])
                                 for c in cases])

    cases = cspline2d_cases()
    emit(lines, "CS2_SH", [v for c in cases for v in c[0].shape])
    emit(lines, "CS2_LAMB", [c[1] for c in cases])
    ragged_out(lines, "CS2_IN", [c[0].ravel() for c in cases])
    ragged_out(lines, "CS2_OUT", [sg.cspline2d(c[0], c[1]).ravel() for c in cases])

    cases = qspline2d_cases()
    emit(lines, "QS2_SH", [v for c in cases for v in c.shape])
    ragged_out(lines, "QS2_IN", [c.ravel() for c in cases])
    ragged_out(lines, "QS2_OUT", [sg.qspline2d(c).ravel() for c in cases])

    cases = symiir1_cases()
    ragged_out(lines, "SI1_IN", [c[0] for c in cases])
    emit(lines, "SI1_C0", [c[1] for c in cases])
    emit(lines, "SI1_Z1", [c[2] for c in cases])
    ragged_out(lines, "SI1_OUT", [sg.symiirorder1(c[0], c[1], c[2])
                                  for c in cases])

    cases = symiir2_cases()
    ragged_out(lines, "SI2_IN", [c[0] for c in cases])
    emit(lines, "SI2_R", [c[1] for c in cases])
    emit(lines, "SI2_OM", [c[2] for c in cases])
    ragged_out(lines, "SI2_OUT", [sg.symiirorder2(c[0], c[1], c[2])
                                  for c in cases])

    cases = spline_filter_cases()
    emit(lines, "SF_SH", [v for c in cases for v in c[0].shape])
    emit(lines, "SF_LM", [c[1] for c in cases])
    ragged_out(lines, "SF_IN", [c[0].ravel() for c in cases])
    ragged_out(lines, "SF_OUT", [sg.spline_filter(c[0], lmbda=c[1]).ravel()
                                 for c in cases])

    cases = oa_cases()
    ragged_out(lines, "OA_A", [c[0] for c in cases])
    ragged_out(lines, "OA_B", [c[1] for c in cases])
    emit(lines, "OA_M", [MODE_OA[c[2]] for c in cases])
    ragged_out(lines, "OA_OUT", [sg.oaconvolve(c[0], c[1], mode=c[2])
                                 for c in cases])

    cases = upfirdn_cases()
    ragged_out(lines, "UP_H", [c[0] for c in cases])
    ragged_out(lines, "UP_X", [c[1] for c in cases])
    emit(lines, "UP_UD", [v for c in cases for v in c[2:4]])
    emit(lines, "UP_M", [MODE_UP[c[4]] for c in cases])
    emit(lines, "UP_C", [c[5] for c in cases])
    ragged_out(lines, "UP_OUT", [sg.upfirdn(c[0], c[1], up=c[2], down=c[3],
                                            mode=c[4], cval=c[5])
                                 for c in cases])

    cases = stft_cases()
    ragged_out(lines, "ST_WIN", [c[0] for c in cases])
    emit(lines, "ST_WH", [v for c in cases for v in
                          [c[1], c[2], MODE_FM[c[3]], MODE_SC[c[4]],
                           c[5], c[6]]])
    ragged_out(lines, "ST_X", [c[7] for c in cases])
    re_parts = []
    im_parts = []
    out_parts = []
    for c in cases:
        ref = sg.ShortTimeFFT(c[0], hop=c[1], fs=c[2], fft_mode=c[3],
                              mfft=c[5] if c[5] > 0 else None,
                              scale_to=(c[4] if c[4] else None),
                              phase_shift=(None if c[6] < 0 else int(c[6])))
        s = ref.stft(c[7])
        re_parts.append(s.real.ravel())
        im_parts.append(s.imag.ravel())
        out_parts.append(ref.istft(s))
    ragged_out(lines, "ST_RE", re_parts)
    ragged_out(lines, "ST_IM", im_parts)
    ragged_out(lines, "ST_OUT", out_parts)

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", OUT, "with", len(lines), "lines")


if __name__ == "__main__":
    main()
