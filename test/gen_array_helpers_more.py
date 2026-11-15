#!/usr/bin/env python3
"""Write the pins of the tests of the small helpers of NumPy.

The script needs NumPy 2.5.4. It runs from the maintainer, from the root
of the package: python3 -I test/gen_array_helpers_more.py. The user of
the package installs nothing.

The file it writes holds numbers only. A ragged list is two buffers: the
values joined end to end, and the length of each part. A flag is 1.0 or
0.0.
"""
import numpy as np

OUT = "test/suite/array_helpers_more_pins.goose"


def emit(lines, name, values):
    lines.append("const %s: f64[>..] = [%s];" % (name, ", ".join(repr(float(v)) for v in values)))


def ragged(lines, name, parts):
    emit(lines, name + "_D", [v for p in parts for v in p])
    emit(lines, name + "_L", [len(p) for p in parts])


def equiv_pins(lines):
    """array_equiv: two arrays with their shapes, and the answer."""
    r = lambda n: list(range(1, n + 1))
    cases = [
        ([3], [1, 2, 3], [2, 3], [1, 2, 3, 1, 2, 3]),
        ([3], [1, 2, 3], [3], [1, 2, 4]),
        ([2], [1, 2], [2, 2], [1, 2, 3, 4]),
        ([2], [1, 2], [3], [1, 2, 3]),
        ([2], [1, 2], [2, 3], r(6)),
        ([2, 1], [1, 2], [2], [1, 2]),
        ([2, 1], [1, 1], [2], [1, 1]),
        ([1], [5], [2, 2], [5, 5, 5, 5]),
        ([1], [5], [2, 2], [5, 5, 5, 6]),
        ([2, 3], r(6), [2, 3], r(6)),
        ([2, 3], r(6), [2, 3], [1, 2, 3, 4, 5, 7]),
        ([2, 2, 1], [1, 2, 1, 2], [1, 2], [1, 2]),
        ([2, 2, 1], [1, 1, 2, 2], [1, 2], [1, 2]),
        ([4], r(4), [2, 2], r(4)),
        ([1, 1], [0], [3], [0, 0, 0]),
        ([3], [0, 0, 0], [1, 1], [0]),
    ]
    sa, da, sb, db, out = [], [], [], [], []
    for s1, d1, s2, d2 in cases:
        a = np.array(d1, float).reshape(s1)
        b = np.array(d2, float).reshape(s2)
        sa.append(s1); da.append(d1); sb.append(s2); db.append(d2)
        out.append(1 if np.array_equiv(a, b) else 0)
    ragged(lines, "EQ_A", da)
    ragged(lines, "EQ_A_SHAPE", sa)
    ragged(lines, "EQ_B", db)
    ragged(lines, "EQ_B_SHAPE", sb)
    emit(lines, "EQ_OUT", out)


def divmod_pins(lines):
    """divmod: a column of dividends and a row of divisors, all pairs."""
    a = [7.0, -7.0, 1.5, -1.5, 0.0, 7.25, -7.25, 1e10, -1e10, 0.3, 5.5, 1e-5]
    b = [2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 1e-3, 0.1, 1e7, -1e7, 0.7, 2.5]
    q, r = np.divmod(np.array(a)[:, None], np.array(b)[None, :])
    emit(lines, "DM_A", a)
    emit(lines, "DM_B", b)
    emit(lines, "DM_Q", q.flatten())
    emit(lines, "DM_R", r.flatten())


def ric_pins(lines):
    """real_if_close: the parts of a complex array, the tolerance, the answer."""
    cases = [
        ([1.0, 2.0], [1e-13, 0.0], 100.0),
        ([1.0, 2.0], [1e-13, 0.0], 1000.0),
        ([1.0, 2.0], [1e-13, 0.0], 0.5),
        ([1.0, 2.0], [1e-13, 0.0], 1e-12),
        ([1.0, 2.0], [1e-13, 0.0], 0.0),
        ([1.0, 2.0], [1e-16, -1e-16], 100.0),
        ([1.0, 2.0], [1e-16, -1e-16], 1.0),
        ([1.0, 2.0], [1e-16, -1e-16], 1.0001),
        ([1.0, 2.0], [2e-14, 0.0], 100.0),
        ([1.0, 2.0], [2e-14, 0.0], 90.0),
        ([3.0, 4.0, 5.0], [0.0, 0.0, 0.0], 100.0),
        ([3.0, 4.0, 5.0], [0.0, 0.0, 3.0], 1e5),
        ([3.0], [-1e-15], 100.0),
        ([3.0], [-1e-13], 100.0),
        ([3.0], [1.0], 100.0),
        ([3.0], [1.0], 1e20),
        ([3.0], [1e-300], 100.0),
        ([3.0], [1e-300], -5.0),
    ]
    re, im, tol, out = [], [], [], []
    for r, i, t in cases:
        z = np.empty(len(r), complex)
        z.real = r
        z.imag = i
        res = np.real_if_close(z, tol=t)
        re.append(r); im.append(i); tol.append(t)
        out.append(0 if np.iscomplexobj(res) else 1)
    ragged(lines, "RIC_RE", re)
    ragged(lines, "RIC_IM", im)
    emit(lines, "RIC_TOL", tol)
    emit(lines, "RIC_OUT", out)


def dsplit_pins(lines):
    """dsplit: the shape, the count of pieces or the list of places, the pieces."""
    cases = [
        ((2, 3, 4), 4), ((2, 3, 4), 2), ((2, 3, 4), 1), ((2, 3, 4), [1, 3]),
        ((2, 2, 6), [2, 4]), ((2, 3, 4, 2), 2), ((1, 1, 8), 4),
        ((2, 2, 6), [1, 2, 5]), ((3, 1, 5), [2]), ((2, 3, 4), [4]),
    ]
    shape, arg, mode, pieces, widths, data = [], [], [], [], [], []
    for sh, ar in cases:
        a = np.arange(int(np.prod(sh)), dtype=float).reshape(sh)
        out = np.dsplit(a, ar)
        shape.append(list(sh))
        arg.append(ar if isinstance(ar, list) else [ar])
        mode.append(1 if isinstance(ar, list) else 0)
        pieces.append(len(out))
        for p in out:
            widths.append(p.shape[2])
            data.append([float(v) for v in p.flatten()])
    ragged(lines, "DS_SHAPE", shape)
    ragged(lines, "DS_ARG", arg)
    emit(lines, "DS_MODE", mode)
    emit(lines, "DS_PIECES", pieces)
    emit(lines, "DS_WIDTH", widths)
    ragged(lines, "DS_OUT", data)


def build_pins(lines):
    """fromfunction and fromiter."""
    emit(lines, "FF_A", np.fromfunction(lambda i, j: i * 10 + j, (3, 4)).flatten())
    emit(lines, "FF_B", np.fromfunction(lambda i: i * i + 0.5, (5,)).flatten())
    emit(lines, "FF_C", np.fromfunction(lambda i, j, k: i * 100 + j * 10 + k, (2, 3, 2)).flatten())
    emit(lines, "FI_A", np.fromiter((0.5 * i * i - i for i in range(7)), float, 7))


def main():
    lines = ["// Generated by test/gen_array_helpers_more.py. Do not edit by hand.", "",
             "// The pins of the tests of the small helpers of NumPy. They come from",
             "// NumPy 2.5.4.",
             "namespace array_helpers_more_pins;", ""]
    equiv_pins(lines)
    divmod_pins(lines)
    ric_pins(lines)
    dsplit_pins(lines)
    build_pins(lines)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
