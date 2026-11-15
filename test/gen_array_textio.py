#!/usr/bin/env python3
"""Write the pins of the tests of the text calls and the files of NumPy.

The script needs NumPy 2.5.4. It runs from the maintainer, from the
root of the package, with the interpreter of the measurement
environment:

    /path/to/venv/bin/python -I test/gen_array_textio.py

The user of the package installs nothing. The file the script writes
holds numbers only.

A text or a file holds its bytes as a run of numbers, and a second list
holds the start and the length of each run. A value holds two numbers,
the value and a code: 0 finite, 1 the missing value, 2 the infinity and
3 the negative infinity. A value that is not finite holds 0 in the
place of the value.

A case of a print holds the kind of the call (1 array2string, 2
array_repr, 3 array_str), the count of dimensions and the three
lengths, the count of values, the first pair of the values, the first
row of the options, and the run of the expected text. The options hold
the precision, the threshold, the edgeitems, the line width, the
suppress flag, the floatmode (0 maxprec, 1 fixed, 2 unique, 3
maxprec_equal) and the sign (0 minus, 1 plus, 2 space). A value of -1
keeps the default of NumPy. A case of array_repr and array_str uses
only the line width, because NumPy takes no other option there.

A case of a read holds the kind (1 loadtxt, 2 genfromtxt), the
expected shape, the count of expected values, the first pair, the run
of the file, the first row of the options, the code of the fill value
and the fill value. The options hold the run of the delimiter (-1 for
the whitespace of any width), the run of the comment mark (-1 for
none), the rows to skip, the rows to hold (-1 for all), the count of
the columns to take and the first column. A case of genfromtxt with a
fill code of 0 holds a finite fill value.

A case of a write holds the shape, the values, the first row of the
options and the run of the expected file. The options hold the format
(0 scientific, 1 fixed, 2 whole), the count of digits, and the runs of
the delimiter, the header, the footer and the mark. A run of -1 holds
an empty text.

A case of save holds the shape, the values and the run of the .npy
bytes of NumPy. A case of fromregex holds the count of values, the
first pair, the run of the pattern, the run of the file and the
expected shape.
"""
import io

import numpy as np
import re as remod

OUT = "test/suite/array_textio_pins.goose"

FLOATMODES = ["maxprec", "fixed", "unique", "maxprec_equal"]
SIGNS = ["-", "+", " "]
FORMATS = {"%.18e": (0.0, 18.0), "%.4f": (1.0, 4.0), "%d": (2.0, 0.0)}


class Pool:
    """The runs of bytes and the start and the length of each run."""

    def __init__(self):
        self.bytes = []
        self.spans = []

    def add(self, data):
        if isinstance(data, str):
            data = data.encode()
        self.spans += [float(len(self.bytes)), float(len(data))]
        self.bytes += [float(b) for b in data]
        return len(self.spans) // 2 - 1


POOL = Pool()
VALUES = []


def vals_of(arr):
    """The first pair and the count of the values of an array."""
    start = len(VALUES) // 2
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    for x in flat:
        code = 0.0
        if x != x:
            code = 1.0
        elif x == np.inf:
            code = 2.0
        elif x == -np.inf:
            code = 3.0
        VALUES.extend([float(x) if code == 0.0 else 0.0, code])
    return float(start), float(len(flat))


def dims_of(a):
    """The shape padded to three lengths with ones."""
    out = [float(d) for d in a.shape]
    while len(out) < 3:
        out.append(1.0)
    return out


def np_print(kind, a, o):
    """The text of NumPy for one print case."""
    k = {}
    if o.get("precision", -1.0) >= 0:
        k["precision"] = int(o["precision"])
    if o.get("threshold", -1.0) >= 0:
        k["threshold"] = int(o["threshold"])
    if o.get("edgeitems", -1.0) >= 0:
        k["edgeitems"] = int(o["edgeitems"])
    if o.get("linewidth", -1.0) >= 0:
        k["max_line_width"] = int(o["linewidth"])
    if o.get("suppress", -1.0) >= 0:
        k["suppress_small"] = bool(o["suppress"])
    if o.get("floatmode", -1.0) >= 0 and kind == 1:
        k["floatmode"] = FLOATMODES[int(o["floatmode"])]
    if o.get("sign", -1.0) >= 0 and kind == 1:
        k["sign"] = SIGNS[int(o["sign"])]
    if kind == 1:
        return str(np.array2string(a, **k))
    if kind == 2:
        return str(np.array_repr(a, **k))
    return str(np.array_str(a, **k))


def print_cases():
    """The cases of array2string, array_repr and array_str."""
    cases, opts = [], []
    a1 = np.array([1.0, -2.5, 3.25])
    a2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    a3 = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    an = np.array([1.0, np.nan, np.inf, -np.inf])
    plan = [
        (1, a1, {}), (1, a2, {}), (1, a3, {}), (1, an, {}),
        (1, a1, {"precision": 2}), (1, a1, {"floatmode": 3}),
        (1, np.array([1e-9, 1.0]), {"suppress": 1}),
        (1, np.arange(20.0), {"threshold": 6}),
        (1, np.arange(30.0), {"linewidth": 40}),
        (1, a1, {"sign": 1}),
        (1, np.arange(200.0).reshape(10, 20), {"threshold": 50}),
        (2, a1, {}), (2, a2, {}), (2, an, {}),
        (2, np.arange(1200.0), {}),
        (2, np.arange(1200.0), {"linewidth": 200}),
        (3, a1, {}),
    ]
    for kind, a, o in plan:
        start, n = vals_of(a)
        opts += [o.get("precision", -1.0), o.get("threshold", -1.0),
                 o.get("edgeitems", -1.0), o.get("linewidth", -1.0),
                 o.get("suppress", -1.0), o.get("floatmode", -1.0),
                 o.get("sign", -1.0)]
        cases += ([float(kind), float(a.ndim)] + dims_of(a)
                  + [n, start, float(len(opts) // 7 - 1),
                     float(POOL.add(np_print(kind, a, o))), 0.0])
    return cases, opts


def load_cases():
    """The cases of loadtxt and genfromtxt."""
    cases, opts, usecols = [], [], []
    table = "1,2\n3,4\n5,6\n7,8\n"

    def one(kind, text, delim, comments, skiprows, max_rows, cols, fill):
        k = {"skiprows": skiprows, "usecols": cols}
        if delim:
            k["delimiter"] = delim
        if comments:
            k["comments"] = comments
        if max_rows >= 0:
            k["max_rows"] = max_rows
        if kind == 1:
            got = np.loadtxt(io.StringIO(text), **k)
        else:
            del k["skiprows"]
            k["skip_header"] = skiprows
            if max_rows >= 0:
                k["max_rows"] = max_rows
            if fill is not None:
                k["filling_values"] = fill
            got = np.genfromtxt(io.StringIO(text), **k)
        start, n = vals_of(got)
        di = POOL.add(delim) if delim else -1
        ci = POOL.add(comments) if comments else -1
        us = len(usecols)
        nu = 0
        if cols:
            nu = len(cols)
            usecols.extend(float(c) for c in cols)
        opts.extend([di, ci, float(skiprows), float(max_rows), float(nu), float(us)])
        fcode, fval = (1.0, 0.0)
        if fill is not None:
            fcode, fval = 0.0, float(fill)
        cases.extend([float(kind), float(got.ndim)] + dims_of(got)
                     + [n, start, float(POOL.add(text)),
                        float(len(opts) // 6 - 1), fcode, fval, 0.0, 0.0])

    one(1, table, ",", None, 0, -1, None, None)
    one(1, table, ",", None, 1, -1, None, None)
    one(1, table, ",", None, 0, -1, (1,), None)
    one(1, table, ",", None, 0, 2, None, None)
    one(1, "# c\n1 2\n3 4", None, "#", 0, -1, None, None)
    one(2, "1,2\n3,\n5,6\n", ",", None, 0, -1, None, None)
    one(2, "1,2\n3,\n5,6\n", ",", None, 0, -1, None, -1)
    return cases, opts, usecols


def save_cases():
    """The cases of savetxt."""
    cases, opts = [], []
    a2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    plan = [
        (a2, "%.18e", " ", "", "", "# "),
        (a2, "%d", ",", "a,b", "end", "#!"),
        (np.array([[1.5, -2.25]]), "%.18e", " ", "", "", "# "),
    ]
    for a, fmt, delim, header, footer, comments in plan:
        out = io.StringIO()
        np.savetxt(out, a, fmt=fmt, delimiter=delim, header=header,
                   footer=footer, comments=comments)
        start, n = vals_of(a)
        fcode, prec = FORMATS[fmt]
        opts += [fcode, prec, float(POOL.add(delim)),
                 float(POOL.add(header)) if header else -1.0,
                 float(POOL.add(footer)) if footer else -1.0,
                 float(POOL.add(comments))]
        cases += ([1.0, float(a.ndim)] + dims_of(a)
                  + [n, start, float(len(opts) // 6 - 1),
                     float(POOL.add(out.getvalue())), 0.0])
    return cases, opts


def npy_cases():
    """The cases of save: the .npy bytes of NumPy."""
    cases = []
    for a in [np.array([1.0, -2.5]),
              np.array([[1.0, 2.0], [3.0, 4.0]]),
              np.array([0.1, -0.0, np.nan, np.inf])]:
        bio = io.BytesIO()
        np.save(bio, a)
        start, n = vals_of(a)
        cases += ([float(a.ndim)] + dims_of(a)
                  + [n, start, float(POOL.add(bio.getvalue())), 0.0])
    return cases


def regex_cases():
    """The case of fromregex."""
    text = "1.5 2.5\n3.5 4.5\n"
    pattern = r"([0-9.]+) ([0-9.]+)"
    got = np.array([[float(m[0]), float(m[1])] for m in remod.findall(pattern, text)])
    start, n = vals_of(got)
    return [n, start, float(POOL.add(pattern)), float(POOL.add(text)),
            float(got.ndim)] + dims_of(got)


def emit(lines, name, values):
    lines.append("const %s: f64[>..] = [%s];" % (name, ", ".join(repr(float(v)) for v in values)))


def main():
    pc, po = print_cases()
    lc, lo, lu = load_cases()
    sc, so = save_cases()
    nc = npy_cases()
    rc = regex_cases()
    lines = ["// Generated by test/gen_array_textio.py. Do not edit by hand.", "",
             "// The pins of the tests of the text calls and the files.",
             "// They come from NumPy 2.5.4.",
             "namespace array_textio_pins;", ""]
    emit(lines, "BYTES", POOL.bytes)
    emit(lines, "SPANS", POOL.spans)
    emit(lines, "VALUES", VALUES)
    emit(lines, "P_CASES", pc)
    emit(lines, "P_OPTS", po)
    emit(lines, "L_CASES", lc)
    emit(lines, "L_OPTS", lo)
    emit(lines, "L_USECOLS", lu)
    emit(lines, "S_CASES", sc)
    emit(lines, "S_OPTS", so)
    emit(lines, "N_CASES", nc)
    emit(lines, "R_CASES", rc)
    lines.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print("pins", len(POOL.spans) // 2, "runs,", len(VALUES) // 2, "values,",
          len(pc) // 10, "prints,", len(lc) // 12, "reads,", len(sc) // 9,
          "writes,", len(nc) // 7, "saves")


main()
