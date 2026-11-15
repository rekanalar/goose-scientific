"""Writes array_integrate_lebedev_table.goose: the table of the rules of
Lebedev for the sphere.

The table holds the points of the rule in the compact form of Lebedev
and Laikov: every order holds a list of blocks, and every block holds
the kind of the grid, the two parameters and the weight of the block.
The points of the grid come from the table of the kinds of the grid.

Needs: Python 3, NumPy 2.5.4, SciPy 1.18.1.
Run: /home/ubuntu/works/goose/tmp/venv/bin/python -I test/gen_lebedev.py
"""
import os
import numpy as np
from scipy.integrate import _lebedev as L

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ORDERS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35,
          41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119,
          125, 131]
DEGREES = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302,
           350, 434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702,
           3074, 3470, 3890, 4334, 4802, 5294, 5810]
COUNTS = {1: 6, 2: 12, 3: 8, 4: 24, 5: 24, 6: 48}
# The symbol of every coordinate: 0 is zero, 1 is a, 2 is -a, 3 is b,
# 4 is -b, 5 is c and 6 is -c.
SYMS = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]


def probe(kind):
    """The pattern of the points of one kind of the grid."""
    a = 0.3
    b = 0.4
    n = COUNTS[kind]
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    w = np.zeros(n)

    class Leb:
        pass

    leb = Leb()
    leb.x, leb.y, leb.z, leb.w = x, y, z, w
    L.get_lebedev_recurrence_points(kind, 0, a, b, 0.125, leb)
    # The values of the parameters that the kind uses.
    # The kinds 1, 2 and 3 take the value of a from the grid itself.
    if kind == 1:
        a = 1.0
    elif kind == 2:
        a = np.sqrt(0.5)
    elif kind == 3:
        a = np.sqrt(1.0 / 3.0)
    vals = {0: 0.0, 1: a, 2: -a, 3: b, 4: -b, 5: 0.0, 6: 0.0}
    if kind == 4:
        vals[3] = np.sqrt(1.0 - 2.0 * a * a)
        vals[4] = -vals[3]
    if kind == 5:
        vals[3] = np.sqrt(1.0 - a * a)
        vals[4] = -vals[3]
    if kind == 6:
        vals[3] = b
        vals[4] = -b
        vals[5] = np.sqrt(1.0 - a * a - b * b)
        vals[6] = -vals[5]
    out = []
    for i in range(n):
        row = []
        for v in (x[i], y[i], z[i]):
            hit = 0
            for sym in range(7):
                if abs(v - vals[sym]) < 1.0e-12:
                    hit = sym
                    break
            if hit == 0 and abs(v) > 1.0e-12:
                raise SystemExit('the point %d holds an unknown value' % i)
            row.append(hit)
        out.append(row)
    return out


def record_calls():
    """The calls of every order, as the package of SciPy makes them."""
    orig = L.get_lebedev_recurrence_points
    calls = []
    table = []

    def wrapper(kind, start, a, b, v, leb):
        calls.append((kind, a, b, v))
        return orig(kind, start, a, b, v, leb)

    L.get_lebedev_recurrence_points = wrapper
    try:
        for degree in DEGREES:
            calls.clear()
            L.get_lebedev_sphere(degree)
            table.append(list(calls))
    finally:
        L.get_lebedev_recurrence_points = orig
    return table


def check(table, patterns):
    """The table builds the points that SciPy gives."""
    for index, degree in enumerate(DEGREES):
        x = np.zeros(degree)
        y = np.zeros(degree)
        z = np.zeros(degree)
        w = np.zeros(degree)

        class Leb:
            pass

        leb = Leb()
        leb.x, leb.y, leb.z, leb.w = x, y, z, w
        at = 0
        for kind, a0, b, v in table[index]:
            count = COUNTS[kind]
            # The kinds 1, 2 and 3 take the value of a from the grid.
            a = a0
            if kind == 1:
                a = 1.0
            elif kind == 2:
                a = np.sqrt(0.5)
            elif kind == 3:
                a = np.sqrt(1.0 / 3.0)
            if kind == 4:
                bb = np.sqrt(1.0 - 2.0 * a * a)
                cc = 0.0
            elif kind == 5:
                bb = np.sqrt(1.0 - a * a)
                cc = 0.0
            elif kind == 6:
                bb = b
                cc = np.sqrt(1.0 - a * a - b * b)
            else:
                bb = b
                cc = 0.0
            for row in patterns[kind]:
                for axis in range(3):
                    sym = row[axis]
                    val = 0.0
                    if sym == 1:
                        val = a
                    elif sym == 2:
                        val = -a
                    elif sym == 3:
                        val = bb
                    elif sym == 4:
                        val = -bb
                    elif sym == 5:
                        val = cc
                    elif sym == 6:
                        val = -cc
                    if axis == 0:
                        x[at] = val
                    elif axis == 1:
                        y[at] = val
                    else:
                        z[at] = val
                w[at] = 4.0 * np.pi * v
                at += 1
        ref_x, ref_w = L.lebedev_rule(ORDERS[index])
        if not (np.allclose(x, ref_x[0]) and np.allclose(y, ref_x[1])
                and np.allclose(z, ref_x[2]) and np.allclose(w, ref_w)):
            raise SystemExit('the order %d does not match SciPy' % ORDERS[index])


def const_i64(name, vals):
    body = ", ".join(str(int(v)) for v in vals)
    return f"const {name}: i64[{len(vals)}] = [{body}];\n"


def const_f64(name, vals):
    body = ", ".join(repr(float(v)) for v in vals)
    return f"const {name}: f64[{len(vals)}] = [{body}];\n"


def main():
    patterns = {kind: probe(kind) for kind in (1, 2, 3, 4, 5, 6)}
    table = record_calls()
    check(table, patterns)
    out = ["// Generated by test/gen_lebedev.py. Do not edit by hand.\n",
           "// The table of the rules of Lebedev for the sphere. The values come from\n",
           "// SciPy 1.18.1.\n",
           "namespace scientific;\n\n"]
    # The patterns of every kind of the grid, one after the other, and
    # the place of every kind in the table.
    flat = []
    off = [0]
    for kind in (1, 2, 3, 4, 5, 6):
        for row in patterns[kind]:
            flat.extend(row)
        off.append(len(flat))
    out.append(const_i64("LEB_KP", flat))
    out.append(const_i64("LEB_KOFF", off))
    kinds = []
    aa = []
    bb = []
    vv = []
    start = []
    count = []
    points = []
    for calls in table:
        start.append(len(kinds))
        count.append(len(calls))
        points.append(sum(COUNTS[c[0]] for c in calls))
        for kind, a, b, v in calls:
            kinds.append(kind)
            aa.append(a)
            bb.append(b)
            vv.append(v)
    out.append(const_i64("LEB_CALL_K", kinds))
    out.append(const_f64("LEB_CALL_A", aa))
    out.append(const_f64("LEB_CALL_B", bb))
    out.append(const_f64("LEB_CALL_V", vv))
    out.append(const_i64("LEB_START", start))
    out.append(const_i64("LEB_COUNT", count))
    out.append(const_i64("LEB_POINTS", points))
    path = os.path.join(ROOT, "array_integrate_lebedev_table.goose")
    with open(path, "w") as f:
        f.writelines(out)
    print("ok", len(kinds), "calls")


if __name__ == "__main__":
    main()
