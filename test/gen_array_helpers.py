#!/usr/bin/env python3
"""Write the pins of the tests of the layout and the memory of the arrays.

The script needs NumPy 2.5.4. It runs from the maintainer. The user of
the package installs nothing.

The file it writes holds numbers only. Every family of cases holds the
inputs and the expected results. A ragged list is two buffers: the values
joined end to end, and the length of each part. The count of the cases
is the first value of the buffer of the counts.
"""
import numpy as np

OUT = "test/suite/array_helpers_pins.goose"


def emit(lines, name, values):
    lines.append("const %s: f64[>..] = [%s];" % (name, ", ".join(repr(float(v)) for v in values)))


def ragged(lines, name, parts):
    flat = [v for p in parts for v in p]
    emit(lines, name + "_D", flat)
    emit(lines, name + "_L", [len(p) for p in parts])


def pack_cases():
    """The cases of the pack and of the unpack."""
    cases = []
    bits9 = [1, 0, 1, 1, 0, 0, 1, 0, 1]
    m3 = np.array([[1, 0, 1], [1, 0, 0], [1, 0, 1]], float)
    cases.append((bits9, 1, -1, 0))
    cases.append((bits9, 1, -1, 1))
    cases.append(([1, 1, 1, 1, 1], 1, -1, 0))
    cases.append(([1] + [0] * 15 + [1], 1, -1, 0))
    cases.append(([1] + [0] * 15 + [1], 1, -1, 1))
    cases.append((m3, 2, 0, 0))
    cases.append((m3, 2, 1, 0))
    cases.append((m3, 2, 0, 1))
    cases.append((m3, 2, 1, 1))
    cases.append(([1, 0] * 8, 1, -1, 0))
    cases.append(([1, 0] * 8, 1, -1, 1))
    cases.append(([1, 1, 0, 1, 1, 0, 1, 0], 1, 0, 0))
    return cases


def unpack_cases():
    """The cases of the unpack: the bytes, the axis, the count and the order."""
    cases = []
    cases.append(([178], 1, -1, -1, 0))
    cases.append(([178], 1, -1, -1, 1))
    cases.append(([178], 1, -1, 3, 0))
    cases.append(([178], 1, -1, 12, 0))
    cases.append(([0, 255], 1, -1, -1, 0))
    cases.append(([255, 0], 1, -1, -1, 1))
    cases.append(([1, 2, 3], 1, -1, 5, 0))
    cases.append(([[178, 1]], 2, 1, -1, 0))
    cases.append(([[178], [1]], 2, 0, -1, 0))
    cases.append(([[178, 1], [1, 178]], 2, 1, 4, 0))
    cases.append(([[178], [1], [255]], 2, 0, 12, 1))
    return cases


def copy_cases():
    """The cases of the copy: the destination, the source and the mask."""
    cases = []
    cases.append(([0.0] * 6, 2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, []))
    cases.append(([0.0] * 6, 2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3,
                  [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
    cases.append(([0.0] * 3, 1, 3, [7.0, 8.0, 9.0], 1, 3, []))
    cases.append(([0.0] * 3, 1, 3, [7.0, 8.0, 9.0], 1, 3, [0.0, 1.0, 0.0]))
    cases.append(([0.0] * 4, 2, 2, [1.0, 2.0], 2, 1, []))
    cases.append(([0.0] * 4, 2, 2, [1.0, 2.0], 2, 1, [1.0, 0.0, 0.0, 1.0]))
    cases.append(([5.0] * 6, 2, 3, [0.0] * 6, 2, 3, []))
    cases.append(([5.0] * 6, 2, 3, [0.0] * 6, 2, 3, [1.0] * 6))
    cases.append(([0.0] * 4, 1, 4, [1.0, 2.0, 3.0, 4.0], 1, 4, [0.0, 0.0, 1.0, 1.0]))
    cases.append(([0.0] * 6, 2, 3, [1.0, 2.0, 3.0], 1, 3, []))
    cases.append(([0.0] * 9, 3, 3, [1.0, 2.0, 3.0], 3, 1, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    return cases


def block_cases():
    """The cases of the grid of the blocks: the blocks in the order of the grid."""
    cases = []
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    c = np.array([[9.0, 10.0], [11.0, 12.0]])
    d = np.array([[13.0, 14.0], [15.0, 16.0]])
    cases.append((2, 2, [a, b, c, d]))
    cases.append((1, 2, [a, b]))
    cases.append((2, 1, [a, c]))
    cases.append((1, 3, [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0, 5.0])]))
    cases.append((2, 2, [np.array([1.0]), np.array([2.0]), np.array([3.0]), np.array([4.0])]))
    cases.append((3, 2, [a, b, c, d, a, b]))
    cases.append((1, 1, [a]))
    cases.append((2, 3, [a, b, a, b, a, b]))
    # A grid of blocks of three dimensions. The blocks of a row follow
    # each other along the last axis, and the rows follow each other
    # along the axis before it.
    e = np.arange(8.0).reshape(2, 2, 2)
    f = np.arange(8.0, 16.0).reshape(2, 2, 2)
    g = np.arange(16.0, 24.0).reshape(2, 2, 2)
    h = np.arange(24.0, 32.0).reshape(2, 2, 2)
    k = np.arange(32.0, 44.0).reshape(2, 3, 2)
    l = np.arange(44.0, 56.0).reshape(2, 3, 2)
    cases.append((2, 2, [e, f, g, h]))
    cases.append((1, 2, [e, f]))
    # The rows may differ on the axis that grows.
    cases.append((2, 2, [e, f, k, l]))
    return cases


def roll_cases():
    """The cases of the roll: the shape, the axis and the place."""
    return [((2, 3, 4), 2, 0), ((2, 3, 4), 0, 1), ((2, 3, 4), 0, 0),
            ((2, 3, 4), 1, 0), ((2, 3, 4), 1, 3), ((2, 3, 4), 2, 3),
            ((4, 4), 0, 1), ((4, 4), 1, 0), ((5,), 0, 0), ((5,), 0, 1),
            ((2, 2, 2, 2), 3, 0), ((2, 2, 2, 2), 0, 4)]


def cut_cases():
    """The cases of the cut: the shape, the places and the axis."""
    return [((3, 4), [1, 3], 1), ((3, 4), [0, 2], 0), ((3, 4), [4], 1),
            ((3, 4), [], 1), ((3, 4), [2], 0), ((6,), [1, 2, 3, 4, 5], 0),
            ((2, 3), [3], 1), ((2, 3), [1, 2], 1), ((4, 5), [2, 4], 0),
            ((4, 5), [0, 1, 2, 3, 4], 1)]


def sort_cases():
    """The cases of the sort of the complex values: the pairs."""
    return [[3.0, 1.0, 1.0, 5.0, 1.0, 2.0, -1.0, 0.0],
            [1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, -1.0, 2.0, 1.0, -2.0, 0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0, 5.0, 1.0, 9.0],
            [0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
            [-1.0, -1.0, -1.0, -2.0],
            [3.0, 0.0, 1.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
            [7.0, -1.0, 7.0, 1.0, 6.0, 0.0]]


def stride_cases():
    """The cases of the strided array: the buffer, the shape and the steps."""
    buf = list(range(10))
    return [(buf, [3, 4], [8, 8]), (buf, [3], [24]), (buf, [2, 5], [8, 8]),
            (buf, [10], [8]), (buf, [1], [8]), (buf, [3, 3], [16, 8]),
            (buf, [4], [16]), (buf, [2, 2], [24, 8]), (buf, [5], [16]),
            (buf, [2], [40])]


def main():
    lines = ["// Generated by test/gen_array_helpers.py. Do not edit by hand.",
             "",
             "// The pins of the tests of the layout and the memory of the arrays.",
             "// They come from NumPy 2.5.4.",
             "namespace array_helpers_pins;",
             ""]

    # The pack.
    pc = pack_cases()
    ins, shp, ax, order, out = [], [], [], [], []
    for values, ndim, axis, order_flag in pc:
        arr = np.array(values, float).reshape(-1) if ndim == 1 else np.array(values, float)
        bitorder = "little" if order_flag else "big"
        got = np.packbits(arr.astype(bool), axis=axis, bitorder=bitorder)
        ins.append(list(arr.ravel()))
        shp.append(list(arr.shape))
        ax.append(axis)
        order.append(order_flag)
        out.append(list(got.ravel()))
    ragged(lines, "PACK_IN", ins)
    ragged(lines, "PACK_SHAPE", shp)
    emit(lines, "PACK_AXIS", ax)
    emit(lines, "PACK_ORDER", order)
    ragged(lines, "PACK_OUT", out)

    # The unpack.
    uc = unpack_cases()
    ins, shp, ax, cnt, order, out = [], [], [], [], [], []
    for values, ndim, axis, count, order_flag in uc:
        arr = np.array(values, float).reshape(-1) if ndim == 1 else np.array(values, float)
        kw = {} if count < 0 else {"count": count}
        bitorder = "little" if order_flag else "big"
        got = np.unpackbits(arr.astype(np.uint8), axis=axis, bitorder=bitorder, **kw)
        ins.append(list(arr.ravel()))
        shp.append(list(arr.shape))
        ax.append(axis)
        cnt.append(count)
        order.append(order_flag)
        out.append(list(got.ravel()))
    ragged(lines, "UNPACK_IN", ins)
    ragged(lines, "UNPACK_SHAPE", shp)
    emit(lines, "UNPACK_AXIS", ax)
    emit(lines, "UNPACK_COUNT", cnt)
    emit(lines, "UNPACK_ORDER", order)
    ragged(lines, "UNPACK_OUT", out)

    # The copy.
    cc = copy_cases()
    dst, dsh, src, ssh, mask, msh, out = [], [], [], [], [], [], []
    for dvals, dr, dc, svals, sr, sc, mvals in cc:
        d = np.array(dvals, float).reshape(dr, dc)
        s = np.array(svals, float)
        sb = s.reshape(sr, sc)
        if mvals:
            m = np.array(mvals, float).reshape(dr, dc)
            got = d.copy()
            np.copyto(got, sb, where=m != 0.0)
        else:
            got = d.copy()
            np.copyto(got, sb)
        dst.append(list(d.ravel()))
        dsh.append([dr, dc])
        src.append(list(s.ravel()))
        ssh.append([sr, sc])
        mask.append(list(mvals))
        msh.append([dr, dc] if mvals else [])
        out.append(list(got.ravel()))
    ragged(lines, "COPY_DST", dst)
    ragged(lines, "COPY_DST_SHAPE", dsh)
    ragged(lines, "COPY_SRC", src)
    ragged(lines, "COPY_SRC_SHAPE", ssh)
    ragged(lines, "COPY_MASK", mask)
    ragged(lines, "COPY_MASK_SHAPE", msh)
    ragged(lines, "COPY_OUT", out)

    # The grid of the blocks.
    bc = block_cases()
    rows, cols, bsh, bval, out, osh = [], [], [], [], [], []
    for r, c, blocks in bc:
        rows.append(r)
        cols.append(c)
        for b in blocks:
            bsh.append(list(b.shape))
            bval.append(list(b.ravel()))
        # The grid of NumPy: the blocks of a row follow each other along
        # the last axis, and the rows follow each other along the axis
        # before it. A grid of blocks of one dimension and more than one
        # row gives one more axis at the front.
        got = np.block([blocks[i * c:(i + 1) * c] for i in range(r)])
        osh.append(list(got.shape))
        out.append(list(got.ravel()))
    emit(lines, "BLOCK_ROWS", rows)
    emit(lines, "BLOCK_COLS", cols)
    ragged(lines, "BLOCK_SHAPE", bsh)
    ragged(lines, "BLOCK_VALUE", bval)
    ragged(lines, "BLOCK_OUT_SHAPE", osh)
    ragged(lines, "BLOCK_OUT", out)

    # The roll.
    rc = roll_cases()
    ish, ax, st, osh, oval = [], [], [], [], []
    for shape, axis, start in rc:
        n = int(np.prod(shape))
        arr = np.arange(float(n)).reshape(shape)
        got = np.rollaxis(arr, axis, start)
        ish.append(list(shape))
        ax.append(axis)
        st.append(start)
        osh.append(list(got.shape))
        oval.append(list(got.ravel()))
    ragged(lines, "ROLL_SHAPE", ish)
    emit(lines, "ROLL_AXIS", ax)
    emit(lines, "ROLL_START", st)
    ragged(lines, "ROLL_OUT_SHAPE", osh)
    ragged(lines, "ROLL_OUT", oval)

    # The cut at a list of places.
    uc2 = cut_cases()
    csh, places, ax, out = [], [], [], []
    for shape, pl, axis in uc2:
        n = int(np.prod(shape))
        arr = np.arange(float(n)).reshape(shape)
        got = np.split(arr, pl, axis=axis)
        csh.append(list(shape))
        places.append(list(pl))
        ax.append(axis)
        for piece in got:
            out.append(list(piece.ravel()))
    ragged(lines, "CUT_SHAPE", csh)
    ragged(lines, "CUT_PLACES", places)
    emit(lines, "CUT_AXIS", ax)
    ragged(lines, "CUT_OUT", out)

    # The sort of the complex values.
    sc = sort_cases()
    sin, sout = [], []
    for pairs in sc:
        z = np.array([complex(pairs[i], pairs[i + 1]) for i in range(0, len(pairs), 2)])
        got = np.sort_complex(z)
        sin.append(list(pairs))
        sout.append([v for c in got for v in (c.real, c.imag)])
    ragged(lines, "SORT_IN", sin)
    ragged(lines, "SORT_OUT", sout)

    # The strided array.
    stc = stride_cases()
    buf, shp, step, out = [], [], [], []
    for values, shape, strides in stc:
        arr = np.array(values, float)
        got = np.lib.stride_tricks.as_strided(arr, shape=tuple(shape),
                                             strides=tuple(strides))
        buf.append(list(values))
        shp.append(list(shape))
        step.append(list(strides))
        out.append([float(v) for v in got.ravel()])
    ragged(lines, "STRIDE_BUFFER", buf)
    ragged(lines, "STRIDE_SHAPE", shp)
    ragged(lines, "STRIDE_STEP", step)
    ragged(lines, "STRIDE_OUT", out)

    lines.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print("wrote", OUT, "with", len(pc), "pack cases,", len(uc), "unpack cases,",
          len(cc), "copy cases,", len(bc), "block cases,", len(rc), "roll cases,",
          len(uc2), "cut cases,", len(sc), "sort cases and", len(stc), "stride cases")


main()
