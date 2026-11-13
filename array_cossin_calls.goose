// The checks and the public calls cossin, cossin_separate and cossin_blocks.
//
namespace scientific;

import mathshim;

// cx_check checks the shape, the split and the values of an operand
// before any value is used.
fn cx_check(x, p: i64, q: i64, tag: const u8[:]) -> Fault {
    let f = cs_check_stack(x.ndim, x.shape, x.data.len, 1, 2, tag);
    if f.kind != F_NONE { return f; }
    let m = x.shape[x.ndim - 1];
    if x.shape[x.ndim - 2] != m {
        return fault(F_SHAPE, tag, "the matrix is not square");
    }
    if p <= 0 || p >= m || q <= 0 || q >= m {
        return fault(F_RANGE, tag, "the split is outside the range of the side");
    }
    let g = carr_finite(x, tag);
    if g.kind != F_NONE { return g; }
    let n = batch_count(shape_batch(x.shape, 2));
    var s: i64 = 0;
    while s < n {
        if cx_orthogonal(x.data, s * m * m, m) > CX_ORTHO_TOL {
            return fault(F_VALUE, tag, "the matrix is not orthogonal");
        }
        s += 1;
    }
    return no_fault();
}

// cx_cut cuts the blocks of one matrix of a stack out of the
// outputs of cx_matrix. The block holds the rows lo to hi and the
// columns clo to chi of the matrix that starts at the place off.
fn cx_cut(d: f64[:], off: i64, m: i64, lo: i64, hi: i64, clo: i64, chi: i64, out: f64[>..]&) {
    var i: i64 = lo;
    while i < hi {
        var j: i64 = clo;
        while j < chi { out.push(d[off + i * m + j]); j += 1; }
        i += 1;
    }
}

// cossin gives the cosine-sine decomposition of an orthogonal matrix.
// The split is at the row p and at the column q. The result holds the
// left factor u, the cosine-sine matrix cs and the right factor vh. A
// skipped factor holds no element. The last two dimensions of the
// array give the matrix. The dimensions before them give the batch.
// The method follows scipy.linalg.cossin.
fn cossin(x, p: i64, q: i64, swap_sign: bool = false, compute_u: bool = true, compute_vh: bool = true) -> Arr, Arr, Arr, Fault {
    let f = cx_check(x, p, q, "cossin");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f; }
    let m = x.shape[x.ndim - 1];
    let batch = shape_batch(x.shape, 2);
    let n = batch_count(batch);
    var th: f64[>..] = [];
    var u: f64[>..] = [];
    var cs: f64[>..] = [];
    var vh: f64[>..] = [];
    var s: i64 = 0;
    while s < n {
        if !cx_matrix(x.data, s * m * m, m, p, q, swap_sign, th, u, cs, vh) {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "cossin", "the decomposition did not complete");
        }
        s += 1;
    }
    let sh = shape_matrix(batch, m, m);
    var ru = empty_arr();
    var rv = empty_arr();
    if compute_u { ru = arr_of(copy(u), sh); }
    if compute_vh { rv = arr_of(copy(vh), sh); }
    return ru, arr_of(copy(cs), sh), rv, no_fault();
}

// cossin_separate gives the blocks and the angles of the
// decomposition of one matrix. The result holds the blocks u1 and u2
// of the left factor, the angles, and the blocks v1h and v2h of the
// right factor. A stack of matrices is a fault, as SciPy fails there.
fn cossin_separate(x, p: i64, q: i64) -> Arr, Arr, Arr, Arr, Arr, Fault {
    if x.ndim != 2 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "cossin_separate", "the operand is not one matrix");
    }
    let f = cx_check(x, p, q, "cossin_separate");
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), f; }
    let m = x.shape[0];
    var th: f64[>..] = [];
    var u: f64[>..] = [];
    var cs: f64[>..] = [];
    var vh: f64[>..] = [];
    if !cx_matrix(x.data, 0, m, p, q, false, th, u, cs, vh) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "cossin_separate", "the decomposition did not complete");
    }
    var u1: f64[>..] = [];
    var u2: f64[>..] = [];
    var v1: f64[>..] = [];
    var v2: f64[>..] = [];
    cx_cut(u, 0, m, 0, p, 0, p, u1);
    cx_cut(u, 0, m, p, m, p, m, u2);
    cx_cut(vh, 0, m, 0, q, 0, q, v1);
    cx_cut(vh, 0, m, q, m, q, m, v2);
    return mat2(copy(u1), p, p), mat2(copy(u2), m - p, m - p), vec1(copy(th)), mat2(copy(v1), q, q), mat2(copy(v2), m - q, m - q), no_fault();
}

// cossin_blocks gives the decomposition of a matrix that the four
// blocks x11, x12, x21 and x22 form. The blocks may form a stack. The
// batch shapes of the four blocks must be equal.
fn cossin_blocks(x11, x12, x21, x22, swap_sign: bool = false, compute_u: bool = true, compute_vh: bool = true) -> Arr, Arr, Arr, Fault {
    let tag = "cossin_blocks";
    let f1 = cs_check_stack(x11.ndim, x11.shape, x11.data.len, 1, 2, tag);
    if f1.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f1; }
    let f2 = cs_check_stack(x12.ndim, x12.shape, x12.data.len, 1, 2, tag);
    if f2.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f2; }
    let f3 = cs_check_stack(x21.ndim, x21.shape, x21.data.len, 1, 2, tag);
    if f3.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f3; }
    let f4 = cs_check_stack(x22.ndim, x22.shape, x22.data.len, 1, 2, tag);
    if f4.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), f4; }
    let p = x11.shape[x11.ndim - 2];
    let q = x11.shape[x11.ndim - 1];
    let mp = x22.shape[x22.ndim - 2];
    let mq = x22.shape[x22.ndim - 1];
    if x12.shape[x12.ndim - 2] != p || x12.shape[x12.ndim - 1] != mq || x21.shape[x21.ndim - 2] != mp || x21.shape[x21.ndim - 1] != q {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the blocks do not fit");
    }
    if p + mp != q + mq {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the blocks do not form a square matrix");
    }
    let b1 = shape_batch(x11.shape, 2);
    if !cx_same(b1, shape_batch(x12.shape, 2)) || !cx_same(b1, shape_batch(x21.shape, 2)) || !cx_same(b1, shape_batch(x22.shape, 2)) {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the blocks hold different batch shapes");
    }
    let m = p + mp;
    let n = batch_count(b1);
    if m > SCHUR_MAX_SIDE || n > MAX_ELEMENTS / (m * m) {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_LIMIT, tag, "the matrix is above the limits");
    }
    // The blocks join, one matrix at a time.
    var d: f64[>..] = [];
    var s: i64 = 0;
    while s < n {
        var i: i64 = 0;
        while i < p {
            var j: i64 = 0;
            while j < q { d.push(x11.data[s * p * q + i * q + j]); j += 1; }
            j = 0;
            while j < mq { d.push(x12.data[s * p * mq + i * mq + j]); j += 1; }
            i += 1;
        }
        i = 0;
        while i < mp {
            var j: i64 = 0;
            while j < q { d.push(x21.data[s * mp * q + i * q + j]); j += 1; }
            j = 0;
            while j < mq { d.push(x22.data[s * mp * mq + i * mq + j]); j += 1; }
            i += 1;
        }
        s += 1;
    }
    let joined = arr_of(copy(d), shape_matrix(b1, m, m));
    return cossin(joined, p, q, swap_sign, compute_u, compute_vh);
}
