// The call svds: the loop of rebuilds, the vectors of the long side and the result.
//
// The helpers of the iteration live in array_sparse_svds.goose.
namespace scientific;

import mathshim;

// svds_complete gives a unit vector that stands orthogonal to the cnt
// vectors that start at off in the buffer b. Each vector holds n values. It
// takes the unit vectors in a fixed order and keeps the first one whose
// remainder norm is at least SVDS_KEEP. When none is, it keeps the best one.
// w must be empty. The call fills it with n values.
fn svds_complete(w: f64[>..]&, b: f64[:], off: i64, cnt: i64, n: i64) {
    var best: f64 = -1.0;
    var at: i64 = 0;
    var t: i64 = 0;
    while t < n {
        var e: f64[>..] = nd_zeros(n);
        e[t] = 1.0;
        svds_orth(e, b, off, cnt, n);
        let nr = lg_norm(e, 0, n);
        if nr > best { best = nr; at = t; }
        if nr >= SVDS_KEEP { break; }
        t += 1;
    }
    var e: f64[>..] = nd_zeros(n);
    e[at] = 1.0;
    svds_orth(e, b, off, cnt, n);
    svds_unit(e, n);
    var i: i64 = 0;
    while i < n { w.push(e[i]); i += 1; }
}

// svds_long builds the k vectors of the long side in lng, of nl values each.
// Vector i takes the place i * nl. A vector is the product of the matrix with
// the vector of the short side, scaled to norm one and orthogonalized
// against the vectors built before. The loop goes from the largest value
// down. A value that is small against the largest one, and a vector that
// loses its norm, take the completion.
fn svds_long(s, c: f64, wide: bool, nl: i64, k: i64, th: f64[:], thmax: f64, ys: f64[:], nd: i64, lng: f64[>..]&) -> Fault {
    var i: i64 = k - 1;
    while i >= 0 {
        var done = false;
        let t = th[i];
        if t > 0.0 && mathshim::root(t) > SVDS_DIRECT_SHARE * mathshim::root(thmax) {
            let y = copy_slice(ys[i * nd..i * nd + nd]);
            let xa, f = svds_mv(s, y, c, wide);
            if f.kind != F_NONE { return f; }
            var w = xa.data;
            if svds_unit(w, nl) > 0.0 {
                svds_orth(w, lng, (i + 1) * nl, k - 1 - i, nl);
                if svds_unit(w, nl) >= SVDS_KEEP {
                    var r: i64 = 0;
                    while r < nl { lng[i * nl + r] = w[r]; r += 1; }
                    done = true;
                }
            }
        }
        if !done {
            var w: f64[>..] = [];
            svds_complete(w, lng, (i + 1) * nl, k - 1 - i, nl);
            var r: i64 = 0;
            while r < nl { lng[i * nl + r] = w[r]; r += 1; }
        }
        i -= 1;
    }
    return no_fault();
}

// svds_refine replaces each singular value by the norm of the product of the
// matrix with the vector of the short side. The square root of a Ritz value
// of the normal operator loses a small value, because the operator squares
// the matrix. The norm of the product does not. A value that is small against
// the largest one keeps its first estimate. A product that is not finite
// gives F_VALUE.
fn svds_refine(s, c: f64, wide: bool, nl: i64, k: i64, th: f64[:], thmax: f64, ys: f64[:], nd: i64, sv: f64[>..]&) -> Fault {
    var i: i64 = 0;
    while i < k {
        if th[i] > 0.0 && mathshim::root(th[i]) > SVDS_DIRECT_SHARE * mathshim::root(thmax) {
            let y = copy_slice(ys[i * nd..i * nd + nd]);
            let xa, f = svds_mv(s, y, c, wide);
            if f.kind != F_NONE { return f; }
            let w = xa.data;
            let nr = lg_norm(w, 0, nl);
            if nr > 0.0 { sv[i] = nr * c; }
        }
        i += 1;
    }
    if !lg_finite(sv, k) { return fault(F_VALUE, "svds", "a singular value is not finite"); }
    return no_fault();
}

// svds_converged tells whether the k wanted Ritz pairs meet the bound. The
// pair at column base + i of the eigenvectors z has the residual
// |last * z[m - 1]|. The bound is tau times the largest Ritz value.
fn svds_converged(z: f64[:], w: f64[:], m: i64, last: f64, base: i64, k: i64, tau: f64) -> bool {
    var thmax = w[m - 1];
    if thmax < 0.0 { thmax = 0.0; }
    var i: i64 = 0;
    while i < k {
        if mathshim::abs_of(last * z[(m - 1) * m + base + i]) > tau * thmax { return false; }
        i += 1;
    }
    return true;
}

// svds_unit_rows gives k unit vectors of n values each, vector i holding a
// one at place i. The zero matrix uses them for both sides.
fn svds_unit_rows(k: i64, n: i64) -> f64[>..] {
    var out: f64[>..] = nd_zeros(k * n);
    var i: i64 = 0;
    while i < k { out[i * n + i] = 1.0; i += 1; }
    return out;
}

// svds_pack builds the result from the vectors of the short side ys, of nd
// values each, and the vectors of the long side lng, of nl values each. A
// matrix that is not wide has its short side on the right. The mode picks
// the sides that the call gives. A skipped side is an empty array.
fn svds_pack(rows: i64, cols: i64, k: i64, wide: bool, mode: i64, sv: f64[:], ys: f64[:], lng: f64[:]) -> Arr, Arr, Arr, Fault {
    var u: Arr = empty_arr();
    var vt: Arr = empty_arr();
    if mode == SVDS_BOTH || mode == SVDS_LEFT {
        var ud: f64[>..] = nd_zeros(rows * k);
        var i: i64 = 0;
        while i < k {
            var r: i64 = 0;
            while r < rows {
                if wide { ud[r * k + i] = ys[i * rows + r]; } else { ud[r * k + i] = lng[i * rows + r]; }
                r += 1;
            }
            i += 1;
        }
        u = mat2(copy(ud), rows, k);
    }
    if mode == SVDS_BOTH || mode == SVDS_RIGHT {
        var vd: f64[>..] = nd_zeros(k * cols);
        var i: i64 = 0;
        while i < k * cols {
            if wide { vd[i] = lng[i]; } else { vd[i] = ys[i]; }
            i += 1;
        }
        vt = mat2(copy(vd), k, cols);
    }
    let sd = copy_slice(sv[0..k]);
    return u, line_arr(copy(sd)), vt, no_fault();
}

// svds gives the k largest singular values of a sparse matrix, or the k
// smallest ones when which is "SM". The values stand in rising order. It
// gives u of rows by k, the values, and vt of k by cols. For triplet i, the
// product of the matrix with row i of vt equals value i times column i of u.
// k, tol, v0, ncv, maxiter and mode follow the contract of the package. A
// tol of zero means the precision of the machine. An empty v0 means a fixed
// start. An ncv of zero means the default size of the space. A maxiter of zero
// means 64 rebuilds. The mode is one of SVDS_BOTH, SVDS_VALUES, SVDS_LEFT and
// SVDS_RIGHT.
//
// An invalid operand gives a fault and three empty arrays. A run that does
// not converge within the rebuilds or the storage gives F_LIMIT.
fn svds(s, k: i64 = 6, which: const u8[:] = "LM", tol: f64 = 0.0, v0: f64[:] = [], ncv: i64 = 0, maxiter: i64 = 0, mode: i64 = 0) -> Arr, Arr, Arr, Fault {
    let tag = "svds";
    let fm = svds_matrix_check(s, tag);
    if fm.kind != F_NONE { return svds_empty(fm); }
    // The short side holds the vectors of the operator. The wide matrix has
    // more columns than rows, so its short side is the side of the rows.
    let wide = s.cols > s.rows;
    var d = s.cols;
    var big = s.rows;
    if wide { d = s.rows; big = s.cols; }
    let fo = svds_options_check(d, k, which, tol, v0, ncv, maxiter, mode, tag);
    if fo.kind != F_NONE { return svds_empty(fo); }
    var m = ncv;
    if m == 0 {
        m = 2 * k + 1;
        if m < 20 { m = 20; }
        if m > d { m = d; }
    }
    if m > MAX_SVDS_SPACE { m = MAX_SVDS_SPACE; }
    let fs = svds_storage_check(s.rows, s.cols, k, m, tag);
    if fs.kind != F_NONE { return svds_empty(fs); }
    var mmax = MAX_SVDS_SPACE;
    if d < mmax { mmax = d; }
    if MAX_ELEMENTS / big - 2 * k - 6 < mmax { mmax = MAX_ELEMENTS / big - 2 * k - 6; }
    var rebuilds = maxiter;
    if rebuilds == 0 { rebuilds = SVDS_DEFAULT_REBUILDS; }
    let c = svds_scale(s);
    if c == 0.0 {
        // A matrix of zeros: every value is zero and each side is a set of unit vectors.
        var sz: f64[>..] = nd_zeros(k);
        let ysz = svds_unit_rows(k, d);
        let lgz = svds_unit_rows(k, big);
        return svds_pack(s.rows, s.cols, k, wide, mode, sz, ysz, lgz);
    }
    var rng: i64 = EXPM_RNG_SEED;
    var start: f64[>..] = [];
    if v0.len == 0 {
        start = svds_fixed(d, rng);
    } else {
        start = copy_slice(v0);
    }
    var tau = tol * tol;
    if tau < 1.0e-14 { tau = 1.0e-14; }
    var base: i64 = 0;
    var turn: i64 = 0;
    var vs: f64[>..] = [];
    var wv: f64[>..] = [];
    var zv: f64[>..] = [];
    while true {
        vs = [];
        var al: f64[>..] = [];
        var be: f64[>..] = [];
        let last, fl = svds_lanczos(s, c, wide, d, m, start, vs, al, be);
        if fl.kind != F_NONE { return svds_empty(fl); }
        wv = [];
        zv = [];
        let fe = svds_eig(al, be, m, wv, zv);
        if fe.kind != F_NONE { return svds_empty(fe); }
        if !lg_finite(wv, m) { return svds_empty(fault(F_VALUE, tag, "a Ritz value is not finite")); }
        base = 0;
        if which == "LM" { base = m - k; }
        if m >= d || svds_converged(zv, wv, m, last, base, k, tau) { break; }
        if turn >= rebuilds || m >= mmax {
            return svds_empty(fault(F_LIMIT, tag, "the iteration did not converge within the rebuilds or the storage"));
        }
        m = 2 * m;
        if m > mmax { m = mmax; }
        turn += 1;
    }
    var thmax = wv[m - 1];
    if thmax < 0.0 { thmax = 0.0; }
    var th: f64[>..] = [];
    var sv: f64[>..] = [];
    var i: i64 = 0;
    while i < k {
        var t = wv[base + i];
        if t < 0.0 { t = 0.0; }
        th.push(t);
        sv.push(mathshim::root(t) * c);
        i += 1;
    }
    if !lg_finite(sv, k) { return svds_empty(fault(F_VALUE, tag, "a singular value is not finite")); }
    // The vectors of the short side: the Ritz vectors of the basis.
    var ys: f64[>..] = nd_zeros(k * d);
    i = 0;
    while i < k {
        var t: i64 = 0;
        while t < m {
            let coef = zv[t * m + base + i];
            var r: i64 = 0;
            while r < d { ys[i * d + r] = ys[i * d + r] + coef * vs[t * d + r]; r += 1; }
            t += 1;
        }
        i += 1;
    }
    let fr = svds_refine(s, c, wide, big, k, th, thmax, ys, d, sv);
    if fr.kind != F_NONE { return svds_empty(fr); }
    if mode == SVDS_VALUES { return empty_arr(), line_arr(copy(sv)), empty_arr(), no_fault(); }
    // The long side is the left side of a tall matrix and the right side of a wide one.
    var need_long = mode == SVDS_BOTH;
    if !wide && mode == SVDS_LEFT { need_long = true; }
    if wide && mode == SVDS_RIGHT { need_long = true; }
    var lng: f64[>..] = [];
    if need_long {
        lng = nd_zeros(k * big);
        let fv = svds_long(s, c, wide, big, k, th, thmax, ys, d, lng);
        if fv.kind != F_NONE { return svds_empty(fv); }
    }
    return svds_pack(s.rows, s.cols, k, wide, mode, sv, ys, lng);
}
