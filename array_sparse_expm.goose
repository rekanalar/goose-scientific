// The sparse one-norm estimate and the matrix exponential.
//
// onenormest gives a lower bound of the one-norm of a matrix of compressed
// rows. The block algorithm of Higham and Tisseur does the work. The
// package draws the columns of the block with a generator of a fixed seed,
// so two calls on the same matrix give the same estimate. SciPy draws at
// random, so its estimate can differ between two runs.
//
// expm_multiply gives the product of the matrix exponential of a matrix of
// compressed rows with a vector or with a matrix of columns. The work keeps
// the matrix sparse. The series of the shifted matrix is applied to the
// right side, so no dense matrix is ever built. The package uses its own
// order and its own scale. SciPy takes them from an estimate of the norms
// of the powers of the matrix.
namespace scientific;

import mathshim;

// The largest scale of the exponential. A matrix with a larger one-norm
// gives a fault of kind F_LIMIT.
const MAX_EXPM_SCALE: i64 = 100000;

// The order of the series of the exponential.
const MAX_EXPM_TERMS: i64 = 20;

// The largest count of the times of the range form.
const MAX_EXPM_STEPS: i64 = 4096;

// A matrix of this side or less gets the exact one-norm.
const EXPM_EXACT_SIDE: i64 = 4;

// The generator of the estimate. The package keeps its own, so the estimate
// is the same on every run and on every build of the compiler.
const EXPM_RNG_MOD: i64 = 2147483647;
const EXPM_RNG_MUL: i64 = 48271;
const EXPM_RNG_SEED: i64 = 12345;

// expm_matrix_check checks the matrix of the estimate and of the
// exponential. The matrix is square and every stored value is finite.
fn expm_matrix_check(s, tag: const u8[:]) -> Fault {
    let f = csr_check(s.rows, s.cols, tag);
    if f.kind != F_NONE { return f; }
    if s.rows != s.cols { return fault(F_SHAPE, tag, "the matrix is not square"); }
    if s.nnz > MAX_SPARSE_NNZ { return fault(F_LIMIT, tag, "the count of the stored values stands above the limit"); }
    let b = bmat_check_block(s, tag);
    if b.kind != F_NONE { return b; }
    var k: i64 = 0;
    while k < s.nnz {
        if mathshim::is_finite_of(csr_val(s, k)) == 0.0 { return fault(F_VALUE, tag, "a stored value is not finite"); }
        k += 1;
    }
    return no_fault();
}

// expm_side_check checks the right side of the exponential against the
// matrix. The right side holds one column or many columns. Its shape must
// agree with its count of dimensions and with its count of elements.
fn expm_side_check(s, b, tag: const u8[:]) -> Fault {
    let f = check_stack(b.ndim, b.shape, b.data.len, 1, tag);
    if f.kind != F_NONE { return f; }
    if b.ndim != 1 && b.ndim != 2 { return fault(F_SHAPE, tag, "the right side holds neither one column nor many columns"); }
    if b.shape[0] != s.rows { return fault(F_SHAPE, tag, "the count of the rows of the right side differs from the side of the matrix"); }
    var k: i64 = 1;
    if b.ndim == 2 { k = b.shape[1]; }
    if s.rows > MAX_ELEMENTS / k { return fault(F_LIMIT, tag, "the count of the columns of the right side stands above the limit"); }
    var i: i64 = 0;
    while i < b.data.len {
        if mathshim::is_finite_of(b.data[i]) == 0.0 { return fault(F_VALUE, tag, "a value of the right side is not finite"); }
        i += 1;
    }
    return no_fault();
}

// expm_sums gives the sum of the absolute values of every column, the value
// of every diagonal entry and the sum of the absolute values of the stored
// entries of the diagonal.
fn expm_sums(s) -> Arr, Arr, Arr {
    var ca: f64[>..] = [];
    var dg: f64[>..] = [];
    var da: f64[>..] = [];
    var i: i64 = 0;
    while i < s.cols { ca.push(0.0); dg.push(0.0); da.push(0.0); i += 1; }
    var r: i64 = 0;
    while r < s.rows {
        var k = csr_ptr(s, r);
        while k < csr_ptr(s, r + 1) {
            let j = csr_col(s, k);
            let v = csr_val(s, k);
            ca[j] = ca[j] + mathshim::abs_of(v);
            if j == r { dg[r] = dg[r] + v; da[r] = da[r] + mathshim::abs_of(v); }
            k += 1;
        }
        r += 1;
    }
    return line_arr(copy(ca)), line_arr(copy(dg)), line_arr(copy(da));
}

// expm_column_of gives the column j of the matrix.
fn expm_column_of(s, j: i64) -> Arr {
    var out: f64[>..] = [];
    var r: i64 = 0;
    while r < s.rows {
        var t: f64 = 0.0;
        var k = csr_ptr(s, r);
        while k < csr_ptr(s, r + 1) {
            if csr_col(s, k) == j { t += csr_val(s, k); }
            k += 1;
        }
        out.push(t);
        r += 1;
    }
    return line_arr(copy(out));
}

// expm_rng_sign gives a sign of one. The state of the generator is a
// reference, so the caller keeps it.
fn expm_rng_sign(state: i64&) -> f64 {
    state = (state * EXPM_RNG_MUL) % EXPM_RNG_MOD;
    if ((state >> 16) & 1) == 1 { return 1.0; }
    return -1.0;
}

// expm_parallel tells whether two columns of a block stand parallel. Every
// value of the block is a sign of one.
fn expm_parallel(x: f64[:], off_a: i64, off_b: i64, n: i64) -> bool {
    var d: f64 = 0.0;
    var i: i64 = 0;
    while i < n { d += x[off_a + i] * x[off_b + i]; i += 1; }
    return d == n as f64;
}

// expm_block_product gives the product of the matrix with one column of the
// block. The column starts at off. The product starts at poff.
fn expm_block_product(s, x: f64[:], off: i64, n: i64, out: f64[>..]&, poff: i64) {
    var r: i64 = 0;
    while r < n {
        var t: f64 = 0.0;
        var k = csr_ptr(s, r);
        while k < csr_ptr(s, r + 1) { t += csr_val(s, k) * x[off + csr_col(s, k)]; k += 1; }
        out[poff + r] = t;
        r += 1;
    }
}

// expm_trans_product gives the product of the transpose of the matrix with
// one column of the block. The column starts at off. The product starts at
// poff.
fn expm_trans_product(s, x: f64[:], off: i64, n: i64, out: f64[>..]&, poff: i64) {
    var i: i64 = 0;
    while i < n { out[poff + i] = 0.0; i += 1; }
    var r: i64 = 0;
    while r < n {
        let xv = x[off + r];
        if xv != 0.0 {
            var k = csr_ptr(s, r);
            while k < csr_ptr(s, r + 1) {
                out[poff + csr_col(s, k)] += csr_val(s, k) * xv;
                k += 1;
            }
        }
        r += 1;
    }
}

// onenormest_exact gives the exact one-norm of the matrix. It gives the
// unit vector of the largest column and the column of the matrix.
fn onenormest_exact(s) -> f64, Arr, Arr {
    let n = s.rows;
    let ca, dg, da = expm_sums(s);
    var best: i64 = 0;
    var i: i64 = 1;
    while i < n { if ca.data[i] > ca.data[best] { best = i; } i += 1; }
    var v: f64[>..] = [];
    i = 0;
    while i < n { v.push(0.0); i += 1; }
    v[best] = 1.0;
    return ca.data[best], line_arr(copy(v)), expm_column_of(s, best);
}

// onenormest_block gives the estimate of the one-norm with the block
// algorithm of Higham and Tisseur. The block holds its columns in one flat
// buffer of n * t values.
fn onenormest_block(s, t: i64, itmax: i64) -> f64, Arr, Arr {
    let n = s.rows;
    let nt = n * t;
    var x: f64[>..] = [];
    var y: f64[>..] = [];
    var sg: f64[>..] = [];
    var z: f64[>..] = [];
    var h: f64[>..] = [];
    var bv: f64[>..] = [];
    var bw: f64[>..] = [];
    var used: i64[>..] = [];
    var i: i64 = 0;
    while i < nt { x.push(0.0); y.push(0.0); sg.push(0.0); z.push(0.0); i += 1; }
    i = 0;
    while i < n { h.push(0.0); bv.push(0.0); bw.push(0.0); used.push(0); i += 1; }
    // The first column of the block holds ones. The other columns hold a
    // sign of one. A column that stands parallel to another one is drawn
    // again.
    var state: i64 = EXPM_RNG_SEED;
    var c: i64 = 0;
    while c < t {
        var tries: i64 = 0;
        var ok: bool = false;
        while !ok && tries < 100 {
            var r: i64 = 0;
            while r < n {
                if c == 0 { x[r] = 1.0; } else { x[c * n + r] = expm_rng_sign(state); }
                r += 1;
            }
            // The candidate is compared with the columns that stand
            // before it in the block.
            ok = true;
            var p: i64 = 0;
            while ok && p < c {
                if expm_parallel(x, c * n, p * n, n) { ok = false; }
                p += 1;
            }
            tries += 1;
        }
        c += 1;
    }
    // Every column of the block has a one-norm of one.
    i = 0;
    while i < nt { x[i] = x[i] / n as f64; i += 1; }
    var est: f64 = 0.0;
    var est_old: f64 = 0.0;
    var k: i64 = 1;
    var stop: bool = false;
    while !stop && k <= itmax {
        // The product of the matrix with the block. Each column of the
        // block costs one product.
        c = 0;
        while c < t {
            expm_block_product(s, x, c * n, n, y, c * n);
            c += 1;
        }
        // The estimate is the largest of the sums of the absolute values of
        // the columns of the product.
        var e: f64 = 0.0;
        var bj: i64 = 0;
        c = 0;
        while c < t {
            var m: f64 = 0.0;
            var r: i64 = 0;
            while r < n { m += mathshim::abs_of(y[c * n + r]); r += 1; }
            if m > e { e = m; bj = c; }
            c += 1;
        }
        if e > est_old || k == 1 {
            est = e;
            var r: i64 = 0;
            while r < n { bv[r] = x[bj * n + r]; bw[r] = y[bj * n + r]; r += 1; }
        }
        if k >= 2 && e <= est_old { est = est_old; stop = true; }
        est_old = e;
        if !stop && k == itmax { stop = true; }
        if !stop {
            // The sign of the product. A zero rounds up to one.
            c = 0;
            while c < t {
                var r: i64 = 0;
                while r < n {
                    let v = y[c * n + r];
                    var sv: f64 = 1.0;
                    if v != 0.0 { sv = v / mathshim::abs_of(v); }
                    sg[c * n + r] = sv;
                    r += 1;
                }
                c += 1;
            }
            // Every column of the sign block that stands parallel to another
            // one ends the algorithm.
            var allpar: bool = true;
            c = 0;
            while allpar && c < t {
                var has: bool = false;
                var p: i64 = 0;
                while !has && p < t {
                    if p != c && expm_parallel(sg, c * n, p * n, n) { has = true; }
                    p += 1;
                }
                if !has { allpar = false; }
                c += 1;
            }
            if allpar { stop = true; }
        }
        if !stop {
            // The product of the transpose gives the order of the rows of
            // the next block.
            c = 0;
            while c < t {
                expm_trans_product(s, sg, c * n, n, z, c * n);
                c += 1;
            }
            var r: i64 = 0;
            while r < n {
                var m: f64 = 0.0;
                c = 0;
                while c < t {
                    let a = mathshim::abs_of(z[c * n + r]);
                    if a > m { m = a; }
                    c += 1;
                }
                h[r] = m;
                r += 1;
            }
            // The columns of the next block are the unit vectors of the
            // rows with the largest values.
            i = 0;
            while i < n { used[i] = 0; i += 1; }
            var slot: i64 = 0;
            while slot < t {
                var pick: i64 = -1;
                var besth: f64 = -1.0;
                r = 0;
                while r < n {
                    if used[r] == 0 && h[r] > besth { besth = h[r]; pick = r; }
                    r += 1;
                }
                if pick < 0 { break; }
                used[pick] = 1;
                var q: i64 = 0;
                while q < n { x[slot * n + q] = 0.0; q += 1; }
                x[slot * n + pick] = 1.0;
                slot += 1;
            }
            if slot < t { stop = true; }
        }
        k += 1;
    }
    return est, line_arr(copy(bv)), line_arr(copy(bw));
}

// onenormest_pick keeps each vector only when the caller asks for it.
fn onenormest_pick(e: f64, v, w, compute_v: bool, compute_w: bool) -> f64, Arr, Arr, Fault {
    if compute_v && compute_w { return e, clone(v), clone(w), no_fault(); }
    if compute_v { return e, clone(v), empty_arr(), no_fault(); }
    if compute_w { return e, empty_arr(), clone(w), no_fault(); }
    return e, empty_arr(), empty_arr(), no_fault();
}

// onenormest gives a lower bound of the one-norm of a matrix of compressed
// rows. It gives the estimate, the vector that reaches it and the product
// of the matrix with that vector. The two vectors stay empty unless the
// caller asks for them.
//
// A side of four or less gives the exact one-norm. A count of the columns
// of the block at or above the side gives it too. The other matrices run
// the block algorithm.
fn onenormest(s, t: i64 = 2, itmax: i64 = 5, compute_v: bool = false, compute_w: bool = false) -> f64, Arr, Arr, Fault {
    let f = expm_matrix_check(s, "onenormest");
    if f.kind != F_NONE { return 0.0, empty_arr(), empty_arr(), f; }
    if t < 1 { return 0.0, empty_arr(), empty_arr(), fault(F_RANGE, "onenormest", "the count of the columns of the block stands below one"); }
    if itmax < 1 || itmax > 100 { return 0.0, empty_arr(), empty_arr(), fault(F_RANGE, "onenormest", "the count of the rounds stands outside one to 100"); }
    if t > MAX_SPARSE_SIDE { return 0.0, empty_arr(), empty_arr(), fault(F_LIMIT, "onenormest", "the count of the columns of the block stands above the limit"); }
    let n = s.rows;
    if t >= n || n <= EXPM_EXACT_SIDE {
        let e, vv, ww = onenormest_exact(s);
        if mathshim::is_finite_of(e) == 0.0 { return 0.0, empty_arr(), empty_arr(), fault(F_VALUE, "onenormest", "the estimate is not finite"); }
        return onenormest_pick(e, vv, ww, compute_v, compute_w);
    }
    if n > MAX_ELEMENTS / t { return 0.0, empty_arr(), empty_arr(), fault(F_LIMIT, "onenormest", "the count of the elements of the block stands above the limit"); }
    let e, vv, ww = onenormest_block(s, t, itmax);
    if mathshim::is_finite_of(e) == 0.0 { return 0.0, empty_arr(), empty_arr(), fault(F_VALUE, "onenormest", "the estimate is not finite"); }
    return onenormest_pick(e, vv, ww, compute_v, compute_w);
}
// expm_shift gives the shift of the exponential and the one-norm of the
// shifted matrix. The shift is the mean of the diagonal. The shifted matrix
// is A - mu I.
fn expm_shift(s) -> f64, f64 {
    let n = s.rows;
    let ca, dg, da = expm_sums(s);
    var mu: f64 = 0.0;
    var i: i64 = 0;
    while i < n { mu += dg.data[i]; i += 1; }
    mu = mu / n as f64;
    var norm: f64 = 0.0;
    i = 0;
    while i < n {
        let c = ca.data[i] - da.data[i] + mathshim::abs_of(dg.data[i] - mu);
        if c > norm { norm = c; }
        i += 1;
    }
    return mu, norm;
}

// expm_scale_of gives the scale of the series at the time t. The scale is
// the one-norm of the shifted matrix at that time.
fn expm_scale_of(norm: f64, t: f64, tag: const u8[:]) -> i64, Fault {
    let sc = mathshim::abs_of(t) * norm;
    if sc > MAX_EXPM_SCALE as f64 { return 0, fault(F_LIMIT, tag, "the scale of the exponential stands above the limit"); }
    var scale: i64 = 1;
    if sc > 1.0 { scale = mathshim::ceil_of(sc) as i64; }
    return scale, no_fault();
}

// expm_apply gives the product of the shifted matrix with a vector. The
// shifted matrix is A - mu I.
fn expm_apply(s, mu: f64, x: f64[:], out: f64[>..]&, n: i64) {
    var r: i64 = 0;
    while r < n {
        var t: f64 = 0.0;
        var k = csr_ptr(s, r);
        while k < csr_ptr(s, r + 1) { t += csr_val(s, k) * x[csr_col(s, k)]; k += 1; }
        out[r] = t - mu * x[r];
        r += 1;
    }
}

// expm_series gives the product of the series of the shifted matrix with a
// vector. The series has the order MAX_EXPM_TERMS. The scale divides the
// shifted matrix. The time multiplies it.
fn expm_series(s, mu: f64, t: f64, sc: f64, x: f64[:], out: f64[>..]&, term: f64[>..]&, tmp: f64[>..]&) {
    let n = s.rows;
    var i: i64 = 0;
    while i < n { term[i] = x[i]; out[i] = x[i]; i += 1; }
    var k: i64 = 1;
    while k <= MAX_EXPM_TERMS {
        expm_apply(s, mu, term, tmp, n);
        i = 0;
        let c = t / (sc * k as f64);
        while i < n { term[i] = c * tmp[i]; out[i] += term[i]; i += 1; }
        k += 1;
    }
}

// expm_action gives the product of the matrix exponential of the matrix at
// the time t with one column of the right side. The column holds n values.
fn expm_action(s, mu: f64, scale: i64, t: f64, col: f64[:]) -> Arr {
    let n = s.rows;
    var cur: f64[>..] = [];
    var nxt: f64[>..] = [];
    var term: f64[>..] = [];
    var tmp: f64[>..] = [];
    var i: i64 = 0;
    while i < n { cur.push(col[i]); nxt.push(0.0); term.push(0.0); tmp.push(0.0); i += 1; }
    // The series of the shifted matrix runs once for each step of the
    // scale. The work only uses products of the matrix with a vector, so
    // the matrix stays sparse.
    var iter: i64 = 0;
    while iter < scale {
        expm_series(s, mu, t, scale as f64, cur, nxt, term, tmp);
        i = 0;
        while i < n { cur[i] = nxt[i]; i += 1; }
        iter += 1;
    }
    let fac = mathshim::power_e(t * mu);
    var res: f64[>..] = [];
    i = 0;
    while i < n { res.push(fac * cur[i]); i += 1; }
    return line_arr(copy(res));
}

// expm_columns gives the product of the matrix exponential at the time t
// with every column of the right side. The columns stand one after the
// other, each with n values.
fn expm_columns(s, mu: f64, scale: i64, t: f64, b, n: i64, k: i64) -> Arr {
    var out: f64[>..] = [];
    var c: i64 = 0;
    while c < k {
        var col: f64[>..] = [];
        var r: i64 = 0;
        // The array of the package holds its elements in row-major order.
        while r < n { col.push(b.data[r * k + c]); r += 1; }
        let res = expm_action(s, mu, scale, t, col);
        r = 0;
        while r < n { out.push(res.data[r]); r += 1; }
        c += 1;
    }
    return line_arr(copy(out));
}

// expm_rows gives the rows of the columns. The columns stand one after the
// other. Every row holds one value for each column.
fn expm_rows(cols: f64[:], n: i64, k: i64) -> Arr {
    var out: f64[>..] = [];
    var r: i64 = 0;
    while r < n {
        var c: i64 = 0;
        while c < k { out.push(cols[c * n + r]); c += 1; }
        r += 1;
    }
    return line_arr(copy(out));
}

// expm_multiply gives the product of the matrix exponential of the matrix
// with the right side. The right side holds one column or many columns. The
// result holds the same shape.
fn expm_multiply(s, b) -> Arr, Fault {
    let f = expm_matrix_check(s, "expm_multiply");
    if f.kind != F_NONE { return empty_arr(), f; }
    let g = expm_side_check(s, b, "expm_multiply");
    if g.kind != F_NONE { return empty_arr(), g; }
    let mu, norm = expm_shift(s);
    let scale, h = expm_scale_of(norm, 1.0, "expm_multiply");
    if h.kind != F_NONE { return empty_arr(), h; }
    let n = s.rows;
    var k: i64 = 1;
    if b.ndim == 2 { k = b.shape[1]; }
    let cols = expm_columns(s, mu, scale, 1.0, b, n, k);
    var i: i64 = 0;
    while i < cols.data.len {
        if mathshim::is_finite_of(cols.data[i]) == 0.0 { return empty_arr(), fault(F_VALUE, "expm_multiply", "a value of the result is not finite"); }
        i += 1;
    }
    if b.ndim == 1 { return line_arr(copy(cols.data)), no_fault(); }
    let rows = expm_rows(cols.data, n, k);
    return mat2(copy(rows.data), n, k), no_fault();
}

// expm_multiply_range gives the product of the matrix exponential of the
// matrix with the right side at each of the times. The times follow the
// rule of numpy.linspace. The result holds one row for each time. A right
// side of many columns gives a block for each time.
fn expm_multiply_range(s, b, start: f64, stop: f64, num: i64, endpoint: bool) -> Arr, Fault {
    let f = expm_matrix_check(s, "expm_multiply_range");
    if f.kind != F_NONE { return empty_arr(), f; }
    let g = expm_side_check(s, b, "expm_multiply_range");
    if g.kind != F_NONE { return empty_arr(), g; }
    if mathshim::is_finite_of(start) == 0.0 || mathshim::is_finite_of(stop) == 0.0 { return empty_arr(), fault(F_VALUE, "expm_multiply_range", "the start or the stop is not finite"); }
    if start > stop { return empty_arr(), fault(F_RANGE, "expm_multiply_range", "the start passes the stop"); }
    if num < 2 { return empty_arr(), fault(F_RANGE, "expm_multiply_range", "the count of the times stands below two"); }
    if num > MAX_EXPM_STEPS { return empty_arr(), fault(F_LIMIT, "expm_multiply_range", "the count of the times stands above the limit"); }
    let n = s.rows;
    var k: i64 = 1;
    if b.ndim == 2 { k = b.shape[1]; }
    let nk = n * k;
    if num > MAX_ELEMENTS / nk { return empty_arr(), fault(F_LIMIT, "expm_multiply_range", "the count of the elements of the result stands above the limit"); }
    let mu, norm = expm_shift(s);
    // The times follow the rule of numpy.linspace.
    var step: f64 = 0.0;
    if endpoint { step = (stop - start) / (num - 1) as f64; } else { step = (stop - start) / num as f64; }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < num {
        var t = start + step * i as f64;
        if endpoint && i == num - 1 { t = stop; }
        let scale, h = expm_scale_of(norm, t, "expm_multiply_range");
        if h.kind != F_NONE { return empty_arr(), h; }
        let cols = expm_columns(s, mu, scale, t, b, n, k);
        let rows = expm_rows(cols.data, n, k);
        var r: i64 = 0;
        while r < rows.data.len {
            if mathshim::is_finite_of(rows.data[r]) == 0.0 { return empty_arr(), fault(F_VALUE, "expm_multiply_range", "a value of the result is not finite"); }
            out.push(rows.data[r]);
            r += 1;
        }
        i += 1;
    }
    if b.ndim == 1 { return mat2(copy(out), num, n), no_fault(); }
    return arr_of(copy(out), [num, n, k]), no_fault();
}
