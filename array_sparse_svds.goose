// The partial singular value decomposition of a sparse matrix: svds.
//
// svds gives the k largest or the k smallest singular values of a matrix of
// compressed rows, with the vectors. It follows scipy.sparse.linalg.svds.
//
// The call never builds the dense matrix and never builds the normal
// matrix. It uses the matrix only through its product with a vector and its
// product with the transpose. The iteration is the Lanczos method with full
// reorthogonalization on the operator of the short side. The values come
// from the Ritz values of the tridiagonal matrix.
//
// The vectors of the short side are Ritz vectors. The vectors of the long
// side are the products of the matrix with them, orthogonalized again. A
// singular value that is small against the largest one has a Ritz residual
// that the square of the operator magnifies. The vectors of such a value
// are accurate to about 1.0e-14 times the largest value divided by the value.
//
// The package takes no solver choice, no random generator, no options and no
// linear operator. The path of the iteration may differ from ARPACK. The
// values agree when the tolerance is tight, and the vectors agree up to sign.
namespace scientific;

import mathshim;

// The choice of the result. Both sides, the values only, the left side only
// or the right side only.
const SVDS_BOTH: i64 = 0;
const SVDS_VALUES: i64 = 1;
const SVDS_LEFT: i64 = 2;
const SVDS_RIGHT: i64 = 3;

// The largest Krylov space. The tridiagonal solver then runs on a matrix of
// at most this side.
const MAX_SVDS_SPACE: i64 = 512;

// The count of rebuilds when maxiter is zero.
const SVDS_DEFAULT_REBUILDS: i64 = 64;

// A value at or below this share of the largest one takes its vector of the
// long side from the completion. A breakdown is a norm at or below this
// share of the scale of the matrix.
const SVDS_DIRECT_SHARE: f64 = 1.0e-10;
const SVDS_BREAKDOWN: f64 = 1.0e-12;

// The least norm that a vector must keep after the orthogonalization.
const SVDS_KEEP: f64 = 0.5;

// svds_empty is the result of every fault.
fn svds_empty(f: Fault) -> Arr, Arr, Arr, Fault {
    return empty_arr(), empty_arr(), empty_arr(), f;
}

// svds_matrix_check checks the matrix. It has no square test. Every stored
// value must be finite.
fn svds_matrix_check(s, tag: const u8[:]) -> Fault {
    let f = csr_check(s.rows, s.cols, tag);
    if f.kind != F_NONE { return f; }
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

// svds_options_check checks the counts, the tolerance, the choice, the start
// vector and the mode. The checks run in the order of the contract.
fn svds_options_check(d: i64, k: i64, which: const u8[:], tol: f64, v0: f64[:], ncv: i64, maxiter: i64, mode: i64, tag: const u8[:]) -> Fault {
    if k < 1 || k >= d { return fault(F_RANGE, tag, "k stands outside one to the smaller side minus one"); }
    if k >= MAX_SVDS_SPACE { return fault(F_LIMIT, tag, "k stands at or above the limit of the space"); }
    if ncv < 0 || (ncv > 0 && (ncv <= k || ncv >= d)) {
        return fault(F_RANGE, tag, "ncv stands outside k + 1 to the smaller side minus one");
    }
    if ncv > MAX_SVDS_SPACE { return fault(F_LIMIT, tag, "ncv stands above the limit of the space"); }
    if maxiter < 0 { return fault(F_RANGE, tag, "the count of rebuilds is below zero"); }
    if maxiter > MAX_KRYLOV_ITER { return fault(F_LIMIT, tag, "the count of rebuilds stands above the limit"); }
    if !(tol >= 0.0) || tol - tol != 0.0 { return fault(F_VALUE, tag, "the tolerance is negative or not finite"); }
    if which != "LM" && which != "SM" { return fault(F_VALUE, tag, "which is neither LM nor SM"); }
    if v0.len != 0 {
        if v0.len != d { return fault(F_SHAPE, tag, "the start vector does not hold one value per side of the short side"); }
        if !lg_finite(v0, v0.len) { return fault(F_VALUE, tag, "the start vector holds a value that is not finite"); }
        if lg_norm(v0, 0, v0.len) == 0.0 { return fault(F_VALUE, tag, "the start vector is zero"); }
    }
    if mode < SVDS_BOTH || mode > SVDS_RIGHT { return fault(F_RANGE, tag, "the mode stands outside zero to three"); }
    return no_fault();
}

// svds_storage_check checks the buffers against MAX_ELEMENTS before any
// buffer grows. The call holds m basis vectors, k vectors of each side and
// six work vectors of the larger side. It also holds the two outputs. Each
// check uses a division, so no product overflows.
fn svds_storage_check(rows: i64, cols: i64, k: i64, m: i64, tag: const u8[:]) -> Fault {
    var big = rows;
    if cols > big { big = cols; }
    if m + 2 * k + 6 > MAX_ELEMENTS / big {
        return fault(F_LIMIT, tag, "the vectors of the call stand above the element limit");
    }
    if k > MAX_ELEMENTS / rows || k > MAX_ELEMENTS / cols {
        return fault(F_LIMIT, tag, "the vectors of the result stand above the element limit");
    }
    return no_fault();
}

// svds_scale gives the largest absolute stored value.
fn svds_scale(s) -> f64 {
    var c: f64 = 0.0;
    var k: i64 = 0;
    while k < s.nnz {
        let a = mathshim::abs_of(csr_val(s, k));
        if a > c { c = a; }
        k += 1;
    }
    return c;
}

// svds_fixed gives the fixed start vector. The package keeps its own
// generator, so the vector is the same on every run and on every build of
// the compiler. The values stand between -0.5 and 0.5.
fn svds_fixed(n: i64, state: i64&) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        state = (state * EXPM_RNG_MUL) % EXPM_RNG_MOD;
        out.push((state as f64) / (EXPM_RNG_MOD as f64) - 0.5);
        i += 1;
    }
    return out;
}

// svds_op gives the operator of the short side applied to the vector x of d
// values. The matrix is scaled by 1/c. A product that is not finite gives
// F_VALUE. The operator is A^T A for a matrix that is not wide. It is
// A A^T for a wide matrix.
fn svds_op(s, x: f64[:], c: f64, wide: bool) -> Arr, Fault {
    let y, f = svds_mv(s, x, c, wide);
    if f.kind != F_NONE { return empty_arr(), f; }
    let z, f2 = svds_mv(s, y.data, c, !wide);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    return z, no_fault();
}

// svds_mv gives the product of the matrix with the vector y, divided by the
// scale c. The product is the transpose of the matrix when the matrix is
// wide. A value that is not finite gives F_VALUE.
fn svds_mv(s, y: f64[:], c: f64, wide: bool) -> Arr, Fault {
    var xa: Arr = empty_arr();
    var f: Fault = no_fault();
    if wide { xa, f = csr_matvec_t(s, y); } else { xa, f = csr_matvec(s, y); }
    if f.kind != F_NONE { return empty_arr(), f; }
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < xa.data.len { w.push(xa.data[i] / c); i += 1; }
    if !lg_finite(w, w.len) { return empty_arr(), fault(F_VALUE, "svds", "a product is not finite"); }
    return line_arr(copy(w)), no_fault();
}

// svds_orth removes from w its part along the cnt vectors that start at off
// in the buffer b. Each vector holds n values. It runs two passes.
fn svds_orth(w: f64[>..]&, b: f64[:], off: i64, cnt: i64, n: i64) {
    var pass: i64 = 0;
    while pass < 2 {
        var r: i64 = 0;
        while r < cnt {
            let h = lg_dot(b, off + r * n, w, 0, n);
            var i: i64 = 0;
            while i < n { w[i] = w[i] - h * b[off + r * n + i]; i += 1; }
            r += 1;
        }
        pass += 1;
    }
}

// svds_scale_to_one divides the values of w by their largest absolute
// value. The norm of a vector whose values stand near the largest number of
// the format overflows. The scale keeps the norm inside the format.
fn svds_scale_to_one(w: f64[>..]&, n: i64) {
    var mx: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        let a = mathshim::abs_of(w[i]);
        if a > mx { mx = a; }
        i += 1;
    }
    if mx > 0.0 {
        i = 0;
        while i < n { w[i] = w[i] / mx; i += 1; }
    }
}

// svds_unit scales w to norm one. It gives the norm before the scaling.
fn svds_unit(w: f64[>..]&, n: i64) -> f64 {
    let nr = lg_norm(w, 0, n);
    if nr > 0.0 {
        var i: i64 = 0;
        while i < n { w[i] = w[i] / nr; i += 1; }
    }
    return nr;
}

// svds_fresh adds a vector to the basis at a breakdown. The vector comes from
// the generator of the package and stands orthogonal to the cnt vectors that
// the basis holds. It gives false when no such vector is found.
fn svds_fresh(vs: f64[>..]&, d: i64, cnt: i64, rng: i64&) -> bool {
    var tries: i64 = 0;
    while tries < 8 {
        let fresh = svds_fixed(d, rng);
        svds_orth(fresh, vs, 0, cnt, d);
        svds_orth(fresh, vs, 0, cnt, d);
        if svds_unit(fresh, d) > SVDS_BREAKDOWN {
            var i: i64 = 0;
            while i < d { vs.push(fresh[i]); i += 1; }
            return true;
        }
        tries += 1;
    }
    return false;
}

// svds_lanczos builds m Lanczos vectors in vs, with the diagonal al and the
// off diagonal be of the tridiagonal matrix. It gives the last off diagonal
// value, which the convergence test needs. A breakdown goes on with a fresh
// vector of the generator and a zero off diagonal value.
fn svds_lanczos(s, c: f64, wide: bool, d: i64, m: i64, start: f64[:], vs: f64[>..]&, al: f64[>..]&, be: f64[>..]&) -> f64, Fault {
    let first = copy_slice(start);
    svds_scale_to_one(first, d);
    svds_unit(first, d);
    var i: i64 = 0;
    while i < d { vs.push(first[i]); i += 1; }
    var rng: i64 = EXPM_RNG_SEED + 1;
    var last: f64 = 0.0;
    var j: i64 = 0;
    while j < m {
        let x = copy_slice(vs[j * d..j * d + d]);
        let w0, f = svds_op(s, x, c, wide);
        if f.kind != F_NONE { return 0.0, f; }
        var w: f64[>..] = copy(w0.data);
        al.push(lg_dot(vs, j * d, w, 0, d));
        svds_orth(w, vs, 0, j + 1, d);
        let beta = lg_norm(w, 0, d);
        if beta - beta != 0.0 { return 0.0, fault(F_VALUE, "svds", "the iteration gave a value that is not finite"); }
        if j == m - 1 { last = beta; j += 1; continue; }
        if beta > SVDS_BREAKDOWN {
            be.push(beta);
            i = 0;
            while i < d { vs.push(w[i] / beta); i += 1; }
        } else {
            be.push(0.0);
            if !svds_fresh(vs, d, j + 1, rng) { return 0.0, fault(F_VALUE, "svds", "the iteration found no new direction"); }
        }
        j += 1;
    }
    return last, no_fault();
}

// svds_eig gives the eigenvalues w, in rising order, and the eigenvectors z
// of the tridiagonal matrix of the diagonal al and the off diagonal be. The
// vector of the value j stands in the column j of z, which holds m by m
// values. A breakdown of the iteration sets an off diagonal value to zero.
// The matrix then falls into blocks. The solver of the tridiagonal matrix
// gives vectors with a leak across a block of equal values. So each block
// goes to the solver alone, and a block of one value is its own vector.
fn svds_eig(al: f64[:], be: f64[:], m: i64, w: f64[>..]&, z: f64[>..]&) -> Fault {
    // One entry per pair: its value, the start and the length of its block,
    // and the place of its vector in the list vec.
    var val: f64[>..] = [];
    var from: f64[>..] = [];
    var len: f64[>..] = [];
    var off: f64[>..] = [];
    var vec: f64[>..] = [];
    var a: i64 = 0;
    while a < m {
        var b = a;
        while b < m - 1 && be[b] != 0.0 { b += 1; }
        let n = b - a + 1;
        var d: f64[>..] = [];
        var e: f64[>..] = [];
        var q: i64 = 0;
        while q < n { d.push(al[a + q]); q += 1; }
        q = 0;
        while q < n - 1 { e.push(be[a + q]); q += 1; }
        if n == 1 {
            val.push(al[a]);
            from.push(a as f64);
            len.push(1.0);
            off.push(vec.len as f64);
            vec.push(1.0);
        } else {
            let wb, zb, f = eigh_tridiagonal(d, e);
            if f.kind != F_NONE { return f; }
            if wb.data.len != n || zb.data.len != n * n { return fault(F_VALUE, "svds", "the Ritz pairs do not fit the block"); }
            q = 0;
            while q < n {
                val.push(wb.data[q]);
                from.push(a as f64);
                len.push(n as f64);
                off.push(vec.len as f64);
                var r: i64 = 0;
                while r < n { vec.push(zb.data[r * n + q]); r += 1; }
                q += 1;
            }
        }
        a = b + 1;
    }
    // Order the pairs by value with an insertion sort of an index list.
    var idx: f64[>..] = [];
    var i: i64 = 0;
    while i < m { idx.push(i as f64); i += 1; }
    i = 1;
    while i < m {
        let keep = idx[i];
        let kv = val[keep as i64];
        var j: i64 = i - 1;
        while j >= 0 && val[idx[j] as i64] > kv {
            idx[j + 1] = idx[j];
            j -= 1;
        }
        idx[j + 1] = keep;
        i += 1;
    }
    z = nd_zeros(m * m);
    var col: i64 = 0;
    while col < m {
        let src = idx[col] as i64;
        w.push(val[src]);
        let st = from[src] as i64;
        let ln = len[src] as i64;
        let o = off[src] as i64;
        var r2: i64 = 0;
        while r2 < ln { z[(st + r2) * m + col] = vec[o + r2]; r2 += 1; }
        col += 1;
    }
    return no_fault();
}
