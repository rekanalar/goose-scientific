// The banded matrix: the ranks, the residual and the tridiagonal eigenvalue solver.
//
// The tridiagonal eigenvalue solver, the banded Cholesky and the RQ
// factorization of scipy.linalg: eigvalsh_tridiagonal,
// eigh_tridiagonal, cholesky_banded, cho_solve_banded, solveh_banded
// and rq. Every loop is bounded by the sizes of the operands.
namespace scientific;

import mathshim;
import array_decomp;

// bd_pivot gives the smallest pivot that the inverse iteration
// accepts. The size of the matrix scales it.
fn bd_pivot(d: f64[:], e: f64[:]) -> f64 {
    var big: f64 = 0.0;
    var i: i64 = 0;
    while i < d.len {
        var v = d[i];
        if v < 0.0 { v = -v; }
        if v > big { big = v; }
        i += 1;
    }
    i = 0;
    while i < e.len {
        var v = e[i];
        if v < 0.0 { v = -v; }
        if v > big { big = v; }
        i += 1;
    }
    if big == 0.0 { big = 1.0; }
    return big * 2.220446049250313e-16;
}

// The largest size of a tridiagonal matrix. The cost of the bisection
// grows with the square of the size, so that an input cannot ask for a
// result that takes too long.
const BD_MAX_SIDE: i64 = 4096;

// bd_check holds the shape and the finite checks of the two diagonals.
// The off diagonal holds one value fewer than the diagonal.
fn bd_check(d: f64[:], e: f64[:], tag: const u8[:]) -> Fault {
    if d.len < 1 {
        return fault(F_SHAPE, tag, "the diagonal holds no value");
    }
    if d.len > BD_MAX_SIDE {
        return fault(F_LIMIT, tag, "the diagonal holds more values than the limit");
    }
    if e.len != d.len - 1 {
        return fault(F_SHAPE, tag, "the off diagonal must hold one value fewer than the diagonal");
    }
    var i: i64 = 0;
    while i < d.len {
        if !(d[i] - d[i] == 0.0) { return fault(F_VALUE, tag, "the diagonal holds a value that is not finite"); }
        i += 1;
    }
    i = 0;
    while i < e.len {
        if !(e[i] - e[i] == 0.0) { return fault(F_VALUE, tag, "the off diagonal holds a value that is not finite"); }
        i += 1;
    }
    return no_fault();
}

// bd_count gives the count of the eigenvalues below x through the
// Sturm sequence. The sequence holds the leading minors of the matrix
// of T minus x on the diagonal. A zero term sits out of the count of
// the sign changes, so that a value on a pole still counts the values
// below it. A zero off diagonal value ends a block, and the count of
// each block adds up. A matrix that falls into blocks has a leading
// minor of zero, and the sequence of the whole matrix then holds no
// sign change at all.
fn bd_count(x: f64, d: f64[:], e: f64[:], n: i64) -> i64 {
    var cnt: i64 = 0;
    var a: i64 = 0;
    while a < n {
        var b: i64 = a;
        while b < n - 1 && e[b] != 0.0 { b += 1; }
        cnt += bd_count_block(x, d[a..b + 1], e[a..b], b - a + 1);
        a = b + 1;
    }
    return cnt;
}

// bd_count_block gives the count of the eigenvalues below x of a block
// that has no zero off diagonal value. The Sturm sequence holds the
// leading minors of the matrix of T minus x on the diagonal. A zero term
// sits out of the count of the sign changes, so that a value on a pole
// still counts the values below it.
fn bd_count_block(x: f64, d: f64[:], e: f64[:], n: i64) -> i64 {
    var m1: f64 = 1.0;
    var m2: f64 = 0.0;
    var sign: i64 = 1;
    var cnt: i64 = 0;
    var i: i64 = 0;
    while i < n {
        var a = m1;
        if a > 1.0e150 || a < -1.0e150 {
            a = a / 1.0e150;
            m1 = m1 / 1.0e150;
            m2 = m2 / 1.0e150;
        }
        var t = (d[i] - x) * a;
        if i > 0 { t -= e[i - 1] * e[i - 1] * m2; }
        if t != 0.0 {
            var s: i64 = 1;
            if t < 0.0 { s = -1; }
            if s != sign { cnt += 1; sign = s; }
        }
        m2 = m1;
        m1 = t;
        i += 1;
    }
    return cnt;
}

// bd_rank bisects for the eigenvalue of the rank k, where the ranks
// start at zero and the smallest value holds rank zero. The bounds of
// Gershgorin give the search range.
fn bd_rank(d: f64[:], e: f64[:], n: i64, k: i64) -> f64 {
    var lo: f64 = 0.0;
    var hi: f64 = 0.0;
    var first = true;
    var i: i64 = 0;
    while i < n {
        var rad: f64 = 0.0;
        if i > 0 { rad += mathshim::abs_of(e[i - 1]); }
        if i < n - 1 { rad += mathshim::abs_of(e[i]); }
        if first || d[i] - rad < lo { lo = d[i] - rad; }
        if first || d[i] + rad > hi { hi = d[i] + rad; }
        first = false;
        i += 1;
    }
    var big = mathshim::abs_of(lo);
    let b2 = mathshim::abs_of(hi);
    if b2 > big { big = b2; }
    if big < 1.0 { big = 1.0; }
    while hi - lo > 4.0 * 2.220446049250313e-16 * big {
        let mid = (lo + hi) / 2.0;
        if bd_count(mid, d, e, n) > k { hi = mid; } else { lo = mid; }
    }
    return (lo + hi) / 2.0;
}

// bd_residual gives the size of the difference between the matrix
// times the vector and the value times the vector.
fn bd_residual(d: f64[:], e: f64[:], n: i64, lam: f64, v: f64[:]) -> f64 {
    var big: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        var acc = d[i] * v[i];
        if i > 0 { acc += e[i - 1] * v[i - 1]; }
        if i < n - 1 { acc += e[i] * v[i + 1]; }
        var t = acc - lam * v[i];
        if t < 0.0 { t = -t; }
        if t > big { big = t; }
        i += 1;
    }
    return big;
}

// bd_inverse runs the inverse iteration on the tridiagonal matrix for
// the shift lam. A pivot of a small size takes the place of a zero
// pivot. The pattern picks the start vector, so that a vector that
// holds no part of the eigenvector does not win. The result holds the
// size of the residual.
fn bd_inverse(d: f64[:], e: f64[:], n: i64, lam: f64, piv: f64, pattern: i64, out: f64[>..]&) -> f64 {
    var mu: f64[>..] = [];
    var mlt: f64[>..] = [];
    var v = d[0] - lam;
    if mathshim::abs_of(v) < piv { v = -piv; }
    mu.push(v);
    var i: i64 = 1;
    while i < n {
        let m2 = e[i - 1] / mu[i - 1];
        mlt.push(m2);
        v = d[i] - lam - m2 * e[i - 1];
        if mathshim::abs_of(v) < piv { v = -piv; }
        mu.push(v);
        i += 1;
    }
    i = 0;
    while i < n {
        if pattern == 0 { out.push(1.0); }
        else if pattern == 1 {
            if i % 2 == 0 { out.push(1.0); } else { out.push(-1.0); }
        } else { out.push((i + 1) * 1.0); }
        i += 1;
    }
    // The forward step of the elimination.
    i = 1;
    while i < n {
        out[i] = out[i] - mlt[i - 1] * out[i - 1];
        i += 1;
    }
    // The back step.
    out[n - 1] = out[n - 1] / mu[n - 1];
    i = n - 2;
    while i >= 0 {
        out[i] = (out[i] - e[i] * out[i + 1]) / mu[i];
        i -= 1;
    }
    // The normalization.
    var norm: f64 = 0.0;
    i = 0;
    while i < n { norm += out[i] * out[i]; i += 1; }
    norm = mathshim::root(norm);
    if norm > 0.0 {
        i = 0;
        while i < n { out[i] = out[i] / norm; i += 1; }
    }
    return bd_residual(d, e, n, lam, out);
}

// bd_orthogonal takes the part of the vector that the vectors of prev
// do not hold.
fn bd_orthogonal(v: f64[>..]&, prev: f64[:], np: i64, n: i64) {
    var q: i64 = 0;
    while q < np {
        var dot: f64 = 0.0;
        var i: i64 = 0;
        while i < n { dot += prev[q * n + i] * v[i]; i += 1; }
        i = 0;
        while i < n { v[i] -= dot * prev[q * n + i]; i += 1; }
        q += 1;
    }
    var norm: f64 = 0.0;
    var i: i64 = 0;
    while i < n { norm += v[i] * v[i]; i += 1; }
    norm = mathshim::root(norm);
    if norm > 0.0 {
        i = 0;
        while i < n { v[i] = v[i] / norm; i += 1; }
    }
}

// bd_vector gives the eigenvector of the value lam. The inverse
// iteration runs with several start vectors, and the result with the
// smallest residual wins. The vectors of prev hold the earlier results
// of the same matrix, and the turn takes their part out.
fn bd_vector(d: f64[:], e: f64[:], n: i64, lam: f64, piv: f64, prev: f64[:], np: i64, out: f64[>..]&) -> f64 {
    var best: f64[>..] = [];
    var bestres: f64 = 1.0e30;
    var pattern: i64 = 0;
    while pattern < 4 {
        var trial: f64[>..] = [];
        bd_inverse(d, e, n, lam, piv, pattern, trial);
        bd_orthogonal(trial, prev, np, n);
        let res = bd_residual(d, e, n, lam, trial);
        if res < bestres {
            best = copy(trial);
            bestres = res;
        }
        if res <= 1.0e-11 { break; }
        pattern += 1;
    }
    var i: i64 = 0;
    while i < n { out.push(best[i]); i += 1; }
    return bestres;
}

// bd_select_range checks the range of the select value. The ranks run
// from zero to n minus one, and the value range holds the lower bound
// and the upper bound.
fn bd_select_range(select: const u8[:], sr: f64[:], n: i64, tag: const u8[:]) -> Fault {
    if select == "a" { return no_fault(); }
    if sr.len != 2 {
        return fault(F_VALUE, tag, "the select range holds not two values");
    }
    if !(sr[0] - sr[0] == 0.0) || !(sr[1] - sr[1] == 0.0) {
        return fault(F_VALUE, tag, "the select range holds a value that is not finite");
    }
    if select == "i" {
        if sr[0] < 0.0 || sr[1] > n * 1.0 - 1.0 || sr[0] > sr[1] {
            return fault(F_VALUE, tag, "the ranks sit outside the matrix");
        }
        if sr[0] != mathshim::floor_of(sr[0]) || sr[1] != mathshim::floor_of(sr[1]) {
            return fault(F_VALUE, tag, "a rank holds a value that is not a whole number");
        }
    } else if select == "v" {
        if sr[0] >= sr[1] {
            return fault(F_VALUE, tag, "the lower value sits at or above the upper value");
        }
    } else {
        return fault(F_VALUE, tag, "the select is not a, i or v");
    }
    return no_fault();
}

// bd_first_ranks gives the first and the last rank of the select. The
// ranks of a value range count the values below the lower bound and
// the values below the upper bound.
fn bd_first_ranks(d: f64[:], e: f64[:], n: i64, select: const u8[:], sr: f64[:]) -> i64, i64 {
    if select == "i" {
        return sr[0] as i64, sr[1] as i64;
    }
    let lo = bd_count(sr[0], d, e, n);
    let hi = bd_count(sr[1], d, e, n);
    return lo, hi - 1;
}

// eigvalsh_tridiagonal gives the eigenvalues of the symmetric
// tridiagonal matrix of the diagonal d and the off diagonal e, in an
// ascending order. The select value a takes all, i takes a range of
// ranks from one to n, and v takes the values inside the range.
fn eigvalsh_tridiagonal(d: f64[:], e: f64[:], select: const u8[:] = "a", select_range: f64[:] = []) -> Arr, Fault {
    let fc = bd_check(d, e, "eigvalsh_tridiagonal");
    if fc.kind != F_NONE { return empty_arr(), fc; }
    let n = d.len;
    let fs = bd_select_range(select, select_range, n, "eigvalsh_tridiagonal");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    var k1: i64 = 0;
    var k2: i64 = n - 1;
    if select != "a" {
        let kk1, kk2 = bd_first_ranks(d, e, n, select, select_range);
        k1 = kk1;
        k2 = kk2;
    }
    var out: f64[>..] = [];
    var k = k1;
    while k <= k2 {
        out.push(bd_rank(d, e, n, k));
        k += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// eigh_tridiagonal gives the eigenvalues and the eigenvectors of the
// symmetric tridiagonal matrix of the diagonal d and the off diagonal
// e. The flag eigvals_only holds back the vectors. The values come
// from the bisection and the vectors from the inverse iteration. The
// vectors hold no fixed sign.
fn eigh_tridiagonal(d: f64[:], e: f64[:], eigvals_only: bool = false, select: const u8[:] = "a", select_range: f64[:] = []) -> Arr, Arr, Fault {
    let fc = bd_check(d, e, "eigh_tridiagonal");
    if fc.kind != F_NONE { return empty_arr(), empty_arr(), fc; }
    let n = d.len;
    let fs = bd_select_range(select, select_range, n, "eigh_tridiagonal");
    if fs.kind != F_NONE { return empty_arr(), empty_arr(), fs; }
    if n * n > MAX_ELEMENTS {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "eigh_tridiagonal", "the vectors hold more elements than the limit");
    }
    let piv = bd_pivot(d, e);
    var k1: i64 = 0;
    var k2: i64 = n - 1;
    if select != "a" {
        let kk1, kk2 = bd_first_ranks(d, e, n, select, select_range);
        k1 = kk1;
        k2 = kk2;
    }
    // A matrix with a negligible off diagonal value falls into blocks.
    if bd_has_split(d, e, n) {
        let sw, sz = bd_eigh_split(d, e, n, k1, k2, !eigvals_only);
        if eigvals_only { return line_arr(copy(sw)), empty_arr(), no_fault(); }
        return line_arr(copy(sw)), mat2(copy(sz), n, k2 - k1 + 1), no_fault();
    }
    var w: f64[>..] = [];
    var z: f64[>..] = [];
    var prev: f64[>..] = [];
    var np: i64 = 0;
    var k = k1;
    while k <= k2 {
        let lam = bd_rank(d, e, n, k);
        w.push(lam);
        if !eigvals_only {
            var vec: f64[>..] = [];
            bd_vector(d, e, n, lam, piv, prev, np, vec);
            var i: i64 = 0;
            while i < n { prev.push(vec[i]); i += 1; }
            np += 1;
        }
        k += 1;
    }
    if eigvals_only { return line_arr(copy(w)), empty_arr(), no_fault(); }
    // The vectors sit in the columns of the result, and the rows of
    // the buffer hold the values in the row-major order.
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < np { z.push(prev[j * n + i]); j += 1; }
        i += 1;
    }
    if n == 1 { return line_arr(copy(w)), line_arr(copy(z)), no_fault(); }
    return line_arr(copy(w)), mat2(copy(z), n, np), no_fault();
}

