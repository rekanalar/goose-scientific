// The null space, the range, the matrix exponential and the banded LU solver.
//
namespace scientific;

import mathshim;

// svd_rank counts the singular values above the tolerance. A negative
// rcond means the machine epsilon times the larger side.
fn svd_rank(s: f64[:], m: i64, n: i64, rcond: f64) -> i64 {
    var rc = rcond;
    if rc < 0.0 {
        var big = m;
        if n > big { big = n; }
        rc = 2.220446049250313e-16 * big;
    }
    var top: f64 = 0.0;
    var i: i64 = 0;
    while i < s.len { if s[i] > top { top = s[i]; } i += 1; }
    var count: i64 = 0;
    i = 0;
    while i < s.len { if s[i] > top * rc { count += 1; } i += 1; }
    return count;
}

// null_space gives an orthonormal basis of the null space of a matrix,
// one vector per column. A matrix of full column rank holds no vector,
// and the call is a fault, because the package holds no empty array.
// The signs of the vectors may differ from SciPy.
fn null_space(a, rcond: f64 = -1.0) -> Arr, Fault {
    let c = check_matrix(a, "null_space");
    if c.kind != F_NONE { return empty_arr(), c; }
    let u, s, vt, f = svd(a);
    if f.kind != F_NONE { return empty_arr(), f; }
    let m = a.shape[0];
    let n = a.shape[1];
    let num = svd_rank(s.data, m, n, rcond);
    if num >= n {
        return empty_arr(), fault(F_VALUE, "null_space", "the matrix holds no null vector");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = num;
        while j < n { d.push(vt.data[j * n + i]); j += 1; }
        i += 1;
    }
    return mat2(copy(d), n, n - num), no_fault();
}

// orth gives an orthonormal basis of the range of a matrix, one vector
// per column. A zero matrix holds no vector, and the call is a fault.
// The signs of the vectors may differ from SciPy.
fn orth(a, rcond: f64 = -1.0) -> Arr, Fault {
    let c = check_matrix(a, "orth");
    if c.kind != F_NONE { return empty_arr(), c; }
    let u, s, vt, f = svd(a);
    if f.kind != F_NONE { return empty_arr(), f; }
    let m = a.shape[0];
    let n = a.shape[1];
    let num = svd_rank(s.data, m, n, rcond);
    if num < 1 {
        return empty_arr(), fault(F_VALUE, "orth", "the matrix holds no range vector");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < m {
        var j: i64 = 0;
        while j < num { d.push(u.data[i * m + j]); j += 1; }
        i += 1;
    }
    return mat2(copy(d), m, num), no_fault();
}

// mat_mul_n multiplies two square matrices of n rows held in buffers.
fn mat_mul_n(a: f64[:], b: f64[:], n: i64) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { d.push(0.0); i += 1; }
    i = 0;
    while i < n {
        var k: i64 = 0;
        while k < n {
            let v = a[i * n + k];
            if v != 0.0 {
                var j: i64 = 0;
                while j < n { d[i * n + j] = d[i * n + j] + v * b[k * n + j]; j += 1; }
            }
            k += 1;
        }
        i += 1;
    }
    return d;
}

// lin3 gives cx x + cy y + cz z for three buffers of one length.
fn lin3(x: f64[:], cx: f64, y: f64[:], cy: f64, z: f64[:], cz: f64) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len { d.push(cx * x[i] + cy * y[i] + cz * z[i]); i += 1; }
    return d;
}

// expm gives the exponential of a square matrix. The method is the
// scaling and squaring of Higham with a Pade approximant of degree 13.
// A side above 256 is a fault, and so is a matrix with a value that is
// not finite.
fn expm(a) -> Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), fault(F_SHAPE, "expm", "the matrix is not square");
    }
    let n = a.shape[0];
    if n > MAX_EXPM_SIDE {
        return empty_arr(), fault(F_LIMIT, "expm", "the side holds a value above the limit");
    }
    // The largest column sum of the sizes.
    var norm: f64 = 0.0;
    var j: i64 = 0;
    while j < n {
        var t: f64 = 0.0;
        var i: i64 = 0;
        while i < n { t += mathshim::abs_of(a.data[i * n + j]); i += 1; }
        if t > norm || t != t { norm = t; }
        j += 1;
    }
    if !(norm < 1.0e300) {
        return empty_arr(), fault(F_VALUE, "expm", "the matrix holds a value that is not finite");
    }
    var s: i64 = 0;
    var scaled = norm;
    while scaled > 5.371920351148152 { scaled = scaled / 2.0; s += 1; }
    var scale: f64 = 1.0;
    var k: i64 = 0;
    while k < s { scale = scale / 2.0; k += 1; }
    var m1: f64[>..] = [];
    k = 0;
    while k < n * n { m1.push(a.data[k] * scale); k += 1; }
    let a2 = mat_mul_n(m1, m1, n);
    let a4 = mat_mul_n(a2, a2, n);
    let a6 = mat_mul_n(a4, a2, n);
    // z = a6 (b13 a6 + b11 a4 + b9 a2) + b7 a6 + b5 a4 + b3 a2 + b1 i
    var z = mat_mul_n(a6, lin3(a6, 1.0, a4, 16380.0, a2, 40840800.0), n);
    let z2 = lin3(a6, 33522128640.0, a4, 10559470521600.0, a2, 1187353796428800.0);
    k = 0;
    while k < n * n { z[k] = z[k] + z2[k]; k += 1; }
    k = 0;
    while k < n { z[k * n + k] = z[k * n + k] + 32382376266240000.0; k += 1; }
    let u = mat_mul_n(m1, z, n);
    // v = a6 (b12 a6 + b10 a4 + b8 a2) + b6 a6 + b4 a4 + b2 a2 + b0 i
    var v = mat_mul_n(a6, lin3(a6, 182.0, a4, 960960.0, a2, 1323241920.0), n);
    let v2 = lin3(a6, 670442572800.0, a4, 129060195264000.0, a2, 7771770303897600.0);
    k = 0;
    while k < n * n { v[k] = v[k] + v2[k]; k += 1; }
    k = 0;
    while k < n { v[k * n + k] = v[k * n + k] + 64764752532480000.0; k += 1; }
    // The result solves (v - u) x = v + u.
    var den: f64[>..] = [];
    var x: f64[>..] = [];
    k = 0;
    while k < n * n { den.push(v[k] - u[k]); x.push(v[k] + u[k]); k += 1; }
    var piv: i64[>..] = [];
    lapack_lu(den, n, n, piv);
    k = 0;
    while k < n {
        if den[k * n + k] == 0.0 {
            return empty_arr(), fault(F_VALUE, "expm", "the approximant is singular");
        }
        k += 1;
    }
    var pf: f64[>..] = [];
    k = 0;
    while k < n { pf.push(piv[k]); k += 1; }
    lu_apply(den, n, pf, x, n, false);
    k = 0;
    while k < s { let sq = mat_mul_n(x, x, n); x = copy(sq); k += 1; }
    return mat2(copy(x), n, n), no_fault();
}

// band_lu factors a band matrix in place with the partial pivoting of
// LAPACK dgbtf2. The buffer w holds 2*kl+ku+1 rows of n columns. The
// rows kl and below hold the band. On exit the rows 0 to kl+ku hold U
// and the rows below hold the multipliers. The row of swaps holds the
// row that swapped with each column. The result is false when a pivot
// is zero.
fn band_lu(w: f64[>..]&, piv: i64[>..]&, n: i64, kl: i64, ku: i64) -> bool {
    let kv = kl + ku;
    var ju: i64 = 0;
    var j: i64 = 0;
    while j < n {
        var km = kl;
        if n - 1 - j < km { km = n - 1 - j; }
        var jp: i64 = 0;
        var best = mathshim::abs_of(w[kv * n + j]);
        var i: i64 = 1;
        while i <= km {
            let v = mathshim::abs_of(w[(kv + i) * n + j]);
            if v > best { best = v; jp = i; }
            i += 1;
        }
        piv.push(j + jp);
        if w[(kv + jp) * n + j] == 0.0 { return false; }
        var top = j + ku + jp;
        if n - 1 < top { top = n - 1; }
        if top > ju { ju = top; }
        if jp != 0 {
            var t: i64 = 0;
            while t <= ju - j {
                let p = (kv + jp - t) * n + j + t;
                let q = (kv - t) * n + j + t;
                let tmp = w[p];
                w[p] = w[q];
                w[q] = tmp;
                t += 1;
            }
        }
        if km > 0 {
            let d = w[kv * n + j];
            i = 1;
            while i <= km { w[(kv + i) * n + j] = w[(kv + i) * n + j] / d; i += 1; }
            var c: i64 = 1;
            while c <= ju - j {
                let y = w[(kv - c) * n + j + c];
                var r: i64 = 1;
                while r <= km {
                    w[(kv + r - c) * n + j + c] -= w[(kv + r) * n + j] * y;
                    r += 1;
                }
                c += 1;
            }
        }
        j += 1;
    }
    return true;
}

// solve_banded solves a system with a banded matrix. The pair (l, u)
// holds the count of the lower and the upper diagonals. The matrix ab
// holds the diagonals in the band form of SciPy. The solve factors the
// band with partial pivoting, as LAPACK dgbsv does, so the cost grows
// with the side and the band. A singular matrix is a fault.
fn solve_banded(l: i64, u: i64, ab, b) -> Arr, Fault {
    if l < 0 || u < 0 || ab.ndim != 2 || ab.shape[0] != l + u + 1 {
        return empty_arr(), fault(F_SHAPE, "solve_banded", "the band does not hold l + u + 1 rows");
    }
    let n = ab.shape[1];
    if b.ndim < 1 || b.ndim > 2 || b.shape[0] != n {
        return empty_arr(), fault(F_SHAPE, "solve_banded", "the right side does not agree with the band");
    }
    var nrhs: i64 = 1;
    if b.ndim == 2 { nrhs = b.shape[1]; }
    let kv = l + u;
    var w: f64[>..] = [];
    var i: i64 = 0;
    while i < l * n { w.push(0.0); i += 1; }
    i = 0;
    while i < (l + u + 1) * n { w.push(ab.data[i]); i += 1; }
    i = 0;
    while i < l * n { w.push(0.0); i += 1; }
    var piv: i64[>..] = [];
    if !band_lu(w, piv, n, l, u) {
        return empty_arr(), fault(F_VALUE, "solve_banded", "the matrix is singular");
    }
    var x = copy_slice(b.data);
    var k: i64 = 0;
    while k < nrhs {
        var j: i64 = 0;
        while j < n - 1 {
            var lm = l;
            if n - 1 - j < lm { lm = n - 1 - j; }
            let p = piv[j];
            if p != j {
                let t = x[p * nrhs + k];
                x[p * nrhs + k] = x[j * nrhs + k];
                x[j * nrhs + k] = t;
            }
            var t: i64 = 1;
            while t <= lm {
                x[(j + t) * nrhs + k] -= w[(kv + t) * n + j] * x[j * nrhs + k];
                t += 1;
            }
            j += 1;
        }
        var r = n - 1;
        while r >= 0 {
            var top = r + kv;
            if n - 1 < top { top = n - 1; }
            var c = r + 1;
            while c <= top {
                x[r * nrhs + k] -= w[(kv + r - c) * n + c] * x[c * nrhs + k];
                c += 1;
            }
            x[r * nrhs + k] = x[r * nrhs + k] / w[kv * n + r];
            r -= 1;
        }
        k += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    if b.ndim == 2 { s.push(nrhs); }
    return Arr { ndim: b.ndim, shape: s, data: copy(x) }, no_fault();
}
