// The structured solvers and the remainder of the linear algebra of
// scipy.linalg: solve_toeplitz, solve_circulant, matmul_toeplitz,
// diagsvd, orthogonal_procrustes, subspace_angles and pinvh. Every
// loop is bounded by the sizes of the operands.
namespace scientific;

import mathshim;
import array_fft;
import array_svd;
import array_linalg;
import array_scilinalg;

// st_row gives the element of the Toeplitz matrix of the first column
// c and the first row r at the row i and the column j.
fn st_row(c: f64[:], r: f64[:], i: i64, j: i64) -> f64 {
    if i >= j { return c[i - j]; }
    if r.len == 0 { return c[j - i]; }
    return r[j - i];
}

// solve_toeplitz solves the Toeplitz system T x = b. The column c
// holds the first column and the row r the first row, whose first
// element stays unread. An empty r means a symmetric matrix. The
// operand b holds one or several columns. The solver builds the
// matrix and runs the dense solve of the package. SciPy runs the
// recursion of Levinson. The values agree. The cost grows with the
// cube of n, where the recursion grows with the square.
fn solve_toeplitz(c: f64[:], b, r: f64[:] = []) -> Arr, Fault {
    let n = c.len;
    if n < 1 { return empty_arr(), fault(F_SHAPE, "solve_toeplitz", "the column holds no value"); }
    if n > MAX_ELEMENTS / n { return empty_arr(), fault(F_LIMIT, "solve_toeplitz", "the matrix holds more elements than the limit"); }
    if b.ndim < 1 || b.ndim > 2 || b.shape[0] != n {
        return empty_arr(), fault(F_SHAPE, "solve_toeplitz", "the right side does not match the column");
    }
    var k: i64 = 1;
    if b.ndim == 2 { k = b.shape[1]; }
    if n * k > MAX_ELEMENTS { return empty_arr(), fault(F_LIMIT, "solve_toeplitz", "the right side holds more elements than the limit"); }
    var i: i64 = 0;
    while i < n {
        if !(c[i] - c[i] == 0.0) { return empty_arr(), fault(F_VALUE, "solve_toeplitz", "the column holds a value that is not finite"); }
        i += 1;
    }
    i = 0;
    while i < r.len {
        if !(r[i] - r[i] == 0.0) { return empty_arr(), fault(F_VALUE, "solve_toeplitz", "the row holds a value that is not finite"); }
        i += 1;
    }
    i = 0;
    while i < n * k {
        if !(b.data[i] - b.data[i] == 0.0) { return empty_arr(), fault(F_VALUE, "solve_toeplitz", "the right side holds a value that is not finite"); }
        i += 1;
    }
    var d: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n { d.push(st_row(c, r, i, j)); j += 1; }
        i += 1;
    }
    let t = mat2(copy(d), n, n);
    return solve(t, b);
}

// solve_circulant solves the circulant system C x = b through the
// transform. The column c holds the first column and b one or several
// columns. A frequency of the transform at or below the cutoff makes
// the call a fault, as the value "raise" of SciPy does.
fn solve_circulant(c: f64[:], b, tol: f64 = -1.0) -> Arr, Fault {
    let n = c.len;
    if n < 1 { return empty_arr(), fault(F_SHAPE, "solve_circulant", "the column holds no value"); }
    if b.ndim < 1 || b.ndim > 2 || b.shape[0] != n {
        return empty_arr(), fault(F_SHAPE, "solve_circulant", "the right side does not match the column");
    }
    var k: i64 = 1;
    if b.ndim == 2 { k = b.shape[1]; }
    var i: i64 = 0;
    while i < n {
        if !(c[i] - c[i] == 0.0) { return empty_arr(), fault(F_VALUE, "solve_circulant", "the column holds a value that is not finite"); }
        i += 1;
    }
    i = 0;
    while i < n * k {
        if !(b.data[i] - b.data[i] == 0.0) { return empty_arr(), fault(F_VALUE, "solve_circulant", "the right side holds a value that is not finite"); }
        i += 1;
    }
    var zer: f64[>..] = [];
    i = 0;
    while i < n { zer.push(0.0); i += 1; }
    // The transform of each column of c and of b.
    var fc_r: f64[>..] = [];
    var fc_i: f64[>..] = [];
    var fb_r: f64[>..] = [];
    var fb_i: f64[>..] = [];
    var col: i64 = 0;
    while col < k {
        var bc2: f64[>..] = [];
        i = 0;
        while i < n { bc2.push(b.data[i * k + col]); i += 1; }
        let cfr, cfi, f1 = fft_parts(c, zer);
        if f1.kind != F_NONE { return empty_arr(), f1; }
        let bfr, bfi, f2 = fft_parts(bc2, zer);
        if f2.kind != F_NONE { return empty_arr(), f2; }
        i = 0;
        while i < n {
            fc_r.push(cfr.data[i]);
            fc_i.push(cfi.data[i]);
            fb_r.push(bfr.data[i]);
            fb_i.push(bfi.data[i]);
            i += 1;
        }
        col += 1;
    }
    // The cutoff is the greatest modulus times n and the machine
    // epsilon, as in SciPy. A value of tol at or above zero replaces it.
    var big: f64 = 0.0;
    i = 0;
    while i < n * k {
        let av = mathshim::root(fc_r[i] * fc_r[i] + fc_i[i] * fc_i[i]);
        if av > big { big = av; }
        i += 1;
    }
    var cut = big * (n * 1.0) * 2.220446049250313e-16;
    if tol >= 0.0 { cut = tol; }
    var q_r: f64[>..] = [];
    var q_i: f64[>..] = [];
    i = 0;
    while i < n * k {
        let dr = fc_r[i];
        let di = fc_i[i];
        let av = mathshim::root(dr * dr + di * di);
        if av <= cut {
            return empty_arr(), fault(F_VALUE, "solve_circulant", "the circulant matrix is near singular");
        }
        let den = dr * dr + di * di;
        let er = fb_r[i];
        let ei = fb_i[i];
        q_r.push((er * dr + ei * di) / den);
        q_i.push((ei * dr - er * di) / den);
        i += 1;
    }
    // The inverse transform of each column of q, real values only.
    var x: f64[>..] = [];
    col = 0;
    while col < k {
        var qr2: f64[>..] = [];
        var qi2: f64[>..] = [];
        i = 0;
        while i < n { qr2.push(q_r[i * k + col]); qi2.push(q_i[i * k + col]); i += 1; }
        let xr, xi, f3 = ifft(qr2, qi2);
        if f3.kind != F_NONE { return empty_arr(), f3; }
        i = 0;
        while i < n { x.push(xr.data[i]); i += 1; }
        col += 1;
    }
    if k == 1 { return line_arr(copy(x)), no_fault(); }
    return mat2(copy(x), n, k), no_fault();
}

// matmul_toeplitz gives the product of the Toeplitz matrix of the
// first column c and the first row r with the vector or the matrix x.
fn matmul_toeplitz(c: f64[:], x, r: f64[:] = []) -> Arr, Fault {
    let n = c.len;
    if n < 1 { return empty_arr(), fault(F_SHAPE, "matmul_toeplitz", "the column holds no value"); }
    if x.ndim < 1 || x.ndim > 2 || x.shape[0] != n {
        return empty_arr(), fault(F_SHAPE, "matmul_toeplitz", "the operand does not match the column");
    }
    var k: i64 = 1;
    if x.ndim == 2 { k = x.shape[1]; }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < k {
            var s: f64 = 0.0;
            var q: i64 = 0;
            while q < n { s += st_row(c, r, i, q) * x.data[q * k + j]; q += 1; }
            d.push(s);
            j += 1;
        }
        i += 1;
    }
    if x.ndim == 1 { return line_arr(copy(d)), no_fault(); }
    return mat2(copy(d), n, k), no_fault();
}

// diagsvd gives the diagonal matrix of the singular values s, of the
// shape m by n.
fn diagsvd(s: f64[:], m: i64, n: i64) -> Arr, Fault {
    // SciPy builds the diagonal of the smaller side. A count of values
    // above the smaller side is a fault, as in SciPy.
    var side = m;
    if n < m { side = n; }
    if s.len != m && s.len != n { return empty_arr(), fault(F_SHAPE, "diagsvd", "the values must number m or n"); }
    if s.len > side {
        return empty_arr(), fault(F_SHAPE, "diagsvd", "the values must number the smaller side");
    }
    if m < 1 || n < 1 || m * n > MAX_ELEMENTS {
        return empty_arr(), fault(F_SHAPE, "diagsvd", "the shape holds no value or too many");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < m {
        var j: i64 = 0;
        while j < n {
            if i == j && i < s.len { d.push(s[i]); } else { d.push(0.0); }
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), m, n), no_fault();
}

// orthogonal_procrustes gives the orthogonal matrix R of the problem
// of the smallest norm of A R - B, through the singular values of
// A transpose B. The number is the sum of the singular values.
fn orthogonal_procrustes(a, b) -> Arr, f64, Fault {
    if a.ndim != 2 || b.ndim != 2 || a.shape[0] != b.shape[0] || a.shape[1] != b.shape[1] {
        return empty_arr(), 0.0, fault(F_SHAPE, "orthogonal_procrustes", "the operands are not two matrices of one shape");
    }
    let at, ft = transpose(a);
    if ft.kind != F_NONE { return empty_arr(), 0.0, ft; }
    let p, fp = matmul(at, b);
    if fp.kind != F_NONE { return empty_arr(), 0.0, fp; }
    let u, s, vt, f = svd(p);
    if f.kind != F_NONE { return empty_arr(), 0.0, f; }
    let rm, fm = matmul(u, vt);
    if fm.kind != F_NONE { return empty_arr(), 0.0, fm; }
    var sc: f64 = 0.0;
    var i: i64 = 0;
    while i < s.data.len { sc += s.data[i]; i += 1; }
    return rm, sc, no_fault();
}

// subspace_angles gives the principal angles between the column
// spaces of two matrices with the same count of rows. The angles are
// the arcs of the circle between zero and half pi, from the largest
// to the smallest.
fn subspace_angles(a, b) -> Arr, Fault {
    if a.ndim != 2 || b.ndim != 2 || a.shape[0] != b.shape[0] {
        return empty_arr(), fault(F_SHAPE, "subspace_angles", "the operands are not two matrices with the same count of rows");
    }
    let qa, fa = orth(a);
    if fa.kind != F_NONE { return empty_arr(), fa; }
    let qb, fb = orth(b);
    if fb.kind != F_NONE { return empty_arr(), fb; }
    let ca = qa.shape[1];
    let cb = qb.shape[1];
    let at, ft = transpose(qa);
    if ft.kind != F_NONE { return empty_arr(), ft; }
    let p, fp = matmul(at, qb);
    if fp.kind != F_NONE { return empty_arr(), fp; }
    let sigma, fs1 = svdvals(p);
    if fs1.kind != F_NONE { return empty_arr(), fs1; }
    let l = sigma.data.len;
    var mask: bool[>..] = [];
    var any_mask = false;
    var i: i64 = 0;
    while i < l {
        let m2 = sigma.data[i] * sigma.data[i] >= 0.5;
        mask.push(m2);
        if m2 { any_mask = true; }
        i += 1;
    }
    var mu: f64[>..] = [];
    if any_mask {
        // The matrix of the sines, and its singular values.
        var d: f64[>..] = [];
        if ca >= cb {
            let pr, fp2 = matmul(qa, p);
            if fp2.kind != F_NONE { return empty_arr(), fp2; }
            i = 0;
            while i < qb.data.len { d.push(qb.data[i] - pr.data[i]); i += 1; }
        } else {
            let pt, fp2 = transpose(p);
            if fp2.kind != F_NONE { return empty_arr(), fp2; }
            let pr, fp3 = matmul(qb, pt);
            if fp3.kind != F_NONE { return empty_arr(), fp3; }
            i = 0;
            while i < qa.data.len { d.push(qa.data[i] - pr.data[i]); i += 1; }
        }
        var rows = a.shape[0];
        var cols = cb;
        if ca < cb { cols = ca; }
        let bb = mat2(copy(d), rows, cols);
        let s2, fs2 = svdvals(bb);
        if fs2.kind != F_NONE { return empty_arr(), fs2; }
        i = 0;
        while i < l {
            var v: f64 = 0.0;
            if i < s2.data.len { v = s2.data[i]; }
            if v > 1.0 { v = 1.0; }
            if v < -1.0 { v = -1.0; }
            mu.push(mathshim::arcsine_of(v));
            i += 1;
        }
    }
    var out: f64[>..] = [];
    i = 0;
    while i < l {
        if mask[i] { out.push(mu[i]); }
        else {
            var v = sigma.data[l - 1 - i];
            if v > 1.0 { v = 1.0; }
            if v < -1.0 { v = -1.0; }
            out.push(mathshim::arccosine_of(v));
        }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// pinvh gives the inverse of a symmetric matrix through the eigen
// values and the eigen vectors. An eigen value at or below the cutoff
// stays out. The count of the values above the cutoff is the rank.
// A negative rtol means the default, the count of rows times the
// machine epsilon, as in SciPy.
fn pinvh(a, atol: f64 = 0.0, rtol: f64 = -1.0) -> Arr, i64, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), 0, fault(F_SHAPE, "pinvh", "the matrix is not square");
    }
    if !(atol >= 0.0) { return empty_arr(), 0, fault(F_VALUE, "pinvh", "the tolerance atol is negative or not finite"); }
    if !(rtol > 0.0) && rtol != -1.0 {
        return empty_arr(), 0, fault(F_VALUE, "pinvh", "the tolerance rtol is not above zero");
    }
    let n = a.shape[0];
    let s, u, f = eigh(a);
    if f.kind != F_NONE { return empty_arr(), 0, f; }
    var max_s: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        let av = mathshim::abs_of(s.data[i]);
        if av > max_s { max_s = av; }
        i += 1;
    }
    var rt = rtol;
    if rtol == -1.0 { rt = (n * 1.0) * 2.220446049250313e-16; }
    let cut = atol + max_s * rt;
    var keep: i64[>..] = [];
    var psigma: f64[>..] = [];
    i = 0;
    while i < n {
        if mathshim::abs_of(s.data[i]) > cut {
            keep.push(i);
            psigma.push(1.0 / s.data[i]);
        }
        i += 1;
    }
    if keep.len == 0 {
        return empty_arr(), 0, fault(F_VALUE, "pinvh", "no eigenvalue is above the cutoff");
    }
    var d: f64[>..] = [];
    i = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var acc: f64 = 0.0;
            var q: i64 = 0;
            while q < keep.len {
                acc += u.data[i * n + keep[q]] * psigma[q] * u.data[j * n + keep[q]];
                q += 1;
            }
            d.push(acc);
            j += 1;
        }
        i += 1;
    }
    return mat2(copy(d), n, n), keep.len, no_fault();
}
