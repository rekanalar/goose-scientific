// The iteration of the QZ algorithm and the swap of two eigenvalues.
//
namespace scientific;

import mathshim;

// qz_iterate runs the QZ iteration on a pair in the Hessenberg and
// triangular form. The limit gives the turns that one eigenvalue may
// take. The call gives false when the limit is not enough.
fn qz_iterate(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, limit: i64) -> bool {
    var bnorm: f64 = QZ_SAFMIN;
    var i: i64 = 0;
    while i < n * n {
        if mathshim::abs_of(b[i]) > bnorm { bnorm = mathshim::abs_of(b[i]); }
        i += 1;
    }
    var btol = QZ_EPS * bnorm;
    if btol < QZ_SAFMIN { btol = QZ_SAFMIN; }
    var ihi = n - 1;
    var it: i64 = 0;
    while ihi >= 0 {
        // The window starts under the lowest small entry of the
        // subdiagonal. Such an entry is set to zero.
        var lo: i64 = 0;
        var k = ihi;
        while k > 0 {
            var tol = QZ_EPS * (mathshim::abs_of(a[k * n + k]) + mathshim::abs_of(a[(k - 1) * n + k - 1]));
            if tol < QZ_SAFMIN { tol = QZ_SAFMIN; }
            if mathshim::abs_of(a[k * n + k - 1]) <= tol {
                a[k * n + k - 1] = 0.0;
                lo = k;
                k = 0;
            } else {
                k -= 1;
            }
        }
        var handled = false;
        var j = lo;
        while j <= ihi && !handled {
            if mathshim::abs_of(b[j * n + j]) <= btol {
                ihi = qz_zero_b(a, b, q, z, n, lo, ihi, j);
                it = 0;
                handled = true;
            }
            j += 1;
        }
        if !handled {
            if lo == ihi {
                ihi -= 1;
                it = 0;
            } else {
                if lo == ihi - 1 {
                    qz_block2(a, b, q, z, n, lo);
                    ihi -= 2;
                    it = 0;
                } else {
                    it += 1;
                    if it > limit { return false; }
                    qz_step(a, b, q, z, n, lo, ihi, it);
                }
            }
        }
    }
    qz_signs(a, b, q, n);
    return true;
}

// qz_scale_of gives the power of two 2^k for the count entries of d
// from the place from. The largest absolute entry divided by it lies in
// [1, 2). A buffer of zeros gives 1. The loop runs at most 2100 times.
fn qz_scale_of(d: f64[:], from: i64, count: i64) -> f64 {
    var m: f64 = 0.0;
    var i: i64 = from;
    while i < from + count {
        if mathshim::abs_of(d[i]) > m { m = mathshim::abs_of(d[i]); }
        i += 1;
    }
    if m == 0.0 { return 1.0; }
    var p: f64 = 1.0;
    while m >= 2.0 { m = m / 2.0; p = p * 2.0; }
    while m < 1.0 { m = m * 2.0; p = p / 2.0; }
    return p;
}

// qz_scale_apply divides the count entries of d from the place from by
// the power of two s, or multiplies them by s when up is true. A power
// of two changes no digit of a value that stays a normal number.
fn qz_scale_apply(d: f64[>..]&, from: i64, count: i64, s: f64, up: bool) {
    var i: i64 = from;
    while i < from + count {
        if up { d[i] = d[i] * s; } else { d[i] = d[i] / s; }
        i += 1;
    }
}

// qz_roots2 gives the real part and the squared modulus of the
// eigenvalues of the block of two rows at the place j. The block of b
// is diagonal. The eigenvalues are those of b^-1 a.
fn qz_roots2(a: f64[:], b: f64[:], n: i64, j: i64) -> f64, f64, f64 {
    let b11 = b[j * n + j];
    let b22 = b[(j + 1) * n + j + 1];
    if b11 == 0.0 || b22 == 0.0 { return 0.0, 0.0, 0.0; }
    let tr = a[j * n + j] / b11 + a[(j + 1) * n + j + 1] / b22;
    let det = (a[j * n + j] * a[(j + 1) * n + j + 1] - a[j * n + j + 1] * a[(j + 1) * n + j]) / (b11 * b22);
    var im2 = det - tr * tr / 4.0;
    if im2 < 0.0 { im2 = 0.0; }
    return tr / 2.0, mathshim::root(im2), det;
}

// qz_chosen tells whether a rule selects the block of sz rows at the
// place j. The rules follow SciPy. A pair of complex eigenvalues has
// one decision for both values. An eigenvalue with beta zero holds in
// no half plane and in no unit circle. It lies outside the unit
// circle when alpha is not zero. The forms a and b hold the pair
// divided by the scales sa and sb. A rule on the modulus compares
// the true values alpha and beta, so the ratio of the scales never
// forms.
fn qz_chosen(a: f64[:], b: f64[:], n: i64, j: i64, sz: i64, rule: i64, sa: f64, sb: f64) -> bool {
    if sz == 2 {
        let re, im, det = qz_roots2(a, b, n, j);
        if rule == QZ_LHP { return re < 0.0; }
        if rule == QZ_RHP { return re > 0.0; }
        let m = mathshim::root(det) * sa;
        if rule == QZ_IUC { return m < sb; }
        return m > sb;
    }
    let al = mathshim::abs_of(a[j * n + j]);
    let be = mathshim::abs_of(b[j * n + j]);
    if rule == QZ_OUC {
        if be == 0.0 { return al != 0.0; }
        return al * sa > be * sb;
    }
    if be == 0.0 { return false; }
    let r = a[j * n + j] / b[j * n + j];
    if rule == QZ_LHP { return r < 0.0; }
    if rule == QZ_RHP { return r > 0.0; }
    return al * sa < be * sb;
}

// qz_solve solves the system m x = rhs of size k by elimination with
// partial pivoting. The system holds its rows one after the other. The
// call scales each row by its largest entry. It gives false when a row
// holds only zeros or a pivot falls below QZ_PIVOT. The solution
// replaces rhs.
fn qz_solve(m: f64[>..]&, rhs: f64[>..]&, k: i64) -> bool {
    // Each row gets the scale of its largest entry, so the pivot test
    // reads against a row of size 1. A row of zeros is singular.
    var i: i64 = 0;
    while i < k {
        var rmax: f64 = 0.0;
        var t0: i64 = 0;
        while t0 < k {
            if mathshim::abs_of(m[i * k + t0]) > rmax { rmax = mathshim::abs_of(m[i * k + t0]); }
            t0 += 1;
        }
        if rmax == 0.0 { return false; }
        t0 = 0;
        while t0 < k { m[i * k + t0] = m[i * k + t0] / rmax; t0 += 1; }
        rhs[i] = rhs[i] / rmax;
        i += 1;
    }
    let big: f64 = 1.0;
    var c: i64 = 0;
    while c < k {
        var p = c;
        var r: i64 = c + 1;
        while r < k {
            if mathshim::abs_of(m[r * k + c]) > mathshim::abs_of(m[p * k + c]) { p = r; }
            r += 1;
        }
        if mathshim::abs_of(m[p * k + c]) <= QZ_PIVOT * big { return false; }
        if p != c {
            var t: i64 = 0;
            while t < k {
                let x = m[c * k + t];
                m[c * k + t] = m[p * k + t];
                m[p * k + t] = x;
                t += 1;
            }
            let y = rhs[c];
            rhs[c] = rhs[p];
            rhs[p] = y;
        }
        r = c + 1;
        while r < k {
            let f = m[r * k + c] / m[c * k + c];
            var t2: i64 = c;
            while t2 < k { m[r * k + t2] = m[r * k + t2] - f * m[c * k + t2]; t2 += 1; }
            rhs[r] = rhs[r] - f * rhs[c];
            r += 1;
        }
        c += 1;
    }
    c = k - 1;
    while c >= 0 {
        var s = rhs[c];
        var t3: i64 = c + 1;
        while t3 < k { s -= m[c * k + t3] * rhs[t3]; t3 += 1; }
        rhs[c] = s / m[c * k + c];
        c -= 1;
    }
    return true;
}

// qz_basis gives an orthogonal matrix of side m whose first n2 columns
// span the columns of [X; -I]. The matrix x holds n1 rows and n2
// columns, with m = n1 + n2.
fn qz_basis(x: f64[:], n1: i64, n2: i64) -> f64[>..] {
    let m = n1 + n2;
    var s: f64[>..] = [];
    var r: i64 = 0;
    while r < m {
        var c: i64 = 0;
        while c < n2 {
            if r < n1 { s.push(x[r * n2 + c]); }
            else {
                if r - n1 == c { s.push(-1.0); } else { s.push(0.0); }
            }
            c += 1;
        }
        r += 1;
    }
    var w = qz_identity(m);
    var c2: i64 = 0;
    while c2 < n2 {
        var col: f64[>..] = [];
        r = c2;
        while r < m { col.push(s[r * n2 + c2]); r += 1; }
        let v, tau = qz_house(col);
        qz_left(s, n2, v, tau, c2, 0, n2);
        qz_right(w, m, v, tau, c2, 0, m);
        c2 += 1;
    }
    return w;
}

// qz_mul_cols multiplies the columns j to j + m of d by the matrix w
// from the right, for every row.
fn qz_mul_cols(d: f64[>..]&, n: i64, j: i64, m: i64, w: f64[:]) {
    var tmp: f64[>..] = [];
    var r: i64 = 0;
    while r < n {
        tmp.clear();
        var c: i64 = 0;
        while c < m {
            var s: f64 = 0.0;
            var t: i64 = 0;
            while t < m { s += d[r * n + j + t] * w[t * m + c]; t += 1; }
            tmp.push(s);
            c += 1;
        }
        c = 0;
        while c < m { d[r * n + j + c] = tmp[c]; c += 1; }
        r += 1;
    }
}

// qz_mul_rows multiplies the rows j to j + m of d by the transpose of
// w from the left, for every column.
fn qz_mul_rows(d: f64[>..]&, n: i64, j: i64, m: i64, w: f64[:]) {
    var tmp: f64[>..] = [];
    var c: i64 = 0;
    while c < n {
        tmp.clear();
        var r: i64 = 0;
        while r < m {
            var s: f64 = 0.0;
            var t: i64 = 0;
            while t < m { s += w[t * m + r] * d[(j + t) * n + c]; t += 1; }
            tmp.push(s);
            r += 1;
        }
        r = 0;
        while r < m { d[(j + r) * n + c] = tmp[r]; r += 1; }
        c += 1;
    }
}

// qz_restore puts the block of sz rows at the place jj back into the
// standard form after a swap.
fn qz_restore(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, jj: i64, sz: i64) {
    if sz != 2 { return; }
    let c, s = qz_givens(b[jj * n + jj], b[(jj + 1) * n + jj]);
    qz_rot_rows(a, n, jj, jj + 1, c, s, jj, n);
    qz_rot_rows(b, n, jj, jj + 1, c, s, jj, n);
    qz_rot_cols(q, n, jj, jj + 1, c, s, 0, n);
    b[(jj + 1) * n + jj] = 0.0;
    qz_block2(a, b, q, z, n, jj);
}

// qz_swap swaps the block of n1 rows at the place j with the block of
// n2 rows that follows it. The call gives false when the system has no
// safe solution, or when the residual of the swap lies above QZ_SWAP_TOL.
fn qz_swap(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, j: i64, n1: i64, n2: i64) -> bool {
    let r = qz_swap_resid(a, b, q, z, n, j, n1, n2);
    return r >= 0.0 && r <= QZ_SWAP_TOL;
}

// qz_swap_resid does the swap of qz_swap and gives its residual. The
// system of the swap is A11 R - L A22 = A12 and B11 R - L B22 = B12.
// The residual is the largest value that the swap must clear, in
// relation to the largest entry of the window. A system without a safe
// solution gives -1.
fn qz_swap_resid(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, j: i64, n1: i64, n2: i64) -> f64 {
    let m = n1 + n2;
    let k = n1 * n2;
    var sys: f64[>..] = [];
    var rhs: f64[>..] = [];
    var i: i64 = 0;
    while i < 4 * k * k { sys.push(0.0); i += 1; }
    i = 0;
    while i < 2 * k { rhs.push(0.0); i += 1; }
    let w = 2 * k;
    var c: i64 = 0;
    while c < n2 {
        var r: i64 = 0;
        while r < n1 {
            let row = c * n1 + r;
            var t: i64 = 0;
            while t < n1 {
                sys[row * w + c * n1 + t] = sys[row * w + c * n1 + t] + a[(j + r) * n + j + t];
                sys[(k + row) * w + c * n1 + t] = sys[(k + row) * w + c * n1 + t] + b[(j + r) * n + j + t];
                t += 1;
            }
            t = 0;
            while t < n2 {
                sys[row * w + k + t * n1 + r] = sys[row * w + k + t * n1 + r] - a[(j + n1 + t) * n + j + n1 + c];
                sys[(k + row) * w + k + t * n1 + r] = sys[(k + row) * w + k + t * n1 + r] - b[(j + n1 + t) * n + j + n1 + c];
                t += 1;
            }
            rhs[row] = a[(j + r) * n + j + n1 + c];
            rhs[k + row] = b[(j + r) * n + j + n1 + c];
            r += 1;
        }
        c += 1;
    }
    if !qz_solve(sys, rhs, w) { return -1.0; }
    var rm: f64[>..] = [];
    var lm: f64[>..] = [];
    var r2: i64 = 0;
    while r2 < n1 {
        var c2: i64 = 0;
        while c2 < n2 {
            rm.push(rhs[c2 * n1 + r2]);
            lm.push(rhs[k + c2 * n1 + r2]);
            c2 += 1;
        }
        r2 += 1;
    }
    let zw = qz_basis(rm, n1, n2);
    let qw = qz_basis(lm, n1, n2);
    qz_mul_cols(a, n, j, m, zw);
    qz_mul_cols(b, n, j, m, zw);
    qz_mul_cols(z, n, j, m, zw);
    qz_mul_rows(a, n, j, m, qw);
    qz_mul_rows(b, n, j, m, qw);
    qz_mul_cols(q, n, j, m, qw);
    var low: f64 = 0.0;
    var win: f64 = 0.0;
    var wr: i64 = j;
    while wr < j + m {
        var wc: i64 = j;
        while wc < j + m {
            let lowrow = wr >= j + n2 && wc < j + n2;
            let va = mathshim::abs_of(a[wr * n + wc]);
            let vb = mathshim::abs_of(b[wr * n + wc]);
            if va > win { win = va; }
            if vb > win { win = vb; }
            if lowrow && va > low { low = va; }
            if lowrow && vb > low { low = vb; }
            wc += 1;
        }
        wr += 1;
    }
    var rr: i64 = j + n2;
    while rr < j + m {
        var cc: i64 = j;
        while cc < j + n2 {
            a[rr * n + cc] = 0.0;
            b[rr * n + cc] = 0.0;
            cc += 1;
        }
        rr += 1;
    }
    qz_restore(a, b, q, z, n, j, n2);
    qz_restore(a, b, q, z, n, j + n2, n1);
    if win == 0.0 { return 0.0; }
    return low / win;
}

