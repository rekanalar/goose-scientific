// The generalized Schur form of a pair of real matrices. The calls
// follow scipy.linalg.qz and scipy.linalg.ordqz. The pair A and B
// gives the matrices AA and BB and the orthogonal matrices Q and Z,
// so that A = Q AA Z^T and B = Q BB Z^T. The matrix BB is upper
// triangular. The matrix AA is upper triangular, except for one block
// of 2 by 2 for each pair of complex eigenvalues.
//
// The method is the one of LAPACK dgges. A Householder step and
// Givens rotations reduce the pair to the Hessenberg-triangular form.
// The double-shift QZ iteration of Moler and Stewart follows. Swaps of
// adjacent blocks put the chosen eigenvalues first. The factors equal
// those of LAPACK up to the sign of each column. The order of the
// eigenvalues can differ for a pair that the iteration orders in
// another way.
namespace scientific;

import mathshim;

const QZ_EPS: f64 = 2.220446049250313e-16;
const QZ_SAFMIN: f64 = 2.2250738585072014e-308;

// The most turns that the iteration takes for one eigenvalue.
const QZ_TURNS: i64 = 30;

// A swap of two blocks fails when a pivot of its system falls below
// this part of the largest entry of the system.
const QZ_PIVOT: f64 = 1.0e-14;

// A swap whose residual lies above this value is refused.
const QZ_SWAP_TOL: f64 = 1.0e-11;

// The codes of the rules of sort.
const QZ_NONE: i64 = 0;
const QZ_LHP: i64 = 1;
const QZ_RHP: i64 = 2;
const QZ_IUC: i64 = 3;
const QZ_OUC: i64 = 4;

// qz_hypot gives the length of the vector (a, b) without an overflow.
fn qz_hypot(a: f64, b: f64) -> f64 {
    let x = mathshim::abs_of(a);
    let y = mathshim::abs_of(b);
    if x == 0.0 { return y; }
    if y == 0.0 { return x; }
    if x >= y {
        let t1 = y / x;
        return x * mathshim::root(1.0 + t1 * t1);
    }
    let t2 = x / y;
    return y * mathshim::root(1.0 + t2 * t2);
}

// qz_givens gives c and s so that the rotation [c s; -s c] turns the
// vector (a, b) into (r, 0).
fn qz_givens(a: f64, b: f64) -> f64, f64 {
    if b == 0.0 { return 1.0, 0.0; }
    if a == 0.0 { return 0.0, 1.0; }
    let r = qz_hypot(a, b);
    return a / r, b / r;
}

// qz_house gives the vector v and the factor tau of the reflector
// I - tau v v^T that turns x into a multiple of the first unit
// vector. The vector holds a scaled copy of x, which keeps the sums of
// squares inside the range of the numbers.
fn qz_house(x: f64[:]) -> f64[>..], f64 {
    let k = x.len;
    var m: f64 = 0.0;
    var i: i64 = 0;
    while i < k {
        let t = mathshim::abs_of(x[i]);
        if t > m { m = t; }
        i += 1;
    }
    var v: f64[>..] = [];
    if m == 0.0 {
        v.push(1.0);
        i = 1;
        while i < k { v.push(0.0); i += 1; }
        return v, 0.0;
    }
    var ss: f64 = 0.0;
    i = 0;
    while i < k {
        let t = x[i] / m;
        v.push(t);
        ss += t * t;
        i += 1;
    }
    var beta = mathshim::root(ss);
    if v[0] > 0.0 { beta = -beta; }
    v[0] = v[0] - beta;
    var vv: f64 = 0.0;
    i = 0;
    while i < k { vv += v[i] * v[i]; i += 1; }
    if vv == 0.0 { return v, 0.0; }
    return v, 2.0 / vv;
}

// qz_left applies the reflector to the rows r0 and below of the
// columns c0 to c1 of the matrix d, which holds n columns.
fn qz_left(d: f64[>..]&, n: i64, v: f64[:], tau: f64, r0: i64, c0: i64, c1: i64) {
    if tau == 0.0 { return; }
    var c: i64 = c0;
    while c < c1 {
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i < v.len { s += v[i] * d[(r0 + i) * n + c]; i += 1; }
        s = s * tau;
        i = 0;
        while i < v.len { d[(r0 + i) * n + c] = d[(r0 + i) * n + c] - s * v[i]; i += 1; }
        c += 1;
    }
}

// qz_right applies the reflector to the columns c0 and above of the
// rows r0 to r1 of the matrix d.
fn qz_right(d: f64[>..]&, n: i64, v: f64[:], tau: f64, c0: i64, r0: i64, r1: i64) {
    if tau == 0.0 { return; }
    var r: i64 = r0;
    while r < r1 {
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i < v.len { s += d[r * n + c0 + i] * v[i]; i += 1; }
        s = s * tau;
        i = 0;
        while i < v.len { d[r * n + c0 + i] = d[r * n + c0 + i] - s * v[i]; i += 1; }
        r += 1;
    }
}

// qz_rot_rows turns the rows i and j of d in the columns c0 to c1.
fn qz_rot_rows(d: f64[>..]&, n: i64, i: i64, j: i64, c: f64, s: f64, c0: i64, c1: i64) {
    var k: i64 = c0;
    while k < c1 {
        let x = d[i * n + k];
        let y = d[j * n + k];
        d[i * n + k] = c * x + s * y;
        d[j * n + k] = c * y - s * x;
        k += 1;
    }
}

// qz_rot_cols turns the columns i and j of d in the rows r0 to r1.
fn qz_rot_cols(d: f64[>..]&, n: i64, i: i64, j: i64, c: f64, s: f64, r0: i64, r1: i64) {
    var k: i64 = r0;
    while k < r1 {
        let x = d[k * n + i];
        let y = d[k * n + j];
        d[k * n + i] = c * x + s * y;
        d[k * n + j] = c * y - s * x;
        k += 1;
    }
}

// qz_identity gives the unit matrix of side n.
fn qz_identity(n: i64) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { d.push(0.0); i += 1; }
    i = 0;
    while i < n { d[i * n + i] = 1.0; i += 1; }
    return d;
}

// qz_colrot clears the entry d(row, col - 1) of the matrix b, then it
// turns the columns col - 1 and col of the three matrices. The b
// entry on the diagonal is the pivot. The rows of a and b reach down
// to the row lim.
fn qz_colrot(a: f64[>..]&, b: f64[>..]&, z: f64[>..]&, n: i64, row: i64, col: i64, la: i64, lb: i64) {
    let r = qz_hypot(b[row * n + col - 1], b[row * n + col]);
    if r == 0.0 { return; }
    let c2 = b[row * n + col] / r;
    let s2 = -b[row * n + col - 1] / r;
    qz_rot_cols(a, n, col - 1, col, c2, s2, 0, la);
    qz_rot_cols(b, n, col - 1, col, c2, s2, 0, lb);
    qz_rot_cols(z, n, col - 1, col, c2, s2, 0, n);
    b[row * n + col - 1] = 0.0;
}

// qz_reduce turns the pair into the form where b is upper triangular
// and a is upper Hessenberg. The matrices q and z start as unit
// matrices and collect the transforms.
fn qz_reduce(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64) {
    var j: i64 = 0;
    while j < n - 1 {
        var x: f64[>..] = [];
        var i: i64 = j;
        while i < n { x.push(b[i * n + j]); i += 1; }
        let v, tau = qz_house(x);
        qz_left(b, n, v, tau, j, 0, n);
        qz_left(a, n, v, tau, j, 0, n);
        qz_right(q, n, v, tau, j, 0, n);
        i = j + 1;
        while i < n { b[i * n + j] = 0.0; i += 1; }
        j += 1;
    }
    j = 0;
    while j < n - 2 {
        var i: i64 = n - 1;
        while i > j + 1 {
            let c, s = qz_givens(a[(i - 1) * n + j], a[i * n + j]);
            qz_rot_rows(a, n, i - 1, i, c, s, 0, n);
            qz_rot_rows(b, n, i - 1, i, c, s, 0, n);
            qz_rot_cols(q, n, i - 1, i, c, s, 0, n);
            a[i * n + j] = 0.0;
            qz_colrot(a, b, z, n, i, i, n, n);
            i -= 1;
        }
        j += 1;
    }
}

// qz_block2 puts the block of two rows at the place j into the
// standard form. A block with real eigenvalues splits into two
// triangular parts. A block with a pair of complex eigenvalues gets a
// diagonal b with entries that are not negative. The block of b is
// upper triangular when the call starts.
fn qz_block2(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, j: i64) {
    let a11 = a[j * n + j];
    let a12 = a[j * n + j + 1];
    let a21 = a[(j + 1) * n + j];
    let a22 = a[(j + 1) * n + j + 1];
    let b11 = b[j * n + j];
    let b12 = b[j * n + j + 1];
    let b22 = b[(j + 1) * n + j + 1];
    let qa = b11 * b22;
    let qb = -(a11 * b22 + a22 * b11) + b12 * a21;
    let qc = a11 * a22 - a12 * a21;
    let disc = qb * qb - 4.0 * qa * qc;
    if disc >= 0.0 {
        // The eigenvalues are real. The first one gives a right
        // vector, and a left rotation then clears the entry below.
        let sq = mathshim::root(disc);
        var lam: f64 = 0.0;
        if qa != 0.0 {
            var sg = sq;
            if qb < 0.0 { sg = -sq; }
            let t = -(qb + sg) / 2.0;
            lam = t / qa;
        } else {
            if qb != 0.0 { lam = -qc / qb; }
        }
        let r0a = a11 - lam * b11;
        let r0b = a12 - lam * b12;
        let r1a = a21;
        let r1b = a22 - lam * b22;
        var x0: f64 = 1.0;
        var x1: f64 = 0.0;
        if mathshim::abs_of(r0a) + mathshim::abs_of(r0b) >= mathshim::abs_of(r1a) + mathshim::abs_of(r1b) {
            x0 = -r0b;
            x1 = r0a;
        } else {
            x0 = -r1b;
            x1 = r1a;
        }
        let nx = qz_hypot(x0, x1);
        if nx == 0.0 {
            x0 = 1.0;
            x1 = 0.0;
        } else {
            x0 = x0 / nx;
            x1 = x1 / nx;
        }
        qz_rot_cols(a, n, j, j + 1, x0, x1, 0, j + 2);
        qz_rot_cols(b, n, j, j + 1, x0, x1, 0, j + 2);
        qz_rot_cols(z, n, j, j + 1, x0, x1, 0, n);
        var pa = b[j * n + j];
        var pb = b[(j + 1) * n + j];
        if mathshim::abs_of(pa) + mathshim::abs_of(pb) < mathshim::abs_of(a[j * n + j]) + mathshim::abs_of(a[(j + 1) * n + j]) {
            pa = a[j * n + j];
            pb = a[(j + 1) * n + j];
        }
        let c, s = qz_givens(pa, pb);
        qz_rot_rows(a, n, j, j + 1, c, s, j, n);
        qz_rot_rows(b, n, j, j + 1, c, s, j, n);
        qz_rot_cols(q, n, j, j + 1, c, s, 0, n);
        a[(j + 1) * n + j] = 0.0;
        b[(j + 1) * n + j] = 0.0;
        return;
    }
    // A pair of complex eigenvalues. A rotation on each side makes
    // the block of b diagonal. The angles follow from the sums and the
    // differences of the entries of the 2 by 2 block of b.
    let e = (b11 + b22) / 2.0;
    let f = (b11 - b22) / 2.0;
    let g = b12 / 2.0;
    let h = -b12 / 2.0;
    let ang1 = mathshim::arctangent2_of(g, f);
    let ang2 = mathshim::arctangent2_of(h, e);
    let beta = (ang2 - ang1) / 2.0;
    let gamma = (ang2 + ang1) / 2.0;
    let cu = mathshim::cosine(gamma);
    let su = mathshim::sine(gamma);
    let cv = mathshim::cosine(beta);
    let sv = -mathshim::sine(beta);
    qz_rot_rows(a, n, j, j + 1, cu, su, j, n);
    qz_rot_rows(b, n, j, j + 1, cu, su, j, n);
    qz_rot_cols(q, n, j, j + 1, cu, su, 0, n);
    qz_rot_cols(a, n, j, j + 1, cv, sv, 0, j + 2);
    qz_rot_cols(b, n, j, j + 1, cv, sv, 0, j + 2);
    qz_rot_cols(z, n, j, j + 1, cv, sv, 0, n);
    b[(j + 1) * n + j] = 0.0;
    b[j * n + j + 1] = 0.0;
    // A negative diagonal entry turns the sign of its row.
    var k: i64 = j;
    while k < j + 2 {
        if b[k * n + k] < 0.0 {
            var c2: i64 = j;
            while c2 < n {
                a[k * n + c2] = -a[k * n + c2];
                b[k * n + c2] = -b[k * n + c2];
                c2 += 1;
            }
            var r2: i64 = 0;
            while r2 < n { q[r2 * n + k] = -q[r2 * n + k]; r2 += 1; }
        }
        k += 1;
    }
}

// qz_zero_b removes a diagonal entry of b that is zero. The entry
// sits at the place j of the window lo to ihi. At the bottom a
// rotation of the columns clears the last row of a. Elsewhere the zero
// moves up one row at a time, to the top of the window, and a rotation
// of the rows then clears the entry of a under the zero. The call
// gives the new bottom of the window.
fn qz_zero_b(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, lo: i64, ihi: i64, jstart: i64) -> i64 {
    var j = jstart;
    b[j * n + j] = 0.0;
    if j == ihi {
        if ihi > lo {
            let r = qz_hypot(a[ihi * n + ihi - 1], a[ihi * n + ihi]);
            if r != 0.0 {
                let c2 = a[ihi * n + ihi] / r;
                let s2 = -a[ihi * n + ihi - 1] / r;
                qz_rot_cols(a, n, ihi - 1, ihi, c2, s2, 0, ihi + 1);
                qz_rot_cols(b, n, ihi - 1, ihi, c2, s2, 0, ihi + 1);
                qz_rot_cols(z, n, ihi - 1, ihi, c2, s2, 0, n);
                a[ihi * n + ihi - 1] = 0.0;
                b[ihi * n + ihi - 1] = 0.0;
            }
        }
        return ihi - 1;
    }
    while j > lo {
        let r = qz_hypot(b[(j - 1) * n + j - 1], b[(j - 1) * n + j]);
        if r != 0.0 {
            let c2 = b[(j - 1) * n + j] / r;
            let s2 = -b[(j - 1) * n + j - 1] / r;
            qz_rot_cols(a, n, j - 1, j, c2, s2, 0, j + 2);
            qz_rot_cols(b, n, j - 1, j, c2, s2, 0, j + 1);
            qz_rot_cols(z, n, j - 1, j, c2, s2, 0, n);
        }
        b[(j - 1) * n + j - 1] = 0.0;
        let c, s = qz_givens(a[j * n + j - 1], a[(j + 1) * n + j - 1]);
        qz_rot_rows(a, n, j, j + 1, c, s, j - 1, n);
        qz_rot_rows(b, n, j, j + 1, c, s, j, n);
        qz_rot_cols(q, n, j, j + 1, c, s, 0, n);
        a[(j + 1) * n + j - 1] = 0.0;
        j -= 1;
    }
    let c3, s3 = qz_givens(a[j * n + j], a[(j + 1) * n + j]);
    qz_rot_rows(a, n, j, j + 1, c3, s3, j, n);
    qz_rot_rows(b, n, j, j + 1, c3, s3, j, n);
    qz_rot_cols(q, n, j, j + 1, c3, s3, 0, n);
    a[(j + 1) * n + j] = 0.0;
    return ihi;
}

// qz_step runs one double-shift step on the window l to e. The
// window holds at least three rows. The turn it gives the count of
// turns since the last deflation. A tenth turn takes a shift that
// breaks a cycle.
fn qz_step(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, l: i64, e: i64, it: i64) {
    let w = e - l + 1;
    var x1: f64 = 0.0;
    var x2: f64 = 0.0;
    var x3: f64 = 0.0;
    if it % 10 == 0 {
        x1 = mathshim::abs_of(a[l * n + l] / b[l * n + l]) + 1.0;
        x2 = 0.5;
        x3 = 0.25;
    } else {
        // The last two rows of A B^-1 in the window give the shifts.
        var y1: f64[>..] = [];
        var y2: f64[>..] = [];
        var rowi: i64 = 0;
        while rowi < 2 {
            let ri = e - 1 + rowi;
            var jj: i64 = 0;
            while jj < w {
                var s: f64 = a[ri * n + l + jj];
                var kk: i64 = 0;
                while kk < jj {
                    if rowi == 0 { s -= y1[kk] * b[(l + kk) * n + l + jj]; }
                    else { s -= y2[kk] * b[(l + kk) * n + l + jj]; }
                    kk += 1;
                }
                s = s / b[(l + jj) * n + l + jj];
                if rowi == 0 { y1.push(s); } else { y2.push(s); }
                jj += 1;
            }
            rowi += 1;
        }
        let m11 = y1[w - 2];
        let m12 = y1[w - 1];
        let m21 = y2[w - 2];
        let m22 = y2[w - 1];
        let sh = m11 + m22;
        let tr = m11 * m22 - m12 * m21;
        let b11 = b[l * n + l];
        let b22 = b[(l + 1) * n + l + 1];
        let b12 = b[l * n + l + 1];
        let k11 = a[l * n + l] / b11;
        let k21 = a[(l + 1) * n + l] / b11;
        let i12 = -b12 / (b11 * b22);
        let i22 = 1.0 / b22;
        let k12 = a[l * n + l] * i12 + a[l * n + l + 1] * i22;
        let k22 = a[(l + 1) * n + l] * i12 + a[(l + 1) * n + l + 1] * i22;
        let k32 = a[(l + 2) * n + l + 1] * i22;
        x1 = k11 * k11 + k12 * k21 - sh * k11 + tr;
        x2 = k21 * (k11 + k22 - sh);
        x3 = k32 * k21;
    }
    var k: i64 = l;
    while k < e - 1 {
        var vec: f64[>..] = [];
        if k == l {
            vec.push(x1);
            vec.push(x2);
            vec.push(x3);
        } else {
            vec.push(a[k * n + k - 1]);
            vec.push(a[(k + 1) * n + k - 1]);
            vec.push(a[(k + 2) * n + k - 1]);
        }
        let v, tau = qz_house(vec);
        var c0 = k - 1;
        if c0 < l { c0 = l; }
        qz_left(a, n, v, tau, k, c0, n);
        qz_left(b, n, v, tau, k, k, n);
        qz_right(q, n, v, tau, k, 0, n);
        if k > l {
            a[(k + 1) * n + k - 1] = 0.0;
            a[(k + 2) * n + k - 1] = 0.0;
        }
        // The reflector that clears the row k + 2 of b acts on the
        // reversed row, so it keeps the form of the earlier columns.
        var row: f64[>..] = [];
        row.push(b[(k + 2) * n + k + 2]);
        row.push(b[(k + 2) * n + k + 1]);
        row.push(b[(k + 2) * n + k]);
        let v2r, tau2 = qz_house(row);
        var v2: f64[>..] = [];
        v2.push(v2r[2]);
        v2.push(v2r[1]);
        v2.push(v2r[0]);
        var rmax = k + 3;
        if rmax > e { rmax = e; }
        qz_right(a, n, v2, tau2, k, 0, rmax + 1);
        qz_right(b, n, v2, tau2, k, 0, k + 3);
        qz_right(z, n, v2, tau2, k, 0, n);
        b[(k + 2) * n + k] = 0.0;
        b[(k + 2) * n + k + 1] = 0.0;
        qz_colrot(a, b, z, n, k + 1, k + 1, rmax + 1, k + 2);
        k += 1;
    }
    // The last pair of rows.
    let kk = e - 1;
    let c, s = qz_givens(a[kk * n + kk - 1], a[(kk + 1) * n + kk - 1]);
    qz_rot_rows(a, n, kk, kk + 1, c, s, kk - 1, n);
    qz_rot_rows(b, n, kk, kk + 1, c, s, kk, n);
    qz_rot_cols(q, n, kk, kk + 1, c, s, 0, n);
    a[(kk + 1) * n + kk - 1] = 0.0;
    qz_colrot(a, b, z, n, kk + 1, kk + 1, e + 1, e + 1);
}

// qz_in_block tells whether the row j belongs to a block of two rows.
fn qz_in_block(a: f64[:], n: i64, j: i64) -> bool {
    if j + 1 < n && a[(j + 1) * n + j] != 0.0 { return true; }
    if j > 0 && a[j * n + j - 1] != 0.0 { return true; }
    return false;
}

// qz_signs turns the sign of each row with a real eigenvalue whose
// entry of b is negative. The diagonal of b is then not negative.
fn qz_signs(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, n: i64) {
    var j: i64 = 0;
    while j < n {
        if b[j * n + j] < 0.0 && !qz_in_block(a, n, j) {
            var c: i64 = j;
            while c < n {
                a[j * n + c] = -a[j * n + c];
                b[j * n + c] = -b[j * n + c];
                c += 1;
            }
            var r: i64 = 0;
            while r < n { q[r * n + j] = -q[r * n + j]; r += 1; }
        }
        j += 1;
    }
}

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

// qz_flags_of sets the flag of each diagonal block once. A flag is 1 for
// a block that the rule chooses and 0 for the others. The forms a and b
// hold the pair divided by the scales sa and sb.
fn qz_flags_of(a: f64[:], b: f64[:], n: i64, rule: i64, sa: f64, sb: f64, flags: f64[>..]&) {
    var j: i64 = 0;
    while j < n {
        var sz: i64 = 1;
        if j + 1 < n && a[(j + 1) * n + j] != 0.0 { sz = 2; }
        if qz_chosen(a, b, n, j, sz, rule, sa, sb) { flags.push(1.0); } else { flags.push(0.0); }
        j += sz;
    }
}

// qz_flags_by sets the flag of each diagonal block once. The block of
// the caller chooses the block. It takes the alpha and the beta of the
// block, and gives true for a chosen block. The values come back to the
// scale of the caller, so the block reads the true alpha and the true
// beta. A pair of complex eigenvalues is one block. It gets the value
// with the positive imaginary part.
fn qz_flags_by<F: fn(Cplx, f64) -> bool>(a: f64[:], b: f64[:], n: i64, sa: f64, sb: f64, flags: f64[>..]&) {
    var j: i64 = 0;
    while j < n {
        var sz: i64 = 1;
        if j + 1 < n && a[(j + 1) * n + j] != 0.0 { sz = 2; }
        var al = Cplx { re: 0.0, im: 0.0 };
        var be: f64 = 0.0;
        if sz == 2 {
            let re, im, det = qz_roots2(a, b, n, j);
            let bt = mathshim::root(b[j * n + j] * b[(j + 1) * n + j + 1]);
            al = Cplx { re: re * bt * sa, im: im * bt * sa };
            be = bt * sb;
        } else {
            var sg: f64 = 1.0;
            if b[j * n + j] < 0.0 { sg = -1.0; }
            al = Cplx { re: a[j * n + j] * sa * sg, im: 0.0 };
            be = mathshim::abs_of(b[j * n + j]) * sb;
        }
        if F(al, be) { flags.push(1.0); } else { flags.push(0.0); }
        j += sz;
    }
}

// qz_partition puts the blocks that the flags choose before the others.
// The order inside each group stays as it was. A swap moves the flags
// with the blocks, and each swap lowers the count of unchosen blocks
// before a chosen block. The loop ends after at most n * n swaps, for
// any flags. The call gives false when a swap fails.
fn qz_partition(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, flags: f64[>..]&) -> bool {
    var again = true;
    while again {
        again = false;
        var j: i64 = 0;
        var k: i64 = 0;
        var pj: i64 = -1;
        var pk: i64 = -1;
        var psz: i64 = 0;
        while j < n && !again {
            var sz: i64 = 1;
            if j + 1 < n && a[(j + 1) * n + j] != 0.0 { sz = 2; }
            if flags[k] > 0.5 && pj >= 0 && flags[pk] < 0.5 {
                if !qz_swap(a, b, q, z, n, pj, psz, sz) { return false; }
                flags[pk] = 1.0;
                flags[k] = 0.0;
                again = true;
            } else {
                pj = j;
                pk = k;
                psz = sz;
                j += sz;
                k += 1;
            }
        }
    }
    qz_signs(a, b, q, n);
    return true;
}

// qz_eigs appends the eigenvalues of the pair in the form
// alpha / beta. The values alpha come as pairs of the real and the
// imaginary part. A pair of complex eigenvalues gives the value with
// the positive imaginary part first. Both values of the pair hold the
// same beta, the geometric mean of the two entries of b.
fn qz_eigs(a: f64[:], b: f64[:], n: i64, al: f64[>..]&, be: f64[>..]&) {
    var j: i64 = 0;
    while j < n {
        if j + 1 < n && a[(j + 1) * n + j] != 0.0 {
            let re, im, det = qz_roots2(a, b, n, j);
            let beta = mathshim::root(b[j * n + j] * b[(j + 1) * n + j + 1]);
            al.push(re * beta);
            al.push(im * beta);
            be.push(beta);
            al.push(re * beta);
            al.push(-im * beta);
            be.push(beta);
            j += 2;
        } else {
            al.push(a[j * n + j]);
            al.push(0.0);
            be.push(b[j * n + j]);
            j += 1;
        }
    }
}

// qz_cright multiplies the columns j and j + 1 of the complex matrix d
// by the matrix w of two rows from the right, in the rows r0 to r1. The
// matrix w holds four complex values: w00, w01, w10 and w11.
fn qz_cright(d: f64[>..]&, n: i64, j: i64, w: f64[:], r0: i64, r1: i64) {
    var r: i64 = r0;
    while r < r1 {
        let x = Cplx { re: d[2 * (r * n + j)], im: d[2 * (r * n + j) + 1] };
        let y = Cplx { re: d[2 * (r * n + j + 1)], im: d[2 * (r * n + j + 1) + 1] };
        let w00 = Cplx { re: w[0], im: w[1] };
        let w01 = Cplx { re: w[2], im: w[3] };
        let w10 = Cplx { re: w[4], im: w[5] };
        let w11 = Cplx { re: w[6], im: w[7] };
        let nx = c_add(c_mul(x, w00), c_mul(y, w10));
        let ny = c_add(c_mul(x, w01), c_mul(y, w11));
        d[2 * (r * n + j)] = nx.re;
        d[2 * (r * n + j) + 1] = nx.im;
        d[2 * (r * n + j + 1)] = ny.re;
        d[2 * (r * n + j + 1) + 1] = ny.im;
        r += 1;
    }
}

// qz_cleft_h multiplies the rows j and j + 1 of the complex matrix d by
// the conjugate transpose of w from the left, in the columns c0 to c1.
fn qz_cleft_h(d: f64[>..]&, n: i64, j: i64, w: f64[:], c0: i64, c1: i64) {
    var c: i64 = c0;
    while c < c1 {
        let x = Cplx { re: d[2 * (j * n + c)], im: d[2 * (j * n + c) + 1] };
        let y = Cplx { re: d[2 * ((j + 1) * n + c)], im: d[2 * ((j + 1) * n + c) + 1] };
        let h00 = Cplx { re: w[0], im: -w[1] };
        let h01 = Cplx { re: w[4], im: -w[5] };
        let h10 = Cplx { re: w[2], im: -w[3] };
        let h11 = Cplx { re: w[6], im: -w[7] };
        let nx = c_add(c_mul(h00, x), c_mul(h01, y));
        let ny = c_add(c_mul(h10, x), c_mul(h11, y));
        d[2 * (j * n + c)] = nx.re;
        d[2 * (j * n + c) + 1] = nx.im;
        d[2 * ((j + 1) * n + c)] = ny.re;
        d[2 * ((j + 1) * n + c) + 1] = ny.im;
        c += 1;
    }
}

// qz_unitary2 gives the unitary matrix of two rows whose first column
// is the unit vector (x0, x1). Its second column is (-conj x1, conj x0).
fn qz_unitary2(x0: Cplx, x1: Cplx) -> f64[>..] {
    var w: f64[>..] = [];
    w.push(x0.re); w.push(x0.im);
    w.push(-x1.re); w.push(x1.im);
    w.push(x1.re); w.push(x1.im);
    w.push(x0.re); w.push(-x0.im);
    return w;
}

// qz_unit_of turns the vector (x0, x1) into a unit vector.
fn qz_unit_of(x0: Cplx, x1: Cplx) -> Cplx, Cplx {
    let nr = qz_hypot(c_abs(x0), c_abs(x1));
    if nr == 0.0 { return Cplx { re: 1.0, im: 0.0 }, Cplx { re: 0.0, im: 0.0 }; }
    return c_scale(x0, 1.0 / nr), c_scale(x1, 1.0 / nr);
}

// qz_to_complex turns the real form of a pair into the triangular
// complex form. Each block of two rows gets two unitary matrices. The
// first column of the right one is the eigenvector of the block. The
// first column of the left one points along b times that vector. A
// phase on each row then makes the diagonal of b real. The results
// hold the real part and the imaginary part of each value side by side.
// The call appends them to the outputs.
fn qz_to_complex(a: f64[:], b: f64[:], q: f64[:], z: f64[:], n: i64, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&) {
    var ca: f64[>..] = [];
    var cb: f64[>..] = [];
    var cq: f64[>..] = [];
    var cz: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n {
        ca.push(a[i]); ca.push(0.0);
        cb.push(b[i]); cb.push(0.0);
        cq.push(q[i]); cq.push(0.0);
        cz.push(z[i]); cz.push(0.0);
        i += 1;
    }
    var j: i64 = 0;
    while j < n {
        if j + 1 < n && a[(j + 1) * n + j] != 0.0 {
            let re, im, det = qz_roots2(a, b, n, j);
            let lam = Cplx { re: re, im: im };
            let b00 = Cplx { re: b[j * n + j], im: 0.0 };
            let b11 = Cplx { re: b[(j + 1) * n + j + 1], im: 0.0 };
            let r00 = c_sub(Cplx { re: a[j * n + j], im: 0.0 }, c_mul(lam, b00));
            let r01 = Cplx { re: a[j * n + j + 1], im: 0.0 };
            let r10 = Cplx { re: a[(j + 1) * n + j], im: 0.0 };
            let r11 = c_sub(Cplx { re: a[(j + 1) * n + j + 1], im: 0.0 }, c_mul(lam, b11));
            var x0 = Cplx { re: 0.0, im: 0.0 };
            var x1 = Cplx { re: 0.0, im: 0.0 };
            if c_abs(r00) + c_abs(r01) >= c_abs(r10) + c_abs(r11) {
                x0 = c_scale(r01, -1.0);
                x1 = r00;
            } else {
                x0 = c_scale(r11, -1.0);
                x1 = r10;
            }
            let ux0, ux1 = qz_unit_of(x0, x1);
            let vw = qz_unitary2(ux0, ux1);
            let y0, y1 = qz_unit_of(c_mul(b00, ux0), c_mul(b11, ux1));
            let uw = qz_unitary2(y0, y1);
            qz_cright(ca, n, j, vw, 0, j + 2);
            qz_cright(cb, n, j, vw, 0, j + 2);
            qz_cright(cz, n, j, vw, 0, n);
            qz_cleft_h(ca, n, j, uw, j, n);
            qz_cleft_h(cb, n, j, uw, j, n);
            qz_cright(cq, n, j, uw, 0, n);
            ca[2 * ((j + 1) * n + j)] = 0.0;
            ca[2 * ((j + 1) * n + j) + 1] = 0.0;
            cb[2 * ((j + 1) * n + j)] = 0.0;
            cb[2 * ((j + 1) * n + j) + 1] = 0.0;
            j += 2;
        } else {
            j += 1;
        }
    }
    var k: i64 = 0;
    while k < n {
        let d = Cplx { re: cb[2 * (k * n + k)], im: cb[2 * (k * n + k) + 1] };
        let mag = c_abs(d);
        if mag > 0.0 && d.im != 0.0 {
            let inv = Cplx { re: d.re / mag, im: -d.im / mag };
            let ph = Cplx { re: d.re / mag, im: d.im / mag };
            var c: i64 = 0;
            while c < n {
                let x = c_mul(Cplx { re: ca[2 * (k * n + c)], im: ca[2 * (k * n + c) + 1] }, inv);
                ca[2 * (k * n + c)] = x.re;
                ca[2 * (k * n + c) + 1] = x.im;
                let y = c_mul(Cplx { re: cb[2 * (k * n + c)], im: cb[2 * (k * n + c) + 1] }, inv);
                cb[2 * (k * n + c)] = y.re;
                cb[2 * (k * n + c) + 1] = y.im;
                c += 1;
            }
            var r: i64 = 0;
            while r < n {
                let w = c_mul(Cplx { re: cq[2 * (r * n + k)], im: cq[2 * (r * n + k) + 1] }, ph);
                cq[2 * (r * n + k)] = w.re;
                cq[2 * (r * n + k) + 1] = w.im;
                r += 1;
            }
            cb[2 * (k * n + k)] = mag;
            cb[2 * (k * n + k) + 1] = 0.0;
        }
        k += 1;
    }
    i = 0;
    while i < 2 * n * n {
        oa.push(ca[i]);
        ob.push(cb[i]);
        oq.push(cq[i]);
        oz.push(cz[i]);
        i += 1;
    }
}

// qz_finite tells whether every value of a buffer is finite.
fn qz_finite(d: f64[:]) -> bool { return qz_finite_from(d, 0); }

// qz_finite_from does the same for the values from the place from.
fn qz_finite_from(d: f64[:], from: i64) -> bool {
    var i: i64 = from;
    while i < d.len {
        if !(d[i] - d[i] == 0.0) { return false; }
        i += 1;
    }
    return true;
}

// qz_stage_iterate copies one pair of side n, divides each matrix by
// its own power of two and reduces the pair. It runs the iteration.
// The call gives the two scales and a code: 0 on success, 1 when the
// iteration fails and 3 when a value overflows. The pair starts at the
// place ao of ad and at the place bo of bd.
fn qz_stage_iterate(ad: f64[:], ao: i64, bd: f64[:], bo: i64, n: i64, a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&) -> f64, f64, i64 {
    var i: i64 = 0;
    while i < n * n { a.push(ad[ao + i]); b.push(bd[bo + i]); i += 1; }
    let sa = qz_scale_of(a, 0, n * n);
    let sb = qz_scale_of(b, 0, n * n);
    qz_scale_apply(a, 0, n * n, sa, false);
    qz_scale_apply(b, 0, n * n, sb, false);
    let id = qz_identity(n);
    i = 0;
    while i < n * n { q.push(id[i]); z.push(id[i]); i += 1; }
    qz_reduce(a, b, q, z, n);
    if !qz_iterate(a, b, q, z, n, QZ_TURNS) {
        // A value that overflowed turns every test of the iteration
        // false, so the turns run out. That case is an overflow.
        if !(qz_finite(a) && qz_finite(b)) { return sa, sb, 3; }
        return sa, sb, 1;
    }
    return sa, sb, 0;
}

// qz_stage_finish puts the chosen blocks first, when ordered is true,
// and appends the results to the outputs. The flags hold one value for
// each diagonal block. The forms come back to the scale of the pair:
// AA and alpha by sa, BB and beta by sb. The call gives 0 on success,
// 2 when a swap fails and 3 when a value overflows.
fn qz_stage_finish(a: f64[>..]&, b: f64[>..]&, q: f64[>..]&, z: f64[>..]&, n: i64, sa: f64, sb: f64, flags: f64[>..]&, ordered: bool, cmode: bool, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64 {
    let so = oa.len;
    let sl = oal.len;
    let sbe = obe.len;
    if ordered {
        if !qz_partition(a, b, q, z, n, flags) { return 2; }
        qz_eigs(a, b, n, oal, obe);
    }
    if !(qz_finite(a) && qz_finite(b) && qz_finite(q) && qz_finite(z)) { return 3; }
    if cmode {
        qz_to_complex(a, b, q, z, n, oa, ob, oq, oz);
    } else {
        var i: i64 = 0;
        while i < n * n { oa.push(a[i]); ob.push(b[i]); oq.push(q[i]); oz.push(z[i]); i += 1; }
    }
    qz_scale_apply(oa, so, oa.len - so, sa, true);
    qz_scale_apply(ob, so, ob.len - so, sb, true);
    qz_scale_apply(oal, sl, oal.len - sl, sa, true);
    qz_scale_apply(obe, sbe, obe.len - sbe, sb, true);
    // The outputs hold the complex form and alpha and beta, so the
    // check covers what the call emits and not only the work buffers.
    if !(qz_finite_from(oa, so) && qz_finite_from(ob, so) && qz_finite_from(oq, so) && qz_finite_from(oz, so)) { return 3; }
    if !(qz_finite_from(oal, sl) && qz_finite_from(obe, sbe)) { return 3; }
    return 0;
}

// qz_pair decomposes one pair of side n. A rule other than QZ_NONE puts
// the chosen eigenvalues first and appends alpha and beta. The call
// appends the four forms to the outputs, real or complex. It gives 0
// on success, 1 when the iteration fails, 2 when a swap fails and 3
// when a value overflows.
fn qz_pair(ad: f64[:], ao: i64, bd: f64[:], bo: i64, n: i64, rule: i64, cmode: bool, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64 {
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var q: f64[>..] = [];
    var z: f64[>..] = [];
    let sa, sb, st = qz_stage_iterate(ad, ao, bd, bo, n, a, b, q, z);
    if st != 0 { return st; }
    var flags: f64[>..] = [];
    if rule != QZ_NONE { qz_flags_of(a, b, n, rule, sa, sb, flags); }
    return qz_stage_finish(a, b, q, z, n, sa, sb, flags, rule != QZ_NONE, cmode, oa, ob, oq, oz, oal, obe);
}

// qz_pair_by decomposes one pair of side n. The block of the caller
// chooses the eigenvalues that come first. The call appends the four
// forms to the outputs, real or complex. It gives 0 on success, 1 when
// the iteration fails, 2 when a swap fails and 3 when a value overflows.
fn qz_pair_by<F: fn(Cplx, f64) -> bool>(ad: f64[:], ao: i64, bd: f64[:], bo: i64, n: i64, cmode: bool, oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64 {
    var a: f64[>..] = [];
    var b: f64[>..] = [];
    var q: f64[>..] = [];
    var z: f64[>..] = [];
    let sa, sb, st = qz_stage_iterate(ad, ao, bd, bo, n, a, b, q, z);
    if st != 0 { return st; }
    var flags: f64[>..] = [];
    qz_flags_by(a, b, n, sa, sb, flags) { al, be => F(al, be) };
    return qz_stage_finish(a, b, q, z, n, sa, sb, flags, true, cmode, oa, ob, oq, oz, oal, obe);
}

// qz_rule gives the code of a rule of sort, or -1 for a name that no
// rule holds.
fn qz_rule(sort: const u8[:]) -> i64 {
    if sort == "lhp" { return QZ_LHP; }
    if sort == "rhp" { return QZ_RHP; }
    if sort == "iuc" { return QZ_IUC; }
    if sort == "ouc" { return QZ_OUC; }
    return -1;
}

// qz_check runs the checks of a call for the whole stack, in the order
// of the shape checks, the rule and the finite values. The width is 1
// for a real operand and 2 for a complex one. The call gives the batch
// shape of the result, the code of the rule and a fault. The slices of
// the broadcast go to ia and ib.
fn qz_check(a, b, width: i64, sort: const u8[:], ordered: bool, tag: const u8[:], ia: i64[>..]&, ib: i64[>..]&) -> i64[>..], i64, Fault {
    var none: i64[>..] = [];
    let fa = cs_check_stack(a.ndim, a.shape, a.data.len, width, 2, tag);
    if fa.kind != F_NONE { return none, QZ_NONE, fa; }
    let fb = cs_check_stack(b.ndim, b.shape, b.data.len, width, 2, tag);
    if fb.kind != F_NONE { return none, QZ_NONE, fb; }
    let m = a.shape[a.ndim - 1];
    if a.shape[a.ndim - 2] != m || b.shape[b.ndim - 1] != m || b.shape[b.ndim - 2] != m {
        return none, QZ_NONE, fault(F_SHAPE, tag, "the matrices are not square or hold different sides");
    }
    if m < 1 { return none, QZ_NONE, fault(F_SHAPE, tag, "the matrices hold no element"); }
    let ba = shape_batch(a.shape, 2);
    let bb = shape_batch(b.shape, 2);
    let cb, fw = cs_batch_walk(ba, bb, MAX_ELEMENTS / (m * m), tag, ia, ib);
    if fw.kind != F_NONE { return none, QZ_NONE, fw; }
    var rule: i64 = QZ_NONE;
    if ordered {
        rule = qz_rule(sort);
        if rule < 0 { return none, QZ_NONE, fault(F_VALUE, tag, "the rule of sort is not lhp, rhp, iuc or ouc"); }
    }
    let ga = carr_finite(a, tag);
    if ga.kind != F_NONE { return none, QZ_NONE, ga; }
    let gb = carr_finite(b, tag);
    if gb.kind != F_NONE { return none, QZ_NONE, gb; }
    return cb, rule, no_fault();
}

// qz_status turns the code of a pair into a fault.
fn qz_status(st: i64, tag: const u8[:]) -> Fault {
    if st == 1 { return fault(F_LIMIT, tag, "the iteration did not converge"); }
    if st == 2 { return fault(F_VALUE, tag, "the reorder failed"); }
    if st == 3 { return fault(F_VALUE, tag, "a value overflowed"); }
    return no_fault();
}

// qz_all runs the decomposition of a real pair for the whole stack.
// The call gives the batch shape of the result and a fault.
fn qz_all(a, b, sort: const u8[:], ordered: bool, cmode: bool, tag: const u8[:], oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64[>..], Fault {
    var none: i64[>..] = [];
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    let cb, rule, fc = qz_check(a, b, 1, sort, ordered, tag, ia, ib);
    if fc.kind != F_NONE { return none, fc; }
    let m = a.shape[a.ndim - 1];
    let count = batch_count(cb);
    var s: i64 = 0;
    while s < count {
        let st = qz_pair(a.data, ia[s] * m * m, b.data, ib[s] * m * m, m, rule, cmode, oa, ob, oq, oz, oal, obe);
        let fs = qz_status(st, tag);
        if fs.kind != F_NONE { return none, fs; }
        s += 1;
    }
    return cb, no_fault();
}

// qz_all_by runs the decomposition of a real pair for the whole stack.
// The block of the caller chooses the eigenvalues that come first. The
// call gives the batch shape of the result and a fault.
fn qz_all_by<F: fn(Cplx, f64) -> bool>(a, b, cmode: bool, tag: const u8[:], oa: f64[>..]&, ob: f64[>..]&, oq: f64[>..]&, oz: f64[>..]&, oal: f64[>..]&, obe: f64[>..]&) -> i64[>..], Fault {
    var none: i64[>..] = [];
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    let cb, rule, fc = qz_check(a, b, 1, "", false, tag, ia, ib);
    if fc.kind != F_NONE { return none, fc; }
    let m = a.shape[a.ndim - 1];
    let count = batch_count(cb);
    var s: i64 = 0;
    while s < count {
        let st = qz_pair_by(a.data, ia[s] * m * m, b.data, ib[s] * m * m, m, cmode, oa, ob, oq, oz, oal, obe) { al, be => F(al, be) };
        let fs = qz_status(st, tag);
        if fs.kind != F_NONE { return none, fs; }
        s += 1;
    }
    return cb, no_fault();
}

// qz_vec_shape gives the shape of the batch followed by the side m.
fn qz_vec_shape(cb: i64[:], m: i64) -> i64[>..] {
    var s: i64[>..] = [];
    var i: i64 = 0;
    while i < cb.len { s.push(cb[i]); i += 1; }
    s.push(m);
    return s;
}

// qz gives the generalized Schur form of a pair of real matrices. The
// result holds AA, BB, Q and Z, with A = Q AA Z^T and B = Q BB Z^T.
// The last two dimensions of each operand give the matrix. The
// dimensions before them give the batch, and the batches broadcast.
fn qz(a, b) -> Arr, Arr, Arr, Arr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all(a, b, "", false, false, "qz", oa, ob, oq, oz, oal, obe);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    return arr_of(copy(oa), sh), arr_of(copy(ob), sh), arr_of(copy(oq), sh), arr_of(copy(oz), sh), no_fault();
}

// ordqz gives the generalized Schur form with the chosen eigenvalues
// first. The rule is lhp, rhp, iuc or ouc, as in SciPy. The result holds
// AA, BB, alpha, beta, Q and Z. The eigenvalues are alpha / beta.
fn ordqz(a, b, sort: const u8[:]) -> Arr, Arr, CArr, Arr, Arr, Arr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all(a, b, sort, true, false, "ordqz", oa, ob, oq, oz, oal, obe);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), carr_empty(), empty_arr(), empty_arr(), empty_arr(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    let sv = qz_vec_shape(cb, m);
    return arr_of(copy(oa), sh), arr_of(copy(ob), sh), carr_make(sv, copy(oal)), arr_of(copy(obe), sv), arr_of(copy(oq), sh), arr_of(copy(oz), sh), no_fault();
}

// qz_complex gives the triangular complex form of the generalized
// Schur form of a real pair. The matrices AA and BB are upper
// triangular. The diagonal of BB is real and not negative. The
// products Q AA Z^H and Q BB Z^H equal A and B.
fn qz_complex(a, b) -> CArr, CArr, CArr, CArr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all(a, b, "", false, true, "qz_complex", oa, ob, oq, oz, oal, obe);
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), carr_empty(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    return carr_make(sh, copy(oa)), carr_make(sh, copy(ob)), carr_make(sh, copy(oq)), carr_make(sh, copy(oz)), no_fault();
}

// ordqz_complex gives the triangular complex form with the chosen
// eigenvalues first. The result holds AA, BB, alpha, beta, Q and Z.
fn ordqz_complex(a, b, sort: const u8[:]) -> CArr, CArr, CArr, Arr, CArr, CArr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all(a, b, sort, true, true, "ordqz_complex", oa, ob, oq, oz, oal, obe);
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), empty_arr(), carr_empty(), carr_empty(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    let sv = qz_vec_shape(cb, m);
    return carr_make(sh, copy(oa)), carr_make(sh, copy(ob)), carr_make(sv, copy(oal)), arr_of(copy(obe), sv), carr_make(sh, copy(oq)), carr_make(sh, copy(oz)), no_fault();
}

// ordqz_by gives the generalized Schur form with the eigenvalues that
// the block of the caller chooses first. The block takes the alpha and
// the beta of a diagonal block and gives true for a chosen block. The
// result holds AA, BB, alpha, beta, Q and Z. The eigenvalues are
// alpha / beta.
fn ordqz_by<F: fn(Cplx, f64) -> bool>(a, b) -> Arr, Arr, CArr, Arr, Arr, Arr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all_by(a, b, false, "ordqz_by", oa, ob, oq, oz, oal, obe) { al, be => F(al, be) };
    if f.kind != F_NONE { return empty_arr(), empty_arr(), carr_empty(), empty_arr(), empty_arr(), empty_arr(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    let sv = qz_vec_shape(cb, m);
    return arr_of(copy(oa), sh), arr_of(copy(ob), sh), carr_make(sv, copy(oal)), arr_of(copy(obe), sv), arr_of(copy(oq), sh), arr_of(copy(oz), sh), no_fault();
}

// ordqz_complex_by gives the triangular complex form with the
// eigenvalues that the block of the caller chooses first. The result
// holds AA, BB, alpha, beta, Q and Z.
fn ordqz_complex_by<F: fn(Cplx, f64) -> bool>(a, b) -> CArr, CArr, CArr, Arr, CArr, CArr, Fault {
    var oa: f64[>..] = [];
    var ob: f64[>..] = [];
    var oq: f64[>..] = [];
    var oz: f64[>..] = [];
    var oal: f64[>..] = [];
    var obe: f64[>..] = [];
    let cb, f = qz_all_by(a, b, true, "ordqz_complex_by", oa, ob, oq, oz, oal, obe) { al, be => F(al, be) };
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), empty_arr(), carr_empty(), carr_empty(), f; }
    let m = a.shape[a.ndim - 1];
    let sh = shape_matrix(cb, m, m);
    let sv = qz_vec_shape(cb, m);
    return carr_make(sh, copy(oa)), carr_make(sh, copy(ob)), carr_make(sv, copy(oal)), arr_of(copy(obe), sv), carr_make(sh, copy(oq)), carr_make(sh, copy(oz)), no_fault();
}
