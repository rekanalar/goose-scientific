// The helpers of the QZ algorithm: the rotations, the reduction and the signs.
//
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

