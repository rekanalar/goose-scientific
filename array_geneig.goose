// The generalized eigenproblem of a pair of matrices: A v = w B v.
//
// ceig_pair and ceigvals_pair take a pair of general complex matrices. The
// method takes the generalized Schur form of the pair, then solves the
// triangular pair by back substitution. ceigh_pair and ceigvalsh_pair take a
// Hermitian matrix A and a Hermitian positive definite matrix B. The method
// factors B with the Cholesky method, reduces the problem to one Hermitian
// matrix and runs the complex Jacobi method. The calls with real operands
// take the same path. The calls follow scipy.linalg.eig, eigvals, eigh and
// eigvalsh.
namespace scientific;

import mathshim;

// g_check checks the two operands of a call. Both are square, of one side, at
// most 256, and every part holds a finite value.
fn g_check(a, b, tag: const u8[:]) -> Fault {
    let fa = c_eig_check(a, tag);
    if fa.kind != F_NONE { return fa; }
    let fb = c_eig_check(b, tag);
    if fb.kind != F_NONE { return fb; }
    if a.shape[0] != b.shape[0] {
        return fault(F_SHAPE, tag, "the two matrices hold different sides");
    }
    return no_fault();
}

// g_schur gives the generalized Schur form of a checked pair. The results are
// AA, BB and Z. The diagonal of BB is real and not negative.
fn g_schur(a, b, tag: const u8[:]) -> CArr, CArr, CArr, Fault {
    let f = g_check(a, b, tag);
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), f; }
    let aa, bb, q, z, fq = qz_c(copy(a), copy(b));
    if fq.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), fq; }
    return aa, bb, z, no_fault();
}

// g_largest gives the largest absolute part of a buffer. It gives one for a
// buffer of zeros.
fn g_largest(a: f64[:]) -> f64 {
    var big: f64 = 0.0;
    var i: i64 = 0;
    while i < a.len {
        var x = a[i];
        if x < 0.0 { x = -x; }
        if x > big { big = x; }
        i += 1;
    }
    if big == 0.0 { return 1.0; }
    return big;
}

// g_vectors gives the eigenvectors of a triangular pair. The matrices AA and
// BB are scaled by their largest parts first, so the products cannot
// overflow. The column k of the vectors solves (beta AA - alpha BB) x = 0
// with x[k] equal to one, then takes the unitary matrix Z. A denominator below
// 2.2e-16 times the largest entry of the column takes that value. Each column
// takes the length one and a real largest component above zero.
fn g_vectors(aa: f64[:], bb: f64[:], z: f64[:], n: i64, vr: f64[>..]&, vi: f64[>..]&) {
    let sa = g_largest(aa);
    let sb = g_largest(bb);
    var k: i64 = 0;
    while k < n {
        let ak = (k * n + k) * 2;
        let alr = aa[ak] / sa;
        let ali = aa[ak + 1] / sa;
        let ber = bb[ak] / sb;
        let bei = bb[ak + 1] / sb;
        // The entries t(i, j) of beta AA - alpha BB with i <= j <= k.
        var tr = c_zeros(n * n);
        var ti = c_zeros(n * n);
        var tmax: f64 = 0.0;
        var i: i64 = 0;
        while i <= k {
            var j = i;
            while j <= k {
                let at = (i * n + j) * 2;
                let x1, y1 = c_cmul(ber, bei, aa[at] / sa, aa[at + 1] / sa);
                let x2, y2 = c_cmul(alr, ali, bb[at] / sb, bb[at + 1] / sb);
                tr[i * n + j] = x1 - x2;
                ti[i * n + j] = y1 - y2;
                var m = tr[i * n + j];
                if m < 0.0 { m = -m; }
                if m > tmax { tmax = m; }
                m = ti[i * n + j];
                if m < 0.0 { m = -m; }
                if m > tmax { tmax = m; }
                j += 1;
            }
            i += 1;
        }
        var thr = C_EIG_EPS * tmax;
        if thr < 1.0e-300 { thr = 1.0e-300; }
        var xr = c_zeros(n);
        var xi = c_zeros(n);
        xr[k] = 1.0;
        i = k - 1;
        while i >= 0 {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var j = i + 1;
            while j <= k {
                let pr, pi = c_cmul(tr[i * n + j], ti[i * n + j], xr[j], xi[j]);
                sr += pr;
                si += pi;
                j += 1;
            }
            var dr = tr[i * n + i];
            var di = ti[i * n + i];
            if mathshim::root(dr * dr + di * di) < thr {
                dr = thr;
                di = 0.0;
            }
            // x = -s / d. The quotient divides by the larger part of d, so a
            // tiny denominator cannot underflow to zero when it is squared.
            var qr: f64 = 0.0;
            var qi: f64 = 0.0;
            var adr = dr;
            if adr < 0.0 { adr = -adr; }
            var adi = di;
            if adi < 0.0 { adi = -adi; }
            if adr >= adi {
                let ratio = di / dr;
                let den = dr + di * ratio;
                qr = (sr + si * ratio) / den;
                qi = (si - sr * ratio) / den;
            } else {
                let ratio = dr / di;
                let den = dr * ratio + di;
                qr = (sr * ratio + si) / den;
                qi = (si * ratio - sr) / den;
            }
            xr[i] = -qr;
            xi[i] = -qi;
            // A column that grows past 1e100 takes a smaller scale.
            var big = xr[i];
            if big < 0.0 { big = -big; }
            var bi = xi[i];
            if bi < 0.0 { bi = -bi; }
            if bi > big { big = bi; }
            if big > 1.0e100 {
                var u = i;
                while u <= k {
                    xr[u] = xr[u] * 1.0e-100;
                    xi[u] = xi[u] * 1.0e-100;
                    u += 1;
                }
            }
            i -= 1;
        }
        // The vector is Z x.
        var cr = c_zeros(n);
        var ci = c_zeros(n);
        var r: i64 = 0;
        while r < n {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var j: i64 = 0;
            while j <= k {
                let pr, pi = c_cmul(z[(r * n + j) * 2], z[(r * n + j) * 2 + 1], xr[j], xi[j]);
                sr += pr;
                si += pi;
                j += 1;
            }
            cr[r] = sr;
            ci[r] = si;
            r += 1;
        }
        c_unit_column(cr, ci, n);
        r = 0;
        while r < n {
            vr[r * n + k] = cr[r];
            vi[r * n + k] = ci[r];
            r += 1;
        }
        k += 1;
    }
    return;
}

// g_interleave joins two buffers of real and imaginary parts.
fn g_interleave(re: f64[:], im: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < re.len {
        d.push(re[i]);
        d.push(im[i]);
        i += 1;
    }
    return d;
}

// ceig_pair gives alpha, beta and the eigenvectors of a pair of complex square
// matrices. The eigenvalue is alpha divided by beta. The order follows the
// diagonal of the generalized Schur form. Each vector has the length one and
// a real largest component above zero. SciPy scales its vectors in another
// way.
fn ceig_pair(a, b) -> CArr, CArr, CArr, Fault {
    let aa, bb, z, f = g_schur(a, b, "ceig_pair");
    if f.kind != F_NONE { return carr_empty(), carr_empty(), carr_empty(), f; }
    let n = a.shape[0];
    var vr = c_zeros(n * n);
    var vi = c_zeros(n * n);
    g_vectors(aa.data, bb.data, z.data, n, vr, vi);
    var al: f64[>..] = [];
    var be: f64[>..] = [];
    var k: i64 = 0;
    while k < n {
        al.push(aa.data[2 * (k * n + k)]);
        al.push(aa.data[2 * (k * n + k) + 1]);
        be.push(bb.data[2 * (k * n + k)]);
        be.push(0.0);
        k += 1;
    }
    return CArr { ndim: 1, shape: c_eig_shape(n, 1), data: al }, CArr { ndim: 1, shape: c_eig_shape(n, 1), data: be }, CArr { ndim: 2, shape: c_eig_shape(n, 2), data: g_interleave(vr, vi) }, no_fault();
}

// ceigvals_pair gives alpha divided by beta for each eigenvalue. A zero beta
// gives the real part infinity and the imaginary part zero. A zero alpha with a
// zero beta gives not a number.
fn ceigvals_pair(a, b) -> CArr, Fault {
    let aa, bb, z, f = g_schur(a, b, "ceigvals_pair");
    if f.kind != F_NONE { return carr_empty(), f; }
    let n = a.shape[0];
    var w: f64[>..] = [];
    var k: i64 = 0;
    while k < n {
        let at = 2 * (k * n + k);
        let ar = aa.data[at];
        let ai = aa.data[at + 1];
        let be = bb.data[at];
        if be != 0.0 {
            w.push(ar / be);
            w.push(ai / be);
        } else if ar == 0.0 && ai == 0.0 {
            let nan = 0.0 / 0.0;
            w.push(nan);
            w.push(nan);
        } else {
            w.push(1.0 / 0.0);
            w.push(0.0);
        }
        k += 1;
    }
    return CArr { ndim: 1, shape: c_eig_shape(n, 1), data: w }, no_fault();
}

// eig_pair takes real operands. It gives the complex results of the same
// values.
fn eig_pair(a, b) -> CArr, CArr, CArr, Fault {
    return ceig_pair(complex_real(a), complex_real(b));
}

// eigvals_pair takes real operands. It gives the complex eigenvalues.
fn eigvals_pair(a, b) -> CArr, Fault {
    return ceigvals_pair(complex_real(a), complex_real(b));
}

// g_lower_solve solves L X = B for the lower triangle L, one column at a time.
// The buffers hold n by n complex matrices.
fn g_lower_solve(l: f64[:], b: f64[:], n: i64) -> f64[>..] {
    var x = c_zeros(2 * n * n);
    var col: i64 = 0;
    while col < n {
        var i: i64 = 0;
        while i < n {
            var sr = b[2 * (i * n + col)];
            var si = b[2 * (i * n + col) + 1];
            var j: i64 = 0;
            while j < i {
                let pr, pi = c_cmul(l[2 * (i * n + j)], l[2 * (i * n + j) + 1], x[2 * (j * n + col)], x[2 * (j * n + col) + 1]);
                sr -= pr;
                si -= pi;
                j += 1;
            }
            let d = l[2 * (i * n + i)];
            x[2 * (i * n + col)] = sr / d;
            x[2 * (i * n + col) + 1] = si / d;
            i += 1;
        }
        col += 1;
    }
    return x;
}

// g_upper_solve solves L^H X = Y for the lower triangle L, one column at a
// time. The buffers hold n by n complex matrices.
fn g_upper_solve(l: f64[:], y: f64[:], n: i64) -> f64[>..] {
    var x = c_zeros(2 * n * n);
    var col: i64 = 0;
    while col < n {
        var i = n - 1;
        while i >= 0 {
            var sr = y[2 * (i * n + col)];
            var si = y[2 * (i * n + col) + 1];
            var j = i + 1;
            while j < n {
                // The entry of L^H at (i, j) is the conjugate of L at (j, i).
                let pr, pi = c_cmul(l[2 * (j * n + i)], -l[2 * (j * n + i) + 1], x[2 * (j * n + col)], x[2 * (j * n + col) + 1]);
                sr -= pr;
                si -= pi;
                j += 1;
            }
            let d = l[2 * (i * n + i)];
            x[2 * (i * n + col)] = sr / d;
            x[2 * (i * n + col) + 1] = si / d;
            i -= 1;
        }
        col += 1;
    }
    return x;
}

// g_reduce gives the eigenvalues and the eigenvectors of a Hermitian pair with
// a positive definite B. A and B come from the chosen triangle. They are scaled
// apart. The result C = L^-1 A L^-H holds the Hermitian matrix of the Jacobi
// method. A fault of kind F_VALUE tells that B is not positive definite.
fn g_reduce(a, b, n: i64, lower: bool, w: f64[>..]&, v: f64[>..]&) -> Fault {
    let sa = c_scale_of(a.data);
    let sb = c_scale_of(b.data);
    var ar: f64[>..] = [];
    var ai: f64[>..] = [];
    c_hermitian_from_triangle(a.data, n, lower, sa, ar, ai);
    var br: f64[>..] = [];
    var bi: f64[>..] = [];
    c_hermitian_from_triangle(b.data, n, lower, sb, br, bi);
    let bc = CArr { ndim: 2, shape: c_eig_shape(n, 2), data: g_interleave(br, bi) };
    let l, fl = ccholesky(bc);
    if fl.kind != F_NONE { return fl; }
    let y = g_lower_solve(l.data, g_interleave(ar, ai), n);
    let yh = c_conj_transpose(y, n, n);
    let cm = g_lower_solve(l.data, yh, n);
    // C is Hermitian up to rounding. The mean of C and its conjugate transpose
    // makes it exact.
    let ch = c_conj_transpose(cm, n, n);
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n {
        hr.push(0.5 * (cm[2 * i] + ch[2 * i]));
        hi.push(0.5 * (cm[2 * i + 1] + ch[2 * i + 1]));
        i += 1;
    }
    var vals: f64[>..] = [];
    var yr: f64[>..] = [];
    var yi: f64[>..] = [];
    c_hermitian_jacobi(hr, hi, n, vals, yr, yi);
    if !qz_finite(vals) { return fault(F_VALUE, "ceigh_pair", "the reduced matrix holds a value that is not finite"); }
    i = 0;
    while i < n {
        w.push(vals[i] * sa / sb);
        i += 1;
    }
    // The vectors are L^-H y. Each column takes a real largest component above
    // zero. The length stays as B-orthonormality fixes it.
    let x = g_upper_solve(l.data, g_interleave(yr, yi), n);
    var col: i64 = 0;
    while col < n {
        var best: f64 = -1.0;
        var at: i64 = 0;
        var r: i64 = 0;
        var top: f64 = 0.0;
        while r < n {
            let p = x[2 * (r * n + col)];
            let q = x[2 * (r * n + col) + 1];
            var m = p;
            if m < 0.0 { m = -m; }
            if m > top { top = m; }
            m = q;
            if m < 0.0 { m = -m; }
            if m > top { top = m; }
            r += 1;
        }
        if top == 0.0 { top = 1.0; }
        r = 0;
        while r < n {
            let p = x[2 * (r * n + col)] / top;
            let q = x[2 * (r * n + col) + 1] / top;
            if p * p + q * q > best {
                best = p * p + q * q;
                at = r;
            }
            r += 1;
        }
        let m = mathshim::root(best);
        let pr = x[2 * (at * n + col)] / top / m;
        let pi = -x[2 * (at * n + col) + 1] / top / m;
        r = 0;
        while r < n {
            let zr, zi = c_cmul(pr, pi, x[2 * (r * n + col)], x[2 * (r * n + col) + 1]);
            v[2 * (r * n + col)] = zr;
            v[2 * (r * n + col) + 1] = zi;
            r += 1;
        }
        v[2 * (at * n + col) + 1] = 0.0;
        col += 1;
    }
    return no_fault();
}

// ceigh_pair gives the real eigenvalues in ascending order and the vectors of a
// Hermitian matrix A with a Hermitian positive definite matrix B. The vectors
// satisfy V^H B V equal to the unit matrix. The largest component of each
// vector is real and above zero. The call reads the triangle that uplo names.
fn ceigh_pair(a, b, uplo: const u8[:] = "L") -> Arr, CArr, Fault {
    let f = g_check(a, b, "ceigh_pair");
    if f.kind != F_NONE { return empty_arr(), carr_empty(), f; }
    let side = c_uplo(uplo);
    if side == 0 {
        return empty_arr(), carr_empty(), fault(F_VALUE, "ceigh_pair", "the choice of the triangle is not L or U");
    }
    let n = a.shape[0];
    var w: f64[>..] = [];
    var v = c_zeros(2 * n * n);
    let fr = g_reduce(a, b, n, side == 1, w, v);
    if fr.kind != F_NONE { return empty_arr(), carr_empty(), fr; }
    let vals, fv = from_values(w);
    if fv.kind != F_NONE { return empty_arr(), carr_empty(), fv; }
    return vals, CArr { ndim: 2, shape: c_eig_shape(n, 2), data: v }, no_fault();
}

// ceigvalsh_pair gives the eigenvalues of ceigh_pair.
fn ceigvalsh_pair(a, b, uplo: const u8[:] = "L") -> Arr, Fault {
    let vals, v, f = ceigh_pair(a, b, uplo);
    return vals, f;
}

// eigh_pair takes real operands. It gives the real eigenvalues and the real
// vectors.
fn eigh_pair(a, b, uplo: const u8[:] = "L") -> Arr, Arr, Fault {
    let vals, v, f = ceigh_pair(complex_real(a), complex_real(b), uplo);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let n = a.shape[0];
    var re: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n {
        re.push(v.data[2 * i]);
        i += 1;
    }
    let vr, fr = from_rows(n, n, re);
    return vals, vr, fr;
}

// eigvalsh_pair takes real operands. It gives the real eigenvalues.
fn eigvalsh_pair(a, b, uplo: const u8[:] = "L") -> Arr, Fault {
    let vals, v, f = eigh_pair(a, b, uplo);
    return vals, f;
}
