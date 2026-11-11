// The eigenvalues and the eigenvectors of a complex matrix.
//
// ceig and ceigvals take a general matrix. The method reduces the matrix to
// the Hessenberg form with Householder turns. A shifted QR iteration with
// Givens turns then gives the Schur form. Back substitution on the triangle
// gives the vectors. ceigh and ceigvalsh take a Hermitian matrix and use the
// complex Jacobi method. The calls follow numpy.linalg.eig, eigvals, eigh and
// eigvalsh.
namespace scientific;

import mathshim;

// The side of a matrix of these calls. A side above the limit gives a fault of
// kind F_LIMIT.
const C_EIG_SIDE: i64 = 256;

// The steps that the iteration spends on one eigenvalue.
const C_EIG_STEPS: i64 = 30;

// The sweeps of the Jacobi method.
const C_EIG_SWEEPS: i64 = 60;

// The machine epsilon of a double.
const C_EIG_EPS: f64 = 2.220446049250313e-16;

// c_eig_check checks an operand of a call. The matrix is square, its side is
// at most 256 and every part holds a finite value.
fn c_eig_check(a, tag: const u8[:]) -> Fault {
    if a.ndim != 2 {
        return fault(F_SHAPE, tag, "the operand holds a count of dimensions that is not two");
    }
    let f = carr_check(a, tag);
    if f.kind != F_NONE { return f; }
    if a.shape[0] != a.shape[1] {
        return fault(F_SHAPE, tag, "the matrix is not square");
    }
    if a.shape[0] > C_EIG_SIDE {
        return fault(F_LIMIT, tag, "the side of the matrix is above 256");
    }
    if !carr_finite(a) {
        return fault(F_VALUE, tag, "the operand holds a value that is not finite");
    }
    return no_fault();
}

// c_zeros gives a buffer of count zero values.
fn c_zeros(count: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < count {
        out.push(0.0);
        i += 1;
    }
    return out;
}

// c_csqrt gives the root of a complex number with a non negative real part.
fn c_csqrt(re: f64, im: f64) -> f64, f64 {
    let m = mathshim::root(re * re + im * im);
    var sr = mathshim::root((m + re) / 2.0);
    var si = mathshim::root((m - re) / 2.0);
    if im < 0.0 { si = -si; }
    return sr, si;
}

// c_split reads the matrix of a complex array into a buffer of real parts and
// a buffer of imaginary parts. The values divide by the scale.
fn c_split(a: f64[:], n: i64, scale: f64, re: f64[>..]&, im: f64[>..]&) {
    var i: i64 = 0;
    while i < n * n {
        re.push(a[2 * i] / scale);
        im.push(a[2 * i + 1] / scale);
        i += 1;
    }
    return;
}

// c_hessenberg reduces the matrix to the Hessenberg form with Householder
// turns. The buffers hr and hi hold the matrix. The buffers qr and qi hold the
// unitary matrix that gathers the turns. They start as the unit matrix.
fn c_hessenberg(hr: f64[>..]&, hi: f64[>..]&, qr: f64[>..]&, qi: f64[>..]&, n: i64) {
    var k: i64 = 0;
    while k < n - 2 {
        // The part of the column below the subdiagonal.
        var nx: f64 = 0.0;
        var r = k + 1;
        while r < n {
            nx += hr[r * n + k] * hr[r * n + k] + hi[r * n + k] * hi[r * n + k];
            r += 1;
        }
        nx = mathshim::root(nx);
        if nx > 0.0 {
            // The vector of the turn. Its first part holds the phase of the
            // first entry of the column.
            let x0r = hr[(k + 1) * n + k];
            let x0i = hi[(k + 1) * n + k];
            let am = mathshim::root(x0r * x0r + x0i * x0i);
            var phr: f64 = 1.0;
            var phi: f64 = 0.0;
            if am > 0.0 {
                phr = x0r / am;
                phi = x0i / am;
            }
            var vr: f64[>..] = [];
            var vi: f64[>..] = [];
            r = k + 1;
            while r < n {
                vr.push(hr[r * n + k]);
                vi.push(hi[r * n + k]);
                r += 1;
            }
            vr[0] = vr[0] + phr * nx;
            vi[0] = vi[0] + phi * nx;
            var vn: f64 = 0.0;
            var i: i64 = 0;
            while i < vr.len {
                vn += vr[i] * vr[i] + vi[i] * vi[i];
                i += 1;
            }
            vn = mathshim::root(vn);
            i = 0;
            while i < vr.len {
                vr[i] = vr[i] / vn;
                vi[i] = vi[i] / vn;
                i += 1;
            }
            let m = vr.len;
            // The turn acts on the rows k+1 to n-1 from the left.
            var c: i64 = 0;
            while c < n {
                var sr: f64 = 0.0;
                var si: f64 = 0.0;
                i = 0;
                while i < m {
                    let f = (k + 1 + i) * n + c;
                    // The sum of conj(v) times the entry.
                    sr += vr[i] * hr[f] + vi[i] * hi[f];
                    si += vr[i] * hi[f] - vi[i] * hr[f];
                    i += 1;
                }
                i = 0;
                while i < m {
                    let f = (k + 1 + i) * n + c;
                    let zr, zi = c_cmul(vr[i], vi[i], sr, si);
                    hr[f] = hr[f] - 2.0 * zr;
                    hi[f] = hi[f] - 2.0 * zi;
                    i += 1;
                }
                c += 1;
            }
            // The turn acts on the columns k+1 to n-1 from the right.
            // The unitary matrix takes the same turn.
            var rr: i64 = 0;
            while rr < n {
                var sr: f64 = 0.0;
                var si: f64 = 0.0;
                var qsr: f64 = 0.0;
                var qsi: f64 = 0.0;
                i = 0;
                while i < m {
                    let f = rr * n + k + 1 + i;
                    let xr, xi = c_cmul(hr[f], hi[f], vr[i], vi[i]);
                    sr += xr;
                    si += xi;
                    let yr, yi = c_cmul(qr[f], qi[f], vr[i], vi[i]);
                    qsr += yr;
                    qsi += yi;
                    i += 1;
                }
                i = 0;
                while i < m {
                    let f = rr * n + k + 1 + i;
                    // The entry loses 2 s conj(v).
                    let zr, zi = c_cmul(sr, si, vr[i], -vi[i]);
                    hr[f] = hr[f] - 2.0 * zr;
                    hi[f] = hi[f] - 2.0 * zi;
                    let wr, wi = c_cmul(qsr, qsi, vr[i], -vi[i]);
                    qr[f] = qr[f] - 2.0 * wr;
                    qi[f] = qi[f] - 2.0 * wi;
                    i += 1;
                }
                rr += 1;
            }
            // The column holds zero below the subdiagonal.
            r = k + 2;
            while r < n {
                hr[r * n + k] = 0.0;
                hi[r * n + k] = 0.0;
                r += 1;
            }
        }
        k += 1;
    }
    return;
}

// c_schur turns a matrix of the Hessenberg form into the Schur form. The
// buffers hr and hi hold the matrix and end as the upper triangle. The
// buffers qr and qi gather the turns. The call gives false when one
// eigenvalue needs more steps than the limit.
fn c_schur(hr: f64[>..]&, hi: f64[>..]&, qr: f64[>..]&, qi: f64[>..]&, n: i64, limit: i64) -> bool {
    var top = n - 1;
    var steps: i64 = 0;
    while top > 0 {
        // Find the lowest subdiagonal entry that holds no value.
        var l = top;
        while l > 0 {
            var s = mathshim::root(hr[(l - 1) * n + l - 1] * hr[(l - 1) * n + l - 1] + hi[(l - 1) * n + l - 1] * hi[(l - 1) * n + l - 1]);
            s += mathshim::root(hr[l * n + l] * hr[l * n + l] + hi[l * n + l] * hi[l * n + l]);
            if s == 0.0 { s = 1.0; }
            let e = mathshim::root(hr[l * n + l - 1] * hr[l * n + l - 1] + hi[l * n + l - 1] * hi[l * n + l - 1]);
            if e <= C_EIG_EPS * s {
                hr[l * n + l - 1] = 0.0;
                hi[l * n + l - 1] = 0.0;
                break;
            }
            l -= 1;
        }
        if l == top {
            top -= 1;
            steps = 0;
        } else {
            steps += 1;
            if steps > limit { return false; }
            // The Wilkinson shift. It is the eigenvalue of the last block of
            // two rows that lies nearer to the last diagonal entry.
            let ar = hr[(top - 1) * n + top - 1];
            let ai = hi[(top - 1) * n + top - 1];
            let br = hr[(top - 1) * n + top];
            let bi = hi[(top - 1) * n + top];
            let cr = hr[top * n + top - 1];
            let ci = hi[top * n + top - 1];
            let dr = hr[top * n + top];
            let di = hi[top * n + top];
            let tr = (ar + dr) / 2.0;
            let ti = (ai + di) / 2.0;
            let hr2 = (ar - dr) / 2.0;
            let hi2 = (ai - di) / 2.0;
            let sqr, sqi = c_cmul(hr2, hi2, hr2, hi2);
            let bcr, bci = c_cmul(br, bi, cr, ci);
            let rr, ri = c_csqrt(sqr + bcr, sqi + bci);
            let m1r = tr + rr;
            let m1i = ti + ri;
            let m2r = tr - rr;
            let m2i = ti - ri;
            let d1 = (m1r - dr) * (m1r - dr) + (m1i - di) * (m1i - di);
            let d2 = (m2r - dr) * (m2r - dr) + (m2i - di) * (m2i - di);
            var mur = m2r;
            var mui = m2i;
            if d1 < d2 {
                mur = m1r;
                mui = m1i;
            }
            // The 10th and the 20th step use a shift of another kind.
            if steps == 10 || steps == 20 {
                mur = mathshim::root(cr * cr + ci * ci);
                if top > 1 {
                    mur += mathshim::root(hr[(top - 1) * n + top - 2] * hr[(top - 1) * n + top - 2] + hi[(top - 1) * n + top - 2] * hi[(top - 1) * n + top - 2]);
                }
                mui = 0.0;
            }
            var k = l;
            while k <= top {
                hr[k * n + k] = hr[k * n + k] - mur;
                hi[k * n + k] = hi[k * n + k] - mui;
                k += 1;
            }
            // The turns of the step. The turn k zeroes the entry below the
            // diagonal in the column k.
            var gcr: f64[>..] = [];
            var gci: f64[>..] = [];
            var gsr: f64[>..] = [];
            var gsi: f64[>..] = [];
            k = l;
            while k < top {
                let xr = hr[k * n + k];
                let xi = hi[k * n + k];
                let yr = hr[(k + 1) * n + k];
                let yi = hi[(k + 1) * n + k];
                let rn = mathshim::root(xr * xr + xi * xi + yr * yr + yi * yi);
                var cvr: f64 = 1.0;
                var cvi: f64 = 0.0;
                var svr: f64 = 0.0;
                var svi: f64 = 0.0;
                if rn > 0.0 {
                    cvr = xr / rn;
                    cvi = xi / rn;
                    svr = yr / rn;
                    svi = yi / rn;
                }
                gcr.push(cvr);
                gci.push(cvi);
                gsr.push(svr);
                gsi.push(svi);
                // The turn is [[conj c, conj s], [-s, c]] on the rows k, k+1.
                var j = k;
                while j < n {
                    let p = k * n + j;
                    let q = (k + 1) * n + j;
                    let t1r, t1i = c_cmul(cvr, -cvi, hr[p], hi[p]);
                    let t2r, t2i = c_cmul(svr, -svi, hr[q], hi[q]);
                    let u1r, u1i = c_cmul(svr, svi, hr[p], hi[p]);
                    let u2r, u2i = c_cmul(cvr, cvi, hr[q], hi[q]);
                    hr[p] = t1r + t2r;
                    hi[p] = t1i + t2i;
                    hr[q] = u2r - u1r;
                    hi[q] = u2i - u1i;
                    j += 1;
                }
                k += 1;
            }
            // The turns act on the columns from the right, with the conjugate
            // transpose. The unitary matrix takes the same turns.
            k = l;
            while k < top {
                let cvr = gcr[k - l];
                let cvi = gci[k - l];
                let svr = gsr[k - l];
                let svi = gsi[k - l];
                var rw: i64 = 0;
                while rw <= k + 1 && rw < n {
                    let p = rw * n + k;
                    let q = rw * n + k + 1;
                    // The matrix G^H is [[c, -conj s], [s, conj c]].
                    let a1r, a1i = c_cmul(hr[p], hi[p], cvr, cvi);
                    let a2r, a2i = c_cmul(hr[q], hi[q], svr, svi);
                    let b1r, b1i = c_cmul(hr[p], hi[p], svr, -svi);
                    let b2r, b2i = c_cmul(hr[q], hi[q], cvr, -cvi);
                    hr[p] = a1r + a2r;
                    hi[p] = a1i + a2i;
                    hr[q] = b2r - b1r;
                    hi[q] = b2i - b1i;
                    rw += 1;
                }
                rw = 0;
                while rw < n {
                    let p = rw * n + k;
                    let q = rw * n + k + 1;
                    let a1r, a1i = c_cmul(qr[p], qi[p], cvr, cvi);
                    let a2r, a2i = c_cmul(qr[q], qi[q], svr, svi);
                    let b1r, b1i = c_cmul(qr[p], qi[p], svr, -svi);
                    let b2r, b2i = c_cmul(qr[q], qi[q], cvr, -cvi);
                    qr[p] = a1r + a2r;
                    qi[p] = a1i + a2i;
                    qr[q] = b2r - b1r;
                    qi[q] = b2i - b1i;
                    rw += 1;
                }
                k += 1;
            }
            k = l;
            while k <= top {
                hr[k * n + k] = hr[k * n + k] + mur;
                hi[k * n + k] = hi[k * n + k] + mui;
                k += 1;
            }
        }
    }
    // The entries below the diagonal hold zero.
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < i {
            hr[i * n + j] = 0.0;
            hi[i * n + j] = 0.0;
            j += 1;
        }
        i += 1;
    }
    return true;
}

// c_schur_vectors gives the eigenvectors of a matrix from its Schur form. The
// buffers tr and ti hold the triangle. The buffers qr and qi hold the unitary
// matrix. The results go into the buffers vr and vi. Each column has the length
// one and a real largest component.
fn c_schur_vectors(tr: f64[:], ti: f64[:], qr: f64[:], qi: f64[:], n: i64, vr: f64[>..]&, vi: f64[>..]&) {
    // A denominator below the small value takes that value.
    var big: f64 = 0.0;
    var i: i64 = 0;
    while i < n * n {
        let m = mathshim::root(tr[i] * tr[i] + ti[i] * ti[i]);
        if m > big { big = m; }
        i += 1;
    }
    if big == 0.0 { big = 1.0; }
    let small = C_EIG_EPS * big;
    var xr = c_zeros(n * n);
    var xi = c_zeros(n * n);
    var k: i64 = 0;
    while k < n {
        xr[k * n + k] = 1.0;
        var r = k - 1;
        while r >= 0 {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var j = r + 1;
            while j <= k {
                let zr, zi = c_cmul(tr[r * n + j], ti[r * n + j], xr[j * n + k], xi[j * n + k]);
                sr += zr;
                si += zi;
                j += 1;
            }
            var dr = tr[r * n + r] - tr[k * n + k];
            var di = ti[r * n + r] - ti[k * n + k];
            if mathshim::root(dr * dr + di * di) < small {
                dr = small;
                di = 0.0;
            }
            let dd = dr * dr + di * di;
            // The entry is minus s over d.
            let zr, zi = c_cmul(-sr, -si, dr, -di);
            xr[r * n + k] = zr / dd;
            xi[r * n + k] = zi / dd;
            r -= 1;
        }
        k += 1;
    }
    // The vectors of the matrix are Q times x. Each column takes the length
    // one and a real largest component.
    var col: i64 = 0;
    while col < n {
        var cr = c_zeros(n);
        var ci = c_zeros(n);
        var r: i64 = 0;
        while r < n {
            var sr: f64 = 0.0;
            var si: f64 = 0.0;
            var j: i64 = 0;
            while j <= col {
                let zr, zi = c_cmul(qr[r * n + j], qi[r * n + j], xr[j * n + col], xi[j * n + col]);
                sr += zr;
                si += zi;
                j += 1;
            }
            cr[r] = sr;
            ci[r] = si;
            r += 1;
        }
        c_unit_column(cr, ci, n);
        r = 0;
        while r < n {
            vr[r * n + col] = cr[r];
            vi[r * n + col] = ci[r];
            r += 1;
        }
        col += 1;
    }
    return;
}

// c_unit_column gives a column the length one. The phase of the column makes
// the largest component real and above zero. The first component of the
// largest size wins a tie.
fn c_unit_column(cr: f64[>..]&, ci: f64[>..]&, n: i64) {
    var nrm: f64 = 0.0;
    var i: i64 = 0;
    while i < n {
        nrm += cr[i] * cr[i] + ci[i] * ci[i];
        i += 1;
    }
    nrm = mathshim::root(nrm);
    if nrm == 0.0 { return; }
    var best: f64 = -1.0;
    var at: i64 = 0;
    i = 0;
    while i < n {
        cr[i] = cr[i] / nrm;
        ci[i] = ci[i] / nrm;
        let size = cr[i] * cr[i] + ci[i] * ci[i];
        if size > best {
            best = size;
            at = i;
        }
        i += 1;
    }
    let m = mathshim::root(best);
    let pr = cr[at] / m;
    let pi = -ci[at] / m;
    i = 0;
    while i < n {
        let zr, zi = c_cmul(pr, pi, cr[i], ci[i]);
        cr[i] = zr;
        ci[i] = zi;
        i += 1;
    }
    ci[at] = 0.0;
    return;
}

// c_hermitian_from_triangle builds the Hermitian matrix of an operand. The
// call reads the lower triangle when lower holds, else the upper triangle.
// The diagonal takes its real part. The other triangle comes by conjugation.
fn c_hermitian_from_triangle(a: f64[:], n: i64, lower: bool, scale: f64, hr: f64[>..]&, hi: f64[>..]&) {
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            var si = i;
            var sj = j;
            var conj = false;
            // The element that the triangle holds.
            if lower && j > i {
                si = j;
                sj = i;
                conj = true;
            }
            if !lower && j < i {
                si = j;
                sj = i;
                conj = true;
            }
            let f = 2 * (si * n + sj);
            hr.push(a[f] / scale);
            if i == j {
                hi.push(0.0);
            } else if conj {
                hi.push(-a[f + 1] / scale);
            } else {
                hi.push(a[f + 1] / scale);
            }
            j += 1;
        }
        i += 1;
    }
    return;
}

// c_hermitian_jacobi gives the eigenvalues and the eigenvectors of a Hermitian
// matrix. The buffers hr and hi hold the matrix. The buffers w, vr and vi get
// the eigenvalues in ascending order and the vectors. The largest component
// of each vector holds a real value above zero.
fn c_hermitian_jacobi(hr: f64[>..]&, hi: f64[>..]&, n: i64, w: f64[>..]&, vr: f64[>..]&, vi: f64[>..]&) {
    var ur = qz_identity(n);
    var ui = c_zeros(n * n);
    var sweep: i64 = 0;
    while sweep < C_EIG_SWEEPS {
        var turned = false;
        var p: i64 = 0;
        while p < n - 1 {
            var q = p + 1;
            while q < n {
                let gr = hr[p * n + q];
                let gi = hi[p * n + q];
                let g = mathshim::root(gr * gr + gi * gi);
                var dp = hr[p * n + p];
                if dp < 0.0 { dp = -dp; }
                var dq = hr[q * n + q];
                if dq < 0.0 { dq = -dq; }
                if g > 0.0 && g > 1.0e-17 * (dp + dq) {
                    turned = true;
                    let phr = gr / g;
                    let phi = gi / g;
                    let tau = (hr[q * n + q] - hr[p * n + p]) / (2.0 * g);
                    var atau = tau;
                    if atau < 0.0 { atau = -atau; }
                    var t = 1.0 / (atau + mathshim::root(1.0 + tau * tau));
                    if tau < 0.0 { t = -t; }
                    let c = 1.0 / mathshim::root(1.0 + t * t);
                    let s = t * c;
                    // The unitary turn U has c on the diagonal, s ph in the
                    // place p q and -s conj(ph) in the place q p. The matrix
                    // takes H U and then U^H times that.
                    var r: i64 = 0;
                    while r < n {
                        let ar = hr[r * n + p];
                        let ai = hi[r * n + p];
                        let br = hr[r * n + q];
                        let bi = hi[r * n + q];
                        // New p: c a - s conj(ph) b. New q: s ph a + c b.
                        let x1r, x1i = c_cmul(phr, -phi, br, bi);
                        let x2r, x2i = c_cmul(phr, phi, ar, ai);
                        hr[r * n + p] = c * ar - s * x1r;
                        hi[r * n + p] = c * ai - s * x1i;
                        hr[r * n + q] = s * x2r + c * br;
                        hi[r * n + q] = s * x2i + c * bi;
                        let er = ur[r * n + p];
                        let ei = ui[r * n + p];
                        let fr = ur[r * n + q];
                        let fi = ui[r * n + q];
                        let y1r, y1i = c_cmul(phr, -phi, fr, fi);
                        let y2r, y2i = c_cmul(phr, phi, er, ei);
                        ur[r * n + p] = c * er - s * y1r;
                        ui[r * n + p] = c * ei - s * y1i;
                        ur[r * n + q] = s * y2r + c * fr;
                        ui[r * n + q] = s * y2i + c * fi;
                        r += 1;
                    }
                    var col: i64 = 0;
                    while col < n {
                        let mpr = hr[p * n + col];
                        let mpi = hi[p * n + col];
                        let mqr = hr[q * n + col];
                        let mqi = hi[q * n + col];
                        // New row p: c m_p - s ph m_q. New row q: s conj(ph) m_p + c m_q.
                        let z1r, z1i = c_cmul(phr, phi, mqr, mqi);
                        let z2r, z2i = c_cmul(phr, -phi, mpr, mpi);
                        hr[p * n + col] = c * mpr - s * z1r;
                        hi[p * n + col] = c * mpi - s * z1i;
                        hr[q * n + col] = s * z2r + c * mqr;
                        hi[q * n + col] = s * z2i + c * mqi;
                        col += 1;
                    }
                    // The turn zeroes the pair.
                    hr[p * n + q] = 0.0;
                    hi[p * n + q] = 0.0;
                    hr[q * n + p] = 0.0;
                    hi[q * n + p] = 0.0;
                }
                q += 1;
            }
            p += 1;
        }
        sweep += 1;
        if !turned { break; }
    }
    // The eigenvalues come in ascending order.
    var order: i64[>..] = [];
    var i: i64 = 0;
    while i < n {
        order.push(i);
        i += 1;
    }
    i = 1;
    while i < n {
        let key = order[i];
        var j = i - 1;
        while j >= 0 {
            if hr[order[j] * n + order[j]] > hr[key * n + key] {
                order[j + 1] = order[j];
                j -= 1;
            } else { break; }
        }
        order[j + 1] = key;
        i += 1;
    }
    i = 0;
    while i < n {
        w.push(hr[order[i] * n + order[i]]);
        i += 1;
    }
    // The columns follow the order. Each column takes the phase rule.
    i = 0;
    while i < n * n {
        vr.push(0.0);
        vi.push(0.0);
        i += 1;
    }
    var col: i64 = 0;
    while col < n {
        var cr = c_zeros(n);
        var ci = c_zeros(n);
        var r: i64 = 0;
        while r < n {
            cr[r] = ur[r * n + order[col]];
            ci[r] = ui[r * n + order[col]];
            r += 1;
        }
        c_unit_column(cr, ci, n);
        r = 0;
        while r < n {
            vr[r * n + col] = cr[r];
            vi[r * n + col] = ci[r];
            r += 1;
        }
        col += 1;
    }
    return;
}

// c_eig_solve gives the eigenvalues and the eigenvectors of a matrix. The
// operand divides by its scale and the eigenvalues take the scale back. The
// buffer w gets n values with the real and the imaginary part side by side.
// The buffer v gets the vectors in the same layout. The call gives false when
// the iteration does not converge.
fn c_eig_solve(a: f64[:], n: i64, limit: i64, w: f64[>..]&, v: f64[>..]&) -> bool {
    let scale = c_scale_of(a);
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    c_split(a, n, scale, hr, hi);
    var qr = qz_identity(n);
    var qi = c_zeros(n * n);
    c_hessenberg(hr, hi, qr, qi, n);
    if !c_schur(hr, hi, qr, qi, n, limit) { return false; }
    var i: i64 = 0;
    while i < n {
        w.push(hr[i * n + i] * scale);
        w.push(hi[i * n + i] * scale);
        i += 1;
    }
    var vr = c_zeros(n * n);
    var vi = c_zeros(n * n);
    c_schur_vectors(hr, hi, qr, qi, n, vr, vi);
    i = 0;
    while i < n * n {
        v.push(vr[i]);
        v.push(vi[i]);
        i += 1;
    }
    return true;
}

// c_eig_shape gives the shape of one dimension or of a matrix.
fn c_eig_shape(n: i64, dims: i64) -> i64[..8] {
    var s: i64[..8] = [];
    s.push(n);
    if dims == 2 { s.push(n); }
    return s;
}

// ceig gives the eigenvalues and the eigenvectors of a complex square matrix.
// Each vector has the length one and a real largest component, as LAPACK
// gives it. The eigenvalues come in the order of the Schur form. NumPy states
// that its order is not fixed.
fn ceig(a) -> CArr, CArr, Fault {
    let f = c_eig_check(a, "ceig");
    if f.kind != F_NONE { return carr_empty(), carr_empty(), f; }
    let n = a.shape[0];
    var w: f64[>..] = [];
    var v: f64[>..] = [];
    if !c_eig_solve(a.data, n, C_EIG_STEPS, w, v) {
        return carr_empty(), carr_empty(), fault(F_VALUE, "ceig", "the iteration did not converge");
    }
    return CArr { ndim: 1, shape: c_eig_shape(n, 1), data: w }, CArr { ndim: 2, shape: c_eig_shape(n, 2), data: v }, no_fault();
}

// ceigvals gives the eigenvalues of a complex square matrix.
fn ceigvals(a) -> CArr, Fault {
    let f = c_eig_check(a, "ceigvals");
    if f.kind != F_NONE { return carr_empty(), f; }
    let n = a.shape[0];
    var w: f64[>..] = [];
    var v: f64[>..] = [];
    if !c_eig_solve(a.data, n, C_EIG_STEPS, w, v) {
        return carr_empty(), fault(F_VALUE, "ceigvals", "the iteration did not converge");
    }
    return CArr { ndim: 1, shape: c_eig_shape(n, 1), data: w }, no_fault();
}

// c_eigh_solve gives the eigenvalues and the eigenvectors of a Hermitian
// matrix. The call reads one triangle of the operand.
fn c_eigh_solve(a: f64[:], n: i64, lower: bool, w: f64[>..]&, v: f64[>..]&) {
    let scale = c_scale_of(a);
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    c_hermitian_from_triangle(a, n, lower, scale, hr, hi);
    var vr: f64[>..] = [];
    var vi: f64[>..] = [];
    var vals: f64[>..] = [];
    c_hermitian_jacobi(hr, hi, n, vals, vr, vi);
    var i: i64 = 0;
    while i < n {
        w.push(vals[i] * scale);
        i += 1;
    }
    i = 0;
    while i < n * n {
        v.push(vr[i]);
        v.push(vi[i]);
        i += 1;
    }
    return;
}

// c_uplo reads the choice of the triangle. It gives 1 for "L" or "l", 2 for
// "U" or "u" and 0 for any other text, as NumPy does.
fn c_uplo(uplo: const u8[:]) -> i64 {
    if uplo.len != 1 { return 0; }
    if uplo[0] == 76 || uplo[0] == 108 { return 1; }
    if uplo[0] == 85 || uplo[0] == 117 { return 2; }
    return 0;
}

// ceigh gives the real eigenvalues in ascending order and the orthonormal
// eigenvectors of a Hermitian matrix. The call reads the lower triangle, or
// the upper triangle when uplo is "U", and takes the real part of the
// diagonal. NumPy fixes no phase of the vectors. The package makes the largest
// component of each vector real and above zero.
fn ceigh(a, uplo: const u8[:] = "L") -> Arr, CArr, Fault {
    let f = c_eig_check(a, "ceigh");
    if f.kind != F_NONE { return empty_arr(), carr_empty(), f; }
    let side = c_uplo(uplo);
    if side == 0 {
        return empty_arr(), carr_empty(), fault(F_VALUE, "ceigh", "the choice of the triangle is not L or U");
    }
    let n = a.shape[0];
    var w: f64[>..] = [];
    var v: f64[>..] = [];
    c_eigh_solve(a.data, n, side == 1, w, v);
    let vals, fv = from_values(w);
    if fv.kind != F_NONE { return empty_arr(), carr_empty(), fv; }
    return vals, CArr { ndim: 2, shape: c_eig_shape(n, 2), data: v }, no_fault();
}

// ceigvalsh gives the real eigenvalues of a Hermitian matrix in ascending
// order.
fn ceigvalsh(a, uplo: const u8[:] = "L") -> Arr, Fault {
    let vals, v, f = ceigh(a, uplo);
    return vals, f;
}
