// The eigenproblem of a complex Hermitian matrix.
//
namespace scientific;

import mathshim;

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
