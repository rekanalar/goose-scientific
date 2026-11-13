// The Schur decomposition and the LDL decomposition.
//
namespace scientific;

import mathshim;

// schur_split reduces a matrix to the upper Hessenberg form and turns
// the reflectors of the reduction into the change of the basis. The
// matrix h holds the form and the matrix z holds the change, so the
// caller holds a = z h z^t at the end of the reduction.
fn schur_split(a: f64[:], n: i64, h: f64[>..]&, z: f64[>..]&) {
    var i: i64 = 0;
    while i < n * n { h.push(a[i]); i += 1; }
    i = 0;
    while i < n {
        var c: i64 = 0;
        while c < n {
            if c == i { z.push(1.0); } else { z.push(0.0); }
            c += 1;
        }
        i += 1;
    }
    var k: i64 = 0;
    while k < n - 2 {
        var norm_h: f64 = 0.0;
        var r: i64 = k + 1;
        while r < n {
            norm_h += h[r * n + k] * h[r * n + k];
            r += 1;
        }
        norm_h = mathshim::root(norm_h);
        if norm_h > 0.0 {
            var alpha: f64 = -norm_h;
            if h[(k + 1) * n + k] > 0.0 { alpha = norm_h; }
            var v: f64[>..] = [];
            r = k + 1;
            while r < n {
                v.push(h[r * n + k]);
                r += 1;
            }
            v[0] = v[0] - alpha;
            var norm_v: f64 = 0.0;
            r = 0;
            while r < v.len {
                norm_v += v[r] * v[r];
                r += 1;
            }
            norm_v = mathshim::root(norm_v);
            if norm_v > 0.0 {
                r = 0;
                while r < v.len { v[r] = v[r] / norm_v; r += 1; }
                var c: i64 = k;
                while c < n {
                    var s: f64 = 0.0;
                    r = 0;
                    while r < v.len { s += v[r] * h[(k + 1 + r) * n + c]; r += 1; }
                    r = 0;
                    while r < v.len { h[(k + 1 + r) * n + c] = h[(k + 1 + r) * n + c] - 2.0 * s * v[r]; r += 1; }
                    c += 1;
                }
                c = 0;
                while c < n {
                    var s: f64 = 0.0;
                    r = 0;
                    while r < v.len { s += v[r] * h[c * n + k + 1 + r]; r += 1; }
                    r = 0;
                    while r < v.len { h[c * n + k + 1 + r] = h[c * n + k + 1 + r] - 2.0 * s * v[r]; r += 1; }
                    c += 1;
                }
                // The change of the basis takes the turn on the right.
                i = 0;
                while i < n {
                    var s2: f64 = 0.0;
                    r = 0;
                    while r < v.len { s2 += v[r] * z[i * n + k + 1 + r]; r += 1; }
                    r = 0;
                    while r < v.len { z[i * n + k + 1 + r] = z[i * n + k + 1 + r] - 2.0 * s2 * v[r]; r += 1; }
                    i += 1;
                }
            }
        }
        k += 1;
    }
    return;
}

// schur_reflect builds one reflector from the values of v and
// applies it to the rows k to k + nr - 1 and the columns k to
// k + nr - 1. The caller hands over the values in v and takes the
// reflector of the QR algorithm.
fn schur_reflect(h: f64[>..]&, z: f64[>..]&, n: i64, k: i64, m: i64, i: i64, v: f64[>..]&) {
    let nr = if i - k + 1 < 3 { i - k + 1 } else { 3 };
    // The reflector maps the first value of v onto beta and the rest
    // onto zero. tau holds the factor of the reflection.
    let x0 = v[0];
    var norm: f64 = 0.0;
    var j: i64 = 1;
    while j < nr {
        norm += v[j] * v[j];
        j += 1;
    }
    norm = mathshim::root(norm);
    var beta: f64 = x0;
    var tau: f64 = 0.0;
    if norm > 0.0 {
        var sgn: f64 = -1.0;
        if x0 < 0.0 { sgn = 1.0; }
        beta = sgn * mathshim::root(x0 * x0 + norm * norm);
        tau = (beta - x0) / beta;
        let d = x0 - beta;
        j = 1;
        while j < nr {
            v[j] = v[j] / d;
            j += 1;
        }
    }
    // The bulge of the last turn leaves the column k - 1 in place of
    // the values that the reflector removes.
    if k > m {
        h[k * n + (k - 1)] = beta;
        h[(k + 1) * n + (k - 1)] = 0.0;
        if k < i - 1 {
            h[(k + 2) * n + (k - 1)] = 0.0;
        }
    }
    if tau == 0.0 { return; }
    let t0 = tau;
    let t1 = tau * v[1];
    if nr == 3 {
        let t2 = tau * v[2];
        // The rows take the reflector from the left.
        var c: i64 = k;
        while c < n {
            let s0 = h[k * n + c] + v[1] * h[(k + 1) * n + c] + v[2] * h[(k + 2) * n + c];
            h[k * n + c] -= s0 * t0;
            h[(k + 1) * n + c] -= s0 * t1;
            h[(k + 2) * n + c] -= s0 * t2;
            c += 1;
        }
        // The columns take the reflector from the right.
        let rtop = if k + 3 < i { k + 3 } else { i };
        var rr: i64 = 0;
        while rr <= rtop {
            let s0 = h[rr * n + k] + v[1] * h[rr * n + k + 1] + v[2] * h[rr * n + k + 2];
            h[rr * n + k] -= s0 * t0;
            h[rr * n + k + 1] -= s0 * t1;
            h[rr * n + k + 2] -= s0 * t2;
            rr += 1;
        }
        // The change of the basis takes the same reflector.
        var zi: i64 = 0;
        while zi < n {
            let s0 = z[zi * n + k] + v[1] * z[zi * n + k + 1] + v[2] * z[zi * n + k + 2];
            z[zi * n + k] -= s0 * t0;
            z[zi * n + k + 1] -= s0 * t1;
            z[zi * n + k + 2] -= s0 * t2;
            zi += 1;
        }
        return;
    }
    // Two values: the rows and the columns take the shorter reflector.
    var c: i64 = k;
    while c < n {
        let s0 = h[k * n + c] + v[1] * h[(k + 1) * n + c];
        h[k * n + c] -= s0 * t0;
        h[(k + 1) * n + c] -= s0 * t1;
        c += 1;
    }
    var rr: i64 = 0;
    while rr <= i {
        let s0 = h[rr * n + k] + v[1] * h[rr * n + k + 1];
        h[rr * n + k] -= s0 * t0;
        h[rr * n + k + 1] -= s0 * t1;
        rr += 1;
    }
    var zi: i64 = 0;
    while zi < n {
        let s0 = z[zi * n + k] + v[1] * z[zi * n + k + 1];
        z[zi * n + k] -= s0 * t0;
        z[zi * n + k + 1] -= s0 * t1;
        zi += 1;
    }
    return;
}

// schur_split2 turns a block of two rows with two real values into
// two blocks of one row. The rotation holds the eigenvector of the
// first value. A pair of complex values stays as it is.
fn schur_split2(h: f64[>..]&, z: f64[>..]&, n: i64, p: i64) {
    let a = h[p * n + p];
    let b = h[p * n + (p + 1)];
    let c = h[(p + 1) * n + p];
    let d = h[(p + 1) * n + (p + 1)];
    if c == 0.0 { return; }
    let half = (a - d) / 2.0;
    let disc = half * half + b * c;
    if disc < 0.0 { return; }
    let r = mathshim::root(disc);
    var mu: f64 = d + half - r;
    if half < 0.0 { mu = d + half + r; }
    var cs: f64 = 0.0;
    var sn: f64 = 0.0;
    if b == 0.0 {
        let rr = mathshim::root(a * a + c * c);
        cs = a / rr;
        sn = c / rr;
    } else {
        let dd = mu - a;
        let rr = mathshim::root(b * b + dd * dd);
        cs = b / rr;
        sn = dd / rr;
    }
    // The columns take the rotation from the right.
    var i: i64 = 0;
    while i < n {
        let x = h[i * n + p];
        let y = h[i * n + (p + 1)];
        h[i * n + p] = cs * x + sn * y;
        h[i * n + (p + 1)] = -sn * x + cs * y;
        i += 1;
    }
    // The rows take the rotation from the left.
    var j: i64 = 0;
    while j < n {
        let x = h[p * n + j];
        let y = h[(p + 1) * n + j];
        h[p * n + j] = cs * x + sn * y;
        h[(p + 1) * n + j] = -sn * x + cs * y;
        j += 1;
    }
    // The change of the basis takes the same rotation.
    i = 0;
    while i < n {
        let x = z[i * n + p];
        let y = z[i * n + (p + 1)];
        z[i * n + p] = cs * x + sn * y;
        z[i * n + (p + 1)] = -sn * x + cs * y;
        i += 1;
    }
    h[(p + 1) * n + p] = 0.0;
    return;
}

// schur gives the real Schur form of a matrix. The result holds the
// quasi-triangular matrix T with blocks of one and two rows on the
// diagonal, and the orthogonal matrix Z with a = z t z^t. The columns
// of Z hold no fixed sign, because the form is not unique. The
// iteration follows the double shift of the QR algorithm.
fn schur(a) -> Arr, Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "schur", "the operand holds a count of dimensions that is not two");
    }
    if a.shape[0] != a.shape[1] {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "schur", "the operand is not a square matrix");
    }
    let n = a.shape[0];
    var h: f64[>..] = [];
    var z: f64[>..] = [];
    schur_split(a.data, n, h, z);
    let ulp = 2.220446049250313e-16;
    let itmax = 30 * (if 10 > n { 10 } else { n });
    var kdefl: i64 = 0;
    var i: i64 = n - 1;
    while i >= 0 {
        var l: i64 = 0;
        var converged = false;
        var its: i64 = 0;
        while its <= itmax {
            // Look for one small subdiagonal value.
            var k: i64 = i;
            while k > l {
                var tst: f64 = mathshim::root(h[(k - 1) * n + (k - 1)] * h[(k - 1) * n + (k - 1)]) + mathshim::root(h[k * n + k] * h[k * n + k]);
                if k - 2 >= 0 {
                    tst += mathshim::root(h[(k - 1) * n + (k - 2)] * h[(k - 1) * n + (k - 2)]);
                }
                if k + 1 <= n - 1 {
                    tst += mathshim::root(h[(k + 1) * n + k] * h[(k + 1) * n + k]);
                }
                let hk = mathshim::root(h[k * n + (k - 1)] * h[k * n + (k - 1)]);
                if hk <= ulp * tst {
                    var ab = mathshim::root(h[k * n + (k - 1)] * h[k * n + (k - 1)]);
                    var ba = mathshim::root(h[(k - 1) * n + k] * h[(k - 1) * n + k]);
                    if ba > ab {
                        let tmp = ab;
                        ab = ba;
                        ba = tmp;
                    }
                    var aa = mathshim::root(h[k * n + k] * h[k * n + k]);
                    let dd = h[(k - 1) * n + (k - 1)] - h[k * n + k];
                    let dv = mathshim::root(dd * dd);
                    if dv > aa { aa = dv; }
                    var bb = aa;
                    if dv < bb { bb = dv; }
                    let s = aa + ab;
                    if s == 0.0 { break; }
                    if ab / s * ba <= aa / s * bb * ulp { break; }
                }
                k -= 1;
            }
            l = k;
            if l > 0 {
                h[l * n + (l - 1)] = 0.0;
            }
            if l >= i - 1 {
                converged = true;
                break;
            }
            kdefl += 1;
            // The shifts come from the last two rows of the block, with
            // an exceptional shift every ten turns of the deflation.
            var h11: f64 = 0.0;
            var h12: f64 = 0.0;
            var h21: f64 = 0.0;
            var h22: f64 = 0.0;
            if kdefl % 20 == 0 {
                let s1 = mathshim::root(h[i * n + (i - 1)] * h[i * n + (i - 1)]) + mathshim::root(h[(i - 1) * n + (i - 2)] * h[(i - 1) * n + (i - 2)]);
                h11 = 0.75 * s1 + h[i * n + i];
                h12 = -0.4375 * s1;
                h21 = s1;
                h22 = h11;
            } else if kdefl % 10 == 0 {
                let s1 = mathshim::root(h[(l + 1) * n + l] * h[(l + 1) * n + l]) + mathshim::root(h[(l + 2) * n + (l + 1)] * h[(l + 2) * n + (l + 1)]);
                h11 = 0.75 * s1 + h[l * n + l];
                h12 = -0.4375 * s1;
                h21 = s1;
                h22 = h11;
            } else {
                h11 = h[(i - 1) * n + (i - 1)];
                h21 = h[i * n + (i - 1)];
                h12 = h[(i - 1) * n + i];
                h22 = h[i * n + i];
            }
            // The two shifts of the iteration.
            var s = mathshim::root(h11 * h11) + mathshim::root(h12 * h12) + mathshim::root(h21 * h21) + mathshim::root(h22 * h22);
            var rt1r: f64 = 0.0;
            var rt1i: f64 = 0.0;
            var rt2r: f64 = 0.0;
            var rt2i: f64 = 0.0;
            if s != 0.0 {
                h11 = h11 / s;
                h21 = h21 / s;
                h12 = h12 / s;
                h22 = h22 / s;
                let tr = (h11 + h22) / 2.0;
                let det = (h11 - tr) * (h22 - tr) - h12 * h21;
                let rtdisc = mathshim::root(if det >= 0.0 { det } else { -det });
                if det >= 0.0 {
                    rt1r = tr * s;
                    rt2r = rt1r;
                    rt1i = rtdisc * s;
                    rt2i = -rt1i;
                } else {
                    rt1r = tr + rtdisc;
                    rt2r = tr - rtdisc;
                    if mathshim::root((rt1r - h22) * (rt1r - h22)) <= mathshim::root((rt2r - h22) * (rt2r - h22)) {
                        rt1r = rt1r * s;
                        rt2r = rt1r;
                    } else {
                        rt2r = rt2r * s;
                        rt1r = rt2r;
                    }
                }
            }
            // The turn starts at the row where two small values sit.
            var m: i64 = l;
            var v: f64[>..] = [];
            v.push(0.0);
            v.push(0.0);
            v.push(0.0);
            var mm: i64 = i - 2;
            while mm >= l {
                var h21s = h[(mm + 1) * n + mm];
                let s1 = mathshim::root((h[mm * n + mm] - rt2r) * (h[mm * n + mm] - rt2r)) + mathshim::root(rt2i * rt2i) + mathshim::root(h21s * h21s);
                h21s = h21s / s1;
                v[0] = h21s * h[mm * n + (mm + 1)] + (h[mm * n + mm] - rt1r) * ((h[mm * n + mm] - rt2r) / s1) - rt2i / s1 * rt1i;
                v[1] = h21s * (h[mm * n + mm] + h[(mm + 1) * n + (mm + 1)] - rt1r - rt2r);
                v[2] = h21s * h[(mm + 2) * n + (mm + 1)];
                let s2 = mathshim::root(v[0] * v[0]) + mathshim::root(v[1] * v[1]) + mathshim::root(v[2] * v[2]);
                v[0] = v[0] / s2;
                v[1] = v[1] / s2;
                v[2] = v[2] / s2;
                m = mm;
                if mm == l { break; }
                let dsum = mathshim::root(h[(mm - 1) * n + (mm - 1)] * h[(mm - 1) * n + (mm - 1)]) + mathshim::root(h[mm * n + mm] * h[mm * n + mm]) + mathshim::root(h[(mm + 1) * n + (mm + 1)] * h[(mm + 1) * n + (mm + 1)]);
                if mathshim::root(h[mm * n + (mm - 1)] * h[mm * n + (mm - 1)]) * (mathshim::root(v[1] * v[1]) + mathshim::root(v[2] * v[2])) <= ulp * mathshim::root(v[0] * v[0]) * dsum {
                    break;
                }
                mm -= 1;
            }
            // The chase of the bulge.
            k = m;
            while k < i {
                if k > m {
                    let nr = if i - k + 1 < 3 { i - k + 1 } else { 3 };
                    var j: i64 = 0;
                    while j < nr {
                        v[j] = h[(k + j) * n + (k - 1)];
                        j += 1;
                    }
                }
                schur_reflect(h, z, n, k, m, i, v);
                k += 1;
            }
            its += 1;
        }
        if !converged {
            return empty_arr(), empty_arr(), fault(F_LIMIT, "schur", "the iteration did not settle");
        }
        if l == i - 1 {
            schur_split2(h, z, n, i - 1);
        }
        kdefl = 0;
        i = l - 1;
    }
    var sq: i64[..8] = [];
    sq.push(n);
    sq.push(n);
    return Arr { ndim: 2, shape: sq, data: copy(h) }, Arr { ndim: 2, shape: sq, data: copy(z) }, no_fault();
}

// ldl gives the factorization of a symmetric matrix into a unit lower
// triangle, a diagonal, and the transpose of the triangle. The method
// holds no pivoting, so a matrix that needs pivoting is a fault.
fn ldl(a) -> Arr, Arr, Fault {
    if a.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "ldl", "the operand holds a count of dimensions that is not two");
    }
    if a.shape[0] != a.shape[1] {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "ldl", "the operand is not a square matrix");
    }
    let n = a.shape[0];
    if !is_symmetric(a, n) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "ldl", "the operand is not symmetric");
    }
    var scale: f64 = 0.0;
    var i: i64 = 0;
    while i < n * n {
        var v = a.data[i];
        if v < 0.0 { v = -v; }
        if v > scale { scale = v; }
        i += 1;
    }
    if scale == 0.0 { scale = 1.0; }
    var lo: f64[>..] = [];
    var di: f64[>..] = [];
    i = 0;
    while i < n * n { lo.push(0.0); di.push(0.0); i += 1; }
    var j: i64 = 0;
    while j < n {
        var s: f64 = a.data[j * n + j];
        var k: i64 = 0;
        while k < j {
            s -= lo[j * n + k] * lo[j * n + k] * di[k * n + k];
            k += 1;
        }
        if mathshim::root(s * s) <= 1.0e-300 * scale {
            return empty_arr(), empty_arr(), fault(F_VALUE, "ldl", "the matrix needs pivoting");
        }
        di[j * n + j] = s;
        lo[j * n + j] = 1.0;
        i = j + 1;
        while i < n {
            var s2: f64 = a.data[i * n + j];
            k = 0;
            while k < j {
                s2 -= lo[i * n + k] * lo[j * n + k] * di[k * n + k];
                k += 1;
            }
            lo[i * n + j] = s2 / s;
            i += 1;
        }
        j += 1;
    }
    var sq: i64[..8] = [];
    sq.push(n);
    sq.push(n);
    return Arr { ndim: 2, shape: sq, data: copy(lo) }, Arr { ndim: 2, shape: sq, data: copy(di) }, no_fault();
}
