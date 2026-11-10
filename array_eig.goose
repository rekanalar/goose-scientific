// The eigenvalues and the eigenvectors.
//
// The eigenvalues of a symmetric matrix come from the Jacobi method.
// The eigenvalues of a matrix that is not symmetric come from the
// unshifted QR algorithm. The package holds real numbers only, so a
// complex pair is a fault.
namespace scientific;

import mathshim;

// jacobi finds the eigenvalues and the eigenvectors of a symmetric
// matrix. The eigenvalues come in the order of their value, from the
// smallest to the largest. The eigenvectors are the columns of the
// second result.
fn jacobi(a, n: i64) -> Arr, Arr, Fault {
    // The matrix of the work and the matrix of the eigenvectors.
    var m: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { m.push(a.data[i]); i += 1; }
    var v: f64[>..] = [];
    i = 0;
    while i < n * n { v.push(0.0); i += 1; }
    i = 0;
    while i < n { v[i * n + i] = 1.0; i += 1; }
    // The largest element off the diagonal gives the next turn.
    // Each turn clears one pair of elements, so the count of turns grows
    // with the square of the side. The stop value follows the scale of
    // the matrix.
    var scale: f64 = 0.0;
    i = 0;
    while i < n * n {
        let a = mathshim::abs_of(m[i]);
        if a > scale { scale = a; }
        i += 1;
    }
    let stop = 1.0e-15 * scale;
    var sweep: i64 = 0;
    while sweep < 60 * n * n + 100 {
        var p: i64 = 0;
        var q: i64 = 0;
        var best: f64 = 0.0;
        var r: i64 = 0;
        while r < n {
            var c: i64 = r + 1;
            while c < n {
                let x = m[r * n + c];
                if x < 0.0 { let y = -x; if y > best { best = y; p = r; q = c; } }
                else if x > best { best = x; p = r; q = c; }
                c += 1;
            }
            r += 1;
        }
        if best <= stop { break; }
        // The turn of the plane.
        let app = m[p * n + p];
        let aqq = m[q * n + q];
        let apq = m[p * n + q];
        let theta = (aqq - app) / (2.0 * apq);
        var t: f64 = 1.0;
        if theta >= 0.0 { t = 1.0 / (theta + mathshim::root(theta * theta + 1.0)); }
        else { t = -1.0 / (-theta + mathshim::root(theta * theta + 1.0)); }
        let c2 = 1.0 / mathshim::root(t * t + 1.0);
        let s2 = t * c2;
        // The turn touches the rows and the columns.
        var k: i64 = 0;
        while k < n {
            if k != p && k != q {
                let mkp = m[k * n + p];
                let mkq = m[k * n + q];
                m[k * n + p] = c2 * mkp - s2 * mkq;
                m[p * n + k] = m[k * n + p];
                m[k * n + q] = s2 * mkp + c2 * mkq;
                m[q * n + k] = m[k * n + q];
            }
            k += 1;
        }
        m[p * n + p] = c2 * c2 * app - 2.0 * s2 * c2 * apq + s2 * s2 * aqq;
        m[q * n + q] = s2 * s2 * app + 2.0 * s2 * c2 * apq + c2 * c2 * aqq;
        m[p * n + q] = 0.0;
        m[q * n + p] = 0.0;
        // The eigenvectors follow the turn.
        k = 0;
        while k < n {
            let vkp = v[k * n + p];
            let vkq = v[k * n + q];
            v[k * n + p] = c2 * vkp - s2 * vkq;
            v[k * n + q] = s2 * vkp + c2 * vkq;
            k += 1;
        }
        sweep += 1;
    }
    // The eigenvalues come in the order of their value.
    var order: i64[>..] = [];
    i = 0;
    while i < n { order.push(i); i += 1; }
    i = 1;
    while i < n {
        let key = order[i];
        var j = i - 1;
        while j >= 0 {
            let other = order[j];
            if m[other * n + other] > m[key * n + key] {
                order[j + 1] = other;
                j -= 1;
            } else { break; }
        }
        order[j + 1] = key;
        i += 1;
    }
    var vals: f64[>..] = [];
    var vecs: f64[>..] = [];
    i = 0;
    while i < n * n { vecs.push(0.0); i += 1; }
    i = 0;
    while i < n {
        vals.push(m[order[i] * n + order[i]]);
        var r2: i64 = 0;
        while r2 < n {
            vecs[r2 * n + i] = v[r2 * n + order[i]];
            r2 += 1;
        }
        i += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    var sv: i64[..8] = [];
    sv.push(n);
    sv.push(n);
    return Arr { ndim: 1, shape: s, data: vals }, Arr { ndim: 2, shape: sv, data: vecs }, no_fault();
}

// eigvalsh gives the eigenvalues of a symmetric matrix.
fn eigvalsh(a) -> Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), fault(F_SHAPE, "eigvalsh", "the matrix is not square");
    }
    let n = a.shape[0];
    let vals, vecs, f = jacobi(a, n);
    if f.kind != F_NONE { return empty_arr(), f; }
    return vals, no_fault();
}

// eigh gives the eigenvalues and the eigenvectors of a symmetric
// matrix.
fn eigh(a) -> Arr, Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "eigh", "the matrix is not square");
    }
    let n = a.shape[0];
    let vals, vecs, f = jacobi(a, n);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    return vals, vecs, no_fault();
}

// is_symmetric tells whether a matrix holds the same values above and
// below the diagonal.
fn is_symmetric(a, n: i64) -> bool {
    var i: i64 = 0;
    while i < n {
        var j: i64 = i + 1;
        while j < n {
            if a.data[i * n + j] != a.data[j * n + i] { return false; }
            j += 1;
        }
        i += 1;
    }
    return true;
}

// hessenberg reduces a matrix to the upper Hessenberg form, where the
// elements below the first diagonal of the lower triangle are zero.
// The similarity keeps the eigenvalues. The transform holds the
// orthogonal matrix of the change.
fn hessenberg(a: f64[:], n: i64, h: f64[>..]&) {
    var i: i64 = 0;
    while i < n * n { h.push(a[i]); i += 1; }
    var k: i64 = 0;
    while k < n - 2 {
        // The length of the part of the column that is below the turn.
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
                // The turn touches the rows on the right.
                var c: i64 = k;
                while c < n {
                    var s: f64 = 0.0;
                    r = 0;
                    while r < v.len { s += v[r] * h[(k + 1 + r) * n + c]; r += 1; }
                    r = 0;
                    while r < v.len { h[(k + 1 + r) * n + c] = h[(k + 1 + r) * n + c] - 2.0 * s * v[r]; r += 1; }
                    c += 1;
                }
                // The turn touches the columns below.
                c = 0;
                while c < n {
                    var s: f64 = 0.0;
                    r = 0;
                    while r < v.len { s += v[r] * h[c * n + k + 1 + r]; r += 1; }
                    r = 0;
                    while r < v.len { h[c * n + k + 1 + r] = h[c * n + k + 1 + r] - 2.0 * s * v[r]; r += 1; }
                    c += 1;
                }
            }
        }
        k += 1;
    }
    return;
}

// qr_eigen finds the eigenvalues of a matrix that is not symmetric.
// The method is the QR algorithm with a shift. A complex pair is a
// fault.
fn qr_eigen(a, n: i64) -> Arr, Fault {
    var h: f64[>..] = [];
    hessenberg(a.data, n, h);
    var sweep: i64 = 0;
    while sweep < 500 {
        // The shift comes from the last two rows of the matrix.
        var shift: f64 = h[(n - 1) * n + (n - 1)];
        if n > 1 {
            let app = h[(n - 2) * n + (n - 2)];
            let aqq = h[(n - 1) * n + (n - 1)];
            let apq = h[(n - 2) * n + (n - 1)];
            let tr = (app + aqq) / 2.0;
            let disc = tr * tr - (app * aqq - apq * apq);
            if disc > 0.0 {
                let root = mathshim::root(disc);
                var mu = tr - root;
                if tr < 0.0 { mu = tr + root; }
                shift = mu;
            } else {
                shift = tr;
            }
        }
        // The matrix of the work takes the shift out.
        var work: f64[>..] = [];
        var i: i64 = 0;
        while i < n * n { work.push(h[i]); i += 1; }
        i = 0;
        while i < n { work[i * n + i] -= shift; i += 1; }
        // The QR decomposition of the matrix.
        var qq: f64[>..] = [];
        var rr: f64[>..] = [];
        householder_qr(work, n, n, qq, rr);
        // The matrix takes the product of R and Q, and the shift goes
        // back onto the diagonal.
        var next: f64[>..] = [];
        i = 0;
        while i < n * n { next.push(0.0); i += 1; }
        var x: i64 = 0;
        while x < n {
            var y: i64 = 0;
            while y < n {
                var t: f64 = 0.0;
                var z: i64 = 0;
                while z < n {
                    t += rr[x * n + z] * qq[z * n + y];
                    z += 1;
                }
                next[x * n + y] = t;
                y += 1;
            }
            x += 1;
        }
        i = 0;
        while i < n { next[i * n + i] += shift; i += 1; }
        // The change of the matrix must be small, else the loop goes
        // on.
        var change: f64 = 0.0;
        var scale: f64 = 0.0;
        i = 0;
        while i < n {
            let d = mathshim::root(next[i * n + i] * next[i * n + i]);
            if d > scale { scale = d; }
            if i + 1 < n {
                let e = mathshim::root(next[(i + 1) * n + (i + 1)] * next[(i + 1) * n + (i + 1)]);
                if e > scale { scale = e; }
            }
            i += 1;
        }
        i = 1;
        while i < n {
            let e = mathshim::root(next[i * n + i - 1] * next[i * n + i - 1]);
            if e > change { change = e; }
            i += 1;
        }
        h = copy(next);
        sweep += 1;
        if change <= 1e-12 * scale { break; }
        if scale == 0.0 { break; }
    }
    // The eigenvalues lie on the diagonal. A pair of neighbouring
    // values that hold a complex number is a fault.
    var vals: f64[>..] = [];
    var i2: i64 = 0;
    while i2 < n {
        let d = h[i2 * n + i2];
        if i2 + 1 < n {
            let e = h[(i2 + 1) * n + (i2 + 1)];
            let up = h[i2 * n + (i2 + 1)];
            let low = h[(i2 + 1) * n + i2];
            // The pair holds a complex number when the square of the
            // difference of the values is smaller than four times the
            // product of the off values.
            let half = (d - e) / 2.0;
            let disc = half * half + up * low;
            if disc < 0.0 {
                return empty_arr(), fault(F_VALUE, "eig", "the matrix holds a complex pair of eigenvalues");
            }
        }
        vals.push(d);
        i2 += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    return Arr { ndim: 1, shape: s, data: vals }, no_fault();
}

// null_vector gives the vector of an eigenvalue. The value is an
// eigenvalue of the matrix, found by the iteration. The method takes
// the matrix that holds the value on its diagonal.
fn null_vector(a, n: i64, value: f64) -> Arr, Fault {
    // The size of the matrix sets the step of the shift.
    var scale: f64 = 0.0;
    var i: i64 = 0;
    while i < a.data.len {
        let v = mathshim::root(a.data[i] * a.data[i]);
        if v > scale { scale = v; }
        i += 1;
    }
    // The shift keeps the matrix away from a zero pivot.
    let delta = 0.0000000001 * scale;
    let shifted = value + delta;
    // The matrix of the work.
    var m: f64[>..] = [];
    i = 0;
    while i < n * n { m.push(a.data[i]); i += 1; }
    i = 0;
    while i < n { m[i * n + i] -= shifted; i += 1; }
    // The elimination with a partial pivot. The order of the rows is
    // in perm.
    var perm: i64[>..] = [];
    i = 0;
    while i < n { perm.push(i); i += 1; }
    var col: i64 = 0;
    while col < n {
        var best = col;
        var best_val: f64 = 0.0;
        var r: i64 = col;
        while r < n {
            let v = mathshim::root(m[perm[r] * n + col] * m[perm[r] * n + col]);
            if v > best_val { best = r; best_val = v; }
            r += 1;
        }
        if best_val == 0.0 {
            return empty_arr(), fault(F_VALUE, "eig", "the matrix holds no pivot");
        }
        if best != col {
            let t = perm[col];
            perm[col] = perm[best];
            perm[best] = t;
        }
        r = col + 1;
        while r < n {
            let f = m[perm[r] * n + col] / m[perm[col] * n + col];
            m[perm[r] * n + col] = f;
            var c: i64 = col + 1;
            while c < n {
                m[perm[r] * n + c] = m[perm[r] * n + c] - f * m[perm[col] * n + c];
                c += 1;
            }
            r += 1;
        }
        col += 1;
    }
    // The iteration starts with a vector that no eigenvector is
    // parallel to.
    var x: f64[>..] = [];
    i = 0;
    while i < n { x.push((i + 1) as f64); i += 1; }
    var pass: i64 = 0;
    while pass < 4 {
        // The right side holds the vector of the last pass.
        var y: f64[>..] = [];
        i = 0;
        while i < n { y.push(x[perm[i]]); i += 1; }
        // The forward substitution.
        i = 1;
        while i < n {
            var t: f64 = y[i];
            var c: i64 = 0;
            while c < i {
                t -= m[perm[i] * n + c] * y[c];
                c += 1;
            }
            y[i] = t;
            i += 1;
        }
        // The back substitution.
        var z: f64[>..] = [];
        i = 0;
        while i < n { z.push(0.0); i += 1; }
        i = n - 1;
        while i >= 0 {
            var t: f64 = y[i];
            var c: i64 = i + 1;
            while c < n {
                t -= m[perm[i] * n + c] * z[c];
                c += 1;
            }
            z[i] = t / m[perm[i] * n + i];
            i -= 1;
        }
        // The vector holds a length of one.
        var length: f64 = 0.0;
        i = 0;
        while i < n {
            length += z[i] * z[i];
            i += 1;
        }
        length = mathshim::root(length);
        if length == 0.0 {
            return empty_arr(), fault(F_VALUE, "eig", "the matrix holds no vector for one eigenvalue");
        }
        i = 0;
        while i < n {
            x[i] = z[i] / length;
            i += 1;
        }
        pass += 1;
    }
    // The vector must hold the matrix near zero.
    var residual: f64 = 0.0;
    var r2: i64 = 0;
    while r2 < n {
        var t: f64 = 0.0;
        var c: i64 = 0;
        while c < n {
            t += a.data[r2 * n + c] * x[c];
            c += 1;
        }
        let e = mathshim::root((t - value * x[r2]) * (t - value * x[r2]));
        if e > residual { residual = e; }
        r2 += 1;
    }
    if residual > 0.000001 * scale {
        return empty_arr(), fault(F_VALUE, "eig", "the matrix holds no vector for one eigenvalue");
    }
    var d: f64[>..] = [];
    i = 0;
    while i < n {
        d.push(x[i]);
        i += 1;
    }
    var sh: i64[..8] = [];
    sh.push(n);
    return Arr { ndim: 1, shape: sh, data: d }, no_fault();
}


// eigvals gives the eigenvalues of a square matrix.
fn eigvals(a) -> Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), fault(F_SHAPE, "eigvals", "the matrix is not square");
    }
    let n = a.shape[0];
    if is_symmetric(a, n) {
        return eigvalsh(a);
    }
    let vals, f = qr_eigen(a, n);
    if f.kind != F_NONE { return empty_arr(), f; }
    return vals, no_fault();
}

// eig gives the eigenvalues and the eigenvectors of a square matrix.
fn eig(a) -> Arr, Arr, Fault {
    if a.ndim != 2 || a.shape[0] != a.shape[1] {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "eig", "the matrix is not square");
    }
    let n = a.shape[0];
    if is_symmetric(a, n) {
        let vals, vecs, f = eigh(a);
        if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
        return vals, vecs, no_fault();
    }
    let vals, f = qr_eigen(a, n);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    // The vector of an eigenvalue comes from the iteration. The
    // vectors are the columns of the result.
    var vecs_of: f64[>..] = [];
    var e: i64 = 0;
    while e < n {
        let v, fe = null_vector(a, n, vals.data[e]);
        if fe.kind != F_NONE { return empty_arr(), empty_arr(), fe; }
        var r: i64 = 0;
        while r < n {
            vecs_of.push(v.data[r]);
            r += 1;
        }
        e += 1;
    }
    var out: f64[>..] = [];
    var r2: i64 = 0;
    while r2 < n {
        var c: i64 = 0;
        while c < n {
            out.push(vecs_of[c * n + r2]);
            c += 1;
        }
        r2 += 1;
    }
    var s: i64[..8] = [];
    s.push(n);
    var sv: i64[..8] = [];
    sv.push(n);
    sv.push(n);
    return Arr { ndim: 1, shape: s, data: copy(vals.data) }, Arr { ndim: 2, shape: sv, data: out }, no_fault();
}
