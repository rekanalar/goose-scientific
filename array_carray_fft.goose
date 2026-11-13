// The Fourier transforms of complex arrays and the eigenvalues of a complex matrix.
//
namespace scientific;

import mathshim;

// cfft gives the transform of a complex signal of one dimension.
fn cfft(a) -> CArr, Fault {
    if a.ndim != 1 {
        return carr_empty(), fault(F_SHAPE, "cfft", "the operand is not a vector");
    }
    let re = carr_part(a, 0);
    let im = carr_part(a, 1);
    let r, i, f = fft_parts(re.data, im.data);
    if f.kind != F_NONE { return carr_empty(), f; }
    return complex_array(r, i);
}

// cifft gives the inverse transform of a complex signal.
fn cifft(a) -> CArr, Fault {
    if a.ndim != 1 {
        return carr_empty(), fault(F_SHAPE, "cifft", "the operand is not a vector");
    }
    let re = carr_part(a, 0);
    let im = carr_part(a, 1);
    let r, i, f = ifft(re.data, im.data);
    if f.kind != F_NONE { return carr_empty(), f; }
    return complex_array(r, i);
}

// cfftn gives the transform of a complex array along every axis. The
// norm backward leaves the result as it is.
fn cfftn(a) -> CArr, Fault {
    let f = carr_check(a, "cfftn");
    if f.kind != F_NONE { return carr_empty(), f; }
    if !carr_finite(a) {
        return carr_empty(), fault(F_VALUE, "cfftn", "the operand holds a value that is not finite");
    }
    let re = real(a);
    let im = imag(a);
    let r, i, g = fftn_ex(re, im);
    if g.kind != F_NONE { return carr_empty(), g; }
    return complex_array(r, i);
}

// cifftn gives the inverse transform of a complex array along every
// axis. The norm backward divides by the count of the places.
fn cifftn(a) -> CArr, Fault {
    let f = carr_check(a, "cifftn");
    if f.kind != F_NONE { return carr_empty(), f; }
    if !carr_finite(a) {
        return carr_empty(), fault(F_VALUE, "cifftn", "the operand holds a value that is not finite");
    }
    let re = real(a);
    let im = imag(a);
    let r, i, g = ifftn_ex(re, im);
    if g.kind != F_NONE { return carr_empty(), g; }
    return complex_array(r, i);
}
// hqr_value reads the element (i, j) of the work matrix. The indexes
// start at one, as in the method.
fn hqr_at(i: i64, j: i64, n: i64) -> i64 { return (i - 1) * n + (j - 1); }

fn hqr_sign(a: f64, b: f64) -> f64 {
    var x = mathshim::abs_of(a);
    if b < 0.0 { x = -x; }
    return x;
}

// eigvals_complex gives the eigenvalues of a real square matrix, which
// may hold complex pairs. The method is the Hessenberg form and the QR
// iteration with two shifts. The eigenvalues come sorted by the real
// part, then by the imaginary part. NumPy gives them in another order.
fn eigvals_complex(a) -> CArr, Fault {
    if a.ndim != 2 || a.shape.len != 2 || a.shape[0] != a.shape[1] || a.shape[0] < 1 {
        return carr_empty(), fault(F_SHAPE, "eigvals_complex", "the operand is not a square matrix");
    }
    let n = a.shape[0];
    if n > 512 {
        return carr_empty(), fault(F_LIMIT, "eigvals_complex", "the side is above 512");
    }
    let fb = check_stack(a.ndim, a.shape, a.data.len, 1, "eigvals_complex");
    if fb.kind != F_NONE { return carr_empty(), fb; }
    if !carr_finite(a) {
        return carr_empty(), fault(F_VALUE, "eigvals_complex", "the matrix holds a value that is not finite");
    }
    var h: f64[>..] = [];
    hessenberg(a.data, n, h);
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < i - 1 { h[i * n + j] = 0.0; j += 1; }
        i += 1;
    }
    var wr: f64[>..] = [];
    var wi: f64[>..] = [];
    i = 0;
    while i < n { wr.push(0.0); wi.push(0.0); i += 1; }
    var anorm: f64 = 0.0;
    i = 1;
    while i <= n {
        var j = i - 1;
        if j < 1 { j = 1; }
        while j <= n { anorm += mathshim::abs_of(h[hqr_at(i, j, n)]); j += 1; }
        i += 1;
    }
    var nn = n;
    var t: f64 = 0.0;
    var p: f64 = 0.0;
    var q: f64 = 0.0;
    var r: f64 = 0.0;
    var x: f64 = 0.0;
    var y: f64 = 0.0;
    var z: f64 = 0.0;
    var w: f64 = 0.0;
    var s: f64 = 0.0;
    while nn >= 1 {
        var its: i64 = 0;
        var again = true;
        while again {
            var l = nn;
            while l >= 2 {
                s = mathshim::abs_of(h[hqr_at(l - 1, l - 1, n)]) + mathshim::abs_of(h[hqr_at(l, l, n)]);
                if s == 0.0 { s = anorm; }
                if mathshim::abs_of(h[hqr_at(l, l - 1, n)]) + s == s {
                    h[hqr_at(l, l - 1, n)] = 0.0;
                    break;
                }
                l -= 1;
            }
            x = h[hqr_at(nn, nn, n)];
            if l == nn {
                wr[nn - 1] = x + t;
                wi[nn - 1] = 0.0;
                nn -= 1;
                again = false;
            } else {
                y = h[hqr_at(nn - 1, nn - 1, n)];
                w = h[hqr_at(nn, nn - 1, n)] * h[hqr_at(nn - 1, nn, n)];
                if l == nn - 1 {
                    p = 0.5 * (y - x);
                    q = p * p + w;
                    z = mathshim::root(mathshim::abs_of(q));
                    x = x + t;
                    if q >= 0.0 {
                        z = p + hqr_sign(z, p);
                        wr[nn - 2] = x + z;
                        wr[nn - 1] = wr[nn - 2];
                        if z != 0.0 { wr[nn - 1] = x - w / z; }
                        wi[nn - 2] = 0.0;
                        wi[nn - 1] = 0.0;
                    } else {
                        wr[nn - 2] = x + p;
                        wr[nn - 1] = x + p;
                        wi[nn - 2] = z;
                        wi[nn - 1] = -z;
                    }
                    nn -= 2;
                    again = false;
                } else {
                    if its >= 60 {
                        return carr_empty(), fault(F_LIMIT, "eigvals_complex", "the iteration did not converge");
                    }
                    if its == 10 || its == 20 {
                        t += x;
                        i = 1;
                        while i <= nn { h[hqr_at(i, i, n)] = h[hqr_at(i, i, n)] - x; i += 1; }
                        s = mathshim::abs_of(h[hqr_at(nn, nn - 1, n)]) + mathshim::abs_of(h[hqr_at(nn - 1, nn - 2, n)]);
                        x = 0.75 * s;
                        y = x;
                        w = -0.4375 * s * s;
                    }
                    its += 1;
                    var m = nn - 2;
                    var found = false;
                    while m >= l && !found {
                        z = h[hqr_at(m, m, n)];
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / h[hqr_at(m + 1, m, n)] + h[hqr_at(m, m + 1, n)];
                        q = h[hqr_at(m + 1, m + 1, n)] - z - r - s;
                        r = h[hqr_at(m + 2, m + 1, n)];
                        s = mathshim::abs_of(p) + mathshim::abs_of(q) + mathshim::abs_of(r);
                        p = p / s;
                        q = q / s;
                        r = r / s;
                        if m == l { found = true; }
                        else {
                            let u = mathshim::abs_of(h[hqr_at(m, m - 1, n)]) * (mathshim::abs_of(q) + mathshim::abs_of(r));
                            let v = mathshim::abs_of(p) * (mathshim::abs_of(h[hqr_at(m - 1, m - 1, n)]) + mathshim::abs_of(z) + mathshim::abs_of(h[hqr_at(m + 1, m + 1, n)]));
                            if u + v == v { found = true; } else { m -= 1; }
                        }
                    }
                    i = m + 2;
                    while i <= nn {
                        h[hqr_at(i, i - 2, n)] = 0.0;
                        if i != m + 2 { h[hqr_at(i, i - 3, n)] = 0.0; }
                        i += 1;
                    }
                    var k = m;
                    while k <= nn - 1 {
                        if k != m {
                            p = h[hqr_at(k, k - 1, n)];
                            q = h[hqr_at(k + 1, k - 1, n)];
                            r = 0.0;
                            if k != nn - 1 { r = h[hqr_at(k + 2, k - 1, n)]; }
                            x = mathshim::abs_of(p) + mathshim::abs_of(q) + mathshim::abs_of(r);
                            if x != 0.0 { p = p / x; q = q / x; r = r / x; }
                        }
                        s = hqr_sign(mathshim::root(p * p + q * q + r * r), p);
                        if s != 0.0 {
                            if k == m {
                                if l != m { h[hqr_at(k, k - 1, n)] = -h[hqr_at(k, k - 1, n)]; }
                            } else {
                                h[hqr_at(k, k - 1, n)] = -s * x;
                            }
                            p = p + s;
                            x = p / s;
                            y = q / s;
                            z = r / s;
                            q = q / p;
                            r = r / p;
                            var j = k;
                            while j <= nn {
                                p = h[hqr_at(k, j, n)] + q * h[hqr_at(k + 1, j, n)];
                                if k != nn - 1 {
                                    p = p + r * h[hqr_at(k + 2, j, n)];
                                    h[hqr_at(k + 2, j, n)] = h[hqr_at(k + 2, j, n)] - p * z;
                                }
                                h[hqr_at(k + 1, j, n)] = h[hqr_at(k + 1, j, n)] - p * y;
                                h[hqr_at(k, j, n)] = h[hqr_at(k, j, n)] - p * x;
                                j += 1;
                            }
                            var mmin = k + 3;
                            if nn < mmin { mmin = nn; }
                            i = l;
                            while i <= mmin {
                                p = x * h[hqr_at(i, k, n)] + y * h[hqr_at(i, k + 1, n)];
                                if k != nn - 1 {
                                    p = p + z * h[hqr_at(i, k + 2, n)];
                                    h[hqr_at(i, k + 2, n)] = h[hqr_at(i, k + 2, n)] - p * r;
                                }
                                h[hqr_at(i, k + 1, n)] = h[hqr_at(i, k + 1, n)] - p * q;
                                h[hqr_at(i, k, n)] = h[hqr_at(i, k, n)] - p;
                                i += 1;
                            }
                        }
                        k += 1;
                    }
                }
            }
        }
    }
    // Sort by the real part, then by the imaginary part.
    var keys: f64[>..] = [];
    i = 0;
    while i < n { keys.push(wi[i]); i += 1; }
    i = 0;
    while i < n { keys.push(wr[i]); i += 1; }
    let ord = order_places(keys, 2, n);
    var d: f64[>..] = [];
    i = 0;
    while i < n { d.push(wr[ord[i]]); d.push(wi[ord[i]]); i += 1; }
    var s1: i64[..8] = [];
    s1.push(n);
    return carr_make(s1, copy(d)), no_fault();
}

