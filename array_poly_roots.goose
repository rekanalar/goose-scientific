// The roots of a series of an orthogonal basis.
//
// The roots are the eigenvalues of the scaled companion matrix of the
// family. NumPy rotates that matrix by 180 degrees before it solves,
// which reduces the error, and this code does the same. The monomial
// route loses far more digits at high degree.
namespace scientific;

import mathshim;

// The roots of a series of a higher degree than this give a fault. The
// eigenvalue solver of the package stops at the same side.
const PB_ROOTS_MAX_DEGREE: i64 = 512;

// pb_chain_scale gives the scale of place i for Hermite and HermiteE.
// The scales of the places are the running product of the factors
// 1, 1/s(n-1), 1/s(n-2), ... in reverse order.
fn pb_chain_scale(fam: i64, n: i64) -> f64[>..] {
    var out = pb_zeros(n);
    var run: f64 = 1.0;
    out[n - 1] = 1.0;
    var i: i64 = 1;
    while i < n {
        let m = (n - i) as f64;
        var d = m;
        if fam == PB_HERM { d = 2.0 * m; }
        run = run / mathshim::root(d);
        out[n - 1 - i] = run;
        i += 1;
    }
    return out;
}

// pb_companion gives the scaled companion matrix of a series of degree
// n at least 2, row by row. The series holds no trailing zero.
fn pb_companion(fam: i64, c: f64[:]) -> f64[>..] {
    let n = c.len - 1;
    var mat = pb_zeros(n * n);
    let lead = c[n];
    var i: i64 = 0;
    if fam == PB_LEG {
        var scl = pb_zeros(n);
        while i < n { scl[i] = 1.0 / mathshim::root(2 * i + 1); i += 1; }
        i = 0;
        while i < n - 1 {
            let v = (i + 1) * scl[i] * scl[i + 1];
            mat[i * n + i + 1] = v;
            mat[(i + 1) * n + i] = v;
            i += 1;
        }
        i = 0;
        let fac = (n as f64) / (2 * n - 1);
        while i < n { mat[i * n + n - 1] -= (c[i] / lead) * (scl[i] / scl[n - 1]) * fac; i += 1; }
    } else if fam == PB_CHEB {
        var scl = pb_zeros(n);
        let h = mathshim::root(0.5);
        while i < n { scl[i] = h; i += 1; }
        scl[0] = 1.0;
        i = 0;
        while i < n - 1 {
            var v = 0.5;
            if i == 0 { v = h; }
            mat[i * n + i + 1] = v;
            mat[(i + 1) * n + i] = v;
            i += 1;
        }
        i = 0;
        while i < n { mat[i * n + n - 1] -= (c[i] / lead) * (scl[i] / scl[n - 1]) * 0.5; i += 1; }
    } else if fam == PB_HERM || fam == PB_HERME {
        let scl = pb_chain_scale(fam, n);
        while i < n - 1 {
            var v = ((i + 1) as f64);
            if fam == PB_HERM { v = 0.5 * v; }
            v = mathshim::root(v);
            mat[i * n + i + 1] = v;
            mat[(i + 1) * n + i] = v;
            i += 1;
        }
        i = 0;
        var den = lead;
        if fam == PB_HERM { den = 2.0 * lead; }
        while i < n { mat[i * n + n - 1] -= scl[i] * c[i] / den; i += 1; }
    } else {
        while i < n {
            mat[i * n + i] = 2 * i + 1;
            if i < n - 1 {
                mat[i * n + i + 1] = -(i + 1);
                mat[(i + 1) * n + i] = -(i + 1);
            }
            i += 1;
        }
        i = 0;
        while i < n { mat[i * n + n - 1] += (c[i] / lead) * n; i += 1; }
    }
    return mat;
}

// pb_rotate gives the matrix turned by 180 degrees.
fn pb_rotate(mat: f64[:], n: i64) -> f64[>..] {
    var out = pb_zeros(n * n);
    var i: i64 = 0;
    while i < n * n { out[i] = mat[n * n - 1 - i]; i += 1; }
    return out;
}

// pb_one_root gives the root of a series of degree one.
fn pb_one_root(fam: i64, c: f64[:]) -> f64 {
    if fam == PB_HERM { return -0.5 * c[0] / c[1]; }
    if fam == PB_LAG { return 1.0 + c[0] / c[1]; }
    return -c[0] / c[1];
}

// pb_roots checks a series and gives its roots as a real part and an
// imaginary part. The order is by the real part, then the imaginary part.
fn pb_roots(fam: i64, c: f64[:], tag: const u8[:]) -> Arr, Arr, Fault {
    let f = pb_check_series(c, tag);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    let len = poly_trim(c);
    if len < 2 { return line_arr([]), line_arr([]), no_fault(); }
    if len - 1 > PB_ROOTS_MAX_DEGREE {
        return empty_arr(), empty_arr(), fault(F_LIMIT, tag, "the degree is above 512");
    }
    var cut: f64[>..] = [];
    var i: i64 = 0;
    while i < len { cut.push(c[i]); i += 1; }
    if len == 2 {
        let r = pb_one_root(fam, cut);
        if !pb_finite(r) { return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the root is not finite"); }
        return line_arr([r]), line_arr([0.0]), no_fault();
    }
    let n = len - 1;
    let comp = pb_companion(fam, cut);
    if !pb_all_finite(comp) { return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the companion matrix is not finite"); }
    let ev, fe = eigvals_complex(mat2(pb_rotate(comp, n), n, n));
    if fe.kind != F_NONE { return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the eigenvalue solver failed"); }
    var re: f64[>..] = [];
    var im: f64[>..] = [];
    i = 0;
    while i < n { re.push(ev.data[2 * i]); im.push(ev.data[2 * i + 1]); i += 1; }
    if !pb_all_finite(re) || !pb_all_finite(im) {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "a root is not finite");
    }
    // A series whose roots are all real gives imaginary parts of zero. The
    // solver may leave a small part behind, so it goes when it lies below
    // a billionth of the size of the roots.
    var scale: f64 = 1.0;
    i = 0;
    while i < n {
        let a = mathshim::abs_of(re[i]);
        if a > scale { scale = a; }
        i += 1;
    }
    var all_real = true;
    i = 0;
    while i < n {
        if mathshim::abs_of(im[i]) > 1.0e-12 * scale { all_real = false; }
        i += 1;
    }
    if all_real {
        i = 0;
        while i < n { im[i] = 0.0; i += 1; }
    }
    return line_arr(copy(re)), line_arr(copy(im)), no_fault();
}
