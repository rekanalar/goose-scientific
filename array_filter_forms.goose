// The forms of a filter.
//
// A filter has three forms. The coefficients hold the numerator and the
// denominator, from the highest power to the lowest. The zeros, the
// poles and the gain hold the roots of the two polynomials and the
// scale. The state space form holds four matrices. The calls of this
// file change one form into another, as the calls of scipy.signal do:
// zpk2tf, tf2zpk, tf2ss, zpk2ss and ss2tf.
//
// A zero or a pole is a pair of two lists: the real parts and the
// imaginary parts. A real filter holds its roots in pairs of
// conjugates. The state space form holds one input and one output.
namespace scientific;

import mathshim;

// The limits of a filter: 64 zeros, 64 poles and 64 states, so 65
// coefficients.
const FORM_ROOTS_MAX: i64 = 64;
const FORM_COEF_MAX: i64 = 65;

// A leading coefficient of the numerator at or below this size goes, as
// in SciPy.
const FORM_TRIM: f64 = 1.0e-14;

// form_roots_check checks the two lists of a set of roots.
fn form_roots_check(xr: f64[:], xi: f64[:], tag: const u8[:]) -> Fault {
    if xr.len != xi.len {
        return fault(F_SHAPE, tag, "the real parts and the imaginary parts hold a different count");
    }
    if xr.len > FORM_ROOTS_MAX {
        return fault(F_LIMIT, tag, "the list holds more than 64 roots");
    }
    if !all_finite(xr) || !all_finite(xi) {
        return fault(F_VALUE, tag, "a root holds a value that is not finite");
    }
    return no_fault();
}

// form_normalize trims and scales the coefficients as SciPy does. It
// drops the leading zeros of the denominator, divides both lists by the
// first coefficient of the denominator, and drops the leading
// coefficients of the numerator that are at or below FORM_TRIM. One
// coefficient of the numerator always stays.
fn form_normalize(b: f64[:], a: f64[:], tag: const u8[:], nb: f64[>..]&, na: f64[>..]&) -> Fault {
    if b.len < 1 || a.len < 1 {
        return fault(F_VALUE, tag, "a list of coefficients holds no element");
    }
    if b.len > FORM_COEF_MAX || a.len > FORM_COEF_MAX {
        return fault(F_LIMIT, tag, "a list holds more than 65 coefficients");
    }
    if !all_finite(b) || !all_finite(a) {
        return fault(F_VALUE, tag, "a coefficient is not finite");
    }
    var i: i64 = 0;
    while i < a.len && a[i] == 0.0 { i += 1; }
    if i >= a.len {
        return fault(F_VALUE, tag, "the denominator holds only zeros");
    }
    let lead = a[i];
    var j = i;
    while j < a.len { na.push(a[j] / lead); j += 1; }
    var t: i64 = 0;
    while t < b.len - 1 && mathshim::abs_of(b[t] / lead) <= FORM_TRIM { t += 1; }
    j = t;
    while j < b.len { nb.push(b[j] / lead); j += 1; }
    if !all_finite(na) || !all_finite(nb) {
        return fault(F_VALUE, tag, "the division by the first coefficient overflows");
    }
    return no_fault();
}

// zpk2tf gives the numerator and the denominator of a filter from its
// zeros, its poles and its gain. The numerator is the gain times the
// product of (s - zero). The denominator is the product of (s - pole).
// A root without a conjugate partner gives F_SHAPE, because the
// coefficients would not be real.
fn zpk2tf(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64) -> Arr, Arr, Fault {
    let fz = form_roots_check(zr, zi, "zpk2tf");
    if fz.kind != F_NONE { return empty_arr(), empty_arr(), fz; }
    let fp = form_roots_check(pr, pi_, "zpk2tf");
    if fp.kind != F_NONE { return empty_arr(), empty_arr(), fp; }
    if !all_finite([k]) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "zpk2tf", "the gain is not finite");
    }
    let z = rs_zip(zr, zi);
    let p = rs_zip(pr, pi_);
    let b, a, f = zpk_to_ba(z, p, k);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), fault(f.kind, "zpk2tf", f.detail); }
    if !all_finite(b.data) || !all_finite(a.data) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "zpk2tf", "a coefficient overflows");
    }
    return b, a, no_fault();
}

// tf2zpk gives the zeros, the poles and the gain of a filter from its
// numerator and its denominator. The results are the real parts of the
// zeros, the imaginary parts of the zeros, the real parts of the
// poles, the imaginary parts of the poles and the gain. The order of
// the roots is the order of the package, not the order of SciPy.
fn tf2zpk(b: f64[:], a: f64[:]) -> Arr, Arr, Arr, Arr, f64, Fault {
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    let f = form_normalize(b, a, "tf2zpk", nb, na);
    if f.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, f;
    }
    let k = nb[0];
    if k == 0.0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "tf2zpk", "the numerator holds only zeros");
    }
    var i: i64 = 0;
    while i < nb.len { nb[i] = nb[i] / k; i += 1; }
    if !all_finite(nb) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "tf2zpk", "the division by the gain overflows");
    }
    let zr, zi, fz = roots(nb);
    if fz.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, fz;
    }
    let pr, pim, fp = roots(na);
    if fp.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0, fp;
    }
    if !all_finite(zr.data) || !all_finite(zi.data) || !all_finite(pr.data) || !all_finite(pim.data) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "tf2zpk", "a root is not finite");
    }
    return zr, zi, pr, pim, k, no_fault();
}

// form_zero1 gives the matrix of one row and one column that holds zero.
fn form_zero1() -> Arr {
    var d: f64[>..] = [0.0];
    return mat2(copy(d), 1, 1);
}

// tf2ss gives the state space form of a filter from its numerator and
// its denominator. The form is the controller canonical type. The
// results are the matrices A, B and C, and the direct part D. A
// numerator longer than its denominator gives F_SHAPE.
fn tf2ss(b: f64[:], a: f64[:]) -> Arr, Arr, Arr, f64, Fault {
    var nb: f64[>..] = [];
    var na: f64[>..] = [];
    let f = form_normalize(b, a, "tf2ss", nb, na);
    if f.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), 0.0, f;
    }
    if nb.len > na.len {
        return empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_SHAPE, "tf2ss", "the numerator holds more coefficients than the denominator");
    }
    // The numerator gets zeros at the front, up to the length of the denominator.
    var num: f64[>..] = [];
    var i: i64 = 0;
    while i < na.len - nb.len { num.push(0.0); i += 1; }
    i = 0;
    while i < nb.len { num.push(nb[i]); i += 1; }
    let n = na.len - 1;
    if n == 0 {
        return form_zero1(), form_zero1(), form_zero1(), num[0], no_fault();
    }
    var am: f64[>..] = [];
    var bm: f64[>..] = [];
    var cm: f64[>..] = [];
    i = 0;
    while i < n * n { am.push(0.0); i += 1; }
    i = 0;
    while i < n { bm.push(0.0); i += 1; }
    i = 0;
    while i < n {
        am[i] = -na[i + 1];
        cm.push(num[i + 1] - num[0] * na[i + 1]);
        i += 1;
    }
    i = 1;
    while i < n { am[i * n + i - 1] = 1.0; i += 1; }
    bm[0] = 1.0;
    if !all_finite(cm) || !all_finite(am) {
        return empty_arr(), empty_arr(), empty_arr(), 0.0,
            fault(F_VALUE, "tf2ss", "the output matrix overflows");
    }
    return mat2(copy(am), n, n), mat2(copy(bm), n, 1), mat2(copy(cm), 1, n), num[0], no_fault();
}

// zpk2ss gives the state space form of a filter from its zeros, its
// poles and its gain. It is tf2ss of the result of zpk2tf.
fn zpk2ss(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64) -> Arr, Arr, Arr, f64, Fault {
    let b, a, f = zpk2tf(zr, zi, pr, pi_, k);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), 0.0, f; }
    return tf2ss(b.data, a.data);
}

// form_charpoly gives the characteristic polynomial of a real square
// matrix, from the highest power to the lowest. It is the polynomial of
// the eigenvalues.
fn form_charpoly(m, tag: const u8[:]) -> f64[>..], Fault {
    let ev, fe = eigvals_complex(m);
    if fe.kind != F_NONE { return [], fault(fe.kind, tag, fe.detail); }
    var roots_list: Cplx[>..] = [];
    var i: i64 = 0;
    while i < ev.data.len / 2 {
        roots_list.push(Cplx { re: ev.data[2 * i], im: ev.data[2 * i + 1] });
        i += 1;
    }
    var out: f64[>..] = [];
    let fr = roots_poly(roots_list, out);
    if fr.kind != F_NONE { return [], fault(fr.kind, tag, fr.detail); }
    if out.len != ev.data.len / 2 + 1 {
        return [], fault(F_VALUE, tag, "the eigenvalues do not pair into a real polynomial");
    }
    return out, no_fault();
}

// ss2tf gives the numerator and the denominator of a filter from a
// state space form of one input and one output. A is the square matrix
// of the states, b is the input column, c is the output row and d is
// the direct part. The denominator is the polynomial of the eigenvalues
// of A. The numerator is the polynomial of the eigenvalues of A - b c,
// plus (d - 1) times the denominator.
fn ss2tf(a, b: f64[:], c: f64[:], d: f64) -> Arr, Arr, Fault {
    if a.ndim != 2 || a.shape.len != 2 || a.shape[0] != a.shape[1] || a.shape[0] < 1 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "ss2tf", "the matrix of the states is not square");
    }
    let n = a.shape[0];
    if n > FORM_ROOTS_MAX {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "ss2tf", "the matrix of the states is above 64");
    }
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "ss2tf");
    if fs.kind != F_NONE { return empty_arr(), empty_arr(), fs; }
    if b.len != n || c.len != n {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "ss2tf", "the input or the output does not match the matrix of the states");
    }
    if !all_finite(a.data) || !all_finite(b) || !all_finite(c) || !all_finite([d]) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "ss2tf", "a value is not finite");
    }
    let den, fd = form_charpoly(a, "ss2tf");
    if fd.kind != F_NONE { return empty_arr(), empty_arr(), fd; }
    var shifted: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n { shifted.push(a.data[i * n + j] - b[i] * c[j]); j += 1; }
        i += 1;
    }
    let sm = mat2(copy(shifted), n, n);
    let top, ft = form_charpoly(sm, "ss2tf");
    if ft.kind != F_NONE { return empty_arr(), empty_arr(), ft; }
    var num: f64[>..] = [];
    i = 0;
    while i < top.len { num.push(top[i] + (d - 1.0) * den[i]); i += 1; }
    if !all_finite(num) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "ss2tf", "the numerator overflows");
    }
    return flt_arr(copy(num)), flt_arr(copy(den)), no_fault();
}
