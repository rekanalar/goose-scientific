// The derivative and the integral of a series of an orthogonal basis.
//
// Each family follows the loops of numpy.polynomial. The rules differ
// per family, so the loops do not share a body.
namespace scientific;

import mathshim;

// pb_der_once gives the derivative of the series w, which the call
// scales first. The series w holds n + 1 places and may change.
fn pb_der_once(fam: i64, w: f64[>..]&, n: i64) -> f64[>..] {
    var d = pb_zeros(n);
    var j = n;
    if fam == PB_LEG {
        while j > 2 {
            d[j - 1] = (2 * j - 1) * w[j];
            w[j - 2] += w[j];
            j -= 1;
        }
        if n > 1 { d[1] = 3.0 * w[2]; }
        d[0] = w[1];
    } else if fam == PB_CHEB {
        while j > 2 {
            d[j - 1] = (2 * j) * w[j];
            w[j - 2] += j * w[j] / (j - 2);
            j -= 1;
        }
        if n > 1 { d[1] = 4.0 * w[2]; }
        d[0] = w[1];
    } else if fam == PB_HERM {
        while j > 0 { d[j - 1] = (2 * j) * w[j]; j -= 1; }
    } else if fam == PB_HERME {
        while j > 0 { d[j - 1] = j * w[j]; j -= 1; }
    } else {
        while j > 1 {
            d[j - 1] = -w[j];
            w[j - 1] += w[j];
            j -= 1;
        }
        d[0] = -w[1];
    }
    return d;
}

// pb_check_call checks the series, the order and the scale of a call.
fn pb_check_call(c: f64[:], m: i64, scl: f64, tag: const u8[:]) -> Fault {
    var f = pb_check_series(c, tag);
    if f.kind != F_NONE { return f; }
    f = pb_check_order(m, tag);
    if f.kind != F_NONE { return f; }
    return pb_check_scale(scl, tag);
}

// pb_der checks its input and gives the derivative of order m.
fn pb_der(fam: i64, c: f64[:], m: i64, scl: f64, tag: const u8[:]) -> Arr, Fault {
    let f = pb_check_call(c, m, scl, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    var w = copy_slice(c);
    if m == 0 { return line_arr(copy(w)), no_fault(); }
    var n = w.len;
    if m >= n { return line_arr(pb_zeros(1)), no_fault(); }
    var turn: i64 = 0;
    while turn < m {
        n -= 1;
        var i: i64 = 0;
        while scl != 1.0 && i < w.len { w[i] *= scl; i += 1; }
        var d = pb_der_once(fam, w, n);
        w = copy(d);
        turn += 1;
    }
    if !pb_all_finite(w) { return empty_arr(), fault(F_VALUE, tag, "a coefficient of the result is not finite"); }
    return line_arr(copy(w)), no_fault();
}

// pb_int_once gives the antiderivative of the scaled series w with the
// constant of the first place still to set.
fn pb_int_once(fam: i64, w: f64[:]) -> f64[>..] {
    let n = w.len;
    var t = pb_zeros(n + 1);
    var j: i64 = 1;
    if fam == PB_LEG {
        t[1] = w[0];
        if n > 1 { t[2] = w[1] / 3.0; }
        j = 2;
        while j < n {
            let q = w[j] / (2 * j + 1);
            t[j + 1] = q;
            t[j - 1] -= q;
            j += 1;
        }
    } else if fam == PB_CHEB {
        t[1] = w[0];
        if n > 1 { t[2] = w[1] / 4.0; }
        j = 2;
        while j < n {
            t[j + 1] = w[j] / (2 * (j + 1));
            t[j - 1] -= w[j] / (2 * (j - 1));
            j += 1;
        }
    } else if fam == PB_HERM {
        t[1] = w[0] / 2.0;
        while j < n { t[j + 1] = w[j] / (2 * (j + 1)); j += 1; }
    } else if fam == PB_HERME {
        t[1] = w[0];
        while j < n { t[j + 1] = w[j] / (j + 1); j += 1; }
    } else {
        t[0] = w[0];
        t[1] = -w[0];
        while j < n {
            t[j] += w[j];
            t[j + 1] = -w[j];
            j += 1;
        }
    }
    return t;
}

// pb_int checks its input and gives the integral of order m. The
// constants k pad with zeros. The integral at lbnd equals the constant.
fn pb_int(fam: i64, c: f64[:], m: i64, k: f64[:], lbnd: f64, scl: f64, tag: const u8[:]) -> Arr, Fault {
    let f = pb_check_call(c, m, scl, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    if !pb_finite(lbnd) { return empty_arr(), fault(F_VALUE, tag, "the lower bound is not finite"); }
    if k.len > m { return empty_arr(), fault(F_VALUE, tag, "more constants than the order"); }
    var i: i64 = 0;
    while i < k.len {
        if !pb_finite(k[i]) { return empty_arr(), fault(F_VALUE, tag, "a constant is not finite"); }
        i += 1;
    }
    var w = copy_slice(c);
    var turn: i64 = 0;
    while turn < m {
        i = 0;
        while scl != 1.0 && i < w.len { w[i] *= scl; i += 1; }
        var kc: f64 = 0.0;
        if turn < k.len { kc = k[turn]; }
        if w.len == 1 && w[0] == 0.0 {
            w[0] += kc;
        } else {
            var t = pb_int_once(fam, w);
            t[0] += kc - pb_clenshaw(fam, lbnd, t);
            w = copy(t);
        }
        turn += 1;
    }
    if !pb_all_finite(w) { return empty_arr(), fault(F_VALUE, tag, "a coefficient of the result is not finite"); }
    return line_arr(copy(w)), no_fault();
}
