// The arithmetic of polynomials: poly, polyadd, polymul, polydiv, polyder and polyint.
//
namespace scientific;

import mathshim;

// poly gives the coefficients of the monic polynomial of the roots,
// from the highest power.
fn poly(roots: f64[:]) -> Arr, Fault {
    if roots.len < 1 {
        return empty_arr(), fault(F_VALUE, "poly", "the list of roots holds no value");
    }
    if roots.len > MAX_ROOTS_DEGREE {
        return empty_arr(), fault(F_LIMIT, "poly", "the count of roots holds a value above the limit");
    }
    var c: f64[>..] = [1.0];
    var i: i64 = 0;
    while i < roots.len {
        var t: f64[>..] = [];
        var j: i64 = 0;
        while j <= c.len { t.push(0.0); j += 1; }
        j = 0;
        while j < c.len {
            t[j] = t[j] + c[j];
            t[j + 1] = t[j + 1] - roots[i] * c[j];
            j += 1;
        }
        c = copy(t);
        i += 1;
    }
    return line_arr(copy(c)), no_fault();
}

// poly_cplx gives the coefficients of the monic polynomial of complex
// roots, as the real parts and the imaginary parts.
fn poly_cplx(re: f64[:], im: f64[:]) -> Arr, Arr, Fault {
    if re.len < 1 || re.len != im.len {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "poly", "the parts hold no value or differ in count");
    }
    if re.len > MAX_ROOTS_DEGREE {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "poly", "the count of roots holds a value above the limit");
    }
    var c: Cplx[>..] = [Cplx { re: 1.0, im: 0.0 }];
    var i: i64 = 0;
    while i < re.len {
        var t: Cplx[>..] = [];
        var j: i64 = 0;
        while j <= c.len { t.push(Cplx { re: 0.0, im: 0.0 }); j += 1; }
        j = 0;
        while j < c.len {
            t[j] = c_add(t[j], c[j]);
            t[j + 1] = c_sub(t[j + 1], c_mul(Cplx { re: re[i], im: im[i] }, c[j]));
            j += 1;
        }
        c = copy(t);
        i += 1;
    }
    var r: f64[>..] = [];
    var m: f64[>..] = [];
    i = 0;
    while i < c.len { r.push(c[i].re); m.push(c[i].im); i += 1; }
    return line_arr(copy(r)), line_arr(copy(m)), no_fault();
}

// polyadd adds two polynomials from the highest power. The result
// keeps the leading zeros, as NumPy keeps them.
fn polyadd(a: f64[:], b: f64[:]) -> Arr, Fault {
    return poly_sum(a, b, 1.0, "polyadd");
}

// polysub subtracts the second polynomial from the first.
fn polysub(a: f64[:], b: f64[:]) -> Arr, Fault {
    return poly_sum(a, b, -1.0, "polysub");
}

fn poly_sum(a: f64[:], b: f64[:], sign: f64, tag: const u8[:]) -> Arr, Fault {
    if a.len < 1 || b.len < 1 {
        return empty_arr(), fault(F_VALUE, tag, "a polynomial holds no coefficient");
    }
    var n = a.len;
    if b.len > n { n = b.len; }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var v: f64 = 0.0;
        let ia = i - (n - a.len);
        let ib = i - (n - b.len);
        if ia >= 0 { v = a[ia]; }
        if ib >= 0 { v = v + sign * b[ib]; }
        d.push(v);
        i += 1;
    }
    return line_arr(copy(d)), no_fault();
}

// polymul multiplies two polynomials from the highest power.
fn polymul(a: f64[:], b: f64[:]) -> Arr, Fault {
    if a.len < 1 || b.len < 1 {
        return empty_arr(), fault(F_VALUE, "polymul", "a polynomial holds no coefficient");
    }
    var d: f64[>..] = [];
    flt_pmul(a, b, d);
    return line_arr(copy(d)), no_fault();
}

// polydiv divides two polynomials from the highest power and gives the
// quotient and the remainder. The remainder drops its leading zeros.
fn polydiv(u: f64[:], v: f64[:]) -> Arr, Arr, Fault {
    if u.len < 1 || v.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "polydiv", "a polynomial holds no coefficient");
    }
    if v[0] == 0.0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "polydiv", "the leading coefficient of the divisor holds zero");
    }
    let m = u.len;
    let n = v.len;
    if m < n {
        return line_arr([0.0]), line_arr(copy_slice(u)), no_fault();
    }
    var r: f64[>..] = [];
    var i: i64 = 0;
    while i < m { r.push(u[i]); i += 1; }
    var q: f64[>..] = [];
    i = 0;
    while i <= m - n {
        let f = r[i] / v[0];
        q.push(f);
        var j: i64 = 0;
        while j < n { r[i + j] = r[i + j] - f * v[j]; j += 1; }
        i += 1;
    }
    // The remainder holds the last n - 1 places, without leading zeros.
    var rem: f64[>..] = [];
    i = m - n + 1;
    while i < m { rem.push(r[i]); i += 1; }
    var first: i64 = 0;
    while first < rem.len - 1 && rem[first] == 0.0 { first += 1; }
    var out: f64[>..] = [];
    i = first;
    while i < rem.len { out.push(rem[i]); i += 1; }
    if out.len == 0 { out.push(0.0); }
    return line_arr(copy(q)), line_arr(copy(out)), no_fault();
}

fn copy_slice(u: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < u.len { d.push(u[i]); i += 1; }
    return d;
}

// polyder gives the derivative of order m from the highest power. A
// constant gives [0], where NumPy gives an empty array.
fn polyder(p: f64[:], m: i64 = 1) -> Arr, Fault {
    if p.len < 1 || m < 1 {
        return empty_arr(), fault(F_VALUE, "polyder", "the polynomial holds no coefficient or the order is below one");
    }
    var c = copy_slice(p);
    var turn: i64 = 0;
    while turn < m && c.len > 1 {
        var d: f64[>..] = [];
        var i: i64 = 0;
        while i + 1 < c.len { d.push(c[i] * (c.len - 1 - i)); i += 1; }
        c = copy(d);
        turn += 1;
    }
    if turn < m { c = [0.0]; }
    return line_arr(copy(c)), no_fault();
}

// polyint gives the integral of order m from the highest power. The
// list k holds the constants of the integration, one per turn, with
// zero for any constant that the list leaves out.
fn polyint(p: f64[:], m: i64 = 1, k: f64[:] = []) -> Arr, Fault {
    if p.len < 1 || m < 1 || m > 64 {
        return empty_arr(), fault(F_VALUE, "polyint", "the polynomial holds no coefficient or the order stands outside one to 64");
    }
    if k.len > m {
        return empty_arr(), fault(F_VALUE, "polyint", "the list of constants is longer than the order");
    }
    var c = copy_slice(p);
    var turn: i64 = 0;
    while turn < m {
        var d: f64[>..] = [];
        var i: i64 = 0;
        while i < c.len { d.push(c[i] / (c.len - i)); i += 1; }
        if turn < k.len { d.push(k[turn]); } else { d.push(0.0); }
        c = copy(d);
        turn += 1;
    }
    return line_arr(copy(c)), no_fault();
}
