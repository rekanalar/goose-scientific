// The partial fractions of a complex rational function: residue_complex.
//
namespace scientific;

import mathshim;

// rc_trim drops the leading zeros of a complex coefficient list.
fn rc_trim(c: Cplx[:], out: Cplx[>..]&) {
    var i: i64 = 0;
    while i < c.len && c[i].re == 0.0 && c[i].im == 0.0 { i += 1; }
    while i < c.len { out.push(c[i]); i += 1; }
}

// rc_roots gives the roots of a polynomial with complex coefficients,
// from the highest power to the lowest. The method is the rule of
// Durand and Kerner, followed by steps of Newton.
fn rc_roots(c: Cplx[:], out: Cplx[>..]&) {
    let n = c.len - 1;
    if n < 1 { return; }
    var p: Cplx[>..] = [];
    var i: i64 = 0;
    while i < c.len { p.push(c_div(c[i], c[0])); i += 1; }
    var radius: f64 = 1.0;
    i = 1;
    while i < p.len {
        let v = c_abs(p[i]);
        if v > radius { radius = v; }
        i += 1;
    }
    radius = radius + 1.0;
    var z: Cplx[>..] = [];
    i = 0;
    while i < n {
        let angle = 6.283185307179586 * i / n + 0.4;
        z.push(Cplx { re: radius * mathshim::cosine(angle), im: radius * mathshim::sine(angle) });
        i += 1;
    }
    var turn: i64 = 0;
    while turn < 500 {
        var moved: f64 = 0.0;
        i = 0;
        while i < n {
            var den = Cplx { re: 1.0, im: 0.0 };
            var k: i64 = 0;
            while k < n {
                if k != i { den = c_mul(den, c_sub(z[i], z[k])); }
                k += 1;
            }
            let step = c_div(rs_value(p, z[i]), den);
            z[i] = c_sub(z[i], step);
            let size = c_abs(step);
            if size > moved { moved = size; }
            i += 1;
        }
        if moved < 1.0e-16 * radius { break; }
        turn += 1;
    }
    var d: Cplx[>..] = [];
    i = 0;
    while i < n { d.push(c_scale(p[i], (n - i) * 1.0)); i += 1; }
    i = 0;
    while i < n {
        var t: i64 = 0;
        while t < 3 {
            let dv = rs_value(d, z[i]);
            if dv.re == 0.0 && dv.im == 0.0 { break; }
            z[i] = c_sub(z[i], c_div(rs_value(p, z[i]), dv));
            t += 1;
        }
        out.push(z[i]);
        i += 1;
    }
}

// rc_deriv gives the coefficients of the derivative of a complex
// polynomial.
fn rc_deriv(c: Cplx[:], out: Cplx[>..]&) {
    let n = c.len - 1;
    var i: i64 = 0;
    while i < n { out.push(c_scale(c[i], (n - i) * 1.0)); i += 1; }
}

// rc_refine moves the mean of a group of m close roots to the nearby
// root of the derivative of order m minus one, as rs_refine does for
// real coefficients.
fn rc_refine(c: Cplx[:], m: i64, start: Cplx, tol: f64) -> Cplx {
    var d: Cplx[>..] = copy_cplx(c);
    var k: i64 = 1;
    while k < m {
        var nx: Cplx[>..] = [];
        rc_deriv(d, nx);
        d = copy(nx);
        k += 1;
    }
    if d.len < 2 { return start; }
    var d2: Cplx[>..] = [];
    rc_deriv(d, d2);
    var z = start;
    var turn: i64 = 0;
    while turn < 20 {
        let den = rs_value(d2, z);
        if den.re == 0.0 && den.im == 0.0 { break; }
        let step = c_div(rs_value(d, z), den);
        z = c_sub(z, step);
        if c_abs(step) < 1.0e-16 * (1.0 + c_abs(z)) { break; }
        turn += 1;
    }
    if c_abs(c_sub(z, start)) > tol { return start; }
    return z;
}

// rc_divide gives the quotient and the remainder of the long division
// of one complex polynomial by another. The first coefficient of the
// divisor must not be zero.
fn rc_divide(b: Cplx[:], a: Cplx[:], q: Cplx[>..]&, r: Cplx[>..]&) {
    var w: Cplx[>..] = copy_cplx(b);
    let steps = b.len - a.len + 1;
    var i: i64 = 0;
    while i < steps {
        let f = c_div(w[i], a[0]);
        q.push(f);
        var j: i64 = 0;
        while j < a.len {
            w[i + j] = c_sub(w[i + j], c_mul(f, a[j]));
            j += 1;
        }
        i += 1;
    }
    i = steps;
    while i < w.len { r.push(w[i]); i += 1; }
}

// residue_complex gives the partial fraction expansion of b over a for
// coefficients that are complex, as SciPy residue does. The coefficient
// lists come as real parts and imaginary parts, from the highest power
// to the lowest. The result holds the residues, the poles and the
// direct terms, each as real parts and imaginary parts. The other
// rules follow residue.
fn residue_complex(br: f64[:], bi: f64[:], ar: f64[:], ai: f64[:], tol: f64 = 0.001, rtype: const u8[:] = "avg") -> Arr, Arr, Arr, Arr, Arr, Arr, Fault {
    let e = empty_arr();
    var rule: i64 = -1;
    if rtype == "avg" || rtype == "mean" { rule = 0; }
    else if rtype == "min" || rtype == "minimum" { rule = 1; }
    else if rtype == "max" || rtype == "maximum" { rule = 2; }
    if rule < 0 {
        return e, e, e, e, e, e, fault(F_VALUE, "residue_complex", "rtype must be avg, min or max");
    }
    if tol != tol || tol < 0.0 || tol - tol != 0.0 {
        return e, e, e, e, e, e, fault(F_VALUE, "residue_complex", "tol must be a finite number that is not negative");
    }
    if br.len != bi.len || ar.len != ai.len {
        return e, e, e, e, e, e, fault(F_SHAPE, "residue_complex", "the real and imaginary parts hold different counts");
    }
    if !rs_finite(br) || !rs_finite(bi) || !rs_finite(ar) || !rs_finite(ai) {
        return e, e, e, e, e, e, fault(F_VALUE, "residue_complex", "a coefficient is not a finite number");
    }
    if br.len > MAX_RESIDUE_TERMS || ar.len > MAX_RESIDUE_TERMS {
        return e, e, e, e, e, e, fault(F_LIMIT, "residue_complex", "the polynomial holds more terms than the limit of residue");
    }
    var bz: Cplx[>..] = rs_zip(br, bi);
    var az: Cplx[>..] = rs_zip(ar, ai);
    var bt: Cplx[>..] = [];
    var at: Cplx[>..] = [];
    rc_trim(bz, bt);
    rc_trim(az, at);
    if at.len == 0 {
        return e, e, e, e, e, e, fault(F_VALUE, "residue_complex", "the denominator holds no non-zero term");
    }
    var rts: Cplx[>..] = [];
    rc_roots(at, rts);
    var order: i64[>..] = [];
    rs_order(rts, order);
    if bt.len == 0 {
        var zr: f64[>..] = [];
        var pr: f64[>..] = [];
        var pi: f64[>..] = [];
        var i: i64 = 0;
        while i < order.len {
            zr.push(0.0);
            pr.push(rts[order[i]].re);
            pi.push(rts[order[i]].im);
            i += 1;
        }
        return line_arr(copy(zr)), line_arr(copy(zr)), line_arr(copy(pr)), line_arr(copy(pi)), e, e, no_fault();
    }
    var kr: f64[>..] = [];
    var ki: f64[>..] = [];
    var num: Cplx[>..] = [];
    if bt.len < at.len {
        num = copy_cplx(bt);
    } else {
        var q: Cplx[>..] = [];
        var rm: Cplx[>..] = [];
        rc_divide(bt, at, q, rm);
        var qi: i64 = 0;
        while qi < q.len { kr.push(q[qi].re); ki.push(q[qi].im); qi += 1; }
        var lead: i64 = 0;
        while lead < rm.len - 1 && c_abs(rm[lead]) <= 0.00000001 { lead += 1; }
        var t: i64 = lead;
        while t < rm.len { num.push(rm[t]); t += 1; }
        if rm.len == 0 { num.push(Cplx { re: 0.0, im: 0.0 }); }
    }
    var up: Cplx[>..] = [];
    var umult: i64[>..] = [];
    rs_group(rts, tol, rule, up, umult);
    if rule == 0 {
        var ug: i64 = 0;
        while ug < up.len {
            if umult[ug] > 1 { up[ug] = rc_refine(at, umult[ug], up[ug], tol); }
            ug += 1;
        }
    }
    var uo: i64[>..] = [];
    rs_order(up, uo);
    var sp: Cplx[>..] = [];
    var sm: i64[>..] = [];
    var gi: i64 = 0;
    while gi < uo.len { sp.push(up[uo[gi]]); sm.push(umult[uo[gi]]); gi += 1; }
    var res: Cplx[>..] = [];
    if sp.len > 0 { rs_residues(sp, sm, num, res); }
    var rr: f64[>..] = [];
    var ri: f64[>..] = [];
    var pr: f64[>..] = [];
    var pi: f64[>..] = [];
    var i: i64 = 0;
    while i < res.len {
        let v = c_div(res[i], at[0]);
        rr.push(v.re);
        ri.push(v.im);
        i += 1;
    }
    gi = 0;
    while gi < sp.len {
        var m: i64 = 0;
        while m < sm[gi] { pr.push(sp[gi].re); pi.push(sp[gi].im); m += 1; }
        gi += 1;
    }
    return line_arr(copy(rr)), line_arr(copy(ri)), line_arr(copy(pr)), line_arr(copy(pi)), line_arr(copy(kr)), line_arr(copy(ki)), no_fault();
}
