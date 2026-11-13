// The partial fraction expansion of a rational function: the poles and the residues.
//
// The partial fraction expansion of a rational function, as SciPy
// computes it with residue. The poles come from the roots of the
// denominator. Poles that lie within a tolerance of each other form one
// group of a multiple pole. The residues follow from the method of
// deflation through subtraction.
namespace scientific;

import mathshim;

// The largest count of coefficients that residue takes. The method needs
// the products of the factors of the denominator, so the cost grows with
// the cube of the degree.
const MAX_RESIDUE_TERMS: i64 = 257;

// rs_times_pole multiplies a complex polynomial by the factor x minus
// the pole.
fn rs_times_pole(p: Cplx[:], pole: Cplx, out: Cplx[>..]&) {
    let m = Cplx { re: -pole.re, im: -pole.im };
    var k: i64 = 0;
    while k <= p.len {
        var v = Cplx { re: 0.0, im: 0.0 };
        if k < p.len { v = p[k]; }
        if k >= 1 { v = c_add(v, c_mul(p[k - 1], m)); }
        out.push(v);
        k += 1;
    }
}

// rs_divide_pole divides a complex polynomial by the factor x minus the
// pole. It gives the quotient in q and the remainder as the value.
fn rs_divide_pole(u: Cplx[:], pole: Cplx, q: Cplx[>..]&) -> Cplx {
    if u.len < 2 {
        q.push(Cplx { re: 0.0, im: 0.0 });
        return u[0];
    }
    var carry = u[0];
    var k: i64 = 1;
    q.push(carry);
    while k < u.len - 1 {
        carry = c_add(u[k], c_mul(carry, pole));
        q.push(carry);
        k += 1;
    }
    return c_add(u[u.len - 1], c_mul(carry, pole));
}

// rs_value gives the value of a complex polynomial at a point.
fn rs_value(p: Cplx[:], x: Cplx) -> Cplx {
    var y = p[0];
    var i: i64 = 1;
    while i < p.len { y = c_add(c_mul(y, x), p[i]); i += 1; }
    return y;
}

// rs_before says whether the first pole sorts before the second. The
// order is by the size, and a tie keeps the original order.
fn rs_before(a: Cplx, b: Cplx) -> bool {
    return c_abs(a) < c_abs(b);
}

// rs_group joins the roots that lie within the tolerance of each other.
// The scan goes through the roots in order, and each root takes the
// roots that stay unused and lie within the tolerance of it. A group
// gives one pole by the rule: 0 mean, 1 minimum, 2 maximum.
fn rs_group(p: Cplx[:], tol: f64, rule: i64, out: Cplx[>..]&, mult: i64[>..]&) {
    var used: i64[>..] = [];
    var i: i64 = 0;
    while i < p.len { used.push(0); i += 1; }
    i = 0;
    while i < p.len {
        if used[i] != 0 { i += 1; continue; }
        var sum = Cplx { re: 0.0, im: 0.0 };
        var pick = p[i];
        var count: i64 = 0;
        var j: i64 = 0;
        while j < p.len {
            if used[j] == 0 {
                let dx = p[j].re - p[i].re;
                let dy = p[j].im - p[i].im;
                if mathshim::root(dx * dx + dy * dy) <= tol {
                    used[j] = 1;
                    if count == 0 { pick = p[j]; }
                    else if rule == 1 && (p[j].re < pick.re || (p[j].re == pick.re && p[j].im < pick.im)) { pick = p[j]; }
                    else if rule == 2 && (p[j].re > pick.re || (p[j].re == pick.re && p[j].im > pick.im)) { pick = p[j]; }
                    sum = c_add(sum, p[j]);
                    count += 1;
                }
            }
            j += 1;
        }
        if rule == 0 { pick = Cplx { re: sum.re / count, im: sum.im / count }; }
        out.push(pick);
        mult.push(count);
        i += 1;
    }
}

// rs_deriv gives the coefficients of the derivative of a polynomial.
fn rs_deriv(c: f64[:], out: f64[>..]&) {
    let n = c.len - 1;
    var i: i64 = 0;
    while i < n { out.push(c[i] * (n - i)); i += 1; }
}

// rs_refine moves the mean of a group of m close roots to the nearby
// root of the derivative of order m - 1. The group is a simple root of
// that derivative, so the iteration is stable where the roots of the
// group are not. A result that leaves the group stays out.
fn rs_refine(c: f64[:], m: i64, start: Cplx, tol: f64) -> Cplx {
    var d: f64[>..] = copy_slice(c);
    var k: i64 = 1;
    while k < m {
        var nx: f64[>..] = [];
        rs_deriv(d, nx);
        d = copy(nx);
        k += 1;
    }
    if d.len < 2 { return start; }
    var d2: f64[>..] = [];
    rs_deriv(d, d2);
    var z = start;
    var turn: i64 = 0;
    while turn < 20 {
        let den = c_value(d2, z);
        if den.re == 0.0 && den.im == 0.0 { break; }
        let step = c_div(c_value(d, z), den);
        z = c_sub(z, step);
        if c_abs(step) < 1.0e-16 * (1.0 + c_abs(z)) { break; }
        turn += 1;
    }
    let dx = z.re - start.re;
    let dy = z.im - start.im;
    if mathshim::root(dx * dx + dy * dy) > tol { return start; }
    return z;
}

// rs_order gives the indices of the poles sorted by size. The sort is
// stable, so equal sizes keep their order.
fn rs_order(p: Cplx[:], idx: i64[>..]&) {
    var i: i64 = 0;
    while i < p.len { idx.push(i); i += 1; }
    i = 1;
    while i < idx.len {
        let v = idx[i];
        var j = i - 1;
        while j >= 0 && rs_before(p[v], p[idx[j]]) { idx[j + 1] = idx[j]; j -= 1; }
        idx[j + 1] = v;
        i += 1;
    }
}

// rs_residues gives the residues of the numerator over the poles. The
// poles hold one entry per group and mult holds the size of each group.
// A group of size m gives m residues, ordered by the rising power. The
// factor of a group is the product of the other groups.
fn rs_residues(poles: Cplx[:], mult: i64[:], numer0: Cplx[:], res: Cplx[>..]&) {
    let g = poles.len;
    var gi: i64 = 0;
    while gi < g {
        let pole = poles[gi];
        let mu = mult[gi];
        var factor: Cplx[>..] = [Cplx { re: 1.0, im: 0.0 }];
        var j: i64 = 0;
        while j < g {
            if j != gi {
                var m: i64 = 0;
                while m < mult[j] {
                    var nx: Cplx[>..] = [];
                    rs_times_pole(factor, poles[j], nx);
                    factor = copy(nx);
                    m += 1;
                }
            }
            j += 1;
        }
        if mu == 1 {
            res.push(c_div(rs_value(numer0, pole), rs_value(factor, pole)));
        } else {
            var numer: Cplx[>..] = copy_cplx(numer0);
            var fq: Cplx[>..] = [];
            let d = rs_divide_pole(factor, pole, fq);
            var powers: Cplx[>..] = [];
            var m: i64 = 0;
            while m < mu {
                var nq: Cplx[>..] = [];
                let rem = rs_divide_pole(numer, pole, nq);
                let r = c_div(rem, d);
                // The new numerator is the quotient less r times the factor.
                var ln = nq.len;
                if fq.len > ln { ln = fq.len; }
                var out: Cplx[>..] = [];
                var q: i64 = 0;
                while q < ln {
                    var av = Cplx { re: 0.0, im: 0.0 };
                    var bv = Cplx { re: 0.0, im: 0.0 };
                    let ia = q - (ln - nq.len);
                    let ib = q - (ln - fq.len);
                    if ia >= 0 { av = nq[ia]; }
                    if ib >= 0 { bv = c_mul(r, fq[ib]); }
                    out.push(c_sub(av, bv));
                    q += 1;
                }
                numer = copy(out);
                powers.push(r);
                m += 1;
            }
            var q: i64 = powers.len - 1;
            while q >= 0 { res.push(powers[q]); q -= 1; }
        }
        gi += 1;
    }
}

// copy_cplx gives a growable copy of a list of complex numbers.
fn copy_cplx(p: Cplx[:]) -> Cplx[>..] {
    var out: Cplx[>..] = [];
    var i: i64 = 0;
    while i < p.len { out.push(p[i]); i += 1; }
    return out;
}

// rs_finite says whether every value is a finite number.
fn rs_finite(v: f64[:]) -> bool {
    var i: i64 = 0;
    while i < v.len {
        if v[i] != v[i] || v[i] - v[i] != 0.0 { return false; }
        i += 1;
    }
    return true;
}

// rs_trim drops the leading zeros of a coefficient list.
fn rs_trim(c: f64[:], out: f64[>..]&) {
    var i: i64 = 0;
    while i < c.len && c[i] == 0.0 { i += 1; }
    while i < c.len { out.push(c[i]); i += 1; }
}

// rs_unique groups the roots, refines the groups of a repeated root and
// sorts the groups by size. The polynomial c holds the roots.
fn rs_unique(rts: Cplx[:], c: f64[:], tol: f64, rule: i64, sp: Cplx[>..]&, sm: i64[>..]&) {
    var up: Cplx[>..] = [];
    var mult: i64[>..] = [];
    rs_group(rts, tol, rule, up, mult);
    if rule == 0 {
        var ug: i64 = 0;
        while ug < up.len {
            if mult[ug] > 1 { up[ug] = rs_refine(c, mult[ug], up[ug], tol); }
            ug += 1;
        }
    }
    var uo: i64[>..] = [];
    rs_order(up, uo);
    var gi: i64 = 0;
    while gi < uo.len { sp.push(up[uo[gi]]); sm.push(mult[uo[gi]]); gi += 1; }
}

// residue gives the partial fraction expansion of b over a. The
// coefficients run from the highest power to the lowest, and they are
// real. The result holds the residues as real parts and imaginary
// parts, the poles as real parts and imaginary parts, and the direct
// terms. A pole that repeats gives one residue per power, from the
// first power up. The poles sort by size. Poles within tol of each
// other form a group, and rtype picks the pole of a group: "avg",
// "min" or "max".
fn residue(b: f64[:], a: f64[:], tol: f64 = 0.001, rtype: const u8[:] = "avg") -> Arr, Arr, Arr, Arr, Arr, Fault {
    var rule: i64 = -1;
    if rtype == "avg" || rtype == "mean" { rule = 0; }
    else if rtype == "min" || rtype == "minimum" { rule = 1; }
    else if rtype == "max" || rtype == "maximum" { rule = 2; }
    if rule < 0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "residue", "rtype must be avg, min or max");
    }
    if tol != tol || tol < 0.0 || tol - tol != 0.0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "residue", "tol must be a finite number that is not negative");
    }
    if !rs_finite(b) || !rs_finite(a) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "residue", "a coefficient is not a finite number");
    }
    if b.len > MAX_RESIDUE_TERMS || a.len > MAX_RESIDUE_TERMS {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_LIMIT, "residue", "the polynomial holds more terms than the limit of residue");
    }
    var bt: f64[>..] = [];
    var at: f64[>..] = [];
    rs_trim(b, bt);
    rs_trim(a, at);
    if at.len == 0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "residue", "the denominator holds no non-zero term");
    }
    let rts, rf = roots_of(at);
    if rf.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), rf; }
    var order: i64[>..] = [];
    rs_order(rts, order);
    if bt.len == 0 {
        // A zero numerator gives zero residues at the sorted poles.
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
        return line_arr(copy(zr)), line_arr(copy(zr)), line_arr(copy(pr)), line_arr(copy(pi)), empty_arr(), no_fault();
    }
    // A numerator of the same degree or higher gives the direct terms.
    var kk: f64[>..] = [];
    var num: f64[>..] = [];
    if bt.len < at.len {
        num = copy(bt);
    } else {
        let q, rm, df = polydiv(bt, at);
        if df.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), df; }
        kk = copy(q.data);
        var lead: i64 = 0;
        while lead < rm.data.len - 1 && mathshim::abs_of(rm.data[lead]) <= 0.00000001 { lead += 1; }
        var t: i64 = lead;
        while t < rm.data.len { num.push(rm.data[t]); t += 1; }
    }
    var cnum: Cplx[>..] = [];
    var ni: i64 = 0;
    while ni < num.len { cnum.push(Cplx { re: num[ni], im: 0.0 }); ni += 1; }
    var sp: Cplx[>..] = [];
    var sm: i64[>..] = [];
    rs_unique(rts, at, tol, rule, sp, sm);
    var gi: i64 = 0;
    var res: Cplx[>..] = [];
    if sp.len > 0 { rs_residues(sp, sm, cnum, res); }
    var rr: f64[>..] = [];
    var ri: f64[>..] = [];
    var pr: f64[>..] = [];
    var pi: f64[>..] = [];
    // The residues divide by the leading coefficient of the denominator.
    let lead_a = Cplx { re: at[0], im: 0.0 };
    var i: i64 = 0;
    while i < res.len {
        let v = c_div(res[i], lead_a);
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
    return line_arr(copy(rr)), line_arr(copy(ri)), line_arr(copy(pr)), line_arr(copy(pi)), line_arr(copy(kk)), no_fault();
}

// The largest count of roots that unique_roots takes. The grouping
// compares every pair, so the cost grows with the square of the count.
const MAX_UNIQUE_ROOTS: i64 = 4096;

