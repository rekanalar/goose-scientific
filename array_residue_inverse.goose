// The inverse of a partial fraction: invres, invresz, unique_roots and residuez.
//
namespace scientific;

import mathshim;

// rs_rule gives the number of a rule that picks the pole of a group, or
// minus one for a name that does not exist.
fn rs_rule(rtype: const u8[:]) -> i64 {
    if rtype == "avg" || rtype == "mean" { return 0; }
    if rtype == "min" || rtype == "minimum" { return 1; }
    if rtype == "max" || rtype == "maximum" { return 2; }
    return -1;
}

// rs_zip joins real parts and imaginary parts into complex numbers.
fn rs_zip(re: f64[:], im: f64[:]) -> Cplx[>..] {
    var out: Cplx[>..] = [];
    var i: i64 = 0;
    while i < re.len { out.push(Cplx { re: re[i], im: im[i] }); i += 1; }
    return out;
}

// rs_polymul multiplies two complex polynomials.
fn rs_polymul(a: Cplx[:], b: Cplx[:], out: Cplx[>..]&) {
    var k: i64 = 0;
    while k < a.len + b.len - 1 { out.push(Cplx { re: 0.0, im: 0.0 }); k += 1; }
    var i: i64 = 0;
    while i < a.len {
        var j: i64 = 0;
        while j < b.len { out[i + j] = c_add(out[i + j], c_mul(a[i], b[j])); j += 1; }
        i += 1;
    }
}

// rs_reverse gives a complex polynomial with its coefficients reversed.
fn rs_reverse(p: Cplx[:]) -> Cplx[>..] {
    var out: Cplx[>..] = [];
    var i: i64 = p.len - 1;
    while i >= 0 { out.push(p[i]); i -= 1; }
    return out;
}

// rs_add_scaled gives the polynomial acc plus r times the polynomial f.
// The polynomials align at the constant term.
fn rs_add_scaled(acc: Cplx[:], r: Cplx, f: Cplx[:]) -> Cplx[>..] {
    var ln = acc.len;
    if f.len > ln { ln = f.len; }
    var out: Cplx[>..] = [];
    var i: i64 = 0;
    while i < ln {
        var v = Cplx { re: 0.0, im: 0.0 };
        let ia = i - (ln - acc.len);
        let ib = i - (ln - f.len);
        if ia >= 0 { v = acc[ia]; }
        if ib >= 0 { v = c_add(v, c_mul(r, f[ib])); }
        out.push(v);
        i += 1;
    }
    return out;
}

// rs_group_adjacent groups poles that sit side by side and lie within the
// tolerance of the first pole of the group. The first pole stands for
// the group, as SciPy does for every rule.
fn rs_group_adjacent(p: Cplx[:], tol: f64, up: Cplx[>..]&, mult: i64[>..]&) {
    var first = p[0];
    var count: i64 = 1;
    var i: i64 = 1;
    while i < p.len {
        let dx = p[i].re - first.re;
        let dy = p[i].im - first.im;
        if mathshim::root(dx * dx + dy * dy) <= tol {
            count += 1;
        } else {
            up.push(first);
            mult.push(count);
            first = p[i];
            count = 1;
        }
        i += 1;
    }
    up.push(first);
    mult.push(count);
}

// rs_factors gives, for each pole of a group list, the polynomials that
// the expansion needs. A group of m gives m polynomials, from the
// highest power down. Each is the product of the other groups times a
// power of the own factor. The product of all groups goes to den.
fn rs_factors(poles: Cplx[:], mult: i64[:], flat: Cplx[>..]&, off: i64[>..]&, lens: i64[>..]&, den: Cplx[>..]&) {
    var all: Cplx[>..] = [Cplx { re: 1.0, im: 0.0 }];
    var gi: i64 = 0;
    while gi < poles.len {
        var m: i64 = 0;
        while m < mult[gi] {
            var nx: Cplx[>..] = [];
            rs_times_pole(all, poles[gi], nx);
            all = copy(nx);
            m += 1;
        }
        gi += 1;
    }
    var t: i64 = 0;
    while t < all.len { den.push(all[t]); t += 1; }
    gi = 0;
    while gi < poles.len {
        var base: Cplx[>..] = [Cplx { re: 1.0, im: 0.0 }];
        var j: i64 = 0;
        while j < poles.len {
            if j != gi {
                var m: i64 = 0;
                while m < mult[j] {
                    var nx: Cplx[>..] = [];
                    rs_times_pole(base, poles[j], nx);
                    base = copy(nx);
                    m += 1;
                }
            }
            j += 1;
        }
        // The powers of the own factor, from none up to m - 1.
        var cur: Cplx[>..] = copy(base);
        var lv: i64 = 0;
        var starts: i64[>..] = [];
        var store: Cplx[>..] = [];
        while lv < mult[gi] {
            starts.push(store.len);
            var q: i64 = 0;
            while q < cur.len { store.push(cur[q]); q += 1; }
            var nx: Cplx[>..] = [];
            rs_times_pole(cur, poles[gi], nx);
            cur = copy(nx);
            lv += 1;
        }
        // Emit them in the reverse order: the highest power first.
        lv = mult[gi] - 1;
        while lv >= 0 {
            let st = starts[lv];
            var en = store.len;
            if lv + 1 < starts.len { en = starts[lv + 1]; }
            off.push(flat.len);
            lens.push(en - st);
            var q: i64 = st;
            while q < en { flat.push(store[q]); q += 1; }
            lv -= 1;
        }
        gi += 1;
    }
}

// rs_real gives the real parts of a complex polynomial. A coefficient
// with an imaginary part above the noise of the arithmetic is a fault.
fn rs_real(p: Cplx[:], out: f64[>..]&) -> bool {
    var big: f64 = 1.0;
    var i: i64 = 0;
    while i < p.len { if c_abs(p[i]) > big { big = c_abs(p[i]); } i += 1; }
    i = 0;
    while i < p.len {
        if mathshim::abs_of(p[i].im) > 1.0e-9 * big { return false; }
        out.push(p[i].re);
        i += 1;
    }
    return true;
}

// rs_pair_check tests the residues and the poles of the inverse
// expansions. It gives a fault, or no fault.
fn rs_pair_check(name: const u8[:], rr: f64[:], ri: f64[:], pr: f64[:], pi: f64[:], k: f64[:], tol: f64, rule: i64) -> Fault {
    if rr.len != ri.len || pr.len != pi.len || rr.len != pr.len {
        return fault(F_SHAPE, name, "the residues and the poles must hold the same count");
    }
    if pr.len == 0 { return fault(F_VALUE, name, "the expansion holds no pole"); }
    if rule < 0 { return fault(F_VALUE, name, "rtype must be avg, min or max"); }
    if tol != tol || tol < 0.0 || tol - tol != 0.0 {
        return fault(F_VALUE, name, "tol must be a finite number that is not negative");
    }
    if !rs_finite(rr) || !rs_finite(ri) || !rs_finite(pr) || !rs_finite(pi) || !rs_finite(k) {
        return fault(F_VALUE, name, "a value is not a finite number");
    }
    if pr.len > MAX_RESIDUE_TERMS || k.len > MAX_RESIDUE_TERMS {
        return fault(F_LIMIT, name, "the expansion holds more terms than the limit");
    }
    return no_fault();
}

// invres builds the numerator and the denominator from the residues, the
// poles and the direct terms. It is the inverse of residue. Poles of a
// group must sit side by side. The residues of a repeated pole run from
// the first power up. A result with a complex coefficient is a fault.
fn invres(rr: f64[:], ri: f64[:], pr: f64[:], pi: f64[:], k: f64[:], tol: f64 = 0.001, rtype: const u8[:] = "avg") -> Arr, Arr, Fault {
    let chk = rs_pair_check("invres", rr, ri, pr, pi, k, tol, rs_rule(rtype));
    if chk.kind != F_NONE { return empty_arr(), empty_arr(), chk; }
    var kt: f64[>..] = [];
    rs_trim(k, kt);
    let p = rs_zip(pr, pi);
    var up: Cplx[>..] = [];
    var mult: i64[>..] = [];
    rs_group_adjacent(p, tol, up, mult);
    var flat: Cplx[>..] = [];
    var off: i64[>..] = [];
    var lens: i64[>..] = [];
    var den: Cplx[>..] = [];
    rs_factors(up, mult, flat, off, lens, den);
    var num: Cplx[>..] = [];
    if kt.len > 0 {
        var kc: Cplx[>..] = [];
        var i: i64 = 0;
        while i < kt.len { kc.push(Cplx { re: kt[i], im: 0.0 }); i += 1; }
        rs_polymul(kc, den, num);
    }
    var t: i64 = 0;
    while t < rr.len && t < off.len {
        var f: Cplx[>..] = [];
        var q: i64 = 0;
        while q < lens[t] { f.push(flat[off[t] + q]); q += 1; }
        let nn = rs_add_scaled(num, Cplx { re: rr[t], im: ri[t] }, f);
        num = copy(nn);
        t += 1;
    }
    var bo: f64[>..] = [];
    var ao: f64[>..] = [];
    if !rs_real(num, bo) || !rs_real(den, ao) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "invres", "the result holds a complex coefficient");
    }
    return line_arr(copy(bo)), line_arr(copy(ao)), no_fault();
}

// invresz builds the numerator and the denominator in negative powers of
// z. It is the inverse of residuez, and it follows the rules of invres.
fn invresz(rr: f64[:], ri: f64[:], pr: f64[:], pi: f64[:], k: f64[:], tol: f64 = 0.001, rtype: const u8[:] = "avg") -> Arr, Arr, Fault {
    let chk = rs_pair_check("invresz", rr, ri, pr, pi, k, tol, rs_rule(rtype));
    if chk.kind != F_NONE { return empty_arr(), empty_arr(), chk; }
    var kt: f64[>..] = [];
    var kn: i64 = k.len;
    while kn > 0 && k[kn - 1] == 0.0 { kn -= 1; }
    var i: i64 = 0;
    while i < kn { kt.push(k[i]); i += 1; }
    let p = rs_zip(pr, pi);
    var up: Cplx[>..] = [];
    var mult: i64[>..] = [];
    rs_group_adjacent(p, tol, up, mult);
    var flat: Cplx[>..] = [];
    var off: i64[>..] = [];
    var lens: i64[>..] = [];
    var den: Cplx[>..] = [];
    rs_factors(up, mult, flat, off, lens, den);
    var num: Cplx[>..] = [];
    if kt.len > 0 {
        var kc: Cplx[>..] = [];
        i = kt.len - 1;
        while i >= 0 { kc.push(Cplx { re: kt[i], im: 0.0 }); i -= 1; }
        let dr = rs_reverse(den);
        rs_polymul(kc, dr, num);
    }
    var t: i64 = 0;
    while t < rr.len && t < off.len {
        var f: Cplx[>..] = [];
        var q: i64 = lens[t] - 1;
        while q >= 0 { f.push(flat[off[t] + q]); q -= 1; }
        let nn = rs_add_scaled(num, Cplx { re: rr[t], im: ri[t] }, f);
        num = copy(nn);
        t += 1;
    }
    let numr = rs_reverse(num);
    var bo: f64[>..] = [];
    var ao: f64[>..] = [];
    if !rs_real(numr, bo) || !rs_real(den, ao) {
        return empty_arr(), empty_arr(), fault(F_VALUE, "invresz", "the result holds a complex coefficient");
    }
    return line_arr(copy(bo)), line_arr(copy(ao)), no_fault();
}

// unique_roots finds the unique roots and their multiplicity. A root
// joins the group of the first unused root that lies within tol of it.
// The rule "min" or "max" compares by the real part, then by the
// imaginary part. The groups keep the order of their first member.
fn unique_roots(pr: f64[:], pi: f64[:], tol: f64 = 0.001, rtype: const u8[:] = "min") -> Arr, Arr, Arr, Fault {
    let rule = rs_rule(rtype);
    if pr.len != pi.len {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "unique_roots", "the real parts and the imaginary parts must hold the same count");
    }
    if rule < 0 { return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "unique_roots", "rtype must be avg, min or max"); }
    if tol != tol || tol < 0.0 || tol - tol != 0.0 {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "unique_roots", "tol must be a finite number that is not negative");
    }
    if !rs_finite(pr) || !rs_finite(pi) {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "unique_roots", "a root is not a finite number");
    }
    if pr.len > MAX_UNIQUE_ROOTS {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_LIMIT, "unique_roots", "more roots than the limit of unique_roots");
    }
    let p = rs_zip(pr, pi);
    var up: Cplx[>..] = [];
    var mult: i64[>..] = [];
    rs_group(p, tol, rule, up, mult);
    var ur: f64[>..] = [];
    var ui: f64[>..] = [];
    var um: f64[>..] = [];
    var i: i64 = 0;
    while i < up.len { ur.push(up[i].re); ui.push(up[i].im); um.push(mult[i] * 1.0); i += 1; }
    return line_arr(copy(ur)), line_arr(copy(ui)), line_arr(copy(um)), no_fault();
}

// residuez gives the partial fraction expansion of b over a in negative
// powers of z. The coefficients run from the constant term up. The
// result and the rules follow residue, and the poles stay the roots of
// a in z. The direct terms run from the constant term up.
fn residuez(b: f64[:], a: f64[:], tol: f64 = 0.001, rtype: const u8[:] = "avg") -> Arr, Arr, Arr, Arr, Arr, Fault {
    let rule = rs_rule(rtype);
    if rule < 0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "residuez", "rtype must be avg, min or max");
    }
    if tol != tol || tol < 0.0 || tol - tol != 0.0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "residuez", "tol must be a finite number that is not negative");
    }
    if !rs_finite(b) || !rs_finite(a) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "residuez", "a coefficient is not a finite number");
    }
    if b.len > MAX_RESIDUE_TERMS || a.len > MAX_RESIDUE_TERMS {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_LIMIT, "residuez", "the polynomial holds more terms than the limit of residuez");
    }
    // The trailing zeros drop here, where residue drops the leading ones.
    var bn: i64 = b.len;
    while bn > 0 && b[bn - 1] == 0.0 { bn -= 1; }
    var an: i64 = a.len;
    while an > 0 && a[an - 1] == 0.0 { an -= 1; }
    if an == 0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "residuez", "the denominator holds no non-zero term");
    }
    if a[0] == 0.0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "residuez", "the first coefficient of the denominator must not be zero");
    }
    var at: f64[>..] = [];
    var i: i64 = 0;
    while i < an { at.push(a[i]); i += 1; }
    let rts, rf = roots_of(at);
    if rf.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), rf; }
    if bn == 0 {
        var order: i64[>..] = [];
        rs_order(rts, order);
        var zr: f64[>..] = [];
        var pr: f64[>..] = [];
        var pi: f64[>..] = [];
        i = 0;
        while i < order.len { zr.push(0.0); pr.push(rts[order[i]].re); pi.push(rts[order[i]].im); i += 1; }
        return line_arr(copy(zr)), line_arr(copy(zr)), line_arr(copy(pr)), line_arr(copy(pi)), empty_arr(), no_fault();
    }
    var brev: f64[>..] = [];
    i = bn - 1;
    while i >= 0 { brev.push(b[i]); i -= 1; }
    var arev: f64[>..] = [];
    i = an - 1;
    while i >= 0 { arev.push(a[i]); i -= 1; }
    var krev: f64[>..] = [];
    var num: f64[>..] = [];
    if brev.len < arev.len {
        num = copy(brev);
    } else {
        let q, rm, df = polydiv(brev, arev);
        if df.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), df; }
        krev = copy(q.data);
        var lead: i64 = 0;
        while lead < rm.data.len - 1 && mathshim::abs_of(rm.data[lead]) <= 0.00000001 { lead += 1; }
        var t: i64 = lead;
        while t < rm.data.len { num.push(rm.data[t]); t += 1; }
    }
    var cnum: Cplx[>..] = [];
    i = 0;
    while i < num.len { cnum.push(Cplx { re: num[i], im: 0.0 }); i += 1; }
    var sp: Cplx[>..] = [];
    var sm: i64[>..] = [];
    rs_unique(rts, at, tol, rule, sp, sm);
    var inv: Cplx[>..] = [];
    i = 0;
    while i < sp.len { inv.push(c_div(Cplx { re: 1.0, im: 0.0 }, sp[i])); i += 1; }
    var res: Cplx[>..] = [];
    if sp.len > 0 { rs_residues(inv, sm, cnum, res); }
    var rr: f64[>..] = [];
    var ri: f64[>..] = [];
    var pr: f64[>..] = [];
    var pi: f64[>..] = [];
    // Each residue scales by the power of the negated pole and divides by
    // the last coefficient of the denominator.
    var idx: i64 = 0;
    var gi: i64 = 0;
    while gi < sp.len {
        let np = Cplx { re: -sp[gi].re, im: -sp[gi].im };
        var pw = Cplx { re: 1.0, im: 0.0 };
        var m: i64 = 0;
        while m < sm[gi] {
            pw = c_mul(pw, np);
            let v = c_div(c_mul(res[idx], pw), Cplx { re: arev[0], im: 0.0 });
            rr.push(v.re);
            ri.push(v.im);
            pr.push(sp[gi].re);
            pi.push(sp[gi].im);
            idx += 1;
            m += 1;
        }
        gi += 1;
    }
    var kk: f64[>..] = [];
    i = krev.len - 1;
    while i >= 0 { kk.push(krev[i]); i -= 1; }
    return line_arr(copy(rr)), line_arr(copy(ri)), line_arr(copy(pr)), line_arr(copy(pi)), line_arr(copy(kk)), no_fault();
}

