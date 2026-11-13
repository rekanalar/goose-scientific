// The second order sections: the pairing of the poles with the zeros.
//
// The second order sections of a filter.
//
// A filter of a high order loses digits when it holds one long
// polynomial. A cascade of sections of the order two keeps the digits.
// The functions of this file follow SciPy: butter with the sections,
// zpk2sos, tf2sos, sosfilt_zi and sosfiltfilt. The pairing of the
// poles and the zeros is the pairing "nearest" of SciPy.
namespace scientific;

import mathshim;

// The tolerance of the pairing of conjugates: 100 times the machine
// epsilon.
const CONJ_TOL: f64 = 2.220446049250313e-14;

// cx_before tells whether a holds its place before b in the order of
// the real part, then the size of the imaginary part.
fn cx_before(a: Cplx, b: Cplx) -> bool {
    if a.re != b.re { return a.re < b.re; }
    return mathshim::abs_of(a.im) < mathshim::abs_of(b.im);
}

// cx_sort puts a list in the order of cx_before. The sort keeps the
// order of equal elements.
fn cx_sort(a: Cplx[>..]&) {
    var i: i64 = 1;
    while i < a.len {
        let key = a[i];
        var j = i - 1;
        while j >= 0 {
            if !cx_before(key, a[j]) { break; }
            a[j + 1] = a[j];
            j -= 1;
        }
        a[j + 1] = key;
        i += 1;
    }
}

// cx_sort_chunk puts the places from lo to hi of a list in the order
// of the size of the imaginary part. The sort keeps the order of equal
// elements.
fn cx_sort_chunk(a: Cplx[>..]&, lo: i64, hi: i64) {
    var i = lo + 1;
    while i <= hi {
        let key = a[i];
        var j = i - 1;
        while j >= lo {
            if !(mathshim::abs_of(key.im) < mathshim::abs_of(a[j].im)) { break; }
            a[j + 1] = a[j];
            j -= 1;
        }
        a[j + 1] = key;
        i += 1;
    }
}

// cx_delete removes the place idx of a list.
fn cx_delete(a: Cplx[>..]&, idx: i64) {
    var i = idx;
    while i + 1 < a.len { a[i] = a[i + 1]; i += 1; }
    a.pop();
}

// cx_count_real counts the elements whose imaginary part is zero.
fn cx_count_real(a: Cplx[:]) -> i64 {
    var n: i64 = 0;
    var i: i64 = 0;
    while i < a.len { if a[i].im == 0.0 { n += 1; } i += 1; }
    return n;
}

// cx_worst gives the place of the pole that sits nearest to the unit
// circle. The first of equal places wins.
fn cx_worst(a: Cplx[:]) -> i64 {
    var best: i64 = 0;
    var bv = mathshim::abs_of(1.0 - c_abs(a[0]));
    var i: i64 = 1;
    while i < a.len {
        let v = mathshim::abs_of(1.0 - c_abs(a[i]));
        if v < bv { bv = v; best = i; }
        i += 1;
    }
    return best;
}

// cx_nearest gives the place of the element nearest to a point. The
// kind 0 takes any element, 1 takes a real one and 2 takes a complex
// one. The result is -1 when no element fits.
fn cx_nearest(a: Cplx[:], to: Cplx, kind: i32) -> i64 {
    var best: i64 = -1;
    var bv: f64 = 0.0;
    var i: i64 = 0;
    while i < a.len {
        var ok = true;
        if kind == 1 && a[i].im != 0.0 { ok = false; }
        if kind == 2 && a[i].im == 0.0 { ok = false; }
        if ok {
            let v = c_abs(c_sub(a[i], to));
            if best < 0 || v < bv { bv = v; best = i; }
        }
        i += 1;
    }
    return best;
}

// cx_split splits a list into the complex numbers with a positive
// imaginary part, which stand for the pairs, and the real numbers. It
// is the split of SciPy for the pairs of conjugates. A complex number
// with no partner is a fault.
fn cx_split(z: Cplx[>..]&, comp: Cplx[>..]&, real: Cplx[>..]&) -> Fault {
    cx_sort(z);
    var zp: Cplx[>..] = [];
    var zn: Cplx[>..] = [];
    var i: i64 = 0;
    while i < z.len {
        let v = z[i];
        if mathshim::abs_of(v.im) <= CONJ_TOL * c_abs(v) {
            real.push(Cplx { re: v.re, im: 0.0 });
        } else if v.im > 0.0 {
            zp.push(v);
        } else {
            zn.push(v);
        }
        i += 1;
    }
    if zp.len != zn.len {
        return fault(F_VALUE, "zpk2sos", "a complex number holds no conjugate partner");
    }
    // A run of places with the same real part sorts by the size of the
    // imaginary part.
    let m = zp.len - 1;
    i = 0;
    while i < m {
        if zp[i + 1].re - zp[i].re <= CONJ_TOL * c_abs(zp[i]) {
            var j = i;
            while j < m {
                if !(zp[j + 1].re - zp[j].re <= CONJ_TOL * c_abs(zp[j])) { break; }
                j += 1;
            }
            cx_sort_chunk(zp, i, j);
            cx_sort_chunk(zn, i, j);
            i = j;
        } else {
            i += 1;
        }
    }
    i = 0;
    while i < zp.len {
        comp.push(Cplx { re: (zp[i].re + zn[i].re) / 2.0, im: (zp[i].im - zn[i].im) / 2.0 });
        i += 1;
    }
    return no_fault();
}

// cx_join appends two lists to an empty list.
fn cx_join(a: Cplx[:], b: Cplx[:], out: Cplx[>..]&) {
    var i: i64 = 0;
    while i < a.len { out.push(a[i]); i += 1; }
    i = 0;
    while i < b.len { out.push(b[i]); i += 1; }
}

// sos_poly writes the real polynomial of up to two roots into three
// places of a row, right aligned, from the place at. The polynomial
// holds the coefficients from the highest power.
fn sos_poly(row: f64[>..]&, at: i64, r1: Cplx, r2: Cplx, count: i64) {
    var c0: f64 = 1.0;
    var c1: f64 = 0.0;
    var c2: f64 = 0.0;
    if count == 1 {
        c1 = -r1.re;
    } else if count == 2 {
        let s = c_add(r1, r2);
        let pr = c_mul(r1, r2);
        c1 = -s.re;
        c2 = pr.re;
    }
    if count == 0 {
        row[at + 2] = c0;
    } else if count == 1 {
        row[at + 1] = c0;
        row[at + 2] = c1;
    } else {
        row[at] = c0;
        row[at + 1] = c1;
        row[at + 2] = c2;
    }
}

// sos_from_zpk builds the sections of a filter from its zeros, its
// poles and its gain. The method is zpk2sos of SciPy with the pairing
// nearest. The lists change.
fn sos_from_zpk(z: Cplx[>..]&, p: Cplx[>..]&, k: f64) -> Arr, Fault {
    let zero = Cplx { re: 0.0, im: 0.0 };
    while p.len < z.len { p.push(zero); }
    while z.len < p.len { z.push(zero); }
    if p.len == 0 {
        return empty_arr(), fault(F_VALUE, "zpk2sos", "the filter holds no pole and no zero");
    }
    let n_sections = (p.len + 1) / 2;
    if p.len % 2 == 1 { p.push(zero); z.push(zero); }
    var zc: Cplx[>..] = [];
    var zr: Cplx[>..] = [];
    var pc: Cplx[>..] = [];
    var pr: Cplx[>..] = [];
    let f1 = cx_split(z, zc, zr);
    if f1.kind != F_NONE { return empty_arr(), f1; }
    let f2 = cx_split(p, pc, pr);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    var zl: Cplx[>..] = [];
    var pl: Cplx[>..] = [];
    cx_join(zc, zr, zl);
    cx_join(pc, pr, pl);
    var rows: f64[>..] = [];
    var q: i64 = 0;
    while q < n_sections * 6 { rows.push(0.0); q += 1; }
    var si = n_sections - 1;
    while si >= 0 {
        if pl.len < 1 {
            return empty_arr(), fault(F_VALUE, "zpk2sos", "the poles do not pair with the zeros");
        }
        let p1i = cx_worst(pl);
        let p1 = pl[p1i];
        cx_delete(pl, p1i);
        var row: f64[>..] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        if p1.im == 0.0 && cx_count_real(pl) == 0 {
            // One real pole is left.
            let zi = cx_nearest(zl, p1, 1);
            if zi < 0 { return empty_arr(), fault(F_VALUE, "zpk2sos", "a real pole holds no real zero"); }
            let z1 = zl[zi];
            cx_delete(zl, zi);
            sos_poly(row, 0, z1, zero, 2);
            sos_poly(row, 3, p1, zero, 2);
        } else if pl.len + 1 == zl.len && p1.im != 0.0 && cx_count_real(pl) == 1 && cx_count_real(zl) == 1 {
            // A complex pole pairs with a complex zero, and the real
            // zero and the real pole stay for the next section.
            let zi = cx_nearest(zl, p1, 2);
            if zi < 0 { return empty_arr(), fault(F_VALUE, "zpk2sos", "a complex pole holds no complex zero"); }
            let z1 = zl[zi];
            cx_delete(zl, zi);
            sos_poly(row, 0, z1, c_conj(z1), 2);
            sos_poly(row, 3, p1, c_conj(p1), 2);
        } else {
            var p2 = c_conj(p1);
            if p1.im == 0.0 {
                // The second pole is the real pole nearest to the
                // unit circle.
                var reals: Cplx[>..] = [];
                var places: i64[>..] = [];
                var t: i64 = 0;
                while t < pl.len {
                    if pl[t].im == 0.0 { reals.push(pl[t]); places.push(t); }
                    t += 1;
                }
                let pj = places[cx_worst(reals)];
                p2 = pl[pj];
                cx_delete(pl, pj);
            }
            if zl.len > 0 {
                let zi = cx_nearest(zl, p1, 0);
                let z1 = zl[zi];
                cx_delete(zl, zi);
                if z1.im != 0.0 {
                    sos_poly(row, 0, z1, c_conj(z1), 2);
                } else if zl.len > 0 {
                    let zj = cx_nearest(zl, p1, 1);
                    if zj < 0 { return empty_arr(), fault(F_VALUE, "zpk2sos", "a pole holds no second real zero"); }
                    let z2 = zl[zj];
                    cx_delete(zl, zj);
                    sos_poly(row, 0, z1, z2, 2);
                } else {
                    sos_poly(row, 0, z1, zero, 1);
                }
            } else {
                sos_poly(row, 0, zero, zero, 0);
            }
            sos_poly(row, 3, p1, p2, 2);
        }
        var c: i64 = 0;
        while c < 6 { rows[si * 6 + c] = row[c]; c += 1; }
        si -= 1;
    }
    var c: i64 = 0;
    while c < 3 { rows[c] = rows[c] * k; c += 1; }
    var s: i64[..8] = [];
    s.push(n_sections);
    s.push(6);
    return Arr { ndim: 2, shape: s, data: copy(rows) }, no_fault();
}

// zpk2sos gives the sections of a filter from its zeros, its poles
// and its gain. Each list of complex numbers holds the real parts and
// the imaginary parts in two arrays. The complex numbers must come in
// pairs of conjugates.
fn zpk2sos(zr: f64[:], zi: f64[:], pr: f64[:], pi_: f64[:], k: f64) -> Arr, Fault {
    if zr.len != zi.len || pr.len != pi_.len {
        return empty_arr(), fault(F_SHAPE, "zpk2sos", "the parts hold a different count");
    }
    if zr.len > 64 || pr.len > 64 {
        return empty_arr(), fault(F_LIMIT, "zpk2sos", "the filter holds more than 64 poles or zeros");
    }
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    var i: i64 = 0;
    while i < zr.len { z.push(Cplx { re: zr[i], im: zi[i] }); i += 1; }
    i = 0;
    while i < pr.len { p.push(Cplx { re: pr[i], im: pi_[i] }); i += 1; }
    return sos_from_zpk(z, p, k);
}

// tf2sos gives the sections of a filter from its numerator and its
// denominator. The zeros and the poles come from the roots of the two
// polynomials. A filter of a high order loses digits in this path.
fn tf2sos(b: f64[:], a: f64[:]) -> Arr, Fault {
    if b.len < 1 || a.len < 1 || a[0] == 0.0 || b[0] == 0.0 {
        return empty_arr(), fault(F_VALUE, "tf2sos", "a leading coefficient holds zero or the list holds no element");
    }
    if a.len > 65 || b.len > 65 {
        return empty_arr(), fault(F_LIMIT, "tf2sos", "the filter holds more than 64 poles or zeros");
    }
    let k = b[0] / a[0];
    var z: Cplx[>..] = [];
    var p: Cplx[>..] = [];
    if b.len > 1 {
        let zr, zim, f = roots(b);
        if f.kind != F_NONE { return empty_arr(), f; }
        var i: i64 = 0;
        while i < zr.data.len { z.push(Cplx { re: zr.data[i], im: zim.data[i] }); i += 1; }
    }
    if a.len > 1 {
        let pr, pim, f = roots(a);
        if f.kind != F_NONE { return empty_arr(), f; }
        var i: i64 = 0;
        while i < pr.data.len { p.push(Cplx { re: pr.data[i], im: pim.data[i] }); i += 1; }
    }
    return sos_from_zpk(z, p, k);
}

