// The placement of the poles of a closed loop.
//
// place_poles gives the gain matrix K whose closed loop A - B K holds
// the poles that the caller asks for. The call follows the aim of
// scipy.signal.place_poles. The fast path of SciPy, a matrix B of full
// row rank, solves the system of the poles in one step and holds the
// identity as the transfer matrix. The path for a matrix of a lower
// rank tries the constructions of the file next to this one and keeps
// the first gain whose closed loop holds the poles. The poles must
// come in pairs of conjugates. The methods YT and KNV0 run no loop of
// their own here, because the gain of these constructions places the
// poles in one step.
namespace scientific;

import mathshim;

// The largest side of the matrices of the placement.
const PLACE_SIDE_MAX: i64 = 256;

// place_order_poles sorts the poles in the order of SciPy: the real
// poles from the smallest to the largest, then the pairs of
// conjugates. A pair comes as the pole with the part below zero
// first. The parts of a size at or below tol name a real pole. The
// poles of the input come with tol zero, so the test of a real pole is
// exact there. The result holds the real parts and the imaginary
// parts.
fn place_order_poles(wr: f64[:], wi: f64[:], tol: f64, otr: f64[>..]&, oti: f64[>..]&) -> Fault {
    if wr.len != wi.len {
        return fault(F_SHAPE, "place_poles", "the real parts and the imaginary parts hold a different count");
    }
    var real_r: f64[>..] = [];
    var neg_r: f64[>..] = [];
    var neg_i: f64[>..] = [];
    var pos_r: f64[>..] = [];
    var pos_i: f64[>..] = [];
    var i: i64 = 0;
    while i < wr.len {
        if mathshim::abs_of(wi[i]) <= tol {
            real_r.push(wr[i]);
        } else if wi[i] < 0.0 {
            neg_r.push(wr[i]);
            neg_i.push(wi[i]);
        } else {
            pos_r.push(wr[i]);
            pos_i.push(wi[i]);
        }
        i += 1;
    }
    if neg_r.len != pos_r.len {
        return fault(F_VALUE, "place_poles", "the complex poles must come with their conjugates");
    }
    if tol == 0.0 {
        // Every pole with a part below zero needs its exact partner in
        // the list of the input.
        i = 0;
        while i < neg_r.len {
            var found = false;
            var j: i64 = 0;
            while j < wr.len && !found {
                if wr[j] == neg_r[i] && wi[j] == -neg_i[i] { found = true; }
                j += 1;
            }
            if !found {
                return fault(F_VALUE, "place_poles", "the complex poles must come with their conjugates");
            }
            i += 1;
        }
    }
    // The real poles rise.
    var s: i64 = 1;
    while s < real_r.len {
        let key = real_r[s];
        var j = s - 1;
        while j >= 0 && real_r[j] > key { real_r[j + 1] = real_r[j]; j -= 1; }
        real_r[j + 1] = key;
        s += 1;
    }
    // The poles with a part below zero rise by the real part, then by
    // the imaginary part.
    s = 1;
    while s < neg_r.len {
        var kr = neg_r[s];
        var ki = neg_i[s];
        var j = s - 1;
        while j >= 0 && (neg_r[j] > kr || (neg_r[j] == kr && neg_i[j] > ki)) {
            neg_r[j + 1] = neg_r[j];
            neg_i[j + 1] = neg_i[j];
            j -= 1;
        }
        neg_r[j + 1] = kr;
        neg_i[j + 1] = ki;
        s += 1;
    }
    // The poles with a part above zero follow their partners in the
    // same order.
    s = 1;
    while s < pos_r.len {
        var kr = pos_r[s];
        var ki = pos_i[s];
        var j = s - 1;
        while j >= 0 && (pos_r[j] > kr || (pos_r[j] == kr && pos_i[j] < ki)) {
            pos_r[j + 1] = pos_r[j];
            pos_i[j + 1] = pos_i[j];
            j -= 1;
        }
        pos_r[j + 1] = kr;
        pos_i[j + 1] = ki;
        s += 1;
    }
    i = 0;
    while i < real_r.len { otr.push(real_r[i]); oti.push(0.0); i += 1; }
    i = 0;
    while i < neg_r.len {
        otr.push(neg_r[i]); oti.push(neg_i[i]);
        otr.push(pos_r[i]); oti.push(pos_i[i]);
        i += 1;
    }
    return no_fault();
}

// place_poly_poles builds the polynomial of the poles that the caller
// asks for. A real pole adds the factor of one power of s. A pair adds
// the factor of two powers with real coefficients. The coefficients
// come from the highest power, and the leading one is one.
fn place_poly_poles(pr: f64[:], pi_: f64[:], out: f64[>..]&) {
    var poly: f64[>..] = [1.0];
    var i: i64 = 0;
    while i < pr.len {
        var next: f64[>..] = [];
        if pi_[i] == 0.0 {
            var j: i64 = 0;
            while j < poly.len { next.push(poly[j]); j += 1; }
            next.push(0.0);
            j = 0;
            while j < poly.len { next[j + 1] -= poly[j] * pr[i]; j += 1; }
            poly = copy(next);
        } else if pi_[i] > 0.0 {
            // The factor of the pair: s^2 - 2 re s + re^2 + im^2.
            let re = pr[i];
            let im = pi_[i];
            var j: i64 = 0;
            while j < poly.len { next.push(poly[j]); j += 1; }
            next.push(0.0);
            next.push(0.0);
            j = 0;
            while j < poly.len {
                next[j + 1] -= 2.0 * re * poly[j];
                next[j + 2] += (re * re + im * im) * poly[j];
                j += 1;
            }
            poly = copy(next);
        }
        i += 1;
    }
    var j: i64 = 0;
    while j < poly.len { out.push(poly[j]); j += 1; }
}

// place_krylov builds the matrix of controllability of a and one input
// b. The columns hold b, A b and the powers of A on b up to the side
// of a.
fn place_krylov(a, b: f64[:], out: f64[>..]&) {
    let n = a.shape[0];
    var i: i64 = 0;
    while i < n * n { out.push(0.0); i += 1; }
    i = 0;
    while i < n { out[i] = b[i]; i += 1; }
    var cur: f64[>..] = [];
    i = 0;
    while i < n { cur.push(b[i]); i += 1; }
    var col: i64 = 1;
    while col < n {
        var next: f64[>..] = [];
        i = 0;
        while i < n {
            var v: f64 = 0.0;
            var j: i64 = 0;
            while j < n { v += a.data[i * n + j] * cur[j]; j += 1; }
            next.push(v);
            out[col * n + i] = v;
            i += 1;
        }
        cur = copy(next);
        col += 1;
    }
}

// place_poles_ok tells whether the poles of a closed loop hold the
// poles that the caller asked for. Each pole of the request needs an
// unused partner of the closed loop within the tolerance.
fn place_poles_ok(wr: f64[:], wi: f64[:], pr: f64[:], pi_: f64[:]) -> bool {
    if wr.len != pr.len { return false; }
    var scale: f64 = 1.0;
    var i: i64 = 0;
    while i < pr.len {
        let v = mathshim::abs_of(pr[i]) + mathshim::abs_of(pi_[i]);
        if v > scale { scale = v; }
        i += 1;
    }
    let tol = 1.0e-5 * scale;
    var used: i64[>..] = [];
    i = 0;
    while i < wr.len { used.push(0); i += 1; }
    i = 0;
    while i < pr.len {
        var found = false;
        var j: i64 = 0;
        while j < wr.len && !found {
            if used[j] == 0
                && mathshim::abs_of(wr[j] - pr[i]) <= tol
                && mathshim::abs_of(wi[j] - pi_[i]) <= tol {
                used[j] = 1;
                found = true;
            }
            j += 1;
        }
        if !found { return false; }
        i += 1;
    }
    return true;
}

// place_poles gives the gain of the closed loop and the poles. The
// matrix a holds the states and the flat matrix b holds the inputs, in
// n rows of m values, read by rows. The poles come as two lists, the
// real parts and the imaginary parts, and must hold one pole for each
// state, in pairs of conjugates. The results are the gain in m rows of
// n values, the poles of the closed loop, the poles that the caller
// asked for, both in the order of SciPy, the transfer matrix as a
// complex array, the relative tolerance that the call reached and the
// count of the iterations. Both paths place the poles in one step, so
// the tolerance holds NaN on the fast path and zero on the other, and
// the count of the iterations holds zero.
fn place_poles(a, b: f64[:], pr: f64[:], pi_: f64[:], method: const u8[:] = "YT",
               rtol: f64 = 0.001, maxiter: i64 = 30) -> Arr, Arr, Arr, Arr, Arr, CArr, f64, i64, Fault {
    let empty_ca = carr_empty();
    if method != "YT" && method != "KNV0" {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
            fault(F_VALUE, "place_poles", "the method holds a name that is not YT or KNV0");
    }
    if maxiter < 1 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
            fault(F_VALUE, "place_poles", "maxiter holds a value below one");
    }
    if rtol > 1.0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
            fault(F_VALUE, "place_poles", "rtol holds a value above one");
    }
    if a.ndim != 2 || a.shape.len != 2 || a.shape[0] != a.shape[1] || a.shape[0] < 1 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
            fault(F_SHAPE, "place_poles", "the matrix of the states is not square");
    }
    let n = a.shape[0];
    if n > PLACE_SIDE_MAX {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
            fault(F_LIMIT, "place_poles", "the side of the matrices is above the limit");
    }
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, "place_poles");
    if fa.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0, fa;
    }
    if b.len < 1 || b.len % n != 0 {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
            fault(F_SHAPE, "place_poles", "the matrix of the inputs does not split into rows of the states");
    }
    let m = b.len / n;
    if !all_finite(a.data) || !all_finite(b) || !all_finite(pr) || !all_finite(pi_) {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
            fault(F_VALUE, "place_poles", "a value of the matrices or the poles is not finite");
    }
    var orreq: f64[>..] = [];
    var oreqi: f64[>..] = [];
    let fo = place_order_poles(pr, pi_, 0.0, orreq, oreqi);
    if fo.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0, fo;
    }
    if orreq.len != n {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
            fault(F_VALUE, "place_poles", "the poles must hold one pole for each state");
    }
    if method == "KNV0" {
        var i: i64 = 0;
        while i < oreqi.len {
            if oreqi[i] != 0.0 {
                return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
                    fault(F_VALUE, "place_poles", "the method KNV0 holds no complex poles");
            }
            i += 1;
        }
    }
    // The rank of the input decides the path.
    var bg: f64[>..] = [];
    var bi: i64 = 0;
    while bi < b.len { bg.push(b[bi]); bi += 1; }
    let bm = mat2(copy(bg), n, m);
    let rank, fr = matrix_rank(bm);
    if fr.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0, fr;
    }
    // A pole that the caller asks for more times than the rank of the
    // input cannot sit in the closed loop.
    var i: i64 = 0;
    while i < n {
        var count: i64 = 0;
        var j: i64 = 0;
        while j < n {
            if orreq[j] == orreq[i] && oreqi[j] == oreqi[i] { count += 1; }
            j += 1;
        }
        if count > rank {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
                fault(F_VALUE, "place_poles", "one pole is repeated more times than the rank of the input");
        }
        i += 1;
    }
    var gain: f64[>..] = [];
    var holds = false;
    if rank == n {
        // The fast path: the gain comes from one solve of the matrix
        // of the inputs against the poles on the diagonal. A pair of
        // conjugates sits in a block of two by two with real values.
        // SciPy solves A - diag(poles) against B, so the closed loop
        // A - B K holds the poles.
        var diagp: f64[>..] = [];
        i = 0;
        while i < n * n { diagp.push(0.0); i += 1; }
        var idx: i64 = 0;
        while idx < n {
            let re = orreq[idx];
            let im = oreqi[idx];
            diagp[idx * n + idx] = re;
            if im != 0.0 {
                // The partner of the pair sits at the next place, so
                // the pair takes a block of two by two.
                diagp[idx * n + idx + 1] = -im;
                diagp[(idx + 1) * n + idx + 1] = re;
                diagp[(idx + 1) * n + idx] = im;
                idx += 1;
            }
            idx += 1;
        }
        var rhs: f64[>..] = [];
        i = 0;
        while i < n * n { rhs.push(a.data[i] - diagp[i]); i += 1; }
        let rm = mat2(copy(rhs), n, n);
        if m == n {
            let kk, fk = solve(bm, rm);
            if fk.kind == F_NONE {
                gain = copy(kk.data);
                holds = true;
            }
        } else {
            i = 0;
            while i < n * m { gain.push(0.0); i += 1; }
            holds = true;
            var col: i64 = 0;
            while col < n && holds {
                var colv: f64[>..] = [];
                i = 0;
                while i < n { colv.push(rhs[i * n + col]); i += 1; }
                let kk, fk = lstsq(bm, flt_arr(copy(colv)));
                if fk.kind != F_NONE {
                    holds = false;
                } else {
                    i = 0;
                    while i < m { gain[i * n + col] = kk.data[i]; i += 1; }
                }
                col += 1;
            }
            if !holds { gain = []; }
        }
    }
    if !holds {
        // The path of a lower rank tries the constructions in the order
        // of their cost and keeps the first gain whose closed loop
        // holds the poles.
        if m > 1 {
            // The sets of powers run up and down, because some
            // systems hold their reach in the high powers.
            var off: i64 = 0;
            var down = false;
            while off <= 2 && !holds {
                var tryg: f64[>..] = [];
                if place_gen(a, b, n, m, orreq, oreqi, off, down, tryg) {
                    holds = place_gain_ok(a, b, n, m, tryg, orreq, oreqi);
                    if holds { gain = copy(tryg); }
                }
                if down { off += 1; }
                down = !down;
            }
        }
        if !holds {
            // The gain rides on one column of the input that reaches
            // every state. The polynomial of the poles fixes the
            // weights of the powers of A on that column.
            var c: i64 = 0;
            while c < m && !holds {
                var colv: f64[>..] = [];
                i = 0;
                while i < n { colv.push(b[i * m + c]); i += 1; }
                var kry: f64[>..] = [];
                place_krylov(a, colv, kry);
                let km = mat2(copy(kry), n, n);
                let r2, fr2 = matrix_rank(km);
                if fr2.kind == F_NONE && r2 == n {
                    let fp, fpf = form_charpoly(a, "place_poles");
                    if fpf.kind != F_NONE {
                        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0, fpf;
                    }
                    var psi: f64[>..] = [];
                    place_poly_poles(orreq, oreqi, psi);
                    var q: f64[>..] = [];
                    i = 0;
                    while i < n { q.push(psi[i + 1] - fp[i + 1]); i += 1; }
                    // The expansion of the ratio of the two polynomials
                    // at the infinite holds the weights.
                    var wts: f64[>..] = [];
                    wts.push(q[0]);
                    i = 1;
                    while i < n {
                        var v = q[i];
                        var j: i64 = 1;
                        while j <= i { v -= fp[j] * wts[i - j]; j += 1; }
                        wts.push(v);
                        i += 1;
                    }
                    let kt, ft = transpose(km);
                    if ft.kind != F_NONE {
                        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0, ft;
                    }
                    // The matrix of the powers already holds the
                    // powers in its rows, so the solve takes it as it
                    // stands.
                    let kk, fk = solve(km, flt_arr(copy(wts)));
                    if fk.kind == F_NONE {
                        var tryg: f64[>..] = [];
                        i = 0;
                        while i < m * n { tryg.push(0.0); i += 1; }
                        i = 0;
                        while i < n { tryg[c * n + i] = kk.data[i]; i += 1; }
                        if place_gain_ok(a, b, n, m, tryg, orreq, oreqi) {
                            gain = copy(tryg);
                            holds = true;
                        }
                    }
                }
                c += 1;
            }
        }
        if !holds {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0,
                fault(F_VALUE, "place_poles", "the poles cannot sit in the closed loop, so check the reach of the input");
        }
    }
    // The gain holds m rows of n values, so the closed loop is A - B K.
    var am: f64[>..] = [];
    i = 0;
    while i < n * n {
        var v = a.data[i];
        let r = i / n;
        let col = i % n;
        var j: i64 = 0;
        while j < m { v -= b[r * m + j] * gain[j * n + col]; j += 1; }
        am.push(v);
        i += 1;
    }
    let amm = mat2(copy(am), n, n);
    // The poles of the closed loop come in the order of SciPy.
    let cw, fcw = eigvals_complex(amm);
    if fcw.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0, fcw;
    }
    var scale: f64 = 1.0;
    i = 0;
    while i < cw.data.len / 2 {
        let v = mathshim::abs_of(cw.data[2 * i]) + mathshim::abs_of(cw.data[2 * i + 1]);
        if v > scale { scale = v; }
        i += 1;
    }
    var wr: f64[>..] = [];
    var wi: f64[>..] = [];
    i = 0;
    while i < cw.data.len / 2 {
        wr.push(cw.data[2 * i]);
        wi.push(cw.data[2 * i + 1]);
        i += 1;
    }
    var cr: f64[>..] = [];
    var ci: f64[>..] = [];
    let fc = place_order_poles(wr, wi, 1.0e-9 * scale, cr, ci);
    if fc.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0, fc;
    }
    // The transfer matrix holds the vectors of the closed loop on the
    // path of a lower rank and the identity on the fast path. The
    // vectors sit in the columns of the complex decomposition.
    var xr: f64[>..] = [];
    var xi: f64[>..] = [];
    if rank == n {
        i = 0;
        while i < n * n { xr.push(0.0); i += 1; }
        i = 0;
        while i < n { xr[i * n + i] = 1.0; i += 1; }
        i = 0;
        while i < n * n { xi.push(0.0); i += 1; }
    } else {
        // The call for the eigenvectors of a complex matrix takes the
        // real and the imaginary part of every value one after the
        // other. The closed loop holds no imaginary part.
        var amc: f64[>..] = [];
        i = 0;
        while i < n * n {
            amc.push(am[i]);
            amc.push(0.0);
            i += 1;
        }
        let amcs: i64[..8] = [];
        amcs.push(n);
        amcs.push(n);
        let cv, vv, fv = ceig(CArr { ndim: 2, shape: amcs, data: copy(amc) });
        if fv.kind != F_NONE {
            return empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_arr(), empty_ca, 0.0, 0, fv;
        }
        i = 0;
        while i < n * n {
            xr.push(vv.data[2 * i]);
            xi.push(vv.data[2 * i + 1]);
            i += 1;
        }
    }
    var inter: f64[>..] = [];
    i = 0;
    while i < n * n {
        inter.push(xr[i]);
        inter.push(xi[i]);
        i += 1;
    }
    var sh: i64[..8] = [];
    sh.push(n);
    sh.push(n);
    var gs: i64[..8] = [];
    gs.push(m);
    gs.push(n);
    let rtol_out = if rank == n { 0.0 / 0.0 } else { 0.0 };
    return Arr { ndim: 2, shape: gs, data: copy(gain) }, flt_arr(copy(cr)),
        flt_arr(copy(ci)), flt_arr(copy(orreq)), flt_arr(copy(oreqi)),
        CArr { ndim: 2, shape: sh, data: copy(inter) }, rtol_out, 0, no_fault();
}

// place_gain_ok builds the closed loop of a trial gain and tells
// whether its poles hold the poles of the request.
fn place_gain_ok(a, b: f64[:], n: i64, m: i64, g: f64[:], pr: f64[:], pi_: f64[:]) -> bool {
    var am: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n {
        var v = a.data[i];
        let r = i / n;
        let col = i % n;
        var j: i64 = 0;
        while j < m { v -= b[r * m + j] * g[j * n + col]; j += 1; }
        am.push(v);
        i += 1;
    }
    let amm = mat2(copy(am), n, n);
    let cw, f = eigvals_complex(amm);
    if f.kind != F_NONE { return false; }
    var wr: f64[>..] = [];
    var wi: f64[>..] = [];
    i = 0;
    while i < cw.data.len / 2 {
        wr.push(cw.data[2 * i]);
        wi.push(cw.data[2 * i + 1]);
        i += 1;
    }
    return place_poles_ok(wr, wi, pr, pi_);
}
