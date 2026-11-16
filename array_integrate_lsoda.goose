// The LSODA method: the Adams method and the BDF method, and the
// switch between them.
//
// LSODA runs an Adams method of the order up to twelve while the
// system is not stiff, and the backward differentiation formulas of
// the order up to five when it turns stiff. The stiff part runs
// through the calls of the BDF method of SciPy that the package
// holds. The switch follows the paper of Hindmarsh and Petzold,
// "User documentation for LSODA", 2005. The package detects the
// stiffness from the behavior of the steps, so the steps differ from
// the steps of SciPy. The values agree to the tolerance.
//
namespace scientific;

import mathshim;

// The largest count of the past values that the Adams history holds.
const LSODA_HISTORY: i64 = 12;

// lsoda_basis gives the integrals of the Lagrange basis over the
// interval from zero to one. The nodes are the values of u. The
// result holds one value per node.
fn lsoda_basis(u: f64[:]) -> f64[>..] {
    let m = u.len;
    var out: f64[>..] = [];
    var j: i64 = 0;
    while j < m {
        var p: f64[>..] = [1.0];
        var den: f64 = 1.0;
        var k: i64 = 0;
        while k < m {
            if k != j {
                // The product gains one factor of (u - u[k]).
                var np: f64[>..] = [0.0 - u[k] * p[0]];
                var q: i64 = 1;
                while q <= p.len {
                    var lo = p[q - 1];
                    var hi: f64 = 0.0;
                    if q < p.len { hi = p[q]; }
                    np.push(lo - u[k] * hi);
                    q += 1;
                }
                p = copy(np);
                den *= u[j] - u[k];
            }
            k += 1;
        }
        var s: f64 = 0.0;
        var q: i64 = 0;
        while q < p.len { s += p[q] / (q + 1.0); q += 1; }
        out.push(s / den);
        j += 1;
    }
    return out;
}

// lsoda_nodes gives the nodes u of the history. The first node stands
// at the current time and the past nodes lie before it. The corrector
// adds the node at the end of the step.
fn lsoda_nodes(lt: f64[:], t: f64, h: f64, q: i64, corr: bool) -> f64[>..] {
    var u: f64[>..] = [];
    var j: i64 = 0;
    while j < q { u.push((lt[j] - t) / h); j += 1; }
    if corr { u.push(1.0); }
    return u;
}

// lsoda_interp gives the predicted state or the corrected state of
// one order. The flag corr picks the corrector, which adds the node
// at the end of the step and uses the values fp of the function
// there. The result goes into out.
fn lsoda_interp(t: f64, q: i64, h: f64, n: i64, y: f64[:], lt: f64[:], lf: f64[:], corr: bool, fp: f64[:], out: f64[>..]&) {
    var w: f64[>..] = [];
    if corr {
        let uc = lsoda_nodes(lt, t, h, q, true);
        w = lsoda_basis(uc);
    } else {
        let up = lsoda_nodes(lt, t, h, q, false);
        w = lsoda_basis(up);
    }
    var i: i64 = 0;
    while i < n {
        var acc: f64 = 0.0;
        var j: i64 = 0;
        while j < q { acc += lf[j * n + i] * w[j]; j += 1; }
        if corr { acc += fp[i] * w[q]; }
        out.push(y[i] + h * acc);
        i += 1;
    }
}

// lsoda_err gives the error of the Adams pair of one order: the root
// of the mean square of the gap between the corrector and the
// predictor over the scale of the state.
fn lsoda_err(n: i64, y: f64[:], yp: f64[:], yc: f64[:], atol: f64, rtol: f64) -> f64 {
    var sq: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var big = mathshim::abs_of(y[i]);
        if mathshim::abs_of(yp[i]) > big { big = mathshim::abs_of(yp[i]); }
        let sc = atol + rtol * big;
        sq.push(mathshim::abs_of(yc[i] - yp[i]) / sc);
        i += 1;
    }
    if sq.len == 0 { return 0.0; }
    return ivp_rms(sq);
}

// lsoda_step takes one accepted step of the LSODA method. The state
// li holds the mode in the first place: the mode zero is the Adams
// method and the mode one is the BDF method. The other places hold
// the order of the Adams method, the count of the hard rejections,
// the count of the steps of one method and a switch that the step
// before queued. The buffers lt and lf hold the times and the values
// of the Adams history. It gives the
// new state, the values and the derivative at the end of the step,
// and the time of the end.
fn lsoda_step<F: fn(f64, f64[:]) -> f64[:]>(t: f64, t_bound: f64, dir: f64, n: i64, nev: i64, atol: f64, rtol: f64, ntol: f64, max_step: f64, y: f64[:], f: f64[:], lt: f64[>..]&, lf: f64[>..]&, li: i64[>..]&, dm: f64[>..]&, fst: f64[>..]&, ist: i64[>..]&, jm: f64[>..]&, jfac: f64[>..]&, lum: f64[>..]&, piv: i64[>..]&, ynew: f64[>..]&, fnew: f64[>..]&, tn: f64[>..]&) -> bool, Fault {
    var took = false;
    var again = true;
    while again {
        again = false;
        let min_step0 = 10.0 * mathshim::abs_of(t) * 2.220446049250313e-16;
        var min_step = min_step0;
        if min_step < 5.0e-323 { min_step = 5.0e-323; }
        // A switch that the step before queued. The switch happens
        // here, so the rows of the step before stay with their own
        // mode.
        if li.len > 4 && li[4] == 1 {
            li[4] = 0;
            let sp = lsoda_to_bdf(t, dir, n, nev, atol, y, f, mathshim::abs_of(fst[0]), lt, lf, li, dm, fst, ist, jm, jfac, lum, piv) { tt, yy => F(tt, copy(yy)) };
            if sp.kind != F_NONE { return false, sp; }
        }
        if li.len > 4 && li[4] == 2 {
            li[4] = 0;
            li[0] = 0;
            li[1] = 1;
            li[2] = 0;
            li[3] = 0;
            lt = [t];
            lf = [];
            var q0: i64 = 0;
            while q0 < n { lf.push(f[q0]); q0 += 1; }
            fst[0] = mathshim::abs_of(fst[0]);
        }
        if li[0] == 1 {
            // The mode of the BDF method. The step goes through the
            // calls of the method of the package.
            let bf = bdf_step(t, t_bound, dir, n, nev, atol, rtol, ntol, max_step, dm, fst, ist, jm, jfac, lum, piv, ynew, tn) { tt, yy => F(tt, copy(yy)) };
            if bf.kind != F_NONE { return false, bf; }
            let fb = F(tn[0], copy(ynew));
            if fb.len != n + nev {
                return false, fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
            }
            fnew.clear();
            var q1: i64 = 0;
            while q1 < fb.len { fnew.push(fb[q1]); q1 += 1; }
            took = true;
            li[3] += 1;
            // The test of the switch to the Adams method. The method
            // counts the steps between the tests. A small product of
            // the step and the estimate of the stiffness marks a
            // system that the Adams method can serve.
            if li[3] >= 25 {
                li[3] = 0;
                let d = lsoda_stiff(ynew, fnew, n);
                if d > 0.0 && fst[0] * d < 0.1 {
                    li[4] = 2;
                }
            }
            return took, no_fault();
        }
        // The mode of the Adams method.
        if fst.len < 1 { fst.push(mathshim::abs_of(t_bound - t)); }
        if fst[0] < 0.0 { fst[0] = -fst[0]; }
        var h_abs = fst[0];
        if h_abs < min_step { h_abs = min_step; }
        if h_abs > max_step { h_abs = max_step; }
        var q = li[1];
        if q < 1 { q = 1; }
        if q > LSODA_HISTORY { q = LSODA_HISTORY; }
        li[1] = q;
        var h = h_abs * dir;
        var tnew = t + h;
        if dir * (tnew - t_bound) > 0.0 {
            tnew = t_bound;
            h = tnew - t;
            h_abs = mathshim::abs_of(h);
            fst[0] = h_abs;
        }
        var yp: f64[>..] = [];
        lsoda_interp(t, q, h, n, y, lt, lf, false, yp, yp);
        let fp = F(tnew, copy(yp));
        if fp.len != n + nev {
            return false, fault(F_SHAPE, "solve_ivp", "the function holds a different count of values");
        }
        var yc: f64[>..] = [];
        lsoda_interp(t, q, h, n, y, lt, lf, true, fp, yc);
        let e = lsoda_err(n, y, yp, yc, atol, rtol);
        if e - e != 0.0 {
            return false, fault(F_VALUE, "solve_ivp", "the values of the function left the range");
        }
        if e <= 1.0 {
            // The step passes. The corrector holds the state and the
            // values of the function at the end of the step.
            li[2] = 0;
            li[3] += 1;
            // The order and the step for the next turn. The test
            // rates the order below, the order and the order above,
            // and takes the order with the largest predicted step.
            var best = q;
            var best_fac = lsoda_factor(e, q + 1);
            if q > 1 && lt.len >= q - 1 {
                var yp2: f64[>..] = [];
                var yc2: f64[>..] = [];
                lsoda_interp(t, q - 1, h, n, y, lt, lf, false, yp, yp2);
                lsoda_interp(t, q - 1, h, n, y, lt, lf, true, fp, yc2);
                let fd = lsoda_factor(lsoda_err(n, y, yp2, yc2, atol, rtol), q);
                if fd > best_fac { best = q - 1; best_fac = fd; }
            }
            if q < LSODA_HISTORY && lt.len >= q + 1 {
                var yp3: f64[>..] = [];
                var yc3: f64[>..] = [];
                lsoda_interp(t, q + 1, h, n, y, lt, lf, false, yp, yp3);
                lsoda_interp(t, q + 1, h, n, y, lt, lf, true, fp, yc3);
                let fu = lsoda_factor(lsoda_err(n, y, yp3, yc3, atol, rtol), q + 2);
                if fu > best_fac { best = q + 1; best_fac = fu; }
            }
            li[1] = best;
            var grow = best_fac;
            if grow > 10.0 { grow = 10.0; }
            fst[0] = h_abs * grow;
            // The history gains the new point at the front.
            var nt: f64[>..] = [tnew];
            var nf: f64[>..] = [];
            var q2: i64 = 0;
            while q2 < n { nf.push(fp[q2]); q2 += 1; }
            q2 = 0;
            while q2 < lt.len && nt.len < LSODA_HISTORY { nt.push(lt[q2]); q2 += 1; }
            q2 = 0;
            while q2 < lf.len && nf.len < LSODA_HISTORY * n { nf.push(lf[q2]); q2 += 1; }
            lt.clear();
            var q3: i64 = 0;
            while q3 < nt.len { lt.push(nt[q3]); q3 += 1; }
            lf.clear();
            q3 = 0;
            while q3 < nf.len { lf.push(nf[q3]); q3 += 1; }
            ynew.clear();
            q3 = 0;
            while q3 < yc.len { ynew.push(yc[q3]); q3 += 1; }
            fnew.clear();
            q3 = 0;
            while q3 < fp.len { fnew.push(fp[q3]); q3 += 1; }
            tn[0] = tnew;
            return true, no_fault();
        }
        // The step fails. The step shrinks, and a run of hard steps
        // turns the method to the BDF one and takes the step again.
        var factor = 0.2;
        if e > 0.0 && e - e == 0.0 {
            let qf = (q + 1) as f64;
            factor = 0.9 * mathshim::power(e, 0.0 - 1.0 / qf);
            if factor < 0.2 { factor = 0.2; }
            if factor > 1.0 { factor = 1.0; }
        }
        fst[0] = h_abs * factor;
        if factor <= 0.5 {
            li[2] += 1;
            if li[2] >= 3 {
                let sf = lsoda_to_bdf(t, dir, n, nev, atol, y, f, h_abs, lt, lf, li, dm, fst, ist, jm, jfac, lum, piv) { tt, yy => F(tt, copy(yy)) };
                if sf.kind != F_NONE { return false, sf; }
                again = true;
            }
        } else {
            li[2] = 0;
            again = true;
        }
    }
    return false, no_fault();
}

// lsoda_factor gives the predicted growth of the step for one order.
// The error of the pair scales as one power of the step more than the
// order of the predictor q.
fn lsoda_factor(e: f64, q: i64) -> f64 {
    if e <= 0.0 { return 10.0; }
    if e - e != 0.0 { return 0.0; }
    let qf = q as f64;
    return 0.9 * mathshim::power(e, 0.0 - 1.0 / qf);
}

// lsoda_stiff gives the estimate of the stiffness: the root of the
// mean square of the derivative over the root of the mean square of
// the state.
fn lsoda_stiff(y: f64[:], f: f64[:], n: i64) -> f64 {
    var ys: f64[>..] = [];
    var fs: f64[>..] = [];
    var q: i64 = 0;
    while q < n { ys.push(mathshim::abs_of(y[q])); fs.push(mathshim::abs_of(f[q])); q += 1; }
    let ry = ivp_rms(ys);
    let rf = ivp_rms(fs);
    if !(ry > 0.0) { return 0.0; }
    return rf / ry;
}

// lsoda_to_bdf turns the method to the BDF one. The differences start
// from the state and the derivative of the current point, and the
// Jacobian comes from the differences of the function.
fn lsoda_to_bdf<F: fn(f64, f64[:]) -> f64[:]>(t: f64, dir: f64, n: i64, nev: i64, atol: f64, y: f64[:], f: f64[:], h_abs: f64, lt: f64[>..]&, lf: f64[>..]&, li: i64[>..]&, dm: f64[>..]&, fst: f64[>..]&, ist: i64[>..]&, jm: f64[>..]&, jfac: f64[>..]&, lum: f64[>..]&, piv: i64[>..]&) -> Fault {
    li[0] = 1;
    li[1] = 1;
    li[2] = 0;
    li[3] = 0;
    li[4] = 0;
    var q: i64 = 0;
    while q < 8 * n { if q >= dm.len { dm.push(0.0); } q += 1; }
    q = 0;
    while q < n { dm[q] = y[q]; dm[n + q] = f[q] * h_abs * dir; q += 1; }
    q = 0;
    while q < n * n { if q >= jm.len { jm.push(0.0); } q += 1; }
    q = 0;
    while q < n * n { if q >= lum.len { lum.push(0.0); } q += 1; }
    while piv.len < n { piv.push(0); }
    ist = [1, 0, 0];
    fst = [h_abs];
    let jf = bdf_jac(t, dm, n, nev, atol, jfac, jm) { tt, yy => F(tt, copy(yy)) };
    return jf;
}

// lsoda_hermite gives the state inside one step of the Adams mode at
// the fraction xs of the step. The first row of k holds the
// derivative at the start and the second row the derivative at the
// end. The state at the two ends goes in as y and ynew.
fn lsoda_hermite(n: i64, y: f64[:], ynew: f64[:], h: f64, k: f64[:], xs: f64) -> f64[>..] {
    let s = xs;
    let h00 = 2.0 * s * s * s - 3.0 * s * s + 1.0;
    let h01 = 0.0 - 2.0 * s * s * s + 3.0 * s * s;
    let h10 = s * s * s - 2.0 * s * s + s;
    let h11 = s * s * s - s * s;
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        out.push(h00 * y[i] + h01 * ynew[i] + h * (h10 * k[i] + h11 * k[n + i]));
        i += 1;
    }
    return out;
}

// solve_ivp_lsoda integrates the system y' = f(t, y) from t0 to t1
// with the LSODA method. The steps differ from the steps of SciPy, so
// the values agree to the tolerance only.
fn solve_ivp_lsoda<F: fn(f64, f64[:]) -> f64[:]>(t0: f64, t1: f64, y0: f64[:], rtol: f64 = 0.001, atol: f64 = 1.0e-6, max_step: f64 = 1.0e308, t_eval: f64[:] = [], first_step: f64 = 0.0) -> Arr, Arr, Fault {
    let t, y, f = ivp_run(t0, t1, y0, rtol, atol, "LSODA", max_step, t_eval, 0, [], [], false, first_step, false) { tt, yy => F(tt, copy(yy)) };
    return t, y, f;
}
