// The method of Powell with constraints of linear models.
//
// The method holds a trust region of the radius rho. At each turn it
// builds a linear model of the function and of the constraints from a
// forward difference. The step is the projection of the step of the
// steepest descent onto the linear forms of the constraints. The
// quadratic program of the projection is the program of the active
// set. When the step improves the merit, the point moves. Otherwise
// the radius shrinks by the rule of Powell. The method ends when the
// radius reaches its end value.
//
// The merit holds the value of the function and a penalty times the
// largest violation of the constraints. A constraint must be zero or
// above. The radius bounds the accuracy: the point holds no better
// accuracy than the end radius.
//
// The block of the caller gives the value of the function first and
// the values of the constraints after it.
namespace scientific;

import mathshim;

// cb_redrho gives the next radius of the trust region. The rule keeps
// the steps of the shrink in proportion to the end radius.
fn cb_redrho(rho_in: f64, rhoend: f64) -> f64 {
    let q = rho_in / rhoend;
    if q > 250.0 { return 0.1 * rho_in; }
    if q <= 16.0 { return rhoend; }
    return mathshim::root(q) * rhoend;
}

// cb_viol gives the largest violation of the constraints. A
// constraint must be zero or above, so the violation is the part of
// the smallest value below zero.
fn cb_viol(c: f64[:]) -> f64 {
    if c.len < 1 { return 0.0; }
    var lo: f64 = c[0];
    var i: i64 = 1;
    while i < c.len {
        if c[i] < lo { lo = c[i]; }
        i += 1;
    }
    if lo < 0.0 { return -lo; }
    return 0.0;
}

// fmin_cobyla finds the point where a function of several variables is
// at its smallest under constraints. The block of the caller takes a
// vector. It gives the value of the function first and the values of
// the constraints after it. Each constraint must be zero or above at
// the minimum. The method of Powell builds a linear model of the
// function and of the constraints at each turn, and it holds the step
// inside a trust region. The radius of the region starts at rhobeg and
// ends at rhoend. The accuracy of the point never goes below rhoend.
// The call gives the point of the minimum and a fault. The fault of
// the limit means that the method used its count of calls.
fn fmin_cobyla<F: fn(f64[:]) -> f64[:]>(x0: f64[:], rhobeg: f64 = 1.0, rhoend: f64 = 0.0001, maxfun: i64 = 1000, catol: f64 = 0.0002) -> Arr, Fault {
    let n = x0.len;
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "fmin_cobyla", "the start holds no element");
    }
    if !all_finite(x0) {
        return empty_arr(), fault(F_VALUE, "fmin_cobyla", "the start holds a value that is not a number");
    }
    if !(rhobeg > 0.0) || !(rhoend > 0.0) || rhobeg < rhoend {
        return empty_arr(), fault(F_VALUE, "fmin_cobyla", "the radii must hold values above zero and rhobeg at rhoend or above");
    }
    if !(catol >= 0.0) || maxfun < 1 {
        return empty_arr(), fault(F_VALUE, "fmin_cobyla", "the tolerance of the constraints or the count of calls is not valid");
    }
    // The first call gives the count of the constraints.
    let r = F(copy_slice(x0));
    if r.len < 1 {
        return empty_arr(), fault(F_VALUE, "fmin_cobyla", "the function holds no value");
    }
    let m = r.len - 1;
    var x: f64[>..] = copy_slice(x0);
    var f = r[0];
    var c: f64[>..] = [];
    var i: i64 = 1;
    while i < r.len { c.push(r[i]); i += 1; }
    var nf: i64 = 1;
    var cstrv = cb_viol(c);
    // The best point that holds the constraints, and the best point
    // without that condition.
    var bf: f64 = 1.0e308;
    var bfx: f64[>..] = copy_slice(x);
    var bcv = cstrv;
    var bcx: f64[>..] = copy_slice(x);
    var delta = rhobeg;
    let cpen: f64 = 1000.0;
    var done = false;
    while nf < maxfun && !done {
        // The linear model from a forward difference. The matrix gc
        // holds one row per constraint and one column per variable.
        var gf: f64[>..] = [];
        var gc: f64[>..] = [];
        i = 0;
        while i < m * n { gc.push(0.0); i += 1; }
        var k: i64 = 0;
        while k < n {
            let h = 0.0000000149 * (1.0 + mathshim::abs_of(x[k]));
            var probe: f64[>..] = copy_slice(x);
            probe[k] += h;
            let rp = F(copy(probe));
            if rp.len < m + 1 {
                return empty_arr(), fault(F_VALUE, "fmin_cobyla", "the function holds too few values");
            }
            nf += 1;
            gf.push((rp[0] - f) / h);
            var j: i64 = 0;
            while j < m {
                gc[j * n + k] = (rp[j + 1] - c[j]) / h;
                j += 1;
            }
            k += 1;
        }
        // The step. When the constraints hold at the point, the step
        // follows the model of the function. Otherwise the step only
        // repairs the constraints.
        var zg: f64[>..] = [];
        i = 0;
        while i < n { zg.push(0.0); i += 1; }
        var feasible = true;
        i = 0;
        while i < m {
            if c[i] < 0.0 { feasible = false; }
            i += 1;
        }
        var gv: f64[>..] = [];
        if feasible {
            gv = copy_slice(gf);
        } else {
            gv = copy_slice(zg);
        }
        // The right side of the constraints of the step is minus the
        // value of the constraints at the point.
        var rhs: f64[>..] = [];
        i = 0;
        while i < m { rhs.push(-c[i]); i += 1; }
        var no_rows: f64[>..] = [];
        let step, lam, act, ok = sq_qp(cb_identity(n), gv, no_rows, no_rows, gc, rhs, n, 0, m);
        var d: f64[>..] = [];
        if ok {
            d = copy_slice(step.data);
        } else {
            i = 0;
            while i < n { d.push(0.0); i += 1; }
        }
        // The step stays inside the trust region.
        let dn = mathshim::root(mg_dot(d, d));
        if dn > delta {
            i = 0;
            while i < n { d[i] = d[i] * (delta / dn); i += 1; }
        }
        var xt: f64[>..] = [];
        i = 0;
        while i < n { xt.push(x[i] + d[i]); i += 1; }
        let rt = F(copy(xt));
        if rt.len < m + 1 {
            return empty_arr(), fault(F_VALUE, "fmin_cobyla", "the function holds too few values");
        }
        nf += 1;
        let f2 = rt[0];
        var c2: f64[>..] = [];
        i = 1;
        while i < rt.len { c2.push(rt[i]); i += 1; }
        let cstrv2 = cb_viol(c2);
        if cstrv2 <= catol && f2 < bf {
            bf = f2;
            bfx = copy_slice(xt);
        }
        if cstrv2 < bcv {
            bcv = cstrv2;
            bcx = copy_slice(xt);
        }
        let actrem = (f + cpen * cstrv) - (f2 + cpen * cstrv2);
        let shortd = mathshim::root(mg_dot(d, d)) <= 0.000000001 * delta;
        if actrem > 0.0 && !shortd {
            // The merit falls. The point moves.
            x = copy_slice(xt);
            f = f2;
            c = copy_slice(c2);
            cstrv = cstrv2;
        } else {
            // The step fails. The radius of the trust region shrinks.
            delta = cb_redrho(delta, rhoend);
            if delta <= rhoend { done = true; }
        }
    }
    if !done {
        return empty_arr(), fault(F_LIMIT, "fmin_cobyla", "the method used its count of calls before the radius reached its end");
    }
    // No point holds the constraints inside the tolerance. The point
    // with the smallest violation is the answer.
    if bf > 1.0e307 {
        return line_arr(copy(bcx)), no_fault();
    }
    return line_arr(copy(bfx)), no_fault();
}

// cb_identity gives the identity matrix of a size. The program of the
// step uses it as its quadratic term.
fn cb_identity(n: i64) -> f64[>..] {
    return sq_identity(n);
}
