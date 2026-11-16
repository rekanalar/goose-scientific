// The sequential least squares programming of SciPy.
//
// The method builds a model of the function and of the constraints at
// the current point. The model of the function holds a quadratic term.
// The estimate of the Hessian comes from the update of Broyden,
// Fletcher and Goldfarb. The step solves the model under the linear
// forms of the constraints. The rule of the active set finds that
// step. A line search on a merit function accepts the step.
//
// The block of the caller gives the value of the function first and
// the values of the constraints after it. The first neq constraints
// must equal zero. The other constraints must be zero or above. The
// gradients come from a forward difference.
namespace scientific;

import mathshim;

// sq_active_arr gives the numbers of the active rows as a list of
// numbers. The list holds one number per active row.
fn sq_active_arr(active: i64[:]) -> Arr {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < active.len { out.push(active[i]); i += 1; }
    return line_arr(copy(out));
}

// sq_gradjac gives the gradient of the function of the caller and
// the derivatives of the constraints at one point. One forward probe
// per variable fills both. A row of the matrix holds one constraint
// and a column holds one variable. A list of the function that holds
// no value gives a gradient of zero at that variable.
fn sq_gradjac<F: fn(f64[:]) -> f64[:]>(x: f64[:], f: f64, c: f64[:], n: i64) -> f64[>..], f64[>..] {
    var g: f64[>..] = [];
    var jac: f64[>..] = [];
    var j: i64 = 0;
    while j < n {
        let h = 0.0000000149 * (1.0 + mathshim::abs_of(x[j]));
        var probe: f64[>..] = copy_slice(x);
        probe[j] += h;
        let r = F(copy(probe));
        if r.len < 1 { g.push(0.0); } else { g.push((r[0] - f) / h); }
        var i: i64 = 0;
        while i < c.len {
            var v = c[i];
            if i + 1 < r.len { v = r[i + 1]; }
            jac.push((v - c[i]) / h);
            i += 1;
        }
        j += 1;
    }
    return g, jac;
}

// sq_viol gives the measure of the violation of the constraints. The
// equality constraints count by their absolute value. The other
// constraints count by the part below zero.
fn sq_viol(c: f64[:], neq: i64) -> f64 {
    var v: f64 = 0.0;
    var i: i64 = 0;
    while i < c.len {
        if i < neq { v += mathshim::abs_of(c[i]); }
        else if c[i] < 0.0 { v += -c[i]; }
        i += 1;
    }
    return v;
}

// sq_merit gives the value of the merit function of a point. It also
// gives the values that the function gave at the point. The merit
// holds the value of the function and the penalty of the violation.
// A function that holds no value gives the merit of a huge value.
fn sq_merit<F: fn(f64[:]) -> f64[:]>(x: f64[:], mu: f64, neq: i64) -> f64, f64[>..] {
    let r = F(copy_slice(x));
    if r.len < 1 { return 1.0e300, []; }
    var c: f64[>..] = [];
    var i: i64 = 1;
    while i < r.len { c.push(r[i]); i += 1; }
    return r[0] + mu * sq_viol(c, neq), r;
}

// fmin_slsqp finds the point where a function of several variables is
// at its smallest under equality constraints, inequality constraints
// and bounds. The block of the caller takes a vector. It gives the
// value of the function first and the values of the constraints after
// it. The first neq constraints must equal zero. The other constraints
// must be zero or above. The method is the method of the sequential
// least squares programming. The gradients come from a forward
// difference. The call gives the point of the minimum and a fault.
fn fmin_slsqp<F: fn(f64[:]) -> f64[:]>(x0: f64[:], neq: i64, lb: f64[:] = [], ub: f64[:] = [], acc: f64 = 0.000001, iters: i64 = 100) -> Arr, Fault {
    let n = x0.len;
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "fmin_slsqp", "the start holds no element");
    }
    if !all_finite(x0) {
        return empty_arr(), fault(F_VALUE, "fmin_slsqp", "the start holds a value that is not a number");
    }
    if !(acc > 0.0) || iters < 1 || neq < 0 {
        return empty_arr(), fault(F_VALUE, "fmin_slsqp", "the accuracy, the count of turns or neq is not valid");
    }
    var bounded = false;
    if lb.len > 0 || ub.len > 0 {
        bounded = true;
        // SciPy takes a pair of bounds for every variable, so the
        // package takes both lists or neither.
        if lb.len != n || ub.len != n {
            return empty_arr(), fault(F_SHAPE, "fmin_slsqp", "the bounds must hold one value per variable");
        }
        if !all_finite(lb) || !all_finite(ub) {
            return empty_arr(), fault(F_VALUE, "fmin_slsqp", "a bound holds a value that is not a number");
        }
        var i: i64 = 0;
        while i < n {
            if lb.len > 0 && ub.len > 0 && lb[i] > ub[i] {
                return empty_arr(), fault(F_VALUE, "fmin_slsqp", "a lower bound lies above the upper bound");
            }
            i += 1;
        }
    }
    // The first call gives the count of the constraints.
    let r = F(copy_slice(x0));
    if r.len < 1 {
        return empty_arr(), fault(F_VALUE, "fmin_slsqp", "the function holds no value");
    }
    let m = r.len - 1;
    if neq > m {
        return empty_arr(), fault(F_SHAPE, "fmin_slsqp", "neq stands above the count of constraints");
    }
    var x: f64[>..] = copy_slice(x0);
    var f = r[0];
    var c: f64[>..] = [];
    var i: i64 = 1;
    while i < r.len { c.push(r[i]); i += 1; }
    // The estimate of the Hessian starts as the identity.
    var bmat: f64[>..] = sq_identity(n);
    // One forward probe per variable gives the gradient of the
    // function and the derivatives of the constraints at the start.
    var g, cjac = sq_gradjac(x, f, c, n) { p => F(copy(p)) };
    // The radius of the trust region of the step.
    var delta: f64 = 1.0;
    var turn: i64 = 0;
    var done = false;
    while turn < iters && !done {
        turn += 1;
        // The rows of the equality constraints.
        var aeq: f64[>..] = [];
        var beq: f64[>..] = [];
        i = 0;
        while i < neq {
            var j: i64 = 0;
            while j < n { aeq.push(cjac[i * n + j]); j += 1; }
            beq.push(-c[i]);
            i += 1;
        }
        // The rows of the inequality constraints and of the bounds. The
        // list rgrad holds the gradient of each of these rows.
        var ain: f64[>..] = [];
        var bv: f64[>..] = [];
        var rgrad: f64[>..] = [];
        i = neq;
        while i < m {
            var j: i64 = 0;
            while j < n { ain.push(cjac[i * n + j]); j += 1; }
            j = 0;
            while j < n { rgrad.push(cjac[i * n + j]); j += 1; }
            bv.push(c[i]);
            i += 1;
        }
        var count = m - neq;
        if bounded {
            var j: i64 = 0;
            while j < n {
                if lb.len > 0 {
                    var e: f64[>..] = [];
                    var k: i64 = 0;
                    while k < n { if k == j { e.push(1.0); } else { e.push(0.0); } k += 1; }
                    var t: i64 = 0;
                    while t < n { ain.push(e[t]); t += 1; }
                    t = 0;
                    while t < n { rgrad.push(e[t]); t += 1; }
                    bv.push(lb[j] - x[j]);
                    count += 1;
                }
                if ub.len > 0 {
                    var e2: f64[>..] = [];
                    var k: i64 = 0;
                    while k < n { if k == j { e2.push(-1.0); } else { e2.push(0.0); } k += 1; }
                    var t: i64 = 0;
                    while t < n { ain.push(e2[t]); t += 1; }
                    t = 0;
                    while t < n { rgrad.push(e2[t]); t += 1; }
                    bv.push(x[j] - ub[j]);
                    count += 1;
                }
                j += 1;
            }
        }
        let mi = count;
        // The step of the program.
        let step, lam, act, ok = sq_qp(bmat, g, aeq, beq, ain, bv, n, neq, mi);
        if !ok {
            // The system of the program holds no pivot. The estimate of
            // the Hessian starts again.
            bmat = sq_identity(n);
            delta = 0.5 * delta;
            if delta < 0.000000000001 { break; }
            continue;
        }
        var d: f64[>..] = copy_slice(step.data);
        // The step stays inside the trust region.
        let dn = mathshim::root(mg_dot(d, d));
        if dn < 0.00000000000001 {
            // The program holds no move. The point is at its end.
            done = true;
            break;
        }
        if dn > delta {
            var j: i64 = 0;
            while j < n { d[j] = d[j] * (delta / dn); j += 1; }
        }
        // The penalty of the merit function.
        var mu: f64 = 1.0;
        i = 0;
        while i < lam.data.len {
            let v = mathshim::abs_of(lam.data[i]);
            if v > mu { mu = v; }
            i += 1;
        }
        let v0 = sq_viol(c, neq);
        let m0 = f + mu * v0;
        let pred = mg_dot(g, d) - mu * v0;
        // The line search. The step starts at the whole step and halves
        // until the merit falls enough.
        var t: f64 = 1.0;
        var taken = false;
        var rnew: f64[>..] = [];
        var tries: i64 = 0;
        while tries < 30 {
            var trial: f64[>..] = [];
            i = 0;
            while i < n { trial.push(x[i] + t * d[i]); i += 1; }
            let mv, rv = sq_merit(trial, mu, neq) { p => F(copy(p)) };
            if mv <= m0 + 0.1 * t * pred {
                taken = true;
                rnew = copy(rv);
                break;
            }
            t *= 0.5;
            tries += 1;
        }
        if !taken {
            // The step fails. The trust region shrinks.
            delta = 0.5 * delta;
            if delta < 0.000000000001 { break; }
            continue;
        }
        // The new point. The merit gave the values at this point.
        var s: f64[>..] = [];
        var xnew: f64[>..] = [];
        i = 0;
        while i < n {
            s.push(t * d[i]);
            xnew.push(x[i] + s[i]);
            i += 1;
        }
        if rnew.len != r.len {
            return empty_arr(), fault(F_VALUE, "fmin_slsqp", "the function holds a wrong count of values");
        }
        var cnew: f64[>..] = [];
        i = 1;
        while i < rnew.len { cnew.push(rnew[i]); i += 1; }
        let fnew = rnew[0];
        // The value of the function before the step. The test of the
        // end needs it.
        let fold = f;
        // The gradient of the Lagrangian at the old point and at the new
        // point. The update of the Hessian needs the difference.
        let gnew, cjacnew = sq_gradjac(xnew, fnew, cnew, n) { p => F(copy(p)) };
        var glold: f64[>..] = copy_slice(g);
        var glnew: f64[>..] = copy_slice(gnew);
        i = 0;
        while i < neq {
            var j: i64 = 0;
            while j < n {
                glold[j] += lam.data[i] * cjac[i * n + j];
                glnew[j] += lam.data[i] * cjacnew[i * n + j];
                j += 1;
            }
            i += 1;
        }
        // The gradient of the rows of the bounds holds the same values
        // at the new point.
        var rgradnew: f64[>..] = [];
        i = neq;
        while i < m {
            var j: i64 = 0;
            while j < n { rgradnew.push(cjacnew[i * n + j]); j += 1; }
            i += 1;
        }
        i = 0;
        while i < rgrad.len - (m - neq) * n {
            rgradnew.push(rgrad[(m - neq) * n + i]);
            i += 1;
        }
        // The multipliers of the active rows. Each multiplier goes with
        // the row that the active set names.
        var arow: i64 = 0;
        while arow < act.data.len {
            let lm = lam.data[neq + arow];
            let row = act.data[arow] as i64;
            if lm != 0.0 {
                var j: i64 = 0;
                while j < n {
                    glold[j] += lm * rgrad[row * n + j];
                    glnew[j] += lm * rgradnew[row * n + j];
                    j += 1;
                }
            }
            arow += 1;
        }
        // The change of the gradient of the Lagrangian.
        var y: f64[>..] = [];
        i = 0;
        while i < n { y.push(glnew[i] - glold[i]); i += 1; }
        let sy = mg_dot(s, y);
        if sy > 0.00000000001 * mathshim::root(mg_dot(s, s)) * mathshim::root(mg_dot(y, y)) {
            // The update of Broyden, Fletcher and Goldfarb.
            var bs: f64[>..] = [];
            i = 0;
            while i < n {
                var t2: f64 = 0.0;
                var j: i64 = 0;
                while j < n { t2 += bmat[i * n + j] * s[j]; j += 1; }
                bs.push(t2);
                i += 1;
            }
            let sbs = mg_dot(s, bs);
            var bn: f64[>..] = [];
            i = 0;
            while i < n {
                var j: i64 = 0;
                while j < n {
                    var v = bmat[i * n + j] - bs[i] * bs[j] / sbs + y[i] * y[j] / sy;
                    bn.push(v);
                    j += 1;
                }
                i += 1;
            }
            bmat = copy_slice(bn);
        }
        x = copy_slice(xnew);
        f = fnew;
        c = copy_slice(cnew);
        g = copy_slice(gnew);
        // The derivatives hold at the point of the next turn.
        cjac = copy_slice(cjacnew);
        delta = 2.0 * delta;
        if delta > 100.0 { delta = 100.0; }
        // The test of the end. The constraints must hold and the value
        // of the function must stop to change. A small step also ends
        // the method.
        var xmax2 = mathshim::abs_of(x[0]);
        i = 1;
        while i < n {
            let w = mathshim::abs_of(x[i]);
            if w > xmax2 { xmax2 = w; }
            i += 1;
        }
        let df = mathshim::abs_of(f - fold);
        if sq_viol(c, neq) < acc && df <= acc * (1.0 + mathshim::abs_of(fold)) {
            done = true;
        }
        if mathshim::root(mg_dot(s, s)) <= acc * (1.0 + xmax2) {
            done = true;
        }
    }
    if !done {
        return empty_arr(), fault(F_LIMIT, "fmin_slsqp", "the method did not reach the accuracy in the count of turns");
    }
    return line_arr(copy(x)), no_fault();
}
