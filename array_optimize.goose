// The root finding.
//
// A solver takes the function of the caller as a parameter of a
// generic type. The caller passes the function as a block at the end
// of the call. The methods are iterative, because a recursive
// function of the package cannot hold a growable value across the
// recursion.
namespace scientific;

import mathshim;

// brentq finds a root of a function between two points. The method is
// the method of Brent. The method takes the step of the straight
// line, then the step of the quadratic form, and it falls back to
// the middle of the bracket when a step falls outside. The two points
// must hold a change of sign, and the root must lie between them. A
// pair of points that holds no change of sign is a fault.
fn brentq<F: fn(f64) -> f64>(a0: f64, b0: f64, xtol: f64, turns: i64) -> f64, Fault {
    if xtol <= 0.0 {
        return 0.0, fault(F_VALUE, "brentq", "the tolerance holds a value of zero or below");
    }
    if turns < 1 {
        return 0.0, fault(F_VALUE, "brentq", "the count of turns holds a value below one");
    }
    // The point of the work, the point before it, and the point of
    // the other side of the root.
    var xpre = a0;
    var xcur = b0;
    var fpre = F(xpre);
    var fcur = F(xcur);
    if fpre == 0.0 { return xpre, no_fault(); }
    if fcur == 0.0 { return xcur, no_fault(); }
    if (fpre > 0.0 && fcur > 0.0) || (fpre < 0.0 && fcur < 0.0) {
        return 0.0, fault(F_VALUE, "brentq", "the points hold no change of sign");
    }
    // The step holds the distance between the two first points.
    var spre = 0.0;
    var scur = 0.0;
    var xblk = xpre;
    var fblk = fpre;
    var turn: i64 = 0;
    while turn < turns {
        // The bracket holds the root between the two points.
        if (fpre > 0.0 && fcur < 0.0) || (fpre < 0.0 && fcur > 0.0) {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        // The point of the smaller value takes the place of the
        // point of the work. The point before it and the point of the
        // other side take the place of the point of the work too.
        if mathshim::abs_of(fblk) < mathshim::abs_of(fcur) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        // The tolerance of the turn holds the size of the machine.
        let delta = (xtol + 2.220446049250313e-16 * mathshim::abs_of(xcur)) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || mathshim::abs_of(sbis) < delta {
            return xcur, no_fault();
        }
        // The step of the straight line, or the step of the quadratic
        // form. The step must stay inside the bracket, and it must be
        // shorter than the last step.
        if mathshim::abs_of(spre) > delta && mathshim::abs_of(fcur) < mathshim::abs_of(fpre) {
            var stry: f64 = 0.0;
            if xpre == xblk {
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            let limit = mathshim::min_of(mathshim::abs_of(spre), 3.0 * mathshim::abs_of(sbis) - delta);
            if 2.0 * mathshim::abs_of(stry) < limit {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if mathshim::abs_of(scur) > delta {
            xcur = xcur + scur;
        } else {
            if scur > 0.0 { xcur = xcur + delta; } else { xcur = xcur - delta; }
        }
        fcur = F(xcur);
        turn += 1;
    }
    return 0.0, fault(F_LIMIT, "brentq", "the method did not reach the tolerance in the count of turns");
}

// brenth finds a root of a function between two points. The method is
// the method of Brent with the step of the false position. The step
// of the false position holds the root between the two points, so
// the method cannot lose the root.
fn brenth<F: fn(f64) -> f64>(a0: f64, b0: f64, xtol: f64, turns: i64) -> f64, Fault {
    if xtol <= 0.0 {
        return 0.0, fault(F_VALUE, "brenth", "the tolerance holds a value of zero or below");
    }
    if turns < 1 {
        return 0.0, fault(F_VALUE, "brenth", "the count of turns holds a value below one");
    }
    var a = a0;
    var b = b0;
    var fa = F(a);
    var fb = F(b);
    if fa == 0.0 { return a, no_fault(); }
    if fb == 0.0 { return b, no_fault(); }
    if (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0) {
        return 0.0, fault(F_VALUE, "brenth", "the points hold no change of sign");
    }
    var c = a;
    var fc = fa;
    var d = b - a;
    var e = d;
    // The step of the false position takes the weight of the point
    // that did not move. The factor halves at every turn that takes
    // the same side.
    var turn: i64 = 0;
    while turn < turns {
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if mathshim::abs_of(fc) < mathshim::abs_of(fb) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol1 = 2.0 * 2.220446049250313e-16 * mathshim::abs_of(b) + 0.5 * xtol;
        let xm = 0.5 * (c - b);
        if mathshim::abs_of(xm) <= tol1 || fb == 0.0 {
            return b, no_fault();
        }
        // The step comes from the interpolation of the three points.
        if mathshim::abs_of(e) >= tol1 && mathshim::abs_of(fa) > mathshim::abs_of(fb) {
            let s = fb / fa;
            var p: f64 = 0.0;
            var q: f64 = 0.0;
            if a == c {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                let q1 = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * xm * q1 * (q1 - r) - (b - a) * (r - 1.0));
                q = (q1 - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 { q = -q; }
            let min1 = 3.0 * xm * q - mathshim::abs_of(tol1 * q);
            let min2 = mathshim::abs_of(e * q);
            if 2.0 * p < mathshim::min_of(min1, min2) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        // The weight of the point that did not move halves when the
        // step falls on the same side.
        if (fa < 0.0 && fb < 0.0) || (fa > 0.0 && fb > 0.0) {
            fc = fc / 2.0;
        }
        a = b;
        fa = fb;
        if mathshim::abs_of(d) > tol1 {
            b = b + d;
        } else {
            if d > 0.0 { b = b + tol1; } else { b = b - tol1; }
        }
        fb = F(b);
        turn += 1;
    }
    return 0.0, fault(F_LIMIT, "brenth", "the method did not reach the tolerance in the count of turns");
}

// newton finds a root of a function from a start point. It takes the
// derivative of the function as a named function of the caller, and
// the function as a block. A derivative of zero is a fault. A call
// that does not reach the tolerance in the count of turns is a
// fault.
fn newton<D: fn(f64) -> f64, F: fn(f64) -> f64>(x0: f64, xtol: f64, turns: i64) -> f64, Fault {
    if xtol <= 0.0 {
        return 0.0, fault(F_VALUE, "newton", "the tolerance holds a value of zero or below");
    }
    if turns < 1 {
        return 0.0, fault(F_VALUE, "newton", "the count of turns holds a value below one");
    }
    var x = x0;
    var turn: i64 = 0;
    while turn < turns {
        let fx = F(copy(x));
        if mathshim::abs_of(fx) <= xtol {
            return x, no_fault();
        }
        let dfx = D(x);
        if dfx == 0.0 {
            return 0.0, fault(F_VALUE, "newton", "the derivative holds zero");
        }
        let step = fx / dfx;
        x = x - step;
        if mathshim::abs_of(step) <= xtol {
            return x, no_fault();
        }
        turn += 1;
    }
    return 0.0, fault(F_LIMIT, "newton", "the method did not reach the tolerance in the count of turns");
}

// fsolve finds the point where several equations hold zero at once.
// The function of the caller takes a vector and gives a vector. The
// method is the method of Newton with a damping. The step shrinks
// when the new point holds a larger size of the error than the old
// one.
fn fsolve<F: fn(f64[:]) -> f64[:]>(x0: f64[:], xtol: f64, turns: i64) -> Arr, Fault {
    if xtol <= 0.0 {
        return empty_arr(), fault(F_VALUE, "fsolve", "the tolerance holds a value of zero or below");
    }
    if turns < 1 {
        return empty_arr(), fault(F_VALUE, "fsolve", "the count of turns holds a value below one");
    }
    let n = x0.len;
    if n < 1 {
        return empty_arr(), fault(F_VALUE, "fsolve", "the start holds no element");
    }
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < n { x.push(x0[i]); i += 1; }
    var turn: i64 = 0;
    while turn < turns {
        // The values of the equations at the point.
        let fx = F(copy(x));
        if fx.len != n {
            return empty_arr(), fault(F_SHAPE, "fsolve", "the function holds a different count of values");
        }
        // The size of the error.
        var size: f64 = 0.0;
        i = 0;
        while i < n {
            let v = mathshim::abs_of(fx[i]);
            if v > size { size = v; }
            i += 1;
        }
        if size <= xtol {
            var s: i64[..8] = [];
            s.push(n);
            return Arr { ndim: 1, shape: s, data: x }, no_fault();
        }
        // The matrix of the derivatives. The method takes the
        // difference of the equations along each axis. The row of a
        // matrix holds an equation, and the column holds a variable.
        var jac: f64[>..] = [];
        var cell: i64 = 0;
        while cell < n * n { jac.push(0.0); cell += 1; }
        var d: i64 = 0;
        while d < n {
            // A small step along the axis.
            let h = 0.0000001 * (1.0 + mathshim::abs_of(x[d]));
            var probe: f64[>..] = [];
            i = 0;
            while i < n { probe.push(x[i]); i += 1; }
            probe[d] = probe[d] + h;
            let fp = F(copy(probe));
            if fp.len != n {
                return empty_arr(), fault(F_SHAPE, "fsolve", "the function holds a different count of values");
            }
            i = 0;
            while i < n {
                jac[i * n + d] = (fp[i] - fx[i]) / h;
                i += 1;
            }
            d += 1;
        }
        // The step comes from the solving of the system of the
        // derivatives. The right side holds the values of the
        // equations.
        var js: i64[..8] = [];
        js.push(n);
        js.push(n);
        var jac_arr = Arr { ndim: 2, shape: js, data: copy(jac) };
        var rhs: f64[>..] = [];
        i = 0;
        while i < n { rhs.push(fx[i]); i += 1; }
        let rs: i64[..8] = [];
        rs.push(n);
        let rhs_arr = Arr { ndim: 1, shape: rs, data: copy(rhs) };
        let step, fstep = solve(jac_arr, rhs_arr);
        if fstep.kind != F_NONE {
            return empty_arr(), fault(F_VALUE, "fsolve", "the matrix of the derivatives holds no pivot");
        }
        // The damping shrinks the step while the error grows.
        var factor: f64 = 1.0;
        var accepted = false;
        var tries: i64 = 0;
        while tries < 20 {
            var trial: f64[>..] = [];
            i = 0;
            while i < n { trial.push(x[i] - factor * step.data[i]); i += 1; }
            let ft = F(copy(trial));
            if ft.len != n {
                return empty_arr(), fault(F_SHAPE, "fsolve", "the function holds a different count of values");
            }
            var size2: f64 = 0.0;
            i = 0;
            while i < n {
                let v = mathshim::abs_of(ft[i]);
                if v > size2 { size2 = v; }
                i += 1;
            }
            if size2 <= size {
                x = copy(trial);
                accepted = true;
                break;
            }
            factor = factor / 2.0;
            tries += 1;
        }
        if !accepted {
            return empty_arr(), fault(F_VALUE, "fsolve", "the method did not lower the size of the error");
        }
        turn += 1;
    }
    return empty_arr(), fault(F_LIMIT, "fsolve", "the method did not reach the tolerance in the count of turns");
}
