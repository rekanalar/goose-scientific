// The method of Alefeld, Potra and Shi for a root of a function in a
// bracket. The method has the order eight. It uses the inverse
// interpolation of the function and the steps of Newton on a
// quadratic.
namespace scientific;

import mathshim;

// The count of steps of Newton type that the method takes. SciPy holds
// the same limits.
const TOMS748_K_MIN: i64 = 1;
const TOMS748_K_MAX: i64 = 100;

// ts_close says whether two numbers lie near each other. The test is
// the test of NumPy.
fn ts_close(x: f64, y: f64, rtol: f64, atol: f64) -> bool {
    let d = mathshim::abs_of(x - y);
    return d <= atol + rtol * mathshim::abs_of(y);
}

// ts_notclose says whether the values lie far from each other. A value
// of zero or a value that is not finite makes the answer false.
fn ts_notclose(fs: f64[:], rtol: f64, atol: f64) -> bool {
    var i: i64 = 0;
    while i < fs.len {
        if fs[i] == 0.0 { return false; }
        if fs[i] != fs[i] { return false; }
        if mathshim::is_infinite_of(fs[i]) != 0.0 { return false; }
        i += 1;
    }
    i = 0;
    while i < fs.len - 1 {
        var j: i64 = i + 1;
        while j < fs.len {
            if ts_close(fs[i], fs[j], rtol, atol) { return false; }
            j += 1;
        }
        i += 1;
    }
    return true;
}

// ts_secant gives the step of the secant. Two values that hold the
// same number give the missing value.
fn ts_secant(x0: f64, x1: f64, f0: f64, f1: f64) -> f64 {
    if f0 == f1 { return 0.0 / 0.0; }
    var x2: f64 = 0.0;
    if mathshim::abs_of(f1) > mathshim::abs_of(f0) {
        x2 = (-f0 / f1 * x1 + x0) / (1.0 - f0 / f1);
    } else {
        x2 = (-f1 / f0 * x0 + x1) / (1.0 - f1 / f0);
    }
    return x2;
}

// ts_dd gives the divided differences of the points. The result holds
// the value of the first point, the difference of the two first points
// and so on. The forward order takes the difference that starts at the
// first point.
fn ts_dd(xv: f64[:], fv: f64[:], forward: bool) -> f64[>..] {
    var dd: f64[>..] = copy_slice(fv);
    var row: f64[>..] = copy_slice(fv);
    if forward {
        dd[0] = fv[0];
    } else {
        dd[0] = fv[fv.len - 1];
    }
    var i: i64 = 1;
    while i < xv.len {
        var nrow: f64[>..] = [];
        var j: i64 = 0;
        while j + 1 < row.len {
            let denom = xv[i + j] - xv[j];
            nrow.push((row[j + 1] - row[j]) / denom);
            j += 1;
        }
        row = copy_slice(nrow);
        if forward {
            dd[i] = row[0];
        } else {
            dd[i] = row[row.len - 1];
        }
        i += 1;
    }
    return dd;
}

// ts_interp gives the value of the polynomial that passes through the
// points. The method of Neville.
fn ts_interp(xv: f64[:], fv: f64[:], x: f64) -> f64 {
    let n = xv.len;
    var q: f64[>..] = [];
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n { q.push(0.0); d.push(0.0); i += 1; }
    i = 0;
    while i < n {
        q[i * n] = fv[i];
        d[i * n] = fv[i];
        i += 1;
    }
    var k: i64 = 1;
    while k < n {
        var r: i64 = k;
        while r < n {
            let alpha = d[r * n + (k - 1)] - q[(r - 1) * n + (k - 1)];
            let diffik = xv[r - k] - xv[r];
            q[r * n + k] = (xv[r] - x) / diffik * alpha;
            d[r * n + k] = (xv[r - k] - x) / diffik * alpha;
            r += 1;
        }
        k += 1;
    }
    var sum: f64 = 0.0;
    var j: i64 = 1;
    while j < n { sum += q[(n - 1) * n + j]; j += 1; }
    return sum + q[(n - 1) * n];
}

// ts_inverse_zero gives the point where the polynomial through the four
// points holds zero. The points are the values of the function and
// their places.
fn ts_inverse_zero(a: f64, b: f64, c: f64, d: f64, fa: f64, fb: f64, fc: f64, fd: f64) -> f64 {
    var xv: f64[>..] = [fa, fb, fc, fd];
    var fv: f64[>..] = [a, b, c, d];
    return ts_interp(xv, fv, 0.0);
}

// ts_newton_quadratic gives the point that the steps of Newton on a
// quadratic reach. The quadratic passes through the two ends of the
// bracket and the point d. The count of steps is the count of Newton
// steps.
fn ts_newton_quadratic(a: f64, b: f64, fa: f64, fb: f64, d: f64, fd: f64, steps: i64) -> f64 {
    var xv: f64[>..] = [a, b, d];
    var fv: f64[>..] = [fa, fb, fd];
    let dd = ts_dd(xv, fv, true);
    let big = dd[2];
    let mid = dd[1];
    // The quadratic is fa plus mid times the distance to a, plus big
    // times the product of the distances to a and to b.
    if big == 0.0 {
        return a - fa / mid;
    }
    var r = a;
    if big * fa <= 0.0 { r = b; }
    var i: i64 = 0;
    while i < steps {
        let pv = (big * (r - b) + mid) * (r - a) + fa;
        let r1 = r - pv / (mid + big * (2.0 * r - a - b));
        if !(a < r1 && r1 < b) {
            if (a < r && r < b) { return r; }
            return (a + b) / 2.0;
        }
        r = r1;
        i += 1;
    }
    return r;
}

// toms748 finds a root of a function in a bracket. It takes the
// function as a block. The method is the algorithm 748 of Alefeld,
// Potra and Shi. The count k holds the number of steps of Newton type
// that the method takes. The tolerance holds the absolute part and the
// relative part of the test of the end.
fn toms748<F: fn(f64) -> f64>(a: f64, b: f64, xtol: f64 = 2.0e-12, rtol: f64 = 8.881784197001252e-16, turns: i64 = 100, k: i64 = 2) -> f64, Fault {
    if xtol <= 0.0 {
        return 0.0, fault(F_VALUE, "toms748", "the tolerance holds a value of zero or below");
    }
    if rtol < 0.0 {
        return 0.0, fault(F_VALUE, "toms748", "the relative tolerance holds a value below zero");
    }
    if turns < 1 {
        return 0.0, fault(F_VALUE, "toms748", "the count of turns holds a value below one");
    }
    var steps = k;
    if steps < TOMS748_K_MIN { steps = TOMS748_K_MIN; }
    if steps > TOMS748_K_MAX { steps = TOMS748_K_MAX; }
    var ab0 = a;
    var ab1 = b;
    var fa = F(copy(ab0));
    if fa != fa || mathshim::is_infinite_of(fa) != 0.0 {
        return 0.0, fault(F_VALUE, "toms748", "the value of the function is not a finite number");
    }
    if fa == 0.0 { return ab0, no_fault(); }
    var fb = F(copy(ab1));
    if fb != fb || mathshim::is_infinite_of(fb) != 0.0 {
        return 0.0, fault(F_VALUE, "toms748", "the value of the function is not a finite number");
    }
    if fb == 0.0 { return ab1, no_fault(); }
    if fa * fb > 0.0 {
        return 0.0, fault(F_VALUE, "toms748", "the two values of the bracket hold the same sign");
    }
    // The points d and e hold the values outside the bracket.
    var d: f64 = 0.0;
    var fd: f64 = 0.0;
    var e: f64 = 0.0;
    var fe: f64 = 0.0;
    var has_e: bool = false;
    // The first step holds only the two ends of the bracket.
    var c = ts_secant(ab0, ab1, fa, fb);
    if !(ab0 < c && c < ab1) { c = (ab0 + ab1) / 2.0; }
    var fc = F(copy(c));
    if fc == 0.0 { return c, no_fault(); }
    // The bracket moves to the point c. The point that leaves is the
    // point d.
    var keep0: bool = fa * fc > 0.0;
    if keep0 {
        d = ab0;
        fd = fa;
        ab0 = c;
        fa = fc;
    } else {
        d = ab1;
        fd = fb;
        ab1 = c;
        fb = fc;
    }
    var turn: i64 = 1;
    var result: f64 = 0.0;
    var out_fault = no_fault();
    while true {
        turn += 1;
        let eps: f64 = 2.220446049250313e-16;
        let width = ab1 - ab0;
        var found: bool = false;
        var cc: f64 = 0.0;
        var nsteps: i64 = 2;
        while nsteps <= steps + 1 {
            // The interpolation of the function gives the point when the
            // values lie far from each other.
            if has_e {
                var fs: f64[>..] = [fa, fb, fd, fe];
                if ts_notclose(fs, 0.0, 32.0 * eps) {
                    let c0 = ts_inverse_zero(ab0, ab1, d, e, fa, fb, fd, fe);
                    if ab0 < c0 && c0 < ab1 {
                        cc = c0;
                        found = true;
                    }
                }
            }
            if !found {
                cc = ts_newton_quadratic(ab0, ab1, fa, fb, d, fd, nsteps);
                found = true;
            }
            let fcc = F(copy(cc));
            if fcc == 0.0 { return cc, no_fault(); }
            // The bracket moves to the new point.
            e = d;
            fe = fd;
            has_e = true;
            if fa * fcc > 0.0 {
                d = ab0;
                fd = fa;
                ab0 = cc;
                fa = fcc;
            } else {
                d = ab1;
                fd = fb;
                ab1 = cc;
                fb = fcc;
            }
            nsteps += 1;
        }
        // The end with the smaller value leads the step.
        var uix: i64 = 0;
        if mathshim::abs_of(fb) < mathshim::abs_of(fa) { uix = 1; }
        var u = ab0;
        var fu = fa;
        if uix == 1 {
            u = ab1;
            fu = fb;
        }
        var xv: f64[>..] = [ab0, ab1];
        var fv: f64[>..] = [fa, fb];
        let dd = ts_dd(xv, fv, uix == 0);
        cc = u - 2.0 * fu / dd[1];
        if mathshim::abs_of(cc - u) > 0.5 * (ab1 - ab0) {
            cc = (ab0 + ab1) / 2.0;
        } else if ts_close(cc, u, eps, 0.0) {
            // The step did not change. The values of the function may
            // differ much, or the root lies near the end.
            var frs: f64[>..] = [mathshim::frexp_exp_of(fa), mathshim::frexp_exp_of(fb)];
            if frs[uix] < frs[1 - uix] - 50.0 {
                cc = (31.0 * u + (ab0 + ab1 - u)) / 32.0;
            } else {
                var mm: f64 = 1.0;
                if uix == 1 { mm = -1.0; }
                let adj = mm * mathshim::abs_of(cc) * rtol + mm * xtol;
                cc = u + adj;
            }
            if !(ab0 < cc && cc < ab1) { cc = (ab0 + ab1) / 2.0; }
        }
        let fcc = F(copy(cc));
        if fcc == 0.0 { return cc, no_fault(); }
        e = d;
        fe = fd;
        has_e = true;
        if fa * fcc > 0.0 {
            d = ab0;
            fd = fa;
            ab0 = cc;
            fa = fcc;
        } else {
            d = ab1;
            fd = fb;
            ab1 = cc;
            fb = fcc;
        }
        // The bracket must become much smaller. Else the middle is
        // taken.
        if (ab1 - ab0) > 0.5 * width {
            e = d;
            fe = fd;
            let z = (ab0 + ab1) / 2.0;
            let fz = F(copy(z));
            if fz == 0.0 { return z, no_fault(); }
            if fa * fz > 0.0 {
                d = ab0;
                fd = fa;
                ab0 = z;
                fa = fz;
            } else {
                d = ab1;
                fd = fb;
                ab1 = z;
                fb = fz;
            }
        }
        // The end of the method.
        if ts_close(ab0, ab1, rtol, xtol) {
            result = (ab0 + ab1) / 2.0;
            break;
        }
        if turn >= turns {
            out_fault = fault(F_LIMIT, "toms748", "the method did not reach the tolerance in the count of turns");
            break;
        }
    }
    return result, out_fault;
}
