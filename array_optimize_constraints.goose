// The helpers of the bounds and the constraints of the optimization.
//
// SciPy holds the bounds of a problem and the linear constraints in
// the types Bounds and LinearConstraint. The package holds no types of
// this kind, so the calls of this file check the values and give them
// back. The call show_options gives the text of the options that the
// calls of the minimization of the package take.
namespace scientific;

// bounds checks two lists of bounds and gives them back. Every lower
// bound must hold a value at or below the upper bound. A problem of
// at least one variable is needed.
fn bounds(lower: f64[:], upper: f64[:]) -> Arr, Arr, Fault {
    if lower.len < 1 || upper.len < 1 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "bounds", "a list of bounds holds no element");
    }
    if lower.len != upper.len {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "bounds", "the lists of bounds hold a different count");
    }
    var i: i64 = 0;
    while i < lower.len {
        if !all_finite([lower[i], upper[i]]) {
            return empty_arr(), empty_arr(), fault(F_VALUE, "bounds", "a bound is not finite");
        }
        if lower[i] > upper[i] {
            return empty_arr(), empty_arr(), fault(F_VALUE, "bounds", "a lower bound lies above the upper bound");
        }
        i += 1;
    }
    var lo: f64[>..] = [];
    var up: f64[>..] = [];
    i = 0;
    while i < lower.len { lo.push(lower[i]); up.push(upper[i]); i += 1; }
    return flt_arr(copy(lo)), flt_arr(copy(up)), no_fault();
}

// linear_constraint checks the matrix of a linear constraint with its
// bounds and gives the three back. The matrix holds one row per
// constraint, and the bounds hold one value per row. SciPy holds them
// in the type LinearConstraint; the package holds the plain values.
fn linear_constraint(a, lower: f64[:], upper: f64[:]) -> Arr, Arr, Arr, Fault {
    if a.ndim != 2 || a.shape.len != 2 || a.shape[0] < 1 || a.shape[1] < 1 {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "linear_constraint", "the matrix is not of two dimensions");
    }
    let m = a.shape[0];
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "linear_constraint");
    if fs.kind != F_NONE {
        return empty_arr(), empty_arr(), empty_arr(), fs;
    }
    if lower.len != m || upper.len != m {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_SHAPE, "linear_constraint", "the bounds do not hold one value per row");
    }
    var i: i64 = 0;
    while i < m {
        if !all_finite([lower[i], upper[i]]) {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "linear_constraint", "a bound is not finite");
        }
        if lower[i] > upper[i] {
            return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "linear_constraint", "a lower bound lies above the upper bound");
        }
        i += 1;
    }
    if !all_finite(a.data) {
        return empty_arr(), empty_arr(), empty_arr(), fault(F_VALUE, "linear_constraint", "a value of the matrix is not finite");
    }
    var am: f64[>..] = [];
    i = 0;
    while i < a.data.len { am.push(a.data[i]); i += 1; }
    var lo: f64[>..] = [];
    var up: f64[>..] = [];
    i = 0;
    while i < m { lo.push(lower[i]); up.push(upper[i]); i += 1; }
    return mat2(copy(am), m, a.shape[1]), flt_arr(copy(lo)), flt_arr(copy(up)), no_fault();
}

// sq_identity gives the identity matrix of a size.
fn sq_identity(n: i64) -> f64[>..] {
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n {
        if i % (n + 1) == 0 { out.push(1.0); } else { out.push(0.0); }
        i += 1;
    }
    return out;
}

// sq_qp solves a quadratic program. The matrix b holds the estimate
// of the Hessian by rows. The vector g holds the gradient. The matrix
// aeq holds the rows of the equality constraints. The vector beq holds
// their right sides. The matrix ain holds the rows of the inequality
// constraints. The vector bv holds their right sides. The program is
//
//   the least of 0.5 d b d + g d
//   under aeq d = beq and ain d >= bv.
//
// The rule of the active set finds the step. The call gives the step,
// the multipliers of the equality constraints and the multipliers of
// the inequality constraints. It also gives the numbers of the rows of
// the inequality constraints that hold a multiplier. A flag tells
// whether the program holds a step. A system without a pivot gives
// false.
fn sq_qp(b: f64[:], g: f64[:], aeq: f64[:], beq: f64[:], ain: f64[:], bv: f64[:], n: i64, meq: i64, mi: i64) -> Arr, Arr, Arr, bool {
    // The rows of the active set hold the numbers of the constraints.
    var active: i64[>..] = [];
    var d: f64[>..] = [];
    var lam: f64[>..] = [];
    var turn: i64 = 0;
    let limit = 2 * (meq + mi) + 10;
    while turn < limit {
        turn += 1;
        // The count of the rows of the system.
        var rows = meq;
        var k: i64 = 0;
        while k < active.len { rows += 1; k += 1; }
        let size = n + rows;
        // The matrix of the system. The upper left holds the Hessian.
        var m: f64[>..] = [];
        var i: i64 = 0;
        while i < size * size { m.push(0.0); i += 1; }
        i = 0;
        while i < n {
            var j: i64 = 0;
            while j < n { m[i * size + j] = b[i * n + j]; j += 1; }
            i += 1;
        }
        // The rows of the equality constraints. The upper right block
        // holds the transpose of the rows of the constraints.
        var r: i64 = 0;
        while r < meq {
            var i2: i64 = 0;
            while i2 < n {
                m[i2 * size + n + r] = aeq[r * n + i2];
                m[(n + r) * size + i2] = aeq[r * n + i2];
                i2 += 1;
            }
            r += 1;
        }
        // The rows of the active constraints.
        k = 0;
        while k < active.len {
            let row = active[k];
            var i2: i64 = 0;
            while i2 < n {
                m[i2 * size + n + r] = ain[row * n + i2];
                m[(n + r) * size + i2] = ain[row * n + i2];
                i2 += 1;
            }
            r += 1;
            k += 1;
        }
        // The right side of the system.
        var rhs: f64[>..] = [];
        i = 0;
        while i < n { rhs.push(-g[i]); i += 1; }
        i = 0;
        while i < meq { rhs.push(beq[i]); i += 1; }
        k = 0;
        while k < active.len { rhs.push(bv[active[k]]); k += 1; }
        let sol, fsol = solve(mat2(copy(m), size, size), line_arr(copy(rhs)));
        if fsol.kind != F_NONE { return empty_arr(), empty_arr(), empty_arr(), false; }
        // The step holds the first n values of the answer. The values
        // after them hold the multipliers.
        d = [];
        i = 0;
        while i < n { d.push(sol.data[i]); i += 1; }
        // The multipliers of the equality constraints and of the active
        // constraints.
        lam = [];
        i = 0;
        while i < meq { lam.push(sol.data[n + i]); i += 1; }
        k = 0;
        while k < active.len { lam.push(sol.data[n + meq + k]); k += 1; }
        // A constraint of the active set with a negative multiplier
        // leaves the set.
        var drop: i64 = -1;
        k = 0;
        while k < active.len {
            if lam[meq + k] < -0.000000001 { drop = k; break; }
            k += 1;
        }
        if drop >= 0 {
            // The rows of the set move one place to the front.
            var keep: i64[>..] = [];
            k = 0;
            while k < active.len {
                if k != drop { keep.push(active[k]); }
                k += 1;
            }
            active = copy(keep);
            continue;
        }
        // The constraint with the largest violation joins the set.
        var worst: i64 = -1;
        var amount: f64 = -0.000000001;
        var j: i64 = 0;
        while j < mi {
            var in_set = false;
            k = 0;
            while k < active.len {
                if active[k] == j { in_set = true; }
                k += 1;
            }
            if !in_set {
                var s: f64 = 0.0;
                var t: i64 = 0;
                while t < n { s += ain[j * n + t] * d[t]; t += 1; }
                let gap = s - bv[j];
                if gap < amount { amount = gap; worst = j; }
            }
            j += 1;
        }
        if worst >= 0 {
            active.push(worst);
            continue;
        }
        return line_arr(copy(d)), line_arr(copy(lam)), sq_active_arr(active), true;
    }
    return line_arr(copy(d)), line_arr(copy(lam)), sq_active_arr(active), true;
}

// show_options gives the text of the options that the calls of the
// minimization of the package take. SciPy reads the options of a
// solver by the name of the method; the package holds one call per
// method, so the text holds the options of every call.
fn show_options() -> const u8[:] {
    return "minimize: xtol, turns\nminimize_grad: method of bfgs, cg, lbfgsb or newton-cg, bounded, lb, ub, xtol, ftol, gtol, turns\nminimize_constrained: neq, lb, ub, tol, turns\nleast_squares: xtol, turns, lower, upper\ncurve_fit: turns, sigma, lower, upper\ndifferential_evolution: seed, maxiter, popsize, tol, recombination\ndual_annealing: seed, maxiter, initial_temp\nbasinhopping: niter, temperature, stepsize, seed";
}
