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

// show_options gives the text of the options that the calls of the
// minimization of the package take. SciPy reads the options of a
// solver by the name of the method; the package holds one call per
// method, so the text holds the options of every call.
fn show_options() -> const u8[:] {
    return "minimize: xtol, turns\nminimize_grad: method of bfgs, cg, lbfgsb or newton-cg, bounded, lb, ub, xtol, ftol, gtol, turns\nminimize_constrained: neq, lb, ub, tol, turns\nleast_squares: xtol, turns, lower, upper\ncurve_fit: turns, sigma, lower, upper\ndifferential_evolution: seed, maxiter, popsize, tol, recombination\ndual_annealing: seed, maxiter, initial_temp\nbasinhopping: niter, temperature, stepsize, seed";
}
