// The structs of the results of the optimizers and of a root, and the
// struct of a constraint that is not linear.
//
// OptimizeResult replaces the class OptimizeResult of SciPy. It holds
// the point of the answer, the value, the flag of the success, the
// code of the end, the text of the end, and the counts of the calls,
// of the gradient and of the iterations.
//
// RootResults replaces the class RootResults of SciPy. It holds the
// point, the counts and the flag, and the name of the method that ran.
//
// NlConstraint replaces the class NonlinearConstraint of SciPy. It
// holds the two bounds, the kind and the flag of the kept feasibility.
namespace scientific;

// OptimizeResult holds the answer of an optimizer. It replaces the
// class OptimizeResult of SciPy. It holds the point of the answer, the
// value, the flag of the success, the code of the end, the text of the
// end, and the counts of the calls, of the gradient and of the
// iterations. The point sits last, because the language holds one
// growable field in a struct.
struct OptimizeResult {
    fun: f64,
    success: bool,
    status: i64,
    message: const u8[:],
    nfev: i64,
    njev: i64,
    nit: i64,
    x: Arr,
}

// RootResults holds the answer of a root. It replaces the class
// RootResults of SciPy. It holds the point, the counts and the flag,
// and the name of the method that ran.
struct RootResults {
    iterations: i64,
    function_calls: i64,
    converged: bool,
    flag: const u8[:],
    root: Arr,
}

// NlConstraint holds a condition that is not linear. It replaces the
// class NonlinearConstraint of SciPy. It holds the two bounds, the kind
// and the flag of the kept feasibility.
struct NlConstraint {
    lb: f64,
    ub: f64,
    kind: const u8[:],
    keep_feasible: bool,
}

// optimize_result builds the struct of the answer of an optimizer from
// its values. The point is copied, so the caller keeps the struct
// after the buffers of the call move on.
fn optimize_result(x: f64[:], fun: f64, success: bool, status: i64, message: const u8[:], nfev: i64, njev: i64, nit: i64) -> OptimizeResult {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < x.len { d.push(x[i]); i += 1; }
    return OptimizeResult { fun: fun, success: success, status: status, message: message, nfev: nfev, njev: njev, nit: nit, x: flt_arr(copy(d)) };
}

// root_result builds the struct of the answer of a root from its
// values. The point is copied.
fn root_result(root: f64[:], iterations: i64, function_calls: i64, converged: bool, flag: const u8[:]) -> RootResults {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < root.len { d.push(root[i]); i += 1; }
    return RootResults { iterations: iterations, function_calls: function_calls, converged: converged, flag: flag, root: flt_arr(copy(d)) };
}

// nl_constraint builds the struct of a constraint that is not linear.
// The kind holds eq, ineq or both. A kind of eq needs the two bounds
// equal. A kind that is no kind is a fault. Bounds that no point
// satisfies are a fault too.
fn nl_constraint(lb: f64, ub: f64, kind: const u8[:], keep_feasible: bool) -> NlConstraint, Fault {
    if kind != "eq" && kind != "ineq" && kind != "both" {
        return NlConstraint { lb: 0.0, ub: 0.0, kind: "", keep_feasible: false }, fault(F_VALUE, "nl_constraint", "the kind holds none of eq, ineq and both");
    }
    if !(lb <= ub) {
        return NlConstraint { lb: 0.0, ub: 0.0, kind: "", keep_feasible: false }, fault(F_VALUE, "nl_constraint", "no point satisfies the bounds");
    }
    if kind == "eq" && lb != ub {
        return NlConstraint { lb: 0.0, ub: 0.0, kind: "", keep_feasible: false }, fault(F_VALUE, "nl_constraint", "a constraint of the kind eq needs equal bounds");
    }
    return NlConstraint { lb: lb, ub: ub, kind: kind, keep_feasible: keep_feasible }, no_fault();
}
