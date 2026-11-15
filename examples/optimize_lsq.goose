// The least squares, the assignment and the global search. Run it
// with: goose run examples/optimize_lsq.goose
import std;
import scientific from "..";

// poly_res gives the residuals of the fit of a polynomial of the
// second order.
fn poly_res(p: f64[:]) -> f64[>..] {
    var y: f64[>..] = [];
    y.push(1.0);
    y.push(3.1);
    y.push(5.4);
    y.push(8.2);
    y.push(11.1);
    y.push(14.5);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 6 {
        let t = i as f64;
        d.push(p[0] + p[1] * t + p[2] * t * t - y[i]);
        i += 1;
    }
    return d;
}

// poly_jac gives the Jacobian of the same fit.
fn poly_jac(p: f64[:]) -> f64[>..] {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 6 {
        let t = i as f64;
        d.push(1.0);
        d.push(t);
        d.push(t * t);
        i += 1;
    }
    return d;
}

// bowl gives the bowl of the global search.
fn bowl(p: f64[:]) -> f64 {
    return (p[0] - 1.5) * (p[0] - 1.5) + (p[1] + 0.5) * (p[1] + 0.5);
}

fn main() {
    // The fit of a polynomial with the method of Levenberg and
    // Marquardt.
    let r, cov, f = scientific::leastsq([0.0, 0.0, 0.0], poly_jac) { p => poly_res(p) };
    assert(f.kind == scientific::F_NONE);
    print("leastsq point = [", r.x.data[0], ", ", r.x.data[1], ", ", r.x.data[2], "] success = ", r.success, "\n");
    print("leastsq calls = ", r.nfev, " covariance[0][0] = ", cov.data[0], "\n");

    // The same fit with the estimated Jacobian.
    let r2, _, f2 = scientific::leastsq_est([0.0, 0.0, 0.0]) { p => poly_res(p) };
    assert(f2.kind == scientific::F_NONE);
    print("leastsq_est point = [", r2.x.data[0], ", ", r2.x.data[1], ", ", r2.x.data[2], "]\n");

    // The linear least squares with a bound on the second variable.
    let a = scientific::Arr { ndim: 2, shape: [3, 2], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
    let x, f3 = scientific::lsq_linear(a, [1.0, 2.0, 3.0], [], [1.0e308, 1.0]);
    assert(f3.kind == scientific::F_NONE);
    print("lsq_linear point = [", x.data[0], ", ", x.data[1], "]\n");

    // The assignment of two graphs of four nodes.
    let qa = scientific::Arr { ndim: 2, shape: [4, 4], data: [0.0, 80.0, 150.0, 170.0, 80.0, 0.0, 130.0, 100.0, 150.0, 130.0, 0.0, 120.0, 170.0, 100.0, 120.0, 0.0] };
    let qb = scientific::Arr { ndim: 2, shape: [4, 4], data: [0.0, 5.0, 2.0, 7.0, 0.0, 0.0, 3.0, 8.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0] };
    let p, f4 = scientific::quadratic_assignment(qa, qb);
    assert(f4.kind == scientific::F_NONE);
    print("the assignment of the rows = [", p.data[0], ", ", p.data[1], ", ", p.data[2], ", ", p.data[3], "]\n");

    // The global search in a box.
    let g, f5 = scientific::shgo([-5.0, -5.0], [5.0, 5.0], 100, 1) { v => bowl(v) };
    assert(f5.kind == scientific::F_NONE);
    print("shgo point = [", g.data[0], ", ", g.data[1], "]\n");

    // The constraint of a condition that is not linear.
    let c, f6 = scientific::nl_constraint(0.0, 1.0, "ineq", true);
    assert(f6.kind == scientific::F_NONE);
    print("the constraint holds the bounds [", c.lb, ", ", c.ub, "] of the kind ", c.kind, "\n");
}
