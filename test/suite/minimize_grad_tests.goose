// The tests of the group minimize_grad. Run them with SCIENTIFIC_SUITES=minimize_grad.
namespace minimize_grad_tests;

import scientific from "../..";
import os;
import common;

// test_minimize_grad pins the methods that use the gradient against
// SciPy. The blocks give the value and the gradient of the function of
// Rosenbrock and of a quadratic.
fn test_minimize_grad() {
    // The constrained minimization.
    let mx, mv, mf = scientific::minimize_constrained([2.0, 2.0], 1) { p => [p[0] * p[0] + p[1] * p[1], 2.0 * p[0], 2.0 * p[1], p[0] + p[1] - 1.0] };
    assert(mf.kind == scientific::F_NONE && common::rel_close(mx, [0.5, 0.5], 1.0e-6) && common::mathshim_abs(mv.data[0] - 0.5) < 1.0e-6);
    let my, myv, myf = scientific::minimize_constrained([2.0, 2.0], 0) { p => [(p[0] - 1.0) * (p[0] - 1.0) + (p[1] - 2.5) * (p[1] - 2.5), 2.0 * (p[0] - 1.0), 2.0 * (p[1] - 2.5), p[0] - 2.0 * p[1] + 2.0, -p[0] - 2.0 * p[1] + 6.0, -p[0] + 2.0 * p[1] + 2.0] };
    assert(myf.kind == scientific::F_NONE && common::rel_close(my, [1.4, 1.7], 1.0e-6) && common::mathshim_abs(myv.data[0] - 0.8) < 1.0e-6);
    let mz, mzv, mzf = scientific::minimize_constrained([1.0, 0.0], 1, [0.8, -5.0], [5.0, 5.0]) { p => [p[0] * p[0] + p[1] * p[1], 2.0 * p[0], 2.0 * p[1], p[0] + p[1] - 1.0] };
    assert(mzf.kind == scientific::F_NONE && common::rel_close(mz, [0.8, 0.2], 1.0e-6));
    let mb, mbv, mbf = scientific::minimize_constrained([1.0, 0.0], 3) { p => [p[0] * p[0], 2.0 * p[0], 0.0, p[1]] };
    assert(mbf.kind == scientific::F_SHAPE);
    let mc, mcv, mcf = scientific::minimize_constrained([1.0, 0.0], 0, [0.0], [1.0]) { p => [p[0] * p[0], 2.0 * p[0], 0.0] };
    assert(mcf.kind == scientific::F_SHAPE);
    // The method newton-cg.
    let nc, ncv, ncf = scientific::minimize_grad([-1.2, 1.0, 0.8], "newton-cg", false, [], [], 1.0e-8, 1.0e-8, 1.0e-8, 200) { v =>
        var s: f64 = 0.0;
        var g: f64[>..] = [0.0, 0.0, 0.0];
        var j: i64 = 0;
        while j + 1 < v.len {
            let d = v[j+1] - v[j]*v[j];
            s += 100.0 * d * d + (1.0 - v[j]) * (1.0 - v[j]);
            g[j] += -400.0 * v[j] * d - 2.0 * (1.0 - v[j]);
            g[j+1] += 200.0 * d;
            j += 1;
        }
        var o: f64[>..] = [s];
        j = 0;
        while j < g.len { o.push(g[j]); j += 1; }
        o
    };
    assert(ncf.kind == scientific::F_NONE && common::rel_close(nc, [1.0, 1.0, 1.0], 1.0e-6) && ncv.data[0] < 1.0e-10);
    let nd, ndv, ndf = scientific::minimize_grad([0.0, 1.0], "newton-cg", false, [], [], 1.0e-8, 1.0e-8, 1.0e-8, 200) { v =>
        let a = v[0] - 3.0;
        [a * a * a * a + (v[0] + v[1]) * (v[0] + v[1]) + v[1] * v[1] * v[1] * v[1] / 5.0, 4.0 * a * a * a + 2.0 * (v[0] + v[1]), 2.0 * (v[0] + v[1]) + 4.0 * v[1] * v[1] * v[1] / 5.0]
    };
    assert(ndf.kind == scientific::F_NONE && common::rel_close(nd, [2.2299597097408763, -1.3167503728453185], 1.0e-5));
    let ne, nev, nef = scientific::minimize_grad([0.0, 1.0], "newton-cg", true, [0.0, 0.0], [1.0, 1.0]) { v => [v[0] * v[0] + v[1] * v[1], 2.0 * v[0], 2.0 * v[1]] };
    assert(nef.kind == scientific::F_VALUE);
    let a, av, af = scientific::minimize_grad([-1.2, 1.0], "bfgs", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i + 1 < v.len { s += 100.0 * (v[i+1] - v[i]*v[i]) * (v[i+1] - v[i]*v[i]) + (1.0 - v[i]) * (1.0 - v[i]); i += 1; }
        var g: f64[>..] = [];
        var j: i64 = 0;
        while j < v.len { g.push(0.0); j += 1; }
        j = 0;
        while j + 1 < v.len {
            let d = v[j+1] - v[j]*v[j];
            g[j] += -400.0 * v[j] * d - 2.0 * (1.0 - v[j]);
            g[j+1] += 200.0 * d;
            j += 1;
        }
        var o: f64[>..] = [];
        o.push(s);
        j = 0;
        while j < g.len { o.push(g[j]); j += 1; }
        o
    };
    assert(af.kind == scientific::F_NONE && common::rel_close(a, [1.0, 1.0], 1.0e-6) && av.data[0] < 1.0e-12);
    let b, bv, bf = scientific::minimize_grad([-1.2, 1.0], "cg", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i + 1 < v.len { s += 100.0 * (v[i+1] - v[i]*v[i]) * (v[i+1] - v[i]*v[i]) + (1.0 - v[i]) * (1.0 - v[i]); i += 1; }
        var g: f64[>..] = [];
        var j: i64 = 0;
        while j < v.len { g.push(0.0); j += 1; }
        j = 0;
        while j + 1 < v.len {
            let d = v[j+1] - v[j]*v[j];
            g[j] += -400.0 * v[j] * d - 2.0 * (1.0 - v[j]);
            g[j+1] += 200.0 * d;
            j += 1;
        }
        var o: f64[>..] = [];
        o.push(s);
        j = 0;
        while j < g.len { o.push(g[j]); j += 1; }
        o
    };
    assert(bf.kind == scientific::F_NONE && common::rel_close(b, [1.0, 1.0], 1.0e-6) && bv.data[0] < 1.0e-12);
    let c, cv, cf = scientific::minimize_grad([-1.2, 1.0], "lbfgsb", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var s: f64 = 0.0;
        var i: i64 = 0;
        while i + 1 < v.len { s += 100.0 * (v[i+1] - v[i]*v[i]) * (v[i+1] - v[i]*v[i]) + (1.0 - v[i]) * (1.0 - v[i]); i += 1; }
        var g: f64[>..] = [];
        var j: i64 = 0;
        while j < v.len { g.push(0.0); j += 1; }
        j = 0;
        while j + 1 < v.len {
            let d = v[j+1] - v[j]*v[j];
            g[j] += -400.0 * v[j] * d - 2.0 * (1.0 - v[j]);
            g[j+1] += 200.0 * d;
            j += 1;
        }
        var o: f64[>..] = [];
        o.push(s);
        j = 0;
        while j < g.len { o.push(g[j]); j += 1; }
        o
    };
    assert(cf.kind == scientific::F_NONE && common::rel_close(c, [1.0, 1.0], 1.0e-6) && cv.data[0] < 1.0e-12);
    // The quadratic has the point [4, -2] and the value -6.
    let d, dv, df = scientific::minimize_grad([0.0, 0.0], "bfgs", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var o: f64[>..] = [];
        o.push((v[0] - 3.0) * (v[0] - 3.0) + 4.0 * (v[1] + 1.5) * (v[1] + 1.5) + v[0] * v[1]);
        o.push(2.0 * (v[0] - 3.0) + v[1]);
        o.push(8.0 * (v[1] + 1.5) + v[0]);
        o
    };
    assert(df.kind == scientific::F_NONE && common::rel_close(d, [4.0, -2.0], 1.0e-9) && common::rel_close(dv, [-6.0], 1.0e-9));
    // The bounds keep the point inside. SciPy replaces the method BFGS
    // with L-BFGS-B when bounds are given, and the package does the same.
    let e, ev, ef = scientific::minimize_grad([0.0, 0.0], "bfgs", true, [-1.0, -2.0], [2.0, 2.0], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var o: f64[>..] = [];
        o.push((v[0] - 3.0) * (v[0] - 3.0) + 4.0 * (v[1] + 1.5) * (v[1] + 1.5) + v[0] * v[1]);
        o.push(2.0 * (v[0] - 3.0) + v[1]);
        o.push(8.0 * (v[1] + 1.5) + v[0]);
        o
    };
    assert(ef.kind == scientific::F_NONE && common::rel_close(e, [2.0, -1.75], 1.0e-4) && common::rel_close(ev, [-2.25], 1.0e-6));
    // A start outside the bounds moves to the border.
    let g2, gv2, gf2 = scientific::minimize_grad([5.0, 0.0], "lbfgsb", true, [-1.0, -2.0], [2.0, 2.0], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v =>
        var o: f64[>..] = [];
        o.push((v[0] - 3.0) * (v[0] - 3.0) + 4.0 * (v[1] + 1.5) * (v[1] + 1.5) + v[0] * v[1]);
        o.push(2.0 * (v[0] - 3.0) + v[1]);
        o.push(8.0 * (v[1] + 1.5) + v[0]);
        o
    };
    assert(gf2.kind == scientific::F_NONE && common::rel_close(g2, [2.0, -1.75], 1.0e-4));
    // The faults.
    let h1, h1v, h1f = scientific::minimize_grad([0.0, 0.0], "nelder-mead", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h1f.kind != scientific::F_NONE);
    let h2, h2v, h2f = scientific::minimize_grad([], "bfgs", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h2f.kind != scientific::F_NONE);
    let h3, h3v, h3f = scientific::minimize_grad([0.0], "bfgs", false, [], [], 0.0, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h3f.kind != scientific::F_NONE);
    let h4, h4v, h4f = scientific::minimize_grad([0.0], "bfgs", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 0) { v => [v[0] * v[0]] };
    assert(h4f.kind != scientific::F_NONE);
    let h5, h5v, h5f = scientific::minimize_grad([0.0, 0.0], "bfgs", true, [0.0], [1.0], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h5f.kind != scientific::F_NONE);
    let h6, h6v, h6f = scientific::minimize_grad([0.0, 0.0], "bfgs", true, [1.0, 0.0], [0.0, 1.0], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h6f.kind != scientific::F_NONE);
    let h7, h7v, h7f = scientific::minimize_grad([0.0, 0.0], "bfgs", false, [], [], 1.0e-12, 1.0e-12, 1.0e-12, 500) { v => [v[0] * v[0]] };
    assert(h7f.kind != scientific::F_NONE);
}
