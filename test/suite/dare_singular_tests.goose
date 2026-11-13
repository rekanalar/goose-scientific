// The tests of the group dare_singular. Run them with SCIENTIFIC_SUITES=dare_singular.
namespace dare_singular_tests;

import scientific from "../..";
import os;
import common;
import common_2;

// test_dare_singular pins the discrete equation with a singular r
// against SciPy and checks the residual of each answer.
fn test_dare_singular() {
    let a = common::mx(2, 2, [0.9, 0.5, 0.0, 0.7]);
    let b = common::mx(2, 2, [1.0, 0.0, 0.0, 1.0]);
    let q = common::mx(2, 2, [1.0, 0.0, 0.0, 1.0]);
    let none = scientific::empty_arr();
    var v: f64[>..] = [0.1, 0.7];
    let r1 = common::mx(2, 2, [1.0, 1.0, 1.0, 1.0]);
    let r2 = common::mx(2, 2, [1.0, 0.0, 0.0, 0.0]);
    let r3 = common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]);
    let r4 = common::mx(2, 2, [v[0] * v[0], v[0] * v[1], v[1] * v[0], v[1] * v[1]]);
    let x1, f1 = scientific::solve_discrete_are(a, b, q, r1);
    assert(f1.kind == scientific::F_NONE);
    assert(common::all_close(x1, [1.4153846153846148, 0.5538461538461535, 0.5538461538461535, 1.7384615384615383], 1.0e-8));
    let x2, f2 = scientific::solve_discrete_are(a, b, q, r2);
    assert(f2.kind == scientific::F_NONE);
    assert(common::all_close(x2, [1.4744345835777424, 0.26357476865430124, 0.26357476865430124, 1.1464304270301675], 1.0e-8));
    let x3, f3 = scientific::solve_discrete_are(a, b, q, r3);
    assert(f3.kind == scientific::F_NONE);
    assert(common::all_close(x3, [1.0, 0.0, 0.0, 1.0], 1.0e-8));
    // A rank one r of inexact entries has a pivot of rounding size. The
    // call must still give the solution.
    let x4, f4 = scientific::solve_discrete_are(a, b, q, r4);
    assert(f4.kind == scientific::F_NONE);
    assert(common::all_close(x4, [1.0057343920863673, 0.034406352518204605, 0.034406352518204605, 1.2064381151092276], 1.0e-8));
    assert(common_2::qt_dare_resid(a.data, b.data, q.data, r1.data, x1.data, [], [], 2, 2) < 1.0e-8);
    assert(common_2::qt_dare_resid(a.data, b.data, q.data, r2.data, x2.data, [], [], 2, 2) < 1.0e-8);
    assert(common_2::qt_dare_resid(a.data, b.data, q.data, r3.data, x3.data, [], [], 2, 2) < 1.0e-8);
    assert(common_2::qt_dare_resid(a.data, b.data, q.data, r4.data, x4.data, [], [], 2, 2) < 1.0e-8);
    assert(common_2::qt_maxabs([x1.data[1] - x1.data[2], x4.data[1] - x4.data[2]]) < 1.0e-12);
    // The 3 by 3 case with two inputs.
    let a3 = common::mx(3, 3, [0.9, 0.5, 0.1, 0.0, 0.7, 0.2, 0.1, 0.0, 0.6]);
    let b3 = common::mx(3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let q3 = common::mx(3, 3, [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    let x5, f5 = scientific::solve_discrete_are(a3, b3, q3, r1);
    assert(f5.kind == scientific::F_NONE);
    assert(common::all_close(x5, [1.9116803444927128, 1.3211806703351416, -0.14493578196807358, 1.3211806703351416, 3.930864617356723, -0.27909060175522615, -0.14493578196807358, -0.27909060175522615, 3.316520365651271], 1.0e-8));
    assert(common_2::qt_dare_resid(a3.data, b3.data, q3.data, r1.data, x5.data, [], [], 3, 2) < 1.0e-8);
    // The six-operand form with e and s and a singular r.
    let e2 = common::mx(2, 2, [1.0, 0.2, 0.0, 1.5]);
    let s2 = common::mx(2, 2, [0.1, 0.0, 0.0, 0.2]);
    let x6, f6 = scientific::solve_discrete_are(a, b, q, r1, e2, s2);
    assert(f6.kind == scientific::F_NONE);
    assert(common::all_close(x6, [1.0876238193300583, 0.1157375986191378, 0.1157375986191378, 0.5428316361712323], 1.0e-8));
    assert(common_2::qt_dare_resid(a.data, b.data, q.data, r1.data, x6.data, e2.data, s2.data, 2, 2) < 1.0e-8);
    // The example of the SciPy documentation holds a singular r.
    let a7 = common::mx(2, 2, [0.0, 1.0, 0.0, -1.0]);
    let b7 = common::mx(2, 2, [1.0, 0.0, 2.0, 1.0]);
    let q7 = common::mx(2, 2, [-4.0, -4.0, -4.0, 7.0]);
    let r7 = common::mx(2, 2, [9.0, 3.0, 3.0, 1.0]);
    let x7, f7 = scientific::solve_discrete_are(a7, b7, q7, r7);
    assert(f7.kind == scientific::F_NONE);
    assert(common::all_close(x7, [-4.0, -4.0, -4.0, 7.0], 1.0e-8));
    // A zero b with a nonsingular r gives the Lyapunov solution.
    let a8 = common::mx(2, 2, [0.5, 0.1, 0.0, 0.3]);
    let z2 = common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]);
    let id2 = common::mx(2, 2, [1.0, 0.0, 0.0, 1.0]);
    let x8, f8 = scientific::solve_discrete_are(a8, z2, q, id2);
    assert(f8.kind == scientific::F_NONE);
    assert(common_2::qt_dare_resid(a8.data, z2.data, q.data, id2.data, x8.data, [], [], 2, 2) < 1.0e-10);
    // A zero b with a zero r leaves r + b^T x b singular. The equation
    // has no meaning, so the call gives a fault.
    let x9, f9 = scientific::solve_discrete_are(a8, z2, q, z2);
    assert(f9.kind == scientific::F_VALUE);
    // An unstable a that b does not reach has no stabilizing solution.
    let one = common::mx(1, 1, [1.0]);
    let two = common::mx(1, 1, [2.0]);
    let zero1 = common::mx(1, 1, [0.0]);
    let x10, f10 = scientific::solve_discrete_are(two, zero1, one, one);
    assert(f10.kind == scientific::F_VALUE);
    // A scalar deadbeat case with r = 0 gives x = 1, like SciPy.
    let x11, f11 = scientific::solve_discrete_are(one, one, one, zero1);
    assert(f11.kind == scientific::F_NONE);
    assert(common::all_close(x11, [1.0], 1.0e-10));
    // A pair on the unit circle that b does not reach has no solution.
    let x12, f12 = scientific::solve_discrete_are(one, zero1, one, one);
    assert(f12.kind == scientific::F_VALUE);
    return;
}

// test_dare_singular_hostile covers the faults of the equation.
fn test_dare_singular_hostile() {
    let a = common::mx(2, 2, [0.9, 0.5, 0.0, 0.7]);
    let b = common::mx(2, 2, [1.0, 0.0, 0.0, 1.0]);
    let q = common::mx(2, 2, [1.0, 0.0, 0.0, 1.0]);
    let r = common::mx(2, 2, [1.0, 1.0, 1.0, 1.0]);
    let lie = scientific::Arr { ndim: 3, shape: [2, 2], data: [1.0, 0.0, 0.0, 1.0] };
    let x1, f1 = scientific::solve_discrete_are(a, b, q, lie);
    assert(f1.kind == scientific::F_SHAPE);
    let sh = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let x2, f2 = scientific::solve_discrete_are(a, b, q, sh);
    assert(f2.kind == scientific::F_SHAPE);
    let big = scientific::Arr { ndim: 2, shape: [129, 129], data: [1.0] };
    let x3, f3 = scientific::solve_discrete_are(big, b, q, r);
    assert(f3.kind == scientific::F_LIMIT);
    let x4, f4 = scientific::solve_discrete_are(a, common::mx(3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]), q, r);
    assert(f4.kind == scientific::F_SHAPE);
    let nan = 0.0 / 0.0;
    let x5, f5 = scientific::solve_discrete_are(common::mx(2, 2, [0.9, nan, 0.0, 0.7]), b, q, r);
    assert(f5.kind == scientific::F_VALUE);
    let x6, f6 = scientific::solve_discrete_are(a, common::mx(2, 2, [1.0, nan, 0.0, 1.0]), q, r);
    assert(f6.kind == scientific::F_VALUE);
    let x7, f7 = scientific::solve_discrete_are(a, b, common::mx(2, 2, [1.0, 0.0, 0.0, nan]), r);
    assert(f7.kind == scientific::F_VALUE);
    let x8, f8 = scientific::solve_discrete_are(a, b, q, common::mx(2, 2, [1.0, 1.0, nan, 1.0]));
    assert(f8.kind == scientific::F_VALUE);
    let x9, f9 = scientific::solve_discrete_are(a, b, q, common::mx(2, 2, [1.0, 2.0, 0.0, 1.0]));
    assert(f9.kind == scientific::F_VALUE);
    let none = scientific::empty_arr();
    let x10, f10 = scientific::solve_discrete_are(a, b, q, r, common::mx(2, 2, [1.0, 0.0, 0.0, nan]), none);
    assert(f10.kind == scientific::F_VALUE);
    return;
}
