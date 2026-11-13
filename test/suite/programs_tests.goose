// The tests of the group programs. Run them with SCIENTIFIC_SUITES=programs.
namespace programs_tests;

import scientific from "../..";
import os;
import common;

fn test_programs() {
    // Mixed integer programs.
    let kx, kv, kf = scientific::milp([-8.0, -11.0, -6.0, -4.0], [true, true, true, true], [5.0, 7.0, 4.0, 3.0], [14.0], [], [], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]);
    assert(kf.kind == scientific::F_NONE && common::rel_close(kx, [0.0, 1.0, 1.0, 1.0], 1.0e-9) && common::mathshim_abs(kv + 21.0) < 1.0e-9);
    let mx2, mv2, mf2 = scientific::milp([-1.0, -2.0], [true, false], [1.0, 1.0, 1.0, -1.0], [3.5, 1.2], [], []);
    assert(mf2.kind == scientific::F_NONE && common::rel_close(mx2, [0.0, 3.5], 1.0e-9) && common::mathshim_abs(mv2 + 7.0) < 1.0e-9);
    let mx3, mv3, mf3 = scientific::milp([1.0, 1.0], [true, true], [], [], [2.0, 2.0], [3.0]);
    assert(mf3.kind == scientific::F_VALUE);
    let mx4, mv4, mf4 = scientific::milp([1.0, 1.0], [true], [], [], [], []);
    assert(mf4.kind == scientific::F_SHAPE);
    let l1, l1v, l1f = scientific::linprog([-1.0, -2.0], [1.0, 1.0, 1.0, 3.0, 2.0, 1.0], [4.0, 6.0, 5.0], [], [], [], []);
    assert(l1f.kind == scientific::F_NONE && common::all_near(l1, [1.8000000000000007, 1.4]) && common::near(l1v, -4.6000000000000005));
    let l2, l2v, l2f = scientific::linprog([2.0, 3.0, 1.0], [-1.0, -1.0, -1.0], [-10.0], [1.0, -1.0, 0.0], [2.0], [], []);
    assert(l2f.kind == scientific::F_NONE && common::all_near(l2, [2.0, 0.0, 8.0]) && common::near(l2v, 12.0));
    let l3, l3v, l3f = scientific::linprog([1.0, 1.0], [-1.0, -2.0], [-3.0], [], [], [-5.0, -(1.0 / 0.0)], [(1.0 / 0.0), 10.0]);
    assert(l3f.kind == scientific::F_NONE && common::all_near(l3, [-5.0, 4.0]) && common::near(l3v, -1.0));
    let l4, l4v, l4f = scientific::linprog([-3.0, -5.0, -4.0], [2.0, 3.0, 0.0, 0.0, 2.0, 5.0, 3.0, 2.0, 4.0], [8.0, 10.0, 15.0], [], [], [0.0, 1.0, 0.0], [2.0, 3.0, (1.0 / 0.0)]);
    assert(l4f.kind == scientific::F_NONE && common::all_near(l4, [2.0, 1.3333333333333333, 1.4666666666666668]) && common::near(l4v, -18.53333333333333));
    let l5, l5v, l5f = scientific::linprog([1.0, -1.0], [], [], [1.0, 1.0], [2.0], [-(1.0 / 0.0), -1.0], [(1.0 / 0.0), 3.0]);
    assert(l5f.kind == scientific::F_NONE && common::all_near(l5, [-1.0, 3.0]) && common::near(l5v, -4.0));
    let nn, nnr, nnf = scientific::nnls(common::mx(4, 2, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0]), [2.0, 1.0, -1.0, 4.0]);
    assert(nnf.kind == scientific::F_NONE && common::all_near(nn, [1.8333333333333333, 0.0]) && common::near(nnr, 1.3540064007726602));
    let n2, n2r, n2f = scientific::nnls(common::mx(5, 4, [1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 2.0, 1.0, 1.0]), [3.0, -2.0, 1.0, 0.5, 2.0]);
    assert(n2f.kind == scientific::F_NONE && common::all_near(n2, [0.6666666666666664, 0.5833333333333336, 0.0, 0.0]) && common::near(n2r, 2.993047499344661));
    let a1r, a1c, a1f = scientific::linear_sum_assignment(common::mx(3, 3, [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0]), false);
    assert(a1f.kind == scientific::F_NONE && common::all_near(a1r, [0.0, 1.0, 2.0]) && common::all_near(a1c, [1.0, 0.0, 2.0]));
    let a2r, a2c, a2f = scientific::linear_sum_assignment(common::mx(3, 3, [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0]), true);
    assert(a2f.kind == scientific::F_NONE && common::all_near(a2r, [0.0, 1.0, 2.0]) && common::all_near(a2c, [0.0, 2.0, 1.0]));
    let a3r, a3c, a3f = scientific::linear_sum_assignment(common::mx(2, 4, [4.0, 1.0, 3.0, 9.0, 2.0, 0.0, 5.0, 1.0]), false);
    assert(a3f.kind == scientific::F_NONE && common::all_near(a3r, [0.0, 1.0]) && common::all_near(a3c, [1.0, 3.0]));
    let a4r, a4c, a4f = scientific::linear_sum_assignment(common::mx(4, 2, [4.0, 1.0, 2.0, 0.0, 3.0, 2.0, 0.5, 7.0]), false);
    assert(a4f.kind == scientific::F_NONE && common::all_near(a4r, [1.0, 3.0]) && common::all_near(a4c, [1.0, 0.0]));
    let ag, agf = scientific::approx_fprime([1.0, 2.0, -0.5]) { x => x[0] * x[0] * x[1] + common::mathshim_sin(x[2]) };
    assert(agf.kind == scientific::F_NONE && common::all_near(ag, [4.000000029802322, 1.0, 0.8775825649499893]));
    let br, brf = scientific::bisect(0.0, 2.0) { x => x * x - 2.0 };
    assert(brf.kind == scientific::F_NONE && common::near(br, 1.4142135623730951));
    let rr, rrf = scientific::ridder(0.0, 3.0) { x => x * x * x - 2.0 * x - 5.0 + 4.0 };
    assert(rrf.kind == scientific::F_NONE && common::near(rr, 1.6180339887498911));
    let i1, i1v, i1f = scientific::linprog([1.0], [-1.0], [-1.0], [], [], [0.0], [0.5]);
    assert(i1f.kind == scientific::F_VALUE);
    let u1, u1v, u1f = scientific::linprog([-1.0], [], [], [], []);
    assert(u1f.kind == scientific::F_VALUE);
    let s1, s1v, s1f = scientific::linprog([1.0, 2.0], [1.0], [1.0], [], []);
    assert(s1f.kind == scientific::F_SHAPE);
    let c1, c2, cf = scientific::linear_sum_assignment(common::mx(2, 2, [1.0, 1.0 / 0.0, 1.0, 2.0]));
    assert(cf.kind == scientific::F_VALUE);
    let nbr, nbf = scientific::bisect(1.0, 2.0) { x => x * x - 2.0 + 5.0 };
    assert(nbf.kind == scientific::F_VALUE);
}
