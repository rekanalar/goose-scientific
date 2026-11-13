// The tests of the group scilinalg. Run them with SCIENTIFIC_SUITES=scilinalg.
namespace scilinalg_tests;

import scientific from "../..";
import os;
import common;

// The functions of scipy.linalg for factors, special matrices and the
// matrix exponential, pinned against SciPy.
fn test_scilinalg() {
    let a = common::mx(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
    let f, piv, g0 = scientific::lu_factor(a);
    assert(g0.kind == scientific::F_NONE);
    assert(common::all_near(f, [7.0, 8.0, 10.0, 0.14285714285714285, 0.8571428571428572, 1.5714285714285716, 0.5714285714285714, 0.5000000000000002, -0.5]));
    assert(common::same_vals(piv, [3], [2.0, 2.0, 2.0]));
    let x1, g1 = scientific::lu_solve(f, piv, common::av([1.0, 2.0, 3.0]));
    assert(g1.kind == scientific::F_NONE && common::all_near(x1, [-1.0 / 3.0, 2.0 / 3.0, 0.0]));
    let x2, g2 = scientific::lu_solve(f, piv, common::av([1.0, 2.0, 3.0]), true);
    assert(g2.kind == scientific::F_NONE && common::all_near(x2, [1.0, 0.0, 0.0]));
    let p, l, u, g3 = scientific::lu(common::mx(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert(g3.kind == scientific::F_NONE);
    assert(common::same_vals(p, [3, 3], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]));
    assert(common::all_near(l, [1.0, 0.0, 0.2, 1.0, 0.6, 0.5]) && common::all_near(u, [5.0, 6.0, 0.0, 0.8]));
    let sing, sp, g4 = scientific::lu_factor(common::mx(2, 2, [1.0, 2.0, 2.0, 4.0]));
    assert(g4.kind == scientific::F_NONE);
    let sx, g6 = scientific::lu_solve(sing, sp, common::av([1.0, 1.0]));
    assert(g6.kind == scientific::F_VALUE && sx.data.len == 0);
    // solve takes a matrix on the right.
    let sm, g7 = scientific::solve(a, common::mx(3, 2, [1.0, 0.0, 2.0, 1.0, 3.0, 0.0]));
    assert(g7.kind == scientific::F_NONE && sm.ndim == 2 && sm.shape[1] == 2);
    assert(common::near(sm.data[0], -1.0 / 3.0) && common::near(sm.data[2], 2.0 / 3.0) && common::near(sm.data[4], 0.0));
    // triangular solves
    let t = common::mx(3, 3, [2.0, 9.0, 9.0, 1.0, 3.0, 9.0, 4.0, 5.0, 6.0]);
    let r1, g8 = scientific::solve_triangular(t, common::av([2.0, 4.0, 15.0]), true);
    assert(g8.kind == scientific::F_NONE && common::all_near(r1, [1.0, 1.0, 1.0]));
    let r2, g9 = scientific::solve_triangular(t, common::av([2.0, 4.0, 15.0]), false, 1);
    assert(g9.kind == scientific::F_NONE && common::all_near(r2, [1.0, -1.6666666666666667, 3.5]));
    let r3, g10 = scientific::solve_triangular(t, common::mx(3, 1, [2.0, 4.0, 15.0]), true, 0, true);
    assert(g10.kind == scientific::F_NONE && common::all_near(r3, [2.0, 2.0, -3.0]));
    let r4, g11 = scientific::solve_triangular(common::mx(2, 2, [0.0, 1.0, 0.0, 1.0]), common::av([1.0, 1.0]));
    assert(g11.kind == scientific::F_VALUE && r4.data.len == 0);
    // special matrices
    let bd, g12 = scientific::block_diag(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]), common::av([5.0, 6.0, 7.0]));
    assert(g12.kind == scientific::F_NONE && common::same_vals(bd, [3, 5], [1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0]));
    let tp, g13 = scientific::toeplitz([1.0, 2.0, 3.0], [1.0, 5.0, 6.0, 7.0]);
    assert(g13.kind == scientific::F_NONE && common::same_vals(tp, [3, 4], [1.0, 5.0, 6.0, 7.0, 2.0, 1.0, 5.0, 6.0, 3.0, 2.0, 1.0, 5.0]));
    let ts, g14 = scientific::toeplitz([1.0, 2.0, 3.0]);
    assert(g14.kind == scientific::F_NONE && common::same_vals(ts, [3, 3], [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]));
    let hk, g15 = scientific::hankel([1.0, 2.0, 3.0], [3.0, 7.0, 8.0]);
    assert(g15.kind == scientific::F_NONE && common::same_vals(hk, [3, 3], [1.0, 2.0, 3.0, 2.0, 3.0, 7.0, 3.0, 7.0, 8.0]));
    let h2, g16 = scientific::hankel([1.0, 2.0, 3.0]);
    assert(g16.kind == scientific::F_NONE && common::same_vals(h2, [3, 3], [1.0, 2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 0.0, 0.0]));
    let ci, g17 = scientific::circulant([1.0, 2.0, 3.0]);
    assert(g17.kind == scientific::F_NONE && common::same_vals(ci, [3, 3], [1.0, 3.0, 2.0, 2.0, 1.0, 3.0, 3.0, 2.0, 1.0]));
    // null space and range
    let nm = common::mx(2, 3, [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
    let ns, g18 = scientific::null_space(nm);
    assert(g18.kind == scientific::F_NONE && ns.shape[0] == 3 && ns.shape[1] == 2);
    let prod, g19 = scientific::matmul(nm, ns);
    assert(g19.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < prod.data.len { assert(common::mathshim_abs(prod.data[i]) < 0.000000001); i += 1; }
    let gram, g20 = scientific::matmul(scientific::transpose(ns), ns);
    assert(g20.kind == scientific::F_NONE && common::near(gram.data[0], 1.0) && common::near(gram.data[3], 1.0) && common::mathshim_abs(gram.data[1]) < 0.000000001);
    let nf, g21 = scientific::null_space(common::mx(2, 2, [1.0, 0.0, 0.0, 1.0]));
    assert(g21.kind == scientific::F_VALUE && nf.data.len == 0);
    let og, g22 = scientific::orth(common::mx(3, 2, [1.0, 2.0, 2.0, 4.0, 3.0, 6.0]));
    assert(g22.kind == scientific::F_NONE && og.shape[0] == 3 && og.shape[1] == 1);
    assert(common::near(og.data[0] * og.data[0] + og.data[1] * og.data[1] + og.data[2] * og.data[2], 1.0));
    // the matrix exponential
    let e1, g23 = scientific::expm(common::mx(2, 2, [0.0, 1.0, -1.0, 0.0]));
    assert(g23.kind == scientific::F_NONE && common::all_near(e1, [0.5403023058681397, 0.8414709848078965, -0.8414709848078966, 0.5403023058681397]));
    let e2, g24 = scientific::expm(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]));
    assert(g24.kind == scientific::F_NONE && common::all_near(e2, [51.96895619870756, 74.73656456700694, 112.10484685051044, 164.07380304921801]));
    let e3, g25 = scientific::expm(common::mx(2, 2, [10.0, 0.0, 0.0, -20.0]));
    assert(g25.kind == scientific::F_NONE && common::near(e3.data[0], 22026.465794806718) && common::near(e3.data[3] * 1.0e9, 2.061153622438558));
    let e4, g26 = scientific::expm(common::mx(2, 2, [0.0, 50.0, 0.0, 0.0]));
    assert(g26.kind == scientific::F_NONE && common::all_near(e4, [1.0, 50.0, 0.0, 1.0]));
    let e5, g27 = scientific::expm(common::mx(2, 3, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(g27.kind == scientific::F_SHAPE && e5.data.len == 0);
    // the banded solve
    let ab = common::mx(3, 4, [0.0, 2.0, 1.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 1.0, 1.0, 0.0]);
    let sb, g28 = scientific::solve_banded(1, 1, ab, common::av([1.0, 2.0, 3.0, 4.0]));
    assert(g28.kind == scientific::F_NONE && common::all_near(sb, [0.09577464788732393, 0.30845070422535215, 0.3619718309859155, 0.5197183098591549]));
    let sbb, g29 = scientific::solve_banded(1, 0, ab, common::av([1.0, 2.0, 3.0, 4.0]));
    assert(g29.kind == scientific::F_SHAPE && sbb.data.len == 0);
    // a zero on the main diagonal needs a row swap
    let abp = common::mx(3, 4, [0.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0]);
    let sp, g30 = scientific::solve_banded(1, 1, abp, common::av([1.0, 2.0, 3.0, 4.0]));
    assert(g30.kind == scientific::F_NONE && common::all_near(sp, [-10.0, 0.5, 4.0, 1.0]));
    // two lower and one upper diagonal, two right sides
    let ab2 = common::mx(4, 5, [0.0, 1.0, 2.0, 3.0, 4.0, 2.0, -1.0, 3.0, 1.0, 5.0, 1.0, 2.0, -1.0, 4.0, 0.0, 3.0, 1.0, 2.0, 0.0, 0.0]);
    let s2, g31 = scientific::solve_banded(2, 1, ab2, common::mx(5, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0]));
    assert(g31.kind == scientific::F_NONE && s2.shape[0] == 5 && s2.shape[1] == 2);
    assert(common::all_near(s2, [-0.8437499999999986, -3.6874999999999987, 2.687499999999998, 9.374999999999996, 3.2656249999999982, 8.531249999999996, -2.5468749999999982, -9.093749999999996, 2.531249999999999, 4.062499999999998]));
    // a side of 2000 solves at the cost of the band
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 2000 { if bi == 0 { big.push(0.0); } else { big.push(-1.0); } bi += 1; }
    bi = 0;
    while bi < 2000 { big.push(4.0); bi += 1; }
    bi = 0;
    while bi < 2000 { if bi == 1999 { big.push(0.0); } else { big.push(-1.0); } bi += 1; }
    var ones: f64[>..] = [];
    bi = 0;
    while bi < 2000 { ones.push(1.0); bi += 1; }
    let sbig, g32 = scientific::solve_banded(1, 1, common::mx(3, 2000, copy(big)), common::av(copy(ones)));
    assert(g32.kind == scientific::F_NONE && common::near(sbig.data[0], 0.36602540378443865) && common::near(sbig.data[1000], 0.5) && common::near(sbig.data[1999], 0.36602540378443865));
    // a zero matrix is singular
    let sz, g33 = scientific::solve_banded(1, 1, common::mx(3, 2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), common::av([1.0, 2.0]));
    assert(g33.kind == scientific::F_VALUE && sz.data.len == 0);
}
