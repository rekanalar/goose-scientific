// The tests of the group cumulative_simpson. Run them with SCIENTIFIC_SUITES=cumulative_simpson.
namespace cumulative_simpson_tests;

import scientific from "../..";
import os;
import common;

// cumulative_simpson takes a start value, as SciPy does.
fn test_cumulative_simpson_initial() {
    let y = [1.0, 3.0, 2.0, 5.0, 4.0];
    let x = [0.0, 1.0, 2.5, 3.0, 4.5];
    // The axis form.
    var cy: f64[>..] = [];
    var cq: i64 = 0;
    while cq < 24 { cy.push(mathshim::power(cq, 1.5)); cq += 1; }
    let cya, cyf = scientific::from_values(cy);
    let cyr, cyg = scientific::reshape(cya, [2, 3, 4]);
    assert(cyf.kind == scientific::F_NONE && cyg.kind == scientific::F_NONE);
    let ca1, caf1 = scientific::cumulative_simpson_axis(cyr, 1, [], 0.5);
    assert(caf1.kind == scientific::F_NONE && ca1.shape[1] == 2 && common::all_close(ca1, [1.7238576250846034, 2.810113295832983, 4.170619444818321, 5.735831784968369, 9.104569499661586, 12.120226591665965, 15.539826258871042, 19.29334363773743, 26.26680740867807, 29.119546183128683, 32.06946372096823, 35.11342772281442, 64.50198974694078, 70.57957311951756, 76.84041326725652, 83.27919915639917], 1.0e-6));
    let ca2, caf2 = scientific::cumulative_simpson_axis(cyr, 2, [0.0, 1.0, 3.0, 4.5], 1.0, 0.0);
    assert(caf2.kind == scientific::F_NONE && ca2.shape[2] == 4 && common::mathshim_abs(ca2.data[3] - 10.28299733046346) < 1.0e-6 && common::mathshim_abs(ca2.data[23] - 453.835084850994) < 1.0e-5);
    let ca3, caf3 = scientific::cumulative_simpson_axis(cyr, 2, [0.0, 1.0, 3.0]);
    assert(caf3.kind == scientific::F_SHAPE);
    let ca4, caf4 = scientific::cumulative_simpson_axis(cyr, 3);
    assert(caf4.kind == scientific::F_RANGE);
    let a, f = scientific::cumulative_simpson(y, x, 2.0);
    assert(f.kind == scientific::F_NONE && a.data.len == 5);
    assert(common::win_close(a, [2.0, 4.17777777777777778, 8.52777777777777778, 10.34722222222222222, 18.97222222222222222]) || (common::mathshim_abs(a.data[1] - 4.177777777777778) < 1.0e-12 && common::mathshim_abs(a.data[4] - 18.97222222222222) < 1.0e-12));
    let b, g = scientific::cumulative_simpson_dx(y, 0.5, 0.0);
    assert(g.kind == scientific::F_NONE && b.data.len == 5);
    assert(common::mathshim_abs(b.data[0]) < 1.0e-15 && common::mathshim_abs(b.data[1] - 1.125) < 1.0e-12 && common::mathshim_abs(b.data[3] - 4.41666666666666667) < 1.0e-12 && common::mathshim_abs(b.data[4] - 6.83333333333333333) < 1.0e-12);
    let c, h = scientific::cumulative_simpson(y, x);
    assert(h.kind == scientific::F_NONE && c.data.len == 4);
    let d, k = scientific::cumulative_simpson(y, x, 1.0 / 0.0);
    assert(k.kind == scientific::F_VALUE);
}

// cumulative_simpson is pinned against SciPy.
fn test_cumulative_simpson() {
    let x = [0.0, 1.0, 2.5, 4.0, 6.0, 9.5];
    let y = [1.0, 2.0, 0.5, 3.0, 2.0, 4.0];
    let r1, f1 = scientific::cumulative_simpson(y, x);
    assert(f1.kind == scientific::F_NONE && r1.data.len == 5);
    assert(common::all_near(r1, [1.6333333333333333, 3.958333333333333, 6.931547619047619, 12.756944444444445, 21.86489898989899]));
    let r2, f2 = scientific::cumulative_simpson_dx(y, 2.0);
    assert(f2.kind == scientific::F_NONE && common::all_near(r2, [3.4166666666666665, 6.333333333333333, 10.416666666666666, 16.0, 21.5]));
    // A signal of two values falls back to the trapezoid.
    let r3, f3 = scientific::cumulative_simpson([1.0, 3.0], [0.0, 2.0]);
    assert(f3.kind == scientific::F_NONE && common::all_near(r3, [4.0]));
    // The places must rise.
    let b1, g1 = scientific::cumulative_simpson(y, [0.0, 1.0, 1.0, 4.0, 6.0, 9.5]);
    assert(g1.kind == scientific::F_VALUE);
    let b2, g2 = scientific::cumulative_simpson(y, [0.0, 1.0]);
    assert(g2.kind == scientific::F_SHAPE);
    let b3, g3 = scientific::cumulative_simpson_dx(y, 0.0);
    assert(g3.kind == scientific::F_VALUE);
    return;
}
