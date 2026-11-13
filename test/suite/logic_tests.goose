// The tests of the group logic. Run them with SCIENTIFIC_SUITES=logic.
namespace logic_tests;

import scientific from "../..";
import os;

fn test_closeness() {
    // NumPy 2.5.3: the default sizes are a relative 1e-05 and an
    // absolute 1e-08.
    let a, fa = scientific::from_values([1.0e10, 1.0e-7, 1.0, 0.0]);
    let b, fb = scientific::from_values([1.00001e10, 1.0e-8, 1.0001, 0.0]);
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let c, fc = scientific::isclose(a, b);
    assert(fc.kind == scientific::F_NONE && c.data.len == 4);
    assert(c.data[0] == 1.0 && c.data[1] == 0.0 && c.data[2] == 0.0 && c.data[3] == 1.0);
    // The rule leans on the second number.
    let lo, flo = scientific::from_values([1.00001]);
    let one, fone = scientific::from_values([1.0]);
    assert(flo.kind == scientific::F_NONE && fone.kind == scientific::F_NONE);
    let n1, fn1 = scientific::isclose(lo, one, 1.0e-05, 0.0);
    let n2, fn2 = scientific::isclose(one, lo, 1.0e-05, 0.0);
    assert(fn1.kind == scientific::F_NONE && n1.data[0] == 0.0);
    assert(fn2.kind == scientific::F_NONE && n2.data[0] == 1.0);
    // The infinity equals itself. The missing value matches itself only
    // under the flag.
    let inf, finf = scientific::from_values([1.0 / 0.0, -1.0 / 0.0, 1.0 / 0.0, 0.0 / 0.0]);
    assert(finf.kind == scientific::F_NONE);
    let ie, fie = scientific::isclose(inf, inf);
    assert(fie.kind == scientific::F_NONE && ie.data[0] == 1.0 && ie.data[1] == 1.0 && ie.data[2] == 1.0);
    let nn, fnn = scientific::isclose(inf, inf, 1.0e-05, 1.0e-08, true);
    assert(fnn.kind == scientific::F_NONE && nn.data[3] == 1.0);
    let nm, fnm = scientific::isclose(inf, inf, 1.0e-05, 1.0e-08, false);
    assert(fnm.kind == scientific::F_NONE && nm.data[3] == 0.0);
    let pos, fpos = scientific::from_values([1.0 / 0.0]);
    let miss, fmiss = scientific::from_values([0.0 / 0.0]);
    assert(fpos.kind == scientific::F_NONE && fmiss.kind == scientific::F_NONE);
    let mixed, fmx = scientific::isclose(pos, miss);
    assert(fmx.kind == scientific::F_NONE && mixed.data[0] == 0.0);
    let all1, fall1 = scientific::allclose(a, b);
    assert(fall1.kind == scientific::F_NONE && all1 == false);
    let tw1, ftw1 = scientific::from_values([1.0, 2.0]);
    let tw2, ftw2 = scientific::from_values([1.0, 2.0]);
    assert(ftw1.kind == scientific::F_NONE && ftw2.kind == scientific::F_NONE);
    let ok, fok = scientific::allclose(tw1, tw2);
    assert(fok.kind == scientific::F_NONE && ok == true);
    // array_equal takes the shape and the missing value rule.
    let tri, ftri = scientific::from_values([1.0, 2.0, 3.0]);
    assert(ftri.kind == scientific::F_NONE);
    let e1, fe1 = scientific::array_equal(tw1, tri);
    assert(fe1.kind == scientific::F_NONE && e1 == false);
    let r, rr = scientific::from_rows(2, 2, [1.0, 0.0 / 0.0, 3.0, 4.0]);
    assert(rr.kind == scientific::F_NONE);
    let e2, fe2 = scientific::array_equal(r, r);
    assert(fe2.kind == scientific::F_NONE && e2 == false);
    let e3, fe3 = scientific::array_equal(r, r, true);
    assert(fe3.kind == scientific::F_NONE && e3 == true);
    let row12, frow12 = scientific::from_rows(1, 2, [1.0, 2.0]);
    assert(frow12.kind == scientific::F_NONE);
    let e4, fe4 = scientific::array_equal(tw1, row12);
    assert(fe4.kind == scientific::F_NONE && e4 == false);
}
