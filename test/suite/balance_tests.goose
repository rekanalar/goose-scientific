// The tests of the group balance. Run them with SCIENTIFIC_SUITES=balance.
namespace balance_tests;

import scientific from "../..";
import os;
import common;

fn test_balance() {
    // The balance of a matrix follows the factors of LAPACK dgebal.
    let m, fm = scientific::from_rows(3, 3, [1.0, 2.0, 0.0, 9.0, 1.0, 0.01, 1.0, 2.0, 31.41592653589793]);
    let b, t, f = scientific::matrix_balance(m);
    assert(f.kind == scientific::F_NONE);
    common::vchk(b, f, [1.0, 4.0, 0.0, 4.5, 1.0, 0.01, 0.5, 2.0, 31.41592653589793]);
    common::vchk(t, f, [0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let b2, sc, perm, f2 = scientific::matrix_balance_separate(m);
    assert(f2.kind == scientific::F_NONE);
    common::vchk(sc, f2, [0.5, 1.0, 1.0]);
    common::vchk(perm, f2, [0.0, 1.0, 2.0]);
    // A matrix that is not square, a value that is not finite and a
    // side above the limit fault.
    let ns, fns = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let e1, e2, fe1 = scientific::matrix_balance(ns);
    assert(fe1.kind == scientific::F_SHAPE);
    let nan, fnan = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 0.0 / 0.0]);
    let e3, e4, fe2 = scientific::matrix_balance(nan);
    assert(fe2.kind == scientific::F_VALUE);
    // The Frechet derivative of the exponential holds the change of the
    // exponential.
    let a, fa = scientific::from_rows(3, 3, [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 1.0, 0.0, 6.0]);
    let x, der, fd = scientific::expm_frechet(a, a);
    assert(fd.kind == scientific::F_NONE);
    common::vchk(der, fd, [1006.0459388377757, 714.3077260992329, 6152.020064934505, 1587.6745236830175, 1124.902813776815, 9724.14193366317, 992.2236725228236, 635.069809473207, 5967.164301451895]);
    // The condition number of the exponential.
    let b3, fb3 = scientific::from_rows(3, 3, [-0.3, 0.2, 0.6, 0.6, 0.3, -0.1, -0.7, 1.2, 0.9]);
    let k, fk = scientific::expm_cond(b3);
    assert(fk.kind == scientific::F_NONE && common::close(k, 1.7787805864469868));
    let c, fc = scientific::from_rows(2, 2, [0.0, 1.0, -2.0, -3.0]);
    let k2, fk2 = scientific::expm_cond(c);
    assert(fk2.kind == scientific::F_NONE && common::close(k2, 4.01715150000012));
    // A shape that is not square and a direction of another shape fault.
    let e5, e6, fe3 = scientific::expm_frechet(ns, ns);
    assert(fe3.kind == scientific::F_SHAPE);
    let e7, e8, fe4 = scientific::expm_frechet(a, ns);
    assert(fe4.kind == scientific::F_SHAPE);
    let e9, fe5 = scientific::expm_cond(ns);
    assert(fe5.kind == scientific::F_SHAPE);
}
