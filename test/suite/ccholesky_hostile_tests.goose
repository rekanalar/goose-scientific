// The tests of the group ccholesky_hostile. Run them with SCIENTIFIC_SUITES=ccholesky_hostile.
namespace ccholesky_hostile_tests;

import scientific from "../..";
import os;
import common;

// test_ccholesky_hostile covers the faults of the factor.
fn test_ccholesky_hostile() {
    // The matrix is not square.
    let a, f0 = scientific::complex_array(common::mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), common::mx(2, 3, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(f0.kind == scientific::F_NONE);
    let l1, l1f = scientific::ccholesky(a);
    assert(l1f.kind == scientific::F_SHAPE);
    // The shape holds fewer entries than the count of dimensions says.
    let lie = scientific::CArr { ndim: 2, shape: [3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
    let l6, l6f = scientific::ccholesky(lie);
    assert(l6f.kind == scientific::F_SHAPE);
    // The buffer does not match the shape.
    let short = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0] };
    let l2, l2f = scientific::ccholesky(short);
    assert(l2f.kind == scientific::F_SHAPE);
    // The side is above the limit.
    let big = scientific::CArr { ndim: 2, shape: [1025, 1025], data: [1.0] };
    let l3, l3f = scientific::ccholesky(big);
    assert(l3f.kind == scientific::F_LIMIT);
    // A value is not finite.
    let nan = 0.0 / 0.0;
    let nf, f1 = scientific::complex_array(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]), common::mx(2, 2, [nan, 0.0, 0.0, 0.0]));
    assert(f1.kind == scientific::F_NONE);
    let l4, l4f = scientific::ccholesky(nf);
    assert(l4f.kind == scientific::F_VALUE);
    // The matrix is not positive definite.
    let np, f2 = scientific::complex_array(common::mx(2, 2, [0.0, 1.0, 1.0, 0.0]), common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(f2.kind == scientific::F_NONE);
    let l5, l5f = scientific::ccholesky(np);
    assert(l5f.kind == scientific::F_VALUE);
    return;
}
