// The tests of the group cqr_hostile. Run them with SCIENTIFIC_SUITES=cqr_hostile.
namespace cqr_hostile_tests;

import scientific from "../..";
import os;
import common;

// test_cqr_hostile covers the faults of the QR decomposition.
fn test_cqr_hostile() {
    let v, f0 = scientific::complex_from_values([1.0, 2.0], [0.0, 0.0]);
    assert(f0.kind == scientific::F_NONE);
    // A vector is not a matrix.
    let q1, r1, g1 = scientific::cqr(v);
    assert(g1.kind == scientific::F_SHAPE);
    // The buffer does not match the shape.
    let short = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0] };
    let q2, r2, g2 = scientific::cqr(short);
    assert(g2.kind == scientific::F_SHAPE);
    // The shape holds fewer entries than the count of dimensions says.
    let lie = scientific::CArr { ndim: 2, shape: [2], data: [1.0, 0.0, 2.0, 0.0] };
    let q3, r3, g3 = scientific::cqr(lie);
    assert(g3.kind == scientific::F_SHAPE);
    // A value is not finite.
    let nan = 0.0 / 0.0;
    let bad = scientific::CArr { ndim: 2, shape: [1, 2], data: [1.0, 0.0, nan, 0.0] };
    let q4, r4, g4 = scientific::cqr(bad);
    assert(g4.kind == scientific::F_VALUE);
    // A side holds more than the limit.
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 2 * 513 { big.push(1.0); bi += 1; }
    let wide = scientific::CArr { ndim: 2, shape: [1, 513], data: copy(big) };
    let q5, r5, g5 = scientific::cqr(wide);
    assert(g5.kind == scientific::F_LIMIT);
    // A real array reaches the call.
    let q6, r6, g6 = scientific::cqr(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]));
    assert(g6.kind == scientific::F_SHAPE);
    return;
}
