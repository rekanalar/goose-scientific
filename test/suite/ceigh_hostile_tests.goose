// The tests of the group ceigh_hostile. Run them with SCIENTIFIC_SUITES=ceigh_hostile.
namespace ceigh_hostile_tests;

import scientific from "../..";
import os;
import common;

// test_ceigh_hostile covers the faults of the Hermitian eigenvalue calls.
fn test_ceigh_hostile() {
    let m, f0 = scientific::complex_array(common::mx(2, 2, [2.0, 1.0, 1.0, 3.0]), common::mx(2, 2, [0.0, -1.0, 1.0, 0.0]));
    assert(f0.kind == scientific::F_NONE);
    let w1, v1, g1 = scientific::ceigh(m, "X");
    assert(g1.kind == scientific::F_VALUE);
    let w2, v2, g2 = scientific::ceigh(m, "");
    assert(g2.kind == scientific::F_VALUE);
    let w3, v3, g3 = scientific::ceigh(m, "LU");
    assert(g3.kind == scientific::F_VALUE);
    let w4, g4 = scientific::ceigvalsh(m, "x");
    assert(g4.kind == scientific::F_VALUE);
    let rect = scientific::CArr { ndim: 2, shape: [1, 2], data: [1.0, 0.0, 2.0, 0.0] };
    let w5, v5, g5 = scientific::ceigh(rect);
    assert(g5.kind == scientific::F_SHAPE);
    let short = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0] };
    let w6, v6, g6 = scientific::ceigh(short);
    assert(g6.kind == scientific::F_SHAPE);
    let nan = 0.0 / 0.0;
    let bad = scientific::CArr { ndim: 2, shape: [1, 1], data: [nan, 0.0] };
    let w7, v7, g7 = scientific::ceigh(bad);
    assert(g7.kind == scientific::F_VALUE);
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 2 * 257 * 257 { big.push(0.0); bi += 1; }
    let wide = scientific::CArr { ndim: 2, shape: [257, 257], data: copy(big) };
    let w8, v8, g8 = scientific::ceigh(wide);
    assert(g8.kind == scientific::F_LIMIT);
    let w9, g9 = scientific::ceigvalsh(wide);
    assert(g9.kind == scientific::F_LIMIT);
    let w10, v10, g10 = scientific::ceigh(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]));
    assert(g10.kind == scientific::F_SHAPE);
    return;
}
