// The tests of the group ceig. Run them with SCIENTIFIC_SUITES=ceig.
namespace ceig_tests;

import scientific from "../..";
import os;
import common;
import common_2;

// test_ceig_hostile covers the faults of the general eigenvalue calls.
fn test_ceig_hostile() {
    let v, f0 = scientific::complex_from_values([1.0, 2.0], [0.0, 0.0]);
    assert(f0.kind == scientific::F_NONE);
    let w1, v1, g1 = scientific::ceig(v);
    assert(g1.kind == scientific::F_SHAPE);
    let short = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0] };
    let w2, v2, g2 = scientific::ceig(short);
    assert(g2.kind == scientific::F_SHAPE);
    let lie = scientific::CArr { ndim: 2, shape: [2], data: [1.0, 0.0, 2.0, 0.0] };
    let w3, v3, g3 = scientific::ceig(lie);
    assert(g3.kind == scientific::F_SHAPE);
    let rect = scientific::CArr { ndim: 2, shape: [1, 2], data: [1.0, 0.0, 2.0, 0.0] };
    let w4, v4, g4 = scientific::ceig(rect);
    assert(g4.kind == scientific::F_SHAPE);
    let nan = 0.0 / 0.0;
    let bad = scientific::CArr { ndim: 2, shape: [1, 1], data: [nan, 0.0] };
    let w5, v5, g5 = scientific::ceig(bad);
    assert(g5.kind == scientific::F_VALUE);
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 2 * 257 * 257 { big.push(0.0); bi += 1; }
    let wide = scientific::CArr { ndim: 2, shape: [257, 257], data: copy(big) };
    let w6, v6, g6 = scientific::ceig(wide);
    assert(g6.kind == scientific::F_LIMIT);
    let w7, g7 = scientific::ceigvals(wide);
    assert(g7.kind == scientific::F_LIMIT);
    let w8, v8, g8 = scientific::ceig(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]));
    assert(g8.kind == scientific::F_SHAPE);
    // The iteration stops with a limit of one step for one eigenvalue.
    var rng = scientific::default_rng(2970);
    let m = common_2::random_complex(rng, 4, 4);
    var hr: f64[>..] = [];
    var hi: f64[>..] = [];
    scientific::c_split(m.data, 4, 1.0, hr, hi);
    var qr = scientific::qz_identity(4);
    var qi = scientific::c_zeros(16);
    scientific::c_hessenberg(hr, hi, qr, qi, 4);
    assert(!scientific::c_schur(hr, hi, qr, qi, 4, 1));
    return;
}
