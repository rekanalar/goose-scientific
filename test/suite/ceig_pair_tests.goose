// The tests of the group ceig_pair. Run them with SCIENTIFIC_SUITES=ceig_pair.
namespace ceig_pair_tests;

import scientific from "../..";
import os;
import common_3;

// test_ceig_pair_hostile covers the faults of the general pair calls.
fn test_ceig_pair_hostile() {
    let m = common_3::gm(2, [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]);
    let one = scientific::CArr { ndim: 1, shape: [4], data: [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0] };
    let al, be, v, f1 = scientific::ceig_pair(one, m);
    assert(f1.kind == scientific::F_SHAPE);
    let w1, g1 = scientific::ceigvals_pair(m, one);
    assert(g1.kind == scientific::F_SHAPE);
    let short = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0] };
    let al2, be2, v2, f2 = scientific::ceig_pair(m, short);
    assert(f2.kind == scientific::F_SHAPE);
    let rect = scientific::CArr { ndim: 2, shape: [1, 2], data: [1.0, 0.0, 2.0, 0.0] };
    let al3, be3, v3, f3 = scientific::ceig_pair(rect, rect);
    assert(f3.kind == scientific::F_SHAPE);
    let tiny = common_3::gm(1, [1.0], [0.0]);
    let al4, be4, v4, f4 = scientific::ceig_pair(m, tiny);
    assert(f4.kind == scientific::F_SHAPE);
    let nan = 0.0 / 0.0;
    let bad = scientific::CArr { ndim: 2, shape: [1, 1], data: [nan, 0.0] };
    let al5, be5, v5, f5 = scientific::ceig_pair(tiny, bad);
    assert(f5.kind == scientific::F_VALUE);
    let w6, g6 = scientific::ceigvals_pair(bad, tiny);
    assert(g6.kind == scientific::F_VALUE);
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 2 * 257 * 257 { big.push(0.0); bi += 1; }
    let wide = scientific::CArr { ndim: 2, shape: [257, 257], data: copy(big) };
    let al7, be7, v7, f7 = scientific::ceig_pair(wide, wide);
    assert(f7.kind == scientific::F_LIMIT);
    let w8, g8 = scientific::ceigvals_pair(wide, wide);
    assert(g8.kind == scientific::F_LIMIT);
    // A pair of zero matrices gives no fault and eigenvalues that are not numbers.
    let zero = common_3::gm(2, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]);
    let w9, g9 = scientific::ceigvals_pair(zero, zero);
    assert(g9.kind == scientific::F_NONE);
    assert(w9.data[0] != w9.data[0]);
    // The vectors of a singular pencil stay finite and of length one.
    let alz, bez, vz, fz = scientific::ceig_pair(zero, zero);
    assert(fz.kind == scientific::F_NONE);
    assert(common_3::pair_unit(vz));
    let one2 = common_3::gm(2, [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]);
    let alr, ber, vr, fr = scientific::ceig_pair(one2, one2);
    assert(fr.kind == scientific::F_NONE);
    assert(common_3::pair_unit(vr));
    return;
}
