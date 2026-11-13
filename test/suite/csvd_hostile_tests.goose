// The tests of the group csvd_hostile. Run them with SCIENTIFIC_SUITES=csvd_hostile.
namespace csvd_hostile_tests;

import scientific from "../..";
import os;
import common;

// test_csvd_hostile covers the faults of the singular values.
fn test_csvd_hostile() {
    let v, f0 = scientific::complex_from_values([1.0, 2.0], [0.0, 0.0]);
    assert(f0.kind == scientific::F_NONE);
    let u1, s1, vh1, g1 = scientific::csvd(v);
    assert(g1.kind == scientific::F_SHAPE);
    let short = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0] };
    let u2, s2, vh2, g2 = scientific::csvd(short);
    assert(g2.kind == scientific::F_SHAPE);
    let lie = scientific::CArr { ndim: 2, shape: [2], data: [1.0, 0.0, 2.0, 0.0] };
    let u3, s3, vh3, g3 = scientific::csvd(lie);
    assert(g3.kind == scientific::F_SHAPE);
    let nan = 0.0 / 0.0;
    let bad = scientific::CArr { ndim: 2, shape: [1, 2], data: [1.0, 0.0, nan, 0.0] };
    let u4, s4, vh4, g4 = scientific::csvd(bad);
    assert(g4.kind == scientific::F_VALUE);
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 2 * 513 { big.push(1.0); bi += 1; }
    let wide = scientific::CArr { ndim: 2, shape: [1, 513], data: copy(big) };
    let u5, s5, vh5, g5 = scientific::csvd(wide);
    assert(g5.kind == scientific::F_LIMIT);
    let u6, s6, vh6, g6 = scientific::csvd(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]));
    assert(g6.kind == scientific::F_SHAPE);
    return;
}
