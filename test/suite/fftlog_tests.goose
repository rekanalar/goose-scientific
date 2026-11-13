// The tests of the group fftlog. Run them with SCIENTIFIC_SUITES=fftlog.
namespace fftlog_tests;

import scientific from "../..";
import os;
import common;

// The fast Hankel transform, pinned against SciPy.
fn test_fht() {
    var g: f64[>..] = [];
    var i: i64 = 0;
    while i < 16 { let x = -3.0 + 6.0 * i / 15.0; g.push(exp(-x * x)); i += 1; }
    let r1, f1 = scientific::fht(g, 0.1, 0.0, 0.0, 0.0);
    assert(f1.kind == scientific::F_NONE && common::head_close(r1, [0.16658286794408866, 0.2125535499243573, 0.12136785788964644, -0.055993321795547724], 1.0e-11));
    let r2, f2 = scientific::fht(g, 0.4, 0.5, 0.3, 0.7);
    assert(f2.kind == scientific::F_NONE && common::head_close(r2, [0.11360093664708755, 0.08674346332121119, 0.13744050051674025, 0.23965782425244547], 1.0e-11));
    let r3, f3 = scientific::fht(g, 0.4, 2.0, -0.2, -0.3);
    assert(f3.kind == scientific::F_NONE && common::head_close(r3, [-0.0010266310475184393, 0.00022635263065843024, 0.001499351158171856, 0.0048398554983498645], 1.0e-11));
    let r4, f4 = scientific::fht(g, 0.4, 0.5, 0.0, 1.5);
    assert(f4.kind == scientific::F_NONE && common::head_close(r4, [24.602791125026318, 8.050436997621295, 0.8187829868428674, 0.15983414122779394], 1.0e-11));
    let r5, f5 = scientific::fht(g, 0.4, -1.0, 0.0, 0.0);
    assert(f5.kind == scientific::F_NONE && common::head_close(r5, [0.5482819240477328, 0.5403825991188047, 0.5250854033959822, 0.4920255609315086], 1.0e-11));
    let back, fb = scientific::ifht(r2.data, 0.4, 0.5, 0.3, 0.7);
    assert(fb.kind == scientific::F_NONE && common::mathshim_abs(back.data[0] - g[0]) < 1.0e-10 && common::mathshim_abs(back.data[9] - g[9]) < 1.0e-10);
    var h: f64[>..] = [];
    i = 0;
    while i < 15 { let x = -3.0 + 6.0 * i / 14.0; h.push(exp(-x * x)); i += 1; }
    let r6, f6 = scientific::fht(h, 0.3, 1.0, 0.1, 0.2);
    assert(f6.kind == scientific::F_NONE && common::head_close(r6, [0.017451947686648563, 0.05451441806908365, 0.06457579551959093, 0.10553434465142811], 1.0e-11));
    let one, f7 = scientific::fht([2.0], 0.5, 1.0, 0.0, 0.0);
    assert(f7.kind == scientific::F_NONE && one.data.len == 1 && common::mathshim_abs(one.data[0] - 2.0) < 1.0e-12);
    let o1, g1 = scientific::fhtoffset(0.1, 2.0, 0.5, 0.0);
    let o2, g2 = scientific::fhtoffset(0.3, -0.5, 0.2, 0.4);
    assert(g1.kind == scientific::F_NONE && g2.kind == scientific::F_NONE);
    assert(common::mathshim_abs(o1 - 0.5454581477676637) < 1.0e-12 && common::mathshim_abs(o2 - 0.07335295675204884) < 1.0e-12);
    // Faults: a singular bias, an empty signal and bad numbers.
    let s1, h1 = scientific::fht(g, 0.4, 0.5, 0.0, -1.5);
    assert(h1.kind == scientific::F_VALUE);
    let s2, h2 = scientific::ifht(g, 0.4, 0.5, 0.0, 1.5);
    assert(h2.kind == scientific::F_VALUE);
    let e: f64[0] = [];
    let s3, h3 = scientific::fht(e, 0.4, 0.5, 0.0, 0.0);
    assert(h3.kind == scientific::F_SHAPE);
    let s4, h4 = scientific::fht(g, 0.0, 0.5, 0.0, 0.0);
    assert(h4.kind == scientific::F_VALUE);
    let s5, h5 = scientific::fht(g, 0.4, 1.0 / 0.0, 0.0, 0.0);
    assert(h5.kind == scientific::F_VALUE);
    let o3, h6 = scientific::fhtoffset(0.0, 1.0, 0.0, 0.0);
    assert(h6.kind == scientific::F_VALUE);
}
