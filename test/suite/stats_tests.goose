// The tests of the group stats. Run them with SCIENTIFIC_SUITES=stats.
namespace stats_tests;

import scientific from "../..";
import os;
import common;

fn test_stats() {
    // The values come from NumPy 2.5.3.
    let m = common::mat([1.0, 2.0, 4.0, 7.0, 2.0, 1.5, 3.0, 9.0, 5.0, 3.0, 1.0, 0.5], 3, 4);
    let c, f1 = scientific::cov(m);
    assert(f1.kind == scientific::F_NONE && c.shape[0] == 3 && c.shape[1] == 3);
    assert(common::close_all(c, [7.0, 8.583333333333332, -4.916666666666666, 8.583333333333332, 12.0625, -4.9375, -4.916666666666666, -4.9375, 4.229166666666666]));
    let c0, f2 = scientific::cov(m, true, 0);
    assert(f2.kind == scientific::F_NONE && common::close(c0.data[0], 5.25) && common::close(c0.data[4], 9.046875));
    let t = scientific::transpose(m);
    let ct, f3 = scientific::cov(t, false);
    assert(f3.kind == scientific::F_NONE && common::close(ct.data[1], 8.583333333333332) && common::close(ct.data[8], 4.229166666666666));
    let r, f4 = scientific::corrcoef(m);
    assert(f4.kind == scientific::F_NONE);
    assert(common::close_all(r, [1.0, 0.9340890810744523, -0.9036377079347127, 0.9340890810744522, 1.0, -0.6912912266696619, -0.9036377079347127, -0.6912912266696619, 1.0]));
    let row = common::mat([1.0, 2.0, 4.0, 7.0], 1, 4);
    let y = common::mat([3.0, 1.0, 2.0, 8.0], 1, 4);
    let cw, f5 = scientific::cov_with(row, y);
    assert(f5.kind == scientific::F_NONE && common::close_all(cw, [7.0, 6.666666666666666, 6.666666666666666, 9.666666666666666]));
    let rw, f6 = scientific::corrcoef_with(row, y);
    assert(f6.kind == scientific::F_NONE && common::close(rw.data[1], 0.8104408984731077) && common::close(rw.data[0], 1.0));
    let bad = common::mat([1.0, 2.0, 3.0], 1, 3);
    let e1, g1 = scientific::cov_with(row, bad);
    assert(g1.kind == scientific::F_SHAPE);
    // The NaN statistics.
    let nan = 0.0 / 0.0;
    var av: f64[>..] = [3.0, nan, 1.0, 7.0, nan, 5.0];
    let a = scientific::Arr { ndim: 1, shape: [6], data: copy(av) };
    let mx, g2 = scientific::nanargmax(a);
    let mn, g3 = scientific::nanargmin(a);
    assert(g2.kind == scientific::F_NONE && mx == 3 && g3.kind == scientific::F_NONE && mn == 2);
    let md, g4 = scientific::nanmedian(a);
    let pc, g5 = scientific::nanpercentile(a, 30.0);
    let qq, g6 = scientific::nanquantile(a, 0.9, "lower");
    assert(g4.kind == scientific::F_NONE && common::close(md, 4.0) && common::close(pc, 2.8) && common::close(qq, 5.0));
    var only: f64[>..] = [nan, nan];
    let allnan = scientific::Arr { ndim: 1, shape: [2], data: copy(only) };
    let nm, g7 = scientific::nanmedian(allnan);
    assert(g7.kind == scientific::F_NONE && nm != nm);
    let ng, g8 = scientific::nanargmax(allnan);
    assert(g8.kind == scientific::F_VALUE);
    var bv: f64[>..] = [1.0, nan, 3.0, 2.0, 5.0, nan];
    let b = common::mat(bv, 2, 3);
    let cs, g9 = scientific::nancumsum(b, 1);
    let cp, g10 = scientific::nancumprod(b, 0);
    assert(g9.kind == scientific::F_NONE && common::close_all(cs, [1.0, 1.0, 4.0, 2.0, 7.0, 7.0]));
    assert(g10.kind == scientific::F_NONE && common::close_all(cp, [1.0, 1.0, 3.0, 2.0, 5.0, 3.0]));
    // The running forms of the array API.
    let q = common::mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let ci, g11 = scientific::cumulative_sum(q, 1, true);
    assert(g11.kind == scientific::F_NONE && ci.shape[1] == 4 && common::close_all(ci, [0.0, 1.0, 3.0, 6.0, 0.0, 4.0, 9.0, 15.0]));
    let cq, g12 = scientific::cumulative_prod(q, 0, true);
    assert(g12.kind == scientific::F_NONE && cq.shape[0] == 3 && common::close_all(cq, [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 10.0, 18.0]));
    let cn, g13 = scientific::cumulative_sum(q, 0);
    assert(g13.kind == scientific::F_NONE && common::close_all(cn, [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]));
    // The differences with the ends.
    var ev: f64[>..] = [1.0, 4.0, 9.0, 16.0];
    let e = scientific::Arr { ndim: 1, shape: [4], data: copy(ev) };
    let ed, g14 = scientific::ediff1d(e, [-1.0], [100.0, 200.0]);
    assert(g14.kind == scientific::F_NONE && common::close_all(ed, [-1.0, 3.0, 5.0, 7.0, 100.0, 200.0]));
    let one = scientific::Arr { ndim: 1, shape: [1], data: [5.0] };
    let e2, g15 = scientific::ediff1d(one);
    assert(g15.kind == scientific::F_SHAPE);
    // The edges of the bins.
    let he, g16 = scientific::histogram_bin_edges(e, 3);
    let he2, g17 = scientific::histogram_edges(e, 3);
    assert(g16.kind == scientific::F_NONE && common::close_all(he, he2.data));
}
