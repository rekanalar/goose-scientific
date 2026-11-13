// The tests of the group select. Run them with SCIENTIFIC_SUITES=select.
namespace select_tests;

import scientific from "../..";
import os;
import common;

// Selection and order, pinned against NumPy.
fn test_select() {
    let k, g0 = scientific::from_rows(2, 5, [3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0]);
    assert(g0.kind == scientific::F_NONE);
    let ls, g1 = scientific::lexsort(k);
    assert(g1.kind == scientific::F_NONE && common::same_vals(ls, [5], [2.0, 0.0, 4.0, 1.0, 3.0]));
    let kn, g2 = scientific::from_rows(1, 4, [2.0, 1.0, 0.0 / 0.0, 1.0]);
    assert(g2.kind == scientific::F_NONE);
    let ln, g3 = scientific::lexsort(kn);
    assert(g3.kind == scientific::F_NONE && common::same_vals(ln, [4], [1.0, 3.0, 0.0, 2.0]));
    let a, g4 = scientific::reshape(common::seq(6), [2, 3]);
    assert(g4.kind == scientific::F_NONE);
    let c1, g5 = scientific::compress_axis([0.0, 1.0, 1.0], a, 1);
    assert(g5.kind == scientific::F_NONE && common::same_vals(c1, [2, 2], [1.0, 2.0, 4.0, 5.0]));
    let c2, g6 = scientific::compress([1.0, 0.0, 1.0], a);
    assert(g6.kind == scientific::F_NONE && common::same_vals(c2, [2], [0.0, 2.0]));
    let c3, g7 = scientific::compress([0.0, 0.0], a);
    assert(g7.kind == scientific::F_VALUE && c3.data.len == 0);
    let cm, g8 = scientific::from_rows(2, 3, [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    assert(g8.kind == scientific::F_NONE);
    let ex, g9 = scientific::extract(cm, a);
    assert(g9.kind == scientific::F_NONE && common::same_vals(ex, [3], [0.0, 2.0, 5.0]));
    let ch, g10 = scientific::reshape(common::seq(12), [3, 4]);
    assert(g10.kind == scientific::F_NONE);
    let ix = common::av([5.0, -1.0, 1.0, 2.0]);
    let r1, g11 = scientific::choose(ix, ch, "wrap");
    assert(g11.kind == scientific::F_NONE && common::same_vals(r1, [4], [8.0, 9.0, 6.0, 11.0]));
    let r2, g12 = scientific::choose(ix, ch, "clip");
    assert(g12.kind == scientific::F_NONE && common::same_vals(r2, [4], [8.0, 1.0, 6.0, 11.0]));
    let r3, g13 = scientific::choose(ix, ch);
    assert(g13.kind == scientific::F_RANGE && r3.data.len == 0);
    let xs = common::av([-2.0, 0.0, 3.0, 5.0]);
    let cd, g14 = scientific::from_rows(2, 4, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    assert(g14.kind == scientific::F_NONE);
    let pw, g15 = scientific::piecewise(xs, cd, [-1.0]);
    assert(g15.kind == scientific::F_SHAPE && pw.data.len == 0);
    let cd2, g16 = scientific::reshape(cd, [2, 4]);
    assert(g16.kind == scientific::F_NONE);
    let pw2, g17 = scientific::piecewise(xs, cd2, [-1.0, 1.0, 0.5]);
    assert(g17.kind == scientific::F_NONE && common::same_vals(pw2, [4], [-1.0, 0.5, 1.0, 1.0]));
    // setxor1d keeps a NaN of both arrays twice.
    let sx, g18 = scientific::setxor1d(common::av([1.0, 2.0, 3.0, 0.0 / 0.0]), common::av([3.0, 4.0, 0.0 / 0.0]));
    assert(g18.kind == scientific::F_NONE && sx.data.len == 5);
    assert(sx.data[0] == 1.0 && sx.data[1] == 2.0 && sx.data[2] == 4.0 && sx.data[3] != sx.data[3] && sx.data[4] != sx.data[4]);
    let sy, g19 = scientific::setxor1d(common::av([1.0, 2.0, 2.0]), common::av([2.0, 5.0]));
    assert(g19.kind == scientific::F_NONE && common::same_vals(sy, [2], [1.0, 5.0]));
    let sz, g20 = scientific::setxor1d(common::av([1.0]), common::av([1.0]));
    assert(g20.kind == scientific::F_VALUE && sz.data.len == 0);
    // The unique functions keep each NaN apart.
    let u, ui, uv, uc, g21 = scientific::unique_all(common::av([2.0, 1.0, 2.0, 0.0 / 0.0, 0.0 / 0.0, 1.0]));
    assert(g21.kind == scientific::F_NONE && u.data.len == 4);
    assert(u.data[0] == 1.0 && u.data[1] == 2.0 && u.data[2] != u.data[2] && u.data[3] != u.data[3]);
    assert(common::same_vals(ui, [4], [1.0, 0.0, 3.0, 4.0]));
    assert(common::same_vals(uv, [6], [1.0, 0.0, 1.0, 2.0, 3.0, 0.0]));
    assert(common::same_vals(uc, [4], [2.0, 2.0, 1.0, 1.0]));
    let m, g22 = scientific::from_rows(2, 2, [3.0, 1.0, 3.0, 0.0 / 0.0]);
    assert(g22.kind == scientific::F_NONE);
    let um2, um3, g24 = scientific::unique_inverse(m);
    assert(g24.kind == scientific::F_NONE && common::same_vals(um3, [2, 2], [1.0, 0.0, 1.0, 2.0]));
    let cv, cc, g25 = scientific::unique_counts(common::av([5.0, 5.0, 1.0]));
    assert(g25.kind == scientific::F_NONE && common::same_vals(cv, [2], [1.0, 5.0]) && common::same_vals(cc, [2], [1.0, 2.0]));
    let vv, g26 = scientific::unique_values(common::av([3.0, 1.0, 3.0]));
    assert(g26.kind == scientific::F_NONE && common::same_vals(vv, [2], [1.0, 3.0]));
    // The partition holds the sorted element at kth.
    let pa, g27 = scientific::argpartition(common::av([3.0, 1.0, 2.0, 0.0]), 1);
    assert(g27.kind == scientific::F_NONE && common::same_vals(pa, [4], [3.0, 1.0, 2.0, 0.0]));
    let pb, g28 = scientific::argpartition(common::av([3.0, 1.0, 2.0, 0.0]), 4);
    assert(g28.kind == scientific::F_RANGE && pb.data.len == 0);
}
