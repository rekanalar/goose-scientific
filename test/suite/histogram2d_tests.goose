// The tests of the group histogram2d. Run them with SCIENTIFIC_SUITES=histogram2d.
namespace histogram2d_tests;

import scientific from "../..";
import os;

fn test_histogram2d() {
    // NumPy 2.5.3: the pairs fall in a grid of two bins on each axis.
    let x, fx = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    let y, fy = scientific::from_values([0.0, 5.0, 1.0, 2.0]);
    assert(fx.kind == scientific::F_NONE && fy.kind == scientific::F_NONE);
    let h, ex, ey, f = scientific::histogram2d(x, y, 2, 2);
    assert(f.kind == scientific::F_NONE && h.shape[0] == 2 && h.shape[1] == 2);
    assert(h.data[0] == 1.0 && h.data[1] == 1.0 && h.data[2] == 2.0 && h.data[3] == 0.0);
    assert(ex.data[0] == 1.0 && ex.data[1] == 2.5 && ex.data[2] == 4.0);
    assert(ey.data[0] == 0.0 && ey.data[1] == 2.5 && ey.data[2] == 5.0);
    // A value on the right edge of the range falls in the last bin.
    let e5, fe5 = scientific::from_values([1.0, 5.0]);
    let e3, fe3 = scientific::from_values([1.0, 2.0]);
    assert(fe5.kind == scientific::F_NONE && fe3.kind == scientific::F_NONE);
    let hr, _exr, _eyr, fr = scientific::histogram2d(e5, e3, 2, 2);
    assert(fr.kind == scientific::F_NONE && hr.data[0] == 1.0 && hr.data[1] == 0.0 && hr.data[2] == 0.0 && hr.data[3] == 1.0);
    // The inputs must hold the same count.
    let _h1, _e1, _e2, bad = scientific::histogram2d(x, e5, 2, 2);
    assert(bad.kind == scientific::F_SHAPE);
    let _h2, _e3, _e4, bad2 = scientific::histogram2d(x, y, 0, 2);
    assert(bad2.kind == scientific::F_VALUE);
    // The grid of points counts along every column.
    let pts, fpts = scientific::from_rows(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(fpts.kind == scientific::F_NONE);
    let hd, ed, fd = scientific::histogramdd(pts, 2);
    assert(fd.kind == scientific::F_NONE && hd.ndim == 2 && hd.data.len == 4);
    assert(hd.data[0] == 1.0 && hd.data[1] == 0.0 && hd.data[2] == 0.0 && hd.data[3] == 2.0);
    assert(ed.shape[0] == 2 && ed.shape[1] == 3);
    assert(ed.data[0] == 1.0 && ed.data[2] == 5.0 && ed.data[3] == 2.0 && ed.data[5] == 6.0);
    // A column where every point holds one value is a fault, as the
    // histogram of one dimension faults too.
    let one, fone = scientific::from_rows(1, 3, [1.0, 2.0, 3.0]);
    assert(fone.kind == scientific::F_NONE);
    let _h1, _ed1, f1 = scientific::histogramdd(one, 2);
    assert(f1.kind == scientific::F_VALUE);
    // A row per dimension above the limit is a fault.
    let wide, fwide = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    assert(fwide.kind == scientific::F_NONE);
    let _h3, _ed3, fbad = scientific::histogramdd(wide, 0);
    assert(fbad.kind == scientific::F_VALUE);
    let same, fsame = scientific::from_rows(2, 2, [1.0, 5.0, 1.0, 6.0]);
    assert(fsame.kind == scientific::F_NONE);
    let _h4, _ed4, fbad2 = scientific::histogramdd(same, 2);
    assert(fbad2.kind == scientific::F_VALUE);
}
