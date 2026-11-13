// The tests of the group grids. Run them with SCIENTIFIC_SUITES=grids.
namespace grids_tests;

import scientific from "../..";
import os;

fn test_grids() {
    let x, fx = scientific::from_values([1.0, 2.0, 3.0]);
    let y, fy = scientific::from_values([10.0, 20.0]);
    assert(fx.kind == scientific::F_NONE && fy.kind == scientific::F_NONE);
    // The default indexing gives the shape (len(y), len(x)).
    let a, b, f = scientific::meshgrid(x, y);
    assert(f.kind == scientific::F_NONE && a.shape[0] == 2 && a.shape[1] == 3);
    assert(a.data[0] == 1.0 && a.data[2] == 3.0 && a.data[3] == 1.0 && a.data[5] == 3.0);
    assert(b.data[0] == 10.0 && b.data[2] == 10.0 && b.data[3] == 20.0 && b.data[5] == 20.0);
    // The ij indexing gives the shape (len(x), len(y)).
    let c, d, g = scientific::meshgrid(x, y, "ij");
    assert(g.kind == scientific::F_NONE && c.shape[0] == 3 && c.shape[1] == 2);
    assert(c.data[0] == 1.0 && c.data[1] == 1.0 && c.data[2] == 2.0 && c.data[5] == 3.0);
    assert(d.data[0] == 10.0 && d.data[1] == 20.0 && d.data[2] == 10.0 && d.data[5] == 20.0);
    // The sparse form holds one row and one column.
    let e, h, k = scientific::meshgrid(x, y, "xy", true);
    assert(k.kind == scientific::F_NONE && e.shape[0] == 1 && e.shape[1] == 3);
    assert(h.shape[0] == 2 && h.shape[1] == 1 && h.data[1] == 20.0);
    let e2, h2, k2 = scientific::meshgrid(x, y, "ij", true);
    assert(k2.kind == scientific::F_NONE && e2.shape[0] == 3 && e2.shape[1] == 1);
    assert(h2.shape[0] == 1 && h2.shape[1] == 2);
    // A bad indexing and an input of two dimensions are faults.
    let _p, _q, bad = scientific::meshgrid(x, y, "zz");
    assert(bad.kind == scientific::F_VALUE);
    let m, fm = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    assert(fm.kind == scientific::F_NONE);
    let _r, _s, bad2 = scientific::meshgrid(m, y);
    assert(bad2.kind == scientific::F_SHAPE);
    // The cross product of two vectors of three elements.
    let u, fu = scientific::from_values([1.0, 2.0, 3.0]);
    let v, fv = scientific::from_values([4.0, 5.0, 6.0]);
    assert(fu.kind == scientific::F_NONE && fv.kind == scientific::F_NONE);
    let w, fw = scientific::cross(u, v);
    assert(fw.kind == scientific::F_NONE && w.ndim == 1);
    assert(w.data[0] == -3.0 && w.data[1] == 6.0 && w.data[2] == -3.0);
    // Rows of vectors, and one vector against many rows.
    let ra, fra = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rb, frb = scientific::from_rows(2, 3, [7.0, 8.0, 9.0, 1.0, 2.0, 3.0]);
    assert(fra.kind == scientific::F_NONE && frb.kind == scientific::F_NONE);
    let rc, frc = scientific::cross(ra, rb);
    assert(frc.kind == scientific::F_NONE && rc.shape[0] == 2 && rc.shape[1] == 3);
    assert(rc.data[0] == -6.0 && rc.data[1] == 12.0 && rc.data[2] == -6.0);
    assert(rc.data[3] == 3.0 && rc.data[4] == -6.0 && rc.data[5] == 3.0);
    let rd, frd = scientific::cross(ra, v);
    assert(frd.kind == scientific::F_NONE && rd.shape[0] == 2 && rd.data[0] == -3.0);
    assert(rd.data[3] == 0.0 && rd.data[4] == 0.0 && rd.data[5] == 0.0);
    // A vector of two elements is a fault, as in NumPy 2.
    let t2, ft2 = scientific::from_values([1.0, 2.0]);
    assert(ft2.kind == scientific::F_NONE);
    let _z, bad3 = scientific::cross(t2, t2);
    assert(bad3.kind == scientific::F_SHAPE);
    let r3, fr3 = scientific::from_rows(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert(fr3.kind == scientific::F_NONE);
    let _z2, bad4 = scientific::cross(ra, r3);
    assert(bad4.kind == scientific::F_SHAPE);
}
