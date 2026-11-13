// The tests of the group delaunay. Run them with SCIENTIFIC_SUITES=delaunay.
namespace delaunay_tests;

import scientific from "../..";
import os;
import common;

fn test_griddata() {
    let nan = 0.0 / 0.0;
    let pd = [6.251, 8.972, 7.757, 2.252, 3.002, 8.736, 0.053, 8.212, 7.971, 4.679, 3.03, 2.784, 2.549, 4.451, 5.045, 5.535, 9.955, 7.927, 6.222, 9.89, 2.153, 1.602, 6.125, 0.439, 0.357, 5.149, 4.662, 9.172];
    let vd = [-1.83065, -0.264156, -1.404907, -0.647863, 0.926396, -1.762123, 0.041668, 0.520656, -0.651632, -1.848591, 0.772856, 1.652828, 1.195207, -2.935174];
    let xd = [5.0, 5.0, 2.5, 7.5, 9.0, 1.0, 0.1, 0.1, 0.053, 8.212, 20.0, 3.0, 4.4, 4.4];
    let pa = common::gmat(14, 2, pd);
    let xa = common::gmat(7, 2, xd);
    let va = scientific::line_arr(scientific::copy_slice(vd));
    // The triangles of a set in general position are those of SciPy.
    let d, fd = scientific::delaunay(pa);
    assert(fd.kind == scientific::F_NONE);
    assert(d.nsimp == 19);
    let want = [0, 4, 7, 0, 4, 8, 0, 7, 13, 0, 8, 9, 0, 9, 13, 1, 4, 7, 1, 4, 8, 1, 5, 7, 1, 5, 11, 2, 3, 9, 2, 3, 12, 2, 6, 7, 2, 6, 12, 2, 7, 13, 2, 9, 13, 5, 6, 7, 5, 6, 10, 5, 10, 11, 6, 10, 12];
    var found: i64 = 0;
    var t: i64 = 0;
    while t < 19 {
        var c: i64 = 0;
        while c < 19 {
            var hit: i64 = 0;
            var r: i64 = 0;
            while r < 3 {
                var s: i64 = 0;
                while s < 3 { if scientific::dl_corner(d, t * 3 + r) == want[c * 3 + s] { hit += 1; } s += 1; }
                r += 1;
            }
            if hit == 3 { found += 1; }
            c += 1;
        }
        t += 1;
    }
    assert(found == 19);
    let fs, ffs = scientific::delaunay_find_simplex(d, xa);
    assert(ffs.kind == scientific::F_NONE);
    assert(fs.data[2] < 0.0 && fs.data[3] < 0.0 && fs.data[5] < 0.0);
    assert(fs.data[0] >= 0.0 && fs.data[1] >= 0.0 && fs.data[4] >= 0.0 && fs.data[6] >= 0.0);
    let lin, fl = scientific::griddata(pa, va, xa, "linear", nan, false);
    assert(fl.kind == scientific::F_NONE);
    assert(common::grid_close(lin, [0.2081893437715816, -0.759977554500946, nan, nan, -0.647863, nan, -0.35053523060726854], 1.0e-12));
    let near, fn1 = scientific::griddata(pa, va, xa, "nearest", nan, false);
    assert(common::grid_close(near, [0.520656, -1.404907, -0.264156, 0.772856, -0.647863, -0.651632, 0.520656], 1.0e-12));
    let fill, ff = scientific::griddata(pa, va, xa, "linear", -9.0, false);
    assert(common::grid_close(fill, [0.2081893437715816, -0.759977554500946, -9.0, -9.0, -0.647863, -9.0, -0.35053523060726854], 1.0e-12));
    // The scaling of the axes leaves the triangles of the unit square.
    var ps: f64[>..] = [];
    var xs: f64[>..] = [];
    var i: i64 = 0;
    while i < 14 { ps.push(pd[i * 2]); ps.push(pd[i * 2 + 1] * 100.0); i += 1; }
    i = 0;
    while i < 7 { xs.push(xd[i * 2]); xs.push(xd[i * 2 + 1] * 100.0); i += 1; }
    let resc, fr = scientific::griddata(common::gmat(14, 2, ps), va, common::gmat(7, 2, xs), "linear", nan, true);
    assert(common::grid_close(resc, [0.20818934377158171, -0.7599775545009462, nan, nan, -0.647863, nan, -0.35053523060726827], 1.0e-10));
    // One dimension.
    let x1 = scientific::line_arr([3.0, 1.0, 2.0, 5.0]);
    let v1 = scientific::line_arr([30.0, 10.0, 20.0, 50.0]);
    let q1 = scientific::line_arr([0.5, 1.5, 2.5, 4.0, 5.0, 2.0]);
    let l1, fl1 = scientific::griddata(x1, v1, q1, "linear", nan, false);
    assert(common::grid_close(l1, [nan, 15.0, 25.0, 40.0, 50.0, 20.0], 1.0e-12));
    let n1, fn2 = scientific::griddata(x1, v1, q1, "nearest", nan, false);
    assert(common::grid_close(n1, [10.0, 10.0, 20.0, 30.0, 50.0, 20.0], 1.0e-12));
    let q2 = scientific::line_arr([-5.0, 0.99, 5.01, 9.0]);
    let n2, fn3 = scientific::griddata(x1, v1, q2, "nearest", nan, false);
    assert(common::grid_close(n2, [10.0, 10.0, 50.0, 50.0], 1.0e-12));
    // The faults.
    let e1, g1 = scientific::griddata(pa, va, xa, "cubic", nan, false);
    assert(g1.kind == scientific::F_VALUE);
    let e2, g2 = scientific::griddata(pa, scientific::line_arr([1.0, 2.0]), xa, "linear", nan, false);
    assert(g2.kind == scientific::F_SHAPE);
    let e3, g3 = scientific::griddata(pa, va, common::gmat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "linear", nan, false);
    assert(g3.kind == scientific::F_SHAPE);
    let e4, g4 = scientific::delaunay(common::gmat(3, 2, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]));
    assert(g4.kind == scientific::F_VALUE);
    let e5, g5 = scientific::delaunay(common::gmat(3, 2, [0.0, 0.0, 1.0, nan, 2.0, 2.0]));
    assert(g5.kind == scientific::F_VALUE);
    let e6, g6 = scientific::griddata(common::gmat(4, 3, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), scientific::line_arr([1.0, 2.0, 3.0, 4.0]), common::gmat(1, 3, [0.1, 0.1, 0.1]), "linear", nan, false);
    assert(g6.kind == scientific::F_LIMIT);
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 5001 { big.push(bi * 1.0); big.push((bi * bi % 97) * 1.0); bi += 1; }
    let eb, gb = scientific::delaunay(common::gmat(5001, 2, big));
    assert(gb.kind == scientific::F_LIMIT);
    // A repeated point stays out of the triangles.
    let dd, fdd = scientific::delaunay(common::gmat(5, 2, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]));
    assert(fdd.kind == scientific::F_NONE);
    assert(dd.nsimp == 2);
}
