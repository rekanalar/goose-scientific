// The tests of the group moves. Run them with SCIENTIFIC_SUITES=moves.
namespace moves_tests;

import scientific from "../..";
import os;
import common;

fn test_moves() {
    // NumPy 2.5.3 values. The cube holds 0 to 23 in the shape 2, 3, 4.
    let flat, ff = scientific::arange(0.0, 24.0, 1.0);
    assert(ff.kind == scientific::F_NONE);
    let cube, fc = scientific::reshape(flat, [2, 3, 4]);
    assert(fc.kind == scientific::F_NONE);
    let m1, f1 = scientific::moveaxis(cube, 0, -1);
    assert(f1.kind == scientific::F_NONE && m1.shape[0] == 3 && m1.shape[1] == 4 && m1.shape[2] == 2);
    let m2, f2 = scientific::moveaxis(cube, -1, 0);
    assert(f2.kind == scientific::F_NONE && m2.shape[0] == 4 && m2.shape[1] == 2 && m2.shape[2] == 3);
    assert(m2.data[1 * 6 + 0 * 3 + 0] == 1.0 && m2.data[1 * 6 + 0 * 3 + 1] == 5.0 && m2.data[1 * 6 + 0 * 3 + 2] == 9.0);
    let m3, f3 = scientific::moveaxis_many(cube, [0, 1], [-1, -2]);
    assert(f3.kind == scientific::F_NONE && m3.shape[0] == 4 && m3.shape[1] == 3 && m3.shape[2] == 2);
    let _m4, f4 = scientific::moveaxis(cube, 3, 0);
    assert(f4.kind == scientific::F_RANGE);
    let _m5, f5 = scientific::moveaxis_many(cube, [0, 0], [1, 2]);
    assert(f5.kind == scientific::F_VALUE);
    // The flips and the turns of a matrix of two rows and three columns.
    let mat, fm = scientific::from_rows(2, 3, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert(fm.kind == scientific::F_NONE);
    let ud, fud = scientific::flipud(mat);
    assert(fud.kind == scientific::F_NONE && ud.data[0] == 3.0 && ud.data[3] == 0.0);
    let lr, flr = scientific::fliplr(mat);
    assert(flr.kind == scientific::F_NONE && lr.data[0] == 2.0 && lr.data[3] == 5.0);
    let r1, fr1 = scientific::rot90(mat);
    assert(fr1.kind == scientific::F_NONE && r1.shape[0] == 3 && r1.shape[1] == 2);
    assert(r1.data[0] == 2.0 && r1.data[1] == 5.0 && r1.data[2] == 1.0 && r1.data[5] == 3.0);
    let r2, fr2 = scientific::rot90(mat, 2);
    assert(fr2.kind == scientific::F_NONE && r2.data[0] == 5.0 && r2.data[5] == 0.0);
    let r3, fr3 = scientific::rot90(mat, 3);
    assert(fr3.kind == scientific::F_NONE && r3.data[0] == 3.0 && r3.data[1] == 0.0 && r3.data[5] == 2.0);
    let rm, frm = scientific::rot90(mat, -1);
    assert(frm.kind == scientific::F_NONE && rm.data[0] == 3.0 && rm.data[1] == 0.0);
    let r0, fr0 = scientific::rot90(mat, 0);
    assert(fr0.kind == scientific::F_NONE && r0.data[0] == 0.0 && r0.data[5] == 5.0);
    let rc, frc = scientific::rot90(cube);
    assert(frc.kind == scientific::F_NONE && rc.shape[0] == 3 && rc.shape[1] == 2 && rc.shape[2] == 4);
    let rd, frd = scientific::rot90(cube, 1, 1, 2);
    assert(frd.kind == scientific::F_NONE && rd.shape[0] == 2 && rd.shape[1] == 4 && rd.shape[2] == 3);
    let vec, fvec = scientific::from_values([1.0, 2.0]);
    assert(fvec.kind == scientific::F_NONE);
    let _rv, frv = scientific::rot90(vec);
    assert(frv.kind == scientific::F_SHAPE);
    let _lv, flv = scientific::fliplr(vec);
    assert(flv.kind == scientific::F_SHAPE);
    // The broadcast to a shape, and of two arrays.
    let row, frow = scientific::from_values([1.0, 2.0, 3.0]);
    assert(frow.kind == scientific::F_NONE);
    let bt, fbt = scientific::broadcast_to(row, [2, 3]);
    assert(fbt.kind == scientific::F_NONE && bt.shape[0] == 2 && bt.data[3] == 1.0 && bt.data[5] == 3.0);
    let _bad, fbad = scientific::broadcast_to(vec, [3]);
    assert(fbad.kind == scientific::F_SHAPE);
    let _bad2, fbad2 = scientific::broadcast_to(mat, [3]);
    assert(fbad2.kind == scientific::F_SHAPE);
    let col, fcol = scientific::from_rows(2, 1, [1.0, 2.0]);
    let wide, fwide = scientific::from_values([10.0, 20.0, 30.0]);
    assert(fcol.kind == scientific::F_NONE && fwide.kind == scientific::F_NONE);
    let bx, by, fba = scientific::broadcast_arrays(col, wide);
    assert(fba.kind == scientific::F_NONE && bx.shape[0] == 2 && bx.shape[1] == 3);
    assert(bx.data[0] == 1.0 && bx.data[3] == 2.0 && by.data[0] == 10.0 && by.data[5] == 30.0);
    // The choice: NumPy gives [1, 5, -1] for these two conditions.
    let conds, fcn = scientific::from_rows(2, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let picks, fpk = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(fcn.kind == scientific::F_NONE && fpk.kind == scientific::F_NONE);
    let sel, fsel = scientific::select(conds, picks, -1.0);
    assert(fsel.kind == scientific::F_NONE && sel.ndim == 1 && sel.data[0] == 1.0 && sel.data[1] == 5.0 && sel.data[2] == -1.0);
    let _sel2, fsel2 = scientific::select(conds, row, -1.0);
    assert(fsel2.kind == scientific::F_SHAPE);
    // The function along an axis: NumPy gives sums [3, 12] on axis 1,
    // the reversed columns on axis 0, and the bounds [[0, 2], [3, 5]].
    let sums, fsu = scientific::apply_along_axis_scalar(mat, 1, common::line_sum);
    assert(fsu.kind == scientific::F_NONE && sums.ndim == 1 && sums.data[0] == 3.0 && sums.data[1] == 12.0);
    let rev, frev = scientific::apply_along_axis(mat, 0, common::line_reverse);
    assert(frev.kind == scientific::F_NONE && rev.shape[0] == 2 && rev.data[0] == 3.0 && rev.data[2] == 5.0 && rev.data[3] == 0.0);
    let bnd, fbn = scientific::apply_along_axis(mat, 1, common::line_bounds);
    assert(fbn.kind == scientific::F_NONE && bnd.shape[0] == 2 && bnd.shape[1] == 2);
    assert(bnd.data[0] == 0.0 && bnd.data[1] == 2.0 && bnd.data[2] == 3.0 && bnd.data[3] == 5.0);
    let _ax, fax = scientific::apply_along_axis(mat, 2, common::line_reverse);
    assert(fax.kind == scientific::F_RANGE);
}
