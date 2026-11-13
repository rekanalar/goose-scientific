// The helpers that the tests of more than one group share.
namespace common_11;

import scientific from "../..";
import os;
import common;
import common_7;

fn lil_pin_r52() {
    var d, f0 = scientific::lil_new(1, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, 2, [4.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 3.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 2.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 2.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 6, 5).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 2.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, 2, [0.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 3.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 1.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 1.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 5.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 3.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 2.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, 3, [0.0, 4.0, 4.0]).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -2, [0.0, 2.0, 1.0, 0.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 3.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 3.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -3, [2.0, 4.0, 4.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 0.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -4, [2.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 2.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 5.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -4, [2.0, 0.0, 1.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 1.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 3.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 4.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 3, 1).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, 0, [0.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 0.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 0.0, -3.0).kind == scientific::F_NONE);
    assert(common_7::lil_is(d, [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [-3.0, -3.0, 7.0]));
    assert(scientific::lil_nnz(d) == 3);
    let dn, fd = scientific::lil_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [-3.0, -3.0, 7.0]));
    let t, ft = scientific::lil_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_7::lil_is(t, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [-3.0, -3.0, 7.0]));
    let co, fc = scientific::lil_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 3);
    let cs, fs = scientific::lil_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 3);
    let mv, fm = scientific::lil_matvec(d, [-2.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [6.0, 6.0, -14.0]));
    let sm, fz = scientific::lil_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 1.0));
    let dg, fg = scientific::lil_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [-3.0]));
    let sd, fk = scientific::lil_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_7::lil_is(sd, [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [4.5, 4.5, -10.5]));
}

fn lil_pin_r53() {
    var d, f0 = scientific::lil_new(6, 5);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 5.0, 2.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 5.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 1.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 3.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -1, [0.0, 0.0, 4.0, 1.0, 2.0]).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 4, 6).kind == scientific::F_NONE);
    assert(common_7::lil_is(d, [1.0, 3.0], [3.0, 2.0], [1.0, 4.0]));
    assert(scientific::lil_nnz(d) == 2);
    let dn, fd = scientific::lil_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]));
    let t, ft = scientific::lil_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_7::lil_is(t, [2.0, 3.0], [3.0, 1.0], [4.0, 1.0]));
    let co, fc = scientific::lil_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 2);
    let cs, fs = scientific::lil_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 2);
    let mv, fm = scientific::lil_matvec(d, [-2.0, 1.0, 4.0, 0.0, 3.0, -1.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [0.0, 0.0, 0.0, 16.0]));
    let sm, fz = scientific::lil_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 5.0));
    let dg, fg = scientific::lil_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [0.0, 0.0, 0.0, 0.0]));
    let sd, fk = scientific::lil_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_7::lil_is(sd, [1.0, 3.0], [3.0, 2.0], [-1.5, -6.0]));
}

fn lil_pin_r54() {
    var d, f0 = scientific::lil_new(3, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 3.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 2.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -1, [0.0, 2.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 2.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 3.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 2.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 2.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, 3, [4.0, 2.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 2.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, 2, [2.0, 2.0, 4.0]).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 6, 5).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 4.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 4.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, 4, [1.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 5.0, 1.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 0.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 4.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 2.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 3.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 4.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 3.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 1.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 4.0, 7.0).kind == scientific::F_NONE);
    assert(common_7::lil_is(d, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0], [0.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 3.0, 3.0, 4.0, 1.0], [7.0, 2.0, 4.0, 1.0, 2.0, 2.0, 7.0, 2.0, 1.0, 5.0, 7.0, 2.0, 5.0, 7.0, -3.0, -3.0]));
    assert(scientific::lil_nnz(d) == 16);
    let dn, fd = scientific::lil_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [7.0, 0.0, 2.0, 4.0, 1.0, 0.0, 0.0, 2.0, 2.0, 7.0, 0.0, 2.0, 1.0, 5.0, 7.0, 0.0, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 7.0, -3.0, 0.0, -3.0, 0.0, 0.0, 0.0]));
    let t, ft = scientific::lil_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_7::lil_is(t, [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0], [0.0, 2.0, 3.0, 5.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 4.0], [7.0, 2.0, 2.0, -3.0, 2.0, 2.0, 1.0, 4.0, 2.0, 5.0, 5.0, 7.0, 1.0, 7.0, 7.0, -3.0]));
    let co, fc = scientific::lil_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 16);
    let cs, fs = scientific::lil_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 16);
    let mv, fm = scientific::lil_matvec(d, [-2.0, 1.0, 4.0, 0.0, 3.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [-3.0, 29.0, 27.0, 2.0, -9.0, -3.0]));
    let sm, fz = scientific::lil_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 48.0));
    let dg, fg = scientific::lil_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [7.0, 0.0, 1.0, 5.0, -3.0]));
    let sd, fk = scientific::lil_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_7::lil_is(sd, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0], [0.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 3.0, 3.0, 4.0, 1.0], [-10.5, -3.0, -6.0, -1.5, -3.0, -3.0, -10.5, -3.0, -1.5, -7.5, -10.5, -3.0, -7.5, -10.5, 4.5, 4.5]));
}

fn lil_pin_r55() {
    var d, f0 = scientific::lil_new(1, 3);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 2.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 0.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, 2, [1.0, 2.0]).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 2, 6).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 3.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 1.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 5.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 5, 1).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 0.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 0.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 0.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 0.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 1, 2).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, 1, [0.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 0.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 1.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 1.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 1.0, 0.0).kind == scientific::F_NONE);
    assert(common_7::lil_is(d, [0.0], [0.0], [2.0]));
    assert(scientific::lil_nnz(d) == 1);
    let dn, fd = scientific::lil_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [2.0, 0.0]));
    let t, ft = scientific::lil_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_7::lil_is(t, [0.0], [0.0], [2.0]));
    let co, fc = scientific::lil_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 1);
    let cs, fs = scientific::lil_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 1);
    let mv, fm = scientific::lil_matvec(d, [-2.0, 1.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [-4.0]));
    let sm, fz = scientific::lil_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 2.0));
    let dg, fg = scientific::lil_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [2.0]));
    let sd, fk = scientific::lil_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_7::lil_is(sd, [0.0], [0.0], [-3.0]));
}

fn lil_pin_r56() {
    var d, f0 = scientific::lil_new(6, 5);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -2, [2.0, 4.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 2.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 5, 2).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 0.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 1.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -2, [1.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 0.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -3, [2.0, 1.0, 4.0]).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -1, [2.0, 2.0, 2.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 1.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, 0, [4.0, 1.0]).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -1, [4.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(common_7::lil_is(d, [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0], [4.0, 4.0, 1.0, 1.0, 2.0, 2.0, 7.0, 1.0]));
    assert(scientific::lil_nnz(d) == 8);
    let dn, fd = scientific::lil_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [4.0, 0.0, 4.0, 1.0, 1.0, 2.0, 2.0, 7.0, 0.0, 1.0]));
    let t, ft = scientific::lil_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_7::lil_is(t, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0], [4.0, 4.0, 1.0, 2.0, 1.0, 2.0, 7.0, 1.0]));
    let co, fc = scientific::lil_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 8);
    let cs, fs = scientific::lil_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 8);
    let mv, fm = scientific::lil_matvec(d, [-2.0, 1.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [-8.0, -7.0, 0.0, 3.0, 1.0]));
    let sm, fz = scientific::lil_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 22.0));
    let dg, fg = scientific::lil_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [4.0, 1.0]));
    let sd, fk = scientific::lil_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_7::lil_is(sd, [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0], [-6.0, -6.0, -1.5, -1.5, -3.0, -3.0, -10.5, -1.5]));
}

fn lil_pin_r57() {
    var d, f0 = scientific::lil_new(5, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 1.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 3.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 2.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 3.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(common_7::lil_is(d, [], [], []));
    assert(scientific::lil_nnz(d) == 0);
    let dn, fd = scientific::lil_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let t, ft = scientific::lil_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_7::lil_is(t, [], [], []));
    let co, fc = scientific::lil_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 0);
    let cs, fs = scientific::lil_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 0);
    let mv, fm = scientific::lil_matvec(d, [-2.0, 1.0, 4.0, 0.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [0.0, 0.0, 0.0, 0.0, 0.0]));
    let sm, fz = scientific::lil_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 0.0));
    let dg, fg = scientific::lil_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [0.0, 0.0, 0.0, 0.0]));
    let sd, fk = scientific::lil_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_7::lil_is(sd, [], [], []));
}

fn lil_pin_r58() {
    var d, f0 = scientific::lil_new(5, 4);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 1.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 3.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 0.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 3.0, 2.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 2.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 2.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 0.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 1, 2).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 5, 4).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 4, 2).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 0.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 3, 5).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 6, 1).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 5, 1).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 0.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 0.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 0.0, 0.0).kind == scientific::F_NONE);
    assert(common_7::lil_is(d, [1.0, 2.0, 4.0], [0.0, 0.0, 0.0], [2.0, -3.0, 5.0]));
    assert(scientific::lil_nnz(d) == 3);
    let dn, fd = scientific::lil_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [0.0, 2.0, -3.0, 0.0, 5.0]));
    let t, ft = scientific::lil_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_7::lil_is(t, [0.0, 0.0, 0.0], [1.0, 2.0, 4.0], [2.0, -3.0, 5.0]));
    let co, fc = scientific::lil_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 3);
    let cs, fs = scientific::lil_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 3);
    let mv, fm = scientific::lil_matvec(d, [-2.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [0.0, -4.0, 6.0, 0.0, -10.0]));
    let sm, fz = scientific::lil_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 4.0));
    let dg, fg = scientific::lil_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [0.0]));
    let sd, fk = scientific::lil_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_7::lil_is(sd, [1.0, 2.0, 4.0], [0.0, 0.0, 0.0], [-3.0, 4.5, -7.5]));
}

fn lil_pin_r59() {
    var d, f0 = scientific::lil_new(1, 6);
    assert(f0.kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 6, 4).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 1.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 5.0, 3.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 1.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 5.0, 3.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 2.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 7.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 0.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 2.0, -3.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 5.0, 2.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 5, 6).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 2.0, 0.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 4.0, 4.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 3.0, 0.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -1, [1.0, 0.0, 4.0, 1.0]).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -2, [4.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 0.0, 5.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_resize(d, 3, 1).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 0.0, 5.0).kind == scientific::F_NONE);
    assert(scientific::lil_setdiag(d, -2, [1.0, 0.0]).kind == scientific::F_NONE);
    assert(scientific::lil_set(d, 1.0, 0.0, -3.0).kind == scientific::F_NONE);
    assert(common_7::lil_is(d, [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [2.0, -3.0, 1.0]));
    assert(scientific::lil_nnz(d) == 3);
    let dn, fd = scientific::lil_to_dense(d);
    assert(fd.kind == scientific::F_NONE && common::all_near(dn, [2.0, -3.0, 1.0]));
    let t, ft = scientific::lil_transpose(d);
    assert(ft.kind == scientific::F_NONE && common_7::lil_is(t, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [2.0, -3.0, 1.0]));
    let co, fc = scientific::lil_to_coo(d);
    assert(fc.kind == scientific::F_NONE && co.nnz == 3);
    let cs, fs = scientific::lil_to_csr(d);
    assert(fs.kind == scientific::F_NONE && cs.nnz == 3);
    let mv, fm = scientific::lil_matvec(d, [-2.0]);
    assert(fm.kind == scientific::F_NONE && common::all_near(mv, [-4.0, 6.0, -2.0]));
    let sm, fz = scientific::lil_sum(d);
    assert(fz.kind == scientific::F_NONE && common::near(sm, 0.0));
    let dg, fg = scientific::lil_diagonal(d);
    assert(fg.kind == scientific::F_NONE && common::all_near(dg, [2.0]));
    let sd, fk = scientific::lil_scale(d, -1.5);
    assert(fk.kind == scientific::F_NONE && common_7::lil_is(sd, [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [-3.0, 4.5, -1.5]));
}

fn lil_pin_conv() {
    let a, fa = scientific::from_values([0.0, 2.0, 0.0, 3.0, 0.0, 4.0]);
    assert(fa.kind == scientific::F_NONE);
    let a2, fr = scientific::reshape(a, [2, 3]);
    assert(fr.kind == scientific::F_NONE);
    let d, fd = scientific::lil_from_dense(a2);
    assert(fd.kind == scientific::F_NONE && common_7::lil_is(d, [0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [2.0, 3.0, 4.0]));
    let cs, fc = scientific::csr_from_dense(a2);
    let d2, fe = scientific::lil_from_csr(cs);
    assert(fe.kind == scientific::F_NONE && common_7::lil_is(d2, [0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [2.0, 3.0, 4.0]));
    let co, fo = scientific::coo_from_triplets(2, 3, [1.0, 0.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 3.0]);
    assert(fo.kind == scientific::F_NONE);
    let d3, ff = scientific::lil_from_coo(co);
    assert(ff.kind == scientific::F_NONE && common_7::lil_is(d3, [0.0, 1.0], [2.0, 1.0], [2.0, 4.0]));
    // The rows of the sets in any order stand sorted.
    var s, fs = scientific::lil_new(2, 5);
    assert(fs.kind == scientific::F_NONE);
    assert(scientific::lil_set(s, 1.0, 4.0, 1.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(s, 1.0, 0.0, 2.0).kind == scientific::F_NONE);
    assert(scientific::lil_set(s, 1.0, 2.0, 3.0).kind == scientific::F_NONE);
    let rc, rv, fw = scientific::lil_row(s, 1.0);
    assert(fw.kind == scientific::F_NONE && common::all_near(rc, [0.0, 2.0, 4.0]) && common::all_near(rv, [2.0, 3.0, 1.0]));
    let ec, ev, fx = scientific::lil_row(s, 0.0);
    assert(fx.kind == scientific::F_NONE && ec.data.len == 0 && ev.data.len == 0);
}

// The pins of the block matrix and the columnwise diagonals, measured from SciPy.
fn bmat_place(g: scientific::Bmat&, i: f64, j: f64, v: f64[:], rows: i64, cols: i64) {
    let a2 = common::mx(rows, cols, v);
    let c, fc = scientific::csr_from_dense(a2);
    assert(fc.kind == scientific::F_NONE);
    assert(scientific::bmat_put(g, i, j, c).kind == scientific::F_NONE);
}

fn bmat_place_triplets(g: scientific::Bmat&, i: f64, j: f64, rows: i64, cols: i64, ri: f64[:], ci: f64[:], vv: f64[:]) {
    let c, fc = scientific::csr_from_triplets(rows, cols, ri, ci, vv);
    assert(fc.kind == scientific::F_NONE);
    assert(scientific::bmat_put(g, i, j, c).kind == scientific::F_NONE);
}

fn bmat_pin_p1() {
    var g, f0 = scientific::bmat_new(2, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0, 2.0, 0.0, 3.0], 2, 2);
    common_11::bmat_place(g, 0.0, 1.0, [4.0, 5.0], 2, 1);
    common_11::bmat_place(g, 1.0, 0.0, [6.0, 0.0], 1, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0, 0.0, 0.0]));
    assert(m.nnz == 6);
}

fn bmat_pin_p2() {
    var g, f0 = scientific::bmat_new(2, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0, 2.0, 0.0, 3.0], 2, 2);
    common_11::bmat_place(g, 1.0, 1.0, [6.0, 0.0], 1, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0]));
    assert(m.nnz == 4);
}

fn bmat_pin_p3() {
    var g, f0 = scientific::bmat_new(1, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0, 2.0, 0.0, 3.0], 2, 2);
    common_11::bmat_place(g, 0.0, 1.0, [1.0, 2.0, 0.0, 3.0], 2, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 2.0, 1.0, 2.0, 0.0, 3.0, 0.0, 3.0]));
    assert(m.nnz == 6);
}

fn bmat_pin_p4() {
    var g, f0 = scientific::bmat_new(2, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 1.0, [1.0, 2.0, 0.0, 3.0], 2, 2);
    common_11::bmat_place(g, 1.0, 0.0, [1.0, 2.0, 0.0, 3.0], 2, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]));
    assert(m.nnz == 6);
}

fn bmat_pin_p5() {
    var g, f0 = scientific::bmat_new(2, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0, 2.0, 0.0, 3.0], 2, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 2.0, 0.0, 3.0]));
    assert(m.nnz == 3);
}

fn bmat_pin_p6() {
    var g, f0 = scientific::bmat_new(1, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0, 2.0, 0.0, 3.0], 2, 2);
    common_11::bmat_place(g, 0.0, 2.0, [1.0, 2.0, 0.0, 3.0], 2, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 2.0, 1.0, 2.0, 0.0, 3.0, 0.0, 3.0]));
    assert(m.nnz == 6);
}

fn bmat_pin_p2z() {
    var g, f0 = scientific::bmat_new(2, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0, 2.0, 0.0, 3.0], 2, 2);
    common_11::bmat_place_triplets(g, 1.0, 1.0, 1, 2, [0.0, 0.0], [0.0, 1.0], [6.0, 0.0]);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0]));
    assert(m.nnz == 5);
}

fn bmat_pin_p7() {
    let a2 = common::mx(2, 4, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let d, fd = scientific::spdiags(a2, [0.0, 1.0], 3, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 6.0, 0.0, 0.0, 0.0, 2.0, 7.0, 0.0, 0.0, 0.0, 3.0, 8.0]));
    assert(d.nnz == 6);
}

fn bmat_pin_p8() {
    let a2 = common::mx(2, 4, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let d, fd = scientific::spdiags(a2, [0.0, -1.0], 4, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 0.0, 0.0, 5.0, 2.0, 0.0, 0.0, 6.0, 3.0, 0.0, 0.0, 7.0]));
    assert(d.nnz == 6);
}

fn bmat_pin_p9() {
    let a2 = common::mx(1, 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let d, fd = scientific::spdiags(a2, [0.0], 3, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0]));
    assert(d.nnz == 3);
}

fn bmat_pin_p10() {
    let a2 = common::mx(1, 4, [1.0, 2.0, 3.0, 4.0]);
    let d, fd = scientific::spdiags(a2, [-2.0], 3, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_p11() {
    let a2 = common::mx(2, 4, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let d, fd = scientific::spdiags(a2, [1.0, 0.0], 3, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [5.0, 2.0, 0.0, 0.0, 0.0, 6.0, 3.0, 0.0, 0.0, 0.0, 7.0, 4.0]));
    assert(d.nnz == 6);
}

fn bmat_pin_p12() {
    let a2 = common::mx(2, 2, [1.0, 2.0, 5.0, 6.0]);
    let d, fd = scientific::spdiags(a2, [0.0, 1.0], 3, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 6.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 6);
}

fn bmat_pin_r0() {
    var g, f0 = scientific::bmat_new(1, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 2.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 0);
}

fn bmat_pin_r1() {
    var g, f0 = scientific::bmat_new(1, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0], 1, 1);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0]));
    assert(m.nnz == 0);
}

fn bmat_pin_r2() {
    var g, f0 = scientific::bmat_new(2, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0], 1, 1);
    common_11::bmat_place(g, 1.0, 0.0, [2.0, -3.0], 2, 1);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 2.0, -3.0]));
    assert(m.nnz == 2);
}

fn bmat_pin_r3() {
    var g, f0 = scientific::bmat_new(3, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 1.0], 1, 3);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, 5.0, -3.0], 1, 3);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 2, 3);
    common_11::bmat_place(g, 1.0, 1.0, [1.0, 0.0, 5.0, -3.0, 0.0, 0.0], 2, 3);
    common_11::bmat_place(g, 2.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 2.0], 2, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 1.0, 0.0, 5.0, -3.0, 0.0, 0.0, 1.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 8);
}

fn bmat_pin_r4() {
    var g, f0 = scientific::bmat_new(3, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 2.0, [2.0, 5.0], 1, 2);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], 3, 3);
    common_11::bmat_place(g, 1.0, 1.0, [0.0, 0.0, 0.0], 3, 1);
    common_11::bmat_place(g, 1.0, 2.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3, 2);
    common_11::bmat_place(g, 2.0, 0.0, [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3, 3);
    common_11::bmat_place(g, 2.0, 1.0, [0.0, 0.0, -3.0], 3, 1);
    common_11::bmat_place(g, 2.0, 2.0, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -3.0, 0.0, 1.0]));
    assert(m.nnz == 9);
}

fn bmat_pin_r5() {
    var g, f0 = scientific::bmat_new(2, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0, 5.0, 0.0, 0.0, 0.0, 0.0], 2, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 5.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 2);
}

fn bmat_pin_r6() {
    var g, f0 = scientific::bmat_new(2, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, -3.0, 5.0], 1, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, -3.0, 5.0]));
    assert(m.nnz == 2);
}

fn bmat_pin_r7() {
    var g, f0 = scientific::bmat_new(1, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 5.0, 5.0], 3, 1);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, -3.0, 0.0, 0.0, 0.0, 0.0], 3, 2);
    common_11::bmat_place(g, 0.0, 2.0, [0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 3, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, -3.0, 0.0, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 4);
}

fn bmat_pin_r8() {
    var g, f0 = scientific::bmat_new(3, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 5.0], 1, 2);
    common_11::bmat_place(g, 0.0, 1.0, [1.0], 1, 1);
    common_11::bmat_place(g, 0.0, 2.0, [2.0, 0.0], 1, 2);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0], 1, 2);
    common_11::bmat_place(g, 1.0, 2.0, [2.0, 5.0], 1, 2);
    common_11::bmat_place(g, 2.0, 0.0, [-3.0, 0.0, 0.0, 2.0], 2, 2);
    common_11::bmat_place(g, 2.0, 2.0, [1.0, 2.0, 0.0, 0.0], 2, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 5.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, -3.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 9);
}

fn bmat_pin_r9() {
    var g, f0 = scientific::bmat_new(1, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 2, 2);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, 1.0, 0.0, 1.0], 2, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
    assert(m.nnz == 2);
}

fn bmat_pin_r10() {
    var g, f0 = scientific::bmat_new(2, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 5.0, 0.0, 1.0, 0.0], 2, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 5.0, 0.0, 1.0, 0.0]));
    assert(m.nnz == 2);
}

fn bmat_pin_r11() {
    var g, f0 = scientific::bmat_new(3, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 2.0, 0.0, [5.0, 5.0], 2, 1);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [5.0, 5.0]));
    assert(m.nnz == 2);
}

fn bmat_pin_r12() {
    var g, f0 = scientific::bmat_new(2, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 1.0, [5.0, 0.0], 1, 2);
    common_11::bmat_place(g, 1.0, 0.0, [2.0, 5.0], 1, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 5.0, 0.0, 2.0, 5.0, 0.0, 0.0]));
    assert(m.nnz == 3);
}

fn bmat_pin_r13() {
    var g, f0 = scientific::bmat_new(2, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 0);
}

fn bmat_pin_r14() {
    var g, f0 = scientific::bmat_new(3, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 2.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 1.0, -3.0], 1, 3);
    common_11::bmat_place(g, 1.0, 1.0, [-3.0], 1, 1);
    common_11::bmat_place(g, 1.0, 2.0, [0.0, 0.0, 0.0], 1, 3);
    common_11::bmat_place(g, 2.0, 0.0, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -3.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 6);
}

fn bmat_pin_r15() {
    var g, f0 = scientific::bmat_new(3, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 0.0], 3, 1);
    common_11::bmat_place(g, 1.0, 0.0, [5.0], 1, 1);
    common_11::bmat_place(g, 1.0, 1.0, [2.0], 1, 1);
    common_11::bmat_place(g, 2.0, 0.0, [1.0], 1, 1);
    common_11::bmat_place(g, 2.0, 1.0, [0.0], 1, 1);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0, 1.0, 0.0]));
    assert(m.nnz == 3);
}

fn bmat_pin_r16() {
    var g, f0 = scientific::bmat_new(2, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0, 0.0, 0.0, 0.0, 5.0, 5.0], 2, 3);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 3);
}

fn bmat_pin_r17() {
    var g, f0 = scientific::bmat_new(2, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0], 1, 1);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0]));
    assert(m.nnz == 1);
}

fn bmat_pin_r18() {
    var g, f0 = scientific::bmat_new(3, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 1.0, 0.0], 1, 3);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, 2.0], 1, 2);
    common_11::bmat_place(g, 0.0, 2.0, [-3.0, 1.0, -3.0], 1, 3);
    common_11::bmat_place(g, 1.0, 0.0, [5.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 3);
    common_11::bmat_place(g, 1.0, 1.0, [0.0, 0.0, 0.0, 5.0], 2, 2);
    common_11::bmat_place(g, 1.0, 2.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 3);
    common_11::bmat_place(g, 2.0, 0.0, [5.0, 0.0, 0.0, -3.0, 0.0, 0.0], 2, 3);
    common_11::bmat_place(g, 2.0, 2.0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 2, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 1.0, 0.0, 0.0, 2.0, -3.0, 1.0, -3.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 10);
}

fn bmat_pin_r19() {
    var g, f0 = scientific::bmat_new(1, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 1.0], 1, 2);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, 0.0, 0.0], 1, 3);
    common_11::bmat_place(g, 0.0, 2.0, [0.0, -3.0, 5.0], 1, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -3.0, 5.0]));
    assert(m.nnz == 3);
}

fn bmat_pin_r20() {
    var g, f0 = scientific::bmat_new(3, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0, 1.0, 2.0, 1.0, 2.0], 2, 3);
    common_11::bmat_place(g, 2.0, 0.0, [0.0, 2.0, 0.0, 0.0, -3.0, 0.0], 2, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, -3.0, 0.0]));
    assert(m.nnz == 6);
}

fn bmat_pin_r21() {
    var g, f0 = scientific::bmat_new(3, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 2, 2);
    common_11::bmat_place(g, 0.0, 2.0, [5.0, 0.0, 2.0, 0.0, 0.0, 0.0], 2, 3);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0, 2.0, 2.0], 2, 2);
    common_11::bmat_place(g, 1.0, 1.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 3);
    common_11::bmat_place(g, 1.0, 2.0, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3);
    common_11::bmat_place(g, 2.0, 0.0, [5.0, 0.0], 1, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 6);
}

fn bmat_pin_r22() {
    var g, f0 = scientific::bmat_new(3, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [5.0, 0.0], 1, 2);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0, 0.0, 0.0], 2, 2);
    common_11::bmat_place(g, 2.0, 0.0, [-3.0, 0.0, -3.0, 0.0, 0.0, 0.0], 3, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, -3.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 3);
}

fn bmat_pin_r23() {
    var g, f0 = scientific::bmat_new(1, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, 2.0, 0.0, 1.0, 0.0, 0.0], 2, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 2.0, 0.0, 1.0, 0.0, 0.0]));
    assert(m.nnz == 2);
}

fn bmat_pin_r24() {
    var g, f0 = scientific::bmat_new(1, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 2, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]));
    assert(m.nnz == 1);
}

fn bmat_pin_r25() {
    var g, f0 = scientific::bmat_new(2, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [5.0, 0.0], 1, 2);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, 5.0], 1, 2);
    common_11::bmat_place(g, 1.0, 0.0, [5.0, -3.0, 0.0, 0.0, 0.0, 0.0], 3, 2);
    common_11::bmat_place(g, 1.0, 1.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [5.0, 0.0, 0.0, 5.0, 5.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 4);
}

fn bmat_pin_r26() {
    var g, f0 = scientific::bmat_new(1, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 0.0], 1, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0]));
    assert(m.nnz == 0);
}
