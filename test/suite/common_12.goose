// The helpers that the tests of more than one group share.
namespace common_12;

import scientific from "../..";
import os;
import common;
import common_11;

fn bmat_pin_r27() {
    var g, f0 = scientific::bmat_new(1, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [5.0, 0.0, 2.0], 3, 1);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [5.0, 0.0, 2.0]));
    assert(m.nnz == 2);
}

fn bmat_pin_r28() {
    var g, f0 = scientific::bmat_new(1, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [-3.0, -3.0, 0.0], 3, 1);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [-3.0, -3.0, 0.0]));
    assert(m.nnz == 2);
}

fn bmat_pin_r29() {
    var g, f0 = scientific::bmat_new(1, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [2.0, 0.0, 0.0], 3, 1);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, 0.0, 0.0], 3, 1);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 1);
}

fn bmat_pin_r30() {
    var g, f0 = scientific::bmat_new(1, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0], 1, 1);
    common_11::bmat_place(g, 0.0, 1.0, [1.0], 1, 1);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 1.0]));
    assert(m.nnz == 2);
}

fn bmat_pin_r31() {
    var g, f0 = scientific::bmat_new(2, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 2, 2);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0], 1, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 0);
}

fn bmat_pin_r32() {
    var g, f0 = scientific::bmat_new(3, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 1.0, 1.0, [0.0, 0.0, 0.0, 0.0, 1.0, 2.0], 3, 2);
    common_11::bmat_place(g, 2.0, 0.0, [0.0, 0.0, 0.0], 1, 3);
    common_11::bmat_place(g, 2.0, 1.0, [2.0, 0.0], 1, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0]));
    assert(m.nnz == 3);
}

fn bmat_pin_r33() {
    var g, f0 = scientific::bmat_new(2, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 0);
}

fn bmat_pin_r34() {
    var g, f0 = scientific::bmat_new(3, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0], 2, 2);
    common_11::bmat_place(g, 1.0, 0.0, [2.0, 0.0, 2.0, 0.0], 2, 2);
    common_11::bmat_place(g, 2.0, 0.0, [1.0, 0.0], 1, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0]));
    assert(m.nnz == 3);
}

fn bmat_pin_r35() {
    var g, f0 = scientific::bmat_new(1, 3);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [1.0], 1, 1);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0]));
    assert(m.nnz == 1);
}

fn bmat_pin_r36() {
    var g, f0 = scientific::bmat_new(2, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 1.0, 1.0, [2.0, 0.0, 0.0, 2.0, 0.0, -3.0], 2, 3);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [2.0, 0.0, 0.0, 2.0, 0.0, -3.0]));
    assert(m.nnz == 3);
}

fn bmat_pin_r37() {
    var g, f0 = scientific::bmat_new(2, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, 0.0, 0.0, 0.0], 2, 2);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0, 0.0, -3.0, 0.0, 2.0], 2, 3);
    common_11::bmat_place(g, 1.0, 1.0, [0.0, 0.0, 0.0, 5.0], 2, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 2.0, 0.0, 5.0]));
    assert(m.nnz == 3);
}

fn bmat_pin_r38() {
    var g, f0 = scientific::bmat_new(3, 1);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [5.0, 0.0], 1, 2);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(m.nnz == 1);
}

fn bmat_pin_r39() {
    var g, f0 = scientific::bmat_new(3, 2);
    assert(f0.kind == scientific::F_NONE);
    common_11::bmat_place(g, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3, 3);
    common_11::bmat_place(g, 0.0, 1.0, [0.0, 2.0, 0.0, 5.0, 5.0, 0.0], 3, 2);
    common_11::bmat_place(g, 1.0, 0.0, [0.0, 0.0, 0.0], 1, 3);
    common_11::bmat_place(g, 1.0, 1.0, [0.0, -3.0], 1, 2);
    common_11::bmat_place(g, 2.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 3);
    common_11::bmat_place(g, 2.0, 1.0, [0.0, 0.0, 0.0, 1.0], 2, 2);
    let m, fm = scientific::csr_bmat(g);
    assert(fm.kind == scientific::F_NONE);
    let dn, fz = scientific::csr_to_dense(m);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]));
    assert(m.nnz == 5);
}

fn bmat_pin_s0() {
    let a2 = common::mx(2, 4, [-3.0, -3.0, 1.0, 5.0, 2.0, 5.0, 2.0, 2.0]);
    let d, fd = scientific::spdiags(a2, [-4.0, -1.0], 5, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.0, -3.0, 0.0, 0.0]));
    assert(d.nnz == 4);
}

fn bmat_pin_s1() {
    let a2 = common::mx(3, 1, [2.0, 5.0, 5.0]);
    let d, fd = scientific::spdiags(a2, [-2.0, -1.0, 1.0], 3, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0, 0.0]));
    assert(d.nnz == 5);
}

fn bmat_pin_s2() {
    let a2 = common::mx(1, 1, [2.0]);
    let d, fd = scientific::spdiags(a2, [1.0], 2, 2);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s3() {
    let a2 = common::mx(1, 1, [5.0]);
    let d, fd = scientific::spdiags(a2, [0.0], 1, 1);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [5.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s4() {
    let a2 = common::mx(1, 4, [2.0, 0.0, 0.0, 5.0]);
    let d, fd = scientific::spdiags(a2, [3.0], 3, 5);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 2);
}

fn bmat_pin_s5() {
    let a2 = common::mx(2, 4, [-3.0, 1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 0.0]);
    let d, fd = scientific::spdiags(a2, [0.0, 1.0], 2, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [-3.0, 5.0, 0.0, 0.0, 1.0, 5.0]));
    assert(d.nnz == 4);
}

fn bmat_pin_s6() {
    let a2 = common::mx(2, 1, [5.0, -3.0]);
    let d, fd = scientific::spdiags(a2, [-2.0, -1.0], 3, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 3);
}

fn bmat_pin_s7() {
    let a2 = common::mx(3, 6, [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 5.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
    let d, fd = scientific::spdiags(a2, [0.0, 1.0, 4.0], 2, 5);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0]));
    assert(d.nnz == 5);
}

fn bmat_pin_s8() {
    let a2 = common::mx(1, 2, [-3.0, -3.0]);
    let d, fd = scientific::spdiags(a2, [-3.0], 4, 1);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, -3.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s9() {
    let a2 = common::mx(3, 1, [-3.0, 2.0, -3.0]);
    let d, fd = scientific::spdiags(a2, [-4.0, -1.0, 0.0], 5, 1);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [-3.0, 2.0, 0.0, 0.0, -3.0]));
    assert(d.nnz == 3);
}

fn bmat_pin_s10() {
    let a2 = common::mx(2, 2, [2.0, -3.0, 2.0, -3.0]);
    let d, fd = scientific::spdiags(a2, [-4.0, -1.0], 5, 1);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 2.0, 0.0, 0.0, 2.0]));
    assert(d.nnz == 2);
}

fn bmat_pin_s11() {
    let a2 = common::mx(1, 1, [0.0]);
    let d, fd = scientific::spdiags(a2, [-1.0], 4, 1);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s12() {
    let a2 = common::mx(1, 6, [5.0, 5.0, 1.0, 1.0, 1.0, 5.0]);
    let d, fd = scientific::spdiags(a2, [-1.0], 3, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0]));
    assert(d.nnz == 2);
}

fn bmat_pin_s13() {
    let a2 = common::mx(1, 3, [5.0, 5.0, 2.0]);
    let d, fd = scientific::spdiags(a2, [0.0], 1, 2);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [5.0, 0.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s14() {
    let a2 = common::mx(2, 2, [1.0, 2.0, 1.0, 2.0]);
    let d, fd = scientific::spdiags(a2, [-1.0, 2.0], 5, 5);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 7);
}

fn bmat_pin_s15() {
    let a2 = common::mx(3, 5, [-3.0, -3.0, 2.0, 0.0, 2.0, -3.0, 2.0, -3.0, 5.0, 0.0, -3.0, 5.0, 0.0, -3.0, 0.0]);
    let d, fd = scientific::spdiags(a2, [-1.0, 1.0, 2.0], 5, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 2.0, 0.0, -3.0, 0.0, -3.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 6);
}

fn bmat_pin_s16() {
    let a2 = common::mx(1, 1, [-3.0]);
    let d, fd = scientific::spdiags(a2, [0.0], 1, 1);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [-3.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s17() {
    let a2 = common::mx(2, 2, [5.0, 2.0, 2.0, 2.0]);
    let d, fd = scientific::spdiags(a2, [-1.0, 3.0], 3, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]));
    assert(d.nnz == 3);
}

fn bmat_pin_s18() {
    let a2 = common::mx(2, 4, [5.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0]);
    let d, fd = scientific::spdiags(a2, [0.0, 2.0], 4, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 4);
}

fn bmat_pin_s19() {
    let a2 = common::mx(2, 1, [2.0, 2.0]);
    let d, fd = scientific::spdiags(a2, [0.0, 1.0], 1, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [2.0, 0.0, 0.0]));
    assert(d.nnz == 2);
}

fn bmat_pin_s20() {
    let a2 = common::mx(1, 7, [-3.0, 2.0, 0.0, -3.0, 2.0, 0.0, 1.0]);
    let d, fd = scientific::spdiags(a2, [0.0], 1, 5);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [-3.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s21() {
    let a2 = common::mx(2, 3, [-3.0, 5.0, -3.0, -3.0, 0.0, 1.0]);
    let d, fd = scientific::spdiags(a2, [-1.0, 1.0], 4, 2);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, -3.0, 0.0, 0.0, 5.0, 0.0, 0.0]));
    assert(d.nnz == 3);
}

fn bmat_pin_s22() {
    let a2 = common::mx(1, 3, [0.0, 2.0, 1.0]);
    let d, fd = scientific::spdiags(a2, [-1.0], 3, 1);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s23() {
    let a2 = common::mx(1, 3, [5.0, 0.0, 5.0]);
    let d, fd = scientific::spdiags(a2, [-4.0], 5, 2);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s24() {
    let a2 = common::mx(2, 5, [2.0, 0.0, 1.0, 2.0, 2.0, -3.0, 2.0, 0.0, 0.0, 5.0]);
    let d, fd = scientific::spdiags(a2, [-3.0, -2.0], 4, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 2.0, 2.0, 0.0]));
    assert(d.nnz == 3);
}

fn bmat_pin_s25() {
    let a2 = common::mx(1, 2, [1.0, 0.0]);
    let d, fd = scientific::spdiags(a2, [-2.0], 4, 1);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 1.0, 0.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s26() {
    let a2 = common::mx(1, 1, [-3.0]);
    let d, fd = scientific::spdiags(a2, [2.0], 2, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 2);
}

fn bmat_pin_s27() {
    let a2 = common::mx(2, 3, [-3.0, 2.0, 0.0, 2.0, 1.0, 0.0]);
    let d, fd = scientific::spdiags(a2, [1.0, 2.0], 1, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 2.0, 0.0]));
    assert(d.nnz == 2);
}

fn bmat_pin_s28() {
    let a2 = common::mx(3, 5, [1.0, -3.0, 5.0, 1.0, 0.0, -3.0, 0.0, 5.0, 1.0, 0.0, 2.0, 2.0, 1.0, -3.0, -3.0]);
    let d, fd = scientific::spdiags(a2, [-2.0, 0.0, 2.0], 3, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [-3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -3.0, 1.0, 0.0, 5.0, 0.0]));
    assert(d.nnz == 6);
}

fn bmat_pin_s29() {
    let a2 = common::mx(3, 3, [1.0, 1.0, 0.0, -3.0, 0.0, 5.0, 0.0, -3.0, 1.0]);
    let d, fd = scientific::spdiags(a2, [-2.0, -1.0, 0.0], 4, 2);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, -3.0, -3.0, 1.0, 0.0, 0.0, 1.0]));
    assert(d.nnz == 6);
}

fn bmat_pin_s30() {
    let a2 = common::mx(3, 5, [-3.0, 2.0, -3.0, 1.0, 0.0, 5.0, 2.0, 5.0, -3.0, 0.0, 5.0, 5.0, 2.0, 1.0, 5.0]);
    let d, fd = scientific::spdiags(a2, [-2.0, 0.0, 3.0], 4, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [5.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, -3.0, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0, -3.0]));
    assert(d.nnz == 7);
}

fn bmat_pin_s31() {
    let a2 = common::mx(2, 2, [-3.0, -3.0, 5.0, 5.0]);
    let d, fd = scientific::spdiags(a2, [-1.0, 1.0], 4, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 5.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 6);
}

fn bmat_pin_s32() {
    let a2 = common::mx(2, 4, [2.0, 1.0, 1.0, 2.0, 0.0, 1.0, 5.0, 0.0]);
    let d, fd = scientific::spdiags(a2, [-2.0, -1.0], 4, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 5.0]));
    assert(d.nnz == 5);
}

fn bmat_pin_s33() {
    let a2 = common::mx(2, 4, [2.0, 0.0, 2.0, 1.0, 1.0, 1.0, 5.0, 1.0]);
    let d, fd = scientific::spdiags(a2, [0.0, 1.0], 1, 2);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [2.0, 1.0]));
    assert(d.nnz == 2);
}

fn bmat_pin_s34() {
    let a2 = common::mx(1, 1, [0.0]);
    let d, fd = scientific::spdiags(a2, [-4.0], 5, 3);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s35() {
    let a2 = common::mx(1, 5, [5.0, 5.0, 0.0, 5.0, 2.0]);
    let d, fd = scientific::spdiags(a2, [-2.0], 4, 5);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 2);
}

fn bmat_pin_s36() {
    let a2 = common::mx(1, 2, [2.0, 2.0]);
    let d, fd = scientific::spdiags(a2, [1.0], 2, 4);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 2);
}

fn bmat_pin_s37() {
    let a2 = common::mx(1, 1, [1.0]);
    let d, fd = scientific::spdiags(a2, [4.0], 3, 5);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 1);
}

fn bmat_pin_s38() {
    let a2 = common::mx(3, 5, [0.0, 0.0, 0.0, -3.0, 5.0, -3.0, 5.0, 1.0, 2.0, -3.0, 1.0, 1.0, 1.0, -3.0, 1.0]);
    let d, fd = scientific::spdiags(a2, [-4.0, -2.0, 3.0], 5, 5);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
    assert(d.nnz == 6);
}

fn bmat_pin_s39() {
    let a2 = common::mx(1, 1, [5.0]);
    let d, fd = scientific::spdiags(a2, [3.0], 3, 5);
    assert(fd.kind == scientific::F_NONE);
    let dn, fz = scientific::dia_to_dense(d);
    assert(fz.kind == scientific::F_NONE && common::all_near(dn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(d.nnz == 2);
}

// Pins of the sparse LU factor against SciPy 1.18.1.
fn lu_fail(why: const u8[:]) {
    print(why);
    print("\n");
    assert(false);
}

// lu_pin factors the matrix, solves for b, compares the solution and checks
// the identity L U = A[perm_r][:, perm_c].
fn lu_pin(n: i64, ri: f64[:], ci: f64[:], v: f64[:], b: f64[:], x: f64[:]) {
    let a, fa = scientific::csr_from_triplets(n, n, ri, ci, v);
    assert(fa.kind == scientific::F_NONE);
    let f, ff = scientific::sparse_lu(a);
    assert(ff.kind == scientific::F_NONE);
    let got, fg = scientific::splu_solve(f, b);
    assert(fg.kind == scientific::F_NONE && common::all_near(got, x));
    common_12::lu_identity(f, a, n);
}

fn lu_pin_inverse(n: i64, ri: f64[:], ci: f64[:], v: f64[:], inv: f64[:]) {
    let a, fa = scientific::csr_from_triplets(n, n, ri, ci, v);
    assert(fa.kind == scientific::F_NONE);
    let f, ff = scientific::sparse_lu(a);
    assert(ff.kind == scientific::F_NONE);
    let id, fi = scientific::eye(n);
    assert(fi.kind == scientific::F_NONE);
    let got, fg = scientific::splu_solve_matrix(f, id);
    assert(fg.kind == scientific::F_NONE && common::all_near(got, inv));
    common_12::lu_identity(f, a, n);
}

// lu_identity checks the shape of L and U, the count and the identity
// L U = A[perm_r][:, perm_c].
fn lu_identity(f: scientific::Splu&, a: scientific::Csr&, n: i64) {
    let l, fl = scientific::splu_l(f);
    let u, fu = scientific::splu_u(f);
    let pr, fpr = scientific::splu_perm_r(f);
    let pc, fpc = scientific::splu_perm_c(f);
    assert(fl.kind == scientific::F_NONE && fu.kind == scientific::F_NONE);
    assert(fpr.kind == scientific::F_NONE && fpc.kind == scientific::F_NONE);
    assert(l.nnz + u.nnz == scientific::splu_nnz(f));
    let prod, fp = scientific::csr_matmul(l, u);
    assert(fp.kind == scientific::F_NONE);
    let pd, f0 = scientific::csr_to_dense(prod);
    let ld, f1 = scientific::csr_to_dense(l);
    let ud, f2 = scientific::csr_to_dense(u);
    let ad, f3 = scientific::csr_to_dense(a);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE && f2.kind == scientific::F_NONE && f3.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < n {
        var j: i64 = 0;
        while j < n {
            let s = pd.data[i * n + j];
            let want = ad.data[(pr.data[i] as i64) * n + (pc.data[j] as i64)];
            if s - want > 1.0e-10 || want - s > 1.0e-10 { common_12::lu_fail("L U differs from the permuted matrix"); }
            if j > i && ld.data[i * n + j] != 0.0 { common_12::lu_fail("L is not lower"); }
            if j < i && ud.data[i * n + j] != 0.0 { common_12::lu_fail("U is not upper"); }
            if j == i && ld.data[i * n + j] != 1.0 { common_12::lu_fail("L has no unit diagonal"); }
            j += 1;
        }
        i += 1;
    }
}

// lu_matrix gives the matrix [[4, 3], [6, 3]].
fn lu_matrix() -> scientific::Csr {
    let a, fa = scientific::csr_from_triplets(2, 2, [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [4.0, 3.0, 6.0, 3.0]);
    assert(fa.kind == scientific::F_NONE);
    return a;
}

fn lu_good() -> scientific::Splu {
    let a = common_12::lu_matrix();
    let f, ff = scientific::sparse_lu(a);
    assert(ff.kind == scientific::F_NONE && f.buf.len == 18);
    return f;
}

// lu_refuses checks that every call that reads a factor gives a fault
// of kind F_SHAPE and an empty result.
fn lu_refuses(f: scientific::Splu&) {
    let x, f1 = scientific::splu_solve(f, [1.0, 2.0]);
    assert(f1.kind == scientific::F_SHAPE && x.data.len == 0);
    let y, f2 = scientific::splu_solve_matrix(f, common::mx(2, 1, [1.0, 2.0]));
    assert(f2.kind == scientific::F_SHAPE && y.data.len == 0);
    let l, f3 = scientific::splu_l(f);
    assert(f3.kind == scientific::F_SHAPE && l.nnz == 0);
    let u, f4 = scientific::splu_u(f);
    assert(f4.kind == scientific::F_SHAPE && u.nnz == 0);
    let pr, f5 = scientific::splu_perm_r(f);
    assert(f5.kind == scientific::F_SHAPE && pr.data.len == 0);
    let pc, f6 = scientific::splu_perm_c(f);
    assert(f6.kind == scientific::F_SHAPE && pc.data.len == 0);
    assert(scientific::splu_nnz(f) == -1);
}

fn lu_forge_at(at: i64, v: f64) {
    var f = common_12::lu_good();
    f.buf[at] = v;
    common_12::lu_refuses(f);
}

fn lu_forge_counts(n: i64, nl: i64, nu: i64) {
    var f = common_12::lu_good();
    f.n = n;
    f.nnz_l = nl;
    f.nnz_u = nu;
    common_12::lu_refuses(f);
}

fn lu_singular(rows: i64, ri: f64[:], ci: f64[:], v: f64[:]) {
    let a, fa = scientific::csr_from_triplets(rows, rows, ri, ci, v);
    assert(fa.kind == scientific::F_NONE);
    let f, ff = scientific::sparse_lu(a);
    assert(ff.kind == scientific::F_VALUE && f.n == 0 && f.buf.len == 0);
    let x, fx = scientific::spsolve(a, [1.0, 1.0]);
    assert(fx.kind == scientific::F_VALUE && x.data.len == 0);
}

// lu_limited factors [[4, 3], [6, 3]] with lowered limits and checks the
// fault kind. A fault gives an empty factor.
fn lu_limited(max_nnz: i64, max_scratch: i64, kind: i32) {
    let a = common_12::lu_matrix();
    let f, ff = scientific::lu_factor_limited(a, max_nnz, max_scratch);
    assert(ff.kind == kind);
    if kind != scientific::F_NONE { assert(f.n == 0 && f.buf.len == 0); }
    else { assert(f.n == 2 && f.buf.len == 18); }
}

// lu_forged_input gives sparse_lu a matrix whose layout lies.
fn lu_forged_input(at: i64, v: f64, count: i64) {
    var a = common_12::lu_matrix();
    if at >= 0 { a.buf[at] = v; }
    if count >= 0 { a.nnz = count; }
    let f, ff = scientific::sparse_lu(a);
    assert(ff.kind == scientific::F_SHAPE && f.n == 0 && f.buf.len == 0);
}

// expm_csr builds a matrix of compressed rows from entries.
fn expm_csr(n: i64, ri: f64[:], ci: f64[:], v: f64[:]) -> scientific::Csr {
    let a, f = scientific::csr_from_triplets(n, n, ri, ci, v);
    assert(f.kind == scientific::F_NONE);
    return a;
}

// expm_est_pin checks the estimate of the one-norm against the exact one.
// The estimate never passes the exact one-norm. Two calls give one value.
fn expm_est_pin(n: i64, ri: f64[:], ci: f64[:], v: f64[:], exact: f64) {
    let a = common_12::expm_csr(n, ri, ci, v);
    let e1, v1, w1, f1 = scientific::onenormest(a, 2, 5, true, true);
    let e2, v2, w2, f2 = scientific::onenormest(a, 2, 5, false, false);
    assert(f1.kind == scientific::F_NONE && f2.kind == scientific::F_NONE);
    // The estimate never passes the exact one-norm. A side of four or
    // less gives it exactly.
    assert(e1 == e2 && e1 <= exact + 0.000000000001);
    if n <= 4 { assert(common::near(e1, exact)); }
    assert(v2.data.len == 0 && w2.data.len == 0);
    assert(v1.data.len == n && w1.data.len == n);
    var nv: f64 = 0.0;
    var nw: f64 = 0.0;
    var i: i64 = 0;
    while i < n { nv += common::mathshim_abs(v1.data[i]); nw += common::mathshim_abs(w1.data[i]); i += 1; }
    assert(common::near(nv, 1.0) && common::near(nw, e1));
}

// expm_est_vec_pin checks the vector of the estimate and its product.
fn expm_est_vec_pin(n: i64, ri: f64[:], ci: f64[:], v: f64[:], vec: f64[:], prod: f64[:]) {
    let a = common_12::expm_csr(n, ri, ci, v);
    let e, vv, ww, f = scientific::onenormest(a, 2, 5, true, true);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(vv, vec, 0.0000000001) && common::all_close(ww, prod, 0.0000000001));
}

// expm_right builds the right side. One column gives a vector.
fn expm_right(b: f64[:], n: i64) -> scientific::Arr {
    if b.len == n { return common::av(b); }
    return common::mat(b, n, b.len / n);
}

// expm_pin checks the product of the exponential against SciPy.
fn expm_pin(n: i64, ri: f64[:], ci: f64[:], v: f64[:], b: f64[:], x: f64[:]) {
    let a = common_12::expm_csr(n, ri, ci, v);
    let got, f = scientific::expm_multiply(a, common_12::expm_right(b, n));
    assert(f.kind == scientific::F_NONE && got.data.len == x.len);
    var i: i64 = 0;
    while i < x.len {
        assert(common::mathshim_abs(got.data[i] - x[i]) <= 0.0000000001 * (1.0 + common::mathshim_abs(x[i])));
        i += 1;
    }
}

// expm_range_pin checks the rows of the range form against SciPy.
fn expm_range_pin(n: i64, ri: f64[:], ci: f64[:], v: f64[:], b: f64[:], x: f64[:], start: f64, stop: f64, num: i64, endpoint: bool) {
    let a = common_12::expm_csr(n, ri, ci, v);
    let got, f = scientific::expm_multiply_range(a, common_12::expm_right(b, n), start, stop, num, endpoint);
    assert(f.kind == scientific::F_NONE && got.data.len == x.len);
    assert(got.ndim == 2 && got.shape[0] == num && got.shape[1] == n);
    var i: i64 = 0;
    while i < x.len {
        assert(common::mathshim_abs(got.data[i] - x[i]) <= 0.0000000001 * (1.0 + common::mathshim_abs(x[i])));
        i += 1;
    }
}

fn expm_kind_is(f: scientific::Fault&, kind: i32) -> bool {
    return f.kind == kind;
}

// lg_tri builds the tridiagonal matrix of side n with 4 on the diagonal and
// -1 beside it.
fn lg_tri(n: i64) -> scientific::Csr {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        ri.push(i as f64); ci.push(i as f64); v.push(4.0);
        if i + 1 < n {
            ri.push(i as f64); ci.push((i + 1) as f64); v.push(-1.0);
            ri.push((i + 1) as f64); ci.push(i as f64); v.push(-1.0);
        }
        i += 1;
    }
    return common_12::expm_csr(n, ri, ci, v);
}

// lg_diag builds the diagonal matrix of side n with the value d.
fn lg_diag(n: i64, d: f64) -> scientific::Csr {
    var ri: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < n { ri.push(i as f64); v.push(d); i += 1; }
    return common_12::expm_csr(n, ri, ri, v);
}

fn lg_ones(n: i64) -> f64[>..] {
    var b: f64[>..] = [];
    var i: i64 = 0;
    while i < n { b.push(1.0); i += 1; }
    return b;
}

// lg_rel gives the squared relative residual |b - A x|^2 / |b|^2. Both sums
// use a scale, so a right side of any finite size gives a finite value.
fn lg_rel(a: scientific::Csr&, b: f64[:], x: f64[:]) -> f64 {
    var sc: f64 = 0.0;
    var i: i64 = 0;
    while i < b.len { if common::mathshim_abs(b[i]) > sc { sc = common::mathshim_abs(b[i]); } i += 1; }
    let ax, f = scientific::csr_matvec(a, x);
    assert(f.kind == scientific::F_NONE);
    var num: f64 = 0.0;
    var den: f64 = 0.0;
    i = 0;
    while i < b.len {
        let r = (b[i] - ax.data[i]) / sc;
        let q = b[i] / sc;
        num += r * r;
        den += q * q;
        i += 1;
    }
    return num / den;
}

// lg_match checks a solution against the pinned one to 1e-8 and a count of
// zero.
fn lg_match(got: f64[:], c: i64, f: scientific::Fault, x: f64[:]) {
    assert(f.kind == scientific::F_NONE && c == 0 && got.len == x.len);
    var i: i64 = 0;
    while i < x.len {
        assert(common::mathshim_abs(got[i] - x[i]) <= 0.00000001 * (1.0 + common::mathshim_abs(x[i])));
        i += 1;
    }
}

// lg_pin checks both calls against the SciPy solution at a tight tolerance.
fn lg_pin(n: i64, ri: f64[:], ci: f64[:], v: f64[:], b: f64[:], x: f64[:]) {
    let a = common_12::expm_csr(n, ri, ci, v);
    var e0: f64[>..] = [];
    let s1, c1, f1 = scientific::lgmres(a, b, e0, 1.0e-12);
    common_12::lg_match(s1.data, c1, f1, x);
    let s2, c2, f2 = scientific::gcrotmk(a, b, e0, 1.0e-12);
    common_12::lg_match(s2.data, c2, f2, x);
    assert(common_12::lg_rel(a, b, s1.data) <= 1.0e-20 && common_12::lg_rel(a, b, s2.data) <= 1.0e-20);
}

// lg_pin_opts runs both calls with a small inner size and the option
// combinations, so the kept vectors take part in the restarts.
fn lg_pin_opts(n: i64, ri: f64[:], ci: f64[:], v: f64[:], b: f64[:], x: f64[:]) {
    let a = common_12::expm_csr(n, ri, ci, v);
    var e0: f64[>..] = [];
    var store: i64 = 0;
    while store < 2 {
        var pre: i64 = 0;
        while pre < 2 {
            let s1, c1, f1 = scientific::lgmres(a, b, e0, 1.0e-12, 0.0, 1000, 3, 2, store == 0, pre == 1);
            common_12::lg_match(s1.data, c1, f1, x);
            pre += 1;
        }
        store += 1;
    }
    let s2, c2, f2 = scientific::lgmres(a, b, e0, 1.0e-12, 0.0, 1000, 3, 0);
    common_12::lg_match(s2.data, c2, f2, x);
    let s3, c3, f3 = scientific::lgmres(a, b, e0, 1.0e-12, 0.0, 1000, 2, 5);
    common_12::lg_match(s3.data, c3, f3, x);
    var disc: i64 = 0;
    while disc < 2 {
        var old: i64 = 0;
        while old < 2 {
            let s4, c4, f4 = scientific::gcrotmk(a, b, e0, 1.0e-12, 0.0, 1000, 3, 2, disc == 1, old == 0);
            common_12::lg_match(s4.data, c4, f4, x);
            old += 1;
        }
        disc += 1;
    }
    let s5, c5, f5 = scientific::gcrotmk(a, b, e0, 1.0e-12, 0.0, 1000, 3, 0);
    common_12::lg_match(s5.data, c5, f5, x);
    let s6, c6, f6 = scientific::gcrotmk(a, b, e0, 1.0e-12, 0.0, 1000, 4, 6, false, false);
    common_12::lg_match(s6.data, c6, f6, x);
    let s7, c7, f7 = scientific::gcrotmk(a, b, e0, 1.0e-12, 0.0, 1000, 2, 1, false, false);
    common_12::lg_match(s7.data, c7, f7, x);
}

// lg_fault_both gives the same operands to both calls. Each call gives the
// fault kind, an empty result and a count of zero.
fn lg_fault_both(a: scientific::Csr&, b: f64[:], x0: f64[:], rtol: f64, atol: f64, maxiter: i64, m: i64, k: i64, want: i32) {
    let s1, c1, f1 = scientific::lgmres(a, b, x0, rtol, atol, maxiter, m, k);
    assert(f1.kind == want && s1.data.len == 0 && c1 == 0);
    let s2, c2, f2 = scientific::gcrotmk(a, b, x0, rtol, atol, maxiter, m, k);
    assert(f2.kind == want && s2.data.len == 0 && c2 == 0);
}

fn lg_sol6() -> f64[>..] {
    var x: f64[>..] = [0.3658536585365853, 0.46341463414634143, 0.48780487804878037,
                       0.48780487804878037, 0.46341463414634143, 0.3658536585365853];
    return x;
}
