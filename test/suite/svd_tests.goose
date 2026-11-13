// The tests of the group svd. Run them with SCIENTIFIC_SUITES=svd.
namespace svd_tests;

import scientific from "../..";
import os;
import common;

// The singular values, the least squares and the polynomials.
fn test_svd_and_poly() {
    // The singular values of a matrix come in the order of their size.
    let m, f = scientific::from_rows(2, 2, [4.0, 3.0, 6.0, 3.0]);
    let sv, f1 = scientific::svdvals(m);
    assert(f1.kind == scientific::F_NONE && sv.data.len == 2);
    assert(common::close(sv.data[0], 8.33557912) && common::close(sv.data[1], 0.71980602));
    // The product of the left vectors, the values and the right
    // vectors gives the matrix.
    let u, s, vt, f2 = scientific::svd(m);
    assert(f2.kind == scientific::F_NONE && u.shape[0] == 2 && u.shape[1] == 2);
    assert(vt.shape[0] == 2 && vt.shape[1] == 2);
    let prod, f3 = scientific::matmul(u, vt);
    assert(f3.kind == scientific::F_NONE);
    // The product scales the columns of the right vectors.
    let scaled, f4 = scientific::scale(vt, s.data[0]);
    assert(f4.kind == scientific::F_NONE);
    // The left vectors, the values and the right vectors give the
    // matrix back.
    let r00 = u.data[0] * s.data[0] * vt.data[0] + u.data[1] * s.data[1] * vt.data[2];
    let r01 = u.data[0] * s.data[0] * vt.data[1] + u.data[1] * s.data[1] * vt.data[3];
    let r10 = u.data[2] * s.data[0] * vt.data[0] + u.data[3] * s.data[1] * vt.data[2];
    let r11 = u.data[2] * s.data[0] * vt.data[1] + u.data[3] * s.data[1] * vt.data[3];
    assert(common::close(r00, 4.0) && common::close(r01, 3.0) && common::close(r10, 6.0) && common::close(r11, 3.0));
    // A matrix with fewer rows than columns holds a value for each
    // column.
    let wide, f5 = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let wsv, f6 = scientific::svdvals(wide);
    assert(f6.kind == scientific::F_NONE && wsv.data.len == 2);
    // The least squares fits the points.
    let pts, f7 = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 1.0]);
    let rhs, f8 = scientific::from_values([1.0, 2.0]);
    let sol, f9 = scientific::lstsq(pts, rhs);
    assert(f9.kind == scientific::F_NONE && sol.data.len == 2);
    assert(common::close(sol.data[0], 0.6) && common::close(sol.data[1], 0.2));
    // A system with more points than unknowns fits the points.
    let many, f7b = scientific::from_rows(3, 2, [1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);
    let many_rhs, f8b = scientific::from_values([2.0, 3.0, 4.0]);
    let many_sol, f9b = scientific::lstsq(many, many_rhs);
    assert(f9b.kind == scientific::F_NONE && common::close(many_sol.data[0], 1.0) && common::close(many_sol.data[1], 1.0));
    // The rank and the condition of a matrix.
    let rk, f10 = scientific::matrix_rank(m);
    assert(f10.kind == scientific::F_NONE && rk == 2);
    let sing, f11 = scientific::from_rows(2, 2, [1.0, 2.0, 2.0, 4.0]);
    let rk2, f12 = scientific::matrix_rank(sing);
    assert(f12.kind == scientific::F_NONE && rk2 == 1);
    let cd, f13 = scientific::cond(m);
    assert(f13.kind == scientific::F_NONE && common::close(cd, 8.33557912 / 0.71980602));
    // The value of a polynomial.
    let pv = scientific::polyval([1.0, 0.0, -1.0], 2.0);
    assert(common::close(pv, 3.0));
    // The polynomial that fits the points.
    let xs, f14 = scientific::from_values([0.0, 1.0, 2.0]);
    let ys, f15 = scientific::from_values([1.0, 3.0, 5.0]);
    let fit, f16 = scientific::polyfit(xs.data, ys.data, 1);
    assert(f16.kind == scientific::F_NONE && common::close(fit.data[0], 2.0) && common::close(fit.data[1], 1.0));
    // The roots of a polynomial. The parts come in two arrays.
    let rt, rti, f17 = scientific::roots([1.0, -3.0, 2.0]);
    assert(f17.kind == scientific::F_NONE && rt.data.len == 2);
    assert(common::close(rt.data[0], 1.0) || common::close(rt.data[0], 2.0));
    assert(common::close(rt.data[1], 1.0) || common::close(rt.data[1], 2.0));
    // A polynomial with no term is a fault.
    let bad, badi, f18 = scientific::roots([0.0, 0.0]);
    assert(f18.kind == scientific::F_VALUE);
    // The roots of a polynomial with complex roots. The values come
    // from numpy.roots.
    let cr, cri, f20 = scientific::roots([1.0, 0.0, 1.0]);
    assert(f20.kind == scientific::F_NONE && cr.data.len == 2);
    assert(common::close(cr.data[0], 0.0) && common::close(cr.data[1], 0.0));
    assert((common::close(cri.data[0], 1.0) && common::close(cri.data[1], -1.0)) || (common::close(cri.data[0], -1.0) && common::close(cri.data[1], 1.0)));
    let cr2, cri2, f21 = scientific::roots([1.0, 0.0, 0.0, -1.0]);
    assert(f21.kind == scientific::F_NONE && cr2.data.len == 3);
    // The roots come in no fixed order.
    var one: i64 = -1;
    var k: i64 = 0;
    while k < 3 {
        if common::close(cr2.data[k], 1.0) { one = k; }
        k += 1;
    }
    assert(one >= 0 && common::close(cri2.data[one], 0.0));
    var j1: i64 = 0;
    while j1 == one { j1 += 1; }
    var j2 = j1 + 1;
    while j2 == one { j2 += 1; }
    assert(common::close(cr2.data[j1], -0.5) && common::close(cr2.data[j2], -0.5));
    assert((common::close(cri2.data[j1], 0.8660254037844386) && common::close(cri2.data[j2], -0.8660254037844386)) || (common::close(cri2.data[j1], -0.8660254037844386) && common::close(cri2.data[j2], 0.8660254037844386)));
    let cr3, cri3, f22 = scientific::roots([2.0, 3.0, 5.0]);
    assert(f22.kind == scientific::F_NONE && cr3.data.len == 2);
    assert(common::close(cr3.data[0], -0.75) && common::close(cr3.data[1], -0.75));
    assert((common::close(cri3.data[0], 1.3919410907075054) && common::close(cri3.data[1], -1.3919410907075054)) || (common::close(cri3.data[0], -1.3919410907075054) && common::close(cri3.data[1], 1.3919410907075054)));
    // A degree above the limit is a fault, before any work starts.
    var big: f64[>..] = [];
    var bi: i64 = 0;
    while bi < 300 { big.push(1.0); bi += 1; }
    let bigr, bigi, f23 = scientific::roots(big);
    assert(f23.kind == scientific::F_LIMIT);
    // A degree below zero is a fault.
    let bad2, f19 = scientific::polyfit(xs.data, ys.data, -1);
    assert(f19.kind == scientific::F_VALUE);
}

fn test_pinv() {
    // NumPy 2.5.3: the pseudo-inverse of a tall matrix, of a matrix
    // of rank one, of zeros and of a wide matrix.
    let a, fa = scientific::from_rows(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(fa.kind == scientific::F_NONE);
    let p, fp = scientific::pinv(a);
    assert(fp.kind == scientific::F_NONE && p.shape[0] == 2 && p.shape[1] == 3);
    let want = [-1.3333333333333337, -0.33333333333333287, 0.6666666666666665, 1.0833333333333335, 0.3333333333333329, -0.4166666666666665];
    var i: i64 = 0;
    while i < 6 {
        var diff = p.data[i] - want[i];
        if diff < 0.0 { diff = -diff; }
        assert(diff < 1.0e-12);
        i += 1;
    }
    let b, fb = scientific::from_rows(2, 2, [1.0, 2.0, 2.0, 4.0]);
    assert(fb.kind == scientific::F_NONE);
    let pb, fpb = scientific::pinv(b);
    assert(fpb.kind == scientific::F_NONE);
    let wb = [0.04, 0.08, 0.08, 0.16];
    i = 0;
    while i < 4 {
        var diff = pb.data[i] - wb[i];
        if diff < 0.0 { diff = -diff; }
        assert(diff < 1.0e-12);
        i += 1;
    }
    let z, fz = scientific::from_rows(2, 3, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert(fz.kind == scientific::F_NONE);
    let pz, fpz = scientific::pinv(z);
    assert(fpz.kind == scientific::F_NONE && pz.shape[0] == 3 && pz.shape[1] == 2 && pz.data[5] == 0.0);
    // A singular value under the cutoff counts as zero.
    let c, fc = scientific::from_rows(2, 2, [1.0, 0.0, 0.0, 1.0e-20]);
    assert(fc.kind == scientific::F_NONE);
    let pc, fpc = scientific::pinv(c);
    assert(fpc.kind == scientific::F_NONE && pc.data[0] == 1.0 && pc.data[3] == 0.0);
    let pc2, fpc2 = scientific::pinv(c, 1.0e-30);
    assert(fpc2.kind == scientific::F_NONE && pc2.data[3] > 9.0e19);
    // The product of a matrix, its inverse and the matrix gives the matrix.
    let pa, fpa = scientific::matmul(a, p);
    assert(fpa.kind == scientific::F_NONE);
    let back, fback = scientific::matmul(pa, a);
    assert(fback.kind == scientific::F_NONE);
    i = 0;
    while i < 6 {
        var diff = back.data[i] - a.data[i];
        if diff < 0.0 { diff = -diff; }
        assert(diff < 1.0e-10);
        i += 1;
    }
    // A matrix whose first row is zero still gives the values from the
    // largest to the smallest, and the rank of NumPy. NumPy 2.5.3
    // gives the values 2519.1989127, 1136.8725513, 0 and the rank 2.
    let zr, fzr = scientific::from_rows(3, 3, [0.0, 0.0, 0.0, 610.6409992957679, 577.0754169394714, 2332.7120253825424, -336.52552992851207, -914.8298592004128, 735.687023006662]);
    assert(fzr.kind == scientific::F_NONE);
    let sz, fsz = scientific::svdvals(zr);
    assert(fsz.kind == scientific::F_NONE);
    assert(sz.data[0] > 2519.19 && sz.data[0] < 2519.20 && sz.data[1] > 1136.87 && sz.data[1] < 1136.88);
    assert(sz.data[2] < 1.0e-10);
    let rz, frz = scientific::matrix_rank(zr);
    assert(frz.kind == scientific::F_NONE && rz == 2);
    let pz2, fpz2 = scientific::pinv(zr);
    assert(fpz2.kind == scientific::F_NONE && pz2.data[0] == 0.0);
    let v, fv = scientific::from_values([1.0, 2.0]);
    assert(fv.kind == scientific::F_NONE);
    let _q, bad = scientific::pinv(v);
    assert(bad.kind == scientific::F_SHAPE);
}
