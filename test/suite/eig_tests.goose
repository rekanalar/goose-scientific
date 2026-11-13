// The tests of the group eig. Run them with SCIENTIFIC_SUITES=eig.
namespace eig_tests;

import scientific from "../..";
import os;
import common;

// The Cholesky, the QR and the eigenvalues.
fn test_decomp_and_eig() {
    // The Cholesky factor of a symmetric matrix.
    let m, f = scientific::from_rows(2, 2, [4.0, 2.0, 2.0, 3.0]);
    let ch, f1 = scientific::cholesky(m);
    assert(f1.kind == scientific::F_NONE);
    assert(common::close(ch.data[0], 2.0) && common::close(ch.data[2], 1.0) && common::close(ch.data[3], 1.41421356));
    // A matrix with a negative value holds no Cholesky factor.
    let neg, f2 = scientific::from_rows(2, 2, [1.0, 2.0, 2.0, 1.0]);
    let bad, f3 = scientific::cholesky(neg);
    assert(f3.kind == scientific::F_VALUE && bad.data.len == 0);
    let ns, f4 = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let bad2, f5 = scientific::cholesky(ns);
    assert(f5.kind == scientific::F_SHAPE);
    // The QR decomposition gives an orthogonal matrix and an upper
    // triangle. The product of the two gives the matrix.
    let q, r, f6 = scientific::qr(ns);
    assert(f6.kind == scientific::F_NONE && q.shape[0] == 2 && q.shape[1] == 2 && r.shape[0] == 2 && r.shape[1] == 3);
    let qr_prod, f7 = scientific::matmul(q, r);
    assert(f7.kind == scientific::F_NONE);
    assert(common::close(qr_prod.data[0], 1.0) && common::close(qr_prod.data[2], 3.0) && common::close(qr_prod.data[3], 4.0) && common::close(qr_prod.data[5], 6.0));
    let qn, rn, f8 = scientific::qr(m);
    assert(f8.kind == scientific::F_NONE);
    // The signs follow LAPACK: the diagonal of R is negative for a
    // turn, and the last column of a square matrix stays as it is.
    let sq, fsq = scientific::from_rows(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
    let qs, rs, fsq2 = scientific::qr(sq);
    assert(fsq2.kind == scientific::F_NONE);
    assert(common::close(rs.data[0], -8.124038404635959) && common::close(rs.data[4], 0.9045340337332926) && common::close(rs.data[8], 0.40824829046386224));
    assert(common::close(qs.data[0], -0.12309149097933281) && common::close(qs.data[2], 0.4082482904638621));
    // The eigenvalues of a symmetric matrix come in the order of their
    // value.
    let sym, f9 = scientific::from_rows(2, 2, [2.0, 1.0, 1.0, 2.0]);
    let ev, f10 = scientific::eigvalsh(sym);
    assert(f10.kind == scientific::F_NONE && common::close(ev.data[0], 1.0) && common::close(ev.data[1], 3.0));
    // The eigenvectors are the columns of the result.
    let vals, vecs, f11 = scientific::eigh(sym);
    assert(f11.kind == scientific::F_NONE && common::close(vals.data[0], 1.0) && common::close(vals.data[1], 3.0));
    assert(common::close(common::mathshim_abs(vecs.data[1]), 0.70710678));
    // The eigenvalues of a matrix that is not symmetric.
    let gen, f12 = scientific::from_rows(2, 2, [4.0, 1.0, 2.0, 3.0]);
    let gev, f13 = scientific::eigvals(gen);
    assert(f13.kind == scientific::F_NONE && gev.data.len == 2);
    let sum = gev.data[0] + gev.data[1];
    assert(common::close(sum, 7.0));
    // The vector of an eigenvalue holds the matrix near zero.
    let gvals, gvecs, f12b = scientific::eig(gen);
    assert(f12b.kind == scientific::F_NONE);
    let g0 = gvecs.data[0];
    let g1 = gvecs.data[2];
    let av0 = gen.data[0] * g0 + gen.data[1] * g1;
    let av1 = gen.data[2] * g0 + gen.data[3] * g1;
    assert(common::close(av0, gvals.data[0] * g0) && common::close(av1, gvals.data[0] * g1));
    // A matrix with a complex pair holds no real eigenvalues.
    let rot, f14 = scientific::from_rows(2, 2, [0.0, -1.0, 1.0, 0.0]);
    let bad3, f15 = scientific::eigvals(rot);
    assert(f15.kind == scientific::F_VALUE);
    // A matrix that is not square is a fault.
    let bad4, f16 = scientific::eigvalsh(ns);
    assert(f16.kind == scientific::F_SHAPE);
    let bad5, bad6, f17 = scientific::eigh(ns);
    assert(f17.kind == scientific::F_SHAPE);
}
