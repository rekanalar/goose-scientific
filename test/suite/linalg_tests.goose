// The tests of the group linalg. Run them with SCIENTIFIC_SUITES=linalg.
namespace linalg_tests;

import scientific from "../..";
import os;
import common;

// The products and the solving of a system.
fn test_linalg() {
    let a = common::make23();
    // matmul multiplies two matrices.
    let b, f0 = scientific::from_rows(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let p, f = scientific::matmul(a, b);
    assert(f.kind == scientific::F_NONE && p.shape[0] == 2 && p.shape[1] == 2);
    assert(p.data[0] == 22.0 && p.data[1] == 28.0 && p.data[2] == 49.0 && p.data[3] == 64.0);
    // The inner dimensions must agree.
    let bad, f2 = scientific::matmul(a, a);
    assert(f2.kind == scientific::F_SHAPE);
    let row, f3 = scientific::from_values([1.0, 2.0, 3.0]);
    let col, f4 = scientific::matmul(a, row);
    assert(f4.kind == scientific::F_NONE && common::same_vals(col, [2], [14.0, 32.0]));
    let rowp, f4b = scientific::matmul(row, b);
    assert(f4b.kind == scientific::F_NONE && common::same_vals(rowp, [2], [22.0, 28.0]));
    // dot of two rows gives a number.
    let v, f5 = scientific::from_values([1.0, 2.0, 3.0]);
    let w, f6 = scientific::from_values([4.0, 5.0, 6.0]);
    let dt, f7 = scientific::dot(v, w);
    assert(f7.kind == scientific::F_NONE && dt.ndim == 0 && dt.data[0] == 32.0);
    // dot of a row and a matrix gives a row.
    let u, f8a = scientific::from_values([1.0, 2.0]);
    let dr, f8 = scientific::dot(u, a);
    assert(f8.kind == scientific::F_NONE && dr.data.len == 3 && dr.data[0] == 9.0 && dr.data[1] == 12.0 && dr.data[2] == 15.0);
    // dot of a matrix and a row sums over the last axis of the matrix.
    let dc, f9 = scientific::dot(a, v);
    assert(f9.kind == scientific::F_NONE && dc.data.len == 2 && dc.data[0] == 14.0 && dc.data[1] == 32.0);
    let dcb, f9b = scientific::dot(a, u);
    assert(f9b.kind == scientific::F_SHAPE && dcb.data.len == 0);
    // The lengths must agree.
    let dw, f9a = scientific::dot(v, w);
    assert(f9a.kind == scientific::F_NONE && dw.data[0] == 32.0);
    // inner and outer.
    let inn, f10 = scientific::inner(v, v);
    assert(f10.kind == scientific::F_NONE && inn.data[0] == 14.0);
    let ou, f11 = scientific::outer(v, w);
    assert(f11.kind == scientific::F_NONE && ou.shape[0] == 3 && ou.shape[1] == 3);
    assert(ou.data[0] == 4.0 && ou.data[4] == 10.0 && ou.data[8] == 18.0);
    // kron gives the Kronecker product.
    let k1, f12 = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let k2, f13 = scientific::from_rows(1, 2, [5.0, 6.0]);
    let kr, f14 = scientific::kron(k1, k2);
    assert(f14.kind == scientific::F_NONE && kr.shape[0] == 2 && kr.shape[1] == 4);
    assert(kr.data[0] == 5.0 && kr.data[1] == 6.0 && kr.data[4] == 15.0 && kr.data[5] == 18.0);
    // trace and norm.
    let tr, f15 = scientific::trace(k1);
    assert(f15.kind == scientific::F_NONE && tr == 5.0);
    let tr2, f16 = scientific::trace(a);
    assert(f16.kind == scientific::F_SHAPE);
    let nm, f17 = scientific::norm(k1);
    assert(f17.kind == scientific::F_NONE && common::close(nm, 5.47722557));
    let nm2, f18 = scientific::norm(v);
    assert(f18.kind == scientific::F_NONE && common::close(nm2, 3.74165738));
    // solve gives the vector that solves the system.
    let m, f19 = scientific::from_rows(2, 2, [2.0, 1.0, 1.0, 3.0]);
    let rhs, f20 = scientific::from_values([3.0, 4.0]);
    let sol, f21 = scientific::solve(m, rhs);
    assert(f21.kind == scientific::F_NONE && common::close(sol.data[0], 1.0) && common::close(sol.data[1], 1.0));
    // A matrix without a pivot is a fault.
    let sing, f22 = scientific::from_rows(2, 2, [1.0, 2.0, 2.0, 4.0]);
    let bad3, f23 = scientific::solve(sing, rhs);
    assert(f23.kind == scientific::F_VALUE);
    let bad4, f24 = scientific::solve(a, rhs);
    assert(f24.kind == scientific::F_SHAPE);
    // inv gives the inverse.
    let iv, f25 = scientific::inv(m);
    assert(f25.kind == scientific::F_NONE && common::close(iv.data[0], 0.6) && common::close(iv.data[1], -0.2));
    assert(common::close(iv.data[2], -0.2) && common::close(iv.data[3], 0.4));
    // det gives the determinant.
    let dt2, f26 = scientific::det(m);
    assert(f26.kind == scientific::F_NONE && common::close(dt2, 5.0));
    let dt3, f27 = scientific::det(sing);
    assert(f27.kind == scientific::F_VALUE);
    // slogdet gives the sign and the logarithm.
    let sg, lg, f28 = scientific::slogdet(m);
    assert(f28.kind == scientific::F_NONE && sg == 1.0 && common::close(lg, 1.60943791));
    let sg2, lg2, f29 = scientific::slogdet(sing);
    assert(f29.kind == scientific::F_NONE && sg2 == 0.0 && lg2 == 0.0);
    // matrix_power raises a matrix to a power.
    let pw, f30 = scientific::matrix_power(m, 2);
    assert(f30.kind == scientific::F_NONE && common::close(pw.data[0], 5.0) && common::close(pw.data[1], 5.0));
    assert(common::close(pw.data[2], 5.0) && common::close(pw.data[3], 10.0));
    let pw0, f31 = scientific::matrix_power(m, 0);
    assert(f31.kind == scientific::F_NONE && pw0.data[0] == 1.0 && pw0.data[1] == 0.0 && pw0.data[3] == 1.0);
    let pwn, f33 = scientific::matrix_power(m, -1);
    assert(f33.kind == scientific::F_NONE && common::close(pwn.data[0], 0.6));
}
