// The tests of the group sparse. Run them with SCIENTIFIC_SUITES=sparse.
namespace sparse_tests;

import scientific from "../..";
import os;
import common;
import common_3;

fn test_sparse() {
    let sd, sdf = scientific::csr_from_dense(common::mx(5, 4, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 8.0]));
    let se, sef = scientific::csr_from_dense(common::mx(5, 4, [0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, -8.0]));
    assert(sdf.kind == scientific::F_NONE && sef.kind == scientific::F_NONE && sd.nnz == 8);
    let dd, ddf = scientific::csr_to_dense(sd);
    assert(ddf.kind == scientific::F_NONE && common::all_near(dd, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 8.0]));
    let mv, mvf = scientific::csr_matvec(sd, [1.0, -2.0, 0.5, 3.0]);
    assert(mvf.kind == scientific::F_NONE && common::all_near(mv, [2.0, 1.5, 12.0, 0.0, 31.0]));
    let ad, adf = scientific::csr_add(sd, se);
    let add, addf = scientific::csr_to_dense(ad);
    assert(adf.kind == scientific::F_NONE && common::all_near(add, [1.0, 1.0, 2.0, 2.0, 3.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 11.0, 1.0, 1.0, 0.0, 0.0, 7.0, 0.0, 2.0, 0.0]));
    let sb, sbf = scientific::csr_sub(sd, se);
    let sbd, sbdf = scientific::csr_to_dense(sb);
    assert(sbf.kind == scientific::F_NONE && common::all_near(sbd, [1.0, -1.0, 2.0, -2.0, -3.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 7.0, 0.0, -2.0, 16.0]));
    let tr, trf = scientific::csr_transpose(sd);
    let tm, tmf = scientific::csr_matmul(sd, tr);
    let tmd, tmdf = scientific::csr_to_dense(tm);
    assert(trf.kind == scientific::F_NONE && tmf.kind == scientific::F_NONE && common::all_near(tmd, [5.0, 6.0, 4.0, 0.0, 7.0, 6.0, 9.0, 0.0, 0.0, 0.0, 4.0, 0.0, 77.0, 0.0, 76.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 76.0, 0.0, 113.0]));
    let r1, r1f = scientific::csr_sum(sd, 1);
    let r0, r0f = scientific::csr_sum(sd, 0);
    assert(common::all_near(r1, [3.0, 3.0, 15.0, 0.0, 15.0]) && common::all_near(r0, [12.0, 5.0, 5.0, 14.0]));
    let sc = scientific::csr_scale(sd, -2.5);
    let scd, scdf = scientific::csr_to_dense(sc);
    assert(common::all_near(scd, [-2.5, -0.0, -5.0, -0.0, -0.0, -0.0, -7.5, -0.0, -10.0, -12.5, -0.0, -15.0, -0.0, -0.0, -0.0, -0.0, -17.5, -0.0, -0.0, -20.0]));
    let g1, g1f = scientific::csr_get(sd, 2, 3);
    let g2, g2f = scientific::csr_get(sd, 3, 3);
    assert(common::near(g1, 6.0) && g2 == 0.0);
    let dg, dgf = scientific::csr_diagonal(sd);
    assert(common::all_near(dg, [1.0, 0.0, 0.0, 0.0]));
    let co, cof = scientific::csr_from_triplets(3, 3, [0.0, 2.0, 0.0, 1.0, 0.0], [1.0, 2.0, 1.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0, 5.0]);
    let cod, codf = scientific::csr_to_dense(co);
    assert(cof.kind == scientific::F_NONE && co.nnz == 4 && common::all_near(cod, [5.0, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 2.0]));
    let dg1, dg1f = scientific::csr_diags([1.0, 2.0, 3.0, 4.0], 1, 4, 5);
    let dg1d, dg1df = scientific::csr_to_dense(dg1);
    assert(dg1f.kind == scientific::F_NONE && common::all_near(dg1d, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]));
    let dg2, dg2f = scientific::csr_diags([1.0, 2.0, 3.0, 4.0, 5.0], -1, 5, 5);
    let dg2d, dg2df = scientific::csr_to_dense(dg2);
    assert(dg2f.kind == scientific::F_NONE && common::all_near(dg2d, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0]));
    let dg3, dg3f = scientific::csr_diags([1.0, 2.0, 3.0, 4.0, 5.0], 0, 4, 6);
    let dg3d, dg3df = scientific::csr_to_dense(dg3);
    assert(dg3f.kind == scientific::F_NONE && common::all_near(dg3d, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]));
    let dg4, dg4f = scientific::csr_diags([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], -2, 6, 4);
    let dg4d, dg4df = scientific::csr_to_dense(dg4);
    assert(dg4f.kind == scientific::F_NONE && common::all_near(dg4d, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0]));
    let ey, eyf = scientific::csr_eye(3);
    let eyd, eydf = scientific::csr_to_dense(ey);
    assert(common::all_near(eyd, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
    let ta, taf = scientific::csr_from_dense(common::mx(30, 30, [2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0]));
    let sx, sxf = scientific::spsolve(ta, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]);
    assert(sxf.kind == scientific::F_NONE && common::all_near(sx, [159.99999999999974, 318.9999999999995, 475.99999999999926, 629.9999999999991, 779.9999999999987, 924.9999999999985, 1063.9999999999982, 1195.999999999998, 1319.9999999999977, 1434.9999999999975, 1539.9999999999973, 1633.9999999999968, 1715.9999999999964, 1784.999999999996, 1839.9999999999955, 1879.9999999999952, 1903.999999999995, 1910.9999999999948, 1899.9999999999945, 1869.9999999999945, 1819.9999999999948, 1748.999999999995, 1655.9999999999957, 1539.9999999999964, 1399.9999999999968, 1234.9999999999973, 1043.9999999999977, 825.9999999999982, 579.9999999999987, 304.9999999999994]));
    let cx, ci, cf = scientific::cg(ta, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0], 1e-12, 500);
    assert(cf.kind == scientific::F_NONE && ci == 0 && common::all_near(cx, [159.99999999999991, 318.9999999999999, 475.9999999999998, 629.9999999999998, 779.9999999999998, 924.9999999999998, 1063.9999999999993, 1195.9999999999995, 1319.9999999999993, 1434.999999999999, 1539.9999999999986, 1633.9999999999993, 1715.999999999999, 1784.999999999999, 1839.9999999999989, 1879.9999999999993, 1903.9999999999998, 1910.999999999999, 1899.9999999999998, 1869.9999999999993, 1819.9999999999998, 1748.999999999999, 1655.999999999999, 1539.9999999999993, 1399.9999999999995, 1234.9999999999995, 1043.9999999999998, 825.9999999999998, 580.0, 305.0]));
    let pa, paf = scientific::csr_from_dense(common::mx(3, 3, [0.0, 5.0, 0.0, 7.0, 0.0, 1.0, 2.0, 0.0, 0.0]));
    let px, pxf = scientific::spsolve(pa, [10.0, 3.0, 4.0]);
    assert(pxf.kind == scientific::F_NONE && common::all_near(px, [2.0, 2.0, -11.0]));
    let bad1, bad1f = scientific::csr_from_triplets(2, 2, [2.0], [0.0], [1.0]);
    assert(bad1f.kind == scientific::F_RANGE);
    let bad2, bad2f = scientific::csr_add(sd, tr);
    assert(bad2f.kind == scientific::F_SHAPE);
    let bad3, bad3f = scientific::csr_matvec(sd, [1.0]);
    assert(bad3f.kind == scientific::F_SHAPE);
    let bad4, bad4f = scientific::csr_from_triplets(0, 2, [], [], []);
    assert(bad4f.kind == scientific::F_RANGE);
    let nx, ni, nf = scientific::cg(pa, [1.0, 1.0, 1.0]);
    assert(nf.kind == scientific::F_VALUE);
}

// csr_kron, csr_hstack and csr_vstack are pinned against SciPy.
fn test_sparse_join() {
    let a, fa = scientific::csr_from_dense(common::mx(2, 3, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0]));
    let b, fb = scientific::csr_from_dense(common::mx(3, 2, [0.0, 4.0, 5.0, 6.0, 0.0, 7.0]));
    let bt, fbt = scientific::csr_from_dense(common::mx(2, 3, [0.0, 5.0, 0.0, 4.0, 6.0, 7.0]));
    let c, fc = scientific::csr_from_dense(common::mx(2, 3, [2.0, 0.0, 0.0, 0.0, -1.0, 1.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE && fbt.kind == scientific::F_NONE && fc.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kron(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 6 && k.cols == 6 && k.nnz == 12);
    let kd, fkd = scientific::csr_to_dense(k);
    assert(fkd.kind == scientific::F_NONE && common::all_near(kd, [0.0, 4.0, 0.0, 0.0, 0.0, 8.0, 5.0, 6.0, 0.0, 0.0, 10.0, 12.0, 0.0, 7.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 15.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 21.0]));
    let h, fh = scientific::csr_hstack(a, bt);
    assert(fh.kind == scientific::F_NONE && h.cols == 6);
    let hd, fhd = scientific::csr_to_dense(h);
    assert(fhd.kind == scientific::F_NONE && common::all_near(hd, [1.0, 0.0, 2.0, 0.0, 5.0, 0.0, 0.0, 0.0, 3.0, 4.0, 6.0, 7.0]));
    let v, fv = scientific::csr_vstack(a, c);
    assert(fv.kind == scientific::F_NONE && v.rows == 4);
    let vd, fvd = scientific::csr_to_dense(v);
    assert(fvd.kind == scientific::F_NONE && common::all_near(vd, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0, -1.0, 1.0]));
    let e1, g1 = scientific::csr_hstack(a, b);
    assert(g1.kind == scientific::F_SHAPE);
    let e2, g2 = scientific::csr_vstack(a, b);
    assert(g2.kind == scientific::F_SHAPE);
    return;
}

// bicgstab and gmres are pinned against the dense solution of NumPy.
fn test_krylov() {
    let a, fa = scientific::csr_from_dense(common::mx(8, 8, [4.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 4.1, -0.95, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 4.2, -0.9, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 4.3, -0.85, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 4.4, -0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -2.0, 4.5, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 4.6, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 4.7]));
    assert(fa.kind == scientific::F_NONE);
    let b = [1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 1.5, -0.5];
    let want = [0.1387735113564285, -0.28519591834488306, 0.5824268411536323, 0.31942007245880555, 0.00031157404565690866, 0.541786991355806, 0.5832244173464183, 0.14179762440273116];
    let x1, c1, f1 = scientific::bicgstab(a, b, 1.0e-12);
    assert(f1.kind == scientific::F_NONE && c1 == 0 && common::all_near(x1, want));
    let x2, c2, f2 = scientific::gmres(a, b, 1.0e-12);
    assert(f2.kind == scientific::F_NONE && c2 == 0 && common::all_near(x2, want));
    // A restart shorter than the side still reaches the answer.
    let x3, c3, f3 = scientific::gmres(a, b, 1.0e-12, 3);
    assert(f3.kind == scientific::F_NONE && c3 == 0 && common::all_near(x3, want));
    // The right side must fit the matrix.
    let e1, k1, g1 = scientific::gmres(a, [1.0, 2.0]);
    assert(g1.kind == scientific::F_SHAPE);
    let e2, k2, g2 = scientific::bicgstab(a, [1.0, 2.0]);
    assert(g2.kind == scientific::F_SHAPE);
    let e3, k3, g3 = scientific::gmres(a, b, 1.0e-5, 0);
    assert(g3.kind == scientific::F_VALUE);
    return;
}

// The structure functions of sparse matrices are pinned against SciPy.
fn test_sparse_structure() {
    let a, fa = scientific::csr_from_dense(common::mx(3, 4, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0]));
    let b, fb = scientific::csr_from_dense(common::mx(2, 2, [0.0, 7.0, 8.0, 0.0]));
    let s, fs = scientific::csr_from_dense(common::mx(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
    let c, fc = scientific::csr_from_dense(common::mx(3, 4, [1.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE && fs.kind == scientific::F_NONE && fc.kind == scientific::F_NONE);
    let bd, f1 = scientific::csr_block_diag(a, b);
    let bdd, f1d = scientific::csr_to_dense(bd);
    assert(f1.kind == scientific::F_NONE && bd.rows == 5 && bd.cols == 6 && common::all_near(bdd, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0]));
    let lo, f2 = scientific::csr_tril(s, -1);
    let lod, f2d = scientific::csr_to_dense(lo);
    assert(f2.kind == scientific::F_NONE && common::all_near(lod, [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]));
    let up, f3 = scientific::csr_triu(s, 1);
    let upd, f3d = scientific::csr_to_dense(up);
    assert(f3.kind == scientific::F_NONE && common::all_near(upd, [0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]));
    let lo0, f4 = scientific::csr_tril(s);
    let lo0d, f4d = scientific::csr_to_dense(lo0);
    assert(f4.kind == scientific::F_NONE && common::all_near(lo0d, [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]));
    let m, f5 = scientific::csr_multiply(a, c);
    let md, f5d = scientific::csr_to_dense(m);
    assert(f5.kind == scientific::F_NONE && common::all_near(md, [1.0, 0.0, 6.0, 0.0, 0.0, 6.0, 0.0, 16.0, 25.0, 0.0, 0.0, 0.0]));
    let sl, f6 = scientific::csr_slice(a, 1, 3, 1, 4);
    let sld, f6d = scientific::csr_to_dense(sl);
    assert(f6.kind == scientific::F_NONE && sl.rows == 2 && sl.cols == 3 && common::all_near(sld, [3.0, 0.0, 4.0, 0.0, 6.0, 0.0]));
    let bad, f7 = scientific::csr_multiply(a, b);
    assert(f7.kind == scientific::F_SHAPE);
    let bad2, f8 = scientific::csr_slice(a, 2, 2, 0, 1);
    assert(f8.kind == scientific::F_RANGE);
    return;
}

// lsqr and minres are pinned against the dense solutions of NumPy.
fn test_lsqr_minres() {
    let a, fa = scientific::csr_from_dense(common::mx(4, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 0.0, 1.0]));
    let x1, i1, f1 = scientific::lsqr(a, [1.0, 2.0, 2.0, 3.0], 1.0e-12);
    assert(fa.kind == scientific::F_NONE && f1.kind == scientific::F_NONE && i1 == 0 && common::all_loose(x1, [-2.7142857142857144, 2.285714285714286]));
    let w, fw = scientific::csr_from_dense(common::mx(2, 4, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0]));
    let x2, i2, f2 = scientific::lsqr(w, [1.0, 2.0], 1.0e-12);
    assert(fw.kind == scientific::F_NONE && f2.kind == scientific::F_NONE && i2 == 0 && common::all_loose(x2, [0.2, 0.24, 0.4, 0.32]));
    let s, fs = scientific::csr_from_dense(common::mx(3, 3, [2.0, 1.0, 0.0, 1.0, -3.0, 1.0, 0.0, 1.0, 4.0]));
    let x3, i3, f3 = scientific::minres(s, [1.0, 2.0, 3.0], 1.0e-12);
    assert(fs.kind == scientific::F_NONE && f3.kind == scientific::F_NONE && i3 == 0 && common::all_loose(x3, [0.6, -0.2, 0.8]));
    let x4, i4, f4 = scientific::lsqr(a, [1.0, 2.0, 3.0]);
    assert(f4.kind == scientific::F_SHAPE);
    let x5, i5, f5 = scientific::minres(a, [1.0, 2.0, 3.0, 4.0]);
    assert(f5.kind == scientific::F_SHAPE);
    return;
}

fn test_sparse_lsmr_pins0() {
    let qa0, qf0 = scientific::csr_from_dense(common_3::ni_nd([6, 7], [2.288, 0.0, 0.0, 2.946, 0.0, -2.485, 0.0, -0.646, -1.648, -1.656, 1.728, 2.081, -1.541, 2.995, 0.0, 1.993, 2.125, -1.298, 1.513, 0.0, 0.0, 1.987, 0.343, 2.179, 0.707, 2.104, 0.0, 2.557, -0.638, 0.0, 0.0, 0.915, 2.484, 2.713, 0.0, -2.6, 2.954, -2.624, 0.0, 0.0, 0.0, 0.0]));
    assert(qf0.kind == scientific::F_NONE);
    let qr0, qg0 = scientific::lsmr(qa0, [-1.131, -1.613, -0.62, -1.833, -1.243, -1.484], 0.0, 1.0e-6, 1.0e-6, 1.0e8, 2);
    assert(qg0.kind == scientific::F_NONE && qr0.istop == 7 && qr0.itn == 2);
    assert(common::all_near(qr0.x, [0.053815922876502775, -0.28956631297030844, 0.09092752939231807, -0.2674123208405637, -0.4118294765338131, 0.0317097934557706, -0.29779172656609987]));
    let qe0 = common_3::ni_nd([5], [qr0.normr, qr0.normar, qr0.norma, qr0.conda, qr0.normx]);
    assert(common::all_near(qe0, [0.6246961977040671, 1.9122911953768258, 7.5875140181828, 1.2211529402131578, 0.6525418831426354]));
    let qa1, qf1 = scientific::csr_from_dense(common_3::ni_nd([6, 3], [1.04, -0.85, 0.384, 0.0, 2.806, 0.252, -1.177, 1.924, 2.388, 1.212, 0.324, 0.0, 0.477, 0.99, 0.0, 0.0, -1.465, -0.591]));
    assert(qf1.kind == scientific::F_NONE);
    let qr1, qg1 = scientific::lsmr(qa1, [1.48, 0.694, 0.676, -0.093, -0.978, 1.483], 0.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg1.kind == scientific::F_NONE && qr1.istop == 2 && qr1.itn == 3);
    assert(common::all_near(qr1.x, [0.2243500664415456, -0.2533826671899411, 0.5591825275483715]));
    let qe1 = common_3::ni_nd([5], [qr1.normr, qr1.normar, qr1.norma, qr1.conda, qr1.normx]);
    assert(common::all_near(qe1, [2.264764787251016, 0.0000000000000011393754255990604, 5.094644246657464, 1.7516531011259329, 0.6536213180886924]));
    let qa2, qf2 = scientific::csr_from_dense(common_3::ni_nd([7, 4], [0.357, 0.0, 0.169, 0.0, 0.103, 0.0, 0.0, 1.61, -1.353, -1.565, 2.329, 0.0, -0.519, 2.928, -2.227, -0.833, 0.0, -0.812, 0.449, 0.0, -1.703, 0.0, 0.0, 0.0, 0.0, 2.114, 0.363, -1.725]));
    assert(qf2.kind == scientific::F_NONE);
    let qr2, qg2 = scientific::lsmr(qa2, [-1.637, 1.581, -0.046, 0.072, -0.161, 0.077, 1.737], 0.5, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg2.kind == scientific::F_NONE && qr2.istop == 2 && qr2.itn == 4);
    assert(common::all_near(qr2.x, [0.0013756171775571521, 0.9680523507609571, 0.7852600837851971, 0.6779453612415933]));
    let qe2 = common_3::ni_nd([5], [qr2.normr, qr2.normar, qr2.norma, qr2.conda, qr2.normx]);
    assert(common::all_near(qe2, [2.1486317451881374, 0.00000000000009133497783405571, 6.188400520328334, 3.1996833717495625, 1.418932894167925]));
    let qa3, qf3 = scientific::csr_from_dense(common_3::ni_nd([4, 3], [0.0, 0.0, -0.983, 2.604, 2.591, 0.685, 0.89, 0.15, 1.424, 0.0, 0.013, -0.494]));
    assert(qf3.kind == scientific::F_NONE);
    let qr3, qg3 = scientific::lsmr(qa3, [1.729, -0.684, 0.897, 0.906], 2.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg3.kind == scientific::F_NONE && qr3.istop == 2 && qr3.itn == 3);
    assert(common::all_near(qr3.x, [0.046210634338690035, -0.1528839809013136, -0.15262858075454727]));
    let qe3 = common_3::ni_nd([5], [qr3.normr, qr3.normar, qr3.norma, qr3.conda, qr3.normx]);
    assert(common::all_near(qe3, [2.162245610518083, 0.0000000000000007738371882779197, 4.244548503669147, 1.8746911483296742, 0.22091721980272316]));
    let qa4, qf4 = scientific::csr_from_dense(common_3::ni_nd([1, 4], [0.0, -0.185, -2.421, 0.0]));
    assert(qf4.kind == scientific::F_NONE);
    let qr4, qg4 = scientific::lsmr(qa4, [0.598], 0.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg4.kind == scientific::F_NONE && qr4.istop == 1 && qr4.itn == 1);
    assert(common::all_near(qr4.x, [0.0, -0.018765268089070482, -0.24557142726291695, 0.0]));
    let qe4 = common_3::ni_nd([5], [qr4.normr, qr4.normar, qr4.norma, qr4.conda, qr4.normx]);
    assert(common::all_near(qe4, [0.0, 0.0, 2.428058071793177, 1.0, 0.24628735488124598]));
}

fn test_sparse_lsmr_pins1() {
    let qa5, qf5 = scientific::csr_from_dense(common_3::ni_nd([7, 1], [2.252, -2.425, 1.765, -2.199, 0.0, 1.419, -1.989]));
    assert(qf5.kind == scientific::F_NONE);
    let qr5, qg5 = scientific::lsmr(qa5, [0.228, -1.751, -1.427, 0.484, 1.585, -1.685, 0.645], 0.0, 1.0e-6, 1.0e-6, 1.0e8, 2);
    assert(qg5.kind == scientific::F_NONE && qr5.istop == 2 && qr5.itn == 1);
    assert(common::all_near(qr5.x, [-0.10040189948496422]));
    let qe5 = common_3::ni_nd([5], [qr5.normr, qr5.normar, qr5.norma, qr5.conda, qr5.normx]);
    assert(common::all_near(qe5, [3.3023196926542675, 0.0, 4.987247437214239, 1.0, 0.10040189948496422]));
    let qa6, qf6 = scientific::csr_from_dense(common_3::ni_nd([5, 1], [0.0, 1.681, 0.0, -1.953, 0.0]));
    assert(qf6.kind == scientific::F_NONE);
    let qr6, qg6 = scientific::lsmr(qa6, [1.299, -0.793, -1.58, 1.685, 0.795], 0.5, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg6.kind == scientific::F_NONE && qr6.istop == 2 && qr6.itn == 1);
    assert(common::all_near(qr6.x, [-0.6710969713946506]));
    let qe6 = common_3::ni_nd([5], [qr6.normr, qr6.normar, qr6.norma, qr6.conda, qr6.normx]);
    assert(common::all_near(qe6, [2.27614945071287, 0.0000000000000006004391847368178, 2.576813924209507, 1.0, 0.6710969713946506]));
    let qa7, qf7 = scientific::csr_from_dense(common_3::ni_nd([6, 6], [1.509, -2.74, 0.238, 0.973, 0.0, 2.911, -0.194, 0.0, 2.205, 0.0, -2.378, 0.0, -0.629, -2.507, 2.875, 0.0, -2.266, -2.877, 0.0, 1.147, 1.394, 1.241, 0.0, -1.002, 0.0, 2.542, 0.0, 0.637, 0.0, -1.047, 0.0, 0.0, 2.747, 0.332, -2.612, 2.321]));
    assert(qf7.kind == scientific::F_NONE);
    let qr7, qg7 = scientific::lsmr(qa7, [-0.403, 1.622, -0.098, -0.522, -1.417, 0.553], 2.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg7.kind == scientific::F_NONE && qr7.istop == 2 && qr7.itn == 6);
    assert(common::all_near(qr7.x, [-0.13993020986456164, -0.04582116448496119, 0.03498704970852333, -0.2897257024325418, -0.1945340042821097, 0.13200015542221039]));
    let qe7 = common_3::ni_nd([5], [qr7.normr, qr7.normar, qr7.norma, qr7.conda, qr7.normx]);
    assert(common::all_near(qe7, [1.7843570869285077, 0.000000000000014910530547923523, 9.58160790264348, 3.31356195041787, 0.402632138432036]));
    let qa8, qf8 = scientific::csr_from_dense(common_3::ni_nd([4, 2], [1.981, -0.165, -2.06, 1.314, 1.62, 0.0, 0.0, 2.144]));
    assert(qf8.kind == scientific::F_NONE);
    let qr8, qg8 = scientific::lsmr(qa8, [0.105, -1.849, 1.104, -1.823], 0.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg8.kind == scientific::F_NONE && qr8.istop == 2 && qr8.itn == 2);
    assert(common::all_near(qr8.x, [0.2964095296691178, -0.8591693496799014]));
    let qe8 = common_3::ni_nd([5], [qr8.normr, qr8.normar, qr8.norma, qr8.conda, qr8.normx]);
    assert(common::all_near(qe8, [0.8892710837365455, 0.000000000000002590214762922366, 4.14040070524581, 1.5493780704837499, 0.9088622451769313]));
    let qa9, qf9 = scientific::csr_from_dense(common_3::ni_nd([4, 4], [-2.569, 0.0, 0.0, 0.0, -0.803, -1.741, -0.374, -2.96, 0.0, 0.0, 0.703, 1.844, 1.269, 0.285, 2.986, 2.055]));
    assert(qf9.kind == scientific::F_NONE);
    let qr9, qg9 = scientific::lsmr(qa9, [-0.089, 1.032, -1.826, -0.144], 0.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg9.kind == scientific::F_NONE && qr9.istop == 1 && qr9.itn == 4);
    assert(common::all_near(qr9.x, [0.034643830284170084, 1.3618596912954246, 0.6623419582165848, -1.2427475035933822]));
    let qe9 = common_3::ni_nd([5], [qr9.normr, qr9.normar, qr9.norma, qr9.conda, qr9.normx]);
    assert(common::all_near(qe9, [0.00000000000018778870544586474, 0.000000000000961933207271708, 6.156498923901474, 5.183668801196728, 1.9593315801705926]));
}

fn test_sparse_lsmr_pins2() {
    let qa10, qf10 = scientific::csr_from_dense(common_3::ni_nd([7, 1], [0.0, 0.567, 2.52, 1.116, 0.57, -2.033, -2.913]));
    assert(qf10.kind == scientific::F_NONE);
    let qr10, qg10 = scientific::lsmr(qa10, [0.15, 1.784, -0.948, -0.605, 0.13, 1.95, 0.766], 0.5, 1.0e-6, 1.0e-6, 1.0e8, 2);
    assert(qg10.kind == scientific::F_NONE && qr10.istop == 2 && qr10.itn == 1);
    assert(common::all_near(qr10.x, [-0.38720371175027446]));
    let qe10 = common_3::ni_nd([5], [qr10.normr, qr10.normar, qr10.norma, qr10.conda, qr10.normx]);
    assert(common::all_near(qe10, [2.3897599618657672, 0.0, 4.567373753044522, 1.0, 0.38720371175027446]));
    let qa11, qf11 = scientific::csr_from_dense(common_3::ni_nd([7, 3], [0.0, -0.172, -0.146, 0.0, -1.177, -2.74, 2.363, -1.298, 1.125, -0.48, 1.967, -1.956, -1.043, 0.0, 1.389, 2.445, 0.0, -1.543, -1.868, 0.0, 1.632]));
    assert(qf11.kind == scientific::F_NONE);
    let qr11, qg11 = scientific::lsmr(qa11, [-0.427, -1.241, 1.269, 1.301, -1.325, 0.675, 1.221], 2.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg11.kind == scientific::F_NONE && qr11.istop == 2 && qr11.itn == 3);
    assert(common::all_near(qr11.x, [0.2533629937215448, 0.3422420827798817, 0.14174572440425742]));
    let qe11 = common_3::ni_nd([5], [qr11.normr, qr11.normar, qr11.norma, qr11.conda, qr11.normx]);
    assert(common::all_near(qe11, [2.6252585431286675, 0.00000000000000004958424626386846, 6.5522777718897105, 1.3316833427479073, 0.44879204560680735]));
    let qa12, qf12 = scientific::csr_from_dense(common_3::ni_nd([1, 1], [0.813]));
    assert(qf12.kind == scientific::F_NONE);
    let qr12, qg12 = scientific::lsmr(qa12, [0.536], 0.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg12.kind == scientific::F_NONE && qr12.istop == 1 && qr12.itn == 1);
    assert(common::all_near(qr12.x, [0.6592865928659286]));
    let qe12 = common_3::ni_nd([5], [qr12.normr, qr12.normar, qr12.norma, qr12.conda, qr12.normx]);
    assert(common::all_near(qe12, [0.00000000000000007319551552264255, 0.0000000000000000595079541199084, 0.813, 1.2300123001230012, 0.6592865928659286]));
    let qa13, qf13 = scientific::csr_from_dense(common_3::ni_nd([2, 6], [0.0, 1.421, 0.862, 2.606, 0.614, -2.433, -1.397, 0.0, 1.703, 1.431, -0.185, 0.0]));
    assert(qf13.kind == scientific::F_NONE);
    let qr13, qg13 = scientific::lsmr(qa13, [0.288, 0.435], 0.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg13.kind == scientific::F_NONE && qr13.istop == 1 && qr13.itn == 2);
    assert(common::all_near(qr13.x, [-0.09025504683098998, -0.0036248118545480906, 0.10782571918797051, 0.08580404800397609, -0.013518416790830188, 0.0062063105152114564]));
    let qe13 = common_3::ni_nd([5], [qr13.normr, qr13.normar, qr13.norma, qr13.conda, qr13.normx]);
    assert(common::all_near(qe13, [0.00000000000000016815769637902774, 0.000000000000000658227895282267, 4.773238942269704, 1.9431434708473636, 0.16543608734027349]));
    let qa14, qf14 = scientific::csr_from_dense(common_3::ni_nd([3, 7], [-2.951, 1.114, -0.256, -0.232, 2.555, -2.388, 0.0, 2.389, 1.32, 0.0, 1.672, 2.203, 0.0, 0.0, 0.0, 1.418, -0.295, 0.0, -1.368, -0.174, -1.481]));
    assert(qf14.kind == scientific::F_NONE);
    let qr14, qg14 = scientific::lsmr(qa14, [-1.215, -1.323, -0.216], 0.5, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg14.kind == scientific::F_NONE && qr14.istop == 2 && qr14.itn == 3);
    assert(common::all_near(qr14.x, [-0.045409940054804226, -0.27697260469923196, 0.03374126573456056, -0.14039578248110487, -0.26832625744936744, 0.15243158198903622, 0.09325126968173555]));
    let qe14 = common_3::ni_nd([5], [qr14.normr, qr14.normar, qr14.norma, qr14.conda, qr14.normx]);
    assert(common::all_near(qe14, [0.22778764418351335, 0.0000000000000004007655310283012, 6.602321561390356, 1.7267446015200134, 0.45117125582557915]));
}

fn test_sparse_lsmr_pins3() {
    let qa15, qf15 = scientific::csr_from_dense(common_3::ni_nd([5, 2], [0.928, 0.0, 0.0, -0.404, -0.941, 0.575, -0.317, -2.028, -1.233, -2.497]));
    assert(qf15.kind == scientific::F_NONE);
    let qr15, qg15 = scientific::lsmr(qa15, [0.74, 0.218, -1.038, -0.87, 1.908], 2.0, 1.0e-6, 1.0e-6, 1.0e8, 2);
    assert(qg15.kind == scientific::F_NONE && qr15.istop == 2 && qr15.itn == 2);
    assert(common::all_near(qr15.x, [0.05629437052954132, -0.26034114488303645]));
    let qe15 = common_3::ni_nd([5], [qr15.normr, qr15.normar, qr15.norma, qr15.conda, qr15.normx]);
    assert(common::all_near(qe15, [2.265807426621658, 0.00000000000000017197093144335643, 3.7694929367223913, 1.5849163107061839, 0.2663579694177132]));
    let qa16, qf16 = scientific::csr_from_dense(common_3::ni_nd([6, 3], [-0.457, -1.524, -1.786, 0.728, -2.278, -1.351, 1.748, -1.531, 0.917, -0.357, 0.0, -1.436, 2.848, 2.41, 0.24, -2.991, 0.0, -0.877]));
    assert(qf16.kind == scientific::F_NONE);
    let qr16, qg16 = scientific::lsmr(qa16, [1.121, 0.508, -1.29, -0.828, -0.837, -1.1], 0.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg16.kind == scientific::F_NONE && qr16.istop == 2 && qr16.itn == 3);
    assert(common::all_near(qr16.x, [-0.0005109970737800888, -0.14152551929981877, -0.1385860406352762]));
    let qe16 = common_3::ni_nd([5], [qr16.normr, qr16.normar, qr16.norma, qr16.conda, qr16.normx]);
    assert(common::all_near(qe16, [2.257640038455117, 0.0000000000000026621709203406655, 6.737032210105574, 2.038865082894302, 0.19808034831869417]));
    let qa17, qf17 = scientific::csr_from_dense(common_3::ni_nd([8, 3], [2.363, 0.0, 0.366, 0.0, -2.052, 0.146, -1.056, -0.391, 0.0, 1.82, 0.53, -1.544, -2.235, 0.0, -0.527, 1.892, 0.0, -0.703, 0.0, -2.419, 0.0, -2.254, 0.0, 1.61]));
    assert(qf17.kind == scientific::F_NONE);
    let qr17, qg17 = scientific::lsmr(qa17, [1.257, 0.848, 1.873, 0.023, 0.791, -0.981, -1.083, -0.835], 0.0, 1.0e-6, 1.0e-6, 1.0e8, 0);
    assert(qg17.kind == scientific::F_NONE && qr17.istop == 2 && qr17.itn == 3);
    assert(common::all_near(qr17.x, [-0.06721555046513701, 0.0077930505369293475, -0.15235629905358294]));
    let qe17 = common_3::ni_nd([5], [qr17.normr, qr17.normar, qr17.norma, qr17.conda, qr17.normx]);
    assert(common::all_near(qe17, [3.022821432432123, 0.00000000000000027572758250446326, 6.329771085908242, 1.701450290792431, 0.16670663970672348]));
}

fn test_sparse_kronsum_pins0() {
    let a, fa = scientific::csr_from_dense(common::mx(2, 2, [1.0, 2.0, 0.0, 3.0]));
    let b, fb = scientific::csr_from_dense(common::mx(2, 2, [0.0, 5.0, 6.0, 0.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 4 && k.cols == 4 && k.nnz == 10);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [1.0, 2.0, 5.0, 0.0, 0.0, 3.0, 0.0, 5.0, 6.0, 0.0, 1.0, 2.0, 0.0, 6.0, 0.0, 3.0]));
    return;
}

fn test_sparse_kronsum_pins1() {
    let a, fa = scientific::csr_from_dense(common::mx(2, 2, [1.0, 2.0, 0.0, 3.0]));
    let b, fb = scientific::csr_from_dense(common::mx(3, 3, [0.0, 5.0, 1.0, 6.0, 0.0, 0.0, 0.0, 0.0, 2.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 6 && k.cols == 6 && k.nnz == 15);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [1.0, 2.0, 5.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 5.0, 0.0, 1.0, 6.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]));
    return;
}

fn test_sparse_kronsum_pins2() {
    let a, fa = scientific::csr_from_dense(common::mx(1, 1, [2.0]));
    let b, fb = scientific::csr_from_dense(common::mx(1, 1, [3.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 1 && k.cols == 1 && k.nnz == 1);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [5.0]));
    return;
}

fn test_sparse_kronsum_pins3() {
    let a, fa = scientific::csr_from_dense(common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]));
    let b, fb = scientific::csr_from_dense(common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 4 && k.cols == 4 && k.nnz == 0);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    return;
}

fn test_sparse_kronsum_pins4() {
    let a, fa = scientific::csr_from_dense(common::mx(2, 2, [1.0, 0.0, 0.0, 2.0]));
    let b, fb = scientific::csr_from_dense(common::mx(2, 2, [1.0, 0.0, 0.0, 1.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 4 && k.cols == 4 && k.nnz == 4);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0]));
    return;
}

fn test_sparse_kronsum_pins5() {
    let a, fa = scientific::csr_from_dense(common::mx(1, 1, [0.659]));
    let b, fb = scientific::csr_from_dense(common::mx(2, 2, [0.908, 0.0, 0.725, 0.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 2 && k.cols == 2 && k.nnz == 3);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [1.5670000000000002, 0.0, 0.725, 0.659]));
    return;
}

fn test_sparse_kronsum_pins6() {
    let a, fa = scientific::csr_from_dense(common::mx(3, 3, [0.901, 0.683, 0.0, 0.0, 0.962, 0.958, 0.0, 0.775, 0.0]));
    let b, fb = scientific::csr_from_dense(common::mx(3, 3, [0.892, 0.901, 0.0, 0.923, 0.763, 0.0, 0.0, 0.0, 0.853]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 9 && k.cols == 9 && k.nnz == 24);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [1.7930000000000001, 0.683, 0.0, 0.901, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.854, 0.958, 0.0, 0.901, 0.0, 0.0, 0.0, 0.0, 0.0, 0.775, 0.892, 0.0, 0.0, 0.901, 0.0, 0.0, 0.0, 0.923, 0.0, 0.0, 1.6640000000000001, 0.683, 0.0, 0.0, 0.0, 0.0, 0.0, 0.923, 0.0, 0.0, 1.725, 0.958, 0.0, 0.0, 0.0, 0.0, 0.0, 0.923, 0.0, 0.775, 0.763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.754, 0.683, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.815, 0.958, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.775, 0.853]));
    return;
}

fn test_sparse_kronsum_pins7() {
    let a, fa = scientific::csr_from_dense(common::mx(4, 4, [0.781, 0.99, 0.0, 0.0, 0.647, 0.0, 0.769, 0.0, 0.0, 0.0, 0.0, 0.0, 0.873, 0.931, 0.0, 0.0]));
    let b, fb = scientific::csr_from_dense(common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 8 && k.cols == 8 && k.nnz == 12);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [0.781, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.647, 0.0, 0.769, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.873, 0.931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.781, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.647, 0.0, 0.769, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.873, 0.931, 0.0, 0.0]));
    return;
}

fn test_sparse_kronsum_pins8() {
    let a, fa = scientific::csr_from_dense(common::mx(5, 5, [0.0, 0.906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.931, 0.0, 0.0, 0.747, 0.865, 0.666, 0.821, 0.0, 0.0, 0.833, 0.806, 0.0, 0.0, 0.847, 0.0, 0.0, 0.0, 0.814]));
    let b, fb = scientific::csr_from_dense(common::mx(5, 5, [0.0, 0.648, 0.0, 0.0, 0.0, 0.668, 0.0, 0.731, 0.0, 0.0, 0.6, 0.0, 0.0, 0.947, 0.779, 0.0, 0.653, 0.0, 0.0, 0.844, 0.0, 0.974, 0.0, 0.0, 0.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 25 && k.cols == 25 && k.nnz == 95);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [0.0, 0.906, 0.0, 0.0, 0.0, 0.648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.931, 0.0, 0.0, 0.0, 0.648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.747, 0.865, 0.666, 0.821, 0.0, 0.0, 0.0, 0.648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.833, 0.806, 0.0, 0.0, 0.0, 0.0, 0.0, 0.648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.847, 0.0, 0.0, 0.0, 0.814, 0.0, 0.0, 0.0, 0.0, 0.648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.668, 0.0, 0.0, 0.0, 0.0, 0.0, 0.906, 0.0, 0.0, 0.0, 0.731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.668, 0.0, 0.0, 0.0, 0.0, 0.0, 0.931, 0.0, 0.0, 0.0, 0.731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.668, 0.0, 0.0, 0.747, 0.865, 0.666, 0.821, 0.0, 0.0, 0.0, 0.731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.668, 0.0, 0.0, 0.833, 0.806, 0.0, 0.0, 0.0, 0.0, 0.0, 0.731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.668, 0.847, 0.0, 0.0, 0.0, 0.814, 0.0, 0.0, 0.0, 0.0, 0.731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.906, 0.0, 0.0, 0.0, 0.947, 0.0, 0.0, 0.0, 0.0, 0.779, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.931, 0.0, 0.0, 0.0, 0.947, 0.0, 0.0, 0.0, 0.0, 0.779, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.747, 0.865, 0.666, 0.821, 0.0, 0.0, 0.0, 0.947, 0.0, 0.0, 0.0, 0.0, 0.779, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.833, 0.806, 0.0, 0.0, 0.0, 0.0, 0.0, 0.947, 0.0, 0.0, 0.0, 0.0, 0.779, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.847, 0.0, 0.0, 0.0, 0.814, 0.0, 0.0, 0.0, 0.0, 0.947, 0.0, 0.0, 0.0, 0.0, 0.779, 0.0, 0.0, 0.0, 0.0, 0.0, 0.653, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.906, 0.0, 0.0, 0.0, 0.844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.653, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.931, 0.0, 0.0, 0.0, 0.844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.653, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.747, 0.865, 0.666, 0.821, 0.0, 0.0, 0.0, 0.844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.653, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.833, 0.806, 0.0, 0.0, 0.0, 0.0, 0.0, 0.844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.653, 0.0, 0.0, 0.0, 0.0, 0.0, 0.847, 0.0, 0.0, 0.0, 0.814, 0.0, 0.0, 0.0, 0.0, 0.844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.747, 0.865, 0.666, 0.821, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.833, 0.806, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.847, 0.0, 0.0, 0.0, 0.814]));
    return;
}

fn test_sparse_kronsum_pins9() {
    let a, fa = scientific::csr_from_dense(common::mx(2, 2, [0.998, 0.0, 0.671, 0.0]));
    let b, fb = scientific::csr_from_dense(common::mx(4, 4, [0.68, 0.0, 0.0, 0.0, 0.74, 0.0, 0.0, 0.792, 0.865, 0.868, 0.0, 0.0, 0.94, 0.0, 0.0, 0.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 8 && k.cols == 8 && k.nnz == 19);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [1.678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.671, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.74, 0.0, 0.998, 0.0, 0.0, 0.0, 0.792, 0.0, 0.0, 0.74, 0.671, 0.0, 0.0, 0.0, 0.0, 0.792, 0.865, 0.0, 0.868, 0.0, 0.998, 0.0, 0.0, 0.0, 0.0, 0.865, 0.0, 0.868, 0.671, 0.0, 0.0, 0.0, 0.94, 0.0, 0.0, 0.0, 0.0, 0.0, 0.998, 0.0, 0.0, 0.94, 0.0, 0.0, 0.0, 0.0, 0.671, 0.0]));
    return;
}

fn test_sparse_kronsum_pins10() {
    let a, fa = scientific::csr_from_dense(common::mx(3, 3, [0.806, 0.938, 0.0, 0.902, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let b, fb = scientific::csr_from_dense(common::mx(1, 1, [0.604]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 3 && k.cols == 3 && k.nnz == 5);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [1.4100000000000001, 0.938, 0.0, 0.902, 0.604, 0.0, 0.0, 0.0, 0.604]));
    return;
}

fn test_sparse_kronsum_pins11() {
    let a, fa = scientific::csr_from_dense(common::mx(1, 1, [0.765]));
    let b, fb = scientific::csr_from_dense(common::mx(1, 1, [0.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 1 && k.cols == 1 && k.nnz == 1);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [0.765]));
    return;
}

fn test_sparse_kronsum_pins12() {
    let a, fa = scientific::csr_from_dense(common::mx(4, 4, [0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.777, 0.0, 0.841, 0.0, 0.0, 0.888, 0.0, 0.0, 0.607, 0.0]));
    let b, fb = scientific::csr_from_dense(common::mx(5, 5, [0.908, 0.0, 0.0, 0.0, 0.0, 0.0, 0.877, 0.94, 0.607, 0.68, 0.95, 0.628, 0.0, 0.0, 0.0, 0.0, 0.709, 0.0, 0.0, 0.828, 0.0, 0.0, 0.0, 0.779, 0.0]));
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.rows == 20 && k.cols == 20 && k.nnz == 63);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [1.828, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.908, 0.777, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.841, 0.0, 0.908, 0.888, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.607, 0.908, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7970000000000002, 0.0, 0.0, 0.0, 0.94, 0.0, 0.0, 0.0, 0.607, 0.0, 0.0, 0.0, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.877, 0.777, 0.0, 0.0, 0.94, 0.0, 0.0, 0.0, 0.607, 0.0, 0.0, 0.0, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.841, 0.0, 0.877, 0.888, 0.0, 0.0, 0.94, 0.0, 0.0, 0.0, 0.607, 0.0, 0.0, 0.0, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.607, 0.877, 0.0, 0.0, 0.0, 0.94, 0.0, 0.0, 0.0, 0.607, 0.0, 0.0, 0.0, 0.68, 0.95, 0.0, 0.0, 0.0, 0.628, 0.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.628, 0.0, 0.0, 0.0, 0.0, 0.777, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.628, 0.0, 0.841, 0.0, 0.0, 0.888, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.628, 0.0, 0.0, 0.607, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.709, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 0.828, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.709, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.777, 0.0, 0.0, 0.828, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.709, 0.0, 0.0, 0.0, 0.0, 0.0, 0.841, 0.0, 0.0, 0.888, 0.0, 0.0, 0.828, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.709, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.607, 0.0, 0.0, 0.0, 0.0, 0.828, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.779, 0.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.779, 0.0, 0.0, 0.0, 0.0, 0.777, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.779, 0.0, 0.841, 0.0, 0.0, 0.888, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.779, 0.0, 0.0, 0.607, 0.0]));
    return;
}

fn test_sparse_kronsum_pins_zero() {
    let a, fa = scientific::csr_from_triplets(2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 2.0]);
    let b, fb = scientific::csr_eye(2);
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE && a.nnz == 2);
    let k, fk = scientific::csr_kronsum(a, b);
    assert(fk.kind == scientific::F_NONE && k.nnz == 4);
    let kd, fd = scientific::csr_to_dense(k);
    assert(fd.kind == scientific::F_NONE && common::all_near(kd, [1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0]));
    return;
}

// The faults of the block matrix and the columnwise diagonals.
fn test_sparse_bmat_hostile() {
    let a, fa = scientific::from_values([1.0, 2.0, 0.0, 3.0]);
    assert(fa.kind == scientific::F_NONE);
    let a2, fr = scientific::reshape(a, [2, 2]);
    assert(fr.kind == scientific::F_NONE);
    let ac, fc = scientific::csr_from_dense(a2);
    assert(fc.kind == scientific::F_NONE);
    let u, fu = scientific::from_values([1.0]);
    assert(fu.kind == scientific::F_NONE);
    let u2, fv = scientific::reshape(u, [1, 1]);
    assert(fv.kind == scientific::F_NONE);
    let uc, fw = scientific::csr_from_dense(u2);
    assert(fw.kind == scientific::F_NONE);

    // The sides of the grid stand between one and the limit.
    var g0, fg0 = scientific::bmat_new(0, 2);
    assert(fg0.kind == scientific::F_RANGE);
    var g1, fg1 = scientific::bmat_new(2, 0);
    assert(fg1.kind == scientific::F_RANGE);
    var g2, fg2 = scientific::bmat_new(1048577, 2);
    assert(fg2.kind == scientific::F_RANGE);
    var g3, fg3 = scientific::bmat_new(2, 1048577);
    assert(fg3.kind == scientific::F_RANGE);

    // The place of a block stands inside the grid.
    var g, fgn = scientific::bmat_new(2, 2);
    assert(fgn.kind == scientific::F_NONE);
    assert(scientific::bmat_put(g, 0.0 / 0.0, 0.0, ac).kind == scientific::F_RANGE);
    assert(scientific::bmat_put(g, 1.0 / 0.0, 0.0, ac).kind == scientific::F_RANGE);
    assert(scientific::bmat_put(g, 0.5, 0.0, ac).kind == scientific::F_RANGE);
    assert(scientific::bmat_put(g, -1.0, 0.0, ac).kind == scientific::F_RANGE);
    assert(scientific::bmat_put(g, 2.0, 0.0, ac).kind == scientific::F_RANGE);
    assert(scientific::bmat_put(g, 0.0, 0.0 / 0.0, ac).kind == scientific::F_RANGE);
    assert(scientific::bmat_put(g, 0.0, 0.5, ac).kind == scientific::F_RANGE);
    assert(scientific::bmat_put(g, 0.0, -1.0, ac).kind == scientific::F_RANGE);
    assert(scientific::bmat_put(g, 0.0, 2.0, ac).kind == scientific::F_RANGE);
    assert(g.n == 0 && g.buf.len == 0);

    // A place that holds a block already is a fault.
    assert(scientific::bmat_put(g, 0.0, 0.0, ac).kind == scientific::F_NONE);
    assert(scientific::bmat_put(g, 0.0, 0.0, ac).kind == scientific::F_RANGE);
    assert(g.n == 1);

    // The blocks of one grid row share their height.
    var gh, fgh = scientific::bmat_new(2, 2);
    assert(fgh.kind == scientific::F_NONE);
    assert(scientific::bmat_put(gh, 0.0, 0.0, ac).kind == scientific::F_NONE);
    assert(scientific::bmat_put(gh, 0.0, 1.0, uc).kind == scientific::F_SHAPE);
    assert(gh.n == 1);

    // The blocks of one grid column share their width.
    var gw, fgw = scientific::bmat_new(2, 2);
    assert(fgw.kind == scientific::F_NONE);
    assert(scientific::bmat_put(gw, 0.0, 0.0, ac).kind == scientific::F_NONE);
    assert(scientific::bmat_put(gw, 1.0, 0.0, uc).kind == scientific::F_SHAPE);
    assert(gw.n == 1);

    // A value of the block that is not finite is a fault.
    let nv, fnv = scientific::from_values([1.0 / 0.0]);
    assert(fnv.kind == scientific::F_NONE);
    let nv2, fnv2 = scientific::reshape(nv, [1, 1]);
    assert(fnv2.kind == scientific::F_NONE);
    let nc, fnc = scientific::csr_from_dense(nv2);
    assert(fnc.kind == scientific::F_NONE);
    var gv, fgv = scientific::bmat_new(2, 2);
    assert(fgv.kind == scientific::F_NONE);
    assert(scientific::bmat_put(gv, 0.0, 0.0, nc).kind == scientific::F_VALUE);
    assert(gv.n == 0 && gv.buf.len == 0);

    // A grid that holds no block is a fault.
    var ge, fge = scientific::bmat_new(2, 2);
    assert(fge.kind == scientific::F_NONE);
    let er, fer = scientific::csr_bmat(ge);
    assert(fer.kind == scientific::F_RANGE);

    // The count of the stored values stands below the limit.
    var lim, flim = scientific::bmat_new(2, 2);
    assert(flim.kind == scientific::F_NONE);
    lim.n = 1;
    lim.buf.push(0.0); lim.buf.push(0.0); lim.buf.push(1.0); lim.buf.push(1.0); lim.buf.push(134217729.0);
    assert(scientific::bmat_put(lim, 1.0, 0.0, uc).kind == scientific::F_LIMIT);
    assert(lim.n == 1 && lim.buf.len == 5);

    // The buffer of the builder stands below the limit. The block claims
    // so many values that the buffer would grow past the limit.
    var siz, fsiz = scientific::bmat_new(2, 2);
    assert(fsiz.kind == scientific::F_NONE);
    siz.n = 1;
    siz.buf.push(0.0); siz.buf.push(0.0); siz.buf.push(1.0); siz.buf.push(1.0); siz.buf.push(1.0);
    let bigc = scientific::Csr { rows: 1, cols: 1, nnz: 89478485, buf: [] };
    assert(scientific::bmat_put(siz, 1.0, 0.0, bigc).kind == scientific::F_LIMIT);
    assert(siz.n == 1 && siz.buf.len == 5);

    // The layout of the block holds the count and the values together.
    // A block that lies about its count is a fault.
    var gl, fgl = scientific::bmat_new(2, 2);
    assert(fgl.kind == scientific::F_NONE);
    let nan = 0.0 / 0.0;
    // The count says one value, but the row pointers reach three.
    let lie = scientific::Csr { rows: 1, cols: 1, nnz: 1, buf: [0.0, 3.0, 1.0, 1.0, 0.0, nan] };
    assert(scientific::bmat_put(gl, 0.0, 0.0, lie).kind == scientific::F_SHAPE);
    assert(gl.n == 0 && gl.buf.len == 0);
    // The buffer of the block is too short for its count.
    let short = scientific::Csr { rows: 1, cols: 1, nnz: 1, buf: [] };
    assert(scientific::bmat_put(gl, 0.0, 0.0, short).kind == scientific::F_SHAPE);
    assert(gl.n == 0 && gl.buf.len == 0);
    // The first row pointer of the block is not zero.
    let head = scientific::Csr { rows: 1, cols: 1, nnz: 1, buf: [1.0, 2.0, 1.0, 1.0, 0.0, 1.0] };
    assert(scientific::bmat_put(gl, 0.0, 0.0, head).kind == scientific::F_SHAPE);
    // The row pointers of the block do not rise.
    let fall = scientific::Csr { rows: 2, cols: 1, nnz: 2, buf: [0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0] };
    assert(scientific::bmat_put(gl, 0.0, 0.0, fall).kind == scientific::F_SHAPE);
    // A column of the block lies outside the block.
    let wide = scientific::Csr { rows: 1, cols: 1, nnz: 1, buf: [0.0, 1.0, 1.0, 1.0, 0.0, 1.0] };
    assert(scientific::bmat_put(gl, 0.0, 0.0, wide).kind == scientific::F_SHAPE);
    assert(gl.n == 0 && gl.buf.len == 0);

    // The sum of the sides of the grid stands below the limit.
    var big, fbig = scientific::bmat_new(2, 2);
    assert(fbig.kind == scientific::F_NONE);
    big.n = 2;
    big.buf.push(0.0); big.buf.push(0.0); big.buf.push(1073741824.0); big.buf.push(1073741824.0); big.buf.push(0.0);
    big.buf.push(1.0); big.buf.push(0.0); big.buf.push(1073741824.0); big.buf.push(1073741824.0); big.buf.push(0.0);
    let bigm, fbigm = scientific::csr_bmat(big);
    assert(fbigm.kind == scientific::F_RANGE);

    // The data of the diagonals must be a matrix.
    let v1, fv1 = scientific::from_values([1.0, 2.0, 3.0]);
    assert(fv1.kind == scientific::F_NONE);
    let e0, fe0 = scientific::spdiags(v1, [0.0], 2, 2);
    assert(fe0.kind == scientific::F_SHAPE);

    // The count of the offsets must match the count of the rows.
    let m2, fm2 = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    assert(fm2.kind == scientific::F_NONE);
    let m2r, fmr = scientific::reshape(m2, [2, 2]);
    assert(fmr.kind == scientific::F_NONE);
    let e1, fe1 = scientific::spdiags(m2r, [0.0], 2, 2);
    assert(fe1.kind == scientific::F_SHAPE);

    // The sides of the matrix of diagonals stand between one and the limit.
    let e2, fe2 = scientific::spdiags(m2r, [0.0, 1.0], 0, 2);
    assert(fe2.kind == scientific::F_RANGE);
    let e3, fe3 = scientific::spdiags(m2r, [0.0, 1.0], 2, 1073741825);
    assert(fe3.kind == scientific::F_RANGE);

    // The offsets stand inside the matrix.
    let o1, fo1 = scientific::from_values([1.0, 2.0]);
    assert(fo1.kind == scientific::F_NONE);
    let o1r, for1 = scientific::reshape(o1, [1, 2]);
    assert(for1.kind == scientific::F_NONE);
    let e4, fe4 = scientific::spdiags(o1r, [0.0 / 0.0], 2, 2);
    assert(fe4.kind == scientific::F_RANGE);
    let e5, fe5 = scientific::spdiags(o1r, [1.0 / 0.0], 2, 2);
    assert(fe5.kind == scientific::F_RANGE);
    let e6, fe6 = scientific::spdiags(o1r, [0.5], 2, 2);
    assert(fe6.kind == scientific::F_RANGE);
    let e7, fe7 = scientific::spdiags(o1r, [-2.0], 2, 2);
    assert(fe7.kind == scientific::F_RANGE);
    let e8, fe8 = scientific::spdiags(o1r, [2.0], 2, 2);
    assert(fe8.kind == scientific::F_RANGE);

    // Two offsets that are the same are a fault.
    let e9, fe9 = scientific::spdiags(m2r, [0.0, 0.0], 2, 2);
    assert(fe9.kind == scientific::F_RANGE);

    // A value of the data that is not finite is a fault.
    let n1, fn1 = scientific::from_values([1.0 / 0.0, 2.0]);
    assert(fn1.kind == scientific::F_NONE);
    let n1r, fn1r = scientific::reshape(n1, [1, 2]);
    assert(fn1r.kind == scientific::F_NONE);
    let e10, fe10 = scientific::spdiags(n1r, [0.0], 2, 2);
    assert(fe10.kind == scientific::F_VALUE);

    // The count of the stored values stands below the limit.
    let e11, fe11 = scientific::spdiags(m2r, [0.0, 1.0], 1073741824, 1073741824);
    assert(fe11.kind == scientific::F_LIMIT);

    // The grid of blocks gives the same matrix as the calls that stack.
    var gg, fgg = scientific::bmat_new(1, 2);
    assert(fgg.kind == scientific::F_NONE);
    assert(scientific::bmat_put(gg, 0.0, 0.0, ac).kind == scientific::F_NONE);
    assert(scientific::bmat_put(gg, 0.0, 1.0, ac).kind == scientific::F_NONE);
    let gm, fgm = scientific::csr_bmat(gg);
    assert(fgm.kind == scientific::F_NONE);
    let hs, fhs = scientific::csr_hstack(ac, ac);
    assert(fhs.kind == scientific::F_NONE);
    let gd, fgd = scientific::csr_to_dense(gm);
    let hd, fhd = scientific::csr_to_dense(hs);
    assert(fgd.kind == scientific::F_NONE && fhd.kind == scientific::F_NONE && common::all_near(gd, hd.data));

    var gv2, fgv2 = scientific::bmat_new(2, 1);
    assert(fgv2.kind == scientific::F_NONE);
    assert(scientific::bmat_put(gv2, 0.0, 0.0, ac).kind == scientific::F_NONE);
    assert(scientific::bmat_put(gv2, 1.0, 0.0, ac).kind == scientific::F_NONE);
    let vm, fvm = scientific::csr_bmat(gv2);
    assert(fvm.kind == scientific::F_NONE);
    let vs, fvs = scientific::csr_vstack(ac, ac);
    assert(fvs.kind == scientific::F_NONE);
    let vd, fvd = scientific::csr_to_dense(vm);
    let vsd, fvsd = scientific::csr_to_dense(vs);
    assert(fvd.kind == scientific::F_NONE && fvsd.kind == scientific::F_NONE && common::all_near(vd, vsd.data));

    var gb2, fgb2 = scientific::bmat_new(2, 2);
    assert(fgb2.kind == scientific::F_NONE);
    assert(scientific::bmat_put(gb2, 0.0, 0.0, ac).kind == scientific::F_NONE);
    assert(scientific::bmat_put(gb2, 1.0, 1.0, ac).kind == scientific::F_NONE);
    let bm, fbm = scientific::csr_bmat(gb2);
    assert(fbm.kind == scientific::F_NONE);
    let bd, fbd = scientific::csr_block_diag(ac, ac);
    assert(fbd.kind == scientific::F_NONE);
    let bdd, fbdd = scientific::csr_to_dense(bm);
    let bds, fbds = scientific::csr_to_dense(bd);
    assert(fbdd.kind == scientific::F_NONE && fbds.kind == scientific::F_NONE && common::all_near(bdd, bds.data));
}
