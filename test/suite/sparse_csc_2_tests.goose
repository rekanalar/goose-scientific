// The tests of the group sparse_csc. Run them with SCIENTIFIC_SUITES=sparse_csc.
namespace sparse_csc_2_tests;

import scientific from "../..";
import os;
import common_3;
import sparse_csc_tests;

fn test_sparse_csc_pins13() {
    let t39, g39 = scientific::csc_from_dense(common_3::ni_nd([3, 2], [0.0, -0.363, 0.0, -0.034, -2.43, 0.0]));
    assert(g39.kind == scientific::F_NONE);
    let u39, h39 = scientific::csc_transpose(t39);
    assert(h39.kind == scientific::F_NONE);
    let v39, i39 = scientific::csc_to_dense(u39);
    assert(i39.kind == scientific::F_NONE);
    assert(v39.shape == [2, 3] && common_3::ms_same(v39, [0.0, 0.0, -2.43, -0.363, -0.034, 0.0]));
    let w39, j39 = scientific::csc_matvec(t39, [-1.058, 1.149]);
    assert(j39.kind == scientific::F_NONE);
    assert(w39.shape == [3] && common_3::ms_same(w39, [-0.417087, -0.039066000000000004, 2.5709400000000002]));
    let m39, n39 = scientific::csc_matmul(t39, scientific::csc_from_dense(common_3::ni_nd([2, 3], [0.0, -0.406, 0.0, 0.553, 0.149, 0.0])));
    assert(n39.kind == scientific::F_NONE);
    let o39, p39 = scientific::csc_to_dense(m39);
    assert(p39.kind == scientific::F_NONE);
    assert(o39.shape == [3, 3] && common_3::ms_same(o39, [-0.200739, -0.054086999999999996, 0.0, -0.018802000000000003, -0.005066, 0.0, 0.0, 0.9865800000000001, 0.0]));
    let q39, r39 = scientific::csc_add(t39, scientific::csc_from_dense(common_3::ni_nd([3, 2], [0.0, 0.0, 0.0, 3.678, 0.0, 0.0])));
    assert(r39.kind == scientific::F_NONE);
    let s39, t239 = scientific::csc_to_dense(q39);
    assert(t239.kind == scientific::F_NONE);
    assert(s39.shape == [3, 2] && common_3::ms_same(s39, [0.0, -0.363, 0.0, 3.644, -2.43, 0.0]));
    let u239, v239 = scientific::csc_sub(t39, scientific::csc_from_dense(common_3::ni_nd([3, 2], [0.0, 0.0, 0.0, 3.678, 0.0, 0.0])));
    assert(v239.kind == scientific::F_NONE);
    let w239, x239 = scientific::csc_to_dense(u239);
    assert(x239.kind == scientific::F_NONE);
    assert(w239.shape == [3, 2] && common_3::ms_same(w239, [0.0, -0.363, 0.0, -3.7119999999999997, -2.43, 0.0]));
    let y39, z39 = scientific::csc_diagonal(t39);
    assert(z39.kind == scientific::F_NONE);
    assert(y39.shape == [2] && common_3::ms_same(y39, [0.0, -0.034]));
    let aa39, ab39 = scientific::csc_sum(t39, 0);
    assert(ab39.kind == scientific::F_NONE);
    assert(aa39.shape == [2] && common_3::ms_same(aa39, [-2.43, -0.397]));
    let ac39, ad39 = scientific::csc_sum(t39, 1);
    assert(ad39.kind == scientific::F_NONE);
    assert(ac39.shape == [3] && common_3::ms_same(ac39, [-0.363, -0.034, -2.43]));
    let c40, g40 = scientific::csc_from_dense(common_3::ni_nd([5, 6], [0.0, -2.365, 2.825, -3.735, 2.223, 3.853, 0.0, 0.0, 0.0, 1.484, 2.64, 0.0, 3.685, 0.0, -3.172, 0.0, 0.0, 3.933, 0.0, 0.0, 0.0, -3.66, 0.0, 0.0, -3.068, 0.0, 0.0, 0.0, 0.881, -0.071]));
    assert(g40.kind == scientific::F_NONE);
    let d40, h40 = scientific::csc_to_dense(c40);
    assert(h40.kind == scientific::F_NONE);
    assert(d40.shape == [5, 6] && common_3::ms_same(d40, [0.0, -2.365, 2.825, -3.735, 2.223, 3.853, 0.0, 0.0, 0.0, 1.484, 2.64, 0.0, 3.685, 0.0, -3.172, 0.0, 0.0, 3.933, 0.0, 0.0, 0.0, -3.66, 0.0, 0.0, -3.068, 0.0, 0.0, 0.0, 0.881, -0.071]));
    let t41, g41 = scientific::csc_from_dense(common_3::ni_nd([5, 6], [0.0, -2.365, 2.825, -3.735, 2.223, 3.853, 0.0, 0.0, 0.0, 1.484, 2.64, 0.0, 3.685, 0.0, -3.172, 0.0, 0.0, 3.933, 0.0, 0.0, 0.0, -3.66, 0.0, 0.0, -3.068, 0.0, 0.0, 0.0, 0.881, -0.071]));
    assert(g41.kind == scientific::F_NONE);
    let u41, h41 = scientific::csc_transpose(t41);
    assert(h41.kind == scientific::F_NONE);
    let v41, i41 = scientific::csc_to_dense(u41);
    assert(i41.kind == scientific::F_NONE);
    assert(v41.shape == [6, 5] && common_3::ms_same(v41, [0.0, 0.0, 3.685, 0.0, -3.068, -2.365, 0.0, 0.0, 0.0, 0.0, 2.825, 0.0, -3.172, 0.0, 0.0, -3.735, 1.484, 0.0, -3.66, 0.0, 2.223, 2.64, 0.0, 0.0, 0.881, 3.853, 0.0, 3.933, 0.0, -0.071]));
    let w41, j41 = scientific::csc_matvec(t41, [-0.984, 0.366, -0.3, -1.635, -0.946, 1.385]);
    assert(j41.kind == scientific::F_NONE);
    assert(w41.shape == [5] && common_3::ms_same(w41, [7.627082, -4.923780000000001, 2.7727649999999993, 5.984100000000001, 2.087151]));
    let m41, n41 = scientific::csc_matmul(t41, scientific::csc_from_dense(common_3::ni_nd([6, 3], [0.0, 0.0, 2.445, 0.0, 0.0, 0.0, 2.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.32, 0.0, -1.956, -2.759, -0.83])));
    assert(n41.kind == scientific::F_NONE);
    let o41, p41 = scientific::csc_to_dense(m41);
    assert(p41.kind == scientific::F_NONE);
    assert(o41.shape == [5, 3] && common_3::ms_same(o41, [0.8820319999999996, -5.473067, -3.19799, 0.0, 6.1248, 0.0, -17.145508, -10.851147, 5.745435, 0.0, 0.0, 0.0, 0.13887599999999997, 2.2398089999999997, -7.442329999999999]));
    let q41, r41 = scientific::csc_add(t41, scientific::csc_from_dense(common_3::ni_nd([5, 6], [-2.045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.136, -1.003, 0.0, -1.216, 0.36, 0.0, 0.0, 0.0, 1.916, 0.0, -0.752, 2.687, 0.0, 0.78, 1.303, 0.0, 0.0, 0.0, 1.983, -0.308, 0.0, 0.0])));
    assert(r41.kind == scientific::F_NONE);
    let s41, t241 = scientific::csc_to_dense(q41);
    assert(t241.kind == scientific::F_NONE);
    assert(s41.shape == [5, 6] && common_3::ms_same(s41, [-2.045, -2.365, 2.825, -3.735, 2.223, 3.853, 0.0, 0.0, -3.136, 0.4810000000000001, 2.64, -1.216, 4.045, 0.0, -3.172, 0.0, 1.916, 3.933, -0.752, 2.687, 0.0, -2.88, 1.303, 0.0, -3.068, 0.0, 1.983, -0.308, 0.881, -0.071]));
    let u241, v241 = scientific::csc_sub(t41, scientific::csc_from_dense(common_3::ni_nd([5, 6], [-2.045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.136, -1.003, 0.0, -1.216, 0.36, 0.0, 0.0, 0.0, 1.916, 0.0, -0.752, 2.687, 0.0, 0.78, 1.303, 0.0, 0.0, 0.0, 1.983, -0.308, 0.0, 0.0])));
    assert(v241.kind == scientific::F_NONE);
    let w241, x241 = scientific::csc_to_dense(u241);
    assert(x241.kind == scientific::F_NONE);
    assert(w241.shape == [5, 6] && common_3::ms_same(w241, [2.045, -2.365, 2.825, -3.735, 2.223, 3.853, 0.0, 0.0, 3.136, 2.487, 2.64, 1.216, 3.325, 0.0, -3.172, 0.0, -1.916, 3.933, 0.752, -2.687, 0.0, -4.44, -1.303, 0.0, -3.068, 0.0, -1.983, 0.308, 0.881, -0.071]));
    let y41, z41 = scientific::csc_diagonal(t41);
    assert(z41.kind == scientific::F_NONE);
    assert(y41.shape == [5] && common_3::ms_same(y41, [0.0, 0.0, -3.172, -3.66, 0.881]));
    let aa41, ab41 = scientific::csc_sum(t41, 0);
    assert(ab41.kind == scientific::F_NONE);
    assert(aa41.shape == [6] && common_3::ms_same(aa41, [0.617, -2.365, -0.347, -5.911, 5.744, 7.715]));
    let ac41, ad41 = scientific::csc_sum(t41, 1);
    assert(ad41.kind == scientific::F_NONE);
    assert(ac41.shape == [5] && common_3::ms_same(ac41, [2.801, 4.1240000000000006, 4.446, -3.66, -2.2580000000000005]));
}

fn test_sparse_csc_pins14() {
    let c42, g42 = scientific::csc_from_dense(common_3::ni_nd([3, 1], [-0.516, -1.456, 0.312]));
    assert(g42.kind == scientific::F_NONE);
    let d42, h42 = scientific::csc_to_dense(c42);
    assert(h42.kind == scientific::F_NONE);
    assert(d42.shape == [3, 1] && common_3::ms_same(d42, [-0.516, -1.456, 0.312]));
    let t43, g43 = scientific::csc_from_dense(common_3::ni_nd([3, 1], [-0.516, -1.456, 0.312]));
    assert(g43.kind == scientific::F_NONE);
    let u43, h43 = scientific::csc_transpose(t43);
    assert(h43.kind == scientific::F_NONE);
    let v43, i43 = scientific::csc_to_dense(u43);
    assert(i43.kind == scientific::F_NONE);
    assert(v43.shape == [1, 3] && common_3::ms_same(v43, [-0.516, -1.456, 0.312]));
    let w43, j43 = scientific::csc_matvec(t43, [-0.573]);
    assert(j43.kind == scientific::F_NONE);
    assert(w43.shape == [3] && common_3::ms_same(w43, [0.295668, 0.8342879999999999, -0.178776]));
    let m43, n43 = scientific::csc_matmul(t43, scientific::csc_from_dense(common_3::ni_nd([1, 3], [0.0, 0.0, 1.281])));
    assert(n43.kind == scientific::F_NONE);
    let o43, p43 = scientific::csc_to_dense(m43);
    assert(p43.kind == scientific::F_NONE);
    assert(o43.shape == [3, 3] && common_3::ms_same(o43, [0.0, 0.0, -0.660996, 0.0, 0.0, -1.865136, 0.0, 0.0, 0.39967199999999997]));
    let q43, r43 = scientific::csc_add(t43, scientific::csc_from_dense(common_3::ni_nd([3, 1], [-3.466, 0.0, 2.773])));
    assert(r43.kind == scientific::F_NONE);
    let s43, t243 = scientific::csc_to_dense(q43);
    assert(t243.kind == scientific::F_NONE);
    assert(s43.shape == [3, 1] && common_3::ms_same(s43, [-3.982, -1.456, 3.085]));
    let u243, v243 = scientific::csc_sub(t43, scientific::csc_from_dense(common_3::ni_nd([3, 1], [-3.466, 0.0, 2.773])));
    assert(v243.kind == scientific::F_NONE);
    let w243, x243 = scientific::csc_to_dense(u243);
    assert(x243.kind == scientific::F_NONE);
    assert(w243.shape == [3, 1] && common_3::ms_same(w243, [2.95, -1.456, -2.4610000000000003]));
    let y43, z43 = scientific::csc_diagonal(t43);
    assert(z43.kind == scientific::F_NONE);
    assert(y43.shape == [1] && common_3::ms_same(y43, [-0.516]));
    let aa43, ab43 = scientific::csc_sum(t43, 0);
    assert(ab43.kind == scientific::F_NONE);
    assert(aa43.shape == [1] && common_3::ms_same(aa43, [-1.66]));
    let ac43, ad43 = scientific::csc_sum(t43, 1);
    assert(ad43.kind == scientific::F_NONE);
    assert(ac43.shape == [3] && common_3::ms_same(ac43, [-0.516, -1.456, 0.312]));
    let c44, g44 = scientific::csc_from_dense(common_3::ni_nd([2, 2], [1.288, 0.0, 0.0, 0.0]));
    assert(g44.kind == scientific::F_NONE);
    let d44, h44 = scientific::csc_to_dense(c44);
    assert(h44.kind == scientific::F_NONE);
    assert(d44.shape == [2, 2] && common_3::ms_same(d44, [1.288, 0.0, 0.0, 0.0]));
}

fn test_sparse_csc_pins15() {
    let t45, g45 = scientific::csc_from_dense(common_3::ni_nd([2, 2], [1.288, 0.0, 0.0, 0.0]));
    assert(g45.kind == scientific::F_NONE);
    let u45, h45 = scientific::csc_transpose(t45);
    assert(h45.kind == scientific::F_NONE);
    let v45, i45 = scientific::csc_to_dense(u45);
    assert(i45.kind == scientific::F_NONE);
    assert(v45.shape == [2, 2] && common_3::ms_same(v45, [1.288, 0.0, 0.0, 0.0]));
    let w45, j45 = scientific::csc_matvec(t45, [1.251, 1.669]);
    assert(j45.kind == scientific::F_NONE);
    assert(w45.shape == [2] && common_3::ms_same(w45, [1.6112879999999998, 0.0]));
    let m45, n45 = scientific::csc_matmul(t45, scientific::csc_from_dense(common_3::ni_nd([2, 3], [0.0, 3.086, 0.0, 0.0, 0.0, 0.0])));
    assert(n45.kind == scientific::F_NONE);
    let o45, p45 = scientific::csc_to_dense(m45);
    assert(p45.kind == scientific::F_NONE);
    assert(o45.shape == [2, 3] && common_3::ms_same(o45, [0.0, 3.974768, 0.0, 0.0, 0.0, 0.0]));
    let q45, r45 = scientific::csc_add(t45, scientific::csc_from_dense(common_3::ni_nd([2, 2], [0.0, 0.0, 3.127, 1.683])));
    assert(r45.kind == scientific::F_NONE);
    let s45, t245 = scientific::csc_to_dense(q45);
    assert(t245.kind == scientific::F_NONE);
    assert(s45.shape == [2, 2] && common_3::ms_same(s45, [1.288, 0.0, 3.127, 1.683]));
    let u245, v245 = scientific::csc_sub(t45, scientific::csc_from_dense(common_3::ni_nd([2, 2], [0.0, 0.0, 3.127, 1.683])));
    assert(v245.kind == scientific::F_NONE);
    let w245, x245 = scientific::csc_to_dense(u245);
    assert(x245.kind == scientific::F_NONE);
    assert(w245.shape == [2, 2] && common_3::ms_same(w245, [1.288, 0.0, -3.127, -1.683]));
    let y45, z45 = scientific::csc_diagonal(t45);
    assert(z45.kind == scientific::F_NONE);
    assert(y45.shape == [2] && common_3::ms_same(y45, [1.288, 0.0]));
    let aa45, ab45 = scientific::csc_sum(t45, 0);
    assert(ab45.kind == scientific::F_NONE);
    assert(aa45.shape == [2] && common_3::ms_same(aa45, [1.288, 0.0]));
    let ac45, ad45 = scientific::csc_sum(t45, 1);
    assert(ad45.kind == scientific::F_NONE);
    assert(ac45.shape == [2] && common_3::ms_same(ac45, [1.288, 0.0]));
    let c46, g46 = scientific::csc_from_dense(common_3::ni_nd([5, 5], [0.0, 0.0, -2.685, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.345, 0.0, 2.169, 0.0, 0.0, 0.0, 0.0, 3.793, 0.0, 0.0, -0.342, 0.0, 0.07, 0.0, 0.0]));
    assert(g46.kind == scientific::F_NONE);
    let d46, h46 = scientific::csc_to_dense(c46);
    assert(h46.kind == scientific::F_NONE);
    assert(d46.shape == [5, 5] && common_3::ms_same(d46, [0.0, 0.0, -2.685, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.345, 0.0, 2.169, 0.0, 0.0, 0.0, 0.0, 3.793, 0.0, 0.0, -0.342, 0.0, 0.07, 0.0, 0.0]));
    let t47, g47 = scientific::csc_from_dense(common_3::ni_nd([5, 5], [0.0, 0.0, -2.685, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.345, 0.0, 2.169, 0.0, 0.0, 0.0, 0.0, 3.793, 0.0, 0.0, -0.342, 0.0, 0.07, 0.0, 0.0]));
    assert(g47.kind == scientific::F_NONE);
    let u47, h47 = scientific::csc_transpose(t47);
    assert(h47.kind == scientific::F_NONE);
    let v47, i47 = scientific::csc_to_dense(u47);
    assert(i47.kind == scientific::F_NONE);
    assert(v47.shape == [5, 5] && common_3::ms_same(v47, [0.0, 0.0, 3.345, 0.0, -0.342, 0.0, 0.0, 0.0, 0.0, 0.0, -2.685, 0.0, 2.169, 3.793, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let w47, j47 = scientific::csc_matvec(t47, [-1.52, 0.338, -1.183, 1.169, -1.687]);
    assert(j47.kind == scientific::F_NONE);
    assert(w47.shape == [5] && common_3::ms_same(w47, [3.176355, 0.0, -7.650327000000001, -4.487119000000001, 0.4370300000000001]));
    let m47, n47 = scientific::csc_matmul(t47, scientific::csc_from_dense(common_3::ni_nd([5, 3], [2.715, 0.0, 0.0, 0.0, 0.451, 0.0, 0.0, 0.0, 0.0, 0.0, -3.733, -0.577, 0.542, 0.0, 0.0])));
    assert(n47.kind == scientific::F_NONE);
    let o47, p47 = scientific::csc_to_dense(m47);
    assert(p47.kind == scientific::F_NONE);
    assert(o47.shape == [5, 3] && common_3::ms_same(o47, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.081675, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9285300000000001, 0.0, 0.0]));
    let q47, r47 = scientific::csc_add(t47, scientific::csc_from_dense(common_3::ni_nd([5, 5], [0.0, -3.932, -0.684, 0.0, 3.77, -1.555, 0.0, -1.157, -2.955, 0.0, 1.771, 0.87, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.268, 0.0, 0.0, 0.0, 0.0, 0.0, -1.654])));
    assert(r47.kind == scientific::F_NONE);
    let s47, t247 = scientific::csc_to_dense(q47);
    assert(t247.kind == scientific::F_NONE);
    assert(s47.shape == [5, 5] && common_3::ms_same(s47, [0.0, -3.932, -3.369, 0.0, 3.77, -1.555, 0.0, -1.157, -2.955, 0.0, 5.116, 0.87, 2.169, 0.0, 0.35, 0.0, 0.0, 3.793, 1.268, 0.0, -0.342, 0.0, 0.07, 0.0, -1.654]));
    let u247, v247 = scientific::csc_sub(t47, scientific::csc_from_dense(common_3::ni_nd([5, 5], [0.0, -3.932, -0.684, 0.0, 3.77, -1.555, 0.0, -1.157, -2.955, 0.0, 1.771, 0.87, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.268, 0.0, 0.0, 0.0, 0.0, 0.0, -1.654])));
    assert(v247.kind == scientific::F_NONE);
    let w247, x247 = scientific::csc_to_dense(u247);
    assert(x247.kind == scientific::F_NONE);
    assert(w247.shape == [5, 5] && common_3::ms_same(w247, [0.0, 3.932, -2.001, 0.0, -3.77, 1.555, 0.0, 1.157, 2.955, 0.0, 1.5740000000000003, -0.87, 2.169, 0.0, -0.35, 0.0, 0.0, 3.793, -1.268, 0.0, -0.342, 0.0, 0.07, 0.0, 1.654]));
    let y47, z47 = scientific::csc_diagonal(t47);
    assert(z47.kind == scientific::F_NONE);
    assert(y47.shape == [5] && common_3::ms_same(y47, [0.0, 0.0, 2.169, 0.0, 0.0]));
    let aa47, ab47 = scientific::csc_sum(t47, 0);
    assert(ab47.kind == scientific::F_NONE);
    assert(aa47.shape == [5] && common_3::ms_same(aa47, [3.003, 0.0, 3.347, 0.0, 0.0]));
    let ac47, ad47 = scientific::csc_sum(t47, 1);
    assert(ad47.kind == scientific::F_NONE);
    assert(ac47.shape == [5] && common_3::ms_same(ac47, [-2.685, 0.0, 5.514, 3.793, -0.272]));
}

fn test_sparse_csc_pins16() {
    let e48, g48 = scientific::csc_eye(1);
    assert(g48.kind == scientific::F_NONE);
    let d48, h48 = scientific::csc_to_dense(e48);
    assert(h48.kind == scientific::F_NONE);
    assert(d48.shape == [1, 1] && common_3::ms_same(d48, [1.0]));
    let e49, g49 = scientific::csc_eye(4);
    assert(g49.kind == scientific::F_NONE);
    let d49, h49 = scientific::csc_to_dense(e49);
    assert(h49.kind == scientific::F_NONE);
    assert(d49.shape == [4, 4] && common_3::ms_same(d49, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]));
    let e50, g50 = scientific::csc_eye(6);
    assert(g50.kind == scientific::F_NONE);
    let d50, h50 = scientific::csc_to_dense(e50);
    assert(h50.kind == scientific::F_NONE);
    assert(d50.shape == [6, 6] && common_3::ms_same(d50, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]));
    let f51, g51 = scientific::csc_diags([1.0, 2.0, 3.0, 4.0, 5.0], -2, 5, 5);
    assert(g51.kind == scientific::F_NONE);
    let d51, h51 = scientific::csc_to_dense(f51);
    assert(h51.kind == scientific::F_NONE);
    assert(d51.shape == [5, 5] && common_3::ms_same(d51, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]));
    let f52, g52 = scientific::csc_diags([1.0, 2.0, 3.0, 4.0, 5.0], -1, 5, 5);
    assert(g52.kind == scientific::F_NONE);
    let d52, h52 = scientific::csc_to_dense(f52);
    assert(h52.kind == scientific::F_NONE);
    assert(d52.shape == [5, 5] && common_3::ms_same(d52, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0]));
    let f53, g53 = scientific::csc_diags([1.0, 2.0, 3.0, 4.0, 5.0], 0, 5, 5);
    assert(g53.kind == scientific::F_NONE);
    let d53, h53 = scientific::csc_to_dense(f53);
    assert(h53.kind == scientific::F_NONE);
    assert(d53.shape == [5, 5] && common_3::ms_same(d53, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]));
    let f54, g54 = scientific::csc_diags([1.0, 2.0, 3.0, 4.0, 5.0], 1, 5, 5);
    assert(g54.kind == scientific::F_NONE);
    let d54, h54 = scientific::csc_to_dense(f54);
    assert(h54.kind == scientific::F_NONE);
    assert(d54.shape == [5, 5] && common_3::ms_same(d54, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    let f55, g55 = scientific::csc_diags([1.0, 2.0, 3.0, 4.0, 5.0], 2, 5, 5);
    assert(g55.kind == scientific::F_NONE);
    let d55, h55 = scientific::csc_to_dense(f55);
    assert(h55.kind == scientific::F_NONE);
    assert(d55.shape == [5, 5] && common_3::ms_same(d55, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
}

fn test_sparse_csc_pins17() {
    let g56, h56 = scientific::csc_get(scientific::csc_eye(3), 1, 1);
    assert(h56.kind == scientific::F_NONE && g56 == 1.0);
    let i57, j57 = scientific::csc_get(scientific::csc_eye(3), 0, 2);
    assert(j57.kind == scientific::F_NONE && i57 == 0.0);
    let l58, m58 = scientific::csc_from_triplets(2, 2, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]);
    assert(m58.kind == scientific::F_NONE);
    let n58, o58 = scientific::csc_to_dense(l58);
    assert(o58.kind == scientific::F_NONE);
    assert(n58.shape == [2, 2] && common_3::ms_same(n58, [3.0, 0.0, 0.0, 3.0]));
}

fn test_sparse_csc_pins() {
    sparse_csc_tests::test_sparse_csc_pins0();
    sparse_csc_tests::test_sparse_csc_pins1();
    sparse_csc_tests::test_sparse_csc_pins2();
    sparse_csc_tests::test_sparse_csc_pins3();
    sparse_csc_tests::test_sparse_csc_pins4();
    sparse_csc_tests::test_sparse_csc_pins5();
    sparse_csc_tests::test_sparse_csc_pins6();
    sparse_csc_tests::test_sparse_csc_pins7();
    sparse_csc_tests::test_sparse_csc_pins8();
    sparse_csc_tests::test_sparse_csc_pins9();
    sparse_csc_tests::test_sparse_csc_pins10();
    sparse_csc_tests::test_sparse_csc_pins11();
    sparse_csc_tests::test_sparse_csc_pins12();
    sparse_csc_2_tests::test_sparse_csc_pins13();
    sparse_csc_2_tests::test_sparse_csc_pins14();
    sparse_csc_2_tests::test_sparse_csc_pins15();
    sparse_csc_2_tests::test_sparse_csc_pins16();
    sparse_csc_2_tests::test_sparse_csc_pins17();
}

// test_sparse_csc_hostile checks the faults of the compressed columns.
fn test_sparse_csc_hostile() {
    let a = common_3::ni_nd([2, 2], [1.0, 0.0, 0.0, 2.0]);
    // A side below one.
    let c1, g1 = scientific::csc_eye(0);
    assert(g1.kind == scientific::F_RANGE);
    // An entry outside the matrix.
    let c2, g2 = scientific::csc_from_triplets(2, 2, [0.0, 5.0], [0.0, 1.0], [1.0, 2.0]);
    assert(g2.kind == scientific::F_RANGE);
    // The rows, the columns and the values differ in count.
    let c3, g3 = scientific::csc_from_triplets(2, 2, [0.0], [0.0, 1.0], [1.0]);
    assert(g3.kind == scientific::F_SHAPE);
    // A place outside the matrix.
    let c4, g4 = scientific::csc_get(scientific::csc_eye(2), 2, 0);
    assert(g4.kind == scientific::F_RANGE);
    // A vector of the wrong length.
    let c5, g5 = scientific::csc_matvec(scientific::csc_eye(2), [1.0, 2.0, 3.0]);
    assert(g5.kind == scientific::F_SHAPE);
    // A matrix that is not a matrix.
    let c6, g6 = scientific::csc_from_dense(common_3::ni_nd([3], [1.0, 2.0, 3.0]));
    assert(g6.kind == scientific::F_SHAPE);
    // Two matrices of another shape.
    let c7, g7 = scientific::csc_add(scientific::csc_eye(2), scientific::csc_eye(3));
    assert(g7.kind == scientific::F_SHAPE);
    // A diagonal that lies outside the matrix.
    let c8, g8 = scientific::csc_diags([1.0, 2.0], 5, 2, 2);
    assert(g8.kind == scientific::F_RANGE);
    // A diagonal with too few values.
    let c9, g9 = scientific::csc_diags([1.0], 0, 3, 3);
    assert(g9.kind == scientific::F_SHAPE);
    // The values of a matrix of compressed columns equal the values of the
    // compressed rows.
    let c10, g10 = scientific::csc_from_dense(a);
    assert(g10.kind == scientific::F_NONE && c10.nnz == 2 && c10.rows == 2 && c10.cols == 2);
    let c11, g11 = scientific::csc_transpose(c10);
    assert(g11.kind == scientific::F_NONE && c11.nnz == 2);
    let d11, h11 = scientific::csc_to_dense(c11);
    assert(h11.kind == scientific::F_NONE && common_3::ms_same(d11, [1.0, 0.0, 0.0, 2.0]));
}
