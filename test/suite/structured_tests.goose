// The tests of the group structured. Run them with SCIENTIFIC_SUITES=structured.
namespace structured_tests;

import scientific from "../..";
import os;
import common;

fn test_structured() {
    let a0, f0 = scientific::solve_toeplitz([4.0, 1.0, -0.5, 0.25], scientific::line_arr(scientific::copy_slice([1.0, 2.0, 3.0, 4.0])));
    common::mchk(a0, f0, 4, 1, [0.11904761904761904, 0.4880952380952381, 0.40476190476190477, 0.9523809523809523]);
    let a1, f1 = scientific::solve_toeplitz([4.0, 1.0, -0.5, 0.25], common::gmat(4, 2, [1.0, 5.0, 2.0, 1.0, 3.0, 0.0, 4.0, -1.0]), [4.0, 2.0, 1.0, 0.5]);
    common::mchk(a1, f1, 4, 2, [0.0, 1.2857142857142858, 0.14285714285714285, -0.17346938775510204, 0.2346938775510205, 0.43440233236151604, 0.9591836734693876, -0.4606413994169096]);
    let a2, f2 = scientific::solve_toeplitz([0.61, 0.6159, 0.0307, -0.4284, -0.8921, -0.2333, -0.1831, -0.9094], scientific::line_arr(scientific::copy_slice([0.0488, 0.9992, 0.6524, 0.2345, 0.4349, 0.9742, 0.8977, 0.8442])));
    common::mchk(a2, f2, 8, 1, [-10.100568137440284, -4.707556930767106, -32.58518398677538, 5.754538373873798, 5.275488993834518, -33.101076891688244, -4.02292472504455, -9.70125933656567]);
    let a3, f3 = scientific::solve_circulant([1.0, 2.0, 3.0, 4.0], scientific::line_arr(scientific::copy_slice([1.0, 2.0, 3.0, 4.0])));
    common::mchk(a3, f3, 4, 1, [1.0, 0.0, 0.0, 0.0]);
    let a4, f4 = scientific::solve_circulant([1.0, 2.0, 3.0, 4.0], scientific::line_arr(scientific::copy_slice([1.0, 2.0, 3.0, 4.0])), 1.0e-30);
    common::mchk(a4, f4, 4, 1, [1.0, 0.0, 0.0, 0.0]);
    let a5, f5 = scientific::matmul_toeplitz([4.0, 1.0, -0.5, 0.25], scientific::line_arr(scientific::copy_slice([1.0, 2.0, 3.0, 4.0])));
    common::mchk(a5, f5, 4, 1, [5.5, 10.0, 17.499999999999996, 18.25]);
    let a6, f6 = scientific::matmul_toeplitz([4.0, 1.0, -0.5, 0.25], common::gmat(4, 3, [-0.4304, -0.0279, 0.7068, -1.7568, 0.2224, -0.9142, 1.5186, -1.7431, 0.7167, 1.4804, -1.0907, 1.5818]), [4.0, 2.0, 1.0, 0.5]);
    common::mchk(a6, f6, 4, 3, [-2.9764, -1.9552500000000004, 2.5064, -2.9399999999999995, -3.7152000000000007, 0.06519999999999966, 7.493599999999999, -8.91745, 4.7628, 8.211, -6.224074999999999, 7.677699999999999]);
    let a7, f7 = scientific::diagsvd([3.0, 2.0, 1.0], 3, 3);
    common::mchk(a7, f7, 3, 3, [3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
    let a8, f8 = scientific::diagsvd([5.0, 3.0], 3, 2);
    common::mchk(a8, f8, 3, 2, [5.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
    let a9, f9 = scientific::diagsvd([5.0, 3.0], 2, 3);
    common::mchk(a9, f9, 2, 3, [5.0, 0.0, 0.0, 0.0, 3.0, 0.0]);
    let pr, fsc, fpr = scientific::orthogonal_procrustes(common::gmat(5, 3, [0.7444, -0.963, 0.415, -0.9976, 0.0067, -0.1267, -0.5935, -0.3501, 0.6124, -0.3671, -0.7019, 0.397, -0.1029, 0.5979, -0.529]), common::gmat(5, 3, [-0.3604, 0.5998, 0.0141, 0.0128, -0.5276, -0.9709, 0.8664, -0.8283, 0.6899, -0.2642, 0.902, -0.2011, 0.8729, 0.1123, -0.5197]));
    common::mchk(pr, fpr, 3, 3, [0.09060259605101736, 0.9365930694621384, 0.33850345910242385, 0.9934825002902653, -0.10860858485251665, 0.03459330734892455, -0.06916423357357221, -0.333163019454176, 0.9403290441448444]);
    assert(common::mathshim_abs(fsc - 2.7187096293862876) < 1.0e-10);
    let th1, fth1 = scientific::subspace_angles(common::gmat(10, 4, [0.4828, 0.3488, 0.3684, -0.0724, -0.5562, 0.2819, -0.7858, 0.3844, 0.2708, -0.247, 0.597, -0.6119, -0.2191, 0.5959, -0.239, 0.4265, 0.225, 0.882, 0.9834, 0.4474, 0.6177, -0.6943, 0.4258, 0.6952, -0.1975, 0.1065, -0.041, 0.917, -0.3654, -0.1958, -0.9982, -0.1596, 0.2629, 0.8699, 0.8474, -0.3453, 0.9777, -0.6247, 0.6465, -0.6855]), common::gmat(10, 6, [-0.1898, -0.8531, 0.7161, 0.6576, -0.7204, 0.0542, -0.4837, -0.0164, 0.1064, -0.7875, 0.7288, -0.4426, -0.1058, -0.8855, -0.9945, -0.6097, -0.3166, 0.8562, 0.7795, -0.039, -0.0905, 0.334, 0.7175, -0.3254, 0.5873, -0.202, 0.1881, 0.4749, 0.0027, 0.3816, 0.3946, -0.9886, -0.9219, -0.7019, -0.5727, -0.5936, -0.9021, -0.5656, 0.2019, 0.7722, -0.301, -0.2672, -0.1665, 0.3607, 0.5705, 0.8819, -0.2486, 0.4135, -0.3167, 0.648, -0.5387, 0.742, 0.0102, 0.4507, 0.07, -0.3678, -0.0114, -0.8991, -0.914, 0.0599]));
    common::vchk(th1, fth1, [0.9996963616215457, 0.8737513433321569, 0.3643925086009858, 0.04774408693069496]);
    let th2, fth2 = scientific::subspace_angles(common::gmat(3, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), common::gmat(3, 3, [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
    common::vchk(th2, fth2, [0.0, 0.0, 0.0]);
    let p5, rk5, fp5 = scientific::pinvh(common::gmat(5, 5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 3.0, 6.0, 10.0, 15.0, 1.0, 4.0, 10.0, 20.0, 35.0, 1.0, 5.0, 15.0, 35.0, 70.0]));
    assert(rk5 == 5);
    let p3, rk3, fp3 = scientific::pinvh(common::gmat(3, 3, [4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-14]));
    assert(rk3 == 3);
    // The faults.
    let x1, y1 = scientific::solve_toeplitz([1.0, 2.0, 1.0], scientific::line_arr(scientific::copy_slice([1.0, 2.0, 3.0])));
    assert(y1.kind == scientific::F_VALUE);
    let x2, y2 = scientific::solve_toeplitz([1.0, 2.0], scientific::line_arr(scientific::copy_slice([1.0])));
    assert(y2.kind == scientific::F_SHAPE);
    let x3, y3 = scientific::solve_circulant([1.0, 1.0, 1.0], scientific::line_arr(scientific::copy_slice([1.0, 2.0, 3.0])));
    assert(y3.kind == scientific::F_VALUE);
    let x4, y4 = scientific::matmul_toeplitz([1.0, 2.0], scientific::line_arr(scientific::copy_slice([1.0, 2.0, 3.0])));
    assert(y4.kind == scientific::F_SHAPE);
    let x5, y5 = scientific::diagsvd([1.0, 2.0, 3.0], 2, 3);
    assert(y5.kind == scientific::F_SHAPE);
    let e1, sc1, g1 = scientific::orthogonal_procrustes(common::gmat(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), common::gmat(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
    assert(g1.kind == scientific::F_SHAPE);
    let e2, g2 = scientific::subspace_angles(common::gmat(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), common::gmat(4, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
    assert(g2.kind == scientific::F_SHAPE);
    let e3, rk3b, g3 = scientific::pinvh(common::gmat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert(g3.kind == scientific::F_SHAPE);
    let e4, rk4b, g4 = scientific::pinvh(common::gmat(2, 2, [1.0, 2.0, 3.0, 4.0]), 0.0, -2.0);
    assert(g4.kind == scientific::F_VALUE);
    let e5, rk5b, g5 = scientific::pinvh(common::gmat(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(g5.kind == scientific::F_VALUE);
}
