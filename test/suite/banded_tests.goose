// The tests of the group banded. Run them with SCIENTIFIC_SUITES=banded.
namespace banded_tests;

import scientific from "../..";
import os;
import common;

fn test_banded() {
    let d = [2.0, 3.0, 4.0, 5.0, 6.0];
    let e = [1.0, -0.5, 0.25, 2.0];
    let w1, fw1 = scientific::eigvalsh_tridiagonal(d, e);
    common::vchk(w1, fw1, [1.3547922169552884, 3.2727539361611355, 3.4855631998954757, 4.318578517410089, 7.568312129578011]);
    let w2, fw2 = scientific::eigvalsh_tridiagonal(d, e, "i", [1.0, 3.0]);
    common::vchk(w2, fw2, [3.272753936161135, 3.485563199895476, 4.318578517410088]);
    let w3, fw3 = scientific::eigvalsh_tridiagonal(d, e, "v", [2.0, 5.0]);
    common::vchk(w3, fw3, [3.272753936161136, 3.485563199895476, 4.318578517410088]);
    let w4, v4, fw4 = scientific::eigh_tridiagonal(d, e);
    common::vchk(w4, fw4, [1.3547922169552884, 3.2727539361611355, 3.4855631998954766, 4.318578517410088, 7.568312129578013]);
    common::tri_check(d, e, w4, v4);
    let w5, v5, fw5 = scientific::eigh_tridiagonal(d, e, true);
    assert(fw5.kind == scientific::F_NONE && w5.data.len == 5 && v5.data.len == 0);
    // A pair of values that lie close together.
    let d2 = [1.0, 1.0000000001, 3.0, 7.0];
    let e2 = [1.0e-8, 0.5, 1.0];
    let w6, v6, fw6 = scientific::eigh_tridiagonal(d2, e2);
    common::vchk(w6, fw6, [0.8727140436644497, 1.0000000000000007, 2.889084428019311, 7.238201528416239]);
    common::tri_check(d2, e2, w6, v6);
    // A selected range with vectors.
    let w7, v7, fw7 = scientific::eigh_tridiagonal(d, e, false, "v", [2.0, 5.0]);
    common::vchk(w7, fw7, [3.272753936161136, 3.485563199895476, 4.318578517410088]);
    common::tri_check(d, e, w7, v7);
    // The banded Cholesky and the solvers.
    let ab1 = scientific::mat2(scientific::copy_slice([0.0, 1.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0]), 2, 4);
    let c1, fc1 = scientific::cholesky_banded(ab1);
    common::mchk(c1, fc1, 2, 4, [0.0, 0.5, 0.917662935482247, 0.4403152859263555, 2.0, 2.179449471770337, 2.2710998958306754, 2.608854623964239]);
    let ab2 = scientific::mat2(scientific::copy_slice([4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 1.0, 0.0]), 2, 4);
    let c2, fc2 = scientific::cholesky_banded(ab2, true);
    common::mchk(c2, fc2, 2, 4, [2.0, 2.179449471770337, 2.2710998958306754, 2.608854623964239, 0.5, 0.917662935482247, 0.4403152859263555, 0.0]);
    let ab3 = scientific::mat2(scientific::copy_slice([0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0]), 3, 4);
    let c3, fc3 = scientific::cholesky_banded(ab3);
    common::mchk(c3, fc3, 3, 4, [0.0, 0.0, 1.0, 0.4588314677411235, 0.0, 0.5, 1.1470786693528088, 0.7677718959499146, 2.0, 2.179449471770337, 1.9194297398747862, 2.4899799195977463]);
    let b1 = scientific::line_arr(scientific::copy_slice([1.0, 2.0, 3.0, 4.0]));
    let s1, fs1 = scientific::solveh_banded(ab1, b1);
    common::vchk(s1, fs1, [0.1934032983508246, 0.22638680659670163, 0.3373313343328336, 0.5232383808095952]);
    let s2, fs2 = scientific::cho_solve_banded(c1, b1);
    common::vchk(s2, fs2, [0.1934032983508246, 0.22638680659670157, 0.3373313343328337, 0.5232383808095953]);
    let s3, fs3 = scientific::cho_solve_banded(c2, b1, true);
    common::vchk(s3, fs3, [0.1934032983508246, 0.22638680659670157, 0.3373313343328337, 0.5232383808095953]);
    let b2 = scientific::line_arr(scientific::copy_slice([1.0, 0.0, 2.0, -1.0]));
    let s4, fs4 = scientific::solveh_banded(ab3, b2);
    common::vchk(s4, fs4, [0.04838709677419353, -0.2857142857142857, 0.5460829493087558, -0.2580645161290323]);
    // The RQ factorization.
    let m1 = common::gmat(4, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 1.0, 0.0, 2.0]);
    let r1, q1, fr1 = scientific::rq(m1);
    common::mchk(r1, fr1, 4, 3, [-0.8728715609439693, -1.854210138602213, -3.1304951684997055, -0.21821789023599214, -5.074680379332374, -7.155417527999328, 0.0, -8.19756061276768, -12.074767078498866, 0.0, 0.0, -2.23606797749979]);
    common::mchk(q1, fr1, 3, 3, [0.8728715609439696, -0.2182178902359923, -0.4364357804719847, -0.19518001458970657, -0.9759000729485332, 0.09759000729485327, -0.4472135954999579, -0.0, -0.8944271909999157]);
    let r2, q2, fr2 = scientific::rq(m1, "economic");
    common::mchk(r2, fr2, 4, 3, [-0.8728715609439693, -1.854210138602213, -3.1304951684997055, -0.21821789023599214, -5.074680379332374, -7.155417527999328, 0.0, -8.19756061276768, -12.074767078498866, 0.0, 0.0, -2.23606797749979]);
    let r3, q3, fr3 = scientific::rq(m1, "r");
    common::mchk(r3, fr3, 4, 3, [-0.8728715609439693, -1.854210138602213, -3.1304951684997055, -0.21821789023599214, -5.074680379332374, -7.155417527999328, 0.0, -8.19756061276768, -12.074767078498866, 0.0, 0.0, -2.23606797749979]);
    assert(fr3.kind == scientific::F_NONE && q3.data.len == 0);
    let m2 = common::gmat(3, 4, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0]);
    let r4, q4, fr4 = scientific::rq(m2);
    common::mchk(r4, fr4, 3, 4, [0.0, 0.7794837629539552, -1.3416588148232977, -5.2528427244586435, 0.0, 0.0, -0.5791860457433904, -13.178184378905016, 0.0, 0.0, 0.0, -21.702534414210703]);
    common::mchk(q4, fr4, 4, 4, [0.4082482904638634, -0.8164965809277263, 0.40824829046386224, 6.213395723495126e-16, -0.1299139604923246, -0.3247849012308145, -0.5196558419693064, 0.7794837629539569, 0.8027958482139392, 0.12463497186883195, -0.5535259044762768, -0.1832867233365177, -0.41469810982568234, -0.4607756775840915, -0.5068532453425006, -0.599008380859319]);
    let r5, q5, fr5 = scientific::rq(m2, "economic");
    common::mchk(r5, fr5, 3, 3, [0.7794837629539552, -1.3416588148232977, -5.2528427244586435, 0.0, -0.5791860457433904, -13.178184378905016, 0.0, 0.0, -21.702534414210703]);
    common::mchk(q5, fr5, 3, 4, [-0.1299139604923246, -0.3247849012308145, -0.5196558419693064, 0.7794837629539569, 0.8027958482139392, 0.12463497186883195, -0.5535259044762768, -0.1832867233365177, -0.41469810982568234, -0.4607756775840915, -0.5068532453425006, -0.599008380859319]);
    // The alias of the Lyapunov equation.
    let ly1, fly1 = scientific::solve_lyapunov(common::gmat(2, 2, [1.0, 2.0, 3.0, 4.0]), common::gmat(2, 2, [1.0, 0.0, 0.0, 1.0]));
    let ly2, fly2 = scientific::solve_continuous_lyapunov(common::gmat(2, 2, [1.0, 2.0, 3.0, 4.0]), common::gmat(2, 2, [1.0, 0.0, 0.0, 1.0]));
    assert(fly1.kind == scientific::F_NONE && fly2.kind == scientific::F_NONE);
    assert(ly1.data.len == ly2.data.len);
    var q9: i64 = 0;
    while q9 < ly1.data.len {
        assert(common::mathshim_abs(ly1.data[q9] - ly2.data[q9]) < 1.0e-12);
        q9 += 1;
    }
    // The faults.
    let f1, ff1 = scientific::eigvalsh_tridiagonal([2.0, 3.0], [1.0, -0.5]);
    assert(ff1.kind == scientific::F_SHAPE);
    let f2, ff2 = scientific::eigvalsh_tridiagonal(d, e, "x");
    assert(ff2.kind == scientific::F_VALUE);
    let f3, ff3 = scientific::eigvalsh_tridiagonal(d, e, "i", [-1.0, 3.0]);
    assert(ff3.kind == scientific::F_VALUE);
    let f3b, ff3b = scientific::eigvalsh_tridiagonal(d, e, "i", [3.0, 1.0]);
    assert(ff3b.kind == scientific::F_VALUE);
    let f3c, ff3c = scientific::eigvalsh_tridiagonal(d, e, "i", [0.5, 2.0]);
    assert(ff3c.kind == scientific::F_VALUE);
    let f4, ff4 = scientific::eigvalsh_tridiagonal(d, e, "i", [2.0, 9.0]);
    assert(ff4.kind == scientific::F_VALUE);
    let f5, ff5 = scientific::eigvalsh_tridiagonal(d, e, "v", [5.0, 2.0]);
    assert(ff5.kind == scientific::F_VALUE);
    let f6, ff6 = scientific::cholesky_banded(scientific::mat2(scientific::copy_slice([0.0, 1.0, -1.0, 1.0]), 2, 2));
    assert(ff6.kind == scientific::F_VALUE);
    let f7, ff7 = scientific::solveh_banded(ab1, scientific::line_arr(scientific::copy_slice([1.0, 2.0])));
    assert(ff7.kind == scientific::F_SHAPE);
    let f8, f8q, ff8 = scientific::rq(common::gmat(2, 2, [1.0, 0.0 / 0.0, 0.0, 1.0]));
    assert(ff8.kind == scientific::F_VALUE);
    let f9, f9q, ff9 = scientific::rq(common::gmat(2, 2, [1.0, 0.0, 0.0, 1.0]), "wide");
    assert(ff9.kind == scientific::F_VALUE);
}
