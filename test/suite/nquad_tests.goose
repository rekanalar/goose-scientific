// The tests of the group nquad. Run them with SCIENTIFIC_SUITES=nquad.
namespace nquad_tests;

import scientific from "../..";
import os;
import common;

// nquad and quad_vec are pinned against SciPy.
fn test_nquad() {
    let pi = 3.141592653589793;
    let inf = 1.0 / 0.0;
    let v1, e1, f1 = scientific::nquad([0.0, 1.0, 0.0, 2.0]) { p => p[0] * p[1] * p[1] };
    assert(f1.kind == scientific::F_NONE && common::close(v1, 1.3333333333333335, 1.0e-12) && e1 < 1.0e-8);
    // the first value of the point belongs to the first range
    let v1b, e1b, f1b = scientific::nquad([0.0, 1.0, 0.0, 2.0]) { p => p[0] };
    assert(f1b.kind == scientific::F_NONE && common::close(v1b, 1.0, 1.0e-12));
    let v1c, e1c, f1c = scientific::nquad([0.0, 1.0, 0.0, 2.0]) { p => p[1] };
    assert(f1c.kind == scientific::F_NONE && common::close(v1c, 2.0, 1.0e-12));
    let v2, e2, f2 = scientific::nquad([-inf, inf, -inf, inf, -inf, inf]) { p => common::mathshim_exp(-p[0] * p[0] - p[1] * p[1] - p[2] * p[2]) };
    assert(f2.kind == scientific::F_NONE && common::close(v2, 5.568327996830833, 1.0e-7));
    let v3, e3, f3 = scientific::nquad([0.0, pi, 0.0, pi / 2.0]) { p => mathshim::sine(p[0] + p[1]) };
    assert(f3.kind == scientific::F_NONE && common::close(v3, 2.0, 1.0e-12));
    let v4, e4, f4 = scientific::nquad([0.0, 1.0, 0.0, 1.0], 1.49e-8, 1.49e-8, 200) { p => 1.0 / mathshim::root(p[0] * p[1]) };
    assert(f4.kind == scientific::F_NONE && common::close(v4, 4.0, 1.0e-7));
    // one range gives the value of quad
    let v5, e5, f5 = scientific::nquad([0.0, pi]) { p => mathshim::sine(p[0]) };
    let q5, qe5, qf5 = scientific::quad(0.0, pi) { x => mathshim::sine(x) };
    assert(f5.kind == scientific::F_NONE && v5 == q5 && e5 == qe5);
    // a reversed range negates, and an empty range gives zero
    let v6, e6, f6 = scientific::nquad([2.0, 0.0]) { p => p[0] };
    assert(f6.kind == scientific::F_NONE && common::close(v6, -2.0, 1.0e-12));
    let v7, e7, f7 = scientific::nquad([1.0, 1.0, 0.0, 5.0]) { p => p[0] };
    assert(f7.kind == scientific::F_NONE && v7 == 0.0);
    // a ray and a finite range
    let v8, e8, f8 = scientific::nquad([0.0, inf, 0.0, 1.0]) { p => common::mathshim_exp(-p[0]) * p[1] };
    assert(f8.kind == scientific::F_NONE && common::close(v8, 0.5, 1.0e-7));
    // the faults
    let z1, ze1, zf1 = scientific::nquad(common::empty_row()) { p => p[0] };
    assert(zf1.kind == scientific::F_SHAPE);
    let z2, ze2, zf2 = scientific::nquad([0.0, 1.0, 2.0]) { p => p[0] };
    assert(zf2.kind == scientific::F_SHAPE);
    let z3, ze3, zf3 = scientific::nquad([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]) { p => p[0] };
    assert(zf3.kind == scientific::F_LIMIT);
    let z4, ze4, zf4 = scientific::nquad([0.0, 1.0 / 0.0 - 1.0 / 0.0]) { p => p[0] };
    assert(zf4.kind == scientific::F_VALUE);
    let z5, ze5, zf5 = scientific::nquad([0.0, 1.0], -1.0) { p => p[0] };
    assert(zf5.kind == scientific::F_VALUE);
    let z6, ze6, zf6 = scientific::nquad([0.0, 1.0], 1.49e-8, 1.49e-8, 0) { p => p[0] };
    assert(zf6.kind == scientific::F_RANGE);
    let z7, ze7, zf7 = scientific::nquad([0.0, 1.0], 1.0e-30, 1.0e-30, 2) { p => mathshim::sine(100.0 * p[0]) };
    assert(zf7.kind == scientific::F_LIMIT);
    // an inner fault stops the call
    let z8, ze8, zf8 = scientific::nquad([0.0, 1.0, 0.0, 1.0], 1.0e-30, 1.0e-30, 2) { p => mathshim::sine(100.0 * p[0]) * p[1] };
    assert(zf8.kind == scientific::F_LIMIT);

    let r1, re1, rf1 = scientific::quad_vec(0.0, pi) { x, o => { o.push(mathshim::sine(x)); o.push(mathshim::cosine(x)); o.push(x * x); } };
    assert(rf1.kind == scientific::F_NONE && r1.data.len == 3);
    assert(common::close(r1.data[0], 2.0, 1.0e-12) && common::close(r1.data[1], 0.0, 1.0e-12) && common::close(r1.data[2], 10.335425560099939, 1.0e-12) && re1 < 1.0e-8);
    let r2, re2, rf2 = scientific::quad_vec(0.0, pi, 1.0e-200, 1.0e-8, 10000, "max") { x, o => { o.push(mathshim::sine(x)); o.push(mathshim::cosine(x)); o.push(x * x); } };
    assert(rf2.kind == scientific::F_NONE && common::close(r2.data[2], 10.335425560099939, 1.0e-12));
    let r3, re3, rf3 = scientific::quad_vec(0.0, 1.0) { x, o => { o.push(1.0 / mathshim::root(x)); o.push(mathshim::ln(x)); } };
    assert(rf3.kind == scientific::F_NONE && common::close(r3.data[0], 2.0, 1.0e-7) && common::close(r3.data[1], -1.0, 1.0e-7));
    let r4, re4, rf4 = scientific::quad_vec(0.0, 10.0, 1.0e-10, 1.0e-10) { x, o => { o.push(mathshim::sine(50.0 * x)); o.push(mathshim::cosine(30.0 * x)); } };
    assert(rf4.kind == scientific::F_NONE && common::close(r4.data[0], 0.03767699, 1.0e-6) && common::close(r4.data[1], -0.03332519, 1.0e-6));
    let r5, re5, rf5 = scientific::quad_vec(0.0, inf) { x, o => { o.push(common::mathshim_exp(-x)); o.push(common::mathshim_exp(-x)); } };
    assert(rf5.kind == scientific::F_NONE && common::close(r5.data[0], 1.0, 1.0e-8) && common::close(r5.data[1], 1.0, 1.0e-8));
    let r5b, re5b, rf5b = scientific::quad_vec(-inf, inf) { x, o => { o.push(common::mathshim_exp(-x * x)); } };
    assert(rf5b.kind == scientific::F_NONE && common::close(r5b.data[0], 1.7724538509055159, 1.0e-7));
    // a reversed range negates
    let r6, re6, rf6 = scientific::quad_vec(pi, 0.0) { x, o => { o.push(mathshim::sine(x)); } };
    assert(rf6.kind == scientific::F_NONE && common::close(r6.data[0], -2.0, 1.0e-12));
    // an empty range gives zeros of the right length
    let r7, re7, rf7 = scientific::quad_vec(1.0, 1.0) { x, o => { o.push(x); o.push(x); } };
    assert(rf7.kind == scientific::F_NONE && r7.data.len == 2 && r7.data[0] == 0.0);
    // the faults
    let g1, ge1, gf1 = scientific::quad_vec(0.0, 1.0) { x, o => { o.push(x); if x > 0.5 { o.push(x); } } };
    assert(gf1.kind == scientific::F_SHAPE && g1.data.len == 0);
    let g2, ge2, gf2 = scientific::quad_vec(0.0, 1.0) { x, o => { x; } };
    assert(gf2.kind == scientific::F_SHAPE);
    let g3, ge3, gf3 = scientific::quad_vec(0.0, 1.0 / 0.0 - 1.0 / 0.0) { x, o => { o.push(x); } };
    assert(gf3.kind == scientific::F_VALUE);
    let g4, ge4, gf4 = scientific::quad_vec(0.0, 1.0, -1.0) { x, o => { o.push(x); } };
    assert(gf4.kind == scientific::F_VALUE);
    let g5, ge5, gf5 = scientific::quad_vec(0.0, 1.0, 1.0e-200, 1.0e-8, 10000, "1") { x, o => { o.push(x); } };
    assert(gf5.kind == scientific::F_VALUE);
    let g6, ge6, gf6 = scientific::quad_vec(0.0, 1.0, 1.0e-200, 1.0e-8, 0) { x, o => { o.push(x); } };
    assert(gf6.kind == scientific::F_RANGE);
    let g7, ge7, gf7 = scientific::quad_vec(0.0, 1.0, 1.0e-200, 1.0e-12, 2) { x, o => { o.push(mathshim::sine(50.0 * x)); } };
    assert(gf7.kind == scientific::F_LIMIT);
    return;
}
