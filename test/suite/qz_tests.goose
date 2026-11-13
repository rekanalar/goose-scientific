// The tests of the group qz. Run them with SCIENTIFIC_SUITES=qz.
namespace qz_tests;

import scientific from "../..";
import os;
import common;
import common_2;

fn test_qz() {
    // The pinned pair: the diagonals and the factor Q of SciPy 1.18.1.
    let a = scientific::Arr { ndim: 2, shape: [4, 4], data: [2.0, 5.0, 8.0, 7.0, 5.0, 2.0, 2.0, 8.0, 7.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 8.0] };
    let b = scientific::Arr { ndim: 2, shape: [4, 4], data: [0.0, 6.0, 0.0, 0.0, 5.0, 0.0, 2.0, 1.0, 5.0, 2.0, 6.0, 6.0, 4.0, 7.0, 7.0, 7.0] };
    let aa, bb, q, z, f = scientific::qz(a, b);
    assert(f.kind == scientific::F_NONE);
    var da: f64[>..] = [];
    var db: f64[>..] = [];
    var i: i64 = 0;
    while i < 4 { da.push(aa.data[i * 5]); db.push(bb.data[i * 5]); i += 1; }
    common::ct_close(da, [-5.270293137251562, 7.883398770030825, -3.2624551158330526, 1.4312283083412392], 1.0e-10);
    common::ct_close(db, [0.33086466778775225, 5.107944262314686, 5.324800314897024, 7.3340533962845464], 1.0e-10);
    let qs = [0.22562988846509258, 0.4481573789938411, -0.8641145394068567, 0.03939771402348438, -0.8139678162481551, -0.05817839652006695, -0.21830884882022003, 0.5351755925089948, -0.04866640526786121, 0.869549266552417, 0.4475270197776592, 0.2030645725019108, -0.533084541645829, 0.19913390185972188, -0.0732589645121516, -0.8190236166967266];
    var c: i64 = 0;
    while c < 4 {
        var plus: f64 = 0.0;
        var minus: f64 = 0.0;
        var r: i64 = 0;
        while r < 4 {
            plus += common::mathshim_abs(q.data[r * 4 + c] - qs[r * 4 + c]);
            minus += common::mathshim_abs(q.data[r * 4 + c] + qs[r * 4 + c]);
            r += 1;
        }
        assert(plus < 1.0e-9 || minus < 1.0e-9);
        c += 1;
    }
    common::qt_real_case(a, b, 4);

    // Random pairs of side 2 to 10, and the same pairs with a singular b.
    var n: i64 = 2;
    while n <= 10 {
        var k: i64 = 0;
        while k < 6 {
            let pa, pb = common::qt_pair(n, n * 100 + k, k % 2);
            common::qt_real_case(pa, pb, n);
            k += 1;
        }
        n += 1;
    }

    // The singular a, the zeros, the rotation, the side 1 and the identity.
    var sa = copy(a.data);
    i = 0;
    while i < 4 { sa[i * 4 + 1] = 0.0; i += 1; }
    let sing = scientific::Arr { ndim: 2, shape: [4, 4], data: copy(sa) };
    common::qt_real_case(sing, b, 4);
    let zero = scientific::Arr { ndim: 2, shape: [3, 3], data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };
    let zaa, zbb, zal, zbe, zq, zz, zf = scientific::ordqz(zero, zero, "lhp");
    assert(zf.kind == scientific::F_NONE);
    i = 0;
    while i < 3 { assert(zal.data[2 * i] == 0.0 && zbe.data[i] == 0.0); i += 1; }
    let eye2 = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 0.0, 0.0, 1.0] };
    let rot = scientific::Arr { ndim: 2, shape: [2, 2], data: [0.0, 1.0, -1.0, 0.0] };
    common::qt_real_case(rot, eye2, 2);
    let raa, rbb, ral, rbe, rq, rz, rf = scientific::ordqz(rot, eye2, "lhp");
    assert(rf.kind == scientific::F_NONE);
    assert(common::mathshim_abs(ral.data[0] / rbe.data[0]) < 1.0e-12);
    assert(common::mathshim_abs(common::mathshim_abs(ral.data[1] / rbe.data[0]) - 1.0) < 1.0e-12);
    assert(ral.data[1] > 0.0 && ral.data[3] < 0.0);
    assert(rbb.data[0] > 0.0 && rbb.data[3] > 0.0 && rbb.data[1] == 0.0 && rbb.data[2] == 0.0);
    let s1a, s1b, s1q, s1z, s1f = scientific::qz(scientific::Arr { ndim: 2, shape: [1, 1], data: [3.0] }, scientific::Arr { ndim: 2, shape: [1, 1], data: [-2.0] });
    assert(s1f.kind == scientific::F_NONE);
    assert(s1a.data[0] == -3.0 && s1b.data[0] == 2.0);
    common::qt_real_case(eye2, eye2, 2);

    // A stack with a broadcast operand equals the single calls.
    var sd: f64[>..] = [];
    var s: i64 = 0;
    while s < 2 {
        let p1, p2 = common::qt_pair(3, 70 + s, 0);
        i = 0;
        while i < 9 { sd.push(p1.data[i]); i += 1; }
        s += 1;
    }
    let sb1, sb2 = common::qt_pair(3, 99, 0);
    let stack_a = scientific::Arr { ndim: 3, shape: [2, 3, 3], data: copy(sd) };
    let ta, tb, tq, tz, tf = scientific::qz(stack_a, sb2);
    assert(tf.kind == scientific::F_NONE);
    assert(tq.ndim == 3 && tq.shape[0] == 2 && tq.shape[1] == 3);
    s = 0;
    while s < 2 {
        let u1, u2 = common::qt_pair(3, 70 + s, 0);
        let ua, ub, uq, uz, uf = scientific::qz(u1, sb2);
        var d: i64 = 0;
        while d < 9 {
            assert(common::mathshim_abs(ta.data[s * 9 + d] - ua.data[d]) < 1.0e-12);
            assert(common::mathshim_abs(tq.data[s * 9 + d] - uq.data[d]) < 1.0e-12);
            d += 1;
        }
        s += 1;
    }
}

fn test_qz_complex() {
    var n: i64 = 2;
    while n <= 10 {
        var k: i64 = 0;
        while k < 6 {
            let pa, pb = common::qt_pair(n, n * 977 + k, k % 2);
            common_2::qt_complex_case(pa, pb, n);
            k += 1;
        }
        n += 1;
    }
    // The ordered form puts the chosen eigenvalues first.
    let a = scientific::Arr { ndim: 2, shape: [4, 4], data: [2.0, 5.0, 8.0, 7.0, 5.0, 2.0, 2.0, 8.0, 7.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 8.0] };
    let b = scientific::Arr { ndim: 2, shape: [4, 4], data: [0.0, 6.0, 0.0, 0.0, 5.0, 0.0, 2.0, 1.0, 5.0, 2.0, 6.0, 6.0, 4.0, 7.0, 7.0, 7.0] };
    let aa, bb, al, be, q, z, f = scientific::ordqz_complex(a, b, "lhp");
    assert(f.kind == scientific::F_NONE);
    var ar: f64[>..] = [];
    var i: i64 = 0;
    while i < 16 { ar.push(a.data[i]); ar.push(0.0); i += 1; }
    assert(common_2::qt_crecon(q.data, aa.data, z.data, ar, 4) < 1.0e-12);
    assert(be.data.len == 4 && al.data.len == 8);
    assert(al.data[0] / be.data[0] < 0.0 && al.data[2] / be.data[1] < 0.0);
    assert(al.data[4] / be.data[2] > 0.0 && al.data[6] / be.data[3] > 0.0);
    // A pair of complex eigenvalues keeps its place under a rule.
    let rot = scientific::Arr { ndim: 2, shape: [2, 2], data: [0.0, 1.0, -1.0, 0.0] };
    let eye2 = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 0.0, 0.0, 1.0] };
    let c1, c2, c3, c4, c5, c6, cf = scientific::ordqz_complex(rot, eye2, "iuc");
    assert(cf.kind == scientific::F_NONE);
    common_2::qt_complex_case(rot, eye2, 2);

    // A stack with a broadcast operand gives complex results of the batch
    // shape. Each slice equals the single call.
    var sd: f64[>..] = [];
    var s: i64 = 0;
    while s < 2 {
        let p1, p2 = common::qt_pair(3, 330 + s, 0);
        var d: i64 = 0;
        while d < 9 { sd.push(p1.data[d]); d += 1; }
        s += 1;
    }
    let sb1, sb2 = common::qt_pair(3, 399, 0);
    let stack_a = scientific::Arr { ndim: 3, shape: [2, 3, 3], data: copy(sd) };
    let ta, tb, tq, tz, tf = scientific::qz_complex(stack_a, sb2);
    assert(tf.kind == scientific::F_NONE);
    assert(tq.ndim == 3 && tq.shape[0] == 2 && tq.data.len == 36);
    s = 0;
    while s < 2 {
        let u1, u2 = common::qt_pair(3, 330 + s, 0);
        let ua, ub, uq, uz, uf = scientific::qz_complex(u1, sb2);
        var d2: i64 = 0;
        while d2 < 18 { assert(common::mathshim_abs(ta.data[s * 18 + d2] - ua.data[d2]) < 1.0e-12); d2 += 1; }
        s += 1;
    }
    let oa, ob, oal, obe, oq, oz, of = scientific::ordqz_complex(stack_a, sb2, "lhp");
    assert(of.kind == scientific::F_NONE);
    assert(oal.shape[0] == 2 && oal.shape[1] == 3 && obe.shape[1] == 3);
}

fn test_ordqz_by() {
    // The pinned real pair.
    let a = scientific::Arr { ndim: 2, shape: [4, 4], data: [2.0, 5.0, 8.0, 7.0, 5.0, 2.0, 2.0, 8.0, 7.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 8.0] };
    let b = scientific::Arr { ndim: 2, shape: [4, 4], data: [0.0, 6.0, 0.0, 0.0, 5.0, 0.0, 2.0, 1.0, 5.0, 2.0, 6.0, 6.0, 4.0, 7.0, 7.0, 7.0] };
    // A block that chooses the same group as the rule lhp.
    var count: i64 = 0;
    let aa, bb, al, be, q, z, f = scientific::ordqz_by(a, b) { al, be => count += 1; al.re / be < 0.0 };
    assert(f.kind == scientific::F_NONE);
    assert(count > 0 && count <= 4);
    assert(common::qt_recon(q.data, aa.data, z.data, a.data, 0, 4) < 1.0e-12);
    assert(common::qt_recon(q.data, bb.data, z.data, b.data, 0, 4) < 1.0e-12);
    assert(common::qt_orth(q.data, 0, 4) < 1.0e-12);
    assert(common::qt_orth(z.data, 0, 4) < 1.0e-12);
    assert(common::qt_shape(aa.data, bb.data, 0, 4) < 1.0e-12);
    let ra, rb, ral, rbe, rq, rz, rf = scientific::ordqz(a, b, "lhp");
    assert(rf.kind == scientific::F_NONE);
    common_2::qt_eigs_close(al, ral, be, rbe, 1.0e-12);
    // A block that chooses the same group as the rule iuc.
    var count2: i64 = 0;
    let ia, ib, ial, ibe, iq, iz, if2 = scientific::ordqz_by(a, b) { al, be => count2 += 1; mathshim::root(al.re * al.re + al.im * al.im) < be };
    assert(if2.kind == scientific::F_NONE);
    assert(count2 > 0 && count2 <= 4);
    assert(common::qt_recon(iq.data, ia.data, iz.data, a.data, 0, 4) < 1.0e-12);
    let sa, sb, sal, sbe, sq, sz, sf = scientific::ordqz(a, b, "iuc");
    assert(sf.kind == scientific::F_NONE);
    common_2::qt_eigs_close(ial, sal, ibe, sbe, 1.0e-12);

    // A pair with complex eigenvalues is one block. The block gets the
    // value with the positive imaginary part.
    let rot = scientific::Arr { ndim: 2, shape: [2, 2], data: [0.0, 1.0, -1.0, 0.0] };
    let eye2 = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 0.0, 0.0, 1.0] };
    var count3: i64 = 0;
    var imseen: f64 = 0.0;
    let ma, mb, mal, mbe, mq, mz, mf = scientific::ordqz_by(rot, eye2) { al, be => count3 += 1; imseen = al.im; al.im > 0.0 };
    assert(mf.kind == scientific::F_NONE);
    assert(count3 == 1);
    assert(imseen > 0.0);
    assert(mal.data.len == 4);
    // The two values of the pair stay next to each other.
    assert(mal.data[1] > 0.0 && mal.data[3] < 0.0);
    assert(common::mathshim_abs(mal.data[0] - mal.data[2]) < 1.0e-12);

    // A block that always gives true and a block that always gives
    // false leave the order of qz alone.
    let qa, qb, qal, qbe, qq, qz, qf = scientific::ordqz_by(a, b) { al, be => true };
    assert(qf.kind == scientific::F_NONE);
    let pa, pb, pal, pbe, pq, pz, pf = scientific::ordqz_by(a, b) { al, be => false };
    assert(pf.kind == scientific::F_NONE);
    let ua, ub, uq, uz, uf = scientific::qz(a, b);
    assert(uf.kind == scientific::F_NONE);
    var i: i64 = 0;
    while i < 4 {
        let ev = ua.data[i * 5] / ub.data[i * 5];
        assert(common::mathshim_abs(qal.data[2 * i] / qbe.data[i] - ev) < 1.0e-12);
        assert(common::mathshim_abs(pal.data[2 * i] / pbe.data[i] - ev) < 1.0e-12);
        i += 1;
    }

    // A block that alternates its answers ends the call. The result is a
    // valid decomposition and the count of calls stays at or below n.
    var count4: i64 = 0;
    let xa, xb, xal, xbe, xq, xz, xf = scientific::ordqz_by(a, b) { al, be => count4 += 1; (count4 % 2) == 1 };
    assert(xf.kind == scientific::F_NONE);
    assert(count4 <= 4);
    assert(common::qt_recon(xq.data, xa.data, xz.data, a.data, 0, 4) < 1.0e-12);
    assert(common::qt_recon(xq.data, xb.data, xz.data, b.data, 0, 4) < 1.0e-12);

    // The complex form of a real pair and the complex pair itself take
    // the same group as the rule lhp.
    var count5: i64 = 0;
    let ca, cb, cal, cbe, cq, cz, cf = scientific::ordqz_complex_by(a, b) { al, be => count5 += 1; al.re / be < 0.0 };
    assert(cf.kind == scientific::F_NONE);
    assert(count5 > 0 && count5 <= 4);
    assert(common_2::qt_crecon(cq.data, ca.data, cz.data, [2.0, 0.0, 5.0, 0.0, 8.0, 0.0, 7.0, 0.0, 5.0, 0.0, 2.0, 0.0, 2.0, 0.0, 8.0, 0.0, 7.0, 0.0, 5.0, 0.0, 6.0, 0.0, 6.0, 0.0, 5.0, 0.0, 4.0, 0.0, 4.0, 0.0, 8.0, 0.0], 4) < 1.0e-12);
    common_2::qt_eigs_close(cal, ral, cbe, rbe, 1.0e-12);
    let pa2, pb2 = common_2::qt_pinned_pair();
    let xa2 = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(pa2) };
    let xb2 = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(pb2) };
    var count6: i64 = 0;
    let ya, yb, yal, ybe, yq, yz, yf = scientific::ordqz_c_by(copy(xa2), copy(xb2)) { al, be => count6 += 1; al.re / be < 0.0 };
    assert(yf.kind == scientific::F_NONE);
    assert(count6 > 0 && count6 <= 4);
    let za, zb, zal, zbe, zq, zz, zf = scientific::ordqz_c(copy(xa2), copy(xb2), "lhp");
    assert(zf.kind == scientific::F_NONE);
    common_2::qt_eigs_close(yal, zal, ybe, zbe, 1.0e-12);

    // Random pairs with a block of sort. The chosen eigenvalues come
    // first, and the products hold.
    var t: i64 = 0;
    while t < 50 {
        let n = 2 + t % 9;
        let ra, rb = common::qt_pair(n, 800 + t, 0);
        let xa, xb, xal, xbe, xq, xz, xf = scientific::ordqz_by(ra, rb) { al, be => al.re / be < 0.0 };
        assert(xf.kind == scientific::F_NONE);
        assert(common::qt_recon(xq.data, xa.data, xz.data, ra.data, 0, n) < 1.0e-12);
        assert(common::qt_recon(xq.data, xb.data, xz.data, rb.data, 0, n) < 1.0e-12);
        var seen = false;
        var i: i64 = 0;
        while i < n {
            let neg = xal.data[2 * i] / xbe.data[i] < 0.0;
            if neg { assert(!seen); } else { seen = true; }
            i += 1;
        }
        t += 1;
    }
    t = 0;
    while t < 50 {
        let n = 2 + t % 9;
        let ra, rb = common_2::qt_cpair(n, 900 + t);
        let ca = scientific::CArr { ndim: 2, shape: [n, n], data: copy(ra) };
        let cb = scientific::CArr { ndim: 2, shape: [n, n], data: copy(rb) };
        let xa, xb, xal, xbe, xq, xz, xf = scientific::ordqz_c_by(copy(ca), copy(cb)) { al, be => al.re / be < 0.0 };
        assert(xf.kind == scientific::F_NONE);
        assert(common_2::qt_crecon(xq.data, xa.data, xz.data, ra, n) < 1.0e-12);
        assert(common_2::qt_crecon(xq.data, xb.data, xz.data, rb, n) < 1.0e-12);
        var seen = false;
        var i: i64 = 0;
        while i < n {
            let neg = xal.data[2 * i] / xbe.data[i] < 0.0;
            if neg { assert(!seen); } else { seen = true; }
            i += 1;
        }
        t += 1;
    }

    // A stack with a broadcast operand gives results of the batch shape.
    var sd: f64[>..] = [];
    var s: i64 = 0;
    while s < 2 {
        let p1, p2 = common::qt_pair(3, 70 + s, 0);
        var d: i64 = 0;
        while d < 9 { sd.push(p1.data[d]); d += 1; }
        s += 1;
    }
    let sb1, sb2 = common::qt_pair(3, 99, 0);
    let stack_a = scientific::Arr { ndim: 3, shape: [2, 3, 3], data: copy(sd) };
    let ta, tb, tal, tbe, tq, tz, tf = scientific::ordqz_by(stack_a, sb2) { al, be => al.re / be < 0.0 };
    assert(tf.kind == scientific::F_NONE);
    assert(tq.ndim == 3 && tq.shape[0] == 2 && tq.shape[1] == 3);
    assert(tal.shape[0] == 2 && tal.shape[1] == 3 && tbe.shape[1] == 3);
    s = 0;
    while s < 2 {
        let u1, u2 = common::qt_pair(3, 70 + s, 0);
        let ua, ub, ual, ube, uq, uz, uf = scientific::ordqz_by(u1, sb2) { al, be => al.re / be < 0.0 };
        assert(uf.kind == scientific::F_NONE);
        var d2: i64 = 0;
        while d2 < 6 {
            assert(common::mathshim_abs(tal.data[s * 6 + d2] - ual.data[d2]) < 1.0e-12);
            d2 += 1;
        }
        s += 1;
    }

    // The complex form of a stack with a broadcast operand.
    let cta, ctb, ctal, ctbe, ctq, ctz, ctf = scientific::ordqz_complex_by(stack_a, sb2) { al, be => al.re / be < 0.0 };
    assert(ctf.kind == scientific::F_NONE);
    assert(ctq.ndim == 3 && ctq.shape[0] == 2 && ctq.shape[1] == 3);
    assert(ctal.shape[0] == 2 && ctal.shape[1] == 3 && ctbe.shape[1] == 3);
    s = 0;
    while s < 2 {
        let u1, u2 = common::qt_pair(3, 70 + s, 0);
        let ua, ub, ual, ube, uq, uz, uf = scientific::ordqz_complex_by(u1, sb2) { al, be => al.re / be < 0.0 };
        assert(uf.kind == scientific::F_NONE);
        var d3: i64 = 0;
        while d3 < 6 {
            assert(common::mathshim_abs(ctal.data[s * 6 + d3] - ual.data[d3]) < 1.0e-12);
            d3 += 1;
        }
        s += 1;
    }

    // A stack of complex pairs with a broadcast operand.
    var csd: f64[>..] = [];
    s = 0;
    while s < 2 {
        let p1, p2 = common_2::qt_cpair(3, 330 + s);
        var d: i64 = 0;
        while d < 18 { csd.push(p1[d]); d += 1; }
        s += 1;
    }
    let cs1, cs2 = common_2::qt_cpair(3, 399);
    let cstack = scientific::CArr { ndim: 3, shape: [2, 3, 3], data: copy(csd) };
    let cbcast = scientific::CArr { ndim: 2, shape: [3, 3], data: copy(cs2) };
    let cta2, ctb2, ctal2, ctbe2, ctq2, ctz2, ctf2 = scientific::ordqz_c_by(copy(cstack), copy(cbcast)) { al, be => al.re / be < 0.0 };
    assert(ctf2.kind == scientific::F_NONE);
    assert(ctq2.ndim == 3 && ctq2.shape[0] == 2 && ctq2.shape[1] == 3);
    assert(ctal2.shape[0] == 2 && ctal2.shape[1] == 3 && ctbe2.shape[1] == 3);
    s = 0;
    while s < 2 {
        let u1, u2 = common_2::qt_cpair(3, 330 + s);
        let ua, ub, ual, ube, uq, uz, uf = scientific::ordqz_c_by(scientific::CArr { ndim: 2, shape: [3, 3], data: copy(u1) }, copy(cbcast)) { al, be => al.re / be < 0.0 };
        assert(uf.kind == scientific::F_NONE);
        var d4: i64 = 0;
        while d4 < 6 {
            assert(common::mathshim_abs(ctal2.data[s * 6 + d4] - ual.data[d4]) < 1.0e-12);
            d4 += 1;
        }
        s += 1;
    }
}

fn test_qz_hostile() {
    let a = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0] };
    let r1 = scientific::Arr { ndim: 3, shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0] };
    let x1, x2, x3, x4, f1 = scientific::qz(r1, a);
    assert(f1.kind == scientific::F_SHAPE);
    let r2 = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 2.0, 3.0] };
    let y1, y2, y3, y4, f2 = scientific::qz(a, r2);
    assert(f2.kind == scientific::F_SHAPE);
    var big: f64[>..] = [];
    var i: i64 = 0;
    while i < 513 * 513 { big.push(0.0); i += 1; }
    let bg = scientific::Arr { ndim: 2, shape: [513, 513], data: copy(big) };
    let z1, z2, z3, z4, f3 = scientific::qz(bg, bg);
    assert(f3.kind == scientific::F_LIMIT);
    let nan = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 0.0 / 0.0, 3.0, 4.0] };
    let w1, w2, w3, w4, f4 = scientific::qz(nan, a);
    assert(f4.kind == scientific::F_VALUE);
    let inf = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 1.0 / 0.0, 3.0, 4.0] };
    let v1, v2, v3, v4, f5 = scientific::qz(a, inf);
    assert(f5.kind == scientific::F_VALUE);
    let a3 = scientific::Arr { ndim: 2, shape: [3, 3], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
    let u1, u2, u3, u4, f6 = scientific::qz(a, a3);
    assert(f6.kind == scientific::F_SHAPE);
    let rect = scientific::Arr { ndim: 2, shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
    let t1, t2, t3, t4, f7 = scientific::qz(rect, rect);
    assert(f7.kind == scientific::F_SHAPE);
    var s2: f64[>..] = [];
    var s3: f64[>..] = [];
    i = 0;
    while i < 8 { s2.push(1.0); i += 1; }
    i = 0;
    while i < 12 { s3.push(1.0); i += 1; }
    let b2 = scientific::Arr { ndim: 3, shape: [2, 2, 2], data: copy(s2) };
    let b3 = scientific::Arr { ndim: 3, shape: [3, 2, 2], data: copy(s3) };
    let r1a, r1b, r1c, r1d, f8 = scientific::qz(b2, b3);
    assert(f8.kind == scientific::F_SHAPE);
    // Two valid batches whose broadcast holds too many matrices.
    var l1: f64[>..] = [];
    i = 0;
    while i < 6000 * 9 { l1.push(1.0); i += 1; }
    let la = scientific::Arr { ndim: 4, shape: [6000, 1, 3, 3], data: copy(l1) };
    let lb = scientific::Arr { ndim: 4, shape: [1, 6000, 3, 3], data: copy(l1) };
    let m1, m2, m3, m4, f9 = scientific::qz(la, lb);
    assert(f9.kind == scientific::F_LIMIT);
    // The iteration limit and the swap that fails.
    let pa, pb = common::qt_pair(5, 31, 0);
    var ca = copy(pa.data);
    var cb = copy(pb.data);
    var cq = scientific::qz_identity(5);
    var cz = scientific::qz_identity(5);
    scientific::qz_reduce(ca, cb, cq, cz, 5);
    assert(!scientific::qz_iterate(ca, cb, cq, cz, 5, 1));
    var sa: f64[>..] = [1.0, 2.0, 0.0, 1.0];
    var sb: f64[>..] = [1.0, 0.0, 0.0, 1.0];
    var sq = scientific::qz_identity(2);
    var sz = scientific::qz_identity(2);
    assert(!scientific::qz_swap(sa, sb, sq, sz, 2, 0, 1, 1));

    // Entries near the limit of the number format. The call gives a
    // fault, or all results are finite. A NaN never comes back.
    var hd: f64[>..] = [];
    i = 0;
    while i < 9 { hd.push(1.0e300 * (1.0 + (i as f64) / 7.0)); i += 1; }
    hd[1] = -1.0e300;
    hd[5] = 3.0e300;
    let h1 = scientific::Arr { ndim: 2, shape: [3, 3], data: copy(hd) };
    let h2 = scientific::Arr { ndim: 2, shape: [3, 3], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
    let ha, hb, hq, hz, hf = scientific::qz(h1, h2);
    if hf.kind == scientific::F_NONE {
        assert(common_2::qt_finite_all(ha.data) && common_2::qt_finite_all(hb.data) && common_2::qt_finite_all(hq.data) && common_2::qt_finite_all(hz.data));
    } else {
        assert(hf.kind == scientific::F_VALUE);
    }
    print("overflow case fault kind ", hf.kind, "\n");

    // A pair whose complex form overflows. Each entry point gives a
    // fault, or every emitted value is finite.
    let ea = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0e200, 1.0e200, -1.0e200, 1.0e200] };
    let eb = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0e200, 0.0, 0.0, 1.0e200] };
    let c1, c2, c3, c4, cf = scientific::qz_complex(ea, eb);
    assert(cf.kind == scientific::F_NONE && common_2::qt_finite_all(c1.data) && common_2::qt_finite_all(c2.data) && common_2::qt_finite_all(c3.data) && common_2::qt_finite_all(c4.data) || cf.kind == scientific::F_VALUE);
    let o1, o2, oal, obe, o5, o6, of = scientific::ordqz(ea, eb, "lhp");
    assert(of.kind == scientific::F_NONE && common_2::qt_finite_all(oal.data) && common_2::qt_finite_all(obe.data) && common_2::qt_finite_all(o1.data) && common_2::qt_finite_all(o2.data) || of.kind == scientific::F_VALUE);
    let k1, k2, kal, kbe, k5, k6, kf = scientific::ordqz_complex(ea, eb, "lhp");
    assert(kf.kind == scientific::F_NONE && common_2::qt_finite_all(kal.data) && common_2::qt_finite_all(kbe.data) && common_2::qt_finite_all(k1.data) && common_2::qt_finite_all(k2.data) || kf.kind == scientific::F_VALUE);
}

// test_qz_scale_helpers checks the power of two that scales a buffer.
fn test_qz_scale_helpers() {
    let cases = [0.0, 1.0, 1.5, 3.0e300, 4.0e-300, 5.0e-324, 2.0, 0.999];
    var i: i64 = 0;
    while i < cases.len {
        let d = [cases[i]];
        let p = scientific::qz_scale_of(d, 0, 1);
        if cases[i] == 0.0 { assert(p == 1.0); } else {
            let m = cases[i] / p;
            assert(m >= 1.0 && m < 2.0);
            // A power of two halves and doubles without a change. A
            // subnormal value has no room to halve.
            assert(p / 2.0 * 2.0 == p || p < 1.0e-300);
        }
        i += 1;
    }
    var buf: f64[>..] = [8.0, -3.0, 0.5];
    scientific::qz_scale_apply(buf, 0, 3, 4.0, false);
    assert(buf[0] == 2.0 && buf[1] == -0.75 && buf[2] == 0.125);
    scientific::qz_scale_apply(buf, 1, 2, 4.0, true);
    assert(buf[0] == 2.0 && buf[1] == -3.0 && buf[2] == 0.5);
    // The scale reads only the range that it is given.
    let part = [1.0e100, 3.0, 1.0];
    assert(scientific::qz_scale_of(part, 1, 2) == 2.0);
}

// test_qz_time decomposes a pair of side 512, the largest side that the
// package accepts. The work grows with the cube of the side, so the test
// shows that the call finishes.
fn test_qz_time() {
    let pa, pb = common::qt_pair(512, 7, 0);
    let aa, bb, q, z, f = scientific::qz(pa, pb);
    assert(f.kind == scientific::F_NONE);
    assert(common::qt_orth(q.data, 0, 512) < 1.0e-10);
    assert(common::qt_shape(aa.data, bb.data, 0, 512) < 1.0e-10);
}
