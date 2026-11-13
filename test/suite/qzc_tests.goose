// The tests of the group qzc. Run them with SCIENTIFIC_SUITES=qzc.
namespace qzc_tests;

import scientific from "../..";
import os;
import common;
import common_2;

fn test_qz_c() {
    // The pinned complex pair: the eigenvalues and the diagonal of the
    // second form of the reference library.
    let pa, pb = common_2::qt_pinned_pair();
    let a = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(pa) };
    let b = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(pb) };
    let aa, bb, q, z, f = scientific::qz_c(copy(a), copy(b));
    assert(f.kind == scientific::F_NONE);
    assert(common_2::qt_crecon(q.data, aa.data, z.data, pa, 4) < 1.0e-12);
    assert(common_2::qt_crecon(q.data, bb.data, z.data, pb, 4) < 1.0e-12);
    assert(common_2::qt_corth(q.data, 4) < 1.0e-12);
    assert(common_2::qt_corth(z.data, 4) < 1.0e-12);
    assert(common_2::qt_cshape(aa.data, bb.data, 4) < 1.0e-12);
    var db: f64[>..] = [];
    var ev: f64[>..] = [];
    var i: i64 = 0;
    while i < 4 {
        let bt = bb.data[2 * (i * 4 + i)];
        assert(bt >= 0.0);
        db.push(bt);
        ev.push(aa.data[2 * (i * 4 + i)] / bt);
        ev.push(aa.data[2 * (i * 4 + i) + 1] / bt);
        i += 1;
    }
    common::ct_close(db, [4.30961609, 4.67717916, 7.99688329, 8.07965136], 1.0e-8);
    common_2::qt_csorteigs(ev);
    common_2::qt_ceigs_close(ev, [-0.9543240323, 0.7190975118, 0.0913951471, -0.0307145322,
                         0.8267799859, -0.2673308838, 1.5639973587, 0.1699689044], 1.0e-9);

    // Random complex pairs of side 2 to 10.
    var t: i64 = 0;
    while t < 50 {
        let n = 2 + t % 9;
        let ra, rb = common_2::qt_cpair(n, 500 + t);
        let ca = scientific::CArr { ndim: 2, shape: [n, n], data: copy(ra) };
        let cb = scientific::CArr { ndim: 2, shape: [n, n], data: copy(rb) };
        let xa, xb, xq, xz, xf = scientific::qz_c(copy(ca), copy(cb));
        assert(xf.kind == scientific::F_NONE);
        assert(common_2::qt_crecon(xq.data, xa.data, xz.data, ra, n) < 1.0e-12);
        assert(common_2::qt_crecon(xq.data, xb.data, xz.data, rb, n) < 1.0e-12);
        assert(common_2::qt_corth(xq.data, n) < 1.0e-12);
        assert(common_2::qt_corth(xz.data, n) < 1.0e-12);
        assert(common_2::qt_cshape(xa.data, xb.data, n) < 1.0e-12);
        var k: i64 = 0;
        while k < n {
            assert(xb.data[2 * (k * n + k)] >= 0.0);
            assert(xb.data[2 * (k * n + k) + 1] == 0.0);
            k += 1;
        }
        t += 1;
    }

    // A singular second matrix, a singular first matrix, the zeros and
    // a pair of side 1.
    var sb = copy(pb);
    i = 0;
    while i < 4 { sb[2 * (i * 4 + 3)] = 0.0; sb[2 * (i * 4 + 3) + 1] = 0.0; i += 1; }
    let sing = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(sb) };
    let saa, sbb, sq, sz, sf = scientific::qz_c(copy(a), copy(sing));
    assert(sf.kind == scientific::F_NONE);
    assert(common_2::qt_crecon(sq.data, saa.data, sz.data, pa, 4) < 1.0e-12);
    var ta = copy(pa);
    i = 0;
    while i < 4 { ta[2 * (i * 4 + 3)] = 0.0; ta[2 * (i * 4 + 3) + 1] = 0.0; i += 1; }
    let singa = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(ta) };
    let taa, tbb, tq, tz, tf = scientific::qz_c(copy(singa), copy(b));
    assert(tf.kind == scientific::F_NONE);
    assert(common_2::qt_crecon(tq.data, taa.data, tz.data, ta, 4) < 1.0e-12);
    var zd: f64[>..] = [];
    i = 0;
    while i < 18 { zd.push(0.0); i += 1; }
    let zero = scientific::CArr { ndim: 2, shape: [3, 3], data: copy(zd) };
    let zaa, zbb, zq, zz, zf = scientific::qz_c(copy(zero), copy(zero));
    assert(zf.kind == scientific::F_NONE);
    i = 0;
    while i < 18 { assert(zaa.data[i] == 0.0 && zbb.data[i] == 0.0); i += 1; }
    let one_a = scientific::CArr { ndim: 2, shape: [1, 1], data: [3.0, 1.0] };
    let one_b = scientific::CArr { ndim: 2, shape: [1, 1], data: [2.0, 0.0] };
    let oaa, obb, oq, oz, of = scientific::qz_c(copy(one_a), copy(one_b));
    assert(of.kind == scientific::F_NONE);
    assert(oaa.data[0] == 3.0 && oaa.data[1] == 1.0);
    assert(obb.data[0] == 2.0 && obb.data[1] == 0.0);

    // A complex pair with no imaginary part holds the eigenvalues of the
    // real pair with the same real parts.
    var rpa: f64[>..] = [];
    var rpb: f64[>..] = [];
    i = 0;
    while i < 16 {
        rpa.push(pa[2 * i]);
        rpb.push(pb[2 * i]);
        i += 1;
    }
    let real_a = scientific::Arr { ndim: 2, shape: [4, 4], data: copy(rpa) };
    let real_b = scientific::Arr { ndim: 2, shape: [4, 4], data: copy(rpb) };
    let raa, rbb, rq, rz, rf = scientific::qz_complex(real_a, real_b);
    assert(rf.kind == scientific::F_NONE);
    var rev: f64[>..] = [];
    i = 0;
    while i < 4 {
        rev.push(raa.data[2 * (i * 4 + i)] / rbb.data[2 * (i * 4 + i)]);
        rev.push(0.0);
        i += 1;
    }
    common_2::qt_csorteigs(rev);
    var cra: f64[>..] = [];
    var crb: f64[>..] = [];
    i = 0;
    while i < 16 {
        cra.push(rpa[i]);
        cra.push(0.0);
        crb.push(rpb[i]);
        crb.push(0.0);
        i += 1;
    }
    let caa, cbb, cq, cz, cf = scientific::qz_c(scientific::CArr { ndim: 2, shape: [4, 4], data: copy(cra) }, scientific::CArr { ndim: 2, shape: [4, 4], data: copy(crb) });
    assert(cf.kind == scientific::F_NONE);
    var cev: f64[>..] = [];
    i = 0;
    while i < 4 {
        let bt = cbb.data[2 * (i * 4 + i)];
        cev.push(caa.data[2 * (i * 4 + i)] / bt);
        cev.push(caa.data[2 * (i * 4 + i) + 1] / bt);
        i += 1;
    }
    common_2::qt_csorteigs(cev);
    common_2::qt_ceigs_close(cev, rev, 1.0e-9);

    // A stack of two pairs with a broadcast operand equals the single
    // calls.
    var sd: f64[>..] = [];
    var s: i64 = 0;
    while s < 2 {
        let p1, p2 = common_2::qt_cpair(3, 330 + s);
        var d: i64 = 0;
        while d < 18 { sd.push(p1[d]); d += 1; }
        s += 1;
    }
    let sb1, sb2 = common_2::qt_cpair(3, 399);
    let stack_a = scientific::CArr { ndim: 3, shape: [2, 3, 3], data: copy(sd) };
    let stack_b = scientific::CArr { ndim: 2, shape: [3, 3], data: copy(sb2) };
    let ta, tb, tq, tz, tf2 = scientific::qz_c(copy(stack_a), copy(stack_b));
    assert(tf2.kind == scientific::F_NONE);
    assert(tq.ndim == 3 && tq.shape[0] == 2 && tq.shape[1] == 3);
    s = 0;
    while s < 2 {
        let u1, u2 = common_2::qt_cpair(3, 330 + s);
        let ua, ub, uq, uz, uf = scientific::qz_c(scientific::CArr { ndim: 2, shape: [3, 3], data: copy(u1) }, copy(stack_b));
        assert(uf.kind == scientific::F_NONE);
        var d2: i64 = 0;
        while d2 < 18 {
            assert(common::mathshim_abs(ta.data[s * 18 + d2] - ua.data[d2]) < 1.0e-12);
            d2 += 1;
        }
        s += 1;
    }

    // A pair whose two matrices have different scales. The values alpha
    // and beta stay finite.
    var usa: f64[>..] = [];
    var usb: f64[>..] = [];
    i = 0;
    while i < 32 { usa.push(pa[i] * 1.0e300); usb.push(pb[i] * 1.0e-300); i += 1; }
    let ua2 = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(usa) };
    let ub2 = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(usb) };
    let va, vb, val, vbe, vq, vz, vf = scientific::ordqz_c(copy(ua2), copy(ub2), "lhp");
    assert(vf.kind == scientific::F_NONE);
    assert(common_2::qt_finite_all(val.data) && common_2::qt_finite_all(vbe.data));
    assert(vbe.data[0] > 0.0 && vbe.data[0] < 1.0e-299);

    // The same pair at four scales. The eigenvalues stay the same.
    var sc: f64[>..] = [1.0e-300, 1.0e-150, 1.0e150, 1.0e300];
    var ci: i64 = 0;
    while ci < 4 {
        let k = sc[ci];
        var sda: f64[>..] = [];
        var sdb: f64[>..] = [];
        i = 0;
        while i < 32 { sda.push(pa[i] * k); sdb.push(pb[i] * k); i += 1; }
        let ka = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(sda) };
        let kb = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(sdb) };
        let ma, mb, mq, mz, mf = scientific::qz_c(copy(ka), copy(kb));
        assert(mf.kind == scientific::F_NONE);
        assert(common_2::qt_crecon(mq.data, ma.data, mz.data, sda, 4) < 1.0e-12 * common_2::qt_maxabs(sda));
        assert(common_2::qt_crecon(mq.data, mb.data, mz.data, sdb, 4) < 1.0e-12 * common_2::qt_maxabs(sdb));
        var mev: f64[>..] = [];
        i = 0;
        while i < 4 {
            let bt = mb.data[2 * (i * 4 + i)];
            mev.push(ma.data[2 * (i * 4 + i)] / bt);
            mev.push(ma.data[2 * (i * 4 + i) + 1] / bt);
            i += 1;
        }
        common_2::qt_csorteigs(mev);
        common_2::qt_ceigs_close(mev, ev, 1.0e-10);
        ci += 1;
    }
}

fn test_qz_c_limit() {
    // A limit of one turn is not enough for a pair of side 5. The
    // status of that failure gives the fault of the limit.
    let ra, rb = common_2::qt_cpair(5, 91);
    let a: f64[>..] = [];
    let b: f64[>..] = [];
    var i: i64 = 0;
    while i < 50 { a.push(ra[i]); b.push(rb[i]); i += 1; }
    var ca: f64[>..] = [];
    var cb: f64[>..] = [];
    i = 0;
    while i < 50 { ca.push(ra[i]); cb.push(rb[i]); i += 1; }
    var cq = scientific::cq_identity(5);
    var cz = scientific::cq_identity(5);
    scientific::cq_reduce(ca, cb, cq, cz, 5);
    var qa: f64[>..] = [];
    var qb: f64[>..] = [];
    var qq: f64[>..] = [];
    var qz: f64[>..] = [];
    var qal: f64[>..] = [];
    var qbe: f64[>..] = [];
    let st = scientific::cq_pair(a, 0, b, 0, 5, scientific::QZ_NONE, 1, qa, qb, qq, qz, qal, qbe);
    assert(st == 1);
    assert(scientific::qz_status(st, "qz_c").kind == scientific::F_LIMIT);
}
