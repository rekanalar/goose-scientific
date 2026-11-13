// The tests of the group ordqz_c. Run them with SCIENTIFIC_SUITES=ordqz_c.
namespace ordqz_c_tests;

import scientific from "../..";
import os;
import common;
import common_2;

fn test_ordqz_c() {
    // The first group of each rule on the pinned pair.
    let pa, pb = common_2::qt_pinned_pair();
    let a = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(pa) };
    let b = scientific::CArr { ndim: 2, shape: [4, 4], data: copy(pb) };
    var r: i64 = 0;
    while r < 4 {
        let rule = common_2::qt_rule_name(r);
        let aa, bb, al, be, q, z, f = scientific::ordqz_c(copy(a), copy(b), rule);
        assert(f.kind == scientific::F_NONE);
        assert(common_2::qt_crecon(q.data, aa.data, z.data, pa, 4) < 1.0e-12);
        assert(common_2::qt_crecon(q.data, bb.data, z.data, pb, 4) < 1.0e-12);
        assert(common_2::qt_corth(q.data, 4) < 1.0e-12);
        assert(common_2::qt_corth(z.data, 4) < 1.0e-12);
        assert(common_2::qt_cshape(aa.data, bb.data, 4) < 1.0e-12);
        assert(al.data.len == 8 && be.data.len == 4);
        let k = common_2::qt_rule_count(r);
        var g1: f64[>..] = [];
        var g2: f64[>..] = [];
        var i: i64 = 0;
        while i < 4 {
            let e = al.data[2 * i] / be.data[i];
            let g = al.data[2 * i + 1] / be.data[i];
            if i < k { g1.push(e); g1.push(g); } else { g2.push(e); g2.push(g); }
            i += 1;
        }
        common_2::qt_csorteigs(g1);
        common_2::qt_csorteigs(g2);
        var w1: f64[>..] = [];
        var w2: f64[>..] = [];
        if rule == "lhp" {
            w1.push(-0.9543240323); w1.push(0.7190975118);
            w2.push(0.0913951471); w2.push(-0.0307145322);
            w2.push(0.8267799859); w2.push(-0.2673308838);
            w2.push(1.5639973587); w2.push(0.1699689044);
        }
        if rule == "rhp" {
            w1.push(0.0913951471); w1.push(-0.0307145322);
            w1.push(0.8267799859); w1.push(-0.2673308838);
            w1.push(1.5639973587); w1.push(0.1699689044);
            w2.push(-0.9543240323); w2.push(0.7190975118);
        }
        if rule == "iuc" {
            w1.push(0.0913951471); w1.push(-0.0307145322);
            w1.push(0.8267799859); w1.push(-0.2673308838);
            w2.push(-0.9543240323); w2.push(0.7190975118);
            w2.push(1.5639973587); w2.push(0.1699689044);
        }
        if rule == "ouc" {
            w1.push(-0.9543240323); w1.push(0.7190975118);
            w1.push(1.5639973587); w1.push(0.1699689044);
            w2.push(0.0913951471); w2.push(-0.0307145322);
            w2.push(0.8267799859); w2.push(-0.2673308838);
        }
        common_2::qt_ceigs_close(g1, w1, 1.0e-9);
        common_2::qt_ceigs_close(g2, w2, 1.0e-9);
        r += 1;
    }

    // Random pairs under each rule. The chosen eigenvalues come first.
    r = 0;
    while r < 4 {
        let rule = common_2::qt_rule_name(r);
        var t: i64 = 0;
        while t < 50 {
            let n = 2 + t % 9;
            let ra, rb = common_2::qt_cpair(n, 700 + t);
            let ca = scientific::CArr { ndim: 2, shape: [n, n], data: copy(ra) };
            let cb = scientific::CArr { ndim: 2, shape: [n, n], data: copy(rb) };
            let xa, xb, al, be, xq, xz, xf = scientific::ordqz_c(copy(ca), copy(cb), rule);
            assert(xf.kind == scientific::F_NONE);
            assert(common_2::qt_crecon(xq.data, xa.data, xz.data, ra, n) < 1.0e-12);
            assert(common_2::qt_crecon(xq.data, xb.data, xz.data, rb, n) < 1.0e-12);
            assert(common_2::qt_corth(xq.data, n) < 1.0e-12);
            assert(common_2::qt_corth(xz.data, n) < 1.0e-12);
            assert(common_2::qt_cshape(xa.data, xb.data, n) < 1.0e-12);
            assert(al.data.len == 2 * n && be.data.len == n);
            var seen = false;
            var i: i64 = 0;
            while i < n {
                if common::qt_chosen(rule, al.data[2 * i], al.data[2 * i + 1], be.data[i]) {
                    assert(!seen);
                } else {
                    seen = true;
                }
                i += 1;
            }
            t += 1;
        }
        r += 1;
    }

    // An infinite eigenvalue comes first under the outer rule.
    let inf_a = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0] };
    let inf_b = scientific::CArr { ndim: 2, shape: [2, 2], data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let iaa, ibb, ial, ibe, iq, iz, iff = scientific::ordqz_c(copy(inf_a), copy(inf_b), "ouc");
    assert(iff.kind == scientific::F_NONE);
    assert(ibe.data[0] == 0.0 && ial.data[0] != 0.0);

    // A pair of zeros holds no eigenvalue under any rule.
    var zd: f64[>..] = [];
    var i: i64 = 0;
    while i < 8 { zd.push(0.0); i += 1; }
    let zero = scientific::CArr { ndim: 2, shape: [2, 2], data: copy(zd) };
    r = 0;
    while r < 4 {
        let rule = common_2::qt_rule_name(r);
        let zaa, zbb, zal, zbe, zq, zz, zf = scientific::ordqz_c(copy(zero), copy(zero), rule);
        assert(zf.kind == scientific::F_NONE);
        i = 0;
        while i < 2 {
            assert(zal.data[2 * i] == 0.0 && zbe.data[i] == 0.0);
            assert(!common::qt_chosen(rule, zal.data[2 * i], zal.data[2 * i + 1], zbe.data[i]));
            i += 1;
        }
        r += 1;
    }

    // A name that is not a rule gives a fault.
    let ba, bb, bal, bbe, bq, bz, bf = scientific::ordqz_c(copy(a), copy(b), "abc");
    assert(bf.kind == scientific::F_VALUE);
}
