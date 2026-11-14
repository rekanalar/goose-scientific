// The tests of the forms of a filter. Run them with
// SCIENTIFIC_SUITES=filter_forms.
//
// The five calls zpk2tf, tf2zpk, tf2ss, zpk2ss and ss2tf match the pins of
// SciPy 1.18.1. Every limit of the calls has a fault test.
namespace filter_forms_tests;

import scientific from "../..";
import common;
import filter_forms_pins;

// roots_match tells whether two lists of roots hold the same roots. The
// order of the roots is free, so each pinned root needs an unused partner.
fn roots_match(gr: scientific::Arr&, gi: scientific::Arr&, wr: f64[:], wi: f64[:], tol: f64) -> bool {
    let n = wr.len;
    if gr.data.len != n || gi.data.len != n { return false; }
    var used: i64[>..] = [];
    var i: i64 = 0;
    while i < n { used.push(0); i += 1; }
    i = 0;
    while i < n {
        var found = false;
        var j: i64 = 0;
        while j < n && !found {
            if used[j] == 0
                && common::mathshim_abs(gr.data[j] - wr[i]) <= tol * (1.0 + common::mathshim_abs(wr[i]))
                && common::mathshim_abs(gi.data[j] - wi[i]) <= tol * (1.0 + common::mathshim_abs(wi[i])) {
                used[j] = 1;
                found = true;
            }
            j += 1;
        }
        if !found { return false; }
        i += 1;
    }
    return true;
}

// zeros gives a list of n zeros.
fn zeros(n: i64) -> f64[>..] {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < n { v.push(0.0); i += 1; }
    return v;
}

// diag_of builds the matrix of side n that holds the value v on its
// diagonal and zero elsewhere.
fn diag_of(n: i64, v: f64) -> scientific::Arr {
    var d = zeros(n * n);
    var i: i64 = 0;
    while i < n { d[i * n + i] = v; i += 1; }
    return common::mat(d, n, n);
}

fn test_zpk2tf_pins() {
    let cases = filter_forms_pins::ZT_K.len;
    assert(cases >= 8);
    var i: i64 = 0;
    while i < cases {
        let zr = common::part_of(filter_forms_pins::ZT_ZR_D, filter_forms_pins::ZT_ZR_L, i);
        let zi = common::part_of(filter_forms_pins::ZT_ZI_D, filter_forms_pins::ZT_ZI_L, i);
        let pr = common::part_of(filter_forms_pins::ZT_PR_D, filter_forms_pins::ZT_PR_L, i);
        let pim = common::part_of(filter_forms_pins::ZT_PI_D, filter_forms_pins::ZT_PI_L, i);
        let b, a, f = scientific::zpk2tf(zr, zi, pr, pim, filter_forms_pins::ZT_K[i]);
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(b, common::part_of(filter_forms_pins::ZT_B_D, filter_forms_pins::ZT_B_L, i), 1.0e-9));
        assert(common::rel_close(a, common::part_of(filter_forms_pins::ZT_A_D, filter_forms_pins::ZT_A_L, i), 1.0e-9));
        i += 1;
    }
}

fn test_zpk2tf_empty() {
    // No zeros and no poles give the gain and one.
    let b, a, f = scientific::zpk2tf([], [], [], [], 4.0);
    assert(f.kind == scientific::F_NONE);
    assert(b.data.len == 1 && b.data[0] == 4.0);
    assert(a.data.len == 1 && a.data[0] == 1.0);
}

fn test_zpk2tf_faults() {
    // Two lists of a different count.
    let b1, a1, f1 = scientific::zpk2tf([1.0, 2.0], [0.0], [], [], 1.0);
    assert(f1.kind == scientific::F_SHAPE);
    let b2, a2, f2 = scientific::zpk2tf([], [], [1.0], [0.0, 0.0], 1.0);
    assert(f2.kind == scientific::F_SHAPE);
    // A zero without a partner.
    let b3, a3, f3 = scientific::zpk2tf([-0.5], [0.3], [0.9], [0.0], 2.5);
    assert(f3.kind == scientific::F_SHAPE);
    // A pole without a partner.
    let b4, a4, f4 = scientific::zpk2tf([], [], [0.2, 0.2], [0.5, 0.6], 1.0);
    assert(f4.kind == scientific::F_SHAPE);
    // More than 64 zeros, and more than 64 poles.
    let big = zeros(65);
    let b5, a5, f5 = scientific::zpk2tf(big, big, [], [], 1.0);
    assert(f5.kind == scientific::F_LIMIT);
    let b6, a6, f6 = scientific::zpk2tf([], [], big, big, 1.0);
    assert(f6.kind == scientific::F_LIMIT);
    // A value that is not finite.
    let nan = 0.0 / 0.0;
    let b7, a7, f7 = scientific::zpk2tf([nan], [0.0], [], [], 1.0);
    assert(f7.kind == scientific::F_VALUE);
    let b8, a8, f8 = scientific::zpk2tf([], [], [1.0], [1.0 / 0.0], 1.0);
    assert(f8.kind == scientific::F_VALUE);
    let b9, a9, f9 = scientific::zpk2tf([], [], [], [], nan);
    assert(f9.kind == scientific::F_VALUE);
    // The limit itself holds: 64 zeros and 64 poles.
    let ok = zeros(64);
    let b10, a10, f10 = scientific::zpk2tf(ok, ok, ok, ok, 1.0);
    assert(f10.kind == scientific::F_NONE);
    assert(b10.data.len == 65 && a10.data.len == 65);
}

fn test_tf2zpk_pins() {
    let cases = filter_forms_pins::TZ_K.len;
    assert(cases >= 8);
    var i: i64 = 0;
    while i < cases {
        let b = common::part_of(filter_forms_pins::TZ_B_D, filter_forms_pins::TZ_B_L, i);
        let a = common::part_of(filter_forms_pins::TZ_A_D, filter_forms_pins::TZ_A_L, i);
        let zr, zi, pr, pim, k, f = scientific::tf2zpk(b, a);
        assert(f.kind == scientific::F_NONE);
        assert(common::close(k, filter_forms_pins::TZ_K[i], 1.0e-9));
        assert(roots_match(zr, zi, common::part_of(filter_forms_pins::TZ_ZR_D, filter_forms_pins::TZ_ZR_L, i),
                           common::part_of(filter_forms_pins::TZ_ZI_D, filter_forms_pins::TZ_ZI_L, i), 1.0e-6));
        assert(roots_match(pr, pim, common::part_of(filter_forms_pins::TZ_PR_D, filter_forms_pins::TZ_PR_L, i),
                           common::part_of(filter_forms_pins::TZ_PI_D, filter_forms_pins::TZ_PI_L, i), 1.0e-6));
        i += 1;
    }
}

fn test_tf2zpk_shapes() {
    // A denominator of one coefficient holds no pole.
    let zr, zi, pr, pim, k, f = scientific::tf2zpk([4.0, 1.0], [2.0]);
    assert(f.kind == scientific::F_NONE);
    assert(zr.data.len == 1 && pr.data.len == 0 && pim.data.len == 0);
    assert(common::close(k, 2.0, 1.0e-12));
    // The first coefficient of the denominator scales the gain.
    let zr2, zi2, pr2, pi2, k2, f2 = scientific::tf2zpk([3.0, 0.0, 0.0], [2.0, 0.0, 8.0]);
    assert(f2.kind == scientific::F_NONE);
    assert(common::close(k2, 1.5, 1.0e-12));
}

fn test_tf2zpk_faults() {
    let e = zeros(0);
    let z1, z2, z3, z4, k1, f1 = scientific::tf2zpk(e, [1.0, 2.0]);
    assert(f1.kind == scientific::F_VALUE);
    let y1, y2, y3, y4, k2, f2 = scientific::tf2zpk([1.0, 2.0], e);
    assert(f2.kind == scientific::F_VALUE);
    // A denominator of only zeros.
    let w1, w2, w3, w4, k3, f3 = scientific::tf2zpk([1.0], [0.0, 0.0]);
    assert(f3.kind == scientific::F_VALUE);
    // A numerator of only zeros has no gain.
    let v1, v2, v3, v4, k4, f4 = scientific::tf2zpk([0.0, 0.0], [1.0, 2.0]);
    assert(f4.kind == scientific::F_VALUE);
    // A list of more than 65 coefficients.
    let big = zeros(66);
    let u1, u2, u3, u4, k5, f5 = scientific::tf2zpk(big, [1.0]);
    assert(f5.kind == scientific::F_LIMIT);
    let t1, t2, t3, t4, k6, f6 = scientific::tf2zpk([1.0], big);
    assert(f6.kind == scientific::F_LIMIT);
    // A value that is not finite.
    let nan = 0.0 / 0.0;
    let s1, s2, s3, s4, k7, f7 = scientific::tf2zpk([nan, 1.0], [1.0, 2.0]);
    assert(f7.kind == scientific::F_VALUE);
    let r1, r2, r3, r4, k8, f8 = scientific::tf2zpk([1.0, 2.0], [1.0, 1.0 / 0.0]);
    assert(f8.kind == scientific::F_VALUE);
    // A denominator so small that the division overflows.
    let q1, q2, q3, q4, k9, f9 = scientific::tf2zpk([1.0e300, 1.0], [1.0e-300, 1.0]);
    assert(f9.kind == scientific::F_VALUE);
}

// check_ss compares the four parts of a state space form with a pin.
fn check_ss(am: scientific::Arr&, bm: scientific::Arr&, cm: scientific::Arr&, d: f64, n: i64,
            wa: f64[:], wb: f64[:], wc: f64[:], wd: f64) {
    assert(am.ndim == 2 && am.shape[0] == n && am.shape[1] == n);
    assert(bm.ndim == 2 && bm.shape[0] == n && bm.shape[1] == 1);
    assert(cm.ndim == 2 && cm.shape[0] == 1 && cm.shape[1] == n);
    assert(common::rel_close(am, wa, 1.0e-9));
    assert(common::rel_close(bm, wb, 1.0e-9));
    assert(common::rel_close(cm, wc, 1.0e-9));
    assert(common::close(d, wd, 1.0e-9));
}

fn test_tf2ss_pins() {
    let cases = filter_forms_pins::TS_D.len;
    assert(cases >= 8);
    var i: i64 = 0;
    while i < cases {
        let b = common::part_of(filter_forms_pins::TS_B_D, filter_forms_pins::TS_B_L, i);
        let a = common::part_of(filter_forms_pins::TS_A_D, filter_forms_pins::TS_A_L, i);
        let am, bm, cm, d, f = scientific::tf2ss(b, a);
        assert(f.kind == scientific::F_NONE);
        check_ss(am, bm, cm, d, filter_forms_pins::TS_N[i] as i64,
                 common::part_of(filter_forms_pins::TS_AM_D, filter_forms_pins::TS_AM_L, i),
                 common::part_of(filter_forms_pins::TS_BM_D, filter_forms_pins::TS_BM_L, i),
                 common::part_of(filter_forms_pins::TS_CM_D, filter_forms_pins::TS_CM_L, i),
                 filter_forms_pins::TS_D[i]);
        i += 1;
    }
}

fn test_tf2ss_one_state() {
    // A denominator of one coefficient gives one state that holds zero.
    let am, bm, cm, d, f = scientific::tf2ss([5.0], [1.0]);
    assert(f.kind == scientific::F_NONE);
    assert(am.shape[0] == 1 && am.shape[1] == 1 && am.data[0] == 0.0);
    assert(bm.data.len == 1 && bm.data[0] == 0.0);
    assert(cm.data.len == 1 && cm.data[0] == 0.0);
    assert(d == 5.0);
}

fn test_tf2ss_trim() {
    // A leading coefficient of 1e-15 goes. One of 1e-13 stays.
    let a1, b1, c1, d1, f1 = scientific::tf2ss([1.0e-15, 1.0, 2.0], [1.0, 3.0, 2.0]);
    assert(f1.kind == scientific::F_NONE);
    assert(d1 == 0.0);
    assert(common::close(c1.data[0], 1.0, 1.0e-12) && common::close(c1.data[1], 2.0, 1.0e-12));
    let a2, b2, c2, d2, f2 = scientific::tf2ss([1.0e-13, 1.0, 2.0], [1.0, 3.0, 2.0]);
    assert(f2.kind == scientific::F_NONE);
    assert(d2 == 1.0e-13);
}

fn test_tf2ss_faults() {
    // A numerator longer than the denominator.
    let a1, b1, c1, d1, f1 = scientific::tf2ss([1.0, 2.0, 3.0], [1.0, 2.0]);
    assert(f1.kind == scientific::F_SHAPE);
    // An empty list.
    let e = zeros(0);
    let a2, b2, c2, d2, f2 = scientific::tf2ss(e, [1.0, 2.0]);
    assert(f2.kind == scientific::F_VALUE);
    let a3, b3, c3, d3, f3 = scientific::tf2ss([1.0], e);
    assert(f3.kind == scientific::F_VALUE);
    // A denominator of only zeros.
    let a4, b4, c4, d4, f4 = scientific::tf2ss([1.0], [0.0, 0.0]);
    assert(f4.kind == scientific::F_VALUE);
    // More than 65 coefficients, and a value that is not finite.
    let big = zeros(66);
    let a5, b5, c5, d5, f5 = scientific::tf2ss([1.0], big);
    assert(f5.kind == scientific::F_LIMIT);
    let a6, b6, c6, d6, f6 = scientific::tf2ss([1.0], [1.0, 0.0 / 0.0]);
    assert(f6.kind == scientific::F_VALUE);
    // The limit itself holds: 65 coefficients give 64 states.
    var den: f64[>..] = [1.0];
    var i: i64 = 1;
    while i < 65 { den.push(0.0); i += 1; }
    let a7, b7, c7, d7, f7 = scientific::tf2ss([1.0], den);
    assert(f7.kind == scientific::F_NONE);
    assert(a7.shape[0] == 64 && a7.shape[1] == 64);
}

fn test_zpk2ss_pins() {
    let cases = filter_forms_pins::ZS_D.len;
    assert(cases >= 8);
    var i: i64 = 0;
    while i < cases {
        let zr = common::part_of(filter_forms_pins::ZS_ZR_D, filter_forms_pins::ZS_ZR_L, i);
        let zi = common::part_of(filter_forms_pins::ZS_ZI_D, filter_forms_pins::ZS_ZI_L, i);
        let pr = common::part_of(filter_forms_pins::ZS_PR_D, filter_forms_pins::ZS_PR_L, i);
        let pim = common::part_of(filter_forms_pins::ZS_PI_D, filter_forms_pins::ZS_PI_L, i);
        let am, bm, cm, d, f = scientific::zpk2ss(zr, zi, pr, pim, filter_forms_pins::ZS_K[i]);
        assert(f.kind == scientific::F_NONE);
        check_ss(am, bm, cm, d, filter_forms_pins::ZS_N[i] as i64,
                 common::part_of(filter_forms_pins::ZS_AM_D, filter_forms_pins::ZS_AM_L, i),
                 common::part_of(filter_forms_pins::ZS_BM_D, filter_forms_pins::ZS_BM_L, i),
                 common::part_of(filter_forms_pins::ZS_CM_D, filter_forms_pins::ZS_CM_L, i),
                 filter_forms_pins::ZS_D[i]);
        i += 1;
    }
}

fn test_zpk2ss_faults() {
    // A zero without a partner.
    let a1, b1, c1, d1, f1 = scientific::zpk2ss([-0.5], [0.3], [0.9], [0.0], 2.5);
    assert(f1.kind == scientific::F_SHAPE);
    // More zeros than poles make a numerator longer than the denominator.
    let a2, b2, c2, d2, f2 = scientific::zpk2ss([1.0, 2.0], [0.0, 0.0], [1.0], [0.0], 1.0);
    assert(f2.kind == scientific::F_SHAPE);
    // Two lists of a different count.
    let a3, b3, c3, d3, f3 = scientific::zpk2ss([], [], [1.0], [], 1.0);
    assert(f3.kind == scientific::F_SHAPE);
}

fn test_ss2tf_pins() {
    let cases = filter_forms_pins::ST_D.len;
    assert(cases >= 8);
    var i: i64 = 0;
    while i < cases {
        let n = filter_forms_pins::ST_N[i] as i64;
        let am = common::mat(common::part_of(filter_forms_pins::ST_A_D, filter_forms_pins::ST_A_L, i), n, n);
        let bv = common::part_of(filter_forms_pins::ST_B_D, filter_forms_pins::ST_B_L, i);
        let cv = common::part_of(filter_forms_pins::ST_C_D, filter_forms_pins::ST_C_L, i);
        let num, den, f = scientific::ss2tf(am, bv, cv, filter_forms_pins::ST_D[i]);
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(num, common::part_of(filter_forms_pins::ST_NUM_D, filter_forms_pins::ST_NUM_L, i), 1.0e-8));
        assert(common::rel_close(den, common::part_of(filter_forms_pins::ST_DEN_D, filter_forms_pins::ST_DEN_L, i), 1.0e-8));
        i += 1;
    }
}

fn test_ss2tf_faults() {
    // A matrix that is not square.
    let m23 = common::mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let n1, d1, f1 = scientific::ss2tf(m23, [1.0, 2.0], [1.0, 2.0], 0.0);
    assert(f1.kind == scientific::F_SHAPE);
    // A vector in place of a matrix.
    let v = common::av([1.0, 2.0]);
    let n2, d2, f2 = scientific::ss2tf(v, [1.0, 2.0], [1.0, 2.0], 0.0);
    assert(f2.kind == scientific::F_SHAPE);
    // An input or an output that does not match the side.
    let m = diag_of(2, 1.0);
    let n3, d3, f3 = scientific::ss2tf(m, [1.0], [1.0, 2.0], 0.0);
    assert(f3.kind == scientific::F_SHAPE);
    let n4, d4, f4 = scientific::ss2tf(m, [1.0, 2.0], [1.0, 2.0, 3.0], 0.0);
    assert(f4.kind == scientific::F_SHAPE);
    // A matrix of no row, and a forged buffer.
    let none = scientific::Arr { ndim: 2, shape: [0, 0], data: [] };
    let n5, d5, f5 = scientific::ss2tf(none, [], [], 0.0);
    assert(f5.kind == scientific::F_SHAPE);
    let forged = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0, 2.0, 3.0] };
    let n6, d6, f6 = scientific::ss2tf(forged, [1.0, 2.0], [1.0, 2.0], 0.0);
    assert(f6.kind == scientific::F_SHAPE);
    // A side above 64.
    let big = diag_of(65, 0.5);
    let bz = zeros(65);
    let n7, d7, f7 = scientific::ss2tf(big, bz, bz, 0.0);
    assert(f7.kind == scientific::F_LIMIT);
    // A value that is not finite.
    let nan = 0.0 / 0.0;
    let bad = common::mat([nan, 0.0, 0.0, 1.0], 2, 2);
    let n8, d8, f8 = scientific::ss2tf(bad, [1.0, 2.0], [1.0, 2.0], 0.0);
    assert(f8.kind == scientific::F_VALUE);
    let n9, d9, f9 = scientific::ss2tf(m, [1.0, nan], [1.0, 2.0], 0.0);
    assert(f9.kind == scientific::F_VALUE);
    let n10, d10, f10 = scientific::ss2tf(m, [1.0, 2.0], [1.0, 2.0], 1.0 / 0.0);
    assert(f10.kind == scientific::F_VALUE);
    // The limit itself holds: 64 states.
    let edge = diag_of(64, 0.5);
    let bz2 = zeros(64);
    let n11, d11, f11 = scientific::ss2tf(edge, bz2, bz2, 0.0);
    assert(f11.kind == scientific::F_NONE);
    assert(d11.data.len == 65);
}

fn test_forms_round_trip() {
    // The zeros, the poles and the gain go to the coefficients and come back.
    let b, a, f = scientific::zpk2tf([-0.5, -0.5, 0.25], [0.3, -0.3, 0.0], [0.9, 0.2, 0.2], [0.0, 0.4, -0.4], 2.5);
    assert(f.kind == scientific::F_NONE);
    let zr, zi, pr, pim, k, g = scientific::tf2zpk(b.data, a.data);
    assert(g.kind == scientific::F_NONE);
    assert(common::close(k, 2.5, 1.0e-9));
    assert(roots_match(zr, zi, [-0.5, -0.5, 0.25], [0.3, -0.3, 0.0], 1.0e-6));
    assert(roots_match(pr, pim, [0.9, 0.2, 0.2], [0.0, 0.4, -0.4], 1.0e-6));
    // The coefficients go to the state space form and come back.
    let am, bm, cm, d, h = scientific::tf2ss(b.data, a.data);
    assert(h.kind == scientific::F_NONE);
    let nb, na, s = scientific::ss2tf(am, bm.data, cm.data, d);
    assert(s.kind == scientific::F_NONE);
    assert(common::rel_close(nb, b.data, 1.0e-8));
    assert(common::rel_close(na, a.data, 1.0e-8));
}

fn test_design_round_trip() {
    // A design with distinct roots goes to the zeros and back. A repeated root loses digits and may not pair.
    let b, a, f = scientific::cheby2(4, 40.0, [0.3], "lowpass");
    assert(f.kind == scientific::F_NONE);
    assert(b.data.len == 5 && a.data.len == 5);
    let zr, zi, pr, pim, k, g = scientific::tf2zpk(b.data, a.data);
    assert(g.kind == scientific::F_NONE);
    let b2, a2, f2 = scientific::zpk2tf(zr.data, zi.data, pr.data, pim.data, k);
    assert(f2.kind == scientific::F_NONE);
    assert(common::rel_close(b2, b.data, 1.0e-6));
    assert(common::rel_close(a2, a.data, 1.0e-6));
}

fn test_tiny_imaginary_roots() {
    // A root whose imaginary part is tiny must not pair with itself.
    let b1, a1, f1 = scientific::zpk2tf([0.5], [1.0e-11], [], [], 1.0);
    assert(f1.kind == scientific::F_SHAPE);
    // A true pair with a tiny imaginary part keeps its degree.
    let b2, a2, f2 = scientific::zpk2tf([0.5, 0.5], [1.0e-11, -1.0e-11], [], [], 1.0);
    assert(f2.kind == scientific::F_NONE);
    assert(b2.data.len == 3);
    // A nearly zero state matrix gives a fault or a result of the right length, never a stop.
    let e = 1.0e-11;
    let m = common::mat([1.0, -e, e, 0.0], 2, 2);
    let n3, d3, f3 = scientific::ss2tf(m, [1.0, 0.0], [1.0, 0.0], 1.0);
    if f3.kind == scientific::F_NONE { assert(n3.data.len == 3 && d3.data.len == 3); }
    let m2 = common::mat([0.0, -e, e, 0.0], 2, 2);
    let n4, d4, f4 = scientific::ss2tf(m2, [1.0, 1.0], [1.0, 1.0], 1.0);
    if f4.kind == scientific::F_NONE { assert(n4.data.len == 3 && d4.data.len == 3); }
}

fn test_overflow_gives_fault() {
    let b1, a1, f1 = scientific::zpk2tf([], [], [1.0e200, 1.0e200], [0.0, 0.0], 1.0);
    assert(f1.kind == scientific::F_VALUE);
    let b2, a2, f2 = scientific::zpk2tf([2.0], [0.0], [], [], 1.0e308);
    assert(f2.kind == scientific::F_VALUE);
    let a3, b3, c3, d3, f3 = scientific::zpk2ss([], [], [1.0e200, 1.0e200], [0.0, 0.0], 1.0);
    assert(f3.kind == scientific::F_VALUE);
    let z1, z2, z3, z4, k4, f4 = scientific::tf2zpk([1.0e-13, 1.0e300], [1.0, 1.0]);
    assert(f4.kind == scientific::F_VALUE);
}
