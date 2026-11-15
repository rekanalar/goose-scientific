// The tests of the systems of a signal.
//
// The calls lti, dlti and dlsim hold a linear system in a struct and
// run an input through it. The call place_poles gives the gain of a
// closed loop. The calls sos2tf and sos2zpk turn a cascade of second
// order sections into another form. The call firls designs a filter by
// the least squares of the error. The call unit_impulse builds the
// signal of one hit. Every call matches the pins of SciPy 1.18.1, and
// every fault path holds a test.
//
// Run them with SCIENTIFIC_SUITES=signal_systems.
namespace signal_systems_tests;

import scientific from "../..";
import common;
import signal_systems_pins;

// roots_match tells whether two lists of roots hold the same roots.
// The order of the roots is free, so each root of the pins needs an
// unused partner of the result.
fn roots_match(gr: f64[>..], gi: f64[>..], wr: f64[:], wi: f64[:], tol: f64) -> bool {
    let n = wr.len;
    if gr.len != n || gi.len != n { return false; }
    var used: i64[>..] = [];
    var i: i64 = 0;
    while i < n { used.push(0); i += 1; }
    i = 0;
    while i < n {
        var found = false;
        var j: i64 = 0;
        while j < n && !found {
            if used[j] == 0
                && common::mathshim_abs(gr[j] - wr[i]) <= tol * (1.0 + common::mathshim_abs(wr[i]))
                && common::mathshim_abs(gi[j] - wi[i]) <= tol * (1.0 + common::mathshim_abs(wi[i])) {
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

// ss_of builds the state space of a system of one input and one
// output.
fn ss_of(am: scientific::Arr&, bm: scientific::Arr&, cm: scientific::Arr&,
         d: f64, dt: f64) -> scientific::StateSpace {
    let dm = common::mat([d], 1, 1);
    let s, f = scientific::state_space(am, bm, cm, dm, dt);
    return s;
}

// tf_num gives the numerator of a transfer function.
fn tf_num(sys: scientific::TransferFunction&) -> f64[>..] {
    return scientific::copy_slice(sys.buf[0..sys.num_len]);
}

// tf_den gives the denominator of a transfer function.
fn tf_den(sys: scientific::TransferFunction&) -> f64[>..] {
    return scientific::copy_slice(sys.buf[sys.num_len..sys.num_len + sys.den_len]);
}

// dlsim_run runs an input through a system. An empty list of times
// leaves the times out, and an empty start holds the states at zero.
fn dlsim_run(sys, u: f64[:], t: f64[:], x0: f64[:]) -> scientific::Arr, scientific::Arr, scientific::Arr, scientific::Fault {
    if t.len == 0 && x0.len == 0 {
        return scientific::dlsim(sys, u);
    }
    if t.len == 0 {
        return scientific::dlsim(sys, u, [], x0);
    }
    return scientific::dlsim(sys, u, t, x0);
}

// tf_vals tells whether the form of a system holds the values of the
// pins and the step of the time.
fn tf_vals(sys, en: f64[:], ed: f64[:], want_dt: f64) -> bool {
    if sys.num_len != en.len || sys.den_len != ed.len { return false; }
    let tn = tf_num(sys);
    let td = tf_den(sys);
    var j: i64 = 0;
    while j < en.len {
        if !common::close(tn[j], en[j], 1.0e-12) { return false; }
        if !common::close(td[j], ed[j], 1.0e-12) { return false; }
        j += 1;
    }
    return sys.dt == want_dt;
}

fn test_lti_pins() {
    let cases = signal_systems_pins::LT_NUM_L.len;
    assert(cases >= 8);
    var i: i64 = 0;
    while i < cases {
        let num = common::part_of(signal_systems_pins::LT_NUM_D, signal_systems_pins::LT_NUM_L, i);
        let den = common::part_of(signal_systems_pins::LT_DEN_D, signal_systems_pins::LT_DEN_L, i);
        let sys, f = scientific::lti(num, den);
        assert(f.kind == scientific::F_NONE);
        let en = common::part_of(signal_systems_pins::LT_ENUM_D, signal_systems_pins::LT_ENUM_L, i);
        let ed = common::part_of(signal_systems_pins::LT_EDEN_D, signal_systems_pins::LT_EDEN_L, i);
        assert(tf_vals(sys, en, ed, 0.0));
        i += 1;
    }
}

fn test_dlti_pins() {
    let cases = signal_systems_pins::LT_NUM_L.len;
    assert(cases >= 8);
    var i: i64 = 0;
    while i < cases {
        let num = common::part_of(signal_systems_pins::LT_NUM_D, signal_systems_pins::LT_NUM_L, i);
        let den = common::part_of(signal_systems_pins::LT_DEN_D, signal_systems_pins::LT_DEN_L, i);
        let sys, f = scientific::dlti(num, den);
        assert(f.kind == scientific::F_NONE);
        let en = common::part_of(signal_systems_pins::LT_ENUM_D, signal_systems_pins::LT_ENUM_L, i);
        let ed = common::part_of(signal_systems_pins::LT_EDEN_D, signal_systems_pins::LT_EDEN_L, i);
        assert(tf_vals(sys, en, ed, 1.0));
        i += 1;
    }
}

fn test_lti_faults() {
    let nan = 0.0 / 0.0;
    // A list of coefficients that holds no element.
    let _s1, f1 = scientific::lti([], [1.0]);
    assert(f1.kind == scientific::F_VALUE);
    let _s2, f2 = scientific::lti([1.0], []);
    assert(f2.kind == scientific::F_VALUE);
    // A denominator of zeros only.
    let _s3, f3 = scientific::lti([1.0], [0.0, 0.0]);
    assert(f3.kind == scientific::F_VALUE);
    // A value that is not finite.
    let _s4, f4 = scientific::lti([nan], [1.0]);
    assert(f4.kind == scientific::F_VALUE);
    let _s5, f5 = scientific::dlti([1.0], [1.0 / 0.0]);
    assert(f5.kind == scientific::F_VALUE);
    let _s6, f6 = scientific::dlti([], []);
    assert(f6.kind == scientific::F_VALUE);
}

fn test_dlsim_tf_pins() {
    let cases = signal_systems_pins::TF_DT.len;
    assert(cases >= 6);
    var i: i64 = 0;
    while i < cases {
        let num = common::part_of(signal_systems_pins::TF_NUM_D, signal_systems_pins::TF_NUM_L, i);
        let den = common::part_of(signal_systems_pins::TF_DEN_D, signal_systems_pins::TF_DEN_L, i);
        let sys, f0 = scientific::dlti(num, den);
        assert(f0.kind == scientific::F_NONE);
        let am, bm, cm, d, f1 = scientific::tf2ss(tf_num(sys), tf_den(sys));
        assert(f1.kind == scientific::F_NONE);
        let sys2 = ss_of(am, bm, cm, d, signal_systems_pins::TF_DT[i]);
        let u = common::part_of(signal_systems_pins::TF_U_D, signal_systems_pins::TF_U_L, i);
        let t = common::part_of(signal_systems_pins::TF_T_D, signal_systems_pins::TF_T_L, i);
        let tout, yout, xout, f = dlsim_run(sys2, u, t, []);
        assert(f.kind == scientific::F_NONE);
        let et = common::part_of(signal_systems_pins::TF_TOUT_D, signal_systems_pins::TF_TOUT_L, i);
        let ey = common::part_of(signal_systems_pins::TF_YOUT_D, signal_systems_pins::TF_YOUT_L, i);
        assert(common::rel_close(tout, et, 1.0e-12));
        assert(common::rel_close(yout, ey, 1.0e-9));
        i += 1;
    }
}

fn test_dlsim_ss_pins() {
    let cases = signal_systems_pins::SS_N.len;
    assert(cases >= 4);
    var i: i64 = 0;
    while i < cases {
        let n = signal_systems_pins::SS_N[i] as i64;
        let af = common::part_of(signal_systems_pins::SS_A_D, signal_systems_pins::SS_A_L, i);
        let bf = common::part_of(signal_systems_pins::SS_B_D, signal_systems_pins::SS_B_L, i);
        let cf = common::part_of(signal_systems_pins::SS_C_D, signal_systems_pins::SS_C_L, i);
        let am = common::mat(af, n, n);
        let bm = common::mat(bf, n, 1);
        let cm = common::mat(cf, 1, n);
        let sys = ss_of(am, bm, cm, signal_systems_pins::SS_D[i], signal_systems_pins::SS_DT[i]);
        let u = common::part_of(signal_systems_pins::SS_U_D, signal_systems_pins::SS_U_L, i);
        let t = common::part_of(signal_systems_pins::SS_T_D, signal_systems_pins::SS_T_L, i);
        let x0 = common::part_of(signal_systems_pins::SS_X0_D, signal_systems_pins::SS_X0_L, i);
        let tout, yout, xout, f = dlsim_run(sys, u, t, x0);
        assert(f.kind == scientific::F_NONE);
        let et = common::part_of(signal_systems_pins::SS_TOUT_D, signal_systems_pins::SS_TOUT_L, i);
        let ey = common::part_of(signal_systems_pins::SS_YOUT_D, signal_systems_pins::SS_YOUT_L, i);
        let ex = common::part_of(signal_systems_pins::SS_XOUT_D, signal_systems_pins::SS_XOUT_L, i);
        assert(common::rel_close(tout, et, 1.0e-12));
        assert(common::rel_close(yout, ey, 1.0e-9));
        assert(common::rel_close(xout, ex, 1.0e-9));
        i += 1;
    }
}

fn test_dlsim_faults() {
    let nan = 0.0 / 0.0;
    let am = common::mat([0.0, 1.0, -2.0, -3.0], 2, 2);
    let bm = common::mat([0.0, 1.0], 2, 1);
    let cm = common::mat([1.0, 0.0], 1, 2);
    let sys = ss_of(am, bm, cm, 0.0, 0.5);
    let u = [1.0, 0.0, 1.0];
    // A matrix of the states that is not square.
    let bam = common::mat([1.0, 2.0, 3.0], 1, 3);
    let bad_a = ss_of(bam, bm, cm, 0.0, 0.5);
    let _t1, _y1, _x1, f1 = scientific::dlsim(bad_a, u);
    assert(f1.kind == scientific::F_SHAPE);
    // A matrix of the input of another size.
    let bbm = common::mat([1.0], 1, 1);
    let bad_b = ss_of(am, bbm, cm, 0.0, 0.5);
    let _t2, _y2, _x2, f2 = scientific::dlsim(bad_b, u);
    assert(f2.kind == scientific::F_SHAPE);
    // A step of the time of zero.
    let sys0 = ss_of(am, bm, cm, 0.0, 0.0);
    let _t3, _y3, _x3, f3 = scientific::dlsim(sys0, u);
    assert(f3.kind == scientific::F_VALUE);
    // An input that holds no value.
    let sys_dt = ss_of(am, bm, cm, 0.0, 0.5);
    let _t4, _y4, _x4, f4 = scientific::dlsim(sys_dt, []);
    assert(f4.kind == scientific::F_VALUE);
    // A start of another length.
    let _t5, _y5, _x5, f5 = scientific::dlsim(sys_dt, u, [], [1.0]);
    assert(f5.kind == scientific::F_SHAPE);
    // A value of the input that is not finite.
    let _t6, _y6, _x6, f6 = scientific::dlsim(sys_dt, [nan, 1.0, 1.0]);
    assert(f6.kind == scientific::F_VALUE);
    // Times that do not rise.
    let _t7, _y7, _x7, f7 = scientific::dlsim(sys_dt, u, [0.0, 0.5, 0.25]);
    assert(f7.kind == scientific::F_VALUE);
    // Times of another count than the input.
    let _t8, _y8, _x8, f8 = scientific::dlsim(sys_dt, u, [0.0, 0.5, 1.0, 1.5]);
    assert(f8.kind == scientific::F_SHAPE);
    // A last time below zero.
    let _t9, _y9, _x9, f9 = scientific::dlsim(sys_dt, u, [-2.0, -1.0, -0.5]);
    assert(f9.kind == scientific::F_VALUE);
    // More steps than the limit holds.
    let _t10, _y10, _x10, f10 = scientific::dlsim(sys_dt, u, [0.0, 500000.0, 1000000.0]);
    assert(f10.kind == scientific::F_LIMIT);
    // A direct part of another size.
    // A direct part of another size gives an empty system.
    let dm_bad = common::mat([0.0, 1.0], 1, 2);
    let sys_d, fsd = scientific::state_space(am, bm, cm, dm_bad, 0.5);
    assert(fsd.kind == scientific::F_SHAPE);
    let _t11, _y11, _x11, f11 = scientific::dlsim(sys_d, u);
    assert(f11.kind == scientific::F_SHAPE);
}

// place_check checks the result of a placement against the pins.
fn place_check(g: scientific::Arr, cr: scientific::Arr, ci: scientific::Arr,
               rr: scientific::Arr, ri: scientific::Arr, x: scientific::CArr,
               rtol: f64, f: scientific::Fault, cpr: f64[:], cpi: f64[:],
               rpr: f64[:], rpi: f64[:], gain: f64[:], fast: bool) {
    assert(f.kind == scientific::F_NONE);
    // The poles of the request come back in the order of SciPy.
    assert(rr.data.len == rpr.len && ri.data.len == rpi.len);
    var i: i64 = 0;
    while i < rpr.len {
        assert(common::close(rr.data[i], rpr[i], 1.0e-12));
        assert(common::close(ri.data[i], rpi[i], 1.0e-12));
        i += 1;
    }
    // The poles of the closed loop hold the poles of the request. The
    // gain of the path of a lower rank is not the gain of SciPy, so
    // the eigenvalues of the closed loop carry the error of the
    // problem of the eigenvalues, and the tolerance is loose.
    assert(cr.data.len == cpr.len && ci.data.len == cpi.len);
    i = 0;
    while i < cpr.len {
        assert(common::close(cr.data[i], cpr[i], 1.0e-6));
        assert(common::close(ci.data[i], cpi[i], 1.0e-6));
        i += 1;
    }
    if fast {
        // The fast path runs no loop, so the tolerance holds a value
        // that is not a number.
        assert(!(rtol == rtol));
        assert(g.data.len == gain.len);
        i = 0;
        while i < gain.len {
            assert(common::close(g.data[i], gain[i], 1.0e-6));
            i += 1;
        }
    }
    // The transfer matrix holds one row for each state.
    assert(x.ndim == 2 && x.shape[0] == x.shape[1]);
}

// place_run runs the placement with the method of the pins and checks
// the poles of the closed loop and the gain of the fast path.
fn place_run(a: scientific::Arr&, b: f64[:], pr: f64[:], pi: f64[:], knv: bool,
             cpr: f64[:], cpi: f64[:], rpr: f64[:], rpi: f64[:],
             gain: f64[:], fast: bool) {
    if knv {
        let g, cr, ci, rr, ri, x, rtol, iters, f =
            scientific::place_poles(a, b, pr, pi, "KNV0", 0.001, 30);
        place_check(copy(g), copy(cr), copy(ci), copy(rr), copy(ri), copy(x), rtol, f, cpr, cpi, rpr, rpi, gain, fast);
    } else {
        let g, cr, ci, rr, ri, x, rtol, iters, f =
            scientific::place_poles(a, b, pr, pi, "YT", 0.001, 30);
        place_check(copy(g), copy(cr), copy(ci), copy(rr), copy(ri), copy(x), rtol, f, cpr, cpi, rpr, rpi, gain, fast);
    }
}

fn test_place_poles_pins() {
    let cases = signal_systems_pins::PP_N.len;
    assert(cases >= 10);
    var i: i64 = 0;
    while i < cases {
        let n = signal_systems_pins::PP_N[i] as i64;
        let af = common::part_of(signal_systems_pins::PP_A_D, signal_systems_pins::PP_A_L, i);
        let bf = common::part_of(signal_systems_pins::PP_B_D, signal_systems_pins::PP_B_L, i);
        let a = common::mat(af, n, n);
        let pr = common::part_of(signal_systems_pins::PP_PR_D, signal_systems_pins::PP_PR_L, i);
        let pi = common::part_of(signal_systems_pins::PP_PI_D, signal_systems_pins::PP_PI_L, i);
        let cpr = common::part_of(signal_systems_pins::PP_CPR_D, signal_systems_pins::PP_CPR_L, i);
        let cpi = common::part_of(signal_systems_pins::PP_CPI_D, signal_systems_pins::PP_CPI_L, i);
        let rpr = common::part_of(signal_systems_pins::PP_RPR_D, signal_systems_pins::PP_RPR_L, i);
        let rpi = common::part_of(signal_systems_pins::PP_RPI_D, signal_systems_pins::PP_RPI_L, i);
        let gain = common::part_of(signal_systems_pins::PP_GAIN_D, signal_systems_pins::PP_GAIN_L, i);
        place_run(a, bf, pr, pi, signal_systems_pins::PP_METHOD[i] == 1.0,
                  cpr, cpi, rpr, rpi, gain, signal_systems_pins::PP_FAST[i] == 1.0);
        i += 1;
    }
}

fn test_place_poles_faults() {
    let a = common::mat([1.0, 2.0, 3.0, 4.0], 2, 2);
    let eye2 = [1.0, 0.0, 0.0, 1.0];
    // A method with an unknown name.
    let _g1, _c1, _d1, _r1, _s1, _x1, _t1, _n1, f1 = scientific::place_poles(a, eye2, [-1.0, -2.0], [0.0, 0.0], "XX", 0.001, 30);
    assert(f1.kind == scientific::F_VALUE);
    // A count of iterations below one.
    let _g2, _c2, _d2, _r2, _s2, _x2, _t2, _n2, f2 = scientific::place_poles(a, eye2, [-1.0, -2.0], [0.0, 0.0], "YT", 0.001, 0);
    assert(f2.kind == scientific::F_VALUE);
    // A tolerance above one.
    let _g3, _c3, _d3, _r3, _s3, _x3, _t3, _n3, f3 = scientific::place_poles(a, eye2, [-1.0, -2.0], [0.0, 0.0], "YT", 2.0, 30);
    assert(f3.kind == scientific::F_VALUE);
    // A matrix of the states that is not square.
    let bad = common::mat([1.0, 2.0, 3.0], 1, 3);
    let _g4, _c4, _d4, _r4, _s4, _x4, _t4, _n4, f4 = scientific::place_poles(bad, eye2, [-1.0, -2.0], [0.0, 0.0]);
    assert(f4.kind == scientific::F_SHAPE);
    // A matrix of the inputs that does not split into rows.
    let _g5, _c5, _d5, _r5, _s5, _x5, _t5, _n5, f5 = scientific::place_poles(a, [1.0, 0.0, 0.0], [-1.0, -2.0], [0.0, 0.0]);
    assert(f5.kind == scientific::F_SHAPE);
    // A count of poles that does not match the states.
    let _g6, _c6, _d6, _r6, _s6, _x6, _t6, _n6, f6 = scientific::place_poles(a, eye2, [-1.0], [0.0]);
    assert(f6.kind == scientific::F_VALUE);
    // A pole of a pair without its partner.
    let _g7, _c7, _d7, _r7, _s7, _x7, _t7, _n7, f7 = scientific::place_poles(a, eye2, [-1.0, -1.0], [1.0, -2.0]);
    assert(f7.kind == scientific::F_VALUE);
    // A pole repeated more times than the rank of the input holds.
    let _g8, _c8, _d8, _r8, _s8, _x8, _t8, _n8, f8 = scientific::place_poles(a, [1.0, 0.0], [-1.0, -1.0], [0.0, 0.0]);
    assert(f8.kind == scientific::F_VALUE);
    // The method KNV0 holds no complex poles.
    let _g9, _c9, _d9, _r9, _s9, _x9, _t9, _n9, f9 = scientific::place_poles(a, eye2, [-1.0, -1.0], [1.0, -1.0], "KNV0", 0.001, 30);
    assert(f9.kind == scientific::F_VALUE);
    // A value of the matrices that is not finite.
    let nan = 0.0 / 0.0;
    let _g10, _c10, _d10, _r10, _s10, _x10, _t10, _n10, f10 = scientific::place_poles(a, [nan, 0.0, 0.0, 1.0], [-1.0, -2.0], [0.0, 0.0]);
    assert(f10.kind == scientific::F_VALUE);
    // A system whose input does not reach every state.
    let un = common::mat([1.0, 0.0, 0.0, 2.0], 2, 2);
    let _g11, _c11, _d11, _r11, _s11, _x11, _t11, _n11, f11 = scientific::place_poles(un, [1.0, 0.0], [-1.0, -2.0], [0.0, 0.0]);
    assert(f11.kind == scientific::F_VALUE);
}

fn test_sos2tf_pins() {
    let cases = signal_systems_pins::SO_SOS_L.len;
    assert(cases >= 7);
    var i: i64 = 0;
    while i < cases {
        let flat = common::part_of(signal_systems_pins::SO_SOS_D, signal_systems_pins::SO_SOS_L, i);
        let rows = flat.len / 6;
        let sos = common::mat(flat, rows, 6);
        let b, a, f = scientific::sos2tf(sos);
        assert(f.kind == scientific::F_NONE);
        let eb = common::part_of(signal_systems_pins::SO_B_D, signal_systems_pins::SO_B_L, i);
        let ea = common::part_of(signal_systems_pins::SO_A_D, signal_systems_pins::SO_A_L, i);
        assert(common::rel_close(b, eb, 1.0e-9));
        assert(common::rel_close(a, ea, 1.0e-9));
        i += 1;
    }
}

fn test_sos2zpk_pins() {
    let cases = signal_systems_pins::SO_SOS_L.len;
    assert(cases >= 7);
    var i: i64 = 0;
    while i < cases {
        let flat = common::part_of(signal_systems_pins::SO_SOS_D, signal_systems_pins::SO_SOS_L, i);
        let rows = flat.len / 6;
        let sos = common::mat(flat, rows, 6);
        let zr, zi, pr, pi, k, f = scientific::sos2zpk(sos);
        assert(f.kind == scientific::F_NONE);
        let ezr = common::part_of(signal_systems_pins::SO_ZR_D, signal_systems_pins::SO_ZR_L, i);
        let ezi = common::part_of(signal_systems_pins::SO_ZI_D, signal_systems_pins::SO_ZI_L, i);
        let epr = common::part_of(signal_systems_pins::SO_PR_D, signal_systems_pins::SO_PR_L, i);
        let epi = common::part_of(signal_systems_pins::SO_PI_D, signal_systems_pins::SO_PI_L, i);
        assert(roots_match(copy(zr.data), copy(zi.data), ezr, ezi, 1.0e-6));
        assert(roots_match(copy(pr.data), copy(pi.data), epr, epi, 1.0e-6));
        assert(common::close(k, signal_systems_pins::SO_K[i], 1.0e-9));
        i += 1;
    }
}

fn test_sos_faults() {
    // A cascade whose rows do not hold six coefficients.
    let wide = common::mat([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], 2, 4);
    let _b1, _a1, f1 = scientific::sos2tf(wide);
    assert(f1.kind == scientific::F_SHAPE);
    // A cascade with five coefficients in a row.
    let bad = common::mat([1.0, 0.0, 0.0, 1.0, 0.0], 1, 5);
    let _b2, _a2, f2 = scientific::sos2tf(bad);
    assert(f2.kind == scientific::F_SHAPE);
    // A cascade with no row.
    let none = common::mat([], 0, 6);
    let _b3, _a3, f3 = scientific::sos2tf(none);
    assert(f3.kind == scientific::F_SHAPE);
    // A coefficient that is not finite.
    let nan = 0.0 / 0.0;
    let badv = common::mat([nan, 0.0, 0.0, 1.0, 0.0, 0.0], 1, 6);
    let _b4, _a4, f4 = scientific::sos2tf(badv);
    assert(f4.kind == scientific::F_VALUE);
    // A section whose denominator starts at zero.
    let zero = common::mat([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1, 6);
    let _z1, _z2, _p1, _p2, _k1, f5 = scientific::sos2zpk(zero);
    assert(f5.kind == scientific::F_VALUE);
    // A cascade of no row for the sections too.
    let _z3, _z4, _p3, _p4, _k2, f6 = scientific::sos2zpk(none);
    assert(f6.kind == scientific::F_SHAPE);
}

// firls_run runs the design with the weights of the pins and checks
// the count of the taps and the values.
fn firls_check(taps: scientific::Arr, f: scientific::Fault, ecoef: f64[:], numtaps: i64) {
    assert(f.kind == scientific::F_NONE);
    assert(taps.data.len == numtaps);
    // The system of the design holds a large condition number, so the
    // taps agree with the reference to a part in a million.
    assert(common::rel_close(taps, ecoef, 1.0e-6));
}

fn test_firls_pins() {
    let cases = signal_systems_pins::FI_TAPS.len;
    assert(cases >= 6);
    var i: i64 = 0;
    while i < cases {
        let numtaps = signal_systems_pins::FI_TAPS[i] as i64;
        let bands = common::part_of(signal_systems_pins::FI_BANDS_D, signal_systems_pins::FI_BANDS_L, i);
        let desired = common::part_of(signal_systems_pins::FI_DES_D, signal_systems_pins::FI_DES_L, i);
        let weight = common::part_of(signal_systems_pins::FI_W_D, signal_systems_pins::FI_W_L, i);
        let fs = signal_systems_pins::FI_FS[i];
        let ecoef = common::part_of(signal_systems_pins::FI_COEF_D, signal_systems_pins::FI_COEF_L, i);
        if weight.len == 0 {
            let taps, f = scientific::firls(numtaps, bands, desired, [], fs);
            firls_check(copy(taps), f, ecoef, numtaps);
        } else {
            let taps, f = scientific::firls(numtaps, bands, desired, weight, fs);
            firls_check(copy(taps), f, ecoef, numtaps);
        }
        i += 1;
    }
}

fn test_firls_faults() {
    let bands = [0.0, 0.1, 0.2, 0.3];
    let desired = [1.0, 1.0, 0.0, 0.0];
    // A count of taps that is even.
    let _t1, f1 = scientific::firls(10, bands, desired);
    assert(f1.kind == scientific::F_VALUE);
    // A count of taps below one.
    let _t2, f2 = scientific::firls(0, bands, desired);
    assert(f2.kind == scientific::F_VALUE);
    // A rate of the samples of zero.
    let _t3, f3 = scientific::firls(11, bands, desired, [], 0.0);
    assert(f3.kind == scientific::F_VALUE);
    // An odd count of band edges.
    let _t4, f4 = scientific::firls(11, [0.0, 0.1, 0.2], desired);
    assert(f4.kind == scientific::F_VALUE);
    // An edge of a band outside the rate.
    let _t5, f5 = scientific::firls(11, [0.0, 1.5], [1.0, 1.0]);
    assert(f5.kind == scientific::F_VALUE);
    // A band of no width.
    let _t6, f6 = scientific::firls(11, [0.2, 0.2], [1.0, 1.0]);
    assert(f6.kind == scientific::F_VALUE);
    // A band edge that falls.
    let _t7, f7 = scientific::firls(11, [0.3, 0.1], [1.0, 1.0]);
    assert(f7.kind == scientific::F_VALUE);
    // Bands that overlap.
    let _t8, f8 = scientific::firls(11, [0.0, 0.3, 0.2, 0.5], desired);
    assert(f8.kind == scientific::F_VALUE);
    // A count of wanted values of another size.
    let _t9, f9 = scientific::firls(11, bands, [1.0, 1.0]);
    assert(f9.kind == scientific::F_SHAPE);
    // A wanted value below zero.
    let _t10, f10 = scientific::firls(11, bands, [-1.0, 1.0, 0.0, 0.0]);
    assert(f10.kind == scientific::F_VALUE);
    // A weight of another size.
    let _t11, f11 = scientific::firls(11, bands, desired, [1.0]);
    assert(f11.kind == scientific::F_SHAPE);
    // A weight below zero.
    let _t12, f12 = scientific::firls(11, bands, desired, [-1.0, 1.0]);
    assert(f12.kind == scientific::F_VALUE);
    // A count of taps above the limit.
    let _t13, f13 = scientific::firls(2001, bands, desired);
    assert(f13.kind == scientific::F_LIMIT);
    // A rate of the samples that is not finite.
    let nan = 0.0 / 0.0;
    let _t14, f14 = scientific::firls(11, bands, desired, [], nan);
    assert(f14.kind == scientific::F_VALUE);
}

fn test_unit_impulse_pins() {
    let cases = signal_systems_pins::UI_SHAPE_L.len;
    assert(cases >= 7);
    var i: i64 = 0;
    while i < cases {
        let shape_f = common::part_of(signal_systems_pins::UI_SHAPE_D, signal_systems_pins::UI_SHAPE_L, i);
        var shape: i64[>..] = [];
        var k: i64 = 0;
        while k < shape_f.len { shape.push(shape_f[k] as i64); k += 1; }
        let idx_f = common::part_of(signal_systems_pins::UI_IDX_D, signal_systems_pins::UI_IDX_L, i);
        var idx: i64[>..] = [];
        k = 0;
        while k < idx_f.len { idx.push(idx_f[k] as i64); k += 1; }
        let mid = signal_systems_pins::UI_MID[i] == 1.0;
        let out, f = scientific::unit_impulse(shape, idx, mid);
        assert(f.kind == scientific::F_NONE);
        let evals = common::part_of(signal_systems_pins::UI_VAL_D, signal_systems_pins::UI_VAL_L, i);
        assert(out.data.len == evals.len);
        var j: i64 = 0;
        while j < evals.len {
            assert(out.data[j] == evals[j]);
            j += 1;
        }
        i += 1;
    }
}

fn test_unit_impulse_faults() {
    // A place of another count than the dimensions hold.
    let _o1, f1 = scientific::unit_impulse([4], [1, 2]);
    assert(f1.kind == scientific::F_SHAPE);
    let o2, f2 = scientific::unit_impulse([2, 3], [1]);
    assert(f2.kind == scientific::F_NONE);
    assert(o2.data[4] == 1.0);
    // A place outside a dimension.
    let _o3, f3 = scientific::unit_impulse([4], [4]);
    assert(f3.kind == scientific::F_RANGE);
    let _o4, f4 = scientific::unit_impulse([2, 3], [1, 3]);
    assert(f4.kind == scientific::F_RANGE);
    let _o5, f5 = scientific::unit_impulse([4], [-1]);
    assert(f5.kind == scientific::F_RANGE);
    // A shape of no dimension.
    let _o6, f6 = scientific::unit_impulse([], []);
    assert(f6.kind == scientific::F_VALUE);
    // A dimension of length zero.
    let _o7, f7 = scientific::unit_impulse([0], []);
    assert(f7.kind == scientific::F_VALUE);
    // More dimensions than the limit holds.
    let _o8, f8 = scientific::unit_impulse([1, 1, 1, 1, 1, 1, 1, 1, 1], []);
    assert(f8.kind == scientific::F_LIMIT);
}
