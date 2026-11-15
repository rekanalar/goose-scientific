// The tests of the plain calls of the optimization and of the
// responses of the systems. Run them with SCIENTIFIC_SUITES=optimize_more.
//
// The pins come from SciPy 1.18.1. The pins of the Hessian updates
// come from NumPy, because the rule of the update holds no call of
// SciPy. Every fault path of the new calls holds a check.
namespace optimize_more_tests;

import scientific from "../..";
import common;
import optimize_more_pins;

// mat22 builds the matrix of two states from four values.
fn mat22(v: f64[:]) -> scientific::Arr {
    return common::mat(v, 2, 2);
}

// sys_a gives the matrix of the states of the system of the pins.
fn sys_a() -> scientific::Arr {
    return mat22([-2.0, -1.0, 1.0, 0.0]);
}

// sys_b gives the input of the system of the pins.
fn sys_b() -> f64[>..] {
    var v: f64[>..] = [1.0, 0.0];
    return v;
}

// sys_c gives the output of the system of the pins.
fn sys_c() -> f64[>..] {
    var v: f64[>..] = [0.0, 1.0];
    return v;
}

// rel_list tells whether two lists agree to a relative tolerance.
fn rel_list(a: f64[:], v: f64[:], tol: f64) -> bool {
    if a.len != v.len { return false; }
    var i: i64 = 0;
    while i < v.len {
        if common::mathshim_abs(a[i] - v[i]) > tol * (1.0 + common::mathshim_abs(v[i])) { return false; }
        i += 1;
    }
    return true;
}

// head3 gives the first three values of a result.
fn head3(a: scientific::Arr&) -> f64[>..] {
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i < 3 && i < a.data.len { v.push(a.data[i]); i += 1; }
    return v;
}

// rosen_block gives the value and the gradient of the function of
// Rosenbrock in one list.
fn rosen_block(v: f64[:]) -> f64[>..] {
    var s: f64 = 0.0;
    var g: f64[>..] = [];
    var i: i64 = 0;
    while i < v.len { g.push(0.0); i += 1; }
    i = 0;
    while i + 1 < v.len {
        let d = v[i + 1] - v[i] * v[i];
        s += 100.0 * d * d + (1.0 - v[i]) * (1.0 - v[i]);
        g[i] += -400.0 * v[i] * d - 2.0 * (1.0 - v[i]);
        g[i + 1] += 200.0 * d;
        i += 1;
    }
    var o: f64[>..] = [];
    o.push(s);
    i = 0;
    while i < g.len { o.push(g[i]); i += 1; }
    return o;
}

// sqrt_block gives the image of the square root of every value.
fn sqrt_block(v: f64[:]) -> f64[>..] {
    var o: f64[>..] = [];
    var i: i64 = 0;
    while i < v.len { o.push(common::mathshim_sqrt(v[i])); i += 1; }
    return o;
}

// inv_block gives the image of one over one plus the value.
fn inv_block(v: f64[:]) -> f64[>..] {
    var o: f64[>..] = [];
    var i: i64 = 0;
    while i < v.len { o.push(1.0 / (1.0 + v[i])); i += 1; }
    return o;
}

// quartic gives the value of x^4 - 2 x^2 at the first value.
fn quartic(v: f64[:]) -> f64 {
    let x = v[0];
    return x * x * x * x - 2.0 * x * x;
}

// quartic_block gives the value and the gradient of the quartic.
fn quartic_block(v: f64[:]) -> f64[>..] {
    let x = v[0];
    return [x * x * x * x - 2.0 * x * x, 4.0 * x * x * x - 4.0 * x];
}

// parabola_block gives the value and the gradient of (x - 3)^2.
fn parabola_block(v: f64[:]) -> f64[>..] {
    let x = v[0] - 3.0;
    return [x * x, 2.0 * x];
}

// bowl gives the value of the bowl of the pins. The first case of the
// grid holds two directions, so the bowl reads two values. The second
// case holds one direction, so the bowl of one value reads one value.
fn bowl(v: f64[:]) -> f64 {
    if v.len == 1 {
        let x = v[0] - 0.7;
        return x * x;
    }
    return (v[0] - 1.0) * (v[0] - 1.0) + (v[1] + 2.0) * (v[1] + 2.0);
}

// bracket_f gives the value of the cubic of the pins.
fn bracket_f(x: f64) -> f64 {
    return x * x * x - 2.0 * x - 2.0;
}

// square_f gives the value of the square of the pins.
fn square_f(x: f64) -> f64 {
    return (x - 1.0) * (x - 1.0);
}

fn test_rosen_pins() {
    let cases = optimize_more_pins::ROSEN_X_L.len;
    var i: i64 = 0;
    while i < cases {
        let x = common::part_of(optimize_more_pins::ROSEN_X_D, optimize_more_pins::ROSEN_X_L, i);
        assert(common::close(scientific::rosen(x), optimize_more_pins::ROSEN_F[i], 1.0e-12));
        i += 1;
    }
    let g = scientific::rosen_der(optimize_more_pins::RD_X);
    assert(common::rel_close(g, optimize_more_pins::RD_G, 1.0e-12));
    let h = scientific::rosen_hess(optimize_more_pins::RH_X);
    assert(h.ndim == 2 && h.shape[0] == 3 && h.shape[1] == 3);
    assert(common::rel_close(h, optimize_more_pins::RH_H, 1.0e-12));
    let hp = scientific::rosen_hess_prod(optimize_more_pins::RHP_X, optimize_more_pins::RHP_P);
    assert(common::rel_close(hp, optimize_more_pins::RHP_H, 1.0e-12));
}

fn test_check_grad_pins() {
    let cases = optimize_more_pins::CG_X_L.len;
    var i: i64 = 0;
    while i < cases {
        let x = common::part_of(optimize_more_pins::CG_X_D, optimize_more_pins::CG_X_L, i);
        let d, f = scientific::check_grad(x) { v => rosen_block(v) };
        assert(f.kind == scientific::F_NONE);
        // The two runs hold the same arithmetic, so the distance is
        // the same to a few digits of the last place.
        assert(common::close(d, optimize_more_pins::CG_D[i], 1.0e-6));
        i += 1;
    }
}

fn test_fixed_point_pins() {
    let x0 = common::part_of(optimize_more_pins::FP_X_D, optimize_more_pins::FP_X_L, 0);
    let p0, f0 = scientific::fixed_point(x0) { v => sqrt_block(v) };
    assert(f0.kind == scientific::F_NONE);
    assert(common::rel_close(p0, common::part_of(optimize_more_pins::FP_P_D, optimize_more_pins::FP_P_L, 0), 1.0e-9));
    let x1 = common::part_of(optimize_more_pins::FP_X_D, optimize_more_pins::FP_X_L, 1);
    let p1, f1 = scientific::fixed_point(x1) { v => inv_block(v) };
    assert(f1.kind == scientific::F_NONE);
    assert(common::rel_close(p1, common::part_of(optimize_more_pins::FP_P_D, optimize_more_pins::FP_P_L, 1), 1.0e-9));
}

fn test_line_search_pins() {
    let a1, f1 = scientific::line_search([0.0], [1.0], 9.0, -6.0) { v => parabola_block(v) };
    assert(f1.kind == scientific::F_NONE);
    assert(common::close(a1, optimize_more_pins::LS_A[0], 1.0e-9));
    let a2, f2 = scientific::line_search([0.5], [1.0], -0.4375, -1.5) { v => quartic_block(v) };
    assert(f2.kind == scientific::F_NONE);
    assert(common::close(a2, optimize_more_pins::LS_A[1], 1.0e-9));
}

fn test_bracket_pins() {
    let p0, v0, f0 = scientific::bracket(0.1, 1.0) { x => square_f(x) };
    assert(f0.kind == scientific::F_NONE);
    assert(common::rel_close(p0, common::part_of(optimize_more_pins::BR_P_D, optimize_more_pins::BR_P_L, 0), 1.0e-9));
    assert(common::rel_close(v0, common::part_of(optimize_more_pins::BR_V_D, optimize_more_pins::BR_V_L, 0), 1.0e-9));
    let p1, v1, f1 = scientific::bracket(-1.0, 0.0) { x => bracket_f(x) };
    assert(f1.kind == scientific::F_NONE);
    assert(common::rel_close(p1, common::part_of(optimize_more_pins::BR_P_D, optimize_more_pins::BR_P_L, 1), 1.0e-9));
    assert(common::rel_close(v1, common::part_of(optimize_more_pins::BR_V_D, optimize_more_pins::BR_V_L, 1), 1.0e-9));
    let p2, v2, f2 = scientific::bracket(1.0, 4.0) { x => common::mathshim_sin(x) };
    assert(f2.kind == scientific::F_NONE);
    assert(common::rel_close(p2, common::part_of(optimize_more_pins::BR_P_D, optimize_more_pins::BR_P_L, 2), 1.0e-9));
    assert(common::rel_close(v2, common::part_of(optimize_more_pins::BR_V_D, optimize_more_pins::BR_V_L, 2), 1.0e-9));
}

fn test_brute_pins() {
    let cases = optimize_more_pins::BT_P_L.len;
    var i: i64 = 0;
    while i < cases {
        let ns = optimize_more_pins::BT_NS[i] as i64;
        let lower = common::part_of(optimize_more_pins::BT_LO, [2.0, 1.0], i);
        let upper = common::part_of(optimize_more_pins::BT_HI, [2.0, 1.0], i);
        let p, v, f = scientific::brute(lower, upper, ns) { vv => bowl(vv) };
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(p, common::part_of(optimize_more_pins::BT_P_D, optimize_more_pins::BT_P_L, i), 1.0e-9));
        assert(common::close(v, optimize_more_pins::BT_V[i], 1.0e-9));
        i += 1;
    }
}

fn test_isotonic_pins() {
    let cases = optimize_more_pins::ISO_Y_L.len;
    var i: i64 = 0;
    while i < cases {
        let y = common::part_of(optimize_more_pins::ISO_Y_D, optimize_more_pins::ISO_Y_L, i);
        let x, w, f = scientific::isotonic_regression(y);
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(x, common::part_of(optimize_more_pins::ISO_X_D, optimize_more_pins::ISO_X_L, i), 1.0e-12));
        assert(common::rel_close(w, common::part_of(optimize_more_pins::ISO_W_D, optimize_more_pins::ISO_W_L, i), 1.0e-12));
        i += 1;
    }
}

fn test_hessian_update_pins() {
    let b = mat22(optimize_more_pins::BFGS_B);
    let out, f = scientific::bfgs_hessian_update(b, optimize_more_pins::BFGS_S, optimize_more_pins::BFGS_Y);
    assert(f.kind == scientific::F_NONE);
    assert(common::rel_close(out, optimize_more_pins::BFGS_OUT, 1.0e-12));
    let b2 = mat22(optimize_more_pins::SR1_B);
    let out2, f2 = scientific::sr1_hessian_update(b2, optimize_more_pins::SR1_S, optimize_more_pins::SR1_Y);
    assert(f2.kind == scientific::F_NONE);
    assert(common::rel_close(out2, optimize_more_pins::SR1_OUT, 1.0e-12));
    // The skip: a pair that gives no move holds the matrix.
    var y0: f64[>..] = [2.0, -1.0];
    let out3, f3 = scientific::sr1_hessian_update(b2, optimize_more_pins::SR1_S, y0);
    assert(f3.kind == scientific::F_NONE);
    assert(common::rel_close(out3, optimize_more_pins::SR1_B, 1.0e-15));
    // The skip with the limit of the caller.
    let out4, f4 = scientific::sr1_hessian_update(b2, optimize_more_pins::SR1_S, y0, 1.0e-8);
    assert(f4.kind == scientific::F_NONE);
    assert(common::rel_close(out4, optimize_more_pins::SR1_B, 1.0e-15));
}

fn test_bounds_and_constraints() {
    let lo, up, f = scientific::bounds([-1.0, 0.0], [2.0, 5.0]);
    assert(f.kind == scientific::F_NONE);
    assert(lo.data.len == 2 && up.data.len == 2);
    assert(lo.data[0] == -1.0 && up.data[1] == 5.0);
    // The values of the linear constraint come back in the three parts.
    let am, clo, cup, f2 = scientific::linear_constraint(common::mat([1.0, 2.0, 0.0, 1.0], 2, 2), [1.0, -2.0], [4.0, 2.0]);
    assert(f2.kind == scientific::F_NONE);
    assert(am.ndim == 2 && am.shape[0] == 2 && am.shape[1] == 2);
    assert(clo.data[0] == 1.0 && cup.data[1] == 2.0);
    // The text of the options holds the names of the calls.
    let text = scientific::show_options();
    assert(text.len > 0);
}

fn test_impulse_pins() {
    let am = sys_a();
    let t, h, f = scientific::impulse(am, sys_b(), sys_c(), 0.0);
    assert(f.kind == scientific::F_NONE);
    assert(t.data.len == optimize_more_pins::IMP_HN[0] as i64);
    assert(rel_list(head3(t), optimize_more_pins::IMP_T3, 1.0e-9));
    assert(rel_list(head3(h), optimize_more_pins::IMP_H3, 1.0e-7));
    assert(common::close(t.data[t.data.len - 1], optimize_more_pins::IMP_TLAST[0], 1.0e-9));
    // The run of three points of the caller.
    let t2, h2, f2 = scientific::impulse(am, sys_b(), sys_c(), 0.0, [0.0, 0.5, 1.0]);
    assert(f2.kind == scientific::F_NONE);
    assert(common::rel_close(h2, optimize_more_pins::IMP_HT, 1.0e-7));
}

fn test_step_pins() {
    let am = sys_a();
    let t, s, f = scientific::step(am, sys_b(), sys_c(), 0.0);
    assert(f.kind == scientific::F_NONE);
    assert(rel_list(head3(s), optimize_more_pins::STEP_S3, 1.0e-7));
    assert(common::close(s.data[s.data.len - 1], optimize_more_pins::STEP_SLAST[0], 1.0e-7));
    let t2, s2, f2 = scientific::step(am, sys_b(), sys_c(), 0.0, [0.0, 0.5, 1.0]);
    assert(f2.kind == scientific::F_NONE);
    assert(common::rel_close(s2, optimize_more_pins::STEP_ST, 1.0e-7));
}

fn test_lsim_pins() {
    let am = sys_a();
    let u = optimize_more_pins::LSIM_U;
    let t = optimize_more_pins::LSIM_T;
    let tt, y, x, f = scientific::lsim(am, sys_b(), sys_c(), 0.0, u, t);
    assert(f.kind == scientific::F_NONE);
    assert(common::rel_close(tt, t, 1.0e-15));
    assert(common::rel_close(y, optimize_more_pins::LSIM_Y, 1.0e-7));
    assert(x.ndim == 2 && x.shape[0] == 10 && x.shape[1] == 2);
    // The held run.
    let tz, yz, xz, fz = scientific::lsim(am, sys_b(), sys_c(), 0.0, u, t, [], false);
    assert(fz.kind == scientific::F_NONE);
    assert(common::rel_close(yz, optimize_more_pins::LSIM_YZ, 1.0e-7));
    // The run from the start of two states.
    let t0, y0, x0, f0 = scientific::lsim(am, sys_b(), sys_c(), 0.0, u, t, optimize_more_pins::LSIM_X0);
    assert(f0.kind == scientific::F_NONE);
    assert(common::rel_close(y0, optimize_more_pins::LSIM_Y0, 1.0e-7));
    var xr: f64[>..] = [x0.data[0], x0.data[1]];
    var xr2: f64[>..] = [x0.data[2], x0.data[3]];
    assert(rel_list(xr, common::part_of(optimize_more_pins::LSIM_XX_D, optimize_more_pins::LSIM_XX_L, 0), 1.0e-7));
    assert(rel_list(xr2, common::part_of(optimize_more_pins::LSIM_XX_D, optimize_more_pins::LSIM_XX_L, 1), 1.0e-7));
}

fn test_freqresp_bode_pins() {
    let w = optimize_more_pins::FR_W;
    let wr, hr, hi, f = scientific::freqresp([1.0], [1.0, 2.0, 1.0], w);
    assert(f.kind == scientific::F_NONE);
    assert(common::rel_close(hr, optimize_more_pins::FR_HR, 1.0e-9));
    assert(common::rel_close(hi, optimize_more_pins::FR_HI, 1.0e-9));
    let wm, m, p, f2 = scientific::bode([1.0], [1.0, 2.0, 1.0], w);
    assert(f2.kind == scientific::F_NONE);
    assert(common::rel_close(m, optimize_more_pins::BODE_M, 1.0e-9));
    assert(common::rel_close(p, optimize_more_pins::BODE_P, 1.0e-9));
    // The run of the default grid holds one hundred points.
    let wd, md, pd, f3 = scientific::bode([1.0], [1.0, 2.0, 1.0]);
    assert(f3.kind == scientific::F_NONE);
    assert(wd.data.len == optimize_more_pins::BODE_N[0] as i64);
    assert(common::close(wd.data[0], optimize_more_pins::BODE_WFIRST[0], 1.0e-9));
    assert(common::close(wd.data[wd.data.len - 1], optimize_more_pins::BODE_WFIRST[1], 1.0e-9));
    let md2 = head3(md);
    let pd2 = head3(pd);
    var md2v: f64[>..] = [md2[0], md2[1]];
    var pd2v: f64[>..] = [pd2[0], pd2[1]];
    assert(rel_list(md2v, optimize_more_pins::BODE_M2, 1.0e-9));
    assert(rel_list(pd2v, optimize_more_pins::BODE_P2, 1.0e-9));
}

fn test_discrete_pins() {
    let t, y, f = scientific::dstep([1.0], [1.0, 2.0, 1.0], 4);
    assert(f.kind == scientific::F_NONE);
    assert(common::rel_close(t, optimize_more_pins::DSTEP_T, 1.0e-15));
    assert(common::rel_close(y, optimize_more_pins::DSTEP_Y, 1.0e-12));
    let t2, y2, f2 = scientific::dimpulse([1.0], [1.0, 2.0, 1.0], 4);
    assert(f2.kind == scientific::F_NONE);
    assert(common::rel_close(y2, optimize_more_pins::DIMP_Y, 1.0e-12));
    // The step of the sample of one half moves the samples only.
    let t3, y3, f3 = scientific::dstep([1.0], [1.0, 2.0, 1.0], 4, 0.5);
    assert(f3.kind == scientific::F_NONE);
    assert(common::rel_close(t3, optimize_more_pins::DSTEP_T05, 1.0e-15));
    assert(common::rel_close(y3, optimize_more_pins::DSTEP_Y, 1.0e-12));
    // The answers over the grid of the caller.
    let w, hr, hi, f4 = scientific::dfreqresp([1.0], [1.0, 2.0, 1.0], optimize_more_pins::DFR_WR);
    assert(f4.kind == scientific::F_NONE);
    assert(common::rel_close(hr, optimize_more_pins::DFR_HR, 1.0e-9));
    assert(common::rel_close(hi, optimize_more_pins::DFR_HI, 1.0e-9));
    let w2, hr2, hi2, f5 = scientific::dfreqresp([1.0], [1.0, -0.5], optimize_more_pins::DFR2_W);
    assert(f5.kind == scientific::F_NONE);
    assert(common::rel_close(hr2, optimize_more_pins::DFR2_HR_D, 1.0e-9));
    assert(common::rel_close(hi2, optimize_more_pins::DFR2_HI_D, 1.0e-9));
    // The size and the phase at the frequency of one.
    let wm, m, p, f6 = scientific::dbode([1.0], [1.0, 2.0, 1.0], [1.0]);
    assert(f6.kind == scientific::F_NONE);
    assert(common::rel_close(m, optimize_more_pins::DB_M, 1.0e-9));
    assert(common::rel_close(p, optimize_more_pins::DB_P, 1.0e-9));
    // The step of the sample of one half doubles the frequencies.
    let wm2, m2, p2, f7 = scientific::dbode([1.0], [1.0, 2.0, 1.0], [1.0], 100, 0.5);
    assert(f7.kind == scientific::F_NONE);
    assert(common::rel_close(wm2, optimize_more_pins::DB_W05, 1.0e-12));
    assert(common::rel_close(m2, optimize_more_pins::DB_M05, 1.0e-9));
    assert(common::rel_close(p2, optimize_more_pins::DB_P05, 1.0e-9));
    // Two frequencies of the caller.
    let wm3, m3, p3, f8 = scientific::dbode([1.0], [1.0, 2.0, 1.0], [0.5, 1.0]);
    assert(f8.kind == scientific::F_NONE);
    assert(common::rel_close(m3, optimize_more_pins::DB_M2, 1.0e-9));
    assert(common::rel_close(p3, optimize_more_pins::DB_P2, 1.0e-9));
}

fn test_state_form_pins() {
    let am = mat22([1.0, 0.0, 0.0, 1.0]);
    var b: f64[>..] = [0.0, 0.0];
    var c: f64[>..] = [0.0, 0.0];
    let zr, zi, pr, pim, k, f = scientific::ss2zpk(am, b, c, 1.0);
    assert(f.kind == scientific::F_NONE);
    assert(common::rel_close(zr, optimize_more_pins::SS_Z, 1.0e-9));
    assert(common::rel_close(pr, optimize_more_pins::SS_P, 1.0e-9));
    // A double root loses half the digits, so the imaginary parts stay
    // within the square root of the machine epsilon of zero.
    assert(zi.data.len == 2 && common::close(zi.data[0], 0.0, 1.0e-7) && common::close(zi.data[1], 0.0, 1.0e-7));
    assert(pim.data.len == 2 && common::close(pim.data[0], 0.0, 1.0e-7) && common::close(pim.data[1], 0.0, 1.0e-7));
    assert(common::close(k, optimize_more_pins::SS_K[0], 1.0e-12));
    // The check of the shapes gives the four parts back.
    let a2, b2, c2, d2, f2 = scientific::abcd_normalize(am, [1.0, 5.0], [0.0, 0.0], 2.5);
    assert(f2.kind == scientific::F_NONE);
    assert(a2.shape[0] == 2 && a2.shape[1] == 2);
    assert(b2.shape[0] == 2 && b2.shape[1] == 1);
    assert(c2.shape[0] == 1 && c2.shape[1] == 2);
    assert(d2.shape[0] == 1 && d2.shape[1] == 1 && d2.data[0] == 2.5);
}

fn test_faults() {
    // The point of one value holds no gradient.
    let g = scientific::rosen_der([1.0]);
    assert(g.data.len == 0);
    let d1, f1 = scientific::check_grad([]) { v => rosen_block(v) };
    assert(f1.kind == scientific::F_VALUE);
    let p1, f2 = scientific::fixed_point([1.0, 2.0], 0.0) { v => sqrt_block(v) };
    assert(f2.kind == scientific::F_VALUE);
    let b1, b2, f3 = scientific::bracket(0.0, 1.0, 110.0, 0) { x => square_f(x) };
    assert(f3.kind == scientific::F_VALUE);
    let p2, v1, f4 = scientific::brute([1.0], [0.0], 5) { vv => bowl(vv) };
    assert(f4.kind == scientific::F_VALUE);
    let x1, w1, f5 = scientific::isotonic_regression([1.0, 2.0], [1.0]);
    assert(f5.kind == scientific::F_SHAPE);
    // A pair without curvature gives a fault.
    let bm = mat22(optimize_more_pins::BFGS_B);
    var s: f64[>..] = [1.0, 1.0];
    var yv: f64[>..] = [1.0, -1.0];
    let o1, f6 = scientific::bfgs_hessian_update(bm, s, yv);
    assert(f6.kind == scientific::F_VALUE);
    // The bounds of the wrong count and of the wrong order.
    let l1, u1, f7 = scientific::bounds([1.0], [2.0, 3.0]);
    assert(f7.kind == scientific::F_SHAPE);
    let l2, u2, f8 = scientific::bounds([3.0], [2.0]);
    assert(f8.kind == scientific::F_VALUE);
    // The constraint of the bounds of the wrong count.
    let cm, cl, cu, f9 = scientific::linear_constraint(common::mat([1.0, 2.0], 1, 2), [1.0, 2.0], [3.0, 4.0]);
    assert(f9.kind == scientific::F_SHAPE);
    let cm2, cl2, cu2, f10 = scientific::linear_constraint(common::mat([1.0, 2.0], 1, 2), [4.0], [3.0]);
    assert(f10.kind == scientific::F_VALUE);
    // The form of the wrong shape.
    let a1, bb1, cc1, dd1, f11 = scientific::abcd_normalize(mat22([1.0, 0.0, 0.0, 1.0]), [1.0, 2.0, 3.0], [0.0, 0.0], 1.0);
    assert(f11.kind == scientific::F_SHAPE);
    // The times of the wrong count.
    let am = sys_a();
    let t1, y1, x1, f12 = scientific::lsim(am, sys_b(), sys_c(), 0.0, [1.0, 2.0], [0.0, 1.0, 2.0]);
    assert(f12.kind == scientific::F_SHAPE);
    // The count of the samples of zero.
    let t2, y2, f13 = scientific::dstep([1.0], [1.0, 2.0, 1.0], 0);
    assert(f13.kind == scientific::F_RANGE);
    // The step of the sample of zero.
    let t3, y3, p1b, f14 = scientific::dbode([1.0], [1.0, 2.0, 1.0], [1.0], 100, 0.0);
    assert(f14.kind == scientific::F_VALUE);
}

// A matrix that a caller built by hand and broke does not agree with
// its shape. The calls that read the buffer of a matrix give a fault.
fn test_optimize_more_broken_matrix() {
    // The Hessian holds fewer values than its shape asks for.
    let bad = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let out, f1 = scientific::bfgs_hessian_update(bad, [1.0, 2.0], [1.0, 2.0]);
    assert(f1.kind == scientific::F_SHAPE);
    let out2, f2 = scientific::sr1_hessian_update(bad, [1.0, 2.0], [1.0, 2.0]);
    assert(f2.kind == scientific::F_SHAPE);
    // The matrix of the states holds fewer values than its shape asks for.
    let am = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let t, h, f3 = scientific::impulse(am, [1.0, 2.0], [0.0, 0.0], 0.0);
    assert(f3.kind == scientific::F_SHAPE);
}
