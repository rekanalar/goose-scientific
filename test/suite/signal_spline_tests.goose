// The tests of the group signal_spline. Run them with
// SCIENTIFIC_SUITES=signal_spline.
//
// The spline filters, the fast convolution and the short time
// transform are pinned against SciPy 1.18.1.
namespace signal_spline_tests;

import scientific from "../..";
import common;
import signal_spline_pins;

// cs1_check compares one cspline1d result with the pins.
fn cs1_check(i: i64) {
    let at = common::part_start(signal_spline_pins::CS1_IN_L, i);
    let end = common::part_start(signal_spline_pins::CS1_IN_L, i + 1);
    var x: f64[>..] = [];
    var q: i64 = at;
    while q < end {
        x.push(signal_spline_pins::CS1_IN_D[q]);
        q += 1;
    }
    let c, f = scientific::cspline1d(x[..], signal_spline_pins::CS1_LAMB[i]);
    assert(f.kind == scientific::F_NONE);
    let oat = common::part_start(signal_spline_pins::CS1_OUT_L, i);
    let oend = common::part_start(signal_spline_pins::CS1_OUT_L, i + 1);
    var want: f64[>..] = [];
    q = oat;
    while q < oend {
        want.push(signal_spline_pins::CS1_OUT_D[q]);
        q += 1;
    }
    assert(c.data.len == want.len);
    q = 0;
    while q < want.len {
        assert(common::close(c.data[q], want[q], 1.0e-8));
        q += 1;
    }
}

// cs1_run runs every cspline1d and qspline1d case.
fn cs1_run() {
    var i: i64 = 0;
    while i < signal_spline_pins::CS1_L.len {
        cs1_check(i);
        i += 1;
    }
    i = 0;
    while i < signal_spline_pins::QS1_OUT_L.len {
        let x = common::part_of(signal_spline_pins::QS1_IN_D,
                                signal_spline_pins::QS1_IN_L, i);
        let c, f = scientific::qspline1d(x);
        assert(f.kind == scientific::F_NONE);
        let want = common::part_of(signal_spline_pins::QS1_OUT_D,
                                   signal_spline_pins::QS1_OUT_L, i);
        assert(c.data.len == want.len);
        var q: i64 = 0;
        while q < want.len {
            assert(common::close(c.data[q], want[q], 1.0e-8));
            q += 1;
        }
        i += 1;
    }
}

fn test_signal_spline_1d() {
    cs1_run();
}

// ev_check compares one eval result with the pins.
fn ev_check(i: i64) {
    let cj = common::part_of(signal_spline_pins::EV_C_D,
                             signal_spline_pins::EV_C_L, i);
    let nx = common::part_of(signal_spline_pins::EV_N_D,
                             signal_spline_pins::EV_N_L, i);
    let want = common::part_of(signal_spline_pins::EV_OUT_D,
                               signal_spline_pins::EV_OUT_L, i);
    var got: scientific::Arr = scientific::empty_arr();
    if signal_spline_pins::EV_Q[i] == 0.0 {
        let v, f = scientific::cspline1d_eval(cj, nx, signal_spline_pins::EV_DX[i],
                                              signal_spline_pins::EV_X0[i]);
        assert(f.kind == scientific::F_NONE);
        got = copy(v);
    } else {
        let v, f = scientific::qspline1d_eval(cj, nx, signal_spline_pins::EV_DX[i],
                                              signal_spline_pins::EV_X0[i]);
        assert(f.kind == scientific::F_NONE);
        got = copy(v);
    }
    assert(got.data.len == want.len);
    var q: i64 = 0;
    while q < want.len {
        assert(common::close(got.data[q], want[q], 1.0e-8));
        q += 1;
    }
}

fn test_signal_spline_eval() {
    var i: i64 = 0;
    while i < signal_spline_pins::EV_Q.len {
        ev_check(i);
        i += 1;
    }
}

// mat_of builds a matrix of the shape rows by cols from a pin list.
fn mat_of(d, at: i64, rows: i64, cols: i64) -> scientific::Arr {
    var v: f64[>..] = [];
    var q: i64 = at;
    while q < at + rows * cols {
        v.push(d[q]);
        q += 1;
    }
    return common::mat(copy(v), rows, cols);
}

// two_d_check compares one matrix result with the pins.
fn two_d_check(c: scientific::Arr, f: scientific::Fault, i: i64,
               od: f64[:], ol: f64[:]) {
    assert(f.kind == scientific::F_NONE);
    let want = common::part_of(od, ol, i);
    assert(c.data.len == want.len);
    var q: i64 = 0;
    while q < want.len {
        assert(common::close(c.data[q], want[q], 1.0e-8));
        q += 1;
    }
}

fn test_signal_spline_2d() {
    var i: i64 = 0;
    while i < signal_spline_pins::CS2_LAMB.len {
        let rows = signal_spline_pins::CS2_SH[2 * i];
        let cols = signal_spline_pins::CS2_SH[2 * i + 1];
        let at = common::part_start(signal_spline_pins::CS2_IN_L, i);
        let a = mat_of(signal_spline_pins::CS2_IN_D, at, rows as i64, cols as i64);
        let r, f = scientific::cspline2d(a, signal_spline_pins::CS2_LAMB[i]);
        two_d_check(copy(r), f, i, signal_spline_pins::CS2_OUT_D,
                    signal_spline_pins::CS2_OUT_L);
        i += 1;
    }
    i = 0;
    while i < signal_spline_pins::QS2_SH.len / 2 {
        let rows = signal_spline_pins::QS2_SH[2 * i];
        let cols = signal_spline_pins::QS2_SH[2 * i + 1];
        let at = common::part_start(signal_spline_pins::QS2_IN_L, i);
        let a = mat_of(signal_spline_pins::QS2_IN_D, at, rows as i64, cols as i64);
        let r, f = scientific::qspline2d(a);
        two_d_check(copy(r), f, i, signal_spline_pins::QS2_OUT_D,
                    signal_spline_pins::QS2_OUT_L);
        i += 1;
    }
    i = 0;
    while i < signal_spline_pins::SF_LM.len {
        let rows = signal_spline_pins::SF_SH[2 * i];
        let cols = signal_spline_pins::SF_SH[2 * i + 1];
        let at = common::part_start(signal_spline_pins::SF_IN_L, i);
        let a = mat_of(signal_spline_pins::SF_IN_D, at, rows as i64, cols as i64);
        let r, f = scientific::spline_filter(a, signal_spline_pins::SF_LM[i]);
        two_d_check(copy(r), f, i, signal_spline_pins::SF_OUT_D,
                    signal_spline_pins::SF_OUT_L);
        i += 1;
    }
}

fn test_signal_spline_iir() {
    var i: i64 = 0;
    while i < signal_spline_pins::SI1_C0.len {
        let x = common::part_of(signal_spline_pins::SI1_IN_D,
                                signal_spline_pins::SI1_IN_L, i);
        let r, f = scientific::symiirorder1(x, signal_spline_pins::SI1_C0[i],
                                            signal_spline_pins::SI1_Z1[i]);
        assert(f.kind == scientific::F_NONE);
        let want = common::part_of(signal_spline_pins::SI1_OUT_D,
                                   signal_spline_pins::SI1_OUT_L, i);
        assert(r.data.len == want.len);
        var q: i64 = 0;
        while q < want.len {
            assert(common::close(r.data[q], want[q], 1.0e-8));
            q += 1;
        }
        i += 1;
    }
    i = 0;
    while i < signal_spline_pins::SI2_R.len {
        let x = common::part_of(signal_spline_pins::SI2_IN_D,
                                signal_spline_pins::SI2_IN_L, i);
        let r, f = scientific::symiirorder2(x, signal_spline_pins::SI2_R[i],
                                            signal_spline_pins::SI2_OM[i]);
        assert(f.kind == scientific::F_NONE);
        let want = common::part_of(signal_spline_pins::SI2_OUT_D,
                                   signal_spline_pins::SI2_OUT_L, i);
        assert(r.data.len == want.len);
        var q: i64 = 0;
        while q < want.len {
            assert(common::close(r.data[q], want[q], 1.0e-8));
            q += 1;
        }
        i += 1;
    }
}

fn test_signal_spline_oa() {
    var i: i64 = 0;
    while i < signal_spline_pins::OA_M.len {
        let a = common::part_of(signal_spline_pins::OA_A_D,
                                signal_spline_pins::OA_A_L, i);
        let b = common::part_of(signal_spline_pins::OA_B_D,
                                signal_spline_pins::OA_B_L, i);
        var mode: const u8[:] = "full";
        if signal_spline_pins::OA_M[i] == 1.0 { mode = "same"; }
        if signal_spline_pins::OA_M[i] == 2.0 { mode = "valid"; }
        let r, f = scientific::oaconvolve(a, b, mode);
        assert(f.kind == scientific::F_NONE);
        let want = common::part_of(signal_spline_pins::OA_OUT_D,
                                   signal_spline_pins::OA_OUT_L, i);
        assert(r.data.len == want.len);
        var q: i64 = 0;
        while q < want.len {
            assert(common::close(r.data[q], want[q], 1.0e-8));
            q += 1;
        }
        i += 1;
    }
}

// up_mode gives the name of a mode of the upsample filter.
fn up_mode(code: f64) -> const u8[:] {
    if code == 0.0 { return "constant"; }
    if code == 1.0 { return "symmetric"; }
    if code == 2.0 { return "reflect"; }
    if code == 3.0 { return "edge"; }
    if code == 4.0 { return "wrap"; }
    if code == 5.0 { return "smooth"; }
    if code == 6.0 { return "antisymmetric"; }
    if code == 7.0 { return "antireflect"; }
    return "line";
}

fn test_signal_spline_upfirdn() {
    var i: i64 = 0;
    while i < signal_spline_pins::UP_M.len {
        let h = common::part_of(signal_spline_pins::UP_H_D,
                                signal_spline_pins::UP_H_L, i);
        let x = common::part_of(signal_spline_pins::UP_X_D,
                                signal_spline_pins::UP_X_L, i);
        let xa = common::av(x);
        let r, f = scientific::upfirdn(h, xa, signal_spline_pins::UP_UD[2 * i] as i64,
                                       signal_spline_pins::UP_UD[2 * i + 1] as i64,
                                       -1, up_mode(signal_spline_pins::UP_M[i]),
                                       signal_spline_pins::UP_C[i]);
        assert(f.kind == scientific::F_NONE);
        let want = common::part_of(signal_spline_pins::UP_OUT_D,
                                   signal_spline_pins::UP_OUT_L, i);
        assert(r.data.len == want.len);
        var q: i64 = 0;
        while q < want.len {
            q += 1;
        }
        i += 1;
    }
}

// stft_mode gives the name of a mode of the transform.
fn stft_mode(code: f64) -> const u8[:] {
    if code == 1.0 { return "twosided"; }
    if code == 2.0 { return "centered"; }
    return "onesided";
}

// stft_scale gives the name of a scale of the transform.
fn stft_scale(code: f64) -> const u8[:] {
    if code == 1.0 { return "spectrum"; }
    if code == 2.0 { return "psd"; }
    return "";
}

// stft_check compares one forward and one inverse run with the pins.
fn stft_check(i: i64) {
    let win = common::part_of(signal_spline_pins::ST_WIN_D,
                              signal_spline_pins::ST_WIN_L, i);
    let x = common::part_of(signal_spline_pins::ST_X_D,
                            signal_spline_pins::ST_X_L, i);
    let base = 6 * i;
    let hop = signal_spline_pins::ST_WH[base] as i64;
    let fs = signal_spline_pins::ST_WH[base + 1];
    let mode = stft_mode(signal_spline_pins::ST_WH[base + 2]);
    let scale = stft_scale(signal_spline_pins::ST_WH[base + 3]);
    let mfft = signal_spline_pins::ST_WH[base + 4] as i64;
    let ps = signal_spline_pins::ST_WH[base + 5];
    let plan, f = scientific::stft_plan(win, hop, fs, mode, mfft, scale, ps);
    assert(f.kind == scientific::F_NONE);
    let s, fs = scientific::stft_forward(plan, common::av(x), -1);
    assert(fs.kind == scientific::F_NONE);
    let rre = common::part_of(signal_spline_pins::ST_RE_D,
                              signal_spline_pins::ST_RE_L, i);
    let rim = common::part_of(signal_spline_pins::ST_IM_D,
                              signal_spline_pins::ST_IM_L, i);
    assert(s.data.len == 2 * rre.len);
    var q: i64 = 0;
    while q < rre.len {
        assert(common::near(s.data[2 * q], rre[q]));
        assert(common::near(s.data[2 * q + 1], rim[q]));
        q += 1;
    }
    let y, fy = scientific::stft_inverse(plan, s, -1);
    assert(fy.kind == scientific::F_NONE);
    let want = common::part_of(signal_spline_pins::ST_OUT_D,
                               signal_spline_pins::ST_OUT_L, i);
    assert(y.data.len == want.len);
    q = 0;
    while q < want.len {
        assert(common::near(y.data[q], want[q]));
        q += 1;
    }
}

fn test_signal_spline_stft() {
    var i: i64 = 0;
    while i < signal_spline_pins::ST_OUT_L.len {
        stft_check(i);
        i += 1;
    }
}

fn test_signal_spline_faults() {
    // A signal of one value takes no smoothing spline.
    let c1, f1 = scientific::cspline1d([1.5], 2.0);
    assert(f1.kind == scientific::F_SHAPE);
    // The quadratic spline takes no smoothing.
    let c2, f2 = scientific::qspline1d([1.0, 2.0, 3.0], 1.0);
    assert(f2.kind == scientific::F_VALUE);
    // An empty set of coefficients and a zero step give faults.
    let c3, f3 = scientific::cspline1d_eval([], [0.5]);
    assert(f3.kind == scientific::F_VALUE);
    let c4, f4 = scientific::qspline1d_eval([1.0, 2.0], [0.5], 0.0);
    assert(f4.kind == scientific::F_VALUE);
    // The fall-off of the smoothing spline stands outside the range.
    let c5, f5 = scientific::cspline1d([1.0, 2.0, 3.0], 0.005);
    assert(f5.kind == scientific::F_VALUE);
    // A matrix of one dimension gives a shape fault.
    let lie = scientific::Arr { ndim: 2, shape: [3], data: [1.0, 2.0, 3.0] };
    let c6, f6 = scientific::cspline2d(lie);
    assert(f6.kind == scientific::F_SHAPE);
    let ok2 = common::mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
    let c7, f7 = scientific::qspline2d(ok2, 0.5);
    assert(f7.kind == scientific::F_VALUE);
    let c8, f8 = scientific::spline_filter(lie);
    assert(f8.kind == scientific::F_SHAPE);
    // A forged matrix gives a shape fault before any read.
    let forge = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let c9, f9 = scientific::cspline2d(forge);
    assert(f9.kind == scientific::F_SHAPE);
    // The filter of order one refuses a pole at or above one.
    let c10, f10 = scientific::symiirorder1([1.0, 2.0, 3.0], 0.5, 1.0);
    assert(f10.kind == scientific::F_VALUE);
    // A short signal takes no mirror sum.
    let c11, f11 = scientific::symiirorder1([1.0], 0.5, 0.3);
    assert(f11.kind == scientific::F_VALUE);
    let c12, f12 = scientific::symiirorder2([1.0], 0.3, 0.6);
    assert(f12.kind == scientific::F_SHAPE);
    // The filter of order two refuses a pole at or above one.
    let c13, f13 = scientific::symiirorder2([1.0, 2.0, 3.0], 1.0, 0.6);
    assert(f13.kind == scientific::F_VALUE);
    // The overlap refuses an empty signal and an unknown mode.
    let c14, f14 = scientific::oaconvolve([], [1.0]);
    assert(f14.kind == scientific::F_SHAPE);
    let c15, f15 = scientific::oaconvolve([1.0], [1.0], "wide");
    assert(f15.kind == scientific::F_VALUE);
    // The upsample filter refuses bad taps, rates, modes and axes.
    let xa = common::av([1.0, 2.0, 3.0]);
    let c16, f16 = scientific::upfirdn([], xa);
    assert(f16.kind == scientific::F_VALUE);
    let c17, f17 = scientific::upfirdn([1.0], xa, 0, 1);
    assert(f17.kind == scientific::F_VALUE);
    let c18, f18 = scientific::upfirdn([1.0], xa, 1, 1, -1, "round");
    assert(f18.kind == scientific::F_VALUE);
    let c19, f19 = scientific::upfirdn([1.0], xa, 1, 1, 1);
    assert(f19.kind == scientific::F_RANGE);
    // The plan refuses an empty window and an unknown mode.
    let p1, g1 = scientific::stft_plan([], 2, 10.0);
    assert(g1.kind == scientific::F_SHAPE);
    let p2, g2 = scientific::stft_plan([1.0, 2.0, 3.0], 2, 10.0, "three");
    assert(g2.kind == scientific::F_VALUE);
    // The plan refuses a transform shorter than the window.
    let p3, g3 = scientific::stft_plan([1.0, 2.0, 3.0], 2, 10.0, "onesided", 2);
    assert(g3.kind == scientific::F_RANGE);
    // The plan refuses a step above the length of the window.
    let p4, g4 = scientific::stft_plan([1.0, 2.0, 3.0], 4, 10.0);
    assert(g4.kind == scientific::F_VALUE);
    // The transform refuses a bad axis and a short signal.
    let p5, g5 = scientific::stft_plan([1.0, 2.0, 3.0, 4.0], 2, 10.0);
    assert(g5.kind == scientific::F_NONE);
    let z1, h1 = scientific::stft_forward(p5, common::av([1.0]), -1);
    assert(h1.kind == scientific::F_SHAPE);
    var sh: i64[..8] = [];
    sh.push(2);
    sh.push(4);
    let x2d = scientific::Arr { ndim: 2, shape: sh,
                                data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] };
    let z2, h2 = scientific::stft_forward(p5, x2d, 0);
    assert(h2.kind == scientific::F_RANGE);
    // The inverse refuses a wrong count of frequencies.
    let z3, h3 = scientific::stft_forward(p5, common::av([1.0, 2.0, 3.0, 4.0]), -1);
    assert(h3.kind == scientific::F_NONE);
    let wrong = scientific::CArr { ndim: 2, shape: [2, 2],
                                   data: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0] };
    let z4, h4 = scientific::stft_inverse(p5, wrong, -1);
    assert(h4.kind == scientific::F_SHAPE);
    let z5, h5 = scientific::stft_inverse(p5, z3, 0);
    assert(h5.kind == scientific::F_RANGE);
}
