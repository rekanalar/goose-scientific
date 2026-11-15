// The tests of the group signal_spectrum. Run them with SCIENTIFIC_SUITES=signal_spectrum.
//
// The chirp z-transform, the cross spectra, the envelope, the two
// dimensional filters, the windows, the peaks and the waves match the
// pins of SciPy 1.18.1.
namespace signal_spectrum_tests;

import scientific from "../..";
import common;
import signal_spectrum_pins;

const NOP: f64 = 0.0 / 0.0;

// loose_eq tells whether an array holds the values within a relative 1e-5.
fn loose_eq(a: scientific::Arr&, v: f64[:]) -> bool {
    return common::all_loose(a, v);
}

// shape_is tells whether a matrix holds the shape in a pin of two values.
fn shape_is(a: scientific::Arr&, s: f64[:]) -> bool {
    return a.ndim == 2 && a.shape[0] == s[0] as i64 && a.shape[1] == s[1] as i64;
}

// ints_of gives the whole numbers of a pin.
fn ints_of(v: f64[:]) -> i64[>..] {
    var r: i64[>..] = [];
    var i: i64 = 0;
    while i < v.len { r.push(v[i] as i64); i += 1; }
    return r;
}

fn test_sigspec_czt() {
    let x = signal_spectrum_pins::CZ_X;
    let m1 = signal_spectrum_pins::CZ1_M[0] as i64;
    let r1, i1, f1 = scientific::czt(x, m1, signal_spectrum_pins::CZ1_WR[0], signal_spectrum_pins::CZ1_WI[0], signal_spectrum_pins::CZ1_AR[0], signal_spectrum_pins::CZ1_AI[0]);
    assert(f1.kind == scientific::F_NONE);
    assert(loose_eq(r1, signal_spectrum_pins::CZ1_RE) && loose_eq(i1, signal_spectrum_pins::CZ1_IM));
    let m2 = signal_spectrum_pins::CZ2_M[0] as i64;
    let r2, i2, f2 = scientific::czt(x, m2, signal_spectrum_pins::CZ2_WR[0], signal_spectrum_pins::CZ2_WI[0], signal_spectrum_pins::CZ2_AR[0], signal_spectrum_pins::CZ2_AI[0]);
    assert(f2.kind == scientific::F_NONE);
    assert(loose_eq(r2, signal_spectrum_pins::CZ2_RE) && loose_eq(i2, signal_spectrum_pins::CZ2_IM));
    let m3 = signal_spectrum_pins::CZ3_M[0] as i64;
    let r3, i3, f3 = scientific::czt(x, m3, signal_spectrum_pins::CZ3_WR[0], signal_spectrum_pins::CZ3_WI[0], signal_spectrum_pins::CZ3_AR[0], signal_spectrum_pins::CZ3_AI[0]);
    assert(f3.kind == scientific::F_NONE);
    assert(loose_eq(r3, signal_spectrum_pins::CZ3_RE) && loose_eq(i3, signal_spectrum_pins::CZ3_IM));
}

fn test_sigspec_czt_default_is_fft() {
    let x = signal_spectrum_pins::CZ_X;
    let cr, ci, f = scientific::czt(x);
    assert(f.kind == scientific::F_NONE);
    let fr, fi, ff = scientific::fft(x);
    assert(ff.kind == scientific::F_NONE);
    assert(loose_eq(cr, fr.data) && loose_eq(ci, fi.data));
}

fn test_sigspec_czt_points() {
    let pr, pi, f = scientific::czt_points(signal_spectrum_pins::CZ1_M[0] as i64, signal_spectrum_pins::CZ1_WR[0], signal_spectrum_pins::CZ1_WI[0], signal_spectrum_pins::CZ1_AR[0], signal_spectrum_pins::CZ1_AI[0]);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(pr, signal_spectrum_pins::CZ1_PRE, 0.000000001) && common::all_close(pi, signal_spectrum_pins::CZ1_PIM, 0.000000001));
    let qr, qi, f2 = scientific::czt_points(signal_spectrum_pins::CZ3_M[0] as i64, signal_spectrum_pins::CZ3_WR[0], signal_spectrum_pins::CZ3_WI[0], signal_spectrum_pins::CZ3_AR[0], signal_spectrum_pins::CZ3_AI[0]);
    assert(f2.kind == scientific::F_NONE);
    assert(common::all_close(qr, signal_spectrum_pins::CZ3_PRE, 0.000000001) && common::all_close(qi, signal_spectrum_pins::CZ3_PIM, 0.000000001));
    let dr, di, f3 = scientific::czt_points(signal_spectrum_pins::CZDP_M[0] as i64);
    assert(f3.kind == scientific::F_NONE);
    assert(common::all_close(dr, signal_spectrum_pins::CZDP_RE, 0.000000001) && common::all_close(di, signal_spectrum_pins::CZDP_IM, 0.000000001));
    // A point count below one and a zero step are faults.
    let e1, e2, ef = scientific::czt_points(0);
    assert(ef.kind == scientific::F_VALUE);
    let z1, z2, zf = scientific::czt_points(4, 0.0, 0.0, 1.0, 0.0);
    assert(zf.kind == scientific::F_VALUE);
}

fn test_sigspec_zoom_fft() {
    let x = signal_spectrum_pins::CZ_X;
    let a1, b1, f1 = scientific::zoom_fft(x, [signal_spectrum_pins::ZF1_F1[0], signal_spectrum_pins::ZF1_F2[0]], signal_spectrum_pins::ZF1_M[0] as i64, signal_spectrum_pins::ZF1_FS[0], false);
    assert(f1.kind == scientific::F_NONE);
    assert(loose_eq(a1, signal_spectrum_pins::ZF1_RE) && loose_eq(b1, signal_spectrum_pins::ZF1_IM));
    let a2, b2, f2 = scientific::zoom_fft(x, [signal_spectrum_pins::ZF2_F1[0], signal_spectrum_pins::ZF2_F2[0]], signal_spectrum_pins::ZF2_M[0] as i64, signal_spectrum_pins::ZF2_FS[0], true);
    assert(f2.kind == scientific::F_NONE);
    assert(loose_eq(a2, signal_spectrum_pins::ZF2_RE) && loose_eq(b2, signal_spectrum_pins::ZF2_IM));
    let a3, b3, f3 = scientific::zoom_fft(x, [signal_spectrum_pins::ZF3_F2[0]], signal_spectrum_pins::ZF3_M[0] as i64, signal_spectrum_pins::ZF3_FS[0], false);
    assert(f3.kind == scientific::F_NONE);
    assert(loose_eq(a3, signal_spectrum_pins::ZF3_RE) && loose_eq(b3, signal_spectrum_pins::ZF3_IM));
    let e1, e2, ef = scientific::zoom_fft(x, [0.1, 0.2, 0.3]);
    assert(ef.kind == scientific::F_SHAPE);
}

// csd_case checks one cross spectrum, with the window by name and by values.
fn csd_case(name: const u8[:], nperseg: i64, nov: i64, fs: f64, fr: f64[:], re: f64[:], im: f64[:], win: f64[:]) {
    let x = signal_spectrum_pins::CS_X;
    let y = signal_spectrum_pins::CS_Y;
    let gf, gr, gi, f = scientific::csd_ex(x, y, fs, name, NOP, [], nperseg, nov);
    assert(f.kind == scientific::F_NONE);
    assert(loose_eq(gf, fr) && loose_eq(gr, re) && loose_eq(gi, im));
    let hf, hr, hi, f2 = scientific::csd_ex(x, y, fs, "hann", NOP, win, nperseg, nov);
    assert(f2.kind == scientific::F_NONE);
    assert(loose_eq(hr, re) && loose_eq(hi, im));
}

fn test_sigspec_csd() {
    csd_case("hann", signal_spectrum_pins::CS0_NP[0] as i64, signal_spectrum_pins::CS0_NV[0] as i64, signal_spectrum_pins::CS0_FS[0], signal_spectrum_pins::CS0_FR, signal_spectrum_pins::CS0_RE, signal_spectrum_pins::CS0_IM, signal_spectrum_pins::CS0_WIN);
    csd_case("boxcar", signal_spectrum_pins::CS1_NP[0] as i64, signal_spectrum_pins::CS1_NV[0] as i64, signal_spectrum_pins::CS1_FS[0], signal_spectrum_pins::CS1_FR, signal_spectrum_pins::CS1_RE, signal_spectrum_pins::CS1_IM, signal_spectrum_pins::CS1_WIN);
    csd_case("hamming", signal_spectrum_pins::CS2_NP[0] as i64, signal_spectrum_pins::CS2_NV[0] as i64, signal_spectrum_pins::CS2_FS[0], signal_spectrum_pins::CS2_FR, signal_spectrum_pins::CS2_RE, signal_spectrum_pins::CS2_IM, signal_spectrum_pins::CS2_WIN);
}

fn test_sigspec_csd_of_one_signal_is_welch() {
    let x = signal_spectrum_pins::CS_X;
    let gf, gr, gi, f = scientific::csd_ex(x, x, 2.0, "hann", NOP, [], 16, 8);
    assert(f.kind == scientific::F_NONE);
    assert(loose_eq(gr, signal_spectrum_pins::CS0_XX));
    // The imaginary part of the cross spectrum of a signal with itself is zero.
    var i: i64 = 0;
    while i < gi.data.len { assert(gi.data[i] > -0.000000001 && gi.data[i] < 0.000000001); i += 1; }
    let wf, wp, wfa = scientific::welch_ex(x, 2.0, "hann", NOP, [], 16, 8);
    assert(wfa.kind == scientific::F_NONE && loose_eq(wp, gr.data));
}

fn test_sigspec_csd_two_sided_and_median() {
    let x = signal_spectrum_pins::CS_X;
    let y = signal_spectrum_pins::CS_Y;
    let fo, ro, io, f = scientific::csd_ex(x, y, 1.0, "hann", NOP, [], 16, 8);
    assert(f.kind == scientific::F_NONE && ro.data.len == 9);
    let ft, rt, it, f2 = scientific::csd_ex(x, y, 1.0, "hann", NOP, [], 16, 8, 0, true, false);
    assert(f2.kind == scientific::F_NONE && rt.data.len == 16);
    // The two sides of a real pair hold the mirror of the conjugate.
    assert(common::loose(rt.data[1], rt.data[15]) && common::loose(it.data[1], -it.data[15]));
    let fm, rm, im2, f3 = scientific::csd_ex(x, y, 1.0, "hann", NOP, [], 16, 8, 0, true, true, "density", "median");
    assert(f3.kind == scientific::F_NONE && rm.data.len == 9);
}

fn test_sigspec_csd_faults() {
    let x = signal_spectrum_pins::CS_X;
    let a, b, c, f = scientific::csd_ex(x, [1.0, 2.0], 1.0, "hann", NOP, [], 16, 8);
    assert(f.kind == scientific::F_SHAPE);
    let a2, b2, c2, f2 = scientific::csd_ex(x, x, 1.0, "hann", NOP, [], 16, 16);
    assert(f2.kind == scientific::F_RANGE);
    let a3, b3, c3, f3 = scientific::csd_ex(x, x, 0.0, "hann", NOP, [], 16, 8);
    assert(f3.kind == scientific::F_VALUE);
    let a4, b4, c4, f4 = scientific::csd_ex(x, x, 1.0, "hann", NOP, [], 16, 8, 0, true, true, "power");
    assert(f4.kind == scientific::F_VALUE);
    let a5, b5, c5, f5 = scientific::csd_ex(x, x, 1.0, "hann", NOP, [], 1000, 8);
    assert(f5.kind == scientific::F_RANGE);
}

fn test_sigspec_coherence() {
    let x = signal_spectrum_pins::CS_X;
    let y = signal_spectrum_pins::CS_Y;
    // The place of no frequency holds a ratio of the noise of the
    // transform. The pin holds the other places.
    let f0, c0, e0 = scientific::coherence(x, y, 2.0, "hann", NOP, [], 16, 8);
    assert(e0.kind == scientific::F_NONE);
    assert(loose_eq(f0, signal_spectrum_pins::CS0_FR));
    var i: i64 = 1;
    while i < c0.data.len { assert(common::loose(c0.data[i], signal_spectrum_pins::CS0_COH[i - 1])); i += 1; }
    let f1, c1, e1 = scientific::coherence(x, y, 2.0, "boxcar", NOP, [], 16, 8);
    assert(e1.kind == scientific::F_NONE);
    i = 1;
    while i < c1.data.len { assert(common::loose(c1.data[i], signal_spectrum_pins::CS1_COH[i - 1])); i += 1; }
    let f2, c2, e2 = scientific::coherence(x, y, 1.0, "hamming", NOP, [], 8, 4);
    assert(e2.kind == scientific::F_NONE);
    i = 1;
    while i < c2.data.len { assert(common::loose(c2.data[i], signal_spectrum_pins::CS2_COH[i - 1])); i += 1; }
    // Every value lies between zero and one.
    i = 0;
    while i < c0.data.len { assert(c0.data[i] >= 0.0 && c0.data[i] <= 1.0000000001); i += 1; }
}

// lags_case checks one call of correlation_lags.
fn lags_case(m: f64[:], n: f64[:], mode: const u8[:], want: f64[:]) {
    let lag, f = scientific::correlation_lags(m[0] as i64, n[0] as i64, mode);
    assert(f.kind == scientific::F_NONE);
    assert(common::all_close(lag, want, 0.0));
}

fn test_sigspec_correlation_lags() {
    lags_case(signal_spectrum_pins::LG1_M, signal_spectrum_pins::LG1_N, "full", signal_spectrum_pins::LG1_LAG);
    lags_case(signal_spectrum_pins::LG2_M, signal_spectrum_pins::LG2_N, "full", signal_spectrum_pins::LG2_LAG);
    lags_case(signal_spectrum_pins::LG3_M, signal_spectrum_pins::LG3_N, "same", signal_spectrum_pins::LG3_LAG);
    lags_case(signal_spectrum_pins::LG4_M, signal_spectrum_pins::LG4_N, "valid", signal_spectrum_pins::LG4_LAG);
    lags_case(signal_spectrum_pins::LG5_M, signal_spectrum_pins::LG5_N, "same", signal_spectrum_pins::LG5_LAG);
    let e, f = scientific::correlation_lags(5, 5, "wide");
    assert(f.kind == scientific::F_VALUE);
    let e2, f2 = scientific::correlation_lags(0, 5);
    assert(f2.kind == scientific::F_VALUE);
}

fn test_sigspec_envelope() {
    let ea, ra, f1 = scientific::envelope(signal_spectrum_pins::EV1_Z);
    assert(f1.kind == scientific::F_NONE);
    assert(loose_eq(ea, signal_spectrum_pins::EV1_ENV) && loose_eq(ra, signal_spectrum_pins::EV1_RES));
    let eb, rb, f2 = scientific::envelope(signal_spectrum_pins::EV2_Z, 1, 2147483647, 0, false, "all");
    assert(f2.kind == scientific::F_NONE);
    assert(loose_eq(eb, signal_spectrum_pins::EV2_ENV) && loose_eq(rb, signal_spectrum_pins::EV2_RES));
    let ec, rc, f3 = scientific::envelope(signal_spectrum_pins::EV3_Z, 1, 2147483647, 0, true);
    assert(f3.kind == scientific::F_NONE);
    assert(loose_eq(ec, signal_spectrum_pins::EV3_ENV) && loose_eq(rc, signal_spectrum_pins::EV3_RES));
    let ed, rd, f4 = scientific::envelope(signal_spectrum_pins::EV4_Z);
    assert(f4.kind == scientific::F_NONE);
    assert(loose_eq(ed, signal_spectrum_pins::EV4_ENV) && loose_eq(rd, signal_spectrum_pins::EV4_RES));
    // The residual "none" gives no residual.
    let ee, re2, f5 = scientific::envelope(signal_spectrum_pins::EV1_Z, 1, 2147483647, 0, false, "none");
    assert(f5.kind == scientific::F_NONE && re2.data.len == 0 && loose_eq(ee, signal_spectrum_pins::EV1_ENV));
    let eg, rg, f6 = scientific::envelope(signal_spectrum_pins::EV1_Z, 1, 2147483647, 0, false, "wide");
    assert(f6.kind == scientific::F_VALUE);
    let eh, rh, f7 = scientific::envelope([1.0, 2.0, 1.0 / 0.0]);
    assert(f7.kind == scientific::F_VALUE);
}

fn test_sigspec_convolve2d_correlate2d() {
    let a = common::mat(signal_spectrum_pins::T2_A, 7, 8);
    let b = common::mat(signal_spectrum_pins::T2_B, 3, 4);
    let cf, f1 = scientific::convolve2d(a, b, "full");
    assert(f1.kind == scientific::F_NONE && shape_is(cf, signal_spectrum_pins::T2_CV_SH) && loose_eq(cf, signal_spectrum_pins::T2_CV));
    let cs, f2 = scientific::convolve2d(a, b, "same");
    assert(f2.kind == scientific::F_NONE && shape_is(cs, signal_spectrum_pins::T2_CS_SH) && loose_eq(cs, signal_spectrum_pins::T2_CS));
    let cw, f3 = scientific::convolve2d(a, b, "valid");
    assert(f3.kind == scientific::F_NONE && shape_is(cw, signal_spectrum_pins::T2_CW_SH) && loose_eq(cw, signal_spectrum_pins::T2_CW));
    let rf, f4 = scientific::correlate2d(a, b, "full");
    assert(f4.kind == scientific::F_NONE && shape_is(rf, signal_spectrum_pins::T2_RF_SH) && loose_eq(rf, signal_spectrum_pins::T2_RF));
    let rs, f5 = scientific::correlate2d(a, b, "same");
    assert(f5.kind == scientific::F_NONE && shape_is(rs, signal_spectrum_pins::T2_RS_SH) && loose_eq(rs, signal_spectrum_pins::T2_RS));
    let rw, f6 = scientific::correlate2d(a, b, "valid");
    assert(f6.kind == scientific::F_NONE && shape_is(rw, signal_spectrum_pins::T2_RW_SH) && loose_eq(rw, signal_spectrum_pins::T2_RW));
    let bad, f7 = scientific::convolve2d(a, b, "wide");
    assert(f7.kind == scientific::F_VALUE);
    let v = common::av([1.0, 2.0]);
    let bad2, f8 = scientific::convolve2d(v, b);
    assert(f8.kind == scientific::F_SHAPE);
}

fn test_sigspec_rank_filters_2d() {
    let a = common::mat(signal_spectrum_pins::T2_A, 7, 8);
    let m, f1 = scientific::medfilt2d(a, [3]);
    assert(f1.kind == scientific::F_NONE && loose_eq(m, signal_spectrum_pins::T2_MED));
    var dom: f64[>..] = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let o, f2 = scientific::order_filter(a, common::mat(dom, 3, 3), 1);
    assert(f2.kind == scientific::F_NONE && loose_eq(o, signal_spectrum_pins::T2_ORD));
    let bad, f3 = scientific::order_filter(a, common::mat([1.0, 1.0, 1.0, 1.0], 2, 2), 1);
    assert(f3.kind == scientific::F_VALUE);
    let s, f4 = scientific::sepfir2d(a, signal_spectrum_pins::T2_HROW, signal_spectrum_pins::T2_HCOL);
    assert(f4.kind == scientific::F_NONE && loose_eq(s, signal_spectrum_pins::T2_SEP));
}

fn test_sigspec_firwin_2d() {
    let a, f1 = scientific::firwin_2d(5, 7, 0.2, "kaiser", 5.0, false);
    assert(f1.kind == scientific::F_NONE && shape_is(a, signal_spectrum_pins::T2_FIR_SH) && loose_eq(a, signal_spectrum_pins::T2_FIR));
    let c, f2 = scientific::firwin_2d(6, 8, 0.2, "kaiser", 5.0, true);
    assert(f2.kind == scientific::F_NONE && shape_is(c, signal_spectrum_pins::T2_CIR_SH) && loose_eq(c, signal_spectrum_pins::T2_CIR));
    let bad, f3 = scientific::firwin_2d(5, 7, 1.5);
    assert(f3.kind == scientific::F_VALUE);
    let bad2, f4 = scientific::firwin_2d(0, 7, 0.2);
    assert(f4.kind == scientific::F_VALUE);
}

fn test_sigspec_gauss_spline() {
    let x = signal_spectrum_pins::T2_SPL_X;
    let a, f1 = scientific::gauss_spline(x, signal_spectrum_pins::T2_GS_N1_N[0] as i64);
    assert(f1.kind == scientific::F_NONE && loose_eq(a, signal_spectrum_pins::T2_GS_N1));
    let b, f2 = scientific::gauss_spline(x, signal_spectrum_pins::T2_GS_N2_N[0] as i64);
    assert(f2.kind == scientific::F_NONE && loose_eq(b, signal_spectrum_pins::T2_GS_N2));
    let c, f3 = scientific::gauss_spline(x, signal_spectrum_pins::T2_GS_N3_N[0] as i64);
    assert(f3.kind == scientific::F_NONE && loose_eq(c, signal_spectrum_pins::T2_GS_N3));
    let bad, f4 = scientific::gauss_spline(x, 0);
    assert(f4.kind == scientific::F_VALUE);
}

fn test_sigspec_hilbert2_and_wiener() {
    let a = common::mat(signal_spectrum_pins::T2_H3, 6, 5);
    let hr, hi, f1 = scientific::hilbert2(a);
    assert(f1.kind == scientific::F_NONE);
    assert(loose_eq(hr, signal_spectrum_pins::T2_H3_RE) && loose_eq(hi, signal_spectrum_pins::T2_H3_IM));
    let w = common::mat(signal_spectrum_pins::T2_WN, 8, 7);
    let o1, f2 = scientific::wiener(w, [3, 5]);
    assert(f2.kind == scientific::F_NONE && shape_is(o1, signal_spectrum_pins::T2_WN_SH) && loose_eq(o1, signal_spectrum_pins::T2_WN_OUT));
    let o2, f3 = scientific::wiener(w, [3, 5], 0.05);
    assert(f3.kind == scientific::F_NONE && loose_eq(o2, signal_spectrum_pins::T2_WN_OUT2));
    let bad1, bad2, f4 = scientific::hilbert2(common::av([1.0, 2.0]));
    assert(f4.kind == scientific::F_SHAPE);
}

// cola_case checks one window for the two overlap conditions.
fn cola_case(name: const u8[:], win: f64[:], np: f64[:], nv: f64[:], cola: f64[:], nola: f64[:]) {
    let a, fa = scientific::check_cola(name, NOP, [], np[0] as i64, nv[0] as i64);
    assert(fa.kind == scientific::F_NONE && a == (cola[0] == 1.0));
    let b, fb = scientific::check_nola(name, NOP, [], np[0] as i64, nv[0] as i64);
    assert(fb.kind == scientific::F_NONE && b == (nola[0] == 1.0));
    let c, fc = scientific::check_cola("hann", NOP, win, np[0] as i64, nv[0] as i64);
    assert(fc.kind == scientific::F_NONE && c == (cola[0] == 1.0));
}

fn test_sigspec_check_cola_nola() {
    cola_case("hann", signal_spectrum_pins::WD_CO1, signal_spectrum_pins::WD_CO1_NP, signal_spectrum_pins::WD_CO1_NV, signal_spectrum_pins::WD_CO1_COLA, signal_spectrum_pins::WD_CO1_NOLA);
    cola_case("boxcar", signal_spectrum_pins::WD_CO2, signal_spectrum_pins::WD_CO2_NP, signal_spectrum_pins::WD_CO2_NV, signal_spectrum_pins::WD_CO2_COLA, signal_spectrum_pins::WD_CO2_NOLA);
    cola_case("hamming", signal_spectrum_pins::WD_CO3, signal_spectrum_pins::WD_CO3_NP, signal_spectrum_pins::WD_CO3_NV, signal_spectrum_pins::WD_CO3_COLA, signal_spectrum_pins::WD_CO3_NOLA);
    cola_case("bartlett", signal_spectrum_pins::WD_CO4, signal_spectrum_pins::WD_CO4_NP, signal_spectrum_pins::WD_CO4_NV, signal_spectrum_pins::WD_CO4_COLA, signal_spectrum_pins::WD_CO4_NOLA);
    let bad, f = scientific::check_cola("hann", NOP, [], 8, 8);
    assert(f.kind == scientific::F_VALUE);
    let bad2, f2 = scientific::check_nola("hann", NOP, [1.0, 2.0], 8, 4);
    assert(f2.kind == scientific::F_SHAPE);
}

// dual_case checks one dual window.
fn dual_case(win: f64[:], hop: f64[:], want: f64[:], alpha: f64[:]) {
    let d, a, f = scientific::closest_stft_dual_window(win, hop[0] as i64);
    assert(f.kind == scientific::F_NONE);
    assert(loose_eq(d, want) && common::loose(a, alpha[0]));
}

fn test_sigspec_closest_stft_dual_window() {
    dual_case(signal_spectrum_pins::DW_DU1, signal_spectrum_pins::DW_DU1_HOP, signal_spectrum_pins::DW_DU1_D, signal_spectrum_pins::DW_DU1_A);
    dual_case(signal_spectrum_pins::DW_DU2, signal_spectrum_pins::DW_DU2_HOP, signal_spectrum_pins::DW_DU2_D, signal_spectrum_pins::DW_DU2_A);
    dual_case(signal_spectrum_pins::DW_DU3, signal_spectrum_pins::DW_DU3_HOP, signal_spectrum_pins::DW_DU3_D, signal_spectrum_pins::DW_DU3_A);
    let e, a, f = scientific::closest_stft_dual_window(signal_spectrum_pins::DW_DU1, 0);
    assert(f.kind == scientific::F_VALUE);
}

// conv_case checks one choice of the way of convolving.
fn conv_case(sa: f64[:], sb: f64[:], mode: f64[:], r: f64[:]) {
    var m: const u8[:] = "full";
    if mode[0] == 2.0 { m = "same"; }
    if mode[0] == 3.0 { m = "valid"; }
    let got, f = scientific::choose_conv_method(ints_of(sa), ints_of(sb), m);
    assert(f.kind == scientific::F_NONE);
    if r[0] == 1.0 { assert(got == "fft"); } else { assert(got == "direct"); }
}

fn test_sigspec_choose_conv_method() {
    conv_case(signal_spectrum_pins::CV_CM1_A, signal_spectrum_pins::CV_CM1_B, signal_spectrum_pins::CV_CM1_M, signal_spectrum_pins::CV_CM1_R);
    conv_case(signal_spectrum_pins::CV_CM2_A, signal_spectrum_pins::CV_CM2_B, signal_spectrum_pins::CV_CM2_M, signal_spectrum_pins::CV_CM2_R);
    conv_case(signal_spectrum_pins::CV_CM3_A, signal_spectrum_pins::CV_CM3_B, signal_spectrum_pins::CV_CM3_M, signal_spectrum_pins::CV_CM3_R);
    conv_case(signal_spectrum_pins::CV_CM4_A, signal_spectrum_pins::CV_CM4_B, signal_spectrum_pins::CV_CM4_M, signal_spectrum_pins::CV_CM4_R);
    conv_case(signal_spectrum_pins::CV_CM5_A, signal_spectrum_pins::CV_CM5_B, signal_spectrum_pins::CV_CM5_M, signal_spectrum_pins::CV_CM5_R);
    let bad, f = scientific::choose_conv_method([3, 3], [3], "full");
    assert(f.kind == scientific::F_SHAPE);
    let bad2, f2 = scientific::choose_conv_method([3], [3], "wide");
    assert(f2.kind == scientific::F_VALUE);
}

fn test_sigspec_deconvolve() {
    let q1, r1, f1 = scientific::deconvolve(signal_spectrum_pins::DC_DV1_N, signal_spectrum_pins::DC_DV1_D);
    assert(f1.kind == scientific::F_NONE && loose_eq(q1, signal_spectrum_pins::DC_DV1_Q) && common::all_close(r1, signal_spectrum_pins::DC_DV1_R, 0.000000001));
    let q2, r2, f2 = scientific::deconvolve(signal_spectrum_pins::DC_DV2_N, signal_spectrum_pins::DC_DV2_D);
    assert(f2.kind == scientific::F_NONE && loose_eq(q2, signal_spectrum_pins::DC_DV2_Q) && common::all_close(r2, signal_spectrum_pins::DC_DV2_R, 0.000000001));
    let q3, r3, f3 = scientific::deconvolve(signal_spectrum_pins::DC_DV3_N, signal_spectrum_pins::DC_DV3_D);
    assert(f3.kind == scientific::F_NONE && loose_eq(q3, signal_spectrum_pins::DC_DV3_Q) && common::all_close(r3, signal_spectrum_pins::DC_DV3_R, 0.000000001));
    let e1, e2, f4 = scientific::deconvolve([], [1.0]);
    assert(f4.kind == scientific::F_VALUE);
}

fn test_sigspec_lombscargle() {
    let t = signal_spectrum_pins::LS_T;
    let y = signal_spectrum_pins::LS_Y;
    let f = signal_spectrum_pins::LS_F;
    let p, e1 = scientific::lombscargle(t, y, f);
    assert(e1.kind == scientific::F_NONE && loose_eq(p, signal_spectrum_pins::LS_P));
    let pn, e2 = scientific::lombscargle(t, y, f, "normalize");
    assert(e2.kind == scientific::F_NONE && loose_eq(pn, signal_spectrum_pins::LS_PN));
    let pf, e3 = scientific::lombscargle(t, y, f, "power", true);
    assert(e3.kind == scientific::F_NONE && loose_eq(pf, signal_spectrum_pins::LS_PF));
    let bad, e4 = scientific::lombscargle(t, [1.0, 2.0], f);
    assert(e4.kind == scientific::F_SHAPE);
    let bad2, e5 = scientific::lombscargle(t, y, f, "wide");
    assert(e5.kind == scientific::F_VALUE);
}

fn test_sigspec_minimum_phase() {
    let a, f1 = scientific::minimum_phase(signal_spectrum_pins::MP_MP1_H);
    assert(f1.kind == scientific::F_NONE && loose_eq(a, signal_spectrum_pins::MP_MP1));
    let b, f2 = scientific::minimum_phase(signal_spectrum_pins::MP_MP2_H);
    assert(f2.kind == scientific::F_NONE && loose_eq(b, signal_spectrum_pins::MP_MP2));
    let c, f3 = scientific::minimum_phase(signal_spectrum_pins::MP_MP3_H);
    assert(f3.kind == scientific::F_NONE && loose_eq(c, signal_spectrum_pins::MP_MP3));
    let bad, f4 = scientific::minimum_phase([1.0, 2.0]);
    assert(f4.kind == scientific::F_VALUE);
}

fn test_sigspec_whittaker_henderson() {
    let y = signal_spectrum_pins::WH_Y;
    let a, la, f1 = scientific::whittaker_henderson(y, signal_spectrum_pins::WH_WH1_L[0], signal_spectrum_pins::WH_WH1_O[0] as i64);
    assert(f1.kind == scientific::F_NONE && loose_eq(a, signal_spectrum_pins::WH_WH1) && la == signal_spectrum_pins::WH_WH1_L[0]);
    let b, lb, f2 = scientific::whittaker_henderson(y, signal_spectrum_pins::WH_WH2_L[0], signal_spectrum_pins::WH_WH2_O[0] as i64);
    assert(f2.kind == scientific::F_NONE && loose_eq(b, signal_spectrum_pins::WH_WH2));
    let c, lc, f3 = scientific::whittaker_henderson(y, signal_spectrum_pins::WH_WH3_L[0], signal_spectrum_pins::WH_WH3_O[0] as i64);
    assert(f3.kind == scientific::F_NONE && loose_eq(c, signal_spectrum_pins::WH_WH3));
    let d, ld, f4 = scientific::whittaker_henderson(y, signal_spectrum_pins::WH_WH4_L[0], signal_spectrum_pins::WH_WH4_O[0] as i64);
    assert(f4.kind == scientific::F_NONE && loose_eq(d, signal_spectrum_pins::WH_WH4));
    let w, lw, f5 = scientific::whittaker_henderson(y, 100.0, 2, signal_spectrum_pins::WH_W);
    assert(f5.kind == scientific::F_NONE && loose_eq(w, signal_spectrum_pins::WH_W_OUT));
    let bad, lb2, f6 = scientific::whittaker_henderson(y, -1.0);
    assert(f6.kind == scientific::F_VALUE);
    let bad2, lb3, f7 = scientific::whittaker_henderson([1.0, 2.0], 1.0, 2);
    assert(f7.kind == scientific::F_SHAPE);
}

fn test_sigspec_peak_widths() {
    let x = signal_spectrum_pins::PK_SIG;
    let pk = signal_spectrum_pins::PK_PEAKS;
    let w1, h1, l1, r1, f1 = scientific::peak_widths(x, pk, signal_spectrum_pins::PW_W1_REL[0]);
    assert(f1.kind == scientific::F_NONE);
    assert(loose_eq(w1, signal_spectrum_pins::PW_W1_W) && loose_eq(h1, signal_spectrum_pins::PW_W1_H) && loose_eq(l1, signal_spectrum_pins::PW_W1_L) && loose_eq(r1, signal_spectrum_pins::PW_W1_R));
    let w2, h2, l2, r2, f2 = scientific::peak_widths(x, pk, signal_spectrum_pins::PW_W2_REL[0]);
    assert(f2.kind == scientific::F_NONE);
    assert(loose_eq(w2, signal_spectrum_pins::PW_W2_W) && loose_eq(h2, signal_spectrum_pins::PW_W2_H) && loose_eq(l2, signal_spectrum_pins::PW_W2_L) && loose_eq(r2, signal_spectrum_pins::PW_W2_R));
    let w3, h3, l3, r3, f3 = scientific::peak_widths(x, pk, signal_spectrum_pins::PW_W3_REL[0]);
    assert(f3.kind == scientific::F_NONE);
    assert(loose_eq(w3, signal_spectrum_pins::PW_W3_W) && loose_eq(h3, signal_spectrum_pins::PW_W3_H) && loose_eq(l3, signal_spectrum_pins::PW_W3_L) && loose_eq(r3, signal_spectrum_pins::PW_W3_R));
    let a, b, c, d, f4 = scientific::peak_widths(x, [99.0]);
    assert(f4.kind == scientific::F_RANGE);
    let a2, b2, c2, d2, f5 = scientific::peak_widths(x, pk, -0.5);
    assert(f5.kind == scientific::F_VALUE);
}

fn test_sigspec_argrelextrema() {
    let x = signal_spectrum_pins::AR_SIG;
    let mx, f1 = scientific::argrelmax(x);
    assert(f1.kind == scientific::F_NONE && common::all_close(mx, signal_spectrum_pins::AR_MAX, 0.0));
    let mn, f2 = scientific::argrelmin(x);
    assert(f2.kind == scientific::F_NONE && common::all_close(mn, signal_spectrum_pins::AR_MIN, 0.0));
    let m2, f3 = scientific::argrelmax(x, 2);
    assert(f3.kind == scientific::F_NONE && common::all_close(m2, signal_spectrum_pins::AR_MAX2, 0.0));
    let ex, f4 = scientific::argrelextrema(x, 1, false);
    assert(f4.kind == scientific::F_NONE && common::all_close(ex, signal_spectrum_pins::AR_MIN, 0.0));
    let bad, f5 = scientific::argrelmax(x, 0);
    assert(f5.kind == scientific::F_VALUE);
}

fn test_sigspec_ricker_cwt_find_peaks_cwt() {
    let r, f1 = scientific::ricker(signal_spectrum_pins::RK_P[0] as i64, signal_spectrum_pins::RK_W[0]);
    assert(f1.kind == scientific::F_NONE && loose_eq(r, signal_spectrum_pins::RK_VAL));
    let c, f2 = scientific::cwt(signal_spectrum_pins::CW_SIG, signal_spectrum_pins::CW_W);
    assert(f2.kind == scientific::F_NONE && shape_is(c, signal_spectrum_pins::CW_SH) && loose_eq(c, signal_spectrum_pins::CW_MAT));
    let p, f3 = scientific::find_peaks_cwt(signal_spectrum_pins::FC_SIG, signal_spectrum_pins::FC_W);
    assert(f3.kind == scientific::F_NONE && common::all_close(p, signal_spectrum_pins::FC_PEAKS, 0.0));
    let bad, f4 = scientific::ricker(10, 0.0);
    assert(f4.kind == scientific::F_VALUE);
    let bad2, f5 = scientific::find_peaks_cwt([1.0, 2.0], []);
    assert(f5.kind == scientific::F_VALUE);
}

fn test_sigspec_chirp() {
    let t = signal_spectrum_pins::WV_T;
    let f0 = signal_spectrum_pins::WV_F0[0];
    let f1 = signal_spectrum_pins::WV_F1[0];
    let t1 = signal_spectrum_pins::WV_T1[0];
    let a, e1 = scientific::chirp(t, f0, t1, f1, "linear");
    assert(e1.kind == scientific::F_NONE && loose_eq(a, signal_spectrum_pins::WV_CH1));
    let b, e2 = scientific::chirp(t, f0, t1, f1, "quadratic");
    assert(e2.kind == scientific::F_NONE && loose_eq(b, signal_spectrum_pins::WV_CH2));
    let c, e3 = scientific::chirp(t, f0, t1, f1, "logarithmic");
    assert(e3.kind == scientific::F_NONE && loose_eq(c, signal_spectrum_pins::WV_CH3));
    let d, e4 = scientific::chirp(t, f0, t1, f1, "hyperbolic");
    assert(e4.kind == scientific::F_NONE && loose_eq(d, signal_spectrum_pins::WV_CH4));
    let bad, e5 = scientific::chirp(t, f0, t1, f1, "spiral");
    assert(e5.kind == scientific::F_VALUE);
    let bad2, e6 = scientific::chirp(t, 0.0, t1, f1, "logarithmic");
    assert(e6.kind == scientific::F_VALUE);
}

fn test_sigspec_gausspulse() {
    let t = signal_spectrum_pins::WV_T;
    let i, e1 = scientific::gausspulse(t, 1000.0, 0.5, -6.0, "i");
    assert(e1.kind == scientific::F_NONE && loose_eq(i, signal_spectrum_pins::WV_GP1));
    let q, e2 = scientific::gausspulse(t, 1000.0, 0.5, -6.0, "q");
    assert(e2.kind == scientific::F_NONE && loose_eq(q, signal_spectrum_pins::WV_GP2));
    let v, e3 = scientific::gausspulse(t, 1000.0, 0.5, -6.0, "env");
    assert(e3.kind == scientific::F_NONE && loose_eq(v, signal_spectrum_pins::WV_GP3));
    // The cutoff time is the time where the envelope falls to the level.
    let tc, e4 = scientific::gausspulse_cutoff(1000.0, 0.5, -6.0, -60.0);
    assert(e4.kind == scientific::F_NONE && tc > 0.0);
    let env, e5 = scientific::gausspulse([tc], 1000.0, 0.5, -6.0, "env");
    assert(e5.kind == scientific::F_NONE && common::loose(env.data[0], 0.001));
    let bad, e6 = scientific::gausspulse(t, 1000.0, 0.5, 6.0);
    assert(e6.kind == scientific::F_VALUE);
    let bad2, e7 = scientific::gausspulse(t, 1000.0, 0.0);
    assert(e7.kind == scientific::F_VALUE);
}

fn test_sigspec_sawtooth_sweep_poly() {
    let t = signal_spectrum_pins::WV_T;
    let a, e1 = scientific::sawtooth(t, 1.0);
    assert(e1.kind == scientific::F_NONE && loose_eq(a, signal_spectrum_pins::WV_SW1));
    let b, e2 = scientific::sawtooth(t, 0.5);
    assert(e2.kind == scientific::F_NONE && loose_eq(b, signal_spectrum_pins::WV_SW2));
    let n, e3 = scientific::sawtooth([1.0, 2.0], 1.5);
    assert(e3.kind == scientific::F_NONE && n.data[0] != n.data[0] && n.data[1] != n.data[1]);
    let s, e4 = scientific::sweep_poly(t, signal_spectrum_pins::WV_POLY);
    assert(e4.kind == scientific::F_NONE && loose_eq(s, signal_spectrum_pins::WV_SWP));
    let bad, e5 = scientific::sweep_poly(t, []);
    assert(e5.kind == scientific::F_VALUE);
}

fn test_sigspec_gammatone() {
    let b, a, e1 = scientific::gammatone(signal_spectrum_pins::WV_GT_F[0], "fir", 4, 15, 8000.0);
    assert(e1.kind == scientific::F_NONE);
    assert(loose_eq(b, signal_spectrum_pins::WV_GT_B) && loose_eq(a, signal_spectrum_pins::WV_GT_A));
    let ib, ia, e2 = scientific::gammatone(signal_spectrum_pins::WV_GT_F[0], "iir", 4, 0, 8000.0);
    assert(e2.kind == scientific::F_NONE);
    assert(loose_eq(ib, signal_spectrum_pins::WV_GT_IIR_B) && loose_eq(ia, signal_spectrum_pins::WV_GT_IIR_A));
    let b1, a1, e3 = scientific::gammatone(5000.0, "fir", 4, 15, 8000.0);
    assert(e3.kind == scientific::F_VALUE);
    let b2, a2, e4 = scientific::gammatone(200.0, "fir", 30, 15, 8000.0);
    assert(e4.kind == scientific::F_VALUE);
}

fn test_sigspec_vectorstrength() {
    let s, p, e = scientific::vectorstrength(signal_spectrum_pins::WV_VS_E, signal_spectrum_pins::WV_VS_P);
    assert(e.kind == scientific::F_NONE);
    assert(loose_eq(s, signal_spectrum_pins::WV_VS_S) && loose_eq(p, signal_spectrum_pins::WV_VS_PH));
    let bad1, bad2, e2 = scientific::vectorstrength(signal_spectrum_pins::WV_VS_E, [0.0]);
    assert(e2.kind == scientific::F_VALUE);
}

fn test_sigspec_max_len_seq() {
    let s1, t1, e1 = scientific::max_len_seq(signal_spectrum_pins::MX_ML1_N[0] as i64);
    assert(e1.kind == scientific::F_NONE);
    assert(common::all_close(s1, signal_spectrum_pins::MX_ML1_S, 0.0) && common::all_close(t1, signal_spectrum_pins::MX_ML1_ST, 0.0));
    let s2, t2, e2 = scientific::max_len_seq(signal_spectrum_pins::MX_ML2_N[0] as i64);
    assert(e2.kind == scientific::F_NONE);
    assert(common::all_close(s2, signal_spectrum_pins::MX_ML2_S, 0.0) && common::all_close(t2, signal_spectrum_pins::MX_ML2_ST, 0.0));
    let s3, t3, e3 = scientific::max_len_seq(signal_spectrum_pins::MX_ML3_N[0] as i64);
    assert(e3.kind == scientific::F_NONE);
    assert(common::all_close(s3, signal_spectrum_pins::MX_ML3_S, 0.0) && common::all_close(t3, signal_spectrum_pins::MX_ML3_ST, 0.0));
    let s4, t4, e4 = scientific::max_len_seq(6, 10, [3.0, 1.0]);
    assert(e4.kind == scientific::F_NONE);
    assert(common::all_close(s4, signal_spectrum_pins::MX_TAP_S, 0.0) && common::all_close(t4, signal_spectrum_pins::MX_TAP_ST, 0.0));
    let b1, b2, e5 = scientific::max_len_seq(1);
    assert(e5.kind == scientific::F_VALUE);
    let b3, b4, e6 = scientific::max_len_seq(6, 10, [9.0]);
    assert(e6.kind == scientific::F_VALUE);
}

// An array that a caller built by hand and broke does not agree with
// its shape. The calls that read the buffer of an array give a fault.
fn test_sigspec_broken_array() {
    // A matrix whose shape asks for more values than its buffer holds.
    let bad = scientific::Arr { ndim: 2, shape: [3, 3], data: [1.0] };
    let c, f1 = scientific::convolve2d(bad, bad, "full");
    assert(f1.kind == scientific::F_SHAPE);
    let r, f2 = scientific::correlate2d(bad, bad, "full");
    assert(f2.kind == scientific::F_SHAPE);
    // A matrix whose side of the axis holds no element. The check of
    // the shape comes first and gives a fault of a value.
    let zero = scientific::Arr { ndim: 2, shape: [0, 3], data: [1.0, 2.0, 3.0] };
    let s, f3 = scientific::sepfir2d(zero, [1.0], [1.0]);
    assert(f3.kind == scientific::F_VALUE);
    // A matrix whose shape asks for more values than its buffer holds.
    let s4, f3b = scientific::sepfir2d(bad, [1.0], [1.0]);
    assert(f3b.kind == scientific::F_SHAPE);
    // The sections of a filter whose buffer holds fewer values than
    // its shape asks for.
    let sos = scientific::Arr { ndim: 2, shape: [100, 6], data: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] };
    let w, h, g, f4 = scientific::freqz_sos(copy(sos), 6);
    assert(f4.kind == scientific::F_SHAPE);
}
