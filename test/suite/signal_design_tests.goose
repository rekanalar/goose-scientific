// The tests of the design and the transforms of a filter. Run them
// with SCIENTIFIC_SUITES=signal_design.
//
// The pins come from SciPy 1.18.1. The zeros and the poles come in a
// set, so the tests match them as a set.
namespace signal_design_tests;

import scientific from "../..";
import common;
import signal_design_pins;

// roots_match tells whether two lists of roots hold the same roots. The
// order of the roots is free, so each pinned root needs an unused
// partner.
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

// proto_of calls the prototype of the kind that the pin gives. The kind
// holds 0 for Butterworth, 1 for Chebyshev of type I, 2 for
// Chebyshev of type II, 3 for elliptic and 4 for Bessel.
fn proto_of(kind: i64, order: i64, rp: f64, rs: f64) -> scientific::Arr, scientific::Arr, scientific::Arr, scientific::Arr, f64, scientific::Fault {
    if kind == 0 { return scientific::buttap(order); }
    if kind == 1 { return scientific::cheb1ap(order, rp); }
    if kind == 2 { return scientific::cheb2ap(order, rs); }
    if kind == 3 { return scientific::ellipap(order, rp, rs); }
    return scientific::besselap(order, "phase");
}

// band_name gives the name of the band of the code that the pin gives.
fn band_name(code: i64) -> const u8[:] {
    if code == 0 { return "lowpass"; }
    if code == 1 { return "highpass"; }
    if code == 2 { return "bandpass"; }
    return "bandstop";
}

// ftype_name gives the name of the kind of filter of the code that the
// pin gives.
fn ftype_name(code: i64) -> const u8[:] {
    if code == 0 { return "butter"; }
    if code == 1 { return "cheby1"; }
    if code == 2 { return "cheby2"; }
    if code == 3 { return "ellip"; }
    return "bessel";
}

fn test_prototypes() {
    // The five analog prototypes.
    let cases = signal_design_pins::PRO_N.len;
    assert(cases >= 8);
    var i: i64 = 0;
    while i < cases {
        let order = signal_design_pins::PRO_N[i] as i64;
        let rp = signal_design_pins::PRO_RP[i];
        let rs = signal_design_pins::PRO_RS[i];
        let kind = signal_design_pins::PRO_KIND[i] as i64;
        let zr = common::part_of(signal_design_pins::PRO_ZR_D, signal_design_pins::PRO_ZR_L, i);
        let zi = common::part_of(signal_design_pins::PRO_ZI_D, signal_design_pins::PRO_ZI_L, i);
        let pr = common::part_of(signal_design_pins::PRO_PR_D, signal_design_pins::PRO_PR_L, i);
        let pim = common::part_of(signal_design_pins::PRO_PI_D, signal_design_pins::PRO_PI_L, i);
        let k = signal_design_pins::PRO_K[i];
        let gzr, gzi, gpr, gpi, gk, f = proto_of(kind, order, rp, rs);
        assert(f.kind == scientific::F_NONE);
        assert(roots_match(gzr, gzi, zr, zi, 1.0e-9));
        assert(roots_match(gpr, gpi, pr, pim, 1.0e-9));
        assert(common::close(gk, k, 1.0e-9));
        i += 1;
    }
}

fn test_prototype_faults() {
    // An order outside the range.
    let a1, b1, c1, d1, e1, f1 = scientific::buttap(-1);
    assert(f1.kind == scientific::F_VALUE);
    let a2, b2, c2, d2, e2, f2 = scientific::cheb1ap(3, 0.0);
    assert(f2.kind == scientific::F_VALUE);
    let a3, b3, c3, d3, e3, f3 = scientific::cheb2ap(3, 0.0);
    assert(f3.kind == scientific::F_VALUE);
    // The attenuation below the ripple.
    let a4, b4, c4, d4, e4, f4 = scientific::ellipap(3, 10.0, 5.0);
    assert(f4.kind == scientific::F_VALUE);
    // An unknown norm.
    let a5, b5, c5, d5, e5, f5 = scientific::besselap(3, "flat");
    assert(f5.kind == scientific::F_VALUE);
}

fn test_order_selection() {
    // The four calls that choose the order of a filter.
    let cases = signal_design_pins::ORD_GPASS.len;
    assert(cases >= 6);
    var i: i64 = 0;
    while i < cases {
        let wp = common::part_of(signal_design_pins::ORD_WP_D, signal_design_pins::ORD_WP_L, i);
        let ws = common::part_of(signal_design_pins::ORD_WS_D, signal_design_pins::ORD_WS_L, i);
        let gpass = signal_design_pins::ORD_GPASS[i];
        let gstop = signal_design_pins::ORD_GSTOP[i];
        let analog = signal_design_pins::ORD_ANALOG[i] != 0.0;
        let n1, w1, f1 = scientific::buttord(wp, ws, gpass, gstop, analog);
        assert(f1.kind == scientific::F_NONE);
        assert(n1 == signal_design_pins::BU_ORD[i] as i64);
        assert(common::rel_close(w1, common::part_of(signal_design_pins::BU_WN_D, signal_design_pins::BU_WN_L, i), 1.0e-6));
        let n2, w2, f2 = scientific::cheb1ord(wp, ws, gpass, gstop, analog);
        assert(f2.kind == scientific::F_NONE);
        assert(n2 == signal_design_pins::C1_ORD[i] as i64);
        assert(common::rel_close(w2, common::part_of(signal_design_pins::C1_WN_D, signal_design_pins::C1_WN_L, i), 1.0e-6));
        let n3, w3, f3 = scientific::cheb2ord(wp, ws, gpass, gstop, analog);
        assert(f3.kind == scientific::F_NONE);
        assert(n3 == signal_design_pins::C2_ORD[i] as i64);
        assert(common::rel_close(w3, common::part_of(signal_design_pins::C2_WN_D, signal_design_pins::C2_WN_L, i), 1.0e-6));
        let n4, w4, f4 = scientific::ellipord(wp, ws, gpass, gstop, analog);
        assert(f4.kind == scientific::F_NONE);
        assert(n4 == signal_design_pins::EL_ORD[i] as i64);
        assert(common::rel_close(w4, common::part_of(signal_design_pins::EL_WN_D, signal_design_pins::EL_WN_L, i), 1.0e-6));
        i += 1;
    }
}

fn test_order_faults() {
    // A ripple of zero.
    let n1, w1, f1 = scientific::buttord([0.2], [0.8], 0.0, 30.0);
    assert(f1.kind == scientific::F_VALUE);
    // A ripple above the attenuation.
    let n2, w2, f2 = scientific::cheb1ord([0.2], [0.8], 40.0, 30.0);
    assert(f2.kind == scientific::F_VALUE);
    // Two lists of a different count.
    let n3, w3, f3 = scientific::buttord([0.2, 0.4], [0.8], 1.0, 30.0);
    assert(f3.kind == scientific::F_SHAPE);
    // An edge of zero.
    let n4, w4, f4 = scientific::buttord([0.0], [0.8], 1.0, 30.0);
    assert(f4.kind == scientific::F_RANGE);
}

fn test_iirfilter() {
    // The design of a digital and of an analog filter.
    let cases = signal_design_pins::IIR_WN_L.len;
    assert(cases >= 8);
    var i: i64 = 0;
    while i < cases {
        let wn = common::part_of(signal_design_pins::IIR_WN_D, signal_design_pins::IIR_WN_L, i);
        let analog = signal_design_pins::IIR_ANALOG[i] != 0.0;
        let rp = signal_design_pins::IIR_RP[i];
        let rs = signal_design_pins::IIR_RS[i];
        let order = signal_design_pins::IIR_ORD[i] as i64;
        let band = band_name(signal_design_pins::IIR_BAND[i] as i64);
        let ftype = ftype_name(signal_design_pins::IIR_FTYPE[i] as i64);
        let b, a, f = scientific::iirfilter(order, wn, rp, rs, band, ftype, analog);
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(b, common::part_of(signal_design_pins::IIR_B_D, signal_design_pins::IIR_B_L, i), 1.0e-7));
        assert(common::rel_close(a, common::part_of(signal_design_pins::IIR_A_D, signal_design_pins::IIR_A_L, i), 1.0e-7));
        i += 1;
    }
}

fn test_iirdesign() {
    // The design of a filter from its edges.
    let cases = signal_design_pins::DES_WP_L.len;
    assert(cases >= 5);
    var i: i64 = 0;
    while i < cases {
        let wp = common::part_of(signal_design_pins::DES_WP_D, signal_design_pins::DES_WP_L, i);
        let ws = common::part_of(signal_design_pins::DES_WS_D, signal_design_pins::DES_WS_L, i);
        let gpass = signal_design_pins::DES_GPASS[i];
        let gstop = signal_design_pins::DES_GSTOP[i];
        let ftype = ftype_name(signal_design_pins::DES_FTYPE[i] as i64);
        let b, a, f = scientific::iirdesign(wp, ws, gpass, gstop, false, ftype);
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(b, common::part_of(signal_design_pins::DES_B_D, signal_design_pins::DES_B_L, i), 1.0e-6));
        assert(common::rel_close(a, common::part_of(signal_design_pins::DES_A_D, signal_design_pins::DES_A_L, i), 1.0e-6));
        i += 1;
    }
}

fn test_iir_faults() {
    // An unknown band.
    let b1, a1, f1 = scientific::iirfilter(3, [0.2], 1.0, 30.0, "wrong", "butter");
    assert(f1.kind == scientific::F_VALUE);
    // An unknown kind of filter.
    let b2, a2, f2 = scientific::iirfilter(3, [0.2], 1.0, 30.0, "lowpass", "wrong");
    assert(f2.kind == scientific::F_VALUE);
    // A cut outside the range.
    let b3, a3, f3 = scientific::iirfilter(3, [1.5], 1.0, 30.0, "lowpass", "butter");
    assert(f3.kind == scientific::F_RANGE);
    // Two cuts for a low pass.
    let b4, a4, f4 = scientific::iirfilter(3, [0.1, 0.2], 1.0, 30.0, "lowpass", "butter");
    assert(f4.kind == scientific::F_SHAPE);
    // The order outside the range.
    let b5, a5, f5 = scientific::iirfilter(0, [0.2], 1.0, 30.0, "lowpass", "butter");
    assert(f5.kind == scientific::F_VALUE);
    // The pass band outside the stop band.
    let b6, a6, f6 = scientific::iirdesign([0.1, 0.5], [0.3, 0.8], 1.0, 30.0, false, "ellip");
    assert(f6.kind == scientific::F_RANGE);
}

fn test_notch_peak_comb() {
    // The notch, the peak and the comb filters.
    let b1, a1, f1 = scientific::iirnotch(signal_design_pins::NP_W0[0], signal_design_pins::NP_Q[0], signal_design_pins::NP_FS[0]);
    assert(f1.kind == scientific::F_NONE);
    assert(common::rel_close(b1, signal_design_pins::NOTCH_B, 1.0e-9));
    assert(common::rel_close(a1, signal_design_pins::NOTCH_A, 1.0e-9));
    let b2, a2, f2 = scientific::iirpeak(signal_design_pins::NP_W0[0], signal_design_pins::NP_Q[0], signal_design_pins::NP_FS[0]);
    assert(f2.kind == scientific::F_NONE);
    assert(common::rel_close(b2, signal_design_pins::PEAK_B, 1.0e-9));
    assert(common::rel_close(a2, signal_design_pins::PEAK_A, 1.0e-9));
    let b3, a3, f3 = scientific::iircomb(500.0, 30.0, "notch", 2000.0, false);
    assert(f3.kind == scientific::F_NONE);
    assert(common::rel_close(b3, signal_design_pins::COMB_B, 1.0e-9));
    assert(common::rel_close(a3, signal_design_pins::COMB_A, 1.0e-9));
    let b4, a4, f4 = scientific::iircomb(500.0, 30.0, "peak", 2000.0, false);
    assert(f4.kind == scientific::F_NONE);
    assert(common::rel_close(b4, signal_design_pins::COMB_PK_B, 1.0e-9));
    assert(common::rel_close(a4, signal_design_pins::COMB_PK_A, 1.0e-9));
    // The faults.
    let b5, a5, f5 = scientific::iirnotch(2000.0, 30.0, 2000.0);
    assert(f5.kind == scientific::F_RANGE);
    let b6, a6, f6 = scientific::iircomb(500.0, 0.0, "notch", 2000.0, false);
    assert(f6.kind == scientific::F_VALUE);
    let b7, a7, f7 = scientific::iircomb(500.0, 30.0, "wrong", 2000.0, false);
    assert(f7.kind == scientific::F_VALUE);
    // The rate does not divide by the frequency. A rate of 3000 divides
    // by 500, so the case takes the rate 2100.
    let b8, a8, f8 = scientific::iircomb(500.0, 30.0, "notch", 2100.0, false);
    assert(f8.kind == scientific::F_VALUE);
}

fn test_kaiser() {
    // The shape, the attenuation and the count of taps.
    let cases = signal_design_pins::KA_A.len;
    var i: i64 = 0;
    while i < cases {
        assert(common::close(scientific::kaiser_beta(signal_design_pins::KA_A[i]), signal_design_pins::KA_BETA[i], 1.0e-12));
        i += 1;
    }
    i = 0;
    while i < 3 {
        assert(common::close(scientific::kaiser_atten(signal_design_pins::KA_NT[i] as i64, signal_design_pins::KA_W[i]), signal_design_pins::KA_ATTEN[i], 1.0e-12));
        i += 1;
    }
    i = 0;
    while i < 3 {
        let n, beta, f = scientific::kaiserord(signal_design_pins::KO_R[i], signal_design_pins::KO_W[i]);
        assert(f.kind == scientific::F_NONE);
        assert(n == signal_design_pins::KO_NT[i] as i64);
        assert(common::close(beta, signal_design_pins::KO_BETA[i], 1.0e-12));
        i += 1;
    }
    // An attenuation below eight decibels.
    let n1, beta1, f1 = scientific::kaiserord(5.0, 0.1);
    assert(f1.kind == scientific::F_VALUE);
    // A width of zero.
    let n2, beta2, f2 = scientific::kaiserord(30.0, 0.0);
    assert(f2.kind == scientific::F_VALUE);
}

fn test_band_transforms() {
    // The four maps of the band on the coefficients.
    let cases = signal_design_pins::LP_B_L.len;
    assert(cases >= 4);
    var i: i64 = 0;
    while i < cases {
        let b = common::part_of(signal_design_pins::LP_B_D, signal_design_pins::LP_B_L, i);
        let a = common::part_of(signal_design_pins::LP_A_D, signal_design_pins::LP_A_L, i);
        let wo = signal_design_pins::LP_WO[i];
        let b1, a1, f1 = scientific::lp2lp(b, a, wo);
        assert(f1.kind == scientific::F_NONE);
        assert(common::rel_close(b1, common::part_of(signal_design_pins::LPLP_B_D, signal_design_pins::LPLP_B_L, i), 1.0e-9));
        assert(common::rel_close(a1, common::part_of(signal_design_pins::LPLP_A_D, signal_design_pins::LPLP_A_L, i), 1.0e-9));
        let b2, a2, f2 = scientific::lp2hp(b, a, wo);
        assert(f2.kind == scientific::F_NONE);
        assert(common::rel_close(b2, common::part_of(signal_design_pins::LPHP_B_D, signal_design_pins::LPHP_B_L, i), 1.0e-9));
        assert(common::rel_close(a2, common::part_of(signal_design_pins::LPHP_A_D, signal_design_pins::LPHP_A_L, i), 1.0e-9));
        i += 1;
    }
    i = 0;
    while i < signal_design_pins::BP_B_L.len {
        let b = common::part_of(signal_design_pins::BP_B_D, signal_design_pins::BP_B_L, i);
        let a = common::part_of(signal_design_pins::BP_A_D, signal_design_pins::BP_A_L, i);
        let wo = signal_design_pins::BP_WO[i];
        let bw = signal_design_pins::BP_BW[i];
        let b1, a1, f1 = scientific::lp2bp(b, a, wo, bw);
        assert(f1.kind == scientific::F_NONE);
        assert(common::rel_close(b1, common::part_of(signal_design_pins::LPBP_B_D, signal_design_pins::LPBP_B_L, i), 1.0e-9));
        assert(common::rel_close(a1, common::part_of(signal_design_pins::LPBP_A_D, signal_design_pins::LPBP_A_L, i), 1.0e-9));
        let b2, a2, f2 = scientific::lp2bs(b, a, wo, bw);
        assert(f2.kind == scientific::F_NONE);
        assert(common::rel_close(b2, common::part_of(signal_design_pins::LPBS_B_D, signal_design_pins::LPBS_B_L, i), 1.0e-9));
        assert(common::rel_close(a2, common::part_of(signal_design_pins::LPBS_A_D, signal_design_pins::LPBS_A_L, i), 1.0e-9));
        i += 1;
    }
}

fn test_zpk_transforms() {
    // The four maps of the band on the zeros and the poles.
    let cases = signal_design_pins::ZT_K.len;
    assert(cases >= 3);
    var i: i64 = 0;
    while i < cases {
        let zr = common::part_of(signal_design_pins::ZT_ZR_D, signal_design_pins::ZT_ZR_L, i);
        let zi = common::part_of(signal_design_pins::ZT_ZI_D, signal_design_pins::ZT_ZI_L, i);
        let pr = common::part_of(signal_design_pins::ZT_PR_D, signal_design_pins::ZT_PR_L, i);
        let pim = common::part_of(signal_design_pins::ZT_PI_D, signal_design_pins::ZT_PI_L, i);
        let k = signal_design_pins::ZT_K[i];
        let wo = signal_design_pins::ZT_WO[i];
        let bw = signal_design_pins::ZT_BW[i];
        let a1, b1, c1, d1, e1, f1 = scientific::lp2lp_zpk(zr, zi, pr, pim, k, wo);
        assert(f1.kind == scientific::F_NONE);
        assert(roots_match(a1, b1, common::part_of(signal_design_pins::LZPK_ZR_D, signal_design_pins::LZPK_ZR_L, i), common::part_of(signal_design_pins::LZPK_ZI_D, signal_design_pins::LZPK_ZI_L, i), 1.0e-9));
        assert(roots_match(c1, d1, common::part_of(signal_design_pins::LZPK_PR_D, signal_design_pins::LZPK_PR_L, i), common::part_of(signal_design_pins::LZPK_PI_D, signal_design_pins::LZPK_PI_L, i), 1.0e-9));
        assert(common::close(e1, signal_design_pins::LZPK_K[i], 1.0e-9));
        let a2, b2, c2, d2, e2, f2 = scientific::lp2hp_zpk(zr, zi, pr, pim, k, wo);
        assert(f2.kind == scientific::F_NONE);
        assert(roots_match(a2, b2, common::part_of(signal_design_pins::HZPK_ZR_D, signal_design_pins::HZPK_ZR_L, i), common::part_of(signal_design_pins::HZPK_ZI_D, signal_design_pins::HZPK_ZI_L, i), 1.0e-9));
        assert(roots_match(c2, d2, common::part_of(signal_design_pins::HZPK_PR_D, signal_design_pins::HZPK_PR_L, i), common::part_of(signal_design_pins::HZPK_PI_D, signal_design_pins::HZPK_PI_L, i), 1.0e-9));
        assert(common::close(e2, signal_design_pins::HZPK_K[i], 1.0e-9));
        let a3, b3, c3, d3, e3, f3 = scientific::lp2bp_zpk(zr, zi, pr, pim, k, wo, bw);
        assert(f3.kind == scientific::F_NONE);
        assert(roots_match(a3, b3, common::part_of(signal_design_pins::BZPK_ZR_D, signal_design_pins::BZPK_ZR_L, i), common::part_of(signal_design_pins::BZPK_ZI_D, signal_design_pins::BZPK_ZI_L, i), 1.0e-9));
        assert(roots_match(c3, d3, common::part_of(signal_design_pins::BZPK_PR_D, signal_design_pins::BZPK_PR_L, i), common::part_of(signal_design_pins::BZPK_PI_D, signal_design_pins::BZPK_PI_L, i), 1.0e-9));
        assert(common::close(e3, signal_design_pins::BZPK_K[i], 1.0e-9));
        let a4, b4, c4, d4, e4, f4 = scientific::lp2bs_zpk(zr, zi, pr, pim, k, wo, bw);
        assert(f4.kind == scientific::F_NONE);
        assert(roots_match(a4, b4, common::part_of(signal_design_pins::SZPK_ZR_D, signal_design_pins::SZPK_ZR_L, i), common::part_of(signal_design_pins::SZPK_ZI_D, signal_design_pins::SZPK_ZI_L, i), 1.0e-9));
        assert(roots_match(c4, d4, common::part_of(signal_design_pins::SZPK_PR_D, signal_design_pins::SZPK_PR_L, i), common::part_of(signal_design_pins::SZPK_PI_D, signal_design_pins::SZPK_PI_L, i), 1.0e-9));
        assert(common::close(e4, signal_design_pins::SZPK_K[i], 1.0e-9));
        i += 1;
    }
}

fn test_bilinear() {
    // The bilinear rule on the coefficients and on the roots.
    let cases = signal_design_pins::BL_B_L.len;
    assert(cases >= 4);
    var i: i64 = 0;
    while i < cases {
        let b = common::part_of(signal_design_pins::BL_B_D, signal_design_pins::BL_B_L, i);
        let a = common::part_of(signal_design_pins::BL_A_D, signal_design_pins::BL_A_L, i);
        let fs = signal_design_pins::BL_FS[i];
        let b1, a1, f1 = scientific::bilinear(b, a, fs);
        assert(f1.kind == scientific::F_NONE);
        assert(common::rel_close(b1, common::part_of(signal_design_pins::BL_BOUT_D, signal_design_pins::BL_BOUT_L, i), 1.0e-9));
        assert(common::rel_close(a1, common::part_of(signal_design_pins::BL_AOUT_D, signal_design_pins::BL_AOUT_L, i), 1.0e-9));
        i += 1;
    }
    i = 0;
    while i < signal_design_pins::BZ_K.len {
        let zr = common::part_of(signal_design_pins::BZ_ZR_D, signal_design_pins::BZ_ZR_L, i);
        let zi = common::part_of(signal_design_pins::BZ_ZI_D, signal_design_pins::BZ_ZI_L, i);
        let pr = common::part_of(signal_design_pins::BZ_PR_D, signal_design_pins::BZ_PR_L, i);
        let pim = common::part_of(signal_design_pins::BZ_PI_D, signal_design_pins::BZ_PI_L, i);
        let k = signal_design_pins::BZ_K[i];
        let fs = signal_design_pins::BZ_FS[i];
        let a1, b1, c1, d1, e1, f1 = scientific::bilinear_zpk(zr, zi, pr, pim, k, fs);
        assert(f1.kind == scientific::F_NONE);
        assert(roots_match(a1, b1, common::part_of(signal_design_pins::BZ_ZRO_D, signal_design_pins::BZ_ZRO_L, i), common::part_of(signal_design_pins::BZ_ZIO_D, signal_design_pins::BZ_ZIO_L, i), 1.0e-9));
        assert(roots_match(c1, d1, common::part_of(signal_design_pins::BZ_PRO_D, signal_design_pins::BZ_PRO_L, i), common::part_of(signal_design_pins::BZ_PIO_D, signal_design_pins::BZ_PIO_L, i), 1.0e-9));
        assert(common::close(e1, signal_design_pins::BZ_KO[i], 1.0e-9));
        i += 1;
    }
}

fn test_normalize() {
    // The scale of the coefficients.
    let cases = signal_design_pins::NZ_B_L.len;
    assert(cases >= 4);
    var i: i64 = 0;
    while i < cases {
        let b = common::part_of(signal_design_pins::NZ_B_D, signal_design_pins::NZ_B_L, i);
        let a = common::part_of(signal_design_pins::NZ_A_D, signal_design_pins::NZ_A_L, i);
        let b1, a1, f1 = scientific::normalize(b, a);
        assert(f1.kind == scientific::F_NONE);
        assert(common::rel_close(b1, common::part_of(signal_design_pins::NZ_BOUT_D, signal_design_pins::NZ_BOUT_L, i), 1.0e-9));
        assert(common::rel_close(a1, common::part_of(signal_design_pins::NZ_AOUT_D, signal_design_pins::NZ_AOUT_L, i), 1.0e-9));
        i += 1;
    }
    // A denominator of zeros.
    let b1, a1, f1 = scientific::normalize([1.0], [0.0, 0.0]);
    assert(f1.kind == scientific::F_VALUE);
}

fn test_analog_response() {
    // The grid and the response of an analog filter.
    let cases = signal_design_pins::FF_B_L.len;
    assert(cases >= 3);
    var i: i64 = 0;
    while i < cases {
        let b = common::part_of(signal_design_pins::FF_B_D, signal_design_pins::FF_B_L, i);
        let a = common::part_of(signal_design_pins::FF_A_D, signal_design_pins::FF_A_L, i);
        let n = signal_design_pins::FF_N[i] as i64;
        let w, f = scientific::findfreqs(b, a, n);
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(w, common::part_of(signal_design_pins::FF_W_D, signal_design_pins::FF_W_L, i), 1.0e-9));
        i += 1;
    }
    i = 0;
    while i < signal_design_pins::FQ_B_L.len {
        let b = common::part_of(signal_design_pins::FQ_B_D, signal_design_pins::FQ_B_L, i);
        let a = common::part_of(signal_design_pins::FQ_A_D, signal_design_pins::FQ_A_L, i);
        let w, hr, hi, f = scientific::freqs(b, a, 6);
        assert(f.kind == scientific::F_NONE);
        assert(w.data.len == 6);
        assert(common::rel_close(hr, common::part_of(signal_design_pins::FQ_HR_D, signal_design_pins::FQ_HR_L, i), 1.0e-9));
        assert(common::rel_close(hi, common::part_of(signal_design_pins::FQ_HI_D, signal_design_pins::FQ_HI_L, i), 1.0e-9));
        i += 1;
    }
    i = 0;
    while i < signal_design_pins::FZ_K.len {
        let zr = common::part_of(signal_design_pins::FZ_ZR_D, signal_design_pins::FZ_ZR_L, i);
        let zi = common::part_of(signal_design_pins::FZ_ZI_D, signal_design_pins::FZ_ZI_L, i);
        let pr = common::part_of(signal_design_pins::FZ_PR_D, signal_design_pins::FZ_PR_L, i);
        let pim = common::part_of(signal_design_pins::FZ_PI_D, signal_design_pins::FZ_PI_L, i);
        let w, hr, hi, f = scientific::freqs_zpk(zr, zi, pr, pim, signal_design_pins::FZ_K[i], 6);
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(w, common::part_of(signal_design_pins::FZ_W_D, signal_design_pins::FZ_W_L, i), 1.0e-9));
        assert(common::rel_close(hr, common::part_of(signal_design_pins::FZ_HR_D, signal_design_pins::FZ_HR_L, i), 1.0e-9));
        assert(common::rel_close(hi, common::part_of(signal_design_pins::FZ_HI_D, signal_design_pins::FZ_HI_L, i), 1.0e-9));
        i += 1;
    }
}

fn test_digital_response() {
    // The response of a digital filter from its roots and from its
    // sections.
    let cases = signal_design_pins::DZ_B_L.len;
    assert(cases >= 2);
    var i: i64 = 0;
    while i < cases {
        let b = common::part_of(signal_design_pins::DZ_B_D, signal_design_pins::DZ_B_L, i);
        let a = common::part_of(signal_design_pins::DZ_A_D, signal_design_pins::DZ_A_L, i);
        let zr, zi, pr, pim, k, fz2 = scientific::tf2zpk(b, a);
        assert(fz2.kind == scientific::F_NONE);
        let w, hr, hi, f = scientific::freqz_zpk(zr.data, zi.data, pr.data, pim.data, k, 6);
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(hr, common::part_of(signal_design_pins::DZ_HR_D, signal_design_pins::DZ_HR_L, i), 1.0e-7));
        assert(common::rel_close(hi, common::part_of(signal_design_pins::DZ_HI_D, signal_design_pins::DZ_HI_L, i), 1.0e-7));
        i += 1;
    }
    let sos = common::mat(signal_design_pins::SOS_ROWS, 2, 6);
    let w, hr, hi, f = scientific::freqz_sos(copy(sos), 6);
    assert(f.kind == scientific::F_NONE);
    assert(common::rel_close(hr, signal_design_pins::SOS_HR, 1.0e-9));
    assert(common::rel_close(hi, signal_design_pins::SOS_HI, 1.0e-9));
    // The faults.
    let w2, hr2, hi2, f2 = scientific::freqz_sos(common::mat([1.0, 1.0], 1, 2), 6);
    assert(f2.kind == scientific::F_SHAPE);
    let w3, hr3, hi3, f3 = scientific::freqz_sos(common::mat([], 0, 6), 6);
    assert(f3.kind == scientific::F_VALUE);
}

fn test_group_delay() {
    // The group delay of a digital filter.
    let cases = signal_design_pins::GD_B_L.len;
    assert(cases >= 3);
    var i: i64 = 0;
    while i < cases {
        let b = common::part_of(signal_design_pins::GD_B_D, signal_design_pins::GD_B_L, i);
        let a = common::part_of(signal_design_pins::GD_A_D, signal_design_pins::GD_A_L, i);
        let w, gd, f = scientific::group_delay(b, a, 6);
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(w, common::part_of(signal_design_pins::GD_W_D, signal_design_pins::GD_W_L, i), 1.0e-9));
        assert(common::rel_close(gd, common::part_of(signal_design_pins::GD_G_D, signal_design_pins::GD_G_L, i), 1.0e-9));
        i += 1;
    }
}

fn test_discretize() {
    // The discretization of a system with the seven methods.
    let cases = signal_design_pins::DIS_N.len;
    assert(cases >= 3);
    var i: i64 = 0;
    while i < cases {
        let n = signal_design_pins::DIS_N[i] as i64;
        let a = common::mat(common::part_of(signal_design_pins::DIS_A_D, signal_design_pins::DIS_A_L, i), n, n);
        let b = common::mat(common::part_of(signal_design_pins::DIS_B_D, signal_design_pins::DIS_B_L, i), n, 1);
        let c = common::mat(common::part_of(signal_design_pins::DIS_C_D, signal_design_pins::DIS_C_L, i), 1, n);
        let dt = signal_design_pins::DIS_DT[i];
        let ad, bd, cd, dd, f = scientific::cont2discrete(copy(a), copy(b), copy(c), 0.0, dt, "zoh", 0.5);
        assert(f.kind == scientific::F_NONE);
        assert(common::rel_close(ad, common::part_of(signal_design_pins::ZO_AD_D, signal_design_pins::ZO_AD_L, i), 1.0e-9));
        assert(common::rel_close(bd, common::part_of(signal_design_pins::ZO_BD_D, signal_design_pins::ZO_BD_L, i), 1.0e-9));
        assert(common::rel_close(cd, common::part_of(signal_design_pins::ZO_CD_D, signal_design_pins::ZO_CD_L, i), 1.0e-9));
        assert(common::close(dd, signal_design_pins::ZO_DD[i], 1.0e-9));
        let ad2, bd2, cd2, dd2, f2 = scientific::cont2discrete(copy(a), copy(b), copy(c), 0.0, dt, "foh", 0.5);
        assert(f2.kind == scientific::F_NONE);
        assert(common::rel_close(ad2, common::part_of(signal_design_pins::FO_AD_D, signal_design_pins::FO_AD_L, i), 1.0e-9));
        assert(common::rel_close(bd2, common::part_of(signal_design_pins::FO_BD_D, signal_design_pins::FO_BD_L, i), 1.0e-9));
        assert(common::rel_close(cd2, common::part_of(signal_design_pins::FO_CD_D, signal_design_pins::FO_CD_L, i), 1.0e-9));
        assert(common::close(dd2, signal_design_pins::FO_DD[i], 1.0e-9));
        let ad3, bd3, cd3, dd3, f3 = scientific::cont2discrete(copy(a), copy(b), copy(c), 0.0, dt, "bilinear", 0.5);
        assert(f3.kind == scientific::F_NONE);
        assert(common::rel_close(ad3, common::part_of(signal_design_pins::BI_AD_D, signal_design_pins::BI_AD_L, i), 1.0e-9));
        assert(common::rel_close(bd3, common::part_of(signal_design_pins::BI_BD_D, signal_design_pins::BI_BD_L, i), 1.0e-9));
        assert(common::rel_close(cd3, common::part_of(signal_design_pins::BI_CD_D, signal_design_pins::BI_CD_L, i), 1.0e-9));
        assert(common::close(dd3, signal_design_pins::BI_DD[i], 1.0e-9));
        let ad4, bd4, cd4, dd4, f4 = scientific::cont2discrete(copy(a), copy(b), copy(c), 0.0, dt, "euler", 0.5);
        assert(f4.kind == scientific::F_NONE);
        assert(common::rel_close(ad4, common::part_of(signal_design_pins::EU_AD_D, signal_design_pins::EU_AD_L, i), 1.0e-9));
        let ad5, bd5, cd5, dd5, f5 = scientific::cont2discrete(copy(a), copy(b), copy(c), 0.0, dt, "backward_diff", 0.5);
        assert(f5.kind == scientific::F_NONE);
        assert(common::rel_close(ad5, common::part_of(signal_design_pins::BD_AD_D, signal_design_pins::BD_AD_L, i), 1.0e-9));
        let ad6, bd6, cd6, dd6, f6 = scientific::cont2discrete(copy(a), copy(b), copy(c), 0.0, dt, "gbt", 0.3);
        assert(f6.kind == scientific::F_NONE);
        assert(common::rel_close(ad6, common::part_of(signal_design_pins::GB_AD_D, signal_design_pins::GB_AD_L, i), 1.0e-9));
        assert(common::rel_close(bd6, common::part_of(signal_design_pins::GB_BD_D, signal_design_pins::GB_BD_L, i), 1.0e-9));
        assert(common::rel_close(cd6, common::part_of(signal_design_pins::GB_CD_D, signal_design_pins::GB_CD_L, i), 1.0e-9));
        assert(common::close(dd6, signal_design_pins::GB_DD[i], 1.0e-9));
        i += 1;
    }
}

fn test_discretize_faults() {
    // An unknown method.
    let a = common::mat([1.0, 0.0, 0.0, 1.0], 2, 2);
    let b = common::mat([1.0, 0.0], 2, 1);
    let c = common::mat([1.0, 0.0], 1, 2);
    let ad, bd, cd, dd, f = scientific::cont2discrete(copy(a), copy(b), copy(c), 0.0, 1.0, "wrong", 0.5);
    assert(f.kind == scientific::F_VALUE);
    // A step of zero.
    let ad2, bd2, cd2, dd2, f2 = scientific::cont2discrete(copy(a), copy(b), copy(c), 0.0, 0.0, "zoh", 0.5);
    assert(f2.kind == scientific::F_VALUE);
    // The alpha outside the range.
    let ad3, bd3, cd3, dd3, f3 = scientific::cont2discrete(copy(a), copy(b), copy(c), 0.0, 1.0, "gbt", 1.5);
    assert(f3.kind == scientific::F_VALUE);
    // A matrix of the states that is not square.
    let a4 = common::mat([1.0, 0.0, 0.0], 1, 3);
    let ad4, bd4, cd4, dd4, f4 = scientific::cont2discrete(copy(a4), copy(b), copy(c), 0.0, 1.0, "zoh", 0.5);
    assert(f4.kind == scientific::F_SHAPE);
}

// A matrix that a caller built by hand and broke does not agree with
// its shape. The call that reads the buffer of a matrix gives a fault.
fn test_discretize_broken_matrix() {
    // The matrix of the states holds fewer values than its shape asks for.
    let a = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let b = common::mat([1.0, 0.0], 2, 1);
    let c = common::mat([1.0, 0.0], 1, 2);
    let ad, bd, cd, dd, f = scientific::cont2discrete(copy(a), copy(b), copy(c), 0.0, 1.0, "zoh", 0.5);
    assert(f.kind == scientific::F_SHAPE);
}
