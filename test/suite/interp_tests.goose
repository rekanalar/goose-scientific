// The tests of the group interp. Run them with SCIENTIFIC_SUITES=interp.
namespace interp_tests;

import scientific from "../..";
import os;
import common;

fn test_interp() {
    let sp_notaknot, spe_notaknot = scientific::cubic_spline([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], "not-a-knot");
    assert(spe_notaknot.kind == scientific::F_NONE);
    let v_notaknot0, v_notaknot0e = scientific::spline_eval(sp_notaknot, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 0);
    assert(v_notaknot0e.kind == scientific::F_NONE && common::all_near(v_notaknot0, [-5.125380143112701, 2.7121447227191413, 1.3005502683363148, 2.0, 5.315605903398929, 4.108593917710197, 0.8827298747763836, -6.395102862254024]));
    let v_notaknot1, v_notaknot1e = scientific::spline_eval(sp_notaknot, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 1);
    assert(v_notaknot1e.kind == scientific::F_NONE && common::all_near(v_notaknot1, [17.109473762671435, 3.714668753726893, -1.963345557543232, 4.942300238521169, -2.3462435897435885, 0.9818395945140133, -8.238683661299945, -16.451975253428742]));
    let v_notaknot2, v_notaknot2e = scientific::spline_eval(sp_notaknot, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 2);
    assert(v_notaknot2e.kind == scientific::F_NONE && common::all_near(v_notaknot2, [-21.357245080500892, -12.129767441860464, 4.018318425760286, 13.245796064400711, -2.0901252236135957, 0.6850626118067975, -11.53273703041145, -15.844901610017887]));
    assert(common::near(scientific::spline_integrate(sp_notaknot, 0.5, 6.5), 22.039413759689925) && common::near(scientific::spline_integrate(sp_notaknot, -1.0, 8.0), 11.100130441264163));
    let sp_natural, spe_natural = scientific::cubic_spline([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], "natural");
    assert(spe_natural.kind == scientific::F_NONE);
    let v_natural0, v_natural0e = scientific::spline_eval(sp_natural, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 0);
    assert(v_natural0e.kind == scientific::F_NONE && common::all_near(v_natural0, [-0.47429925469863926, 1.9452898574206094, 1.6167854828256645, 2.0, 5.261066104990279, 4.363503996543531, 0.5778990602721948, -2.734466137394686]));
    let v_natural1, v_natural1e = scientific::spline_eval(sp_natural, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 1);
    assert(v_natural1e.kind == scientific::F_NONE && common::all_near(v_natural1, [2.316199503132427, 2.9233025491466837, -2.2127943400302437, 4.750783106502484, -2.3914074314106717, 1.2436379347591275, -5.753152408727587, -4.8229774249297925]));
    let v_natural2, v_natural2e = scientific::spline_eval(sp_natural, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 2);
    assert(v_natural2e.kind == scientific::F_NONE && common::all_near(v_natural2, [3.7943940375891105, -2.2766364225534677, 2.7796500324044073, 14.62929358392741, -1.8042773817239173, -0.14808813998704018, -0.7751458198314953, 3.875729099157482]));
    assert(common::near(scientific::spline_integrate(sp_natural, 0.5, 6.5), 22.19707617736012) && common::near(scientific::spline_integrate(sp_natural, -1.0, 8.0), 20.900039155325125));
    let sp_clamped, spe_clamped = scientific::cubic_spline([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], "clamped");
    assert(spe_clamped.kind == scientific::F_NONE);
    let v_clamped0, v_clamped0e = scientific::spline_eval(sp_clamped, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 0);
    assert(v_clamped0e.kind == scientific::F_NONE && common::all_near(v_clamped0, [2.8222440036900367, 1.4021369926199263, 1.8376728167281673, 2.0, 5.159841328413286, 4.773958726253929, 0.09228736162361573, 3.0953067343173437]));
    let v_clamped1, v_clamped1e = scientific::spline_eval(sp_clamped, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 1);
    assert(v_clamped1e.kind == scientific::F_NONE && common::all_near(v_clamped1, [-8.170472017220174, 2.3635747232472326, -2.3930565805658057, 4.629920049200493, -2.442749077490774, 1.6577101271012697, -1.7932057195571893, 13.694764760147603]));
    let v_clamped2, v_clamped2e = scientific::spline_eval(sp_clamped, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 2);
    assert(v_clamped2e.kind == scientific::F_NONE && common::all_near(v_clamped2, [21.629920049200493, 4.70519680196802, 1.9123001230012289, 15.64514145141451, -1.3098400984009881, -1.4922919229192324, 16.355811808118094, 35.27075645756457]));
    assert(common::near(scientific::spline_integrate(sp_clamped, 0.5, 6.5), 22.348245630893807) && common::near(scientific::spline_integrate(sp_clamped, -1.0, 8.0), 31.6361405801558));
    let sp_pc, spe_pc = scientific::pchip([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0]);
    let v_pc, v_pce = scientific::spline_eval(sp_pc, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5]);
    assert(v_pce.kind == scientific::F_NONE && common::all_near(v_pc, [-0.45000000000000023, 1.8828, 2.5499259259259257, 2.0, 4.352, 4.225037037037038, 0.647099999999998, -2.7375000000000007]));
    let d_pc, d_pce = scientific::spline_eval(sp_pc, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 1);
    assert(d_pce.kind == scientific::F_NONE && common::all_near(d_pc, [2.500000000000001, 2.734666666666667, -0.9955555555555555, 0.0, -0.96, 0.4977777777777778, -6.4830000000000005, -3.875]));
    let sp_ak, spe_ak = scientific::akima([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0]);
    let v_ak, v_ake = scientific::spline_eval(sp_ak, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5]);
    assert(v_ake.kind == scientific::F_NONE && v_ak.data[0] != v_ak.data[0] && v_ak.data[7] != v_ak.data[7]);
    assert(common::near(v_ak.data[1], 1.8439999999999999) && common::near(v_ak.data[4], 4.4) && common::near(v_ak.data[6], 0.6817928571428551));
    let d_ak, d_ake = scientific::spline_eval(sp_ak, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], 1);
    assert(d_ake.kind == scientific::F_NONE && d_ak.data[0] != d_ak.data[0] && common::near(d_ak.data[1], 2.344761904761905) && common::near(d_ak.data[4], -1.0144927536231882) && common::near(d_ak.data[6], -6.703309523809525));
    let sp_pm, spe_pm = scientific::pchip([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [0.0, 1.0, 1.0, 3.0, 3.0, 3.0, 5.0]);
    let v_pm, v_pme = scientific::spline_eval(sp_pm, [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5]);
    assert(v_pme.kind == scientific::F_NONE && common::all_near(v_pm, [-0.575, 0.4218, 1.0, 1.0, 3.0, 3.0, 4.717200000000001, 6.149999999999999]));
    let i_linear, i_lineare = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [0.3, 1.7, 2.5, 3.9, 5.2, 6.9], "linear");
    assert(i_lineare.kind == scientific::F_NONE && common::all_near(i_linear, [1.6, 2.533333333333333, 2.0, 4.4, 4.233333333333333, 0.4499999999999984]));
    let i_nearest, i_neareste = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [0.3, 1.7, 2.5, 3.9, 5.2, 6.9], "nearest");
    assert(i_neareste.kind == scientific::F_NONE && common::all_near(i_nearest, [1.0, 3.0, 2.0, 4.0, 4.0, 0.0]));
    let i_previous, i_previouse = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [0.3, 1.7, 2.5, 3.9, 5.2, 6.9], "previous");
    assert(i_previouse.kind == scientific::F_NONE && common::all_near(i_previous, [1.0, 3.0, 2.0, 5.0, 4.0, 4.5]));
    let i_next, i_nexte = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [0.3, 1.7, 2.5, 3.9, 5.2, 6.9], "next");
    assert(i_nexte.kind == scientific::F_NONE && common::all_near(i_next, [3.0, 2.0, 2.0, 4.0, 4.5, 0.0]));
    let i_cubic, i_cubice = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [0.3, 1.7, 2.5, 3.9, 5.2, 6.9], "cubic");
    assert(i_cubice.kind == scientific::F_NONE && common::all_near(i_cubic, [2.7121447227191404, 1.3005502683363146, 1.9999999999999996, 5.315605903398926, 4.108593917710197, 0.8827298747763835]));
    let i_nt, i_nte = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [1.75, 2.5, 0.5, 6.5], "nearest");
    assert(i_nte.kind == scientific::F_NONE && common::all_near(i_nt, [3.0, 2.0, 1.0, 4.5]));
    let i_ex, i_exe = scientific::interp1d([0.0, 1.0, 2.5, 3.0, 4.5, 6.0, 7.0], [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0], [-0.5, 0.3, 1.7, 2.5, 3.9, 5.2, 6.9, 7.5], "linear", true);
    assert(i_exe.kind == scientific::F_NONE && common::all_near(i_ex, [0.0, 1.5999999999999999, 2.533333333333333, 2.0, 4.4, 4.233333333333333, 0.4499999999999984, -2.25]));
    let sp3, spe3 = scientific::cubic_spline([0.0, 1.0, 3.0], [1.0, 2.0, 0.0]);
    let v3, v3e = scientific::spline_eval(sp3, [0.5, 2.0, 4.0]);
    assert(v3e.kind == scientific::F_NONE && common::all_near(v3, [1.6666666666666665, 1.6666666666666667, -3.000000000000001]));
    let b1, b1e = scientific::interp1d([0.0, 1.0], [0.0, 1.0], [2.0]);
    assert(b1e.kind == scientific::F_RANGE);
    let b2, b2e = scientific::cubic_spline([0.0, 0.0, 1.0], [1.0, 2.0, 3.0]);
    assert(b2e.kind == scientific::F_VALUE);
    let b3, b3e = scientific::akima([0.0, 1.0], [1.0, 2.0]);
    assert(b3e.kind == scientific::F_SHAPE);
    let b4, b4e = scientific::interp1d([0.0, 1.0], [0.0, 1.0], [0.5], "quad");
    assert(b4e.kind == scientific::F_VALUE);
}

// The periodic condition and the given end slopes of cubic_spline are
// pinned against SciPy.
// Cubic splines with a slope or a second derivative at each end.
fn test_spline_mixed_ends() {
    let kx = [0.0, 0.7, 1.5, 2.4, 4.0];
    let ky = [1.0, 2.0, 0.5, 3.0, 2.2];
    let q = [0.3, 1.0, 3.0, 5.0];
    let a, fa = scientific::cubic_spline_ends(kx, ky, 2, 1.5, 2, -0.5);
    let va, ea = scientific::spline_eval(a, q, 0);
    assert(fa.kind == scientific::F_NONE && ea.kind == scientific::F_NONE && common::all_near(va, [1.6730996537914262, 1.4331011255339694, 3.7385485741574564, 0.1614514258425448]));
    let b, fb = scientific::cubic_spline_ends(kx, ky, 1, 0.3, 2, 0.8);
    let vb, eb = scientific::spline_eval(b, q, 0);
    assert(fb.kind == scientific::F_NONE && common::all_near(vb, [1.4392747194115743, 1.5077575474634295, 3.6484652194211025, 1.551534780578896]));
    let c, fc = scientific::cubic_spline_ends(kx, ky, 2, -1.0, 1, 0.2);
    let vc, ec = scientific::spline_eval(c, q, 0);
    assert(fc.kind == scientific::F_NONE && common::all_near(vc, [1.7380019563098024, 1.400141276736955, 3.2582375136809136, 6.394029225950629]));
    let d, fd = scientific::cubic_spline_ends([0.0, 0.7], [1.0, 2.0], 2, 1.0, 2, 2.0);
    let vd, ed = scientific::spline_eval(d, [0.2, 0.5], 0);
    assert(fd.kind == scientific::F_NONE && common::all_near(vd, [1.2142857142857142, 1.635714285714286]));
    let x1, g1 = scientific::cubic_spline_ends(kx, ky, 3, 0.0, 1, 0.0);
    assert(g1.kind == scientific::F_VALUE);
    let x2, g2 = scientific::cubic_spline_ends(kx, ky, 1, 1.0 / 0.0, 1, 0.0);
    assert(g2.kind == scientific::F_VALUE);
}

fn test_spline_ends() {
    let x = [0.0, 1.0, 2.5, 4.0, 5.0];
    let c, f = scientific::cubic_spline(x, [1.0, 3.0, 2.0, 5.0, 1.0], "periodic");
    assert(f.kind == scientific::F_NONE);
    let v, fv = scientific::spline_eval(c, [0.0, 0.5, 1.7, 3.3, 5.0, 6.0, -1.0, 11.2]);
    let ex = [1.0, 1.6375000000000002, 2.6586074074074078, 4.297718518518518, 1.0, 3.0, 5.0, 3.1597925925925923];
    var i: i64 = 0;
    while i < 8 { assert(common::mathshim_abs(v.data[i] - ex[i]) < 1.0e-10); i += 1; }
    let c2, f2 = scientific::cubic_spline(x, [1.0, 3.0, 2.0, 5.0, 2.0], "clamped", 1.5, -0.5);
    assert(f2.kind == scientific::F_NONE);
    let v2, fv2 = scientific::spline_eval(c2, [0.0, 0.5, 1.7, 3.3, 5.0]);
    let ex2 = [1.0, 2.0993055555555555, 2.477471604938272, 4.180187654320987, 2.0];
    i = 0;
    while i < 5 { assert(common::mathshim_abs(v2.data[i] - ex2[i]) < 1.0e-10); i += 1; }
    let b1, g1 = scientific::cubic_spline(x, [1.0, 3.0, 2.0, 5.0, 2.0], "periodic");
    assert(g1.kind == scientific::F_VALUE);
    let b2, g2 = scientific::cubic_spline(x, [1.0, 3.0, 2.0, 5.0, 2.0], "clamped", 1.0 / 0.0, 0.0);
    assert(g2.kind == scientific::F_VALUE);
}
