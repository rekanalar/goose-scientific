// The tests of the group welch_ex. Run them with SCIENTIFIC_SUITES=welch_ex.
namespace welch_ex_tests;

import scientific from "../..";
import os;
import common;

fn test_welch_ex() {
    var xs: f64[>..] = [];
    var i: i64 = 0;
    while i < 200 {
        let n = i * 1.0;
        xs.push(sin(0.3 * n) + 0.5 * cos(1.1 * n) + 0.01 * n + 0.2 * sin(0.07 * n * n));
        i += 1;
    }
    let none: f64[0] = [];
    var hw: f64[>..] = [];
    i = 0;
    while i < 48 { hw.push(0.5 - 0.5 * cos(2.0 * 3.141592653589793 * i / 47.0)); i += 1; }
    let f_base, p_base, e_base = scientific::welch_ex(xs, 2.0, "hann", 0.0 / 0.0, none, 64, 32);
    assert(e_base.kind == scientific::F_NONE && common::welch_probe(p_base, 33, [0.008580803681145851, 0.265786092614749, 2.4438586160106217, 10.921707187095503, 0.025841778947693406, 0.012910099196261316]) && common::welch_probe_f(f_base, 33, [0.0, 0.03125, 0.0625, 0.09375, 0.5, 1.0]));
    let f_lin, p_lin, e_lin = scientific::welch_ex(xs, 2.0, "hann", 0.0 / 0.0, none, 64, 16, 128, "linear");
    assert(e_lin.kind == scientific::F_NONE && common::welch_probe(p_lin, 65, [0.005928647136579688, 0.09505438469406048, 0.08696926365952096, 0.17000381337409184, 0.02311959183871981, 0.004393894745147001]) && common::welch_probe_f(f_lin, 65, [0.0, 0.015625, 0.03125, 0.046875, 0.5, 1.0]));
    let f_kaiser, p_kaiser, e_kaiser = scientific::welch_ex(xs, 1.0, "kaiser", 5.0, none, 50, 10, 0, "constant", true, "spectrum");
    assert(e_kaiser.kind == scientific::F_NONE && common::welch_probe(p_kaiser, 26, [0.004967985115824034, 0.028072205691838454, 0.4024180508815359, 0.2786883805124887, 0.0014065466948576023, 0.0007581433080565957]) && common::welch_probe_f(f_kaiser, 26, [0.0, 0.02, 0.04, 0.06, 0.26, 0.5]));
    let f_median, p_median, e_median = scientific::welch_ex(xs, 1.0, "hann", 0.0 / 0.0, none, 40, 20, 0, "constant", true, "density", "median");
    assert(e_median.kind == scientific::F_NONE && common::welch_probe(p_median, 21, [0.0737352443238109, 7.284685749554018, 18.209944734151254, 3.322510417083082, 0.06183859929895203, 0.00803265837389309]) && common::welch_probe_f(f_median, 21, [0.0, 0.025, 0.05, 0.07500000000000001, 0.25, 0.5]));
    let f_two, p_two, e_two = scientific::welch_ex(xs, 1.0, "hamming", 0.0 / 0.0, none, 33, 11, 0, "none", false);
    assert(e_two.kind == scientific::F_NONE && common::welch_probe(p_two, 33, [28.778141833853134, 10.411548932561196, 4.678189852419481, 0.1288335918926191, 0.012261741660769808, 10.411548932561196]) && common::welch_probe_f(f_two, 33, [0.0, 0.030303030303030304, 0.06060606060606061, 0.09090909090909091, 0.48484848484848486, -0.030303030303030304]));
    let f_arr, p_arr, e_arr = scientific::welch_ex(xs, 1.0, "hann", 0.0 / 0.0, hw, 48, 24);
    assert(e_arr.kind == scientific::F_NONE && common::welch_probe(p_arr, 25, [0.29870136620488025, 1.2961880356992184, 14.399889262552481, 8.4708527851176, 0.04118148634273766, 0.018649780237060792]) && common::welch_probe_f(f_arr, 25, [0.0, 0.020833333333333332, 0.041666666666666664, 0.0625, 0.25, 0.5]));
    // Faults.
    let a1, b1, g1 = scientific::welch_ex(xs, 1.0, "hann", 0.0 / 0.0, none, 64, 32, 32);
    assert(g1.kind == scientific::F_RANGE);
    let a2, b2, g2 = scientific::welch_ex(xs, 1.0, "hann", 0.0 / 0.0, none, 64, 32, 0, "cubic");
    assert(g2.kind == scientific::F_VALUE);
    let a3, b3, g3 = scientific::welch_ex(xs, 1.0, "hann", 0.0 / 0.0, none, 64, 32, 0, "constant", true, "density", "mode");
    assert(g3.kind == scientific::F_VALUE);
    let a4, b4, g4 = scientific::welch_ex(xs, 1.0, "hann", 0.0 / 0.0, hw, 64, 32);
    assert(g4.kind == scientific::F_SHAPE);
    let a5, b5, g5 = scientific::welch_ex(xs, 1.0, "hann", 0.0 / 0.0, none, 300, 32);
    assert(g5.kind == scientific::F_RANGE);
    let a6, b6, g6 = scientific::welch_ex(xs, 1.0, "hann", 0.0 / 0.0, none, 64, 64);
    assert(g6.kind == scientific::F_RANGE);
}
