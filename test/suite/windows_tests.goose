// The tests of the group windows. Run them with SCIENTIFIC_SUITES=windows.
namespace windows_tests;

import scientific from "../..";
import os;
import common;

fn test_windows_more() {
    let w0, f0 = scientific::general_cosine(9, [0.5, 0.3, 0.2]);
    assert(f0.kind == scientific::F_NONE);
    assert(common::win_close(w0, [0.4, 0.2878679656440357, 0.3, 0.7121320343559643, 1.0, 0.7121320343559643, 0.3, 0.2878679656440357, 0.4]));
    let w1, f1 = scientific::general_hamming(8, 0.6, false);
    assert(f1.kind == scientific::F_NONE);
    assert(common::win_close(w1, [0.19999999999999996, 0.31715728752538097, 0.6, 0.882842712474619, 1.0, 0.882842712474619, 0.6, 0.31715728752538097]));
    let w2, f2 = scientific::general_gaussian(9, 1.5, 2.0);
    assert(f2.kind == scientific::F_NONE);
    assert(common::win_close(w2, [0.01831563888873418, 0.18498139990730428, 0.6065306597126334, 0.9394130628134758, 1.0, 0.9394130628134758, 0.6065306597126334, 0.18498139990730428, 0.01831563888873418]));
    let w3, f3 = scientific::taylor(11);
    assert(f3.kind == scientific::F_NONE);
    assert(common::win_close(w3, [0.26248807473298397, 0.40104077462163656, 0.6089456801270016, 0.8084391241352048, 0.9490117072000545, 1.0, 0.9490117072000545, 0.8084391241352048, 0.6089456801270016, 0.40104077462163656, 0.26248807473298397]));
    let w4, f4 = scientific::taylor(10, 6, 40.0, false, false);
    assert(f4.kind == scientific::F_NONE);
    assert(common::win_close(w4, [0.22955793961566962, 0.5046487811660969, 0.9082773809058199, 1.3273435078317122, 1.6468903591489026, 1.7665640626635986, 1.6468903591489026, 1.3273435078317122, 0.9082773809058199, 0.5046487811660971]));
    let w5, f5 = scientific::chebwin(11, 80.0);
    assert(f5.kind == scientific::F_NONE);
    assert(common::win_close(w5, [0.01688772555101567, 0.09690563774467226, 0.2915289995121729, 0.5900876054755189, 0.8788139405478855, 1.0, 0.8788139405478855, 0.5900876054755189, 0.2915289995121729, 0.09690563774467226, 0.01688772555101567]));
    let w6, f6 = scientific::chebwin(10, 60.0);
    assert(f6.kind == scientific::F_NONE);
    assert(common::win_close(w6, [0.044313249478010214, 0.1888932622106898, 0.45729052434559303, 0.7774679588943729, 1.0, 1.0, 0.7774679588943729, 0.45729052434559303, 0.1888932622106898, 0.044313249478010214]));
    let w7, f7 = scientific::chebwin(9, 100.0, false);
    assert(f7.kind == scientific::F_NONE);
    assert(common::win_close(w7, [0.015138861629262326, 0.10443925148320238, 0.3446060821095421, 0.7092668575677474, 1.0, 1.0, 0.7092668575677474, 0.3446060821095421, 0.10443925148320238]));
    let w8, f8 = scientific::kaiser_bessel_derived(10, 4.0);
    assert(f8.kind == scientific::F_NONE);
    assert(common::win_close(w8, [0.17006094170460917, 0.4417398738085654, 0.7071067811865475, 0.897143179145666, 0.9854335472808613, 0.9854335472808613, 0.897143179145666, 0.7071067811865475, 0.4417398738085654, 0.17006094170460917]));
    let a1, g1 = scientific::kaiser_bessel_derived(9, 4.0);
    assert(g1.kind == scientific::F_VALUE);
    let a2, g2 = scientific::chebwin(0, 80.0);
    assert(g2.kind == scientific::F_RANGE);
    let a3, g3 = scientific::general_cosine(5, []);
    assert(g3.kind == scientific::F_SHAPE);
    let a4, g4 = scientific::general_gaussian(5, 0.0, 1.0);
    assert(g4.kind == scientific::F_VALUE);
    let a5, g5 = scientific::taylor(8, 0);
    assert(g5.kind == scientific::F_RANGE);
    let a6, g6 = scientific::general_hamming(1, 0.5);
    assert(g6.kind == scientific::F_NONE && a6.data[0] == 1.0);
}
