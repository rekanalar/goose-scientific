// The tests of the group filter. Run them with SCIENTIFIC_SUITES=filter.
namespace filter_tests;

import scientific from "../..";
import os;
import common;

fn test_filters() {
    // The values come from SciPy 1.18.1.
    let b, a, f = scientific::butter(3, [0.2], "lowpass");
    assert(f.kind == scientific::F_NONE);
    assert(common::close(b.data[0], 0.018098933007514428) && common::close(b.data[1], 0.05429679902254328));
    assert(common::close(a.data[1], -1.7600418803431688) && common::close(a.data[2], 1.182893262037831) && common::close(a.data[3], -0.27805991763454646));
    let hb, ha, f2 = scientific::butter(2, [0.3], "highpass");
    assert(f2.kind == scientific::F_NONE);
    assert(common::close(hb.data[0], 0.5050010290458776) && common::close(hb.data[1], -1.0100020580917553));
    assert(common::close(ha.data[1], -0.7477891782585034) && common::close(ha.data[2], 0.27221493792500717));
    let bb, ba, f3 = scientific::butter(2, [0.1, 0.4], "bandpass");
    assert(f3.kind == scientific::F_NONE);
    assert(bb.data.len == 5 && common::close(bb.data[0], 0.13110643991662596) && common::close(bb.data[2], -0.2622128798332519));
    let sb, sa, f4 = scientific::butter(2, [0.1, 0.4], "bandstop");
    assert(f4.kind == scientific::F_NONE);
    assert(common::close(sb.data[1], -1.6030843431581088) && common::close(sa.data[1], -2.1806578386027993) && common::close(sa.data[3], -1.025510847713418));
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < 40 { x.push(mathshim::sine(0.3 * i) + 0.1 * i); i += 1; }
    let y, f5 = scientific::lfilter(b.data, a.data, x);
    assert(f5.kind == scientific::F_NONE && y.data.len == 40);
    assert(common::close(y.data[0], 0.0) && common::close(y.data[5], 0.6242264193888316) && common::close(y.data[39], 2.6324122648517223));
    let w, f6 = scientific::filtfilt(b.data, a.data, x);
    assert(f6.kind == scientific::F_NONE && w.data.len == 40);
    assert(common::close(w.data[0], -0.0016527642520426794) && common::close(w.data[5], 1.4879722866741751));
    assert(common::close(w.data[20], 1.7229739986405581) && common::close(w.data[39], 3.1457717019846685));
    let zi, f7 = scientific::lfilter_zi(b.data, a.data);
    assert(f7.kind == scientific::F_NONE && zi.data.len == 3);
    assert(common::close(zi.data[0], 0.9819010669924851) && common::close(zi.data[2], 0.29615885064206077));
    let sos: f64[>..] = [0.0674552738890719, 0.1349105477781438, 0.0674552738890719, 1.0, -1.1429805025399011, 0.41280159809618877];
    let q = scientific::Arr { ndim: 2, shape: [1, 6], data: copy(sos) };
    let z, f8 = scientific::sosfilt(q, x);
    assert(f8.kind == scientific::F_NONE);
    assert(common::close(z.data[5], 0.9284098237835398) && common::close(z.data[39], 2.705008195057858));
    let hr, hi, f9 = scientific::hilbert(x[..7]);
    assert(f9.kind == scientific::F_NONE);
    assert(common::close(hr.data[3], x[3]) && common::close(hi.data[0], 0.731648828187718) && common::close(hi.data[6], 0.9299383059240987));
    // The faults.
    let e1, e2, g1 = scientific::butter(0, [0.2], "lowpass");
    assert(g1.kind == scientific::F_VALUE);
    let e3, e4, g2 = scientific::butter(2, [1.0], "lowpass");
    assert(g2.kind == scientific::F_RANGE);
    let e5, e6, g3 = scientific::butter(2, [0.2], "bandpass");
    assert(g3.kind == scientific::F_SHAPE);
    let e7, e8, g4 = scientific::butter(2, [0.4, 0.1], "bandstop");
    assert(g4.kind == scientific::F_RANGE);
    let e9, g5 = scientific::lfilter([1.0], [0.0], x);
    assert(g5.kind == scientific::F_VALUE);
    let e10, g6 = scientific::filtfilt(b.data, a.data, x[..5]);
    assert(g6.kind == scientific::F_SHAPE);
}
