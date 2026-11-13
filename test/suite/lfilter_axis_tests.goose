// The tests of the group lfilter_axis. Run them with SCIENTIFIC_SUITES=lfilter_axis.
namespace lfilter_axis_tests;

import scientific from "../..";
import os;
import common;

fn test_lfilter_axis() {
    var xs: f64[>..] = [];
    var i: i64 = 0;
    while i < 30 {
        let ii = i / 15;
        let jj = (i / 5) % 3;
        xs.push(sin(0.5 * i) + 0.1 * (ii + jj));
        i += 1;
    }
    let x, xf = scientific::reshape(common::av(xs), [2, 3, 5]);
    assert(xf.kind == scientific::F_NONE);
    let b = [0.2, 0.3, 0.1];
    let a = [1.0, -0.5, 0.25];
    let ym1, fym1 = scientific::lfilter_axis(b, a, x, -1);
    assert(fym1.kind == scientific::F_NONE && ym1.shape[0] == 2 && ym1.shape[1] == 3 && ym1.shape[2] == 5 && common::lf_close(ym1, 4, [0.0, 0.8032108646738374, -0.15393198451986784, -0.40783029853372654, 0.633071640034572, -0.06880422217787395, -0.5372843435679575, 0.8855189535325028]));
    let y1, fy1 = scientific::lfilter_axis(b, a, x, 1);
    assert(fy1.kind == scientific::F_NONE && y1.shape[0] == 2 && y1.shape[1] == 3 && y1.shape[2] == 5 && common::lf_close(y1, 4, [0.0, 0.18185948536513635, 0.26763749558003613, 0.09417135548634104, 0.21787164932467637, 0.3463957685320216, -0.05737503178481071, 0.11597022257404183]));
    let y0, fy0 = scientific::lfilter_axis(b, a, x, 0);
    assert(fy0.kind == scientific::F_NONE && y0.shape[0] == 2 && y0.shape[1] == 3 && y0.shape[2] == 5 && common::lf_close(y0, 4, [0.0, 0.18185948536513635, -0.13136049906158564, -0.01588309963978517, 0.40964186476635756, 0.21058463546370865, -0.41832663066612585, 0.4241694663741003]));
    let yfir, ffir = scientific::lfilter_axis([0.5, 0.5], [1.0], x, 1);
    assert(ffir.kind == scientific::F_NONE && common::lf_close(yfir, 4, [0.0, 0.45464871341284085, 0.1703462456480631, -0.16509936294427283, 0.5446791233116909, 0.3469894329426846, -0.1558620192311221, 0.307577590503221]));
    var z0: f64[>..] = [];
    i = 0;
    while i < 12 {
        let ii = i / 6;
        let jj = (i / 2) % 3;
        let kk = i % 2;
        z0.push(0.05 * (ii + 1) * (jj + 1) - 0.02 * kk);
        i += 1;
    }
    let zi, zif = scientific::reshape(common::av(z0), [2, 3, 2]);
    assert(zif.kind == scientific::F_NONE);
    let ya, za, fa = scientific::lfilter_state(b, a, x, zi, -1);
    assert(fa.kind == scientific::F_NONE && za.shape[2] == 2 && common::lf_close(ya, 4, [0.05, 0.7963358646738372, -0.16643198451986785, -0.3428302985337265, 0.763071640034572, 0.13119577782212605, -0.5722843435679577, 0.8480189535325027]) && common::lf_close(za, 1, [0.6082832150712735, -0.10815422348589113, -0.5603655064915034, 0.04876409712906449, 0.485457648916239, 0.006380123366116486, 0.06852593933592299, -0.09589583841794616, -0.29000651832217245, 0.10941379409194592, 0.7833936792482743, -0.12444507714978573]));
    var z1: f64[>..] = [];
    i = 0;
    while i < 20 {
        let ii = i / 10;
        let jj = (i / 5) % 2;
        let kk = i % 5;
        z1.push(0.03 * (ii + 1) + 0.01 * jj * (kk + 1));
        i += 1;
    }
    let zj, zjf = scientific::reshape(common::av(z1), [2, 2, 5]);
    assert(zjf.kind == scientific::F_NONE);
    let yb, zb, fb = scientific::lfilter_state(b, a, x, zj, 1);
    assert(fb.kind == scientific::F_NONE && zb.shape[1] == 2 && common::lf_close(yb, 4, [0.03, 0.21185948536513635, 0.35263749558003615, 0.12417135548634105, 0.27787164932467634, 0.4463957685320217, 0.0826249682151893, 0.16597022257404187]) && common::lf_close(zb, 2, [-0.13270167383926623, -0.07717523158177786, 0.1423945457418834, -0.085603113247269, 0.015342812272430197, 0.02591596275654577, 0.25647839679855977, 0.4029360146982192, 0.025893037424537683, 0.08756817992597658]));
    // The same call on one line matches the plain filter.
    let one, oneff = scientific::lfilter_axis(b, a, common::av([1.0, 2.0, 0.5, -1.0]));
    let plain, pf = scientific::lfilter(b, a, [1.0, 2.0, 0.5, -1.0]);
    assert(oneff.kind == scientific::F_NONE && pf.kind == scientific::F_NONE && common::all_near(one, plain.data));
    // Faults.
    let e1, g1 = scientific::lfilter_axis(b, a, x, 3);
    assert(g1.kind == scientific::F_RANGE);
    let e2, g2 = scientific::lfilter_axis(b, [0.0, 1.0], x);
    assert(g2.kind == scientific::F_VALUE);
    let e3, e4, g3 = scientific::lfilter_state(b, a, x, zj, -1);
    assert(g3.kind == scientific::F_SHAPE);
    let e5, e6, g4 = scientific::lfilter_state(b, a, x, zi, 5);
    assert(g4.kind == scientific::F_RANGE);
}
