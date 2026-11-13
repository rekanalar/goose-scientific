// The tests of the group interpn. Run them with SCIENTIFIC_SUITES=interpn.
namespace interpn_tests;

import scientific from "../..";
import os;
import common;

// interpn is pinned against SciPy.
fn test_interpn() {
    let g = [0.0, 1.0, 2.5, 4.0, -1.0, 0.5, 2.0, 0.0, 2.0];
    let v0 = common::av([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4546487134128409, 2.454648713412841, 0.7384602626041288, 2.7384602626041286, -0.35017548837401463, 1.6498245116259853, 0.32335587945721733, 5.3233558794572176, 0.5252087174427745, 5.525208717442775, -0.24905228953044706, 4.7509477104695526, -0.4089021333016356, 7.591097866698364, -0.6641566726773585, 7.3358433273226415, 0.3149409643133779, 8.314940964313378]);
    let v, vf = scientific::reshape(v0, [4, 3, 2]);
    assert(vf.kind == scientific::F_NONE);
    let pts = common::mx(6, 3, [0.3, -0.2, 0.5, 2.0, 1.0, 1.9, 4.0, 2.0, 2.0, 1.0, 0.5, 0.0, 3.3, 1.7, 1.1, 0.0, -1.0, 0.0]);
    let r1, f1 = scientific::interpn(g, [4, 3, 2], v, pts);
    assert(f1.kind == scientific::F_NONE && common::all_near(r1, [0.33180446189445834, 4.103275036393827, 8.314940964313378, 0.7384602626041288, 3.649571391891415, 0.0]));
    let r2, f2 = scientific::interpn(g, [4, 3, 2], v, pts, "nearest");
    assert(f2.kind == scientific::F_NONE && common::all_near(r2, [0.0, 5.525208717442775, 8.314940964313378, 0.7384602626041288, 8.314940964313378, 0.0]));
    let q = common::mx(8, 3, [0.3, -0.2, 0.5, 2.0, 1.0, 1.9, 4.0, 2.0, 2.0, 1.0, 0.5, 0.0, 3.3, 1.7, 1.1, 0.0, -1.0, 0.0, 5.0, 0.0, 1.0, 1.0, 3.0, 1.0]);
    let r3, f3 = scientific::interpn(g, [4, 3, 2], v, q, "linear", false, -7.0);
    assert(f3.kind == scientific::F_NONE && common::all_near(r3, [0.33180446189445834, 4.103275036393827, 8.314940964313378, 0.7384602626041288, 3.649571391891415, 0.0, -7.0, -7.0]));
    let b1, h1 = scientific::interpn(g, [4, 3, 2], v, q);
    assert(h1.kind == scientific::F_RANGE);
    let b2, h2 = scientific::interpn(g, [4, 3, 2], v, pts, "other");
    assert(h2.kind == scientific::F_VALUE);
    let b3, h3 = scientific::interpn(g, [4, 3, 3], v, pts);
    assert(h3.kind == scientific::F_SHAPE);
    let b4, h4 = scientific::interpn([0.0, 1.0, 1.0, 2.0, 3.0, 4.0], [3, 3], common::mx(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), common::mx(1, 2, [0.5, 1.5]));
    assert(h4.kind == scientific::F_VALUE);
    return;
}
