// The tests of the group bins. Run them with SCIENTIFIC_SUITES=bins.
namespace bins_tests;

import scientific from "../..";
import os;
import common;

// The rules of the count of bins and the weighted covariance are pinned
// against NumPy.
fn test_bins_and_weights() {
    let x = common::av([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 3.0, 4.0, 5.0, 6.0]);
    let e1, f1 = scientific::histogram_edges_rule(x, "auto");
    assert(f1.kind == scientific::F_NONE && common::all_near(e1, [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]));
    let e2, f2 = scientific::histogram_edges_rule(x, "fd");
    assert(f2.kind == scientific::F_NONE && common::all_near(e2, [1.0, 3.25, 5.5, 7.75, 10.0]));
    let e3, f3 = scientific::histogram_edges_rule(x, "sturges");
    assert(f3.kind == scientific::F_NONE && common::all_near(e3, [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]));
    let e4, f4 = scientific::histogram_edges_rule(x, "doane");
    assert(f4.kind == scientific::F_NONE && common::all_near(e4, [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]));
    let e5, f5 = scientific::histogram_edges_rule(x, "scott");
    assert(f5.kind == scientific::F_NONE && common::all_near(e5, [1.0, 4.0, 7.0, 10.0]));
    let e6, f6 = scientific::histogram_edges_rule(x, "rice");
    assert(f6.kind == scientific::F_NONE && common::all_near(e6, [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]));
    let e7, f7 = scientific::histogram_edges_rule(x, "sqrt");
    assert(f7.kind == scientific::F_NONE && common::all_near(e7, [1.0, 2.8, 4.6, 6.4, 8.2, 10.0]));
    // The rule of Stone chooses the count of bins by an estimate.
    let e8, f8 = scientific::histogram_edges_rule(x, "stone");
    assert(f8.kind == scientific::F_NONE && e8.data.len == 101);
    assert(common::near(e8.data[0], 1.0) && common::near(e8.data[1], 1.09) && common::near(e8.data[100], 10.0));
    // The counts of a histogram under a rule.
    let c1, e9, f9 = scientific::histogram_rule(x, "scott");
    assert(f9.kind == scientific::F_NONE && c1.data.len == 3 && common::all_near(c1, [5.0, 9.0, 6.0]));
    let b1, g1 = scientific::histogram_edges_rule(x, "unknown");
    assert(g1.kind == scientific::F_VALUE);
    // The weighted covariance.
    let m = common::mx(2, 5, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let fw = [1.0, 2.0, 3.0, 4.0, 5.0];
    let aw = [1.0, 2.0, 3.0, 4.0, 5.0];
    let c2, g2 = scientific::cov_weights(m, fw);
    assert(g2.kind == scientific::F_NONE && common::all_near(c2, [1.6666666666666667, 1.6666666666666667, 1.6666666666666667, 1.6666666666666665]));
    let c3, g3 = scientific::cov_weights(m, [], aw);
    assert(g3.kind == scientific::F_NONE && common::all_near(c3, [2.058823529411765, 2.058823529411765, 2.058823529411765, 2.0588235294117645]));
    let c4, g4 = scientific::cov_weights(m, fw, aw);
    assert(g4.kind == scientific::F_NONE && common::all_near(c4, [1.15, 1.15, 1.15, 1.15]));
    let c5, g5 = scientific::cov_weights(m, fw, aw, true, 0);
    assert(g5.kind == scientific::F_NONE && common::all_near(c5, [1.0644628099173554, 1.0644628099173554, 1.0644628099173554, 1.0644628099173552]));
    // A weight that does not fit the observations is a fault.
    let b2, g6 = scientific::cov_weights(m, [1.0, 2.0]);
    assert(g6.kind == scientific::F_SHAPE);
    let b3, g7 = scientific::cov_weights(m, [1.0, 2.0, 3.0, 4.0, 5.5]);
    assert(g7.kind == scientific::F_VALUE);
    return;
}
