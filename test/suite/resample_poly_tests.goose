// The tests of the group resample_poly. Run them with SCIENTIFIC_SUITES=resample_poly.
namespace resample_poly_tests;

import scientific from "../..";
import os;
import common;

// resample_poly is pinned against SciPy.
fn test_resample_poly() {
    let x = common::rp_signal();
    let a, f1 = scientific::resample_poly(x, 3, 2);
    assert(f1.kind == scientific::F_NONE && a.data.len == 30);
    assert(common::near(a.data[1], 0.2380285985652429) && common::near(a.data[5], 1.181436686153129) && common::near(a.data[15], 1.1418117248304391) && common::near(a.data[29], 0.9643440493764917));
    let b, f2 = scientific::resample_poly(x, 1, 2);
    assert(f2.kind == scientific::F_NONE && common::all_loose(b, [0.0642898273039825, 0.7568253702817194, 1.333241437659126, 1.5785041508482276, 1.4677133835228036, 1.1564010623373906, 0.7343325113364108, 0.5659863429365632, 0.5420691602674772, 1.15102267428965]));
    let c, f3 = scientific::resample_poly(x, 0, 2);
    assert(f3.kind == scientific::F_RANGE);
    return;
}
