// The tests of the group eigsh. Run them with SCIENTIFIC_SUITES=eigsh.
namespace eigsh_tests;

import scientific from "../..";
import os;
import common;

// eigsh is pinned against SciPy.
fn test_eigsh() {
    let s = common::eg_matrix();
    let la, v1, f1 = scientific::eigsh(s, 3, "LA");
    assert(f1.kind == scientific::F_NONE && common::all_loose(la, [5.835442141035716, 6.132637310688383, 6.500509781468499]));
    let sa, v2, f2 = scientific::eigsh(s, 3, "SA");
    assert(f2.kind == scientific::F_NONE && common::all_loose(sa, [0.39949021853149425, 0.7673626893116112, 1.0645578589642712]));
    let lm, v3, f3 = scientific::eigsh(s, 3, "LM");
    assert(f3.kind == scientific::F_NONE && common::all_loose(lm, [5.83544214103572, 6.132637310688378, 6.500509781468498]));
    let sm, v4, f4 = scientific::eigsh(s, 3, "SM");
    assert(f4.kind == scientific::F_NONE && common::all_loose(sm, [0.3994902185314938, 0.7673626893116103, 1.064557858964272]));
    // The first vector of SA holds a x = value x.
    var col: f64[>..] = [];
    var i: i64 = 0;
    while i < 30 { col.push(v2.data[i * 3]); i += 1; }
    let ax, fx = scientific::csr_matvec(s, col);
    assert(fx.kind == scientific::F_NONE);
    i = 0;
    while i < 30 { assert(common::near(ax.data[i] + 10.0, sa.data[0] * col[i] + 10.0)); i += 1; }
    let bad, vb, fb = scientific::eigsh(s, 30);
    assert(fb.kind == scientific::F_RANGE);
    let bad2, vb2, fb2 = scientific::eigsh(s, 3, "XX");
    assert(fb2.kind == scientific::F_VALUE);
    return;
}
