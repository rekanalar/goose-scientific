// The tests of the group global_minimizers. Run them with SCIENTIFIC_SUITES=global_minimizers.
namespace global_minimizers_tests;

import scientific from "../..";
import os;
import common;

fn test_global_minimizers() {
    var s: i64 = 1;
    while s <= 4 {
        let d, f = scientific::dual_annealing([-5.12, -5.12], [5.12, 5.12], s) { v => common::rastrigin(v) };
        assert(f.kind == scientific::F_NONE);
        assert(d.data.len == 3 && d.data[2] < 1.0e-6);
        s += 1;
    }
    let d2, f2 = scientific::dual_annealing([-5.12, -5.12], [5.12, 5.12], 3) { v => common::rastrigin(v) };
    let d3, f3 = scientific::dual_annealing([-5.12, -5.12], [5.12, 5.12], 3) { v => common::rastrigin(v) };
    assert(d2.data[0] == d3.data[0] && d2.data[2] == d3.data[2]);
    let b, g = scientific::basinhopping([2.5, 2.5], 100, 1.0, 1.0, 3) { v => common::rastrigin(v) };
    assert(g.kind == scientific::F_NONE);
    assert(b.data[2] < 1.0e-6);
    let e1, g1 = scientific::dual_annealing([1.0], [0.0]) { v => v[0] };
    assert(g1.kind == scientific::F_VALUE);
    let e2, g2 = scientific::dual_annealing([0.0, 1.0], [1.0]) { v => v[0] };
    assert(g2.kind == scientific::F_SHAPE);
    let e3, g3 = scientific::dual_annealing([0.0], [1.0], 1, 0) { v => v[0] };
    assert(g3.kind == scientific::F_RANGE);
    let e4, g4 = scientific::dual_annealing([0.0], [1.0]) { v => v[0] / 0.0 };
    assert(g4.kind == scientific::F_VALUE);
    let e5, g5 = scientific::basinhopping([], 10) { v => 0.0 };
    assert(g5.kind == scientific::F_SHAPE);
    let e6, g6 = scientific::basinhopping([1.0], 10, 1.0, 0.0) { v => v[0] };
    assert(g6.kind == scientific::F_RANGE);
    let e7, g7 = scientific::basinhopping([1.0 / 0.0], 10) { v => v[0] };
    assert(g7.kind == scientific::F_VALUE);
}
