// The tests of the group sort. Run them with SCIENTIFIC_SUITES=sort.
namespace sort_tests;

import scientific from "../..";
import os;
import common;

// The sorting, the search and the order statistics.
fn test_sort_and_search() {
    let v, f = scientific::from_values([3.0, 1.0, 2.0, 1.0]);
    // The sort gives a copy in the order of the values.
    let so, f1 = scientific::sort(v);
    assert(f1.kind == scientific::F_NONE && so.data[0] == 1.0 && so.data[1] == 1.0 && so.data[2] == 2.0 && so.data[3] == 3.0);
    assert(v.data[0] == 3.0);
    // argsort gives the flat positions that order the values.
    let ar, f2 = scientific::argsort(v);
    assert(f2.kind == scientific::F_NONE && ar.data[0] == 1.0 && ar.data[1] == 3.0 && ar.data[2] == 2.0 && ar.data[3] == 0.0);
    // A sort along one axis.
    let a = common::make23();
    let sa, f3 = scientific::sort_axis(a, 1);
    assert(f3.kind == scientific::F_NONE && sa.data[0] == 1.0 && sa.data[1] == 2.0 && sa.data[2] == 3.0);
    assert(sa.data[3] == 4.0 && sa.data[5] == 6.0);
    let sb, f4 = scientific::sort_axis(a, 5);
    assert(f4.kind == scientific::F_RANGE);
    // unique gives the values in the order of their first appearance.
    let u, f5 = scientific::unique(v);
    assert(f5.kind == scientific::F_NONE && u.data[0] == 3.0 && u.data[1] == 1.0 && u.data[2] == 2.0);
    let us, f6 = scientific::unique_sorted(v);
    assert(f6.kind == scientific::F_NONE && us.data[0] == 1.0 && us.data[1] == 2.0 && us.data[2] == 3.0);
    // searchsorted gives the place where a value fits.
    let ss, f7 = scientific::searchsorted(so, [0.5, 1.0, 2.5, 9.0]);
    assert(f7.kind == scientific::F_NONE && ss.data[0] == 0.0 && ss.data[1] == 0.0 && ss.data[2] == 3.0 && ss.data[3] == 4.0);
    // The histogram counts the values in equal bins.
    let h, e, f8 = scientific::histogram(v, 2);
    assert(f8.kind == scientific::F_NONE && h.data.len == 2 && h.data[0] == 2.0 && h.data[1] == 2.0);
    assert(e.data.len == 3 && e.data[0] == 1.0 && e.data[2] == 3.0);
    let he, f9 = scientific::histogram_edges(v, 2);
    assert(f9.kind == scientific::F_NONE && he.data.len == 3);
    let hb, he2, f10 = scientific::histogram(v, 0);
    assert(f10.kind == scientific::F_VALUE);
    let one, f11 = scientific::from_values([2.0, 2.0]);
    let hc, he3, f12 = scientific::histogram(one, 2);
    assert(f12.kind == scientific::F_VALUE);
    // bincount counts the whole numbers.
    let w, f13 = scientific::from_values([0.0, 1.0, 1.0, 3.0]);
    let bc, f14 = scientific::bincount(w, 0);
    assert(f14.kind == scientific::F_NONE && bc.data[0] == 1.0 && bc.data[1] == 2.0 && bc.data[2] == 0.0 && bc.data[3] == 1.0);
    let bd, f15 = scientific::from_values([1.5]);
    let bc2, f16 = scientific::bincount(bd, 0);
    assert(f16.kind == scientific::F_VALUE);
    // digitize gives the bin of each number.
    let dg, f17 = scientific::digitize([0.5, 1.5, 2.5], [1.0, 2.0]);
    assert(f17.kind == scientific::F_NONE && dg.data[0] == 0.0 && dg.data[1] == 1.0 && dg.data[2] == 2.0);
    let dgb, f18 = scientific::digitize([1.0], [2.0, 1.0]);
    assert(f18.kind == scientific::F_VALUE);
    // The median and the percentiles.
    let md, f19 = scientific::median(v);
    assert(f19.kind == scientific::F_NONE && md == 1.5);
    let md2, f20 = scientific::median(so);
    assert(md2 == 1.5);
    let pc, f21 = scientific::percentile(v, 50.0);
    assert(common::close(pc, 1.5));
    let pc2, f22 = scientific::percentile(v, 0.0);
    assert(pc2 == 1.0);
    let pc3, f23 = scientific::percentile(v, 100.0);
    assert(pc3 == 3.0);
    let pb, f24 = scientific::percentile(v, 101.0);
    assert(f24.kind == scientific::F_VALUE);
    // roll moves the elements along one axis.
    let rl, f25 = scientific::roll(v, 1, 0);
    assert(f25.kind == scientific::F_NONE && rl.data[0] == 1.0 && rl.data[1] == 3.0 && rl.data[3] == 2.0);
    let rl2, f26 = scientific::roll(v, -1, 0);
    assert(rl2.data[0] == 1.0 && rl2.data[1] == 2.0 && rl2.data[3] == 3.0);
    let rb, f27 = scientific::roll(v, 1, 4);
    assert(f27.kind == scientific::F_RANGE);
}
