// The tests of the group index. Run them with SCIENTIFIC_SUITES=index.
namespace index_tests;

import scientific from "../..";
import os;
import common;

// The slicing, the masks and the selection by position.
fn test_indexing() {
    let a = common::make23();
    // A slice keeps the elements between the bounds.
    let sl, f = scientific::slice_axis(a, 1, 1, 3, 1);
    assert(f.kind == scientific::F_NONE && sl.shape[1] == 2);
    assert(sl.data[0] == 2.0 && sl.data[1] == 3.0 && sl.data[2] == 5.0);
    // A slice of the rows.
    let sl2, f2 = scientific::slice_axis(a, 0, 0, 1, 1);
    assert(f2.kind == scientific::F_NONE && sl2.shape[0] == 1 && sl2.data[2] == 3.0);
    // A step of two keeps every second element.
    let sl3, f3 = scientific::slice_axis(a, 1, 0, 3, 2);
    assert(f3.kind == scientific::F_NONE && sl3.shape[1] == 2 && sl3.data[0] == 1.0 && sl3.data[1] == 3.0 && sl3.data[2] == 4.0);
    // A negative step reads the axis backwards.
    let sl4, f4 = scientific::slice_axis(a, 1, -1, -4, -1);
    assert(f4.kind == scientific::F_NONE && sl4.shape[1] == 3);
    assert(sl4.data[0] == 3.0 && sl4.data[1] == 2.0 && sl4.data[2] == 1.0);
    // A slice that holds no element is a fault.
    let bad, f5 = scientific::slice_axis(a, 1, 2, 2, 1);
    assert(f5.kind == scientific::F_SHAPE && bad.data.len == 0);
    let bad2, f6 = scientific::slice_axis(a, 1, 0, 3, 0);
    assert(f6.kind == scientific::F_VALUE);
    let bad3, f7 = scientific::slice_axis(a, 5, 0, 1, 1);
    assert(f7.kind == scientific::F_RANGE);
    // take gives the elements at the flat positions.
    let tk, f8 = scientific::take(a, [0, 5, 2]);
    assert(f8.kind == scientific::F_NONE && tk.data[0] == 1.0 && tk.data[1] == 6.0 && tk.data[2] == 3.0);
    let tk2, f9 = scientific::take(a, [6]);
    assert(f9.kind == scientific::F_RANGE && tk2.data.len == 0);
    // put writes into the elements at the flat positions.
    let b = common::make23();
    let pf = scientific::put(b, [0, 5], [9.0, 8.0]);
    assert(pf.kind == scientific::F_NONE && b.data[0] == 9.0 && b.data[5] == 8.0 && b.data[1] == 2.0);
    let b2 = common::make23();
    let pf2 = scientific::put(b2, [9], [1.0]);
    assert(pf2.kind == scientific::F_RANGE && b2.data[0] == 1.0);
    let b3 = common::make23();
    let pf3 = scientific::put(b3, [0, 1], [1.0]);
    assert(pf3.kind == scientific::F_SHAPE && b3.data[0] == 1.0);
    // nonzero gives the flat positions of the values that are not zero.
    let z, f10 = scientific::from_values([0.0, 2.0, 0.0, 4.0]);
    let nz, f11 = scientific::nonzero(z);
    assert(f11.kind == scientific::F_NONE && nz.data[0] == 1.0 && nz.data[1] == 3.0 && nz.data.len == 2);
    // a mask keeps the elements that hold true.
    let m, f12 = scientific::mask(z, z);
    assert(f12.kind == scientific::F_NONE && m.data[0] == 2.0 && m.data[1] == 4.0);
    let m2, f13 = scientific::mask(z, z);
    assert(m2.data.len == 2);
    let wrong, f14 = scientific::from_values([1.0]);
    let m3, f15 = scientific::mask(z, wrong);
    assert(f15.kind == scientific::F_SHAPE);
    // take_along_axis takes the elements at the positions of an axis.
    let idx, f16 = scientific::from_rows(2, 1, [2.0, 0.0]);
    let ta, f17 = scientific::take_along_axis(a, idx, 1);
    assert(f17.kind == scientific::F_NONE && ta.data[0] == 3.0 && ta.data[1] == 4.0);
    let badidx, f18 = scientific::from_rows(2, 1, [5.0, 0.0]);
    let ta2, f19 = scientific::take_along_axis(a, badidx, 1);
    assert(f19.kind == scientific::F_RANGE);
    // repeat repeats every element.
    let rp, f20 = scientific::repeat(a, 2);
    assert(f20.kind == scientific::F_NONE && rp.data.len == 12 && rp.data[0] == 1.0 && rp.data[1] == 1.0 && rp.data[2] == 2.0);
    let rp2, f21 = scientific::repeat(a, 0);
    assert(f21.kind == scientific::F_VALUE);
    // tile repeats the whole array along one axis.
    let tl, f22 = scientific::tile(a, 2, 0);
    assert(f22.kind == scientific::F_NONE && tl.shape[0] == 4 && tl.shape[1] == 3);
    assert(tl.data[0] == 1.0 && tl.data[3] == 4.0 && tl.data[6] == 1.0);
    let tl2, f23 = scientific::tile(a, 2, 1);
    assert(f23.kind == scientific::F_NONE && tl2.shape[1] == 6 && tl2.data[3] == 1.0);
    // flatten gives a copy in one dimension.
    let fnn2, f25 = scientific::flatten(a);
    assert(f25.kind == scientific::F_NONE && fnn2.data[5] == 6.0);
    // partition puts the values below the pivot first.
    let pv, f26 = scientific::from_values([5.0, 1.0, 4.0, 2.0]);
    let pt, f27 = scientific::partition(pv, 3.0);
    assert(f27.kind == scientific::F_NONE && pt.data[0] == 1.0 && pt.data[1] == 2.0 && pt.data[2] == 5.0 && pt.data[3] == 4.0);
}
