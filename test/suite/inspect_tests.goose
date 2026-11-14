// The tests of the inspection and the conversion of an array. Run them
// with SCIENTIFIC_SUITES=inspect.
//
// The calls asarray, asanyarray, ascontiguousarray, asfortranarray,
// astype, shape, isscalar, isrealobj, iscomplexobj, isfortran,
// conjugate and flat follow numpy. The package holds one array
// type, so the conversion calls give a copy of the array of the
// caller.
namespace inspect_tests;

import scientific from "../..";

fn test_conversion_gives_a_copy() {
    let a, f_a = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let c, f = scientific::asarray(a);
    assert(f.kind == scientific::F_NONE);
    assert(c.data.len == 6 && c.ndim == 1);
    // A change of the copy leaves the array of the caller as it was.
    var d = c;
    d.data[0] = 99.0;
    assert(a.data[0] == 1.0);
    // The other conversion calls give a copy too.
    let c2, f2 = scientific::asanyarray(a);
    assert(f2.kind == scientific::F_NONE && c2.data.len == 6);
    let c3, f3 = scientific::ascontiguousarray(a);
    assert(f3.kind == scientific::F_NONE && c3.data[5] == 6.0);
    let c4, f4 = scientific::asfortranarray(a);
    assert(f4.kind == scientific::F_NONE && c4.data[5] == 6.0);
    let c6, f6 = scientific::conjugate(a);
    assert(f6.kind == scientific::F_NONE && c6.data[3] == 4.0);
}

fn test_astype_keeps_the_values() {
    let a, f_a = scientific::from_values([1.5, -2.5, 3.0]);
    let c, f = scientific::astype(a, "f64");
    assert(f.kind == scientific::F_NONE);
    assert(c.data.len == 3 && c.data[0] == 1.5 && c.data[1] == -2.5);
    let c2, f2 = scientific::astype(a, "float64");
    assert(f2.kind == scientific::F_NONE && c2.data.len == 3);
    // A type that the package does not hold gives a fault.
    let c3, f3 = scientific::astype(a, "int32");
    assert(f3.kind == scientific::F_VALUE && c3.data.len == 0);
}

fn test_shape_gives_the_shape() {
    let a, f_a = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let s, f = scientific::shape(a);
    assert(f.kind == scientific::F_NONE);
    assert(s.data.len == 2 && s.data[0] == 2.0 && s.data[1] == 3.0);
    let b, f_b = scientific::from_values([1.0, 2.0]);
    let s2, f2 = scientific::shape(b);
    assert(f2.kind == scientific::F_NONE && s2.data.len == 1 && s2.data[0] == 2.0);
}

fn test_the_predicates() {
    let a, f_a = scientific::from_values([1.0, 2.0]);
    assert(scientific::isscalar(1.0));
    assert(scientific::isrealobj(a));
    assert(!scientific::iscomplexobj(a));
    // An array of one dimension gives true, as in NumPy. An array of
    // two dimensions gives false, because the buffer holds the
    // elements in the row major order.
    assert(scientific::isfortran(a));
    let m, f_m = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    assert(!scientific::isfortran(m));
}

fn test_flat_gives_the_elements() {
    let a, f_a = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let f, fg = scientific::flat(a);
    assert(fg.kind == scientific::F_NONE);
    assert(f.data.len == 6 && f.ndim == 1);
    assert(f.data[0] == 1.0 && f.data[5] == 6.0);
}

// A forged array gives a fault and not a crash. The count of
// dimensions does not match the shape, and the buffer holds more
// elements than the shape holds.
fn test_a_forged_array_gives_a_fault() {
    let bad = scientific::Arr { ndim: 2, shape: [2, 3], data: [1.0, 2.0, 3.0] };
    let c, f = scientific::asarray(bad);
    assert(f.kind == scientific::F_SHAPE && c.data.len == 0);
    let c2, f2 = scientific::flat(bad);
    assert(f2.kind == scientific::F_SHAPE && c2.data.len == 0);
    let c4, f4 = scientific::ascontiguousarray(bad);
    assert(f4.kind == scientific::F_SHAPE && c4.data.len == 0);
}
