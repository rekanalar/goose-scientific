// The inspection and the conversion of an array.
//
// The calls asarray, asanyarray, ascontiguousarray, asfortranarray,
// astype, shape, isscalar, isrealobj, iscomplexobj, isfortran,
// conjugate and flat follow numpy. The package holds one array
// type, so the conversion calls give a copy of the array of the
// caller. A call never changes the array of the caller.
namespace scientific;

import std;

// asarray gives the array of the caller as an array. The package holds
// one array type, so the call gives a copy. A forged array gives a
// fault of kind F_SHAPE, because the calls of the package read its
// shape and its buffer.
fn asarray(a) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "asarray");
    if f.kind != F_NONE { return empty_arr(), f; }
    return clone(a), no_fault();
}

// asanyarray gives the array of the caller as an array. The package
// holds no subclass of the array, so the call is the call of asarray.
fn asanyarray(a) -> Arr, Fault { return asarray(a); }

// ascontiguousarray gives a copy of the array of the caller in the row
// major order. The buffer of the package holds the elements in that
// order, so the call gives a copy.
fn ascontiguousarray(a) -> Arr, Fault { return asarray(a); }

// asfortranarray gives a copy of the array of the caller in the
// column major order. The buffer of the package holds the elements in
// the row major order, so the call gives a copy. The order of the
// elements does not change.
fn asfortranarray(a) -> Arr, Fault { return asarray(a); }

// astype gives a copy of the array of the caller with every value
// converted. The package holds the type of f64 only, so the call gives
// a copy. A type that the package does not hold gives a fault.
fn astype(a, kind: const u8[:]) -> Arr, Fault {
    if kind != "f64" && kind != "float64" && kind != "double" {
        return empty_arr(), fault(F_VALUE, "astype", "the package holds the type of f64 only");
    }
    return asarray(a);
}

// shape gives the shape of an array as a row of whole numbers. The
// shape of an array of zero dimensions is an empty row.
fn shape(a) -> Arr, Fault {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.shape.len { d.push(a.shape[i] as f64); i += 1; }
    var s: i64[..8] = [];
    s.push(a.shape.len);
    return Arr { ndim: 1, shape: s, data: d }, no_fault();
}

// isscalar tells whether the value of the caller is a single number
// and not an array. The package holds one array type, so a value of
// the type f64 is a scalar.
fn isscalar(x: f64) -> bool {
    return true;
}

// isrealobj tells whether the array of the caller holds real numbers.
// The array of the package holds real numbers only, so the call gives
// true.
fn isrealobj(a) -> bool { return true; }

// iscomplexobj tells whether the array of the caller holds complex
// numbers. The array of the package holds real numbers only, so the
// call gives false.
fn iscomplexobj(a) -> bool { return false; }

// isfortran tells whether the elements of the array of the caller lie
// in the column major order. The buffer of the package holds the
// elements in the row major order, so the call gives false for an
// array of more than one dimension. An array of one dimension and an
// array of zero dimensions give true, as in NumPy.
fn isfortran(a) -> bool {
    return a.ndim < 2;
}

// conjugate gives the conjugate of every value of the array of the
// caller. The values of the package are real, so the conjugate of a
// value is the value.
fn conjugate(a) -> Arr, Fault { return asarray(a); }

// flat gives the elements of the array of the caller as a row. The row
// holds the elements in the row major order.
fn flat(a) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "flat");
    if f.kind != F_NONE { return empty_arr(), f; }
    return ravel(a);
}
