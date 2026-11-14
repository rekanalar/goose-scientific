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

// asfortranarray gives a copy of the array of the caller. The buffer
// of the package holds the elements in the row major order, so the
// copy keeps that order. NumPy gives an array that isfortran accepts.
// Here isfortran gives false for every array.
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
// and not an array. A number gives true. An array gives false.
fn isscalar(x: f64) -> bool { return true; }
fn isscalar(a: Arr&) -> bool { return false; }
fn isscalar(a: CArr&) -> bool { return false; }

// isrealobj tells whether the array of the caller holds real numbers.
// An array of the type Arr gives true. A complex array gives false.
fn isrealobj(a: Arr&) -> bool { return true; }
fn isrealobj(a: CArr&) -> bool { return false; }

// iscomplexobj tells whether the array of the caller holds complex
// numbers. An array of the type Arr gives false. A complex array gives
// true.
fn iscomplexobj(a: Arr&) -> bool { return false; }
fn iscomplexobj(a: CArr&) -> bool { return true; }

// isfortran tells whether the elements of the array of the caller lie
// in the column major order and not in the row major order. The buffer
// of the package always holds the elements in the row major order, so
// the call gives false. NumPy gives false for an array of one
// dimension too, because that array lies in both orders.
fn isfortran(a: Arr&) -> bool { return false; }
fn isfortran(a: CArr&) -> bool { return false; }

// conjugate gives the conjugate of every value of the array of the
// caller. The values of an array of the type Arr are real, so the
// conjugate of a value is the value. A complex array gives the complex
// conjugate of every element.
fn conjugate(a: Arr&) -> Arr, Fault { return asarray(a); }
fn conjugate(c: CArr&) -> CArr, Fault {
    let f = check_stack(c.ndim, c.shape, c.data.len, 2, "conjugate");
    if f.kind != F_NONE { return carr_empty(), f; }
    return conj(c), no_fault();
}

// flat gives the elements of the array of the caller as a row. The row
// holds the elements in the row major order.
fn flat(a) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "flat");
    if f.kind != F_NONE { return empty_arr(), f; }
    return ravel(a);
}
