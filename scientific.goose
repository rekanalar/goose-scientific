// scientific - numeric arrays and the functions that work on them.
//
// The package holds the N-dimensional array of floating point numbers,
// the operations that broadcast over it, and the whole-array
// reductions. It uses the standard library only. The array stores its
// elements in one flat buffer in row-major order.
//
// An operation builds its result in the value of the caller: a call
// like let c, f = add(a, b); fills c. A read of an array takes an
// untyped parameter, which binds the array by reference and copies
// nothing. Assignment binds a name, it does not copy: use clone for an
// independent array.
//
// docs/packages.md section 4: a package is a directory with an entry
// file named after the package, and its files declare the package as
// their namespace. Import/namespace names use underscores: import
// scientific from "rekanalar/goose-scientific@v0.1.0";
namespace scientific;

import math;
import array_create;
import array_shape;
import array_ops;
import array_logic;
import array_nan;
import array_order;
import array_sets;
import array_stacks;
import array_index;
import array_sort;
import array_linalg;
import array_solve;
import array_decomp;
import array_eig;
import array_svd;
import array_poly;
import array_fft;
import array_random;
import array_optimize;
import array_minimize;
import array_integrate;
import array_interpolate;
import array_signal;
import array_grid;
import array_buffer;
import array_fftn;
import array_moves;
import constants;
import array_constants;

// The kinds of a fault. A fault holds a kind, the name of the check
// and a short detail.
const F_NONE: i32 = 0;
const F_SHAPE: i32 = 1;
const F_RANGE: i32 = 2;
const F_LIMIT: i32 = 3;
const F_VALUE: i32 = 4;

struct Fault {
    kind: i32,
    tag: const u8[:],
    detail: const u8[:],
}

// Arr holds one array: the count of dimensions, the length of each
// dimension, and the elements in row-major order. The shape holds at
// most MAX_DIMS dimensions. Every length is at least 1.
struct Arr {
    ndim: i64,
    shape: i64[..8],
    data: f64[>..],
}

// The limits of the package. MAX_DIMS bounds the shape. MAX_ELEMENTS
// bounds the count of elements of one array.
const MAX_DIMS: i64 = 8;
const MAX_ELEMENTS: i64 = 268435456;

fn no_fault() -> Fault {
    return Fault { kind: F_NONE, tag: "", detail: "" };
}

fn fault(kind: i32, tag: const u8[:], detail: const u8[:]) -> Fault {
    return Fault { kind: kind, tag: tag, detail: detail };
}

// empty_arr is the result that every fault path returns.
fn empty_arr() -> Arr {
    return Arr { ndim: 0, shape: [], data: [] };
}

// kind_name names a fault kind for a test or a log line.
fn kind_name(kind: i32) -> const u8[:] {
    if kind == F_NONE { return "none"; }
    if kind == F_SHAPE { return "shape"; }
    if kind == F_RANGE { return "range"; }
    if kind == F_LIMIT { return "limit"; }
    if kind == F_VALUE { return "value"; }
    return "unknown";
}

// numel gives the count of elements of a shape. A shape longer than
// MAX_DIMS gives -1.
fn numel(shape: i64[:]) -> i64 {
    if shape.len > MAX_DIMS { return -1; }
    var n: i64 = 1;
    var i: i64 = 0;
    while i < shape.len { n *= shape[i]; i += 1; }
    return n;
}

// size gives the count of elements of an array.
fn size(a) -> i64 {
    return a.data.len;
}

// check_shape holds the rules of every shape: at most MAX_DIMS
// dimensions, each at least 1, and at most MAX_ELEMENTS elements.
fn check_shape(shape: i64[:]) -> Fault {
    if shape.len == 0 {
        return fault(F_VALUE, "shape", "the shape holds no dimension");
    }
    if shape.len > MAX_DIMS {
        return fault(F_LIMIT, "shape", "the shape holds more dimensions than the limit");
    }
    var n: i64 = 1;
    var i: i64 = 0;
    while i < shape.len {
        if shape[i] < 1 {
            return fault(F_VALUE, "shape", "a dimension holds a length below one");
        }
        n *= shape[i];
        if n > MAX_ELEMENTS {
            return fault(F_LIMIT, "shape", "the array holds more elements than the limit");
        }
        i += 1;
    }
    return no_fault();
}

// fill builds an array of a shape where every element holds one value.
fn fill(shape: i64[:], v: f64) -> Arr, Fault {
    let f = check_shape(shape);
    if f.kind != F_NONE { return empty_arr(), f; }
    let n = numel(shape);
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n { d.push(v); i += 1; }
    var s: i64[..8] = [];
    i = 0;
    while i < shape.len { s.push(shape[i]); i += 1; }
    return Arr { ndim: shape.len, shape: s, data: d }, no_fault();
}
