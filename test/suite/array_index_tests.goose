// The tests of the group array_index. Run them with SCIENTIFIC_SUITES=array_index.
//
// The index helpers match the pins of NumPy 2.5.4: diag_indices_from,
// tril_indices_from, triu_indices_from, mask_indices, ndindex,
// ndenumerate, may_share_memory, shares_memory and einsum_path.
namespace array_index_tests;

import scientific from "../..";
import array_index_pins;

// same_flat tells whether the buffer of an array holds the pins.
fn same_flat(a: scientific::Arr&, pins: f64[:]) -> bool {
    if a.data.len != pins.len { return false; }
    var i: i64 = 0;
    while i < pins.len {
        if a.data[i] != pins[i] { return false; }
        i += 1;
    }
    return true;
}

fn grid(rows: i64, cols: i64) -> scientific::Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < rows * cols { d.push(i); i += 1; }
    var s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    return scientific::Arr { ndim: 2, shape: s, data: d };
}

fn grid_shift(rows: i64, cols: i64, shift: f64) -> scientific::Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < rows * cols { d.push(i + shift); i += 1; }
    var s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    return scientific::Arr { ndim: 2, shape: s, data: d };
}

fn cube(n: i64) -> scientific::Arr {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < n * n * n { d.push(i); i += 1; }
    var s: i64[..8] = [];
    s.push(n);
    s.push(n);
    s.push(n);
    return scientific::Arr { ndim: 3, shape: s, data: d };
}

fn test_tri_from() {
    let a, f1 = scientific::tril_indices_from(grid(2, 3));
    assert(f1.kind == scientific::F_NONE && same_flat(a, array_index_pins::TRIL_23));
    let b, f2 = scientific::tril_indices_from(grid(3, 4), 1);
    assert(f2.kind == scientific::F_NONE && same_flat(b, array_index_pins::TRIL_34_K1));
    let c, f3 = scientific::triu_indices_from(grid(3, 4), 1);
    assert(f3.kind == scientific::F_NONE && same_flat(c, array_index_pins::TRIU_34_K1));
    let d, f4 = scientific::triu_indices_from(grid(4, 3), -1);
    assert(f4.kind == scientific::F_NONE && same_flat(d, array_index_pins::TRIU_43_KM1));
    let bad, f5 = scientific::tril_indices_from(cube(2));
    assert(f5.kind == scientific::F_SHAPE && bad.data.len == 0);
}

fn test_diag_from() {
    let a, f1 = scientific::diag_indices_from(cube(3));
    assert(f1.kind == scientific::F_NONE && a.shape[0] == 3 && same_flat(a, array_index_pins::DIAG_3_3));
    let rect, f2 = scientific::diag_indices_from(grid(2, 3));
    assert(f2.kind == scientific::F_SHAPE && rect.data.len == 0);
    var s: i64[..8] = [];
    s.push(3);
    let line = scientific::Arr { ndim: 1, shape: s, data: [0.0, 1.0, 2.0] };
    let one, f3 = scientific::diag_indices_from(line);
    assert(f3.kind == scientific::F_SHAPE && one.data.len == 0);
}

fn test_mask_indices() {
    let a, f1 = scientific::mask_indices(4, "triu", 1);
    assert(f1.kind == scientific::F_NONE && same_flat(a, array_index_pins::MASK_TRIU_4_K1));
    let b, f2 = scientific::mask_indices(3, "tril");
    assert(f2.kind == scientific::F_NONE && same_flat(b, array_index_pins::MASK_TRIL_3));
    let bad, f3 = scientific::mask_indices(3, "diag");
    assert(f3.kind == scientific::F_VALUE && bad.data.len == 0);
}

fn test_walks() {
    let p, f1 = scientific::ndindex([2, 3, 2]);
    assert(f1.kind == scientific::F_NONE && p.shape[0] == 12 && p.shape[1] == 3);
    assert(same_flat(p, array_index_pins::NDINDEX_232));
    let bad, f2 = scientific::ndindex([2, 0]);
    assert(f2.kind == scientific::F_VALUE && bad.data.len == 0);
    var a = grid_shift(2, 3, 5.0);
    let places, values, f3 = scientific::ndenumerate(a);
    assert(f3.kind == scientific::F_NONE && places.shape[0] == 6 && places.shape[1] == 2);
    assert(same_flat(places, array_index_pins::ENUM_PLACES));
    assert(same_flat(values, array_index_pins::ENUM_VALUES));
}

fn test_share_memory() {
    let a = grid(2, 2);
    assert(!scientific::may_share_memory(a, a));
    assert(!scientific::shares_memory(a, grid(2, 2)));
}

// path_ok tells whether a path holds the pins and the count of operations.
fn path_ok(p: scientific::Arr&, ops: f64, pins: f64[:], flops: f64[:]) -> bool {
    return same_flat(p, pins) && ops == flops[0];
}

fn test_einsum_path() {
    let p1, o1, f1 = scientific::einsum_path("ij,jk->ik", grid(2, 3), grid(3, 4));
    assert(f1.kind == scientific::F_NONE);
    assert(path_ok(p1, o1, array_index_pins::EP_IJ_JK_PATH, array_index_pins::EP_IJ_JK_FLOPS));
    let p2, o2, f2 = scientific::einsum_path3("ij,jk,kl->il", grid(2, 3), grid(3, 4), grid(4, 5));
    assert(f2.kind == scientific::F_NONE);
    assert(path_ok(p2, o2, array_index_pins::EP_CHAIN_A_PATH, array_index_pins::EP_CHAIN_A_FLOPS));
    let p3, o3, f3 = scientific::einsum_path3("ij,jk,kl->il", grid(2, 2), grid(2, 5), grid(5, 2));
    assert(f3.kind == scientific::F_NONE);
    assert(path_ok(p3, o3, array_index_pins::EP_CHAIN_B_PATH, array_index_pins::EP_CHAIN_B_FLOPS));
    let p4, o4, f4 = scientific::einsum_path1("ii->", grid(3, 3));
    assert(f4.kind == scientific::F_NONE);
    assert(path_ok(p4, o4, array_index_pins::EP_TRACE_PATH, array_index_pins::EP_TRACE_FLOPS));
    let bad, ob, f5 = scientific::einsum_path("ij,jk->ik", grid(2, 3), grid(4, 4));
    assert(f5.kind == scientific::F_SHAPE && bad.data.len == 0);
    let bad2, ob2, f6 = scientific::einsum_path("ij->ik", grid(2, 3), grid(3, 3));
    assert(f6.kind == scientific::F_VALUE && bad2.data.len == 0);
}

// A rule that holds more than eight operands gives a fault. The call
// does not build a path of that rule.
fn test_einsum_path_limit() {
    let p, o, f = scientific::einsum_path("a,b,c,d,e,f,g,h,i->a", grid(2, 2), grid(2, 2));
    assert(f.kind == scientific::F_LIMIT && p.data.len == 0);
}
