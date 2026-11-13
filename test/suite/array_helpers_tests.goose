// The tests of the group array_helpers. Run them with SCIENTIFIC_SUITES=array_helpers.
//
// The pack of the bits, the unpack of the bits, the copy into an array,
// the required form, the grid of the blocks, the roll of an axis, the
// cut at a list of places, the sort of the complex values and the
// strided array match the pins of NumPy 2.5.4.
namespace array_helpers_tests;

import scientific from "../..";
import common;
import array_helpers_pins;

// part_start gives the place of the part i of a ragged list.
fn part_start(lens: f64[:], i: i64) -> i64 {
    var at: i64 = 0;
    var k: i64 = 0;
    while k < i { at += lens[k] as i64; k += 1; }
    return at;
}

// arr_of builds an array of a shape and the values.
fn arr_of(shape: i64[:], values: f64[:]) -> scientific::Arr {
    return scientific::arr_of(scientific::copy_slice(values), shape);
}

// part_of gives the values of the part i of a ragged list.
fn part_of(d: f64[:], lens: f64[:], i: i64) -> f64[>..] {
    var v: f64[>..] = [];
    let at = part_start(lens, i);
    var k: i64 = 0;
    while k < lens[i] as i64 { v.push(d[at + k]); k += 1; }
    return v;
}

// shape_of gives the shape of the part i of a ragged list of shapes.
fn shape_of(d: f64[:], lens: f64[:], i: i64) -> i64[>..] {
    var s: i64[>..] = [];
    let at = part_start(lens, i);
    var k: i64 = 0;
    while k < lens[i] as i64 { s.push(d[at + k] as i64); k += 1; }
    return s;
}

// same_shape tells whether an array holds a shape.
fn same_shape(a: scientific::Arr&, shape: i64[:]) -> bool {
    if a.ndim != shape.len { return false; }
    var i: i64 = 0;
    while i < shape.len {
        if a.shape[i] != shape[i] { return false; }
        i += 1;
    }
    return true;
}

// order_of gives the order of the bits that the flag names.
fn order_of(flag: f64) -> const u8[:] {
    if flag == 0.0 { return "big"; }
    return "little";
}

fn test_pack_bits() {
    let lens = array_helpers_pins::PACK_IN_L;
    var i: i64 = 0;
    while i < lens.len {
        let shape = shape_of(array_helpers_pins::PACK_SHAPE_D, array_helpers_pins::PACK_SHAPE_L, i);
        let values = part_of(array_helpers_pins::PACK_IN_D, lens, i);
        let a = arr_of(shape, values);
        let axis = array_helpers_pins::PACK_AXIS[i] as i64;
        let got, f = scientific::packbits(a, axis, order_of(array_helpers_pins::PACK_ORDER[i]));
        assert(f.kind == scientific::F_NONE);
        let want = part_of(array_helpers_pins::PACK_OUT_D, array_helpers_pins::PACK_OUT_L, i);
        assert(common::all_close(got, want, 0.0));
        // The length of the axis of the result is the count of the bytes.
        let ax = scientific::fix_axis(axis, a.ndim);
        assert(got.shape[ax] == (a.shape[ax] + 7) / 8);
        i += 1;
    }
}

fn test_unpack_bits() {
    let lens = array_helpers_pins::UNPACK_IN_L;
    var i: i64 = 0;
    while i < lens.len {
        let shape = shape_of(array_helpers_pins::UNPACK_SHAPE_D, array_helpers_pins::UNPACK_SHAPE_L, i);
        let values = part_of(array_helpers_pins::UNPACK_IN_D, lens, i);
        let a = arr_of(shape, values);
        let axis = array_helpers_pins::UNPACK_AXIS[i] as i64;
        let count = array_helpers_pins::UNPACK_COUNT[i] as i64;
        let got, f = scientific::unpackbits(a, axis, count, order_of(array_helpers_pins::UNPACK_ORDER[i]));
        assert(f.kind == scientific::F_NONE);
        let want = part_of(array_helpers_pins::UNPACK_OUT_D, array_helpers_pins::UNPACK_OUT_L, i);
        assert(common::all_close(got, want, 0.0));
        i += 1;
    }
}

fn test_copy_into_array() {
    let lens = array_helpers_pins::COPY_DST_L;
    var i: i64 = 0;
    while i < lens.len {
        let dshape = shape_of(array_helpers_pins::COPY_DST_SHAPE_D, array_helpers_pins::COPY_DST_SHAPE_L, i);
        let dvalues = part_of(array_helpers_pins::COPY_DST_D, lens, i);
        var dst = arr_of(dshape, dvalues);
        let sshape = shape_of(array_helpers_pins::COPY_SRC_SHAPE_D, array_helpers_pins::COPY_SRC_SHAPE_L, i);
        let svalues = part_of(array_helpers_pins::COPY_SRC_D, array_helpers_pins::COPY_SRC_L, i);
        let src = arr_of(sshape, svalues);
        let mshape = shape_of(array_helpers_pins::COPY_MASK_SHAPE_D, array_helpers_pins::COPY_MASK_SHAPE_L, i);
        let mvalues = part_of(array_helpers_pins::COPY_MASK_D, array_helpers_pins::COPY_MASK_L, i);
        if mshape.len == 0 {
            let f = scientific::copyto(dst, src);
            assert(f.kind == scientific::F_NONE);
        } else {
            let mask = arr_of(mshape, mvalues);
            let f = scientific::copyto_where(dst, src, mask);
            assert(f.kind == scientific::F_NONE);
        }
        let want = part_of(array_helpers_pins::COPY_OUT_D, array_helpers_pins::COPY_OUT_L, i);
        assert(common::all_close(dst, want, 0.0));
        i += 1;
    }
}

fn test_block_grid() {
    var i: i64 = 0;
    var at: i64 = 0;
    while i < array_helpers_pins::BLOCK_ROWS.len {
        let rows = array_helpers_pins::BLOCK_ROWS[i] as i64;
        let cols = array_helpers_pins::BLOCK_COLS[i] as i64;
        var b, fb = scientific::block_new(rows, cols);
        assert(fb.kind == scientific::F_NONE);
        var k: i64 = 0;
        while k < rows * cols {
            let shape = shape_of(array_helpers_pins::BLOCK_SHAPE_D, array_helpers_pins::BLOCK_SHAPE_L, at);
            let values = part_of(array_helpers_pins::BLOCK_VALUE_D, array_helpers_pins::BLOCK_VALUE_L, at);
            let blk = arr_of(shape, values);
            let f = scientific::block_put(b, k / cols, k % cols, blk);
            assert(f.kind == scientific::F_NONE);
            at += 1;
            k += 1;
        }
        let got, f2 = scientific::block_build(b);
        assert(f2.kind == scientific::F_NONE);
        // The shape of the result is the shape of NumPy.
        let oshape = shape_of(array_helpers_pins::BLOCK_OUT_SHAPE_D, array_helpers_pins::BLOCK_OUT_SHAPE_L, i);
        assert(same_shape(got, oshape));
        let want = part_of(array_helpers_pins::BLOCK_OUT_D, array_helpers_pins::BLOCK_OUT_L, i);
        assert(common::all_close(got, want, 0.0));
        i += 1;
    }
}

fn test_roll_axis() {
    var i: i64 = 0;
    while i < array_helpers_pins::ROLL_AXIS.len {
        let shape = shape_of(array_helpers_pins::ROLL_SHAPE_D, array_helpers_pins::ROLL_SHAPE_L, i);
        var values: f64[>..] = [];
        var k: i64 = 0;
        while k < scientific::numel(shape) { values.push(k as f64); k += 1; }
        let a = arr_of(shape, values);
        let axis = array_helpers_pins::ROLL_AXIS[i] as i64;
        let start = array_helpers_pins::ROLL_START[i] as i64;
        let got, f = scientific::rollaxis(a, axis, start);
        assert(f.kind == scientific::F_NONE);
        let wshape = shape_of(array_helpers_pins::ROLL_OUT_SHAPE_D, array_helpers_pins::ROLL_OUT_SHAPE_L, i);
        assert(same_shape(got, wshape));
        let want = part_of(array_helpers_pins::ROLL_OUT_D, array_helpers_pins::ROLL_OUT_L, i);
        assert(common::all_close(got, want, 0.0));
        i += 1;
    }
}

fn test_split_sections() {
    var i: i64 = 0;
    var at: i64 = 0;
    while i < array_helpers_pins::CUT_AXIS.len {
        let shape = shape_of(array_helpers_pins::CUT_SHAPE_D, array_helpers_pins::CUT_SHAPE_L, i);
        var values: f64[>..] = [];
        var k: i64 = 0;
        while k < scientific::numel(shape) { values.push(k as f64); k += 1; }
        let a = arr_of(shape, values);
        let places = shape_of(array_helpers_pins::CUT_PLACES_D, array_helpers_pins::CUT_PLACES_L, i);
        let axis = array_helpers_pins::CUT_AXIS[i] as i64;
        let p, fp = scientific::split(a, places, axis);
        assert(fp.kind == scientific::F_NONE);
        assert(p.count == places.len + 1);
        var j: i64 = 0;
        while j < p.count {
            let piece = scientific::part(p, j);
            let want = part_of(array_helpers_pins::CUT_OUT_D, array_helpers_pins::CUT_OUT_L, at);
            assert(common::all_close(piece, want, 0.0));
            at += 1;
            j += 1;
        }
        i += 1;
    }
}

fn test_sort_complex() {
    var i: i64 = 0;
    while i < array_helpers_pins::SORT_IN_L.len {
        let pairs = part_of(array_helpers_pins::SORT_IN_D, array_helpers_pins::SORT_IN_L, i);
        var re: f64[>..] = [];
        var im: f64[>..] = [];
        var k: i64 = 0;
        while k < pairs.len {
            re.push(pairs[k]);
            im.push(pairs[k + 1]);
            k += 2;
        }
        let c, f = scientific::complex_from_values(re, im);
        assert(f.kind == scientific::F_NONE);
        let got, f2 = scientific::sort_complex(c);
        assert(f2.kind == scientific::F_NONE);
        let want = part_of(array_helpers_pins::SORT_OUT_D, array_helpers_pins::SORT_OUT_L, i);
        assert(got.data.len == want.len);
        var j: i64 = 0;
        while j < want.len {
            assert(got.data[j] == want[j]);
            j += 1;
        }
        i += 1;
    }
}

fn test_strided_array() {
    var i: i64 = 0;
    while i < array_helpers_pins::STRIDE_SHAPE_L.len {
        let buf = part_of(array_helpers_pins::STRIDE_BUFFER_D, array_helpers_pins::STRIDE_BUFFER_L, i);
        var one: i64[>..] = [];
        one.push(buf.len);
        let a = arr_of(one, buf);
        let shape = shape_of(array_helpers_pins::STRIDE_SHAPE_D, array_helpers_pins::STRIDE_SHAPE_L, i);
        let steps = shape_of(array_helpers_pins::STRIDE_STEP_D, array_helpers_pins::STRIDE_STEP_L, i);
        let got, f = scientific::as_strided(a, shape, steps);
        assert(f.kind == scientific::F_NONE);
        assert(same_shape(got, shape));
        let want = part_of(array_helpers_pins::STRIDE_OUT_D, array_helpers_pins::STRIDE_OUT_L, i);
        assert(common::all_close(got, want, 0.0));
        i += 1;
    }
}

fn test_required_form() {
    let a, f = scientific::from_values([1.0, 2.0, 3.0]);
    assert(f.kind == scientific::F_NONE);
    let r, f2 = scientific::require(a, "float64", "C");
    assert(f2.kind == scientific::F_NONE);
    assert(r.data.len == 3 && r.data[0] == 1.0 && r.data[2] == 3.0);
    assert(r.ndim == 1 && r.shape[0] == 3);
    // The result owns its buffer, so a write into it does not reach the
    // source.
    var w = r;
    w.data[0] = 9.0;
    assert(a.data[0] == 1.0);
    // The empty type and the empty order hold.
    let r2, f3 = scientific::require(a);
    assert(f3.kind == scientific::F_NONE && r2.data.len == 3);
    // The order A, E, O and W hold, because the copy owns its buffer.
    let r3, f4 = scientific::require(a, "f64", "A");
    assert(f4.kind == scientific::F_NONE && r3.data.len == 3);
}

fn test_bits_faults() {
    let a, f = scientific::from_values([1.0, 0.0, 1.0]);
    assert(f.kind == scientific::F_NONE);
    // An axis outside the shape is a fault.
    let _x, f1 = scientific::packbits(a, 5);
    assert(f1.kind == scientific::F_RANGE);
    let _y, f2 = scientific::unpackbits(a, -3);
    assert(f2.kind == scientific::F_RANGE);
    // An order that does not exist is a fault.
    let _z, f3 = scientific::packbits(a, -1, "middle");
    assert(f3.kind == scientific::F_VALUE);
    // A count below minus one is a fault.
    let _w, f4 = scientific::unpackbits(a, -1, -2);
    assert(f4.kind == scientific::F_RANGE);
    // A byte that is not a whole number from 0 to 255 is a fault.
    let b, f5 = scientific::from_values([1.5]);
    assert(f5.kind == scientific::F_NONE);
    let _v, f6 = scientific::unpackbits(b, -1);
    assert(f6.kind == scientific::F_VALUE);
    let c, f7 = scientific::from_values([256.0]);
    assert(f7.kind == scientific::F_NONE);
    let _u, f8 = scientific::unpackbits(c, -1);
    assert(f8.kind == scientific::F_VALUE);
    let d, f9 = scientific::from_values([-1.0]);
    assert(f9.kind == scientific::F_NONE);
    let _t, f10 = scientific::unpackbits(d, -1);
    assert(f10.kind == scientific::F_VALUE);
}

fn test_copy_faults() {
    var dst, f0 = scientific::from_values([0.0, 0.0, 0.0, 0.0]);
    assert(f0.kind == scientific::F_NONE);
    let src, f1 = scientific::from_values([1.0, 2.0]);
    assert(f1.kind == scientific::F_NONE);
    // A source that does not fit the shape of the destination is a fault.
    let f2 = scientific::copyto(dst, src);
    assert(f2.kind == scientific::F_SHAPE);
    // A mask of another shape is a fault.
    let mask, f3 = scientific::from_values([1.0, 0.0]);
    assert(f3.kind == scientific::F_NONE);
    let f4 = scientific::copyto_where(dst, src, mask);
    assert(f4.kind == scientific::F_SHAPE);
    // A type that the package does not hold is a fault.
    let _x, f5 = scientific::require(dst, "int64");
    assert(f5.kind == scientific::F_VALUE);
    // The order F gives a fault, because the package holds one order.
    let _y, f6 = scientific::require(dst, "float64", "F");
    assert(f6.kind == scientific::F_VALUE);
    // An order that does not exist is a fault.
    let _z, f7 = scientific::require(dst, "float64", "Q");
    assert(f7.kind == scientific::F_VALUE);
}

fn test_block_faults() {
    // A side below one is a fault.
    let _a, f1 = scientific::block_new(0, 2);
    assert(f1.kind == scientific::F_RANGE);
    // A grid above the limit of the blocks is a fault.
    let _b, f2 = scientific::block_new(4096, 4096);
    assert(f2.kind == scientific::F_LIMIT);
    // A place outside the grid is a fault.
    var g, f3 = scientific::block_new(2, 2);
    assert(f3.kind == scientific::F_NONE);
    let blk, f4 = scientific::from_values([1.0, 2.0]);
    assert(f4.kind == scientific::F_NONE);
    let f5 = scientific::block_put(g, 2, 0, blk);
    assert(f5.kind == scientific::F_RANGE);
    // A place that holds a block already is a fault.
    let f6 = scientific::block_put(g, 0, 0, blk);
    assert(f6.kind == scientific::F_NONE);
    let f7 = scientific::block_put(g, 0, 0, blk);
    assert(f7.kind == scientific::F_VALUE);
    // A grid with a place without a block is a fault.
    let _x, f8 = scientific::block_build(g);
    assert(f8.kind == scientific::F_SHAPE);
    // Blocks of a row that differ in the axes before the last are a fault.
    var h, f9 = scientific::block_new(1, 2);
    assert(f9.kind == scientific::F_NONE);
    let tall = scientific::zeros([3, 2]);
    let f11 = scientific::block_put(h, 0, 0, tall);
    assert(f11.kind == scientific::F_NONE);
    let f12 = scientific::block_put(h, 0, 1, blk);
    assert(f12.kind == scientific::F_NONE);
    let _y, f13 = scientific::block_build(h);
    assert(f13.kind == scientific::F_SHAPE);
}

fn test_roll_faults() {
    let a, f = scientific::from_values([1.0, 2.0, 3.0]);
    assert(f.kind == scientific::F_NONE);
    // An axis outside the shape is a fault.
    let _x, f1 = scientific::rollaxis(a, 4);
    assert(f1.kind == scientific::F_RANGE);
    let _y, f2 = scientific::rollaxis(a, -3);
    assert(f2.kind == scientific::F_RANGE);
    // A place outside the count of the axes is a fault.
    let _z, f3 = scientific::rollaxis(a, 0, 5);
    assert(f3.kind == scientific::F_RANGE);
    let _w, f4 = scientific::rollaxis(a, 0, -4);
    assert(f4.kind == scientific::F_RANGE);
}

fn test_cut_faults() {
    let a, f = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    assert(f.kind == scientific::F_NONE);
    // A place outside the axis is a fault.
    let _x, f1 = scientific::split(a, [5], 0);
    assert(f1.kind == scientific::F_RANGE);
    let _y, f2 = scientific::split(a, [-1], 0);
    assert(f2.kind == scientific::F_RANGE);
    // A place that falls below the one before it is a fault.
    let _z, f3 = scientific::split(a, [3, 1], 0);
    assert(f3.kind == scientific::F_RANGE);
    // An axis outside the shape is a fault.
    let _w, f4 = scientific::split(a, [1], 3);
    assert(f4.kind == scientific::F_RANGE);
}

fn test_sort_and_stride_faults() {
    // A complex array that does not hold pairs is a fault.
    var c = scientific::CArr { ndim: 1, shape: [1], data: [1.0, 2.0, 3.0] };
    let _x, f1 = scientific::sort_complex(c);
    assert(f1.kind == scientific::F_SHAPE);
    // A step that is not a whole number of 8 bytes is a fault.
    let a, f2 = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    assert(f2.kind == scientific::F_NONE);
    let _y, f3 = scientific::as_strided(a, [2], [4]);
    assert(f3.kind == scientific::F_VALUE);
    // A place outside the buffer is a fault.
    let _z, f4 = scientific::as_strided(a, [3], [24]);
    assert(f4.kind == scientific::F_RANGE);
    // A step that goes back leaves the buffer, so it is a fault.
    let _w, f5 = scientific::as_strided(a, [3], [-8]);
    assert(f5.kind == scientific::F_RANGE);
    // A shape with more dimensions than the limit is a fault.
    let big: i64[>..] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
    let _v, f6 = scientific::as_strided(a, big, [8, 8, 8, 8, 8, 8, 8, 8, 8]);
    assert(f6.kind == scientific::F_LIMIT);
}

// forged builds an array whose count of dimensions does not match its
// shape. A caller that forges such an array must get a fault and not a
// crash of the program.
fn forged(values: f64[:]) -> scientific::Arr {
    var s: i64[..8] = [];
    s.push(values.len as i64);
    return scientific::Arr { ndim: 2, shape: s, data: scientific::copy_slice(values) };
}

fn test_forged_array() {
    // The pack of a forged array is a fault.
    let bad = forged([1.0, 0.0, 1.0, 1.0]);
    let _x, f1 = scientific::packbits(bad, 0);
    assert(f1.kind == scientific::F_SHAPE);
    // The unpack of a forged array is a fault.
    let _y, f2 = scientific::unpackbits(bad, 0);
    assert(f2.kind == scientific::F_SHAPE);
    // The copy into a forged array is a fault.
    var dst = forged([0.0, 0.0, 0.0, 0.0]);
    let src, f3 = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    assert(f3.kind == scientific::F_NONE);
    let f4 = scientific::copyto(dst, src);
    assert(f4.kind == scientific::F_SHAPE);
    // The copy from a forged source is a fault.
    var good, f5 = scientific::from_values([0.0, 0.0, 0.0, 0.0]);
    assert(f5.kind == scientific::F_NONE);
    let f6 = scientific::copyto(good, bad);
    assert(f6.kind == scientific::F_SHAPE);
    // The copy with a forged mask is a fault.
    let f7 = scientific::copyto_where(good, src, bad);
    assert(f7.kind == scientific::F_SHAPE);
    // The roll of a forged array is a fault.
    let _z, f8 = scientific::rollaxis(bad, 0);
    assert(f8.kind == scientific::F_SHAPE);
    // The cut of a forged array is a fault.
    let _w, f9 = scientific::split(bad, [1], 0);
    assert(f9.kind == scientific::F_SHAPE);
    // The cut of a forged array into pieces of a count is a fault.
    let _t, f12 = scientific::split(bad, 2, 0);
    assert(f12.kind == scientific::F_SHAPE);
    // The uneven cut of a forged array is a fault.
    let _u, f13 = scientific::array_split(bad, 2, 0);
    assert(f13.kind == scientific::F_SHAPE);
    // The strided array of a forged array is a fault.
    let _v, f10 = scientific::as_strided(bad, [2], [8, 8]);
    assert(f10.kind == scientific::F_SHAPE);
    // The sort of a forged complex array is a fault.
    var cs: i64[..8] = [];
    cs.push(2);
    let badc = scientific::CArr { ndim: 1, shape: cs, data: [1.0, 2.0, 3.0] };
    let _u, f11 = scientific::sort_complex(badc);
    assert(f11.kind == scientific::F_SHAPE);
}

fn test_element_limit() {
    // The unpack of a count above the limit of the elements is a fault.
    let a, f = scientific::from_values([1.0]);
    assert(f.kind == scientific::F_NONE);
    let _x, f2 = scientific::unpackbits(a, 0, 268435457);
    assert(f2.kind == scientific::F_LIMIT);
}

// forged_grid builds a grid of blocks whose buffer does not hold the
// records that the grid says. A caller that forges such a grid must
// get a fault and not a crash of the program.
fn forged_grid(rows: i64, cols: i64, count: i64, buf: f64[:]) -> scientific::Blocks {
    return scientific::Blocks { rows: rows, cols: cols, count: count, buf: scientific::copy_slice(buf) };
}

fn test_forged_grid() {
    // A grid whose buffer holds no record is a fault.
    let g1 = forged_grid(1, 1, 1, []);
    let _x, f1 = scientific::block_build(g1);
    assert(f1.kind == scientific::F_SHAPE);
    // A grid whose record lies outside the grid is a fault.
    let g2 = forged_grid(1, 1, 1, [5.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]);
    let _y, f2 = scientific::block_build(g2);
    assert(f2.kind == scientific::F_RANGE);
    // A grid whose record holds a shape that does not give its count of
    // values is a fault.
    let g3 = forged_grid(1, 1, 1, [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]);
    let _z, f3 = scientific::block_build(g3);
    assert(f3.kind == scientific::F_SHAPE);
    // A grid whose values lie outside its buffer is a fault.
    let g4 = forged_grid(1, 1, 1, [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let _w, f4 = scientific::block_build(g4);
    assert(f4.kind == scientific::F_SHAPE);
    // A grid with more sides than the limit is a fault.
    let g5 = forged_grid(8192, 1, 8192, []);
    let _v, f5 = scientific::block_build(g5);
    assert(f5.kind == scientific::F_LIMIT);
    // A block that is put in a forged grid is a fault.
    let blk, f6 = scientific::from_values([1.0]);
    assert(f6.kind == scientific::F_NONE);
    let f7 = scientific::block_put(g1, 0, 0, blk);
    assert(f7.kind == scientific::F_SHAPE);
}

fn test_forged_pieces() {
    var sh: i64[..8] = [];
    sh.push(3);
    // A set of pieces whose buffer holds no value is a fault.
    let p1 = scientific::Parts { count: 2, ndim: 1, axis: 0, shape: sh, buf: [] };
    let _x, f1 = scientific::part(p1, 0);
    assert(f1.kind == scientific::F_SHAPE);
    // A set of pieces whose axis stands outside the shape is a fault.
    let p2 = scientific::Parts { count: 1, ndim: 1, axis: 3, shape: sh, buf: [1.0, 1.0, 1.0] };
    let _y, f2 = scientific::part(p2, 0);
    assert(f2.kind == scientific::F_RANGE);
    // A set of pieces whose values lie outside the buffer is a fault.
    let p3 = scientific::Parts { count: 1, ndim: 1, axis: 0, shape: sh, buf: [3.0, 9.0] };
    let _z, f3 = scientific::part(p3, 0);
    assert(f3.kind == scientific::F_SHAPE);
    // A set of pieces whose counts are huge gives a fault. The count of
    // the pieces must not overflow.
    let big: f64[>..] = [];
    big.push(1.0);
    big.push(4503599627370495.0);
    let p4 = scientific::Parts { count: 1, ndim: 1, axis: 0, shape: sh, buf: scientific::copy_slice(big) };
    let _w, f4 = scientific::part(p4, 0);
    assert(f4.kind == scientific::F_SHAPE);
}
