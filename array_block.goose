// The grid of the blocks.
//
// A grid of blocks stands in a builder of the type Blocks. The user
// declares the grid and places the blocks one at a time. One flat buffer
// holds the records and the values of the blocks.
//
// A record holds twelve values: the row of the block, the column of the
// block, the count of the dimensions, the count of the values of the
// block, and the length of the eight dimensions. The values of the block
// follow the record in the buffer.
//
// The blocks of a row follow each other along the last axis, and the
// rows follow each other along the axis before the last. A grid of
// blocks of one dimension and more than one row gives one more axis at
// the front.
namespace scientific;

// The limit of the count of the blocks of a grid.
const MAX_BLOCKS: i64 = 4096;

// The count of the values of a record of the builder.
const BLOCK_RECORD: i64 = 12;

// Blocks holds a grid of blocks.
struct Blocks { rows: i64, cols: i64, count: i64, buf: f64[>..] }

// blocks_empty gives a grid of no block.
fn blocks_empty() -> Blocks {
    return Blocks { rows: 0, cols: 0, count: 0, buf: [] };
}

// block_new gives a builder of a grid. A side below one is a fault, and
// a grid above the limit of the blocks is a fault.
fn block_new(rows: i64, cols: i64) -> Blocks, Fault {
    if rows < 1 || cols < 1 {
        return blocks_empty(), fault(F_RANGE, "block_new", "a side of the grid lies below one");
    }
    if rows > MAX_BLOCKS || cols > MAX_BLOCKS || rows * cols > MAX_BLOCKS {
        return blocks_empty(), fault(F_LIMIT, "block_new", "the grid holds more blocks than the limit");
    }
    return Blocks { rows: rows, cols: cols, count: 0, buf: [] }, no_fault();
}

// block_field reads one value of a record. The value must be a whole
// number that lies inside the given bounds. A forged record gives a
// fault and not a crash of the program.
fn block_field(b, off: i64, at: i64, low: f64, high: f64, tag: const u8[:]) -> f64, Fault {
    let v = b.buf[off + at];
    if !whole_place(v) { return 0.0, fault(F_VALUE, tag, "a value of a record is not a whole number"); }
    if v < low || v > high { return 0.0, fault(F_RANGE, tag, "a value of a record lies outside its bounds"); }
    return v, no_fault();
}

// block_scan gives the place of the record of every place of the grid.
// The table holds one place for each place of the grid, and a place
// that holds no block gives minus one.
//
// The scan reads the records of the buffer, so it checks them first. A
// record must lie inside the buffer, its place must lie inside the
// grid, and its shape must give the count of its values. A forged
// builder gives a fault and not a crash of the program.
fn block_scan(b, tag: const u8[:]) -> i64[>..], Fault {
    if b.rows < 1 || b.cols < 1 {
        let none: i64[>..] = [];
        return none, fault(F_RANGE, tag, "a side of the grid lies below one");
    }
    if b.rows > MAX_BLOCKS || b.cols > MAX_BLOCKS || b.rows * b.cols > MAX_BLOCKS {
        let none: i64[>..] = [];
        return none, fault(F_LIMIT, tag, "the grid holds more blocks than the limit");
    }
    if b.count < 0 || b.count > b.rows * b.cols {
        let none: i64[>..] = [];
        return none, fault(F_SHAPE, tag, "the count of the blocks does not fit the grid");
    }
    var t: i64[>..] = [];
    var i: i64 = 0;
    while i < b.rows * b.cols { t.push(-1); i += 1; }
    var off: i64 = 0;
    i = 0;
    while i < b.count {
        // The record must lie inside the buffer.
        if off < 0 || off + BLOCK_RECORD > b.buf.len {
            return t, fault(F_SHAPE, tag, "a record of the grid lies outside the buffer");
        }
        let rowv, fr = block_field(b, off, 0, 0.0, (b.rows - 1) as f64, tag);
        if fr.kind != F_NONE { return t, fr; }
        let colv, fc = block_field(b, off, 1, 0.0, (b.cols - 1) as f64, tag);
        if fc.kind != F_NONE { return t, fc; }
        let row = rowv as i64;
        let col = colv as i64;
        // A place may hold one block only. The table holds the place of
        // the record of every block that the grid holds.
        if t[row * b.cols + col] >= 0 {
            return t, fault(F_VALUE, tag, "a place of the grid holds two blocks");
        }
        let ndv, fd = block_field(b, off, 2, 1.0, MAX_DIMS as f64, tag);
        if fd.kind != F_NONE { return t, fd; }
        let nd = ndv as i64;
        let nvv, fv = block_field(b, off, 3, 0.0, MAX_ELEMENTS as f64, tag);
        if fv.kind != F_NONE { return t, fv; }
        let nvals = nvv as i64;
        // The shape of the block must give the count of its values.
        var prod: i64 = 1;
        var k: i64 = 0;
        while k < nd {
            let axv, fa = block_field(b, off, 4 + k, 1.0, MAX_ELEMENTS as f64, tag);
            if fa.kind != F_NONE { return t, fa; }
            let ax = axv as i64;
            if prod > MAX_ELEMENTS / ax {
                return t, fault(F_LIMIT, tag, "a block holds more elements than the limit");
            }
            prod = prod * ax;
            k += 1;
        }
        if prod != nvals {
            return t, fault(F_SHAPE, tag, "the shape of a block does not give its count of values");
        }
        // The values of the block must lie inside the buffer.
        if off + BLOCK_RECORD + nvals > b.buf.len {
            return t, fault(F_SHAPE, tag, "the values of a block lie outside the buffer");
        }
        t[row * b.cols + col] = off;
        off = off + BLOCK_RECORD + nvals;
        i += 1;
    }
    // The buffer holds the records and nothing else.
    if off != b.buf.len {
        return t, fault(F_SHAPE, tag, "the buffer of the grid holds values beyond its records");
    }
    return t, no_fault();
}

// block_axis gives the length of the axis of the block at the place. A
// place that lies outside the buffer gives zero, so a forged builder
// never reads outside its buffer.
fn block_axis(b, off: i64, axis: i64) -> i64 {
    if off < 0 || axis < 0 || off + 4 + axis >= b.buf.len { return 0; }
    return b.buf[off + 4 + axis] as i64;
}

// block_count gives the count of the dimensions of the block at the
// place. A place that lies outside the buffer gives zero.
fn block_ndim(b, off: i64) -> i64 {
    if off < 0 || off + 2 >= b.buf.len { return 0; }
    return b.buf[off + 2] as i64;
}

// block_put places a block in the grid. The place must lie inside the
// grid, and it must hold no block yet.
fn block_put(b: Blocks&, row: i64, col: i64, a) -> Fault {
    if row < 0 || row >= b.rows || col < 0 || col >= b.cols {
        return fault(F_RANGE, "block_put", "the place of the block lies outside the grid");
    }
    let t, ft = block_scan(b, "block_put");
    if ft.kind != F_NONE { return ft; }
    if t[row * b.cols + col] >= 0 {
        return fault(F_VALUE, "block_put", "the place of the block holds a block already");
    }
    if a.ndim < 1 || a.ndim > MAX_DIMS {
        return fault(F_SHAPE, "block_put", "the block holds a count of dimensions that is not right");
    }
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "block_put");
    if f.kind != F_NONE { return fault(f.kind, "block_put", f.detail); }
    // The buffer of the builder may not grow above the limit.
    if b.buf.len + BLOCK_RECORD + a.data.len > MAX_ELEMENTS {
        return fault(F_LIMIT, "block_put", "the buffer of the builder holds more values than the limit");
    }
    // The record of the block.
    b.buf.push(row as f64);
    b.buf.push(col as f64);
    b.buf.push(a.ndim as f64);
    b.buf.push(a.data.len as f64);
    var k: i64 = 0;
    while k < MAX_DIMS {
        if k < a.ndim { b.buf.push(a.shape[k] as f64); } else { b.buf.push(0.0); }
        k += 1;
    }
    // The values of the block.
    var i: i64 = 0;
    while i < a.data.len { b.buf.push(a.data[i]); i += 1; }
    b.count += 1;
    return no_fault();
}

// block_width gives the width of a row of the grid. It is the sum of the
// lengths of the last axis of the blocks of the row. The table gives
// the place of every block.
fn block_width(b, t: i64[:], row: i64) -> i64 {
    if row < 0 || row >= b.rows { return 0; }
    if t.len < b.rows * b.cols { return 0; }
    var w: i64 = 0;
    var c: i64 = 0;
    while c < b.cols {
        let off = t[row * b.cols + c];
        w += block_axis(b, off, block_ndim(b, off) - 1);
        c += 1;
    }
    return w;
}

// block_build assembles the grid into one array. Every place of the
// grid must hold a block, and the blocks must agree on their shape as
// NumPy does.
fn block_build(b) -> Arr, Fault {
    if b.count != b.rows * b.cols {
        return empty_arr(), fault(F_SHAPE, "block_build", "the grid holds a place without a block");
    }
    let t, ft = block_scan(b, "block_build");
    if ft.kind != F_NONE { return empty_arr(), ft; }
    let first = t[0];
    let d = block_ndim(b, first);
    // Every block must hold the same count of dimensions.
    var i: i64 = 0;
    while i < b.rows * b.cols {
        if block_ndim(b, t[i]) != d {
            return empty_arr(), fault(F_SHAPE, "block_build", "the blocks hold a different count of dimensions");
        }
        i += 1;
    }
    // The blocks of a row must agree on the axes before the last. The
    // rows may differ on the axis that grows, because the rows follow
    // each other along it.
    var r: i64 = 0;
    while r < b.rows {
        let row0 = t[r * b.cols];
        var c: i64 = 0;
        while c < b.cols {
            let off = t[r * b.cols + c];
            var k: i64 = 0;
            while k < d - 1 {
                if block_axis(b, off, k) != block_axis(b, row0, k) {
                    return empty_arr(), fault(F_SHAPE, "block_build", "the blocks of a row do not agree on the length of an axis");
                }
                k += 1;
            }
            c += 1;
        }
        r += 1;
    }
    // The rows must agree on the axes before the one that grows.
    r = 0;
    while r < b.rows {
        let row0 = t[r * b.cols];
        var k: i64 = 0;
        while k < d - 2 {
            if block_axis(b, row0, k) != block_axis(b, first, k) {
                return empty_arr(), fault(F_SHAPE, "block_build", "the rows do not agree on the length of an axis");
            }
            k += 1;
        }
        r += 1;
    }
    let width = block_width(b, t, 0);
    r = 0;
    while r < b.rows {
        if block_width(b, t, r) != width {
            return empty_arr(), fault(F_SHAPE, "block_build", "the rows do not agree on the width of the last axis");
        }
        r += 1;
    }
    // The shape of the result. The rows grow the axis before the last.
    var s: i64[..8] = [];
    if d == 1 {
        s.push(b.rows);
        s.push(width);
    } else {
        var k: i64 = 0;
        while k < d - 2 { s.push(block_axis(b, first, k)); k += 1; }
        var tall: i64 = 0;
        r = 0;
        while r < b.rows {
            let off = t[r * b.cols];
            tall += block_axis(b, off, d - 2);
            r += 1;
        }
        s.push(tall);
        s.push(width);
    }
    // The count of the values of the result must not pass the limit.
    var total: i64 = 1;
    var q: i64 = 0;
    while q < s.len {
        if total > MAX_ELEMENTS / s[q] {
            return empty_arr(), fault(F_LIMIT, "block_build", "the result holds more elements than the limit");
        }
        total = total * s[q];
        q += 1;
    }
    // The values of the result. The blocks of a row follow each other
    // along the last axis, and the rows follow each other along the
    // axis before the last. The axes before that come first.
    var d2: f64[>..] = [];
    // The count of the axes before the axis that grows. The blocks of
    // the grid agree on them.
    var pre: i64 = 1;
    var k: i64 = 0;
    while k < d - 2 { pre *= block_axis(b, first, k); k += 1; }
    var l: i64 = 0;
    while l < pre {
        r = 0;
        while r < b.rows {
            let at0 = t[r * b.cols];
            // The count of the lines of the blocks of the row. The
            // axis before the last gives them.
            var lines: i64 = 1;
            if d >= 2 { lines = block_axis(b, at0, d - 2); }
            var line: i64 = 0;
            while line < lines {
                var c: i64 = 0;
                while c < b.cols {
                    let off = t[r * b.cols + c];
                    let last = block_axis(b, off, d - 1);
                    let start = off + BLOCK_RECORD;
                    // The place of the line in the values of the block.
                    let at = (l * lines + line) * last;
                    var q: i64 = 0;
                    while q < last { d2.push(b.buf[start + at + q]); q += 1; }
                    c += 1;
                }
                line += 1;
            }
            r += 1;
        }
        l += 1;
    }
    return Arr { ndim: s.len, shape: s, data: d2 }, no_fault();
}
