// The block matrix.
//
// A grid of blocks stands in a builder of the type Bmat. The user
// declares the grid and places the blocks one at a time. One flat buffer
// holds the records of the blocks.
//
// A record starts with five values: the grid row, the grid column, the
// rows of the block, the columns of the block and the count of the stored
// values of the block. Three values follow for each stored value: the
// row, the column and the value.
//
// The height of a grid row is the height of the first block of that row.
// A grid row that holds no block has the height zero and adds nothing.
// The width of a grid column follows the same rule.
//
namespace scientific;

import mathshim;

// The count of the values of the header of a record.
const BMAT_HEAD: i64 = 5;

// The count of the values of one stored value of a record.
const BMAT_STRIDE: i64 = 3;

// Bmat holds a grid of blocks. The field buf comes last.
struct Bmat { grows: i64, gcols: i64, n: i64, buf: f64[>..] }

fn bmat_empty() -> Bmat {
    return Bmat { grows: 0, gcols: 0, n: 0, buf: [] };
}

// bmat_check_grid checks the sides of the grid.
fn bmat_check_grid(grows: i64, gcols: i64, tag: const u8[:]) -> Fault {
    if grows < 1 || gcols < 1 || grows > MAX_BMAT_GRID || gcols > MAX_BMAT_GRID {
        return fault(F_RANGE, tag, "a side of the grid stands outside one to the limit");
    }
    return no_fault();
}

// bmat_check_size checks the size of the buffer before it grows.
fn bmat_check_size(size: i64, tag: const u8[:]) -> Fault {
    if size > MAX_ELEMENTS {
        return fault(F_LIMIT, tag, "the buffer of the builder holds more values than the limit");
    }
    return no_fault();
}

// bmat_int_at gives the value at the index at as a whole number. A value
// that is not a whole number gives zero.
fn bmat_int_at(b, at: i64) -> i64 {
    return bmat_whole_at(b, at) as i64;
}

// bmat_whole_at gives the value at the index at. A value that is not a
// whole number gives zero.
fn bmat_whole_at(b, at: i64) -> f64 {
    if at < 0 || at >= b.buf.len { return 0.0; }
    let v = b.buf[at];
    if !whole_place(v) { return 0.0; }
    return v;
}

// bmat_declared_at gives the count of the stored values that the record at
// the index at claims. A count that is not a whole number gives zero.
fn bmat_declared_at(b, at: i64) -> i64 {
    if at < 0 || at + BMAT_HEAD - 1 >= b.buf.len { return 0; }
    let v = b.buf[at + BMAT_HEAD - 1];
    if !whole_place(v) || v < 0.0 { return 0; }
    return v as i64;
}

// bmat_count_at gives the count of the stored values of the record at the
// index at. The count never exceeds the room that the rest of the buffer
// holds. Every walk of the records uses this count, so a record that lies
// about its count never moves the walk to another place.
fn bmat_count_at(b, at: i64) -> i64 {
    if at < 0 || at + BMAT_HEAD > b.buf.len { return 0; }
    let room = (b.buf.len - at - BMAT_HEAD) / BMAT_STRIDE;
    let cnt = bmat_declared_at(b, at);
    if cnt > room { return room; }
    return cnt;
}

// bmat_records gives the index of the start of every record of the builder.
// Every walk of the records uses this list, so that all of them read the
// same records and the same places.
fn bmat_records(b) -> i64[>..] {
    var out: i64[>..] = [];
    var at: i64 = 0;
    var k: i64 = 0;
    while k < b.n {
        if at + BMAT_HEAD - 1 >= b.buf.len { break; }
        out.push(at);
        at += BMAT_HEAD + BMAT_STRIDE * bmat_count_at(b, at);
        k += 1;
    }
    return out;
}

// bmat_place walks the records of the builder. It gives whether the place
// holds a block, the height of the grid row and the width of the grid
// column.
fn bmat_place(b, recs: i64[>..]&, r: i64, c: i64) -> bool, i64, i64 {
    var taken = false;
    var h: i64 = 0;
    var w: i64 = 0;
    var k: i64 = 0;
    while k < recs.len {
        let at = recs[k];
        let gr = bmat_int_at(b, at);
        let gc = bmat_int_at(b, at + 1);
        if gr == r && gc == c { taken = true; }
        if gr == r && h == 0 { h = bmat_int_at(b, at + 2); }
        if gc == c && w == 0 { w = bmat_int_at(b, at + 3); }
        k += 1;
    }
    return taken, h, w;
}

// bmat_claimed_total gives the count of the stored values that the records
// of the builder claim. The count that a record claims gives the total, so
// that a builder that claims too many values gives the limit fault.
fn bmat_claimed_total(b, recs: i64[>..]&) -> i64 {
    var total: i64 = 0;
    var k: i64 = 0;
    while k < recs.len {
        let at = recs[k];
        let claim = bmat_declared_at(b, at);
        if total + claim > MAX_SPARSE_NNZ { return MAX_SPARSE_NNZ + 1; }
        total += claim;
        k += 1;
    }
    return total;
}

// bmat_new gives a builder of a grid of blocks. The grid holds no block.
fn bmat_new(grows: i64, gcols: i64) -> Bmat, Fault {
    let c = bmat_check_grid(grows, gcols, "bmat_new");
    if c.kind != F_NONE { return bmat_empty(), c; }
    return Bmat { grows: grows, gcols: gcols, n: 0, buf: [] }, no_fault();
}

// bmat_check_block checks the layout of a block. The buffer of a matrix of
// compressed rows holds the row pointers, then the columns, then the
// values. The pointers rise and the last one gives the count of the
// stored values. A block that does not keep this layout gives a fault,
// because the count of the block and the values that the block holds
// would then differ.
fn bmat_check_block(blk, tag: const u8[:]) -> Fault {
    if blk.rows < 1 || blk.cols < 1 || blk.nnz < 0 || blk.rows > MAX_SPARSE_SIDE || blk.cols > MAX_SPARSE_SIDE || blk.nnz > MAX_SPARSE_NNZ {
        return fault(F_SHAPE, tag, "the block holds a side or a count outside its range");
    }
    if blk.buf.len != blk.rows + 1 + 2 * blk.nnz {
        return fault(F_SHAPE, tag, "the buffer of the block does not match its shape");
    }
    var last: i64 = 0;
    var r: i64 = 0;
    while r <= blk.rows {
        let p = blk.buf[r];
        if !whole_place(p) || p < 0.0 {
            return fault(F_SHAPE, tag, "a row pointer of the block is not a whole number");
        }
        let q = p as i64;
        if r == 0 {
            if q != 0 { return fault(F_SHAPE, tag, "the first row pointer of the block is not zero"); }
        } else if q < last {
            return fault(F_SHAPE, tag, "the row pointers of the block do not rise");
        }
        last = q;
        r += 1;
    }
    if last != blk.nnz {
        return fault(F_SHAPE, tag, "the last row pointer of the block differs from its count");
    }
    // The columns of the block lie inside the block. A column outside the
    // block would land in the place of another block.
    var e: i64 = 0;
    while e < blk.nnz {
        let c = blk.buf[blk.rows + 1 + e];
        if !whole_place(c) || c < 0.0 || c >= blk.cols as f64 {
            return fault(F_SHAPE, tag, "a column of the block lies outside the block");
        }
        e += 1;
    }
    return no_fault();
}

// bmat_put places a block at a place of the grid. The blocks of one grid
// row share their height and the blocks of one grid column share their
// width. A place that holds a block already is a fault. A fault leaves
// the builder as it was.
fn bmat_put(b: Bmat&, i: f64, j: f64, blk) -> Fault {
    if !whole_place(i) || !whole_place(j) || i < 0.0 || i >= b.grows as f64 || j < 0.0 || j >= b.gcols as f64 {
        return fault(F_RANGE, "bmat_put", "the place is not a whole place inside the grid");
    }
    let r = i as i64;
    let c = j as i64;
    let recs = bmat_records(b);
    let taken, h, w = bmat_place(b, recs, r, c);
    if taken {
        return fault(F_RANGE, "bmat_put", "the place holds a block already");
    }
    if h > 0 && h != blk.rows {
        return fault(F_SHAPE, "bmat_put", "the height of the block differs from the height of its grid row");
    }
    if w > 0 && w != blk.cols {
        return fault(F_SHAPE, "bmat_put", "the width of the block differs from the width of its grid column");
    }
    // The limits come before the walk over the values of the block. A
    // block that claims many values gives the limit fault before the walk
    // reads them.
    if blk.nnz > MAX_SPARSE_NNZ {
        return fault(F_LIMIT, "bmat_put", "the count of the stored values is above the limit");
    }
    let total = bmat_claimed_total(b, recs);
    if total + blk.nnz > MAX_SPARSE_NNZ {
        return fault(F_LIMIT, "bmat_put", "the count of the stored values is above the limit");
    }
    let z = bmat_check_size(b.buf.len + BMAT_HEAD + BMAT_STRIDE * blk.nnz, "bmat_put");
    if z.kind != F_NONE { return z; }
    // The layout of the block comes next. The check makes the count of the
    // block and the values that the block holds the same number. The
    // limits above use the count that the block claims, so they stand
    // before the check.
    let lay = bmat_check_block(blk, "bmat_put");
    if lay.kind != F_NONE { return lay; }
    var k: i64 = 0;
    while k < blk.nnz {
        if mathshim::is_finite_of(csr_val(blk, k)) == 0.0 {
            return fault(F_VALUE, "bmat_put", "a value of the block is not finite");
        }
        k += 1;
    }
    b.buf.push(i);
    b.buf.push(j);
    b.buf.push(blk.rows as f64);
    b.buf.push(blk.cols as f64);
    b.buf.push(blk.nnz as f64);
    var row: i64 = 0;
    while row < blk.rows {
        var e = csr_ptr(blk, row);
        while e < csr_ptr(blk, row + 1) {
            b.buf.push(row as f64);
            b.buf.push(csr_col(blk, e) as f64);
            b.buf.push(csr_val(blk, e));
            e += 1;
        }
        row += 1;
    }
    b.n += 1;
    return no_fault();
}

// bmat_offsets turns the sizes of the grid rows or the grid columns into
// the offsets of the result. It gives the sum of the sizes. A running
// total stops at the first sum above the limit.
fn bmat_offsets(sizes: i64[>..]&, tag: const u8[:]) -> i64[>..], i64, Fault {
    var out: i64[>..] = [];
    var total: i64 = 0;
    var i: i64 = 0;
    while i < sizes.len {
        out.push(total);
        total += sizes[i];
        if total > MAX_SPARSE_SIDE {
            return out, total, fault(F_RANGE, tag, "the sum of the sizes is above the limit");
        }
        i += 1;
    }
    return out, total, no_fault();
}

// csr_bmat gives the compressed rows of the grid of blocks. The blocks
// stand in the order of the rows and the columns of the grid. A grid
// that holds no block is a fault of kind F_RANGE.
fn csr_bmat(b) -> Csr, Fault {
    let g = bmat_check_grid(b.grows, b.gcols, "csr_bmat");
    if g.kind != F_NONE { return csr_empty(), g; }
    if b.n < 1 {
        return csr_empty(), fault(F_RANGE, "csr_bmat", "the grid holds no block");
    }
    // The height of a grid row is the height of its first block. The
    // width of a grid column is the width of its first block.
    var heights: i64[>..] = [];
    var widths: i64[>..] = [];
    var r: i64 = 0;
    while r < b.grows { heights.push(0); r += 1; }
    var k: i64 = 0;
    while k < b.gcols { widths.push(0); k += 1; }
    // Every record gives its grid row and its grid column. The rule that
    // a record stands inside the grid holds for the whole walk.
    let recs = bmat_records(b);
    var grid_rows: i64[>..] = [];
    var grid_cols: i64[>..] = [];
    var idx: i64 = 0;
    while idx < recs.len {
        let at = recs[idx];
        let gr = bmat_int_at(b, at);
        let gc = bmat_int_at(b, at + 1);
        if gr < 0 || gr >= b.grows || gc < 0 || gc >= b.gcols {
            return csr_empty(), fault(F_RANGE, "csr_bmat", "a record stands outside the grid");
        }
        grid_rows.push(gr);
        grid_cols.push(gc);
        if heights[gr] == 0 { heights[gr] = bmat_int_at(b, at + 2); }
        if widths[gc] == 0 { widths[gc] = bmat_int_at(b, at + 3); }
        idx += 1;
    }
    // The heights and the widths become the offsets of the rows and the
    // columns.
    let row_off, total_h, fh = bmat_offsets(heights, "csr_bmat");
    if fh.kind != F_NONE { return csr_empty(), fh; }
    let col_off, total_w, fw = bmat_offsets(widths, "csr_bmat");
    if fw.kind != F_NONE { return csr_empty(), fw; }
    // A side of the result stands between one and the limit. The rule is
    // the rule of every other sparse format of the package.
    let s = csr_check(total_h, total_w, "csr_bmat");
    if s.kind != F_NONE { return csr_empty(), s; }
    // The offsets of the grid are added to the places of the blocks.
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    idx = 0;
    while idx < recs.len {
        let at = recs[idx];
        let cnt = bmat_count_at(b, at);
        let ro = row_off[grid_rows[idx]];
        let co = col_off[grid_cols[idx]];
        var t: i64 = 0;
        while t < cnt {
            let p = at + BMAT_HEAD + BMAT_STRIDE * t;
            if mathshim::is_finite_of(b.buf[p + 2]) == 0.0 {
                return csr_empty(), fault(F_VALUE, "csr_bmat", "a value of the block is not finite");
            }
            ri.push((ro + bmat_int_at(b, p)) as f64);
            ci.push((co + bmat_int_at(b, p + 1)) as f64);
            vv.push(b.buf[p + 2]);
            t += 1;
        }
        idx += 1;
    }
    return csr_from_triplets(total_h, total_w, ri, ci, vv);
}
