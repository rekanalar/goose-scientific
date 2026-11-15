// The sparse surface: the checks of the formats, the index types and the
// random matrices.
//
// The checks of the formats.
//
// The package holds seven sparse formats. Each format is its own struct.
// A check of a format tells whether the buffer of a matrix agrees with
// its sides. The checks of SciPy carry the names isspmatrix_csr and the
// brothers of the name. The package carries the names csr_valid and the
// brothers of the name. A matrix that a caller built by hand and broke
// does not pass its check.
//
// The index types.
//
// The package holds the index arrays of a matrix as buffers of numbers.
// get_index_dtype names the count of bits that holds the indices of a
// matrix. safely_cast_index_arrays checks every index before the count
// of bits is named.
//
// The random matrices.
//
// random_array gives a matrix of compressed rows with random places and
// random values. The places stand apart from each other. The values come
// from the law of the same name over the range from zero to one.
namespace scientific;

import mathshim;

// The largest index that fits in a 32 bit signed integer.
const MAX_INDEX32: i64 = 2147483647;

// whole_index tells whether a value is a whole number that an index
// buffer can hold.
fn whole_index(v: f64) -> bool {
    return mathshim::is_finite_of(v) != 0.0 && v == mathshim::floor_of(v)
        && mathshim::abs_of(v) <= 9007199254740992.0;
}

// csr_valid tells whether the buffer of a matrix of compressed rows
// agrees with its sides. The pointers grow and end at the count of the
// stored values. Every column stands inside the matrix and every value
// is finite.
fn csr_valid(s) -> bool {
    if s.rows < 1 || s.cols < 1 || s.nnz < 0 { return false; }
    if s.buf.len != s.rows + 1 + 2 * s.nnz { return false; }
    var i: i64 = 0;
    while i <= s.rows {
        if !whole_index(s.buf[i]) || s.buf[i] < 0.0 || s.buf[i] > s.nnz as f64 { return false; }
        if i > 0 && s.buf[i] < s.buf[i - 1] { return false; }
        i += 1;
    }
    if s.buf[0] != 0.0 || s.buf[s.rows] != s.nnz as f64 { return false; }
    var k: i64 = 0;
    while k < s.nnz {
        if !whole_index(s.buf[s.rows + 1 + k]) || s.buf[s.rows + 1 + k] < 0.0
            || s.buf[s.rows + 1 + k] >= s.cols as f64 { return false; }
        if mathshim::is_finite_of(s.buf[s.rows + 1 + s.nnz + k]) == 0.0 { return false; }
        k += 1;
    }
    return true;
}

// csc_valid tells whether the buffer of a matrix of compressed columns
// agrees with its sides.
fn csc_valid(s) -> bool {
    if s.rows < 1 || s.cols < 1 || s.nnz < 0 { return false; }
    if s.buf.len != s.cols + 1 + 2 * s.nnz { return false; }
    var i: i64 = 0;
    while i <= s.cols {
        if !whole_index(s.buf[i]) || s.buf[i] < 0.0 || s.buf[i] > s.nnz as f64 { return false; }
        if i > 0 && s.buf[i] < s.buf[i - 1] { return false; }
        i += 1;
    }
    if s.buf[0] != 0.0 || s.buf[s.cols] != s.nnz as f64 { return false; }
    var k: i64 = 0;
    while k < s.nnz {
        if !whole_index(s.buf[s.cols + 1 + k]) || s.buf[s.cols + 1 + k] < 0.0
            || s.buf[s.cols + 1 + k] >= s.rows as f64 { return false; }
        if mathshim::is_finite_of(s.buf[s.cols + 1 + s.nnz + k]) == 0.0 { return false; }
        k += 1;
    }
    return true;
}

// coo_valid tells whether the buffer of a matrix of coordinates agrees
// with its sides. The rows, the columns and the values follow each other
// in one buffer, in three sections.
fn coo_valid(s) -> bool {
    if s.rows < 1 || s.cols < 1 || s.nnz < 0 { return false; }
    if s.buf.len != 3 * s.nnz { return false; }
    var k: i64 = 0;
    while k < s.nnz {
        if !whole_index(s.buf[k]) || s.buf[k] < 0.0 || s.buf[k] >= s.rows as f64 { return false; }
        if !whole_index(s.buf[s.nnz + k]) || s.buf[s.nnz + k] < 0.0
            || s.buf[s.nnz + k] >= s.cols as f64 { return false; }
        if mathshim::is_finite_of(s.buf[2 * s.nnz + k]) == 0.0 { return false; }
        k += 1;
    }
    return true;
}

// dia_valid tells whether the buffer of a matrix of diagonals agrees
// with its sides. The offsets grow and stand inside the matrix. The
// count of the stored values is the sum of the lengths of the diagonals.
fn dia_valid(s) -> bool {
    if s.rows < 1 || s.cols < 1 || s.ndiag < 0 || s.nnz < 0 { return false; }
    // The buffer must hold the offsets before the loop reads them.
    if s.buf.len < s.ndiag { return false; }
    var sum: i64 = 0;
    var k: i64 = 0;
    while k < s.ndiag {
        if !whole_index(s.buf[k]) { return false; }
        let off = s.buf[k] as i64;
        if off < -(s.cols - 1) || off > s.rows - 1 { return false; }
        if k > 0 && off <= s.buf[k - 1] as i64 { return false; }
        sum += dia_len_of(s.rows, s.cols, off);
        k += 1;
    }
    if s.buf.len != s.ndiag + sum || s.nnz != sum { return false; }
    var i: i64 = s.ndiag;
    while i < s.buf.len {
        if mathshim::is_finite_of(s.buf[i]) == 0.0 { return false; }
        i += 1;
    }
    return true;
}

// dok_valid tells whether the buffer of a dictionary matrix agrees with
// its sides. The table holds whole places into the entries, and the
// entries hold the rows, the columns and the values.
fn dok_valid(s) -> bool {
    if s.rows < 1 || s.cols < 1 || s.n < 0 { return false; }
    if s.cap < 1 || (s.cap & (s.cap - 1)) != 0 { return false; }
    if s.buf.len != s.cap + 3 * s.n { return false; }
    // Each slot of the table holds zero for an empty slot, or the number
    // of an entry plus one. The table holds each entry once, and it keeps
    // at least one empty slot, so a search for a missing key stops.
    var used: i64 = 0;
    var t: i64 = 0;
    while t < s.cap {
        let slot = s.buf[t];
        if !whole_index(slot) || slot < 0.0 || slot > s.n as f64 { return false; }
        if slot > 0.0 { used += 1; }
        t += 1;
    }
    if used != s.n || used >= s.cap { return false; }
    var k: i64 = 0;
    while k < s.n {
        let row = s.buf[s.cap + 3 * k];
        if !whole_index(row) || row < 0.0 || row >= s.rows as f64 { return false; }
        let col = s.buf[s.cap + 3 * k + 1];
        if !whole_index(col) || col < 0.0 || col >= s.cols as f64 { return false; }
        if mathshim::is_finite_of(s.buf[s.cap + 3 * k + 2]) == 0.0 { return false; }
        // The entry must be the one that its own key finds.
        if dok_find(s, row as i64, col as i64) != k { return false; }
        k += 1;
    }
    return true;
}

// lil_valid tells whether the buffer of a list of lists matrix agrees
// with its sides. The offsets of the rows grow. The columns of each row
// grow and stand inside the matrix.
fn lil_valid(s) -> bool {
    if s.rows < 1 || s.cols < 1 || s.n < 0 { return false; }
    if s.buf.len != s.rows + 1 + 2 * s.n { return false; }
    var r: i64 = 0;
    while r <= s.rows {
        if !whole_index(s.buf[r]) || s.buf[r] < 0.0 || s.buf[r] > s.n as f64 { return false; }
        if r > 0 && s.buf[r] < s.buf[r - 1] { return false; }
        r += 1;
    }
    if s.buf[0] != 0.0 || s.buf[s.rows] != s.n as f64 { return false; }
    var e: i64 = 0;
    while e < s.n {
        if !whole_index(s.buf[s.rows + 1 + 2 * e]) { return false; }
        let col = s.buf[s.rows + 1 + 2 * e];
        if col < 0.0 || col >= s.cols as f64 { return false; }
        if mathshim::is_finite_of(s.buf[s.rows + 1 + 2 * e + 1]) == 0.0 { return false; }
        e += 1;
    }
    // The columns of each row grow.
    var r: i64 = 0;
    while r < s.rows {
        let first = s.buf[r] as i64;
        let last = s.buf[r + 1] as i64;
        var k = first + 1;
        while k < last {
            if s.buf[s.rows + 1 + 2 * k] <= s.buf[s.rows + 1 + 2 * (k - 1)] { return false; }
            k += 1;
        }
        r += 1;
    }
    return true;
}

// bsr_valid tells whether the buffer of a matrix of blocks agrees with
// its sides. The pointers of the block rows grow. Every block column
// stands inside the matrix.
fn bsr_valid(s) -> bool {
    if s.rows < 1 || s.cols < 1 || s.br < 1 || s.bc < 1 || s.nblocks < 0 { return false; }
    if s.rows % s.br != 0 || s.cols % s.bc != 0 { return false; }
    let brows = s.rows / s.br;
    let bcols = s.cols / s.bc;
    if s.buf.len != brows + 1 + s.nblocks + s.br * s.bc * s.nblocks { return false; }
    var i: i64 = 0;
    while i <= brows {
        if !whole_index(s.buf[i]) || s.buf[i] < 0.0 || s.buf[i] > s.nblocks as f64 { return false; }
        if i > 0 && s.buf[i] < s.buf[i - 1] { return false; }
        i += 1;
    }
    if s.buf[0] != 0.0 || s.buf[brows] != s.nblocks as f64 { return false; }
    var k: i64 = 0;
    while k < s.nblocks {
        if !whole_index(s.buf[brows + 1 + k]) || s.buf[brows + 1 + k] < 0.0
            || s.buf[brows + 1 + k] >= bcols as f64 { return false; }
        k += 1;
    }
    let base = brows + 1 + s.nblocks;
    var v: i64 = base;
    while v < s.buf.len {
        if mathshim::is_finite_of(s.buf[v]) == 0.0 { return false; }
        v += 1;
    }
    return true;
}

// get_index_dtype names the count of bits that holds every index of a
// matrix. A largest index above the largest index of a 32 bit integer
// takes 64.
fn get_index_dtype(maxval: i64) -> i64 {
    if maxval > MAX_INDEX32 { return 64; }
    return 32;
}

// safely_cast_index_arrays checks the index arrays of a matrix. Every
// index must be a whole number of at least zero, and every index must
// stay at or below the largest value. The call gives the count of bits
// of the type that holds the indices.
fn safely_cast_index_arrays(values: f64[:], low_max: i64 = MAX_INDEX32) -> i64, Fault {
    if low_max < 1 {
        return 0, fault(F_RANGE, "safely_cast_index_arrays", "the limit of the small type must be above zero");
    }
    if low_max > 9007199254740992 {
        return 0, fault(F_RANGE, "safely_cast_index_arrays", "the limit stands above the largest whole number");
    }
    var bits: i64 = 32;
    var i: i64 = 0;
    while i < values.len {
        let v = values[i];
        if !whole_index(v) || v < 0.0 {
            return 0, fault(F_VALUE, "safely_cast_index_arrays", "an index is not a whole number of at least zero");
        }
        if v > low_max as f64 { bits = 64; }
        i += 1;
    }
    return bits, no_fault();
}

// random_array gives a matrix of compressed rows with the places and the
// values of chance. The count of the places is the density times the
// count of the places of the shape, rounded to the nearest whole number.
// The places stand apart from each other. Each value comes from the law
// of one range over zero to one. The values of this call do not equal
// the values of SciPy, because the source of chance of the package
// differs from the source of NumPy.
fn random_array(shape: i64[:], density: f64 = 0.01, seed: i64 = 0) -> Csr, Fault {
    if shape.len < 1 || shape.len > 2 {
        return csr_empty(), fault(F_SHAPE, "random_array", "the shape holds more dimensions than two");
    }
    let rows = shape[0];
    var cols: i64 = 1;
    if shape.len == 2 { cols = shape[1]; }
    let c = csr_check(rows, cols, "random_array");
    if c.kind != F_NONE { return csr_empty(), c; }
    if mathshim::is_finite_of(density) == 0.0 || density < 0.0 || density > 1.0 {
        return csr_empty(), fault(F_VALUE, "random_array", "the density stands outside zero to one");
    }
    let total = rows * cols;
    var count: i64 = mathshim::round_of(density * total as f64) as i64;
    if count > total { count = total; }
    var r = default_rng(seed);
    var mem, f0 = dok_new(rows, cols);
    if f0.kind != F_NONE { return csr_empty(), f0; }
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var tries: i64 = 0;
    while ri.len < count {
        tries += 1;
        if tries > 100 * count + 1000 {
            return csr_empty(), fault(F_LIMIT, "random_array", "the places of chance did not fill the count of the values");
        }
        let flat, ff = integers(r, 0, total);
        if ff.kind != F_NONE { return csr_empty(), ff; }
        let i = flat / cols;
        let j = flat % cols;
        let before = dok_nnz(mem);
        let put = dok_put(mem, i, j, 1.0, "random_array");
        if put.kind != F_NONE { return csr_empty(), put; }
        if dok_nnz(mem) != before {
            ri.push(i as f64);
            ci.push(j as f64);
            vv.push(uniform(r, 0.0, 1.0));
        }
    }
    let s, f1 = csr_from_triplets(rows, cols, ri, ci, vv);
    if f1.kind != F_NONE { return csr_empty(), f1; }
    return s, no_fault();
}

// random gives a matrix of two sides with the places and the values of
// chance. The call names the sides apart.
fn random(rows: i64, cols: i64, density: f64 = 0.01, seed: i64 = 0) -> Csr, Fault {
    var shape: i64[>..] = [];
    shape.push(rows);
    shape.push(cols);
    return random_array(shape, density, seed);
}
