// The sparse matrices of blocks.
//
// A matrix of the type Bsr cuts the matrix into blocks of br rows and bc
// columns. The block size divides the shape. A block is kept when the
// source holds one entry in it. A zero inside a kept block stays in the
// block, so the count of the stored values counts the zeros.
//
// The buffer holds, in this order, the row pointers of the block rows, the
// block column of every block and the values of every block. Each block
// lies in the order of the row.
//
// The blocks of one block row keep the order of their first entry, as the
// blocks of SciPy do. The order of the block columns can therefore fall.
//
// The calls that have no rule of their own go through the compressed
// rows, as the other sparse formats do.
namespace scientific;

// Bsr holds a sparse matrix of blocks.
struct Bsr { rows: i64, cols: i64, br: i64, bc: i64, nblocks: i64, buf: f64[>..] }

// bsr_empty gives a matrix without a block.
fn bsr_empty() -> Bsr {
    return Bsr { rows: 0, cols: 0, br: 1, bc: 1, nblocks: 0, buf: [] };
}

// bsr_ptr gives the place in the blocks where the block row i starts.
fn bsr_ptr(s, i: i64) -> i64 {
    return s.buf[i] as i64;
}

// bsr_col gives the block column of the block k.
fn bsr_col(s, k: i64) -> i64 {
    return s.buf[s.rows / s.br + 1 + k] as i64;
}

// bsr_base gives the place in the buffer of the first value of the first
// block.
fn bsr_base(s) -> i64 {
    return s.rows / s.br + 1 + s.nblocks;
}

// bsr_from_entries builds a matrix from the rows, the columns and the
// values of the entries. The entries of one place add up. A block that
// holds no entry is not kept. The blocks of one block row keep the order
// of their first entry.
fn bsr_from_entries(rows: i64, cols: i64, br: i64, bc: i64, ri: f64[:], ci: f64[:], vv: f64[:]) -> Bsr, Fault {
    let c = csr_check(rows, cols, "bsr_from_entries");
    if c.kind != F_NONE { return bsr_empty(), c; }
    // The block size is checked before anything is multiplied.
    if br < 1 || bc < 1 || rows % br != 0 || cols % bc != 0 {
        return bsr_empty(), fault(F_RANGE, "bsr_from_entries", "the block size does not divide the shape");
    }
    let t = sparse_check_triplets(rows, cols, ri, ci, vv, "bsr_from_entries");
    if t.kind != F_NONE { return bsr_empty(), t; }
    let n = vv.len;
    let nbr = rows / br;
    // The entries in the order of the block row, the block column, the
    // row and the column. The block row is the main key.
    var keys: f64[>..] = [];
    var i: i64 = 0;
    while i < n { keys.push(ci[i]); i += 1; }
    i = 0;
    while i < n { keys.push(ri[i]); i += 1; }
    i = 0;
    while i < n { keys.push(((ci[i] as i64) / bc) as f64); i += 1; }
    i = 0;
    while i < n { keys.push(((ri[i] as i64) / br) as f64); i += 1; }
    var ord: i64[>..] = [];
    if n > 0 { ord = order_places(keys, 4, n); }
    // The blocks: the block row, the block column and the range of the
    // entries of every block. The entries of one block lie in the order
    // of the row and then of the column.
    var gbi: f64[>..] = [];
    var gbj: f64[>..] = [];
    var gbeg: i64[>..] = [];
    var gend: i64[>..] = [];
    var k: i64 = 0;
    while k < n {
        let e = ord[k];
        let bi = (ri[e] as i64) / br;
        let bj = (ci[e] as i64) / bc;
        var m = k + 1;
        while m < n {
            let f = ord[m];
            if ((ri[f] as i64) / br) != bi || ((ci[f] as i64) / bc) != bj { break; }
            m += 1;
        }
        gbi.push(bi as f64);
        gbj.push(bj as f64);
        gbeg.push(k);
        gend.push(m);
        k = m;
    }
    let nb = gbi.len;
    // The blocks of one block row keep the order of their first entry.
    var gkeys: f64[>..] = [];
    i = 0;
    while i < nb { let e = ord[gbeg[i]]; gkeys.push(ci[e]); i += 1; }
    i = 0;
    while i < nb { let e = ord[gbeg[i]]; gkeys.push(ri[e]); i += 1; }
    i = 0;
    while i < nb { gkeys.push(gbi[i]); i += 1; }
    var gord: i64[>..] = [];
    if nb > 0 { gord = order_places(gkeys, 3, nb); }
    // The count of the stored values is checked by a division, so no
    // product can overflow.
    let size = br * bc;
    if nb > MAX_SPARSE_NNZ / size {
        return bsr_empty(), fault(F_LIMIT, "bsr_from_entries", "the count of the stored values is above the limit");
    }
    // The row pointers of the block rows.
    var buf: f64[>..] = [];
    i = 0;
    while i <= nbr { buf.push(0.0); i += 1; }
    i = 0;
    while i < nb {
        let b = (gbi[gord[i]] as i64) + 1;
        buf[b] = buf[b] + 1.0;
        i += 1;
    }
    i = 0;
    while i < nbr { buf[i + 1] = buf[i + 1] + buf[i]; i += 1; }
    // The block column of every block.
    i = 0;
    while i < nb { buf.push(gbj[gord[i]]); i += 1; }
    let base = buf.len;
    i = 0;
    while i < nb * size { buf.push(0.0); i += 1; }
    // The entries of one place add up.
    i = 0;
    while i < nb {
        let g = gord[i];
        var e = gbeg[g];
        while e < gend[g] {
            let f = ord[e];
            let at = base + i * size + ((ri[f] as i64) % br) * bc + ((ci[f] as i64) % bc);
            buf[at] = buf[at] + vv[f];
            // A value that is not finite is a fault, also one that the
            // sum made.
            if mathshim::is_finite_of(buf[at]) == 0.0 {
                return bsr_empty(), fault(F_VALUE, "bsr_from_entries", "a value is not finite");
            }
            e += 1;
        }
        i += 1;
    }
    return Bsr { rows: rows, cols: cols, br: br, bc: bc, nblocks: nb, buf: buf }, no_fault();
}

// bsr_from_csr builds a matrix from a matrix of compressed rows. A stored
// zero keeps its block.
fn bsr_from_csr(s, br: i64, bc: i64) -> Bsr, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        let e = csr_ptr(s, i + 1);
        while k < e {
            ri.push(i as f64); ci.push(csr_col(s, k) as f64); vv.push(csr_val(s, k));
            k += 1;
        }
        i += 1;
    }
    return bsr_from_entries(s.rows, s.cols, br, bc, ri, ci, vv);
}

// bsr_from_dense builds a matrix from a dense matrix. A block that holds
// no value that is not zero is not kept.
fn bsr_from_dense(a, br: i64, bc: i64) -> Bsr, Fault {
    let s, f = csr_from_dense(a);
    if f.kind != F_NONE { return bsr_empty(), f; }
    return bsr_from_csr(s, br, bc);
}

// bsr_from_coo builds a matrix from a matrix of coordinates. The
// duplicates add up, as they add up in the compressed rows.
fn bsr_from_coo(c, br: i64, bc: i64) -> Bsr, Fault {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var k: i64 = 0;
    while k < c.nnz {
        ri.push(coo_row(c, k) as f64);
        ci.push(coo_col(c, k) as f64);
        vv.push(coo_val(c, k));
        k += 1;
    }
    return bsr_from_entries(c.rows, c.cols, br, bc, ri, ci, vv);
}

// bsr_triplets gives the rows, the columns and the values of the entries
// of a matrix of blocks. Every value of every block is an entry, the
// zeros too.
fn bsr_triplets(s) -> f64[>..], f64[>..], f64[>..] {
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    let base = bsr_base(s);
    let size = s.br * s.bc;
    var bi: i64 = 0;
    while bi < s.rows / s.br {
        var k = bsr_ptr(s, bi);
        let end = bsr_ptr(s, bi + 1);
        while k < end {
            let bj = bsr_col(s, k);
            var r: i64 = 0;
            while r < s.br {
                var c: i64 = 0;
                while c < s.bc {
                    ri.push((bi * s.br + r) as f64);
                    ci.push((bj * s.bc + c) as f64);
                    vv.push(s.buf[base + k * size + r * s.bc + c]);
                    c += 1;
                }
                r += 1;
            }
            k += 1;
        }
        bi += 1;
    }
    return ri, ci, vv;
}

// bsr_to_coo gives the coordinates of a matrix of blocks. Every value of
// every block is an entry, the zeros too, as SciPy gives them.
fn bsr_to_coo(s) -> Coo, Fault {
    let ri, ci, vv = bsr_triplets(s);
    return coo_from_triplets(s.rows, s.cols, ri, ci, vv);
}

// bsr_to_csr gives the compressed rows of a matrix of blocks. The zeros
// of the stored blocks stay, as they stay in SciPy.
fn bsr_to_csr(s) -> Csr, Fault {
    let ri, ci, vv = bsr_triplets(s);
    return csr_from_triplets(s.rows, s.cols, ri, ci, vv);
}

// bsr_to_dense gives the dense matrix of a matrix of blocks.
fn bsr_to_dense(s) -> Arr, Fault {
    if s.rows * s.cols > MAX_ELEMENTS {
        return empty_arr(), fault(F_LIMIT, "bsr_to_dense", "the dense matrix holds more elements than the limit");
    }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows * s.cols { d.push(0.0); i += 1; }
    let base = bsr_base(s);
    let size = s.br * s.bc;
    var bi: i64 = 0;
    while bi < s.rows / s.br {
        var k = bsr_ptr(s, bi);
        let end = bsr_ptr(s, bi + 1);
        while k < end {
            let bj = bsr_col(s, k);
            var r: i64 = 0;
            while r < s.br {
                var c: i64 = 0;
                while c < s.bc {
                    d[(bi * s.br + r) * s.cols + (bj * s.bc + c)] = s.buf[base + k * size + r * s.bc + c];
                    c += 1;
                }
                r += 1;
            }
            k += 1;
        }
        bi += 1;
    }
    return mat2(copy(d), s.rows, s.cols), no_fault();
}

// bsr_get gives the value at a place, zero when no block holds the place.
fn bsr_get(s, i: i64, j: i64) -> f64, Fault {
    if i < 0 || i >= s.rows || j < 0 || j >= s.cols {
        return 0.0, fault(F_RANGE, "bsr_get", "the place lies outside the matrix");
    }
    let bi = i / s.br;
    let bj = j / s.bc;
    // The block columns of a block row keep the order of their first
    // entry, so the search walks the blocks of the row.
    let base = bsr_base(s);
    var k = bsr_ptr(s, bi);
    while k < bsr_ptr(s, bi + 1) {
        if bsr_col(s, k) == bj {
            return s.buf[base + k * s.br * s.bc + (i % s.br) * s.bc + (j % s.bc)], no_fault();
        }
        k += 1;
    }
    return 0.0, no_fault();
}

// bsr_scale multiplies every stored value by a number.
fn bsr_scale(s, a: f64) -> Bsr, Fault {
    if mathshim::is_finite_of(a) == 0.0 {
        return bsr_empty(), fault(F_VALUE, "bsr_scale", "the factor is not finite");
    }
    var buf = copy(s.buf);
    var i = bsr_base(s);
    while i < buf.len {
        buf[i] = buf[i] * a;
        // A product that is not finite is a fault. The values of a matrix
        // of blocks are always finite.
        if mathshim::is_finite_of(buf[i]) == 0.0 {
            return bsr_empty(), fault(F_VALUE, "bsr_scale", "a value is not finite");
        }
        i += 1;
    }
    return Bsr { rows: s.rows, cols: s.cols, br: s.br, bc: s.bc, nblocks: s.nblocks, buf: buf }, no_fault();
}

// bsr_matvec gives the product of a matrix and a vector.
fn bsr_matvec(s, x: f64[:]) -> Arr, Fault {
    if x.len != s.cols {
        return empty_arr(), fault(F_SHAPE, "bsr_matvec", "the vector does not hold one value per column");
    }
    var out = nd_zeros(s.rows);
    let base = bsr_base(s);
    let size = s.br * s.bc;
    var bi: i64 = 0;
    while bi < s.rows / s.br {
        var k = bsr_ptr(s, bi);
        while k < bsr_ptr(s, bi + 1) {
            let bj = bsr_col(s, k);
            var r: i64 = 0;
            while r < s.br {
                var c: i64 = 0;
                while c < s.bc {
                    let at = bi * s.br + r;
                    out[at] = out[at] + s.buf[base + k * size + r * s.bc + c] * x[bj * s.bc + c];
                    c += 1;
                }
                r += 1;
            }
            k += 1;
        }
        bi += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// bsr_transpose gives the transpose of a matrix. The block size swaps.
// The blocks lie in the order of the block row of the transpose.
fn bsr_transpose(s) -> Bsr, Fault {
    let c = csr_check(s.cols, s.rows, "bsr_transpose");
    if c.kind != F_NONE { return bsr_empty(), c; }
    let nb = s.nblocks;
    let size = s.br * s.bc;
    let base = bsr_base(s);
    // The block row and the block column of each block of the source.
    var bis: f64[>..] = [];
    var bjs: f64[>..] = [];
    var bi: i64 = 0;
    while bi < s.rows / s.br {
        var k = bsr_ptr(s, bi);
        while k < bsr_ptr(s, bi + 1) {
            bis.push(bi as f64); bjs.push(bsr_col(s, k) as f64);
            k += 1;
        }
        bi += 1;
    }
    // The block columns of the source are the main key.
    var keys: f64[>..] = [];
    var i: i64 = 0;
    while i < nb { keys.push(bis[i]); i += 1; }
    i = 0;
    while i < nb { keys.push(bjs[i]); i += 1; }
    var ord: i64[>..] = [];
    if nb > 0 { ord = order_places(keys, 2, nb); }
    let nbr = s.cols / s.bc;
    var buf: f64[>..] = [];
    i = 0;
    while i <= nbr { buf.push(0.0); i += 1; }
    i = 0;
    while i < nb { let r = (bjs[i] as i64) + 1; buf[r] = buf[r] + 1.0; i += 1; }
    i = 0;
    while i < nbr { buf[i + 1] = buf[i + 1] + buf[i]; i += 1; }
    i = 0;
    while i < nb { buf.push(bis[ord[i]]); i += 1; }
    i = 0;
    while i < nb {
        let k = ord[i];
        var r: i64 = 0;
        while r < s.bc {
            var q: i64 = 0;
            while q < s.br {
                buf.push(s.buf[base + k * size + q * s.bc + r]);
                q += 1;
            }
            r += 1;
        }
        i += 1;
    }
    return Bsr { rows: s.cols, cols: s.rows, br: s.bc, bc: s.br, nblocks: nb, buf: buf }, no_fault();
}

// bsr_diagonal gives the values on the main diagonal.
fn bsr_diagonal(s) -> Arr, Fault {
    var n = s.rows;
    if s.cols < n { n = s.cols; }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < n {
        let v, f = bsr_get(s, i, i);
        if f.kind != F_NONE { return empty_arr(), f; }
        out.push(v);
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// bsr_sum gives the sum of the stored values.
fn bsr_sum(s) -> f64 {
    var t: f64 = 0.0;
    var i = bsr_base(s);
    while i < s.buf.len { t += s.buf[i]; i += 1; }
    return t;
}

// bsr_nnz gives the count of the stored values. A zero inside a stored
// block counts.
fn bsr_nnz(s) -> i64 {
    return s.nblocks * s.br * s.bc;
}

// bsr_blocks gives the row pointers, the block columns and the values of
// the blocks. The values of one block lie in the order of the row.
fn bsr_blocks(s) -> Arr, Arr, Arr, Fault {
    let nbr = s.rows / s.br;
    var p: f64[>..] = [];
    var c: f64[>..] = [];
    var v: f64[>..] = [];
    var i: i64 = 0;
    while i <= nbr { p.push(bsr_ptr(s, i) as f64); i += 1; }
    i = 0;
    while i < s.nblocks { c.push(bsr_col(s, i) as f64); i += 1; }
    i = bsr_base(s);
    while i < s.buf.len { v.push(s.buf[i]); i += 1; }
    return line_arr(copy(p)), line_arr(copy(c)), line_arr(copy(v)), no_fault();
}
