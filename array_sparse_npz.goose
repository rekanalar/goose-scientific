// The archive of the sparse formats: save_npz and load_npz.
//
// SciPy holds a sparse matrix in a zip archive of .npy members. The
// formats csr, csc and bsr hold the members indices and indptr. The
// format dia holds offsets. The format coo holds row and col. Every
// archive holds format, shape and data behind the members of its
// format, in that order.
//
// The writer of this package stores its members, because the standard
// library holds no deflate. SciPy compresses its members unless the
// caller asks for stored members. The reader of this package gives a
// fault of kind F_VALUE for a compressed member, and reads the
// members that stay stored.
//
// The types of the members are the types of NumPy. The values hold
// float64. The indices hold int32 while every index fits, and int64
// when one stands above the largest index of a 32 bit whole number.
// The shape holds int64. The name of the format holds three bytes
// of the type that holds no order. The reader takes these types and
// the writer writes them, so NumPy reads the file of the package and
// the package reads the file of SciPy.
namespace scientific;

import os;


// save_npz writes a matrix of compressed rows into an archive of zip
// files. The members hold the keys of SciPy for the format csr. They
// are the indices, the pointer, the name of the format, the shape and
// the values, in that order. A caller converts another format to the
// compressed rows first. The members stay stored.
fn save_npz(path: const u8[:], m: Csr) -> bool, Fault {
    if !csr_valid(m) {
        return false, fault(F_VALUE, "save_npz", "the matrix does not pass its check");
    }
    let wide = npz_body_wide(m.buf[..m.rows + 1 + m.nnz]);
    var ptrs: u8[>..] = npz_body_i(m.buf[..m.rows + 1], wide);
    var cols: u8[>..] = npz_body_i(m.buf[m.rows + 1..m.rows + 1 + m.nnz], wide);
    var vals: u8[>..] = npz_body_f64(m.buf[m.rows + 1 + m.nnz..m.rows + 1 + 2 * m.nnz]);
    // The type of the indices follows the rule of the index type of
    // SciPy: int32 while every index fits, int64 when one stands above
    // the largest index of a 32 bit whole number.
    var descr: u8[>..] = [];
    descr.append("<i");
    if wide { descr.push('8'); } else { descr.push('4'); }
    var dn_nnz: i64[>..] = [];
    dn_nnz.push(m.nnz);
    var dn_ptr: i64[>..] = [];
    dn_ptr.push(m.rows + 1);
    var dn_two: i64[>..] = [];
    dn_two.push(2);
    var dn_zero: i64[>..] = [];
    var hidx = npz_head(descr, dn_nnz);
    hidx.append(cols[..]);
    var hptr = npz_head(descr, dn_ptr);
    hptr.append(ptrs[..]);
    var hfmt = npz_head("|S3", dn_zero);
    hfmt.append("csr");
    var hshape = npz_head("<i8", dn_two);
    var sb: u8[>..] = [];
    npz_put_i64(sb, m.rows);
    npz_put_i64(sb, m.cols);
    hshape.append(sb[..]);
    var hdata = npz_head("<f8", dn_nnz);
    hdata.append(vals[..]);
    var names: const u8[:][>..] = [];
    var bodies: const u8[:][>..] = [];
    names.push("indices");
    bodies.push(hidx[..]);
    names.push("indptr");
    bodies.push(hptr[..]);
    names.push("format");
    bodies.push(hfmt[..]);
    names.push("shape");
    bodies.push(hshape[..]);
    names.push("data");
    bodies.push(hdata[..]);
    return npz_write(path, names, bodies, "save_npz");
}

// npz_body_wide tells whether a list of indices stands above the
// largest index of a 32 bit whole number, as the rule of the index
// type of SciPy decides.
fn npz_body_wide(vals: f64[:]) -> bool {
    var i: i64 = 0;
    while i < vals.len {
        if vals[i] > MAX_INDEX32 as f64 { return true; }
        i += 1;
    }
    return false;
}

// load_npz reads a matrix from an archive of zip files. It reads the
// name of the format and builds the matrix from the members of that
// format. The formats csr, csc, bsr, dia and coo are read, and the
// matrix comes back as compressed rows.
fn load_npz(path: const u8[:]) -> Csr, Fault {
    let fp = npy_path(path, "load_npz");
    if fp.kind != F_NONE { return csr_empty(), fp; }
    var b: u8[>..] = [];
    if !read_file(path, b) {
        return csr_empty(), fault(F_VALUE, "load_npz", "the file does not read");
    }
    if b.len > TEXT_MAX_BYTES {
        return csr_empty(), fault(F_LIMIT, "load_npz", "the file holds more bytes than the limit");
    }
    return npz_read(copy(b));
}

// npz_read builds the matrix of the bytes of an archive.
fn npz_read(b: u8[>..]) -> Csr, Fault {
    let tag = "load_npz";
    let e, ef = npz_eocd(b, tag);
    if ef.kind != F_NONE { return csr_empty(), ef; }
    let ns, nl, me, us, of, xf = npz_entries(b, e, tag);
    if xf.kind != F_NONE { return csr_empty(), xf; }
    // The name of the format names three bytes in a member that holds
    // no length.
    var code: i64 = -1;
    let fb, f1 = npz_member(b, ns, nl, me, us, of, "format", tag);
    if f1.kind != F_NONE { return csr_empty(), f1; }
    let fk, fs, fbody, f2 = npz_npy_meta(fb, tag);
    if f2.kind != F_NONE { return csr_empty(), f2; }
    if fk != 3 || fs.len != 0 || fb.len - fbody != 3 {
        return csr_empty(), fault(F_VALUE, tag, "the member of the format holds the wrong type");
    }
    if fb[fbody] == 'c' && fb[fbody + 1] == 's' && fb[fbody + 2] == 'r' { code = 0; }
    if fb[fbody] == 'c' && fb[fbody + 1] == 's' && fb[fbody + 2] == 'c' { code = 1; }
    if fb[fbody] == 'b' && fb[fbody + 1] == 's' && fb[fbody + 2] == 'r' { code = 2; }
    if fb[fbody] == 'd' && fb[fbody + 1] == 'i' && fb[fbody + 2] == 'a' { code = 3; }
    if fb[fbody] == 'c' && fb[fbody + 1] == 'o' && fb[fbody + 2] == 'o' { code = 4; }
    // The shape names the two sides in whole numbers of int64.
    let sb, s1 = npz_member(b, ns, nl, me, us, of, "shape", tag);
    if s1.kind != F_NONE { return csr_empty(), s1; }
    let sk, ss, sbody, s2 = npz_npy_meta(sb, tag);
    if s2.kind != F_NONE { return csr_empty(), s2; }
    if (sk != 1 && sk != 2) || ss.len != 1 || ss[0] != 2 {
        return csr_empty(), fault(F_VALUE, tag, "the member of the shape holds the wrong type");
    }
    let sd, s3 = npz_vals_i64(sb, sbody, 2, sk, tag);
    if s3.kind != F_NONE { return csr_empty(), s3; }
    let rows = sd[0] as i64;
    let cols = sd[1] as i64;
    let c = csr_check(rows, cols, tag);
    if c.kind != F_NONE { return csr_empty(), c; }
    if code == 0 { return npz_read_csr(b, ns, nl, me, us, of, rows, cols); }
    if code == 1 { return npz_read_csc(b, ns, nl, me, us, of, rows, cols); }
    if code == 2 { return npz_read_bsr(b, ns, nl, me, us, of, rows, cols); }
    if code == 3 { return npz_read_dia(b, ns, nl, me, us, of, rows, cols); }
    if code == 4 { return npz_read_coo(b, ns, nl, me, us, of, rows, cols); }
    return csr_empty(), fault(F_VALUE, tag, "the archive holds a format that the package does not read");
}

// npz_index_member gives the whole numbers of a member of indices.
fn npz_index_member(b: const u8[:], ns: i64[:], nl: i64[:], me: i64[:], us: i64[:],
                    of: i64[:], want: const u8[:], tag: const u8[:]) -> f64[>..], Fault {
    var out: f64[>..] = [];
    let mb, f1 = npz_member(b, ns, nl, me, us, of, want, tag);
    if f1.kind != F_NONE { return out, f1; }
    let k, s, body, f2 = npz_npy_meta(mb, tag);
    if f2.kind != F_NONE { return out, f2; }
    if (k != 1 && k != 2) || s.len != 1 {
        return out, fault(F_VALUE, tag, "a member holds the wrong type or shape");
    }
    if s[0] > MAX_SPARSE_NNZ {
        return out, fault(F_LIMIT, tag, "a member holds more indices than the limit");
    }
    return npz_vals_i64(mb, body, s[0], k, tag);
}

// npz_data_member gives the values of a member of values and its
// shape. Every value has to be finite.
fn npz_data_member(b: const u8[:], ns: i64[:], nl: i64[:], me: i64[:], us: i64[:],
                   of: i64[:], want: const u8[:], tag: const u8[:]) -> f64[>..], i64[>..], Fault {
    var out: f64[>..] = [];
    var shp: i64[>..] = [];
    let mb, f1 = npz_member(b, ns, nl, me, us, of, want, tag);
    if f1.kind != F_NONE { return out, shp, f1; }
    let k, s, body, f2 = npz_npy_meta(mb, tag);
    if f2.kind != F_NONE { return out, shp, f2; }
    if k != 0 || s.len < 1 {
        return out, shp, fault(F_VALUE, tag, "a member holds the wrong type or shape");
    }
    // The count of the values must stay inside the limit. A wrap of
    // the product would hide a shape above the limit.
    var count: i64 = 1;
    var i: i64 = 0;
    while i < s.len {
        if s[i] > 0 && count > MAX_ELEMENTS / s[i] {
            return out, shp, fault(F_LIMIT, tag, "a member holds more values than the limit");
        }
        count *= s[i];
        i += 1;
    }
    let vals, f3 = npz_vals_f64(mb, body, count, tag);
    if f3.kind != F_NONE { return out, shp, f3; }
    i = 0;
    while i < vals.len {
        if mathshim::is_finite_of(vals[i]) == 0.0 {
            return out, shp, fault(F_VALUE, tag, "a stored value is not finite");
        }
        i += 1;
    }
    return vals, s, no_fault();
}

// npz_read_csr builds the compressed rows of the members of the format
// csr. The pointers have to rise and to end at the count of the
// values. The columns of every row have to rise, so the file holds the
// canonical form.
fn npz_read_csr(b: const u8[:], ns: i64[:], nl: i64[:], me: i64[:], us: i64[:],
                of: i64[:], rows: i64, cols: i64) -> Csr, Fault {
    let idx, f1 = npz_index_member(b, ns, nl, me, us, of, "indices", "load_npz");
    if f1.kind != F_NONE { return csr_empty(), f1; }
    let ptr, f2 = npz_index_member(b, ns, nl, me, us, of, "indptr", "load_npz");
    if f2.kind != F_NONE { return csr_empty(), f2; }
    let vals, shp, f3 = npz_data_member(b, ns, nl, me, us, of, "data", "load_npz");
    if f3.kind != F_NONE { return csr_empty(), f3; }
    let e = npz_check_indexed(ptr, idx, vals, shp, rows, cols);
    if e.kind != F_NONE { return csr_empty(), e; }
    return csr_from_parts(rows, cols, ptr, idx, vals), no_fault();
}

// npz_read_csc builds the compressed columns of the members of the
// format csc, and turns them into compressed rows.
fn npz_read_csc(b: const u8[:], ns: i64[:], nl: i64[:], me: i64[:], us: i64[:],
                of: i64[:], rows: i64, cols: i64) -> Csr, Fault {
    let idx, f1 = npz_index_member(b, ns, nl, me, us, of, "indices", "load_npz");
    if f1.kind != F_NONE { return csr_empty(), f1; }
    let ptr, f2 = npz_index_member(b, ns, nl, me, us, of, "indptr", "load_npz");
    if f2.kind != F_NONE { return csr_empty(), f2; }
    let vals, shp, f3 = npz_data_member(b, ns, nl, me, us, of, "data", "load_npz");
    if f3.kind != F_NONE { return csr_empty(), f3; }
    let e = npz_check_indexed(ptr, idx, vals, shp, cols, rows);
    if e.kind != F_NONE { return csr_empty(), e; }
    let m = csc_from_parts(rows, cols, ptr, idx, vals);
    return csc_to_csr(m);
}

// npz_check_indexed holds the checks that the formats with pointers
// and indices share. The pointers have to rise and to end at the count
// of the values. Every index has to stand inside its side and to rise
// along its row or column.
fn npz_check_indexed(ptr: f64[:], idx: f64[:], vals: f64[:], shp: i64[:],
                     seg: i64, other: i64) -> Fault {
    let tag = "load_npz";
    if ptr.len != seg + 1 || shp.len != 1 {
        return fault(F_SHAPE, tag, "a member does not fit the shape");
    }
    if idx.len != vals.len {
        return fault(F_SHAPE, tag, "the count of the values does not fit");
    }
    if idx.len > MAX_SPARSE_NNZ {
        return fault(F_LIMIT, tag, "the count of the stored values is above the limit");
    }
    if !whole_index(ptr[0]) || ptr[0] != 0.0 {
        return fault(F_SHAPE, tag, "the first pointer is not zero");
    }
    var i: i64 = 0;
    while i < seg {
        if !whole_index(ptr[i]) || !whole_index(ptr[i + 1]) || ptr[i] > ptr[i + 1] {
            return fault(F_SHAPE, tag, "the pointers do not rise");
        }
        i += 1;
    }
    if !whole_index(ptr[seg]) || ptr[seg] != idx.len as f64 {
        return fault(F_SHAPE, tag, "the last pointer differs from the count of the values");
    }
    i = 0;
    while i < seg {
        var prev: i64 = -1;
        var k = ptr[i] as i64;
        let e = ptr[i + 1] as i64;
        while k < e {
            let c = idx[k];
            if !whole_index(c) || c < 0.0 || c >= other as f64 {
                return fault(F_RANGE, tag, "an index stands outside the matrix");
            }
            if (c as i64) <= prev {
                return fault(F_SHAPE, tag, "the indices of a segment do not rise");
            }
            prev = c as i64;
            k += 1;
        }
        i += 1;
    }
    return no_fault();
}

// npz_read_bsr builds the blocks of the members of the format bsr, and
// turns them into compressed rows. The file holds no block size. The
// two lengths of the values name it, as SciPy names it when it reads
// the file.
fn npz_read_bsr(b: const u8[:], ns: i64[:], nl: i64[:], me: i64[:], us: i64[:],
                of: i64[:], rows: i64, cols: i64) -> Csr, Fault {
    let tag = "load_npz";
    let idx, f1 = npz_index_member(b, ns, nl, me, us, of, "indices", tag);
    if f1.kind != F_NONE { return csr_empty(), f1; }
    let ptr, f2 = npz_index_member(b, ns, nl, me, us, of, "indptr", tag);
    if f2.kind != F_NONE { return csr_empty(), f2; }
    let vals, shp, f3 = npz_data_member(b, ns, nl, me, us, of, "data", tag);
    if f3.kind != F_NONE { return csr_empty(), f3; }
    if shp.len != 3 {
        return csr_empty(), fault(F_SHAPE, tag, "the values of blocks do not hold three lengths");
    }
    let br = shp[1];
    let bc = shp[2];
    if br < 1 || bc < 1 || rows % br != 0 || cols % bc != 0 {
        return csr_empty(), fault(F_SHAPE, tag, "the block size does not divide the shape");
    }
    let brows = rows / br;
    let bcols = cols / bc;
    let nblocks = idx.len;
    if ptr.len != brows + 1 {
        return csr_empty(), fault(F_SHAPE, tag, "the member of the pointer does not fit the shape");
    }
    if nblocks > MAX_SPARSE_NNZ {
        return csr_empty(), fault(F_LIMIT, tag, "the member holds more blocks than the limit");
    }
    let per = br * bc;
    if per > MAX_ELEMENTS || vals.len != nblocks * per {
        return csr_empty(), fault(F_SHAPE, tag, "the count of the values does not fit");
    }
    if !whole_index(ptr[0]) || ptr[0] != 0.0 {
        return csr_empty(), fault(F_SHAPE, tag, "the first pointer is not zero");
    }
    var i: i64 = 0;
    while i < brows {
        if !whole_index(ptr[i]) || !whole_index(ptr[i + 1]) || ptr[i] > ptr[i + 1] {
            return csr_empty(), fault(F_SHAPE, tag, "the pointers do not rise");
        }
        i += 1;
    }
    if !whole_index(ptr[brows]) || ptr[brows] != nblocks as f64 {
        return csr_empty(), fault(F_SHAPE, tag, "the last pointer differs from the count of the blocks");
    }
    i = 0;
    while i < nblocks {
        let c = idx[i];
        if !whole_index(c) || c < 0.0 || c >= bcols as f64 {
            return csr_empty(), fault(F_RANGE, tag, "a block stands outside the matrix");
        }
        i += 1;
    }
    var buf: f64[>..] = [];
    i = 0;
    while i < ptr.len { buf.push(ptr[i]); i += 1; }
    i = 0;
    while i < nblocks { buf.push(idx[i]); i += 1; }
    i = 0;
    while i < vals.len { buf.push(vals[i]); i += 1; }
    let m = Bsr { rows: rows, cols: cols, br: br, bc: bc, nblocks: nblocks, buf: copy(buf) };
    return bsr_to_csr(m);
}

// npz_read_dia builds the diagonals of the members of the format dia,
// and turns them into compressed rows. The value of a diagonal at the
// column j sits at the place j of the values of that diagonal. The
// values of one diagonal are therefore taken in the order of the row.
fn npz_read_dia(b: const u8[:], ns: i64[:], nl: i64[:], me: i64[:], us: i64[:],
                of: i64[:], rows: i64, cols: i64) -> Csr, Fault {
    let tag = "load_npz";
    let offs, f1 = npz_index_member(b, ns, nl, me, us, of, "offsets", tag);
    if f1.kind != F_NONE { return csr_empty(), f1; }
    let vals, shp, f3 = npz_data_member(b, ns, nl, me, us, of, "data", tag);
    if f3.kind != F_NONE { return csr_empty(), f3; }
    if shp.len != 2 || shp[0] != offs.len || shp[1] < 1 {
        return csr_empty(), fault(F_SHAPE, tag, "the member of the values does not fit the diagonals");
    }
    // The offsets are checked before any cast, because the cast of a
    // number outside the whole numbers gives no defined value.
    let oc = dia_check_offsets(rows, cols, offs, tag);
    if oc.kind != F_NONE { return csr_empty(), oc; }
    let wid = shp[1];
    if offs.len > MAX_SPARSE_NNZ {
        return csr_empty(), fault(F_LIMIT, tag, "the member holds more diagonals than the limit");
    }
    var dvals: f64[>..] = [];
    var k: i64 = 0;
    while k < offs.len {
        let off = offs[k] as i64;
        let first = dia_first_of(off);
        let len = dia_len_of(rows, cols, off);
        var t: i64 = 0;
        while t < len {
            let col = first + t + off;
            if col < wid { dvals.push(vals[k * wid + col]); } else { dvals.push(0.0); }
            t += 1;
        }
        k += 1;
    }
    let d, df = dia_from_diagonals(rows, cols, offs, dvals);
    if df.kind != F_NONE { return csr_empty(), df; }
    return dia_to_csr(d);
}

// npz_read_coo builds the coordinates of the members of the format
// coo, and turns them into compressed rows, where the entries of one
// place add up.
fn npz_read_coo(b: const u8[:], ns: i64[:], nl: i64[:], me: i64[:], us: i64[:],
                of: i64[:], rows: i64, cols: i64) -> Csr, Fault {
    let tag = "load_npz";
    let r, f1 = npz_index_member(b, ns, nl, me, us, of, "row", tag);
    if f1.kind != F_NONE { return csr_empty(), f1; }
    let c, f2 = npz_index_member(b, ns, nl, me, us, of, "col", tag);
    if f2.kind != F_NONE { return csr_empty(), f2; }
    let vals, shp, f3 = npz_data_member(b, ns, nl, me, us, of, "data", tag);
    if f3.kind != F_NONE { return csr_empty(), f3; }
    if shp.len != 1 || r.len != c.len || vals.len != r.len {
        return csr_empty(), fault(F_SHAPE, tag, "the members do not hold one count");
    }
    let m, mf = coo_from_triplets(rows, cols, r, c, vals);
    if mf.kind != F_NONE { return csr_empty(), mf; }
    return coo_to_csr(m);
}
