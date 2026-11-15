// The tests of the group sparse_npz. Run them with
// SCIENTIFIC_SUITES=sparse_npz.
//
// The pins come from SciPy 1.18.1 and NumPy 2.5.4 through the script
// test/gen_sparse_npz.py. The tests of the files write beside this
// program and remove what they write.
namespace sparse_npz_tests;

import scientific from "../..";
import os;
import common;
import sparse_npz_pins;

var NPZ_BAD: i64 = 0;

// u16_at and u32_at give the whole numbers of the bytes of an archive.
fn u16_at(b: const u8[:], at: i64) -> i64 {
    return (b[at] as i64) | ((b[at + 1] as i64) << 8);
}

fn u32_at(b: const u8[:], at: i64) -> i64 {
    return u16_at(b, at) | (u16_at(b, at + 2) << 16);
}

// span_bytes gives the bytes of one run of the pins.
fn span_bytes(bytes: f64[:], spans: f64[:], i: i64) -> u8[>..] {
    var out: u8[>..] = [];
    let at = spans[2 * i] as i64;
    let n = spans[2 * i + 1] as i64;
    var k: i64 = 0;
    while k < n { out.push(bytes[at + k] as u8); k += 1; }
    return out;
}

// dense_of gives the matrix of the pins of the archives.
fn dense_of() -> scientific::Arr {
    var s: i64[>..] = [];
    s.push(sparse_npz_pins::A_SHAPE[0] as i64);
    s.push(sparse_npz_pins::A_SHAPE[1] as i64);
    return common::arr_of(s, sparse_npz_pins::A_DENSE);
}

// same_dense tells whether the matrix of the pins is the matrix that
// the caller holds.
fn same_dense(a: scientific::Arr&) -> bool {
    if a.data.len != sparse_npz_pins::A_DENSE.len { return false; }
    var k: i64 = 0;
    while k < a.data.len {
        if a.data[k] != sparse_npz_pins::A_DENSE[k] { return false; }
        k += 1;
    }
    return true;
}

// test_sparse_npz_members writes an archive of the package and reads
// the directory of it. The names and the sizes of the members agree
// with the pins of SciPy.
fn test_sparse_npz_members() {
    let a = dense_of();
    let m, mf = scientific::csr_from_dense(a);
    assert(mf.kind == scientific::F_NONE);
    let ok, f = scientific::save_npz("sci_npz_sweep.npz", copy(m));
    assert(ok && f.kind == scientific::F_NONE);
    var z: u8[>..] = [];
    assert(read_file("sci_npz_sweep.npz", z));
    var at = z.len - 22;
    while at >= 0 && u32_at(z, at) != 101010256 { at -= 1; }
    assert(at >= 0);
    let count = u16_at(z, at + 10);
    if count != sparse_npz_pins::A_COUNTS[0] as i64 {
        print("the archive holds ", count, " members, the pin holds ",
              sparse_npz_pins::A_COUNTS[0], "\n");
        NPZ_BAD += 1;
    }
    var p = u32_at(z, at + 16);
    var k: i64 = 0;
    while k < count {
        let nl = u16_at(z, p + 28);
        let sz = u32_at(z, p + 24);
        let span = k * 2;
        let want_len = sparse_npz_pins::A_NAME_SPANS[span + 1] as i64;
        if nl != want_len {
            print("member ", k, " holds a name of ", nl, " bytes, the pin holds ",
                  want_len, "\n");
            NPZ_BAD += 1;
        } else {
            let ns = sparse_npz_pins::A_NAME_SPANS[span] as i64;
            var same = true;
            var q: i64 = 0;
            while q < nl {
                if z[p + 46 + q] != sparse_npz_pins::A_NAMES[ns + q] as u8 { same = false; }
                q += 1;
            }
            if !same {
                print("member ", k, " holds another name\n");
                NPZ_BAD += 1;
            }
        }
        let want_size = sparse_npz_pins::A_SIZES[k] as i64;
        if sz != want_size {
            print("member ", k, " holds ", sz, " bytes, the pin holds ",
                  want_size, "\n");
            NPZ_BAD += 1;
        }
        p += 46 + nl + u16_at(z, p + 30) + u16_at(z, p + 32);
        k += 1;
    }
    assert(delete_file("sci_npz_sweep.npz"));
    assert(NPZ_BAD == 0);
}

// test_sparse_npz_round writes a matrix and reads it back. The matrix
// that comes back holds the values of the matrix that went out.
fn test_sparse_npz_round() {
    let a = dense_of();
    let m, mf = scientific::csr_from_dense(a);
    assert(mf.kind == scientific::F_NONE);
    let ok, f = scientific::save_npz("sci_npz_round.npz", copy(m));
    assert(ok && f.kind == scientific::F_NONE);
    let b, fb = scientific::load_npz("sci_npz_round.npz");
    if fb.kind != scientific::F_NONE {
        print("load_npz fault ", fb.kind, "\n");
        NPZ_BAD += 1;
    } else {
        let d, fd = scientific::csr_to_dense(b);
        assert(fd.kind == scientific::F_NONE);
        if !same_dense(d) {
            print("the matrix that came back holds other values\n");
            NPZ_BAD += 1;
        }
    }
    assert(delete_file("sci_npz_round.npz"));
    assert(NPZ_BAD == 0);
}

// test_sparse_npz_load_scipy reads the five archives of SciPy. Every
// archive holds the same matrix, and the matrix of the pins is that
// matrix.
fn test_sparse_npz_load_scipy() {
    let count = sparse_npz_pins::A_COUNTS.len;
    var i: i64 = 0;
    while i < count {
        let arch = span_bytes(sparse_npz_pins::Z_BYTES, sparse_npz_pins::Z_SPANS, i);
        assert(write_file("sci_npz_in.npz", arch));
        let b, f = scientific::load_npz("sci_npz_in.npz");
        if f.kind != scientific::F_NONE {
            print("load_npz of the archive ", i, " fault ", f.kind, "\n");
            NPZ_BAD += 1;
        } else {
            let d, fd = scientific::csr_to_dense(b);
            assert(fd.kind == scientific::F_NONE);
            if !same_dense(d) {
                print("the archive ", i, " holds other values\n");
                NPZ_BAD += 1;
            }
        }
        assert(delete_file("sci_npz_in.npz"));
        i += 1;
    }
    assert(NPZ_BAD == 0);
}

// test_sparse_npz_spilu solves a right side with the incomplete
// factorization of the package. The answer agrees with the answer of
// SciPy to a relative tolerance of 1e-8.
fn test_sparse_npz_spilu() {
    let count = sparse_npz_pins::S_CASES.len / 7;
    var i: i64 = 0;
    while i < count {
        let r = i * 7;
        let n = sparse_npz_pins::S_CASES[r] as i64;
        let nnz = sparse_npz_pins::S_CASES[r + 1] as i64;
        let at = sparse_npz_pins::S_CASES[r + 2] as i64;
        let b_at = sparse_npz_pins::S_CASES[r + 3] as i64;
        let x_at = sparse_npz_pins::S_CASES[r + 4] as i64;
        var ri: f64[>..] = [];
        var ci: f64[>..] = [];
        var vv: f64[>..] = [];
        var k: i64 = 0;
        while k < nnz {
            ri.push(sparse_npz_pins::S_ROWS[at + k]);
            ci.push(sparse_npz_pins::S_COLS[at + k]);
            vv.push(sparse_npz_pins::S_VALS[at + k]);
            k += 1;
        }
        let m, mf = scientific::csr_from_triplets(n, n, ri, ci, vv);
        assert(mf.kind == scientific::F_NONE);
        let fct, ff = scientific::spilu(m, sparse_npz_pins::S_CASES[r + 5],
            sparse_npz_pins::S_CASES[r + 6], "basic");
        if ff.kind != scientific::F_NONE {
            print("spilu case ", i, " fault ", ff.kind, "\n");
            NPZ_BAD += 1;
        } else {
            var b: f64[>..] = [];
            k = 0;
            while k < n { b.push(sparse_npz_pins::S_B[b_at + k]); k += 1; }
            let x, fx = scientific::splu_solve(fct, b);
            assert(fx.kind == scientific::F_NONE);
            // The size of the answer gives the relative tolerance.
            var scale: f64 = 0.0;
            k = 0;
            while k < n {
                var a = sparse_npz_pins::S_X[x_at + k];
                if a < 0.0 { a = -a; }
                if a > scale { scale = a; }
                k += 1;
            }
            k = 0;
            while k < n {
                var d = x.data[k] - sparse_npz_pins::S_X[x_at + k];
                if d < 0.0 { d = -d; }
                if d > 1.0e-8 * scale {
                    print("spilu case ", i, " holds ", x.data[k], " at ", k,
                          " and the pin holds ", sparse_npz_pins::S_X[x_at + k], "\n");
                    NPZ_BAD += 1;
                }
                k += 1;
            }
        }
        i += 1;
    }
    assert(NPZ_BAD == 0);
}

// test_sparse_npz_drop checks the rule of the drop on a matrix whose
// elimination makes a fill. A high tolerance drops that fill, so the
// factor holds fewer values than the factor of a tolerance of zero,
// and the solve still runs.
fn test_sparse_npz_drop() {
    let m, mf = scientific::csr_from_triplets(3, 3,
        [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        [0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0],
        [4.0, 1.0, 2.0, 1.0, 5.0, 2.0, 6.0]);
    assert(mf.kind == scientific::F_NONE);
    let keep, f0 = scientific::spilu(m, 0.0, 10.0, "basic");
    assert(f0.kind == scientific::F_NONE);
    let drop, f1 = scientific::spilu(m, 0.5, 10.0, "basic");
    assert(f1.kind == scientific::F_NONE);
    if scientific::splu_nnz(drop) >= scientific::splu_nnz(keep) {
        print("the rule of the drop holds no entry back\n");
        NPZ_BAD += 1;
    }
    let x, fx = scientific::splu_solve(drop, [1.0, 2.0, 3.0]);
    assert(fx.kind == scientific::F_NONE);
    assert(NPZ_BAD == 0);
}

// test_sparse_npz_spilu_faults checks the faults of the incomplete
// factorization.
fn test_sparse_npz_spilu_faults() {
    let m, mf = scientific::csr_from_triplets(2, 2,
        [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [4.0, 1.0, 1.0, 4.0]);
    assert(mf.kind == scientific::F_NONE);
    // A drop tolerance outside zero to one.
    let _a, f1 = scientific::spilu(m, -1.0, 10.0, "basic");
    assert(f1.kind == scientific::F_VALUE);
    let _b, f2 = scientific::spilu(m, 2.0, 10.0, "basic");
    assert(f2.kind == scientific::F_VALUE);
    // A fill factor below one.
    let _c, f3 = scientific::spilu(m, 0.0, 0.5, "basic");
    assert(f3.kind == scientific::F_VALUE);
    // A rule that the package does not hold.
    let _d, f4 = scientific::spilu(m, 0.0, 10.0, "area");
    assert(f4.kind == scientific::F_VALUE);
    // A matrix that is not square.
    let r, rf = scientific::csr_from_triplets(1, 2, [0.0], [0.0], [1.0]);
    assert(rf.kind == scientific::F_NONE);
    let _e, f5 = scientific::spilu(r, 0.0, 10.0, "basic");
    assert(f5.kind == scientific::F_SHAPE);
    // A matrix that needs a row exchange.
    let s, sf = scientific::csr_from_triplets(2, 2,
        [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]);
    assert(sf.kind == scientific::F_NONE);
    let _f, f6 = scientific::spilu(s, 0.0, 10.0, "basic");
    assert(f6.kind == scientific::F_VALUE);
    // A matrix whose elimination leaves a pivot of zero.
    let t, tf = scientific::csr_from_triplets(2, 2,
        [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
    assert(tf.kind == scientific::F_NONE);
    let _g, f7 = scientific::spilu(t, 0.0, 10.0, "basic");
    assert(f7.kind == scientific::F_VALUE);
    // A matrix of a forged layout gives a fault before any value is read.
    var fbuf: f64[>..] = [];
    var k4: i64 = 0;
    while k4 < 4 { fbuf.push(0.0); k4 += 1; }
    let forged = scientific::Csr { rows: 2, cols: 2, nnz: 1, buf: copy(fbuf) };
    let _h, f8 = scientific::spilu(forged, 0.0, 10.0, "basic");
    assert(f8.kind == scientific::F_SHAPE);
    assert(NPZ_BAD == 0);
}

// npz_build writes an archive of the parts that the test gives. The
// parts follow the types of SciPy. The type of the values names the
// type that the head of the values claims.
fn npz_build(path: const u8[:], rows: i64, cols: i64, fmt: const u8[:],
             descr: const u8[:], ptr: f64[:], idx: f64[:], vals: f64[:]) -> bool, scientific::Fault {
    var wide = false;
    var i: i64 = 0;
    while i < ptr.len {
        if ptr[i] > 2147483647.0 { wide = true; }
        i += 1;
    }
    i = 0;
    while i < idx.len {
        if idx[i] > 2147483647.0 { wide = true; }
        i += 1;
    }
    var idescr: u8[>..] = [];
    idescr.append("<i");
    if wide { idescr.push('8'); } else { idescr.push('4'); }
    var dn_nnz: i64[>..] = [];
    dn_nnz.push(idx.len);
    var dn_ptr: i64[>..] = [];
    dn_ptr.push(ptr.len);
    var dn_two: i64[>..] = [];
    dn_two.push(2);
    var dn_zero: i64[>..] = [];
    var hidx = scientific::npz_head(idescr, dn_nnz);
    var bi = scientific::npz_body_i(idx, wide);
    hidx.append(bi[..]);
    var hptr = scientific::npz_head(idescr, dn_ptr);
    var bp = scientific::npz_body_i(ptr, wide);
    hptr.append(bp[..]);
    var hfmt = scientific::npz_head("|S3", dn_zero);
    hfmt.append(fmt);
    var hshape = scientific::npz_head("<i8", dn_two);
    var sb: u8[>..] = [];
    scientific::npz_put_i64(sb, rows);
    scientific::npz_put_i64(sb, cols);
    hshape.append(sb[..]);
    var hdata = scientific::npz_head(descr, dn_nnz);
    var bv = scientific::npz_body_f64(vals);
    hdata.append(bv[..]);
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
    return scientific::npz_write(path, names, bodies, "npz_tests");
}

// test_sparse_npz_faults checks the faults of the reader and of the
// writer of the archive.
fn test_sparse_npz_faults() {
    // A file that holds no archive.
    assert(write_file("sci_npz_bad.npz", "no archive here"));
    let _a, f1 = scientific::load_npz("sci_npz_bad.npz");
    assert(f1.kind == scientific::F_VALUE);
    // A path with a NUL byte is refused before any read.
    var p: u8[>..] = [];
    p.append("sci_npz");
    p.push(0);
    p.append(".npz");
    let _n, f0 = scientific::load_npz(p[..]);
    assert(f0.kind == scientific::F_VALUE);
    // A file that does not read.
    let _b, f2 = scientific::load_npz("sci_npz_none.npz");
    assert(f2.kind == scientific::F_VALUE);
    // A file that is cut behind the record of the end.
    let ok0, ff0 = npz_build("sci_npz_full.npz", 2, 2, "csr", "<f8",
        [0.0, 1.0, 2.0], [0.0, 1.0], [4.0, 5.0]);
    assert(ok0 && ff0.kind == scientific::F_NONE);
    var z: u8[>..] = [];
    assert(read_file("sci_npz_full.npz", z));
    assert(write_file("sci_npz_bad.npz", z[..z.len - 10]));
    let _c, f3 = scientific::load_npz("sci_npz_bad.npz");
    assert(f3.kind == scientific::F_VALUE);
    // A member that is compressed. The kind of the storage stands in
    // the head of the member and in the directory of the archive.
    assert(read_file("sci_npz_full.npz", z));
    var at = z.len - 22;
    while at >= 0 && u32_at(z, at) != 101010256 { at -= 1; }
    assert(at >= 0);
    let cd = u32_at(z, at + 16);
    let off = u32_at(z, cd + 42);
    z[cd + 10] = 8;
    z[off + 8] = 8;
    assert(write_file("sci_npz_bad.npz", z));
    let _d, f4 = scientific::load_npz("sci_npz_bad.npz");
    assert(f4.kind == scientific::F_VALUE);
    // A name with a byte outside the letters and the digits.
    assert(read_file("sci_npz_full.npz", z));
    z[cd + 46] = '/';
    z[off + 30] = '/';
    assert(write_file("sci_npz_bad.npz", z));
    let _e, f5 = scientific::load_npz("sci_npz_bad.npz");
    assert(f5.kind == scientific::F_VALUE);
    // Values that claim the type of the whole numbers.
    let ok1, ff1 = npz_build("sci_npz_bad.npz", 2, 2, "csr", "<i4",
        [0.0, 1.0, 2.0], [0.0, 1.0], [4.0, 5.0]);
    assert(ok1 && ff1.kind == scientific::F_NONE);
    let _f, f6 = scientific::load_npz("sci_npz_bad.npz");
    assert(f6.kind == scientific::F_VALUE);
    // A pointer that does not end at the count of the values.
    let ok2, ff2 = npz_build("sci_npz_bad.npz", 2, 2, "csr", "<f8",
        [0.0, 5.0, 2.0], [0.0, 1.0], [4.0, 5.0]);
    assert(ok2 && ff2.kind == scientific::F_NONE);
    let _g, f7 = scientific::load_npz("sci_npz_bad.npz");
    assert(f7.kind == scientific::F_SHAPE);
    // An index that stands outside the matrix.
    let ok3, ff3 = npz_build("sci_npz_bad.npz", 2, 2, "csr", "<f8",
        [0.0, 1.0, 2.0], [0.0, 9.0], [4.0, 5.0]);
    assert(ok3 && ff3.kind == scientific::F_NONE);
    let _h, f8 = scientific::load_npz("sci_npz_bad.npz");
    assert(f8.kind == scientific::F_RANGE);
    // Columns that do not rise inside one row.
    let ok4, ff4 = npz_build("sci_npz_bad.npz", 2, 2, "csr", "<f8",
        [0.0, 2.0, 2.0], [1.0, 0.0], [4.0, 5.0]);
    assert(ok4 && ff4.kind == scientific::F_NONE);
    let _i, f9 = scientific::load_npz("sci_npz_bad.npz");
    assert(f9.kind == scientific::F_SHAPE);
    // A value that is not finite.
    let ok5, ff5 = npz_build("sci_npz_bad.npz", 2, 2, "csr", "<f8",
        [0.0, 1.0, 2.0], [0.0, 1.0], [4.0, 0.0 / 0.0]);
    assert(ok5 && ff5.kind == scientific::F_NONE);
    let _j, f10 = scientific::load_npz("sci_npz_bad.npz");
    assert(f10.kind == scientific::F_VALUE);
    // A name of a format that no archive of the five holds.
    let ok6, ff6 = npz_build("sci_npz_bad.npz", 2, 2, "xyz", "<f8",
        [0.0, 1.0, 2.0], [0.0, 1.0], [4.0, 5.0]);
    assert(ok6 && ff6.kind == scientific::F_NONE);
    let _k, f11 = scientific::load_npz("sci_npz_bad.npz");
    assert(f11.kind == scientific::F_VALUE);
    // An archive with more members than the limit.
    var names: const u8[:][>..] = [];
    var bodies: const u8[:][>..] = [];
    var empty: u8[>..] = [];
    // The names of the members live in one buffer, so every slice of
    // a name outlives the list.
    var all: u8[>..] = [];
    var nlens: i64[>..] = [];
    var k2: i64 = 0;
    while k2 < 257 {
        let before = all.len;
        all.append("m");
        scientific::npy_put_int(all, k2);
        nlens.push(all.len - before);
        bodies.push(empty[..]);
        k2 += 1;
    }
    var at: i64 = 0;
    k2 = 0;
    while k2 < 257 {
        names.push(all[at..at + nlens[k2]]);
        at += nlens[k2];
        k2 += 1;
    }
    let ok7, ff7 = scientific::npz_write("sci_npz_bad.npz", names, bodies, "npz_tests");
    assert(ok7 && ff7.kind == scientific::F_NONE);
    let _l, f12 = scientific::load_npz("sci_npz_bad.npz");
    assert(f12.kind == scientific::F_LIMIT);
    // A matrix of a forged layout is a fault of the writer.
    var fbuf: f64[>..] = [];
    var k3: i64 = 0;
    while k3 < 9 { fbuf.push(0.0); k3 += 1; }
    fbuf[1] = 1.0;
    fbuf[2] = 0.0;
    let forged = scientific::Csr { rows: 2, cols: 2, nnz: 3, buf: copy(fbuf) };
    let ok8, f13 = scientific::save_npz("sci_npz_bad.npz", copy(forged));
    assert(!ok8 && f13.kind == scientific::F_VALUE);
    // A path with a NUL byte is refused before any write.
    var p2: u8[>..] = [];
    p2.append("sci_npz");
    p2.push(0);
    p2.append(".npz");
    let good = dense_of();
    let gm, gmf = scientific::csr_from_dense(good);
    assert(gmf.kind == scientific::F_NONE);
    let ok9, f14 = scientific::save_npz(p2[..], copy(gm));
    assert(!ok9 && f14.kind == scientific::F_VALUE);
    assert(delete_file("sci_npz_bad.npz"));
    assert(delete_file("sci_npz_full.npz"));
    assert(NPZ_BAD == 0);
}
