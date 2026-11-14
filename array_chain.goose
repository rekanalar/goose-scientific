// The product of a chain of matrices, as numpy.linalg.multi_dot.
//
// Goose has no array of arrays, so the caller puts the arrays into a
// Chain one by one and then asks for the product. The call picks the
// order of the products by the cost of the scalar multiplications. A
// first array of one dimension reads as a row. A last array of one
// dimension reads as a column.
namespace scientific;

// MAX_CHAIN is the largest count of arrays that a chain holds.
const MAX_CHAIN: i64 = 256;

// COST_MAX is the value at which the costs of the order stop. A cost
// that stands at this value never wraps into a small number.
const COST_MAX: i64 = 9223372036854775807;

// Chain holds the arrays of a chain product. The lists hold one entry
// for each array: the count of dimensions, the rows, the columns and the
// place of the values in the buffer. An array of one dimension holds
// its length in both the rows and the columns. The buffer is the last
// field.
struct Chain {
    count: i64,
    nd: i64[..MAX_CHAIN],
    rows: i64[..MAX_CHAIN],
    cols: i64[..MAX_CHAIN],
    off: i64[..MAX_CHAIN],
    data: f64[>..]
}

// chain_new gives a chain that holds no array.
fn chain_new() -> Chain {
    return Chain { count: 0, nd: [], rows: [], cols: [], off: [], data: [] };
}

// chain_add puts one array into the chain. The chain grows, so the
// caller keeps the new chain in a name of its own. An array that does not
// fit gives a fault and the chain as it was.
fn chain_add(c: Chain&, a) -> Chain, Fault {
    let tag = "chain_add";
    // The chain is public, so a caller can forge one. The lists and the
    // count of a forged chain must agree, or the push below runs past
    // the capacity of a list.
    if c.count < 0 || c.nd.len != c.count || c.rows.len != c.count || c.cols.len != c.count || c.off.len != c.count {
        return copy(c), fault(F_SHAPE, tag, "the lists of the chain do not match its count");
    }
    if c.count >= MAX_CHAIN {
        return copy(c), fault(F_LIMIT, tag, "the chain holds more arrays than the limit");
    }
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if f.kind != F_NONE { return copy(c), f; }
    if a.ndim < 1 || a.ndim > 2 {
        return copy(c), fault(F_SHAPE, tag, "an array of a chain holds one or two dimensions");
    }
    if !all_finite(a.data) {
        return copy(c), fault(F_VALUE, tag, "a value of the array is not finite");
    }
    if c.data.len + a.data.len > MAX_ELEMENTS {
        return copy(c), fault(F_LIMIT, tag, "the chain holds more values than the limit");
    }
    var out = Chain { count: c.count + 1, nd: c.nd, rows: c.rows, cols: c.cols, off: c.off, data: copy(c.data) };
    out.nd.push(a.ndim);
    out.rows.push(a.shape[0]);
    if a.ndim == 2 {
        out.cols.push(a.shape[1]);
    } else {
        out.cols.push(1);
    }
    out.off.push(out.data.len);
    out.data.append(a.data);
    return out, no_fault();
}

// sat_mul multiplies two counts and stops at COST_MAX.
fn sat_mul(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 { return 0; }
    if a > COST_MAX / b { return COST_MAX; }
    return a * b;
}

// sat_add adds two counts and stops at COST_MAX.
fn sat_add(a: i64, b: i64) -> i64 {
    if a > COST_MAX - b { return COST_MAX; }
    return a + b;
}

// mdot_order fills s with the split of every part of the chain, by the
// dynamic program of Cormen. The sizes p hold the rows of each array
// and the columns of the last one. The first split of the lowest cost
// wins, as in NumPy.
fn mdot_order(p: i64[:], n: i64, s: i64[>..]&) {
    var m: i64[>..] = [];
    var i: i64 = 0;
    while i < n * n { m.push(0); s.push(0); i += 1; }
    var l: i64 = 1;
    while l < n {
        i = 0;
        while i < n - l {
            let j = i + l;
            var best = COST_MAX;
            var bk = i;
            var k = i;
            while k < j {
                let step = sat_mul(sat_mul(p[i], p[k + 1]), p[j + 1]);
                let q = sat_add(sat_add(m[i * n + k], m[(k + 1) * n + j]), step);
                if k == i || q < best { best = q; bk = k; }
                k += 1;
            }
            m[i * n + j] = best;
            s[i * n + j] = bk;
            i += 1;
        }
        l += 1;
    }
}

// mdot_take copies len values from the chain buffer or from the pool.
fn mdot_take(c: Chain&, pool: f64[>..]&, src: i64, off: i64, len: i64) -> f64[>..] {
    if src == 0 { return copy_slice(c.data[off..off + len]); }
    return copy_slice(pool[off..off + len]);
}

// mdot_run multiplies the arrays i to j by the splits. It gives the
// place of the result: the source 0 is the chain buffer and the source
// 1 is the pool, which holds the results of the products.
recursive fn mdot_run(c: Chain&, p: i64[:], s: i64[:], n: i64, i: i64, j: i64, pool: f64[>..]&) -> i64, i64, Fault {
    if i == j { return 0, c.off[i], no_fault(); }
    let k = s[i * n + j];
    let ls, lo, lf = mdot_run(c, p, s, n, i, k, pool);
    if lf.kind != F_NONE { return 0, 0, lf; }
    let rs, ro, rf = mdot_run(c, p, s, n, k + 1, j, pool);
    if rf.kind != F_NONE { return 0, 0, rf; }
    let size = sat_mul(p[i], p[j + 1]);
    if size > MAX_ELEMENTS || pool.len + size > MAX_ELEMENTS {
        return 0, 0, fault(F_LIMIT, "multi_dot", "a product holds more values than the limit");
    }
    let x = mat2(mdot_take(c, pool, ls, lo, p[i] * p[k + 1]), p[i], p[k + 1]);
    let y = mat2(mdot_take(c, pool, rs, ro, p[k + 1] * p[j + 1]), p[k + 1], p[j + 1]);
    let z, zf = matmul(x, y);
    if zf.kind != F_NONE { return 0, 0, zf; }
    let at = pool.len;
    pool.append(z.data);
    return 1, at, no_fault();
}

// mdot_check tests a chain against its own lists before any read. The
// chain is public, so a caller can forge one.
fn mdot_check(c: Chain&) -> Fault {
    let tag = "multi_dot";
    let n = c.count;
    if n < 2 { return fault(F_SHAPE, tag, "the chain holds fewer than two arrays"); }
    if n > MAX_CHAIN || c.nd.len != n || c.rows.len != n || c.cols.len != n || c.off.len != n {
        return fault(F_SHAPE, tag, "the lists of the chain do not match its count");
    }
    var i: i64 = 0;
    while i < n {
        if c.nd[i] < 1 || c.nd[i] > 2 || c.rows[i] < 1 || c.cols[i] < 1 || c.off[i] < 0 || c.off[i] > c.data.len {
            return fault(F_SHAPE, tag, "the chain holds an array that does not match its lists");
        }
        var len = sat_mul(c.rows[i], c.cols[i]);
        if c.nd[i] == 1 { len = c.rows[i]; }
        if len > MAX_ELEMENTS {
            return fault(F_LIMIT, tag, "an array of the chain holds more values than the limit");
        }
        if len > c.data.len - c.off[i] {
            return fault(F_SHAPE, tag, "the chain holds an array that does not match its buffer");
        }
        if i > 0 && i < n - 1 && c.nd[i] != 2 {
            return fault(F_SHAPE, tag, "an array in the middle of the chain holds not two dimensions");
        }
        i += 1;
    }
    if !all_finite(c.data) { return fault(F_VALUE, tag, "a value of the chain is not finite"); }
    return no_fault();
}

// multi_dot gives the product of the arrays of a chain. The chain holds
// at least two arrays. The first and the last one may hold one
// dimension. The others hold two. Two vector ends give one element.
fn multi_dot(c: Chain&) -> Arr, Fault {
    let cf = mdot_check(c);
    if cf.kind != F_NONE { return empty_arr(), cf; }
    let n = c.count;
    let first1 = c.nd[0] == 1;
    let last1 = c.nd[n - 1] == 1;
    // The rows and the columns of every array after the ends have their
    // place. A first array of one dimension is a row of one line. A last
    // one is a column of one column.
    var rr: i64[>..] = [];
    var cc: i64[>..] = [];
    var i: i64 = 0;
    while i < n {
        var r = c.rows[i];
        var k = c.cols[i];
        if c.nd[i] == 1 {
            if i == 0 { r = 1; k = c.rows[i]; }
            if i == n - 1 { r = c.rows[i]; k = 1; }
        }
        rr.push(r);
        cc.push(k);
        i += 1;
    }
    // The sizes: the rows of each array, then the columns of the last.
    var p: i64[>..] = [];
    i = 0;
    while i < n { p.push(rr[i]); i += 1; }
    p.push(cc[n - 1]);
    // The neighbors must agree on the inner size.
    i = 0;
    while i < n - 1 {
        if cc[i] != rr[i + 1] {
            return empty_arr(), fault(F_SHAPE, "multi_dot", "the arrays do not align");
        }
        i += 1;
    }
    var s: i64[>..] = [];
    mdot_order(p, n, s);
    var pool: f64[>..] = [];
    let src, at, rf = mdot_run(c, p, s, n, 0, n - 1, pool);
    if rf.kind != F_NONE { return empty_arr(), rf; }
    var shape: i64[>..] = [];
    if first1 && last1 { shape.push(1); }
    else if first1 { shape.push(p[n]); }
    else if last1 { shape.push(p[0]); }
    else { shape.push(p[0]); shape.push(p[n]); }
    return arr_of(mdot_take(c, pool, src, at, p[0] * p[n]), shape), no_fault();
}
