// The blocks of a tridiagonal matrix.
//
// A tridiagonal matrix with a zero or a negligible off diagonal value
// falls into blocks. Each block holds its own eigenvalues and its own
// eigenvectors. The call eigh_tridiagonal solves the blocks one by one
// and merges the pairs by value.
namespace scientific;

import mathshim;

// The epsilon of the machine.
const BD_EPS: f64 = 2.220446049250313e-16;

// bd_split_at tells whether the off diagonal value e[i] divides the matrix
// into two blocks. The value divides when it is zero. It also divides
// when it is negligible against the scale of its place. The scale of the
// place is the sum of the two diagonal values next to it. The scale of
// the whole matrix does not enter the rule. A coupling that a large value
// elsewhere hides must not be dropped, because it can still be the whole
// story of the values next to it.
fn bd_split_at(d: f64[:], e: f64[:], i: i64) -> bool {
    if e[i] == 0.0 { return true; }
    // The scale of the place is the half sum of the two diagonal values,
    // and the limit is twice that. The half sum of two values near the
    // limit of the number format does not overflow.
    let sc = mathshim::abs_of(d[i]) * 0.5 + mathshim::abs_of(d[i + 1]) * 0.5;
    return mathshim::abs_of(e[i]) <= BD_EPS * sc * 2.0;
}

// bd_has_split tells whether the matrix falls into more than one block.
fn bd_has_split(d: f64[:], e: f64[:], n: i64) -> bool {
    var i: i64 = 0;
    while i < e.len {
        if bd_split_at(d, e, i) { return true; }
        i += 1;
    }
    return false;
}

// bd_blocks gives the start and the length of every block of the matrix.
// The lists hold one entry per block, in the order of the rows.
fn bd_blocks(d: f64[:], e: f64[:], n: i64) -> i64[>..], i64[>..] {
    var starts: i64[>..] = [];
    var lens: i64[>..] = [];
    var a: i64 = 0;
    var i: i64 = 0;
    while i < e.len {
        if bd_split_at(d, e, i) {
            starts.push(a);
            lens.push(i + 1 - a);
            a = i + 1;
        }
        i += 1;
    }
    starts.push(a);
    lens.push(n - a);
    return starts, lens;
}

// bd_block_values gives the values of the block of the rows a to a plus
// len minus one. A block of one row gives its diagonal value.
fn bd_block_values(d: f64[:], e: f64[:], a: i64, len: i64) -> f64[>..] {
    if len == 1 {
        var one: f64[>..] = [];
        one.push(d[a]);
        return one;
    }
    return bd_rank_all(d[a..a + len], e[a..a + len - 1], len, 0, len - 1);
}

// bd_block_vectors gives the vectors of the block of the rows a to a plus
// len minus one. It gives the cnt vectors of the first cnt values of the
// block. The vectors sit in the columns of a buffer of len by cnt values.
fn bd_block_vectors(d: f64[:], e: f64[:], a: i64, len: i64, vals: f64[:], cnt: i64) -> f64[>..] {
    var out: f64[>..] = [];
    if len == 1 {
        out.push(1.0);
        return out;
    }
    let dblk = d[a..a + len];
    let eblk = e[a..a + len - 1];
    let piv = bd_pivot(dblk, eblk);
    var prev: f64[>..] = [];
    var r: i64 = 0;
    while r < cnt {
        var vec: f64[>..] = [];
        bd_vector(dblk, eblk, len, vals[r], piv, prev, r, vec);
        var i: i64 = 0;
        while i < len { out.push(vec[i]); i += 1; }
        var j: i64 = 0;
        while j < len { prev.push(vec[j]); j += 1; }
        r += 1;
    }
    return out;
}

// bd_merge_order sorts the pairs of all blocks by their values. The pairs
// of an equal value keep the order of their blocks. The list holds the
// place of every pair.
fn bd_merge_order(vals: f64[:]) -> i64[>..] {
    var order: i64[>..] = [];
    var i: i64 = 0;
    while i < vals.len { order.push(i); i += 1; }
    i = 1;
    while i < order.len {
        let keep = order[i];
        let kv = vals[keep];
        var j: i64 = i - 1;
        while j >= 0 && vals[order[j]] > kv {
            order[j + 1] = order[j];
            j -= 1;
        }
        order[j + 1] = keep;
        i += 1;
    }
    return order;
}

// bd_eigh_split gives the values and the vectors of a matrix that falls
// into blocks. Each block is solved alone. The pairs of all blocks merge
// by value, so the merged list holds the ranks of the whole matrix. The
// pairs of the ranks k1 to k2 come back. The values rise. The vectors
// stand in the columns of a buffer of n by count values.
fn bd_eigh_split(d: f64[:], e: f64[:], n: i64, k1: i64, k2: i64, with_vectors: bool) -> f64[>..], f64[>..] {
    let starts, lens = bd_blocks(d, e, n);
    let nb = starts.len;
    // The value of every pair, with the block and the rank inside it.
    var vals: f64[>..] = [];
    var owner: i64[>..] = [];
    var local: i64[>..] = [];
    // The place of the first pair of each block in the list of the pairs.
    var pstart: i64[>..] = [];
    var b: i64 = 0;
    while b < nb {
        let a = starts[b];
        let len = lens[b];
        pstart.push(vals.len);
        let bv = bd_block_values(d, e, a, len);
        var k: i64 = 0;
        while k < len {
            vals.push(bv[k]);
            owner.push(b);
            local.push(k);
            k += 1;
        }
        b += 1;
    }
    let order = bd_merge_order(vals);
    var w: f64[>..] = [];
    var z: f64[>..] = [];
    if k1 > k2 { return w, z; }
    let count = k2 - k1 + 1;
    if with_vectors { z = nd_zeros(n * count); }
    // The number of vectors that each block must give.
    var need: i64[>..] = [];
    var off: i64[>..] = [];
    b = 0;
    while b < nb { need.push(0); off.push(0); b += 1; }
    var c: i64 = 0;
    while c < count {
        let p = order[k1 + c];
        let bb = owner[p];
        let r = local[p] + 1;
        if r > need[bb] { need[bb] = r; }
        c += 1;
    }
    // The vectors of the blocks that hold a wanted pair.
    var buf: f64[>..] = [];
    if with_vectors {
        b = 0;
        while b < nb {
            off[b] = buf.len;
            let a = starts[b];
            let len = lens[b];
            if need[b] > 0 {
                let bvals: f64[:] = vals[pstart[b]..pstart[b] + len];
                let bv = bd_block_vectors(d, e, a, len, bvals, need[b]);
                var q: i64 = 0;
                while q < bv.len { buf.push(bv[q]); q += 1; }
            }
            b += 1;
        }
    }
    // The wanted pairs, in the order of their values.
    c = 0;
    while c < count {
        let p = order[k1 + c];
        w.push(vals[p]);
        if with_vectors {
            let bb = owner[p];
            let r = local[p];
            let a = starts[bb];
            let len = lens[bb];
            var q: i64 = 0;
            while q < len {
                z[(a + q) * count + c] = buf[off[bb] + r * len + q];
                q += 1;
            }
        }
        c += 1;
    }
    return w, z;
}
