// The kept pairs of the restarted solvers lgmres and gcrotmk.
//
// A kept pair holds two vectors of n values. lgmres keeps the pairs (dx, ax)
// of its outer iterations. gcrotmk keeps the pairs (c, u). One vector of a
// pair may be absent. A flag tells which one: for lgmres it stands for the
// second vector (ax), and for gcrotmk for the first one (c).
namespace scientific;

import mathshim;

// The largest count of kept pairs that a call takes as input.
const MAX_KEPT_PAIRS: i64 = 64;

struct Kept { count: i64, n: i64, buf: f64[>..] }

fn kept_empty() -> Kept {
    return Kept { count: 0, n: 0, buf: [] };
}

// kept_stride gives the count of values of one pair of vectors of n values.
fn kept_stride(n: i64) -> i64 {
    return 2 * n + 1;
}

// kept_base gives the place of the flag of the pair p.
fn kept_base(p: i64, n: i64) -> i64 {
    return p * kept_stride(n);
}

// kept_first_off gives the place of the first vector of the pair p.
fn kept_first_off(p: i64, n: i64) -> i64 {
    return kept_base(p, n) + 1;
}

// kept_second_off gives the place of the second vector of the pair p.
fn kept_second_off(p: i64, n: i64) -> i64 {
    return kept_base(p, n) + 1 + n;
}

// kept_check checks a list of kept pairs against the side n of the system.
fn kept_check(kp, n: i64, tag: const u8[:]) -> Fault {
    if kp.count == 0 && kp.buf.len == 0 { return no_fault(); }
    if kp.count < 0 || kp.n != n { return fault(F_SHAPE, tag, "the kept pairs do not fit the matrix"); }
    if kp.count > MAX_KEPT_PAIRS { return fault(F_LIMIT, tag, "the count of kept pairs stands above the limit"); }
    if kp.buf.len != kp.count * kept_stride(n) { return fault(F_SHAPE, tag, "the buffer of the kept pairs has the wrong length"); }
    var p: i64 = 0;
    while p < kp.count {
        let flag = kp.buf[kept_base(p, n)];
        if flag != 0.0 && flag != 1.0 { return fault(F_VALUE, tag, "a flag of a kept pair is neither zero nor one"); }
        p += 1;
    }
    if !lg_finite(kp.buf, kp.buf.len) { return fault(F_VALUE, tag, "a kept pair holds a value that is not finite"); }
    return no_fault();
}

// precond_present tells whether the call applies a preconditioner. An empty
// matrix means none.

fn kept_put(buf: f64[>..]&, p: i64, n: i64, flag: f64, a: f64[:], ao: i64, c: f64[:], co: i64) {
    buf[kept_base(p, n)] = flag;
    let fa = kept_first_off(p, n);
    let fb = kept_second_off(p, n);
    var i: i64 = 0;
    while i < n {
        buf[fa + i] = a[ao + i];
        buf[fb + i] = c[co + i];
        i += 1;
    }
}

// kept_copy gives a copy of a list of kept pairs.

fn kept_copy(kp) -> Kept {
    return Kept { count: kp.count, n: kp.n, buf: copy_slice(kp.buf) };
}

// kept_drop_oldest removes the drop oldest rows of the two buffers of kept
// vectors, which hold count rows of n values, and shifts the rest to the
// front.
fn kept_drop_oldest(a: f64[>..]&, b: f64[>..]&, count: i64, drop: i64, n: i64) {
    var slot: i64 = drop;
    while slot < count {
        var i: i64 = 0;
        while i < n {
            a[(slot - drop) * n + i] = a[slot * n + i];
            b[(slot - drop) * n + i] = b[slot * n + i];
            i += 1;
        }
        slot += 1;
    }
}
