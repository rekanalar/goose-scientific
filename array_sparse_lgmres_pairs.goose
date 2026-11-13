// The kept pairs of lgmres.
//
// lgmres keeps the pairs (dx, ax) of its outer iterations. The flag of a
// pair stands for the second vector (ax).
namespace scientific;

// lg_pack gives the kept pairs (dx, ax) of lgmres. With use_av false the
// second vector is absent, and its row holds zeros.
fn lg_pack(ov: f64[:], oav: f64[:], count: i64, n: i64, use_av: bool, want: bool) -> Kept {
    if !want { return kept_empty(); }
    var buf: f64[>..] = nd_zeros(count * kept_stride(n));
    var flag: f64 = 0.0;
    if use_av { flag = 1.0; }
    var p: i64 = 0;
    while p < count {
        kept_put(buf, p, n, flag, ov, p * n, oav, p * n);
        p += 1;
    }
    return Kept { count: count, n: n, buf: buf };
}

// gk_pack gives the kept pairs (c, u) of gcrotmk and, last, the pair (absent,
// x) that holds the solution. With discard true the c vectors are absent.
