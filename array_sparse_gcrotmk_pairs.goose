// The kept pairs of gcrotmk.
//
// gcrotmk keeps the pairs (c, u). The flag of a pair stands for the first
// vector (c). The last pair of a solve holds the solution, and its flag is
// zero.
namespace scientific;

import mathshim;

fn gk_pack(cc: f64[:], uu: f64[:], nc: i64, n: i64, x: f64[:], discard: bool, want: bool) -> Kept {
    if !want { return kept_empty(); }
    var buf: f64[>..] = nd_zeros((nc + 1) * kept_stride(n));
    let zr: f64[>..] = nd_zeros(n);
    var flag: f64 = 1.0;
    if discard { flag = 0.0; }
    var p: i64 = 0;
    while p < nc {
        if discard { kept_put(buf, p, n, flag, zr, 0, uu, p * n); }
        else { kept_put(buf, p, n, flag, cc, p * n, uu, p * n); }
        p += 1;
    }
    kept_put(buf, nc, n, 0.0, zr, 0, x, 0);
    return Kept { count: nc + 1, n: n, buf: buf };
}

// gk_seed takes the pairs of the caller as the first kept pairs of gcrotmk.
// A pair without c gets c = A u. The pairs are made orthonormal in c, and u
// follows, so that c stays A u. A pair that adds a share of 1e-12 or less of
// the first one is dropped. Then the solution and the residual project onto
// the pairs. The call gives the count of pairs that stay.

fn gk_seed(s, cu, n: i64, cc: f64[>..]&, uu: f64[>..]&, tc: f64[>..]&, tu: f64[>..]&,
           x: f64[>..]&, r: f64[>..]&) -> i64, Fault {
    var nnew: i64 = 0;
    var first: f64 = 0.0;
    var j: i64 = 0;
    while j < cu.count {
        let fa = kept_first_off(j, n);
        let fb = kept_second_off(j, n);
        var i: i64 = 0;
        while i < n { tu[i] = cu.buf[fb + i]; i += 1; }
        if cu.buf[kept_base(j, n)] == 1.0 {
            i = 0;
            while i < n { tc[i] = cu.buf[fa + i]; i += 1; }
        } else {
            expm_block_product(s, tu, 0, n, tc, 0);
        }
        var p: i64 = 0;
        while p < nnew {
            let a = lg_dot(cc, p * n, tc, 0, n);
            i = 0;
            while i < n {
                tc[i] = tc[i] - a * cc[p * n + i];
                tu[i] = tu[i] - a * uu[p * n + i];
                i += 1;
            }
            p += 1;
        }
        let an = lg_norm(tc, 0, n);
        if an - an != 0.0 { return 0, fault(F_VALUE, "gcrotmk", "a kept pair gives a value that is not finite"); }
        if nnew == 0 { first = an; }
        if an > 0.0 && an > 1.0e-12 * first && 1.0 / an - 1.0 / an == 0.0 {
            i = 0;
            while i < n {
                cc[nnew * n + i] = tc[i] / an;
                uu[nnew * n + i] = tu[i] / an;
                i += 1;
            }
            nnew += 1;
        }
        j += 1;
    }
    var p: i64 = 0;
    while p < nnew {
        let yc = lg_dot(cc, p * n, r, 0, n);
        var i: i64 = 0;
        while i < n {
            x[i] = x[i] + yc * uu[p * n + i];
            r[i] = r[i] - yc * cc[p * n + i];
            i += 1;
        }
        p += 1;
    }
    return nnew, no_fault();
}
