// The piecewise polynomial in the basis of Bernstein: the object
// BPoly with the breakpoints, the coefficients of each piece and the
// evaluation with derivatives. The names follow
// scipy.interpolate.BPoly. The coefficient of the order a of the
// piece i stands at the place i times the degree plus one plus a of
// the coefficients.
namespace scientific;

import mathshim;

// BPoly holds the count of the breakpoints, the degree, the two
// rules of the outside and one buffer with the breakpoints followed
// by the coefficients of the pieces.
struct BPoly { n: i64, k: i64, extrap: bool, periodic: bool, buf: f64[>..] }

fn bpoly_empty() -> BPoly {
    return BPoly { n: 0, k: 0, extrap: true, periodic: false, buf: [] };
}

// bpoly holds the pieces of the caller. The breakpoints must rise
// strictly, and each piece needs degree plus one coefficients.
fn bpoly(c: f64[:], x: f64[:], extrap: bool = true, periodic: bool = false) -> BPoly, Fault {
    if x.len < 2 {
        return bpoly_empty(), fault(F_SHAPE, "bpoly", "the breakpoints need two values at least");
    }
    var i: i64 = 0;
    while i < x.len {
        if !(x[i] - x[i] == 0.0) {
            return bpoly_empty(), fault(F_VALUE, "bpoly", "a breakpoint is not finite");
        }
        if i > 0 && x[i] <= x[i - 1] {
            return bpoly_empty(), fault(F_VALUE, "bpoly", "the breakpoints do not rise strictly");
        }
        i += 1;
    }
    if periodic && !(x[x.len - 1] - x[0] == 0.0) && x[x.len - 1] <= x[0] {
        return bpoly_empty(), fault(F_VALUE, "bpoly", "a periodic polynomial needs a span of points");
    }
    // The count of the coefficients fixes the degree.
    let m = x.len - 1;
    if c.len % m != 0 {
        return bpoly_empty(), fault(F_SHAPE, "bpoly", "the coefficients do not split into the pieces");
    }
    let k = c.len / m - 1;
    if k < 0 {
        return bpoly_empty(), fault(F_SHAPE, "bpoly", "each piece needs one coefficient at least");
    }
    i = 0;
    while i < c.len {
        if !(c[i] - c[i] == 0.0) {
            return bpoly_empty(), fault(F_VALUE, "bpoly", "a coefficient is not finite");
        }
        i += 1;
    }
    var buf: f64[>..] = copy_slice(x);
    i = 0;
    while i < c.len { buf.push(c[i]); i += 1; }
    return BPoly { n: x.len, k: k, extrap: extrap, periodic: periodic, buf: buf }, no_fault();
}

// bpoly_breaks_of gives the breakpoints of the polynomial.
fn bpoly_breaks_of(b) -> f64[>..] {
    var t: f64[>..] = [];
    var i: i64 = 0;
    while i < b.n { t.push(b.buf[i]); i += 1; }
    return t;
}

// bpoly_coefs_of gives the coefficients of the pieces of the
// polynomial.
fn bpoly_coefs_of(b) -> f64[>..] {
    var c: f64[>..] = [];
    var i: i64 = b.n;
    while i < b.buf.len { c.push(b.buf[i]); i += 1; }
    return c;
}

// bpoly_binom gives the binomial coefficient of k over a.
fn bpoly_binom(k: i64, a: i64) -> f64 {
    if a < 0 || a > k { return 0.0; }
    var r: i64 = 1;
    var i: i64 = 0;
    while i < a {
        r = r * (k - i) / (i + 1);
        i += 1;
    }
    return r as f64;
}

// bpoly_piece finds the piece that holds the point. The pieces are
// half open, and the last piece is closed, as the pieces of SciPy
// are.
fn bpoly_piece(b, v: f64) -> i64 {
    var lo: i64 = 0;
    var hi: i64 = b.n - 2;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if v >= b.buf[mid] { lo = mid; } else { hi = mid; }
    }
    if v >= b.buf[b.n - 2] { return b.n - 2; }
    return lo;
}

// bpoly_value gives the value of the polynomial of the piece i and
// the degree k at the place t of zero to one.
fn bpoly_value(c: f64[:], base: i64, k: i64, t: f64) -> f64 {
    var v: f64 = 0.0;
    var a: i64 = 0;
    while a <= k {
        var ta: f64 = 1.0;
        var j: i64 = 0;
        while j < a { ta = ta * t; j += 1; }
        var tb: f64 = 1.0;
        j = 0;
        while j < k - a { tb = tb * (1.0 - t); j += 1; }
        v += c[base + a] * bpoly_binom(k, a) * ta * tb;
        a += 1;
    }
    return v;
}

// bpoly_reduce gives the coefficients of the derivative of one order.
// The rule of the derivative of the basis of Bernstein holds: the
// coefficient a becomes the degree times the step of the coefficients
// a and a plus one, over the width of the piece.
fn bpoly_reduce(c: f64[:], base: i64, k: i64, dx: f64, out: f64[>..]&) {
    var a: i64 = 0;
    while a < k {
        out.push((k as f64) * (c[base + a + 1] - c[base + a]) / dx);
        a += 1;
    }
}

// bpoly_eval gives the value of the nu-th derivative of the
// polynomial at the points. A derivative above the degree is zero. A
// point outside the breakpoints takes the first or the last piece,
// unless the rule of the outside forbids it. A periodic polynomial
// repeats the span.
fn bpoly_eval(b, xs: f64[:], nu: i64 = 0) -> Arr, Fault {
    if b.n < 2 {
        return empty_arr(), fault(F_SHAPE, "bpoly_eval", "the polynomial holds no pieces");
    }
    if nu < 0 {
        return empty_arr(), fault(F_RANGE, "bpoly_eval", "the order of the derivative is below zero");
    }
    if xs.len < 1 {
        return empty_arr(), fault(F_SHAPE, "bpoly_eval", "the list of points holds no value");
    }
    var i: i64 = 0;
    while i < xs.len {
        if !(xs[i] - xs[i] == 0.0) {
            return empty_arr(), fault(F_VALUE, "bpoly_eval", "a point is not finite");
        }
        i += 1;
    }
    let lo = b.buf[0];
    let hi = b.buf[b.n - 1];
    var out: f64[>..] = [];
    i = 0;
    while i < xs.len {
        var v = xs[i];
        if b.periodic {
            let per = hi - lo;
            var w = (v - lo) / per;
            let fl = w - (w % 1.0);
            v = v - fl * per;
        }
        if v != v || (!b.extrap && !b.periodic && (v < lo || v > hi)) {
            out.push(0.0 / 0.0);
            i += 1;
            continue;
        }
        let pi = bpoly_piece(b, v);
        let dx = b.buf[pi + 1] - b.buf[pi];
        let t = (v - b.buf[pi]) / dx;
        // The coefficients of the piece, reduced by the derivative.
        var cc: f64[>..] = [];
        var a: i64 = 0;
        while a <= b.k {
            cc.push(b.buf[b.n + pi * (b.k + 1) + a]);
            a += 1;
        }
        var kk = b.k;
        var turn: i64 = 0;
        while turn < nu && kk > 0 {
            var nc: f64[>..] = [];
            bpoly_reduce(cc, 0, kk, dx, nc);
            cc = copy(nc);
            kk -= 1;
            turn += 1;
        }
        if nu > b.k {
            out.push(0.0);
        } else {
            out.push(bpoly_value(cc, 0, kk, t));
        }
        i += 1;
    }
    return line_arr(copy(out)), no_fault();
}

// bpoly_derivative gives the polynomial of the order nu lower. The
// derivative of a polynomial of degree zero is the zero polynomial.
fn bpoly_derivative(b: BPoly, nu: i64 = 1) -> BPoly, Fault {
    if b.n < 2 {
        return bpoly_empty(), fault(F_SHAPE, "bpoly_derivative", "the polynomial holds no pieces");
    }
    if nu < 0 {
        return bpoly_antideriv_core(copy(b), -nu);
    }
    return bpoly_deriv_core(copy(b), nu);
}

// bpoly_deriv_core lowers the degree of the polynomial by the count
// nu. The count holds zero or more.
fn bpoly_deriv_core(b: BPoly, nu: i64) -> BPoly, Fault {
    var m = b.n - 1;
    var k = b.k;
    var co: f64[>..] = [];
    var i: i64 = 0;
    while i < m * (k + 1) { co.push(b.buf[b.n + i]); i += 1; }
    var turn: i64 = 0;
    while turn < nu {
        var nc: f64[>..] = [];
        i = 0;
        while i < m {
            let dx = b.buf[i + 1] - b.buf[i];
            bpoly_reduce(co, i * (k + 1), k, dx, nc);
            i += 1;
        }
        co = copy(nc);
        k -= 1;
        turn += 1;
    }
    if co.len != m * (k + 1) || k < 0 {
        // The derivative of a polynomial of degree zero is the zero
        // polynomial of degree zero.
        k = 0;
        co = [];
        var z: i64 = 0;
        while z < m { co.push(0.0); z += 1; }
    }
    var buf: f64[>..] = [];
    i = 0;
    while i < b.n { buf.push(b.buf[i]); i += 1; }
    i = 0;
    while i < m * (k + 1) { buf.push(co[i]); i += 1; }
    return BPoly { n: b.n, k: k, extrap: b.extrap, periodic: b.periodic, buf: buf }, no_fault();
}

// bpoly_antiderivative gives the polynomial whose derivative is the
// polynomial of the caller. The constant of the first piece is zero.
fn bpoly_antiderivative(b: BPoly, nu: i64 = 1) -> BPoly, Fault {
    if b.n < 2 {
        return bpoly_empty(), fault(F_SHAPE, "bpoly_antiderivative", "the polynomial holds no pieces");
    }
    if nu <= 0 {
        return bpoly_deriv_core(copy(b), -nu);
    }
    return bpoly_antideriv_core(copy(b), nu);
}

// bpoly_antideriv_core raises the degree of the polynomial by the
// count nu. The count holds one or more.
fn bpoly_antideriv_core(b: BPoly, nu: i64) -> BPoly, Fault {
    let m = b.n - 1;
    let k = b.k + 1;
    // The coefficients of the pieces, raised nu times.
    var co: f64[>..] = [];
    var i: i64 = 0;
    while i < m * (b.k + 1) { co.push(b.buf[b.n + i]); i += 1; }
    var turn: i64 = 0;
    while turn < nu {
        // One raise: the degree grows by one, and the first
        // coefficient of every piece is zero.
        let deg = co.len / m;
        var nc: f64[>..] = [];
        i = 0;
        while i < m {
            let dx = b.buf[i + 1] - b.buf[i];
            var run: f64 = 0.0;
            var a: i64 = 0;
            nc.push(0.0);
            while a < deg {
                run += co[i * deg + a] / (deg as f64);
                nc.push(run * dx);
                a += 1;
            }
            i += 1;
        }
        // The pieces after the first hold the jump of the value at
        // their left end, as the rule of SciPy holds. The jumps come
        // from the last row of the pieces before the fix.
        var last: f64[>..] = [];
        i = 0;
        while i < m { last.push(nc[i * (deg + 1) + deg]); i += 1; }
        var run: f64 = 0.0;
        i = 1;
        while i < m {
            run += last[i - 1];
            var a: i64 = 0;
            while a < deg + 1 {
                nc[i * (deg + 1) + a] += run;
                a += 1;
            }
            i += 1;
        }
        co = copy(nc);
        turn += 1;
    }
    var buf: f64[>..] = [];
    i = 0;
    while i < b.n { buf.push(b.buf[i]); i += 1; }
    i = 0;
    while i < co.len { buf.push(co[i]); i += 1; }
    return BPoly { n: b.n, k: k, extrap: b.extrap, periodic: b.periodic, buf: buf }, no_fault();
}

// bpoly_integrate gives the integral of the polynomial between a and
// b. The ends may lie outside the breakpoints, and the polynomial
// takes its end pieces there. A periodic polynomial repeats its span.
// bpoly_integrate_flip holds the integral from bb to a, which is the
// negative of the integral from a to bb. It sits apart, so the local
// values of bpoly_integrate end before the recursive call.
recursive fn bpoly_integrate_flip(b: BPoly&, a: f64, bb: f64) -> f64, Fault {
    let r, f = bpoly_integrate(b, bb, a);
    return 0.0 - r, f;
}

recursive fn bpoly_integrate(b: BPoly&, a: f64, bb: f64) -> f64, Fault {
    if b.n < 2 {
        return 0.0, fault(F_SHAPE, "bpoly_integrate", "the polynomial holds no pieces");
    }
    if b.periodic && bb < a {
        return bpoly_integrate_flip(b, bb, a);
    }
    let anti, f = bpoly_antiderivative(copy(b), 1);
    if f.kind != F_NONE { return 0.0, f; }
    if b.periodic {
        let lo = b.buf[0];
        let hi = b.buf[b.n - 1];
        let per = hi - lo;
        let spans = mathshim::floor_of((bb - a) / per);
        let v_hi, f3 = bpoly_eval(anti, [hi], 0);
        let v_lo, f4 = bpoly_eval(anti, [lo], 0);
        if f3.kind != F_NONE || f4.kind != F_NONE {
            return 0.0, fault(F_VALUE, "bpoly_integrate", "the antiderivative holds no value at the ends");
        }
        var res = spans * (v_hi.data[0] - v_lo.data[0]);
        var ra = lo + (a - lo) - mathshim::floor_of((a - lo) / per) * per;
        let rb = ra + (bb - a - spans * per);
        if rb <= hi {
            let v1, f5 = bpoly_eval(anti, [ra], 0);
            let v2, f6 = bpoly_eval(anti, [rb], 0);
            if f5.kind != F_NONE || f6.kind != F_NONE {
                return 0.0, fault(F_VALUE, "bpoly_integrate", "the antiderivative holds no value at the ends");
            }
            res += v2.data[0] - v1.data[0];
        } else {
            let v1, f5 = bpoly_eval(anti, [ra], 0);
            let v2, f6 = bpoly_eval(anti, [hi], 0);
            let v3, f7 = bpoly_eval(anti, [lo + (ra + (bb - a - spans * per) - hi)], 0);
            if f5.kind != F_NONE || f6.kind != F_NONE || f7.kind != F_NONE {
                return 0.0, fault(F_VALUE, "bpoly_integrate", "the antiderivative holds no value at the ends");
            }
            res += v2.data[0] - v1.data[0] + v3.data[0] - v_lo.data[0];
        }
        return res, no_fault();
    }
    let va, f8 = bpoly_eval(anti, [a], 0);
    let vb, f9 = bpoly_eval(anti, [bb], 0);
    if f8.kind != F_NONE || f9.kind != F_NONE {
        return 0.0, fault(F_VALUE, "bpoly_integrate", "the antiderivative holds no value at the ends");
    }
    return vb.data[0] - va.data[0], no_fault();
}
