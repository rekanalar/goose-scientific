// The spline filter of one dimension: the orders, the modes and the coefficients.
//
// The interpolating transforms of arrays of N dimensions, as in
// scipy.ndimage: spline_filter1d, spline_filter, map_coordinates,
// affine_transform, shift, zoom and rotate.
//
// A spline of order 2 to 5 reads coefficients that a recursive filter
// computes first. The orders 0 and 1 read the values. The filter and the
// reading follow the mode. The modes are "mirror", "reflect" (also
// "grid-mirror"), "wrap", "grid-wrap", "nearest", "constant" and
// "grid-constant". The arrays hold real values.
namespace scientific;

import mathshim;

// The cost of a call is the count of taps: the count of output values
// times the taps of one value. A coordinate may reach 4.0e15 at most.
const TR_COST: i64 = 1000000000;
const TR_COORD: f64 = 4.0e15;
const TR_PAD: i64 = 12;

// tr_mode gives the number of a mode name, or -1. The numbers are 0 for
// mirror, 1 for reflect, 2 for wrap, 3 for grid-wrap, 4 for nearest, 5 for
// constant and 6 for grid-constant.
fn tr_mode(name: const u8[:]) -> i64 {
    if name == "mirror" { return 0; }
    if name == "reflect" || name == "grid-mirror" { return 1; }
    if name == "wrap" { return 2; }
    if name == "grid-wrap" { return 3; }
    if name == "nearest" { return 4; }
    if name == "constant" { return 5; }
    if name == "grid-constant" { return 6; }
    return -1;
}

// tr_filter_mode gives the boundary rule of the spline filter for a mode:
// 0 mirrors, 1 reflects and 2 wraps with the period of the length.
fn tr_filter_mode(mode: i64) -> i64 {
    if mode == 1 || mode == 4 { return 1; }
    if mode == 3 { return 2; }
    return 0;
}

// tr_poles gives the poles of the spline filter of an order. It gives the count.
fn tr_poles(order: i64, z: f64[..4]&) -> i64 {
    if order == 2 { z.push(mathshim::root(8.0) - 3.0); return 1; }
    if order == 3 { z.push(mathshim::root(3.0) - 2.0); return 1; }
    if order == 4 {
        let r = mathshim::root(438976.0);
        let s = mathshim::root(304.0);
        z.push(mathshim::root(664.0 - r) + s - 19.0);
        z.push(mathshim::root(664.0 + r) - s - 19.0);
        return 2;
    }
    if order == 5 {
        let r = mathshim::root(4436.25);
        let s = mathshim::root(26.25);
        z.push(mathshim::root(67.5 - r) + s - 6.5);
        z.push(mathshim::root(67.5 + r) - s - 6.5);
        return 2;
    }
    return 0;
}

// tr_src gives the index inside 0 to n - 1 that stands for the index i
// under a boundary rule of the filter.
fn tr_src(i: i64, n: i64, fm: i64) -> i64 {
    if i >= 0 && i < n { return i; }
    if fm == 2 {
        var w = i % n;
        if w < 0 { w += n; }
        return w;
    }
    if fm == 0 {
        if n == 1 { return 0; }
        let p = 2 * n - 2;
        var m = i % p;
        if m < 0 { m += p; }
        if m >= n { m = p - m; }
        return m;
    }
    let p = 2 * n;
    var m = i % p;
    if m < 0 { m += p; }
    if m >= n { m = p - 1 - m; }
    return m;
}

// tr_filter_line runs the spline filter of an order over the first n values
// of a line. The rule fm is the boundary rule.
fn tr_filter_line(c: f64[>..]&, n: i64, order: i64, fm: i64) {
    if n < 2 || order < 2 { return; }
    var z: f64[..4] = [];
    let np = tr_poles(order, z);
    var gain: f64 = 1.0;
    var k: i64 = 0;
    while k < np { gain *= (1.0 - z[k]) * (1.0 - 1.0 / z[k]); k += 1; }
    var i: i64 = 0;
    while i < n { c[i] *= gain; i += 1; }
    k = 0;
    while k < np {
        let zk = z[k];
        var az = zk;
        if az < 0.0 { az = -az; }
        let horizon = (mathshim::ceil_of(mathshim::ln(1.0e-15) / mathshim::ln(az))) as i64;
        var s: f64 = 0.0;
        var zi: f64 = 1.0;
        var j: i64 = 0;
        while j <= horizon {
            s += zi * c[tr_src(-j, n, fm)];
            zi *= zk;
            j += 1;
        }
        c[0] = s;
        i = 1;
        while i < n { c[i] += zk * c[i - 1]; i += 1; }
        if fm == 0 {
            c[n - 1] = (zk / (zk * zk - 1.0)) * (c[n - 1] + zk * c[n - 2]);
        } else if fm == 1 {
            c[n - 1] = c[n - 1] * zk / (zk - 1.0);
        } else {
            var t: f64 = 0.0;
            zi = zk;
            j = 0;
            while j <= horizon {
                t += zi * c[tr_src(n - 1 + j, n, fm)];
                zi *= zk;
                j += 1;
            }
            c[n - 1] = -t;
        }
        i = n - 2;
        while i >= 0 { c[i] = zk * (c[i + 1] - c[i]); i -= 1; }
        k += 1;
    }
}

// tr_filter_axis runs the filter along one axis of an array.
fn tr_filter_axis(data: f64[>..]&, shape: i64[:], axis: i64, order: i64, fm: i64) {
    let n = shape[axis];
    var inner: i64 = 1;
    var k: i64 = axis + 1;
    while k < shape.len { inner *= shape[k]; k += 1; }
    let outer = data.len / (n * inner);
    var line: f64[>..] = nd_zeros(n);
    var o: i64 = 0;
    while o < outer {
        var r: i64 = 0;
        while r < inner {
            let base = o * n * inner + r;
            var q: i64 = 0;
            while q < n { line[q] = data[base + q * inner]; q += 1; }
            tr_filter_line(line, n, order, fm);
            q = 0;
            while q < n { data[base + q * inner] = line[q]; q += 1; }
            r += 1;
        }
        o += 1;
    }
}

// tr_checked checks an array, an order and a mode. It gives the mode number.
fn tr_checked(a, order: i64, mode: const u8[:], tag: const u8[:]) -> i64, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if f.kind != F_NONE { return -1, f; }
    if order < 0 || order > 5 { return -1, fault(F_RANGE, tag, "the order of the spline stands outside 0 to 5"); }
    let m = tr_mode(mode);
    if m < 0 { return -1, fault(F_VALUE, tag, "the mode holds an unknown name"); }
    return m, no_fault();
}

// spline_filter1d gives the coefficients of the spline of an order along one
// axis. The orders 0 and 1 give the array itself.
fn spline_filter1d(a, order: i64 = 3, axis: i64 = -1, mode: const u8[:] = "mirror") -> Arr, Fault {
    let m, f = tr_checked(a, order, mode, "spline_filter1d");
    if f.kind != F_NONE { return empty_arr(), f; }
    let ax = nd_axis(axis, a.ndim);
    if ax < 0 { return empty_arr(), fault(F_RANGE, "spline_filter1d", "the axis stands outside the array"); }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    tr_filter_axis(d, a.shape, ax, order, tr_filter_mode(m));
    return arr_of(copy(d), a.shape), no_fault();
}

// spline_filter gives the coefficients of the spline of an order from 2 to 5,
// one axis after the other.
fn spline_filter(a, order: i64 = 3, mode: const u8[:] = "mirror") -> Arr, Fault {
    let m, f = tr_checked(a, order, mode, "spline_filter");
    if f.kind != F_NONE { return empty_arr(), f; }
    if order < 2 { return empty_arr(), fault(F_RANGE, "spline_filter", "the order of the spline stands outside 2 to 5"); }
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < a.data.len { d.push(a.data[i]); i += 1; }
    var ax: i64 = 0;
    while ax < a.ndim {
        tr_filter_axis(d, a.shape, ax, order, tr_filter_mode(m));
        ax += 1;
    }
    return arr_of(copy(d), a.shape), no_fault();
}

// tr_pad copies an array into a larger one with TR_PAD values on each side.
// The new values repeat the edge, or they hold cval. The check of the size
// comes before the allocation.
fn tr_pad(a, edge: bool, cval: f64, pshape: i64[..8]&, tag: const u8[:]) -> f64[>..], Fault {
    var total: i64 = 1;
    var d: i64 = 0;
    while d < a.ndim {
        let sz = a.shape[d] + 2 * TR_PAD;
        if total > MAX_ELEMENTS / sz {
            return [], fault(F_LIMIT, tag, "the padded array holds more elements than the limit");
        }
        total *= sz;
        pshape.push(sz);
        d += 1;
    }
    var out: f64[>..] = [];
    var ix: i64[..8] = [];
    d = 0;
    while d < a.ndim { ix.push(0); d += 1; }
    var p: i64 = 0;
    while p < total {
        var flat: i64 = 0;
        var outside = false;
        d = 0;
        while d < a.ndim {
            var s = ix[d] - TR_PAD;
            if s < 0 || s >= a.shape[d] {
                outside = true;
                if s < 0 { s = 0; } else { s = a.shape[d] - 1; }
            }
            flat = flat * a.shape[d] + s;
            d += 1;
        }
        if outside && !edge { out.push(cval); } else { out.push(a.data[flat]); }
        d = a.ndim - 1;
        while d >= 0 {
            ix[d] += 1;
            if ix[d] < pshape[d] { d = -1; } else { ix[d] = 0; d -= 1; }
        }
        p += 1;
    }
    return out, no_fault();
}

// tr_bspline gives the cardinal B-spline of an order at t.
fn tr_bspline(n: i64, t: f64) -> f64 {
    var s: f64 = 0.0;
    var binom: f64 = 1.0;
    var k: i64 = 0;
    while k <= n + 1 {
        let v = t + ((n + 1) as f64) / 2.0 - (k as f64);
        if v > 0.0 {
            var pw: f64 = 1.0;
            var e: i64 = 0;
            while e < n { pw *= v; e += 1; }
            if k % 2 == 0 { s += binom * pw; } else { s -= binom * pw; }
        }
        binom = binom * ((n + 1 - k) as f64) / ((k + 1) as f64);
        k += 1;
    }
    var fact: f64 = 1.0;
    k = 2;
    while k <= n { fact *= k as f64; k += 1; }
    return s / fact;
}

// tr_trunc gives the integer part of x as a float.
fn tr_trunc(x: f64) -> f64 {
    let i = x as i64;
    return i as f64;
}

// tr_fold_coord maps a coordinate into the array by the rule of the mode.
fn tr_fold_coord(c0: f64, n: i64, mode: i64) -> f64 {
    var c = c0;
    let nf = n as f64;
    if mode == 0 {
        if n <= 1 { return 0.0; }
        let p = 2.0 * nf - 2.0;
        if c < 0.0 {
            c = p * tr_trunc(-c / p) + c;
            if c <= 1.0 - nf { c += p; } else { c = -c; }
        } else if c >= nf {
            c -= p * tr_trunc(c / p);
            if c >= nf { c = p - c; }
        }
        return c;
    }
    if mode == 1 {
        if n <= 1 { return 0.0; }
        let p = 2.0 * nf;
        if c < -0.5 {
            c = p * tr_trunc(-c / p) + c;
            if c < -nf { c += p; } else { c = -c - 1.0; }
        } else if c > nf - 0.5 {
            c -= p * tr_trunc(c / p);
            if c >= nf { c = p - c - 1.0; }
        }
        return c;
    }
    if mode == 2 {
        if n <= 1 { return 0.0; }
        let p = nf - 1.0;
        if c < 0.0 {
            c += p * (tr_trunc(-c / p) + 1.0);
        } else if c > p {
            c -= p * tr_trunc(c / p);
        }
        return c;
    }
    if mode == 3 {
        if n <= 1 { return 0.0; }
        if c < 0.0 {
            c += nf * (tr_trunc(-c / nf) + 1.0);
            if c == nf { c = 0.0; }
        } else if c >= nf {
            c -= nf * tr_trunc(c / nf);
        }
        return c;
    }
    if mode == 4 {
        if c < 0.0 { return 0.0; }
        if c > nf - 1.0 { return nf - 1.0; }
        return c;
    }
    return c;
}

// tr_fold_idx maps the index of a tap into the array by the rule of the
// mode. The wrap mode folds like the mirror mode for a spline of order 2 or
// more.
fn tr_fold_idx(i0: i64, n: i64, mode0: i64, order: i64) -> i64 {
    var mode = mode0;
    if mode == 2 && order >= 2 { mode = 0; }
    if mode == 5 { mode = 0; }
    var i = i0;
    if mode == 0 { return tr_src(i, n, 0); }
    if mode == 1 { return tr_src(i, n, 1); }
    if mode == 2 {
        if n == 1 { return 0; }
        let sz = n - 1;
        if i < 0 { i += sz * ((-i) / sz + 1); } else if i > sz { i -= sz * (i / sz); }
        return i;
    }
    if mode == 3 { return tr_src(i, n, 2); }
    if mode == 4 {
        if i < 0 { return 0; }
        if i > n - 1 { return n - 1; }
    }
    return i;
}

// tr_run evaluates the spline at one coordinate for each place of the output.
// The kinds of coordinates are 0 for a list in cbuf, 1 for a zoom and a shift
// of each axis, and 2 for a matrix and an offset. The buffer out holds
// zeros, one for each output place.
fn tr_run(flt: f64[:], fshape: i64[:], npad: i64, order: i64, mode: i64, cval: f64, kind: i64, cbuf: f64[:], zs: f64[:], sh: f64[:], grid: bool, mat: f64[:], off: f64[:], oshape: i64[:], out: f64[>..]&, tag: const u8[:]) -> Fault {
    let nd = fshape.len;
    let nout = oshape.len;
    let npts = out.len;
    var ntap = order + 1;
    if order == 0 { ntap = 1; }
    var tidx: i64[>..] = [];
    var tw: f64[>..] = nd_zeros(nd * ntap);
    var i: i64 = 0;
    while i < nd * ntap { tidx.push(0); i += 1; }
    var ix: i64[..8] = [];
    var d: i64 = 0;
    while d < nout { ix.push(0); d += 1; }
    var jv: i64[..8] = [];
    d = 0;
    while d < nd { jv.push(0); d += 1; }
    var p: i64 = 0;
    while p < npts {
        var outside = false;
        d = 0;
        while d < nd {
            var c: f64 = 0.0;
            if kind == 0 {
                c = cbuf[d * npts + p];
            } else if kind == 1 {
                c = ix[d] as f64;
                if sh.len > 0 { c += sh[d]; }
                if zs.len > 0 {
                    if grid { c = (c + 0.5) * zs[d] - 0.5; } else { c *= zs[d]; }
                }
            } else {
                c = off[d];
                var j: i64 = 0;
                while j < nout { c += mat[d * nout + j] * (ix[j] as f64); j += 1; }
            }
            c += npad as f64;
            if !(c >= -TR_COORD && c <= TR_COORD) {
                return fault(F_VALUE, tag, "a coordinate is not finite or reaches beyond 4.0e15");
            }
            let n = fshape[d];
            if mode == 5 {
                if c < 0.0 || c > (n as f64) - 1.0 { outside = true; }
            } else if mode != 6 {
                c = tr_fold_coord(c, n, mode);
            }
            if !outside {
                var start: i64 = 0;
                if order == 0 {
                    start = mathshim::floor_of(c + 0.5) as i64;
                } else if order % 2 == 1 {
                    start = (mathshim::floor_of(c) as i64) - order / 2;
                } else {
                    start = (mathshim::floor_of(c + 0.5) as i64) - order / 2;
                }
                var j: i64 = 0;
                while j < ntap {
                    var w: f64 = 1.0;
                    if order > 0 { w = tr_bspline(order, c - ((start + j) as f64)); }
                    tw[d * ntap + j] = w;
                    if mode == 6 { tidx[d * ntap + j] = start + j; } else { tidx[d * ntap + j] = tr_fold_idx(start + j, n, mode, order); }
                    j += 1;
                }
            }
            d += 1;
        }
        if outside {
            out[p] = cval;
        } else {
            var s: f64 = 0.0;
            d = 0;
            while d < nd { jv[d] = 0; d += 1; }
            var more = true;
            while more {
                var w: f64 = 1.0;
                var flat: i64 = 0;
                var ok = true;
                d = 0;
                while d < nd {
                    let ti = tidx[d * ntap + jv[d]];
                    w *= tw[d * ntap + jv[d]];
                    if ti < 0 || ti >= fshape[d] { ok = false; } else { flat = flat * fshape[d] + ti; }
                    d += 1;
                }
                if ok { s += w * flt[flat]; } else { s += w * cval; }
                d = nd - 1;
                while d >= 0 {
                    jv[d] += 1;
                    if jv[d] < ntap { d = -2; } else { jv[d] = 0; d -= 1; }
                }
                if d == -1 { more = false; }
            }
            out[p] = s;
        }
        d = nout - 1;
        while d >= 0 {
            ix[d] += 1;
            if ix[d] < oshape[d] { d = -1; } else { ix[d] = 0; d -= 1; }
        }
        p += 1;
    }
    return no_fault();
}

// tr_eval prepares the coefficients and runs tr_run. The shape of the output
// holds at least one place on each axis.
fn tr_eval(a, order: i64, mode: i64, cval: f64, prefilter: bool, kind: i64, cbuf: f64[:], zs: f64[:], sh: f64[:], grid: bool, mat: f64[:], off: f64[:], oshape: i64[:], tag: const u8[:]) -> Arr, Fault {
    let nd = a.ndim;
    var taps: i64 = 1;
    var d: i64 = 0;
    var nt = order + 1;
    if order == 0 { nt = 1; }
    while d < nd {
        if taps > TR_COST / nt { return empty_arr(), fault(F_LIMIT, tag, "the cost of the transform is above the limit"); }
        taps *= nt;
        d += 1;
    }
    let fo = check_shape(oshape);
    if fo.kind != F_NONE { return empty_arr(), fault(fo.kind, tag, fo.detail); }
    let npts = numel(oshape);
    if npts > TR_COST / taps { return empty_arr(), fault(F_LIMIT, tag, "the cost of the transform is above the limit"); }
    var fshape: i64[..8] = [];
    var flt: f64[>..] = [];
    var npad: i64 = 0;
    if prefilter && order > 1 {
        if mode == 4 || mode == 6 {
            npad = TR_PAD;
            let pd, pf = tr_pad(a, mode == 4, cval, fshape, tag);
            if pf.kind != F_NONE { return empty_arr(), pf; }
            flt = copy(pd);
        } else {
            d = 0;
            while d < nd { fshape.push(a.shape[d]); d += 1; }
            var i: i64 = 0;
            while i < a.data.len { flt.push(a.data[i]); i += 1; }
        }
        let fm = tr_filter_mode(mode);
        var ax: i64 = 0;
        while ax < nd { tr_filter_axis(flt, fshape, ax, order, fm); ax += 1; }
    } else {
        d = 0;
        while d < nd { fshape.push(a.shape[d]); d += 1; }
        var i: i64 = 0;
        while i < a.data.len { flt.push(a.data[i]); i += 1; }
    }
    var out: f64[>..] = nd_zeros(npts);
    let g = tr_run(flt, fshape, npad, order, mode, cval, kind, cbuf, zs, sh, grid, mat, off, oshape, out, tag);
    if g.kind != F_NONE { return empty_arr(), g; }
    return arr_of(copy(out), oshape), no_fault();
}

