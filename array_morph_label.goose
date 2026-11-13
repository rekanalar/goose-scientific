// The labeling of regions and the Euclidean distance transform.
//
namespace scientific;

import mathshim;

// mo_label finds the connected parts of the nonzero values. A part gets a
// label from 1 in the order of its first value in row-major order.
fn mo_label(a, st, tag: const u8[:]) -> Arr, i64, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), 0, fa; }
    let fs = check_stack(st.ndim, st.shape, st.data.len, 1, tag);
    if fs.kind != F_NONE { return empty_arr(), 0, fs; }
    if st.ndim != a.ndim {
        return empty_arr(), 0, fault(F_SHAPE, tag, "the structure holds another count of dimensions than the array");
    }
    var d: i64 = 0;
    while d < a.ndim {
        if st.shape[d] != 3 {
            return empty_arr(), 0, fault(F_VALUE, tag, "the structure holds a size other than 3 on an axis");
        }
        d += 1;
    }
    var q: i64 = 0;
    while q < st.data.len {
        let mirror = st.data[st.data.len - 1 - q];
        if (st.data[q] != 0.0) != (mirror != 0.0) {
            return empty_arr(), 0, fault(F_VALUE, tag, "the structure is not symmetric");
        }
        q += 1;
    }
    var offs: i64[>..] = [];
    let t = mo_places(st, false, offs);
    let n = a.data.len;
    var lab: f64[>..] = nd_zeros(n);
    var stack: i64[>..] = [];
    var cx: i64[..8] = [];
    d = 0;
    while d < a.ndim { cx.push(0); d += 1; }
    var count: i64 = 0;
    var p: i64 = 0;
    while p < n {
        if a.data[p] != 0.0 && lab[p] == 0.0 {
            count += 1;
            lab[p] = count as f64;
            stack.push(p);
            while stack.len > 0 {
                let cur = stack[stack.len - 1];
                stack.pop();
                var rem = cur;
                d = a.ndim - 1;
                while d >= 0 { cx[d] = rem % a.shape[d]; rem /= a.shape[d]; d -= 1; }
                var k: i64 = 0;
                while k < t {
                    var flat: i64 = 0;
                    var inside = true;
                    d = 0;
                    while d < a.ndim && inside {
                        let y = cx[d] + offs[k * a.ndim + d] - 1;
                        if y < 0 || y >= a.shape[d] { inside = false; } else { flat = flat * a.shape[d] + y; }
                        d += 1;
                    }
                    if inside && a.data[flat] != 0.0 && lab[flat] == 0.0 {
                        lab[flat] = count as f64;
                        stack.push(flat);
                    }
                    k += 1;
                }
            }
        }
        p += 1;
    }
    return arr_of(copy(lab), a.shape), count, no_fault();
}

// label numbers the connected parts of the foreground. The connectivity
// picks the neighbors, as generate_binary_structure does.
fn label(a, connectivity: i64 = 1) -> Arr, i64, Fault {
    let st, f = mo_default(a, connectivity);
    if f.kind != F_NONE { return empty_arr(), 0, f; }
    return mo_label(a, st, "label");
}

// label_st numbers the parts with a structure of size 3 on every axis. The
// structure must be symmetric.
fn label_st(a, structure) -> Arr, i64, Fault {
    return mo_label(a, structure, "label_st");
}

// mo_edt gives the Euclidean distance of each foreground value to the
// nearest background value. A pass along each axis takes the lower
// envelope of the parabolas of the earlier passes.
fn mo_edt(a, sampling: f64[:], want: bool, tag: const u8[:]) -> Arr, Arr, Fault {
    let fa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if fa.kind != F_NONE { return empty_arr(), empty_arr(), fa; }
    let nd = a.ndim;
    if sampling.len != 0 && sampling.len != 1 && sampling.len != nd {
        return empty_arr(), empty_arr(), fault(F_SHAPE, tag, "the sampling holds a count that is not one for each axis");
    }
    var sp: f64[>..] = [];
    var d: i64 = 0;
    while d < nd {
        var s: f64 = 1.0;
        if sampling.len == 1 { s = sampling[0]; }
        if sampling.len > 1 { s = sampling[d]; }
        if !(s - s == 0.0) || !(s > 0.0) || s * (a.shape[d] as f64) > 1.0e100 {
            return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the sampling holds a value that is not above zero or is too large");
        }
        sp.push(s);
        d += 1;
    }
    let n = a.data.len;
    let inf = 1.0 / 0.0;
    var f: f64[>..] = nd_zeros(n);
    var background: i64 = 0;
    var p: i64 = 0;
    while p < n {
        if a.data[p] == 0.0 { background += 1; } else { f[p] = inf; }
        p += 1;
    }
    if background == 0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, tag, "the array holds no background value");
    }
    var feat: i64[>..] = [];
    if want {
        var cx: i64[..8] = [];
        d = 0;
        while d < nd { cx.push(0); d += 1; }
        p = 0;
        while p < n {
            d = 0;
            while d < nd { feat.push(cx[d]); d += 1; }
            d = nd - 1;
            while d >= 0 {
                cx[d] += 1;
                if cx[d] < a.shape[d] { d = -1; } else { cx[d] = 0; d -= 1; }
            }
            p += 1;
        }
    }
    d = 0;
    while d < nd {
        mo_edt_axis(f, feat, a.shape, d, sp[d], want, nd);
        d += 1;
    }
    var dist: f64[>..] = nd_zeros(n);
    p = 0;
    while p < n {
        if f[p] != 0.0 { dist[p] = mathshim::root(f[p]); }
        p += 1;
    }
    var idx: f64[>..] = [];
    var ishape: i64[..8] = [];
    if want {
        ishape.push(nd);
        d = 0;
        while d < nd { ishape.push(a.shape[d]); d += 1; }
        d = 0;
        while d < nd {
            p = 0;
            while p < n { idx.push(feat[p * nd + d] as f64); p += 1; }
            d += 1;
        }
    }
    var ires = empty_arr();
    if want { ires = arr_of(copy(idx), ishape); }
    return arr_of(copy(dist), a.shape), ires, no_fault();
}

// mo_edt_axis takes the lower envelope of parabolas along one axis. The
// values of f hold squared distances, and inf stands for none yet.
fn mo_edt_axis(f: f64[>..]&, feat: i64[>..]&, shape: i64[:], axis: i64, s: f64, want: bool, nd: i64) {
    let n = shape[axis];
    var inner: i64 = 1;
    var k: i64 = axis + 1;
    while k < shape.len { inner *= shape[k]; k += 1; }
    let outer = f.len / (n * inner);
    let inf = 1.0 / 0.0;
    var line: f64[>..] = nd_zeros(n);
    var res: f64[>..] = nd_zeros(n);
    var v: i64[>..] = [];
    var z: f64[>..] = [];
    var from: i64[>..] = [];
    var q: i64 = 0;
    while q < n { v.push(0); from.push(0); q += 1; }
    q = 0;
    while q <= n { z.push(0.0); q += 1; }
    var tmp: i64[>..] = [];
    if want {
        q = 0;
        while q < n * nd { tmp.push(0); q += 1; }
    }
    var o: i64 = 0;
    while o < outer {
        var r: i64 = 0;
        while r < inner {
            let base = o * n * inner + r;
            q = 0;
            while q < n { line[q] = f[base + q * inner]; q += 1; }
            var top: i64 = -1;
            q = 0;
            while q < n {
                if line[q] != inf {
                    let xq = s * (q as f64);
                    if top < 0 {
                        top = 0;
                        v[0] = q;
                        z[0] = -inf;
                        z[1] = inf;
                    } else {
                        var sx: f64 = 0.0;
                        var again = true;
                        while again {
                            let xv = s * (v[top] as f64);
                            sx = ((line[q] + xq * xq) - (line[v[top]] + xv * xv)) / (2.0 * (xq - xv));
                            if sx <= z[top] && top > 0 { top -= 1; } else { again = false; }
                        }
                        if sx <= z[top] {
                            v[top] = q;
                        } else {
                            top += 1;
                            v[top] = q;
                            z[top] = sx;
                            z[top + 1] = inf;
                        }
                    }
                }
                q += 1;
            }
            if top >= 0 {
                var j: i64 = 0;
                q = 0;
                while q < n {
                    let xq = s * (q as f64);
                    while z[j + 1] < xq { j += 1; }
                    let xv = s * (v[j] as f64);
                    res[q] = (xq - xv) * (xq - xv) + line[v[j]];
                    from[q] = v[j];
                    q += 1;
                }
                q = 0;
                while q < n { f[base + q * inner] = res[q]; q += 1; }
                if want {
                    q = 0;
                    while q < n {
                        var c: i64 = 0;
                        while c < nd { tmp[q * nd + c] = feat[(base + from[q] * inner) * nd + c]; c += 1; }
                        q += 1;
                    }
                    q = 0;
                    while q < n {
                        var c: i64 = 0;
                        while c < nd { feat[(base + q * inner) * nd + c] = tmp[q * nd + c]; c += 1; }
                        q += 1;
                    }
                }
            }
            r += 1;
        }
        o += 1;
    }
}

// distance_transform_edt gives the Euclidean distance of each foreground
// value to the nearest background value. The sampling holds none, one, or
// one spacing for each axis.
fn distance_transform_edt(a, sampling: f64[:] = []) -> Arr, Fault {
    let d, i, f = mo_edt(a, sampling, false, "distance_transform_edt");
    return d, f;
}

// distance_transform_edt_indices gives the distances and the index of the
// nearest background value. The array of indices holds one row of positions for
// each axis, so its shape is the count of axes followed by the shape.
fn distance_transform_edt_indices(a, sampling: f64[:] = []) -> Arr, Arr, Fault {
    return mo_edt(a, sampling, true, "distance_transform_edt_indices");
}
