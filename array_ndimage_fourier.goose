// The Fourier filters of an image and the watershed of the distance.
//
// The Fourier filters.
//
// fourier_gaussian, fourier_uniform, fourier_ellipsoid and
// fourier_shift multiply the transform of an array with the response of
// the filter. The input is a complex array, so a caller converts a real
// array with complex_real first. The transform runs along every
// dimension, and the frequency of every dimension follows the table of
// the transform. The formulas of the responses follow the formulas of
// the package of SciPy.
//
// The watershed of the distance.
//
// watershed_ift labels the regions of an image by the watershed of the
// distance transform. The cost of a step is the difference of the
// values of the two nodes. Each region grows from its marker, and the
// label of a node follows the cheapest path from a marker.
namespace scientific;

import mathshim;

// The kinds of the filters. The count of the kinds stands in the limit
// of the count of the dimensions of the ellipsoid: the ellipsoid holds
// three dimensions at most.
const FT_GAUSSIAN: i64 = 0;
const FT_UNIFORM: i64 = 1;
const FT_ELLIPSOID: i64 = 2;
const FT_SHIFT: i64 = 3;

// bessel_j1 gives the function of Bessel of the first kind and the
// order one. Below 25 the method is the backward recurrence of Miller,
// scaled by the sum J0 + 2 J2 + 2 J4 + ... = 1. From 25 on the method
// is the asymptotic series of Hankel. Both hold about 14 digits.
recursive fn bessel_j1(x: f64) -> f64 {
    if x < 0.0 { return -bessel_j1(-x); }
    if x == 0.0 { return 0.0; }
    if x < 25.0 {
        var top = (x as i64) + 40;
        if top % 2 == 1 { top += 1; }
        var bjp: f64 = 0.0;
        var bj: f64 = 1.0;
        var even: f64 = 0.0;
        var one: f64 = 0.0;
        var n = top;
        while n >= 1 {
            let bjm = 2.0 * (n as f64) / x * bj - bjp;
            bjp = bj;
            bj = bjm;
            if mathshim::abs_of(bj) > 1.0e100 {
                bj *= 1.0e-100;
                bjp *= 1.0e-100;
                even *= 1.0e-100;
                one *= 1.0e-100;
            }
            if (n - 1) % 2 == 0 {
                if n == 1 { even += bj; } else { even += 2.0 * bj; }
            }
            if n == 1 { one = bjp; }
            n -= 1;
        }
        return one / even;
    }
    // The series of Hankel for the order one: mu = 4 nu squared = 4.
    let eight = 8.0 * x;
    var t: f64 = 1.0;
    var p: f64 = 1.0;
    var q: f64 = 0.0;
    var k: i64 = 1;
    while k < 40 {
        let odd = (2 * k - 1) as f64;
        let nt = t * (4.0 - odd * odd) / ((k as f64) * eight);
        if mathshim::abs_of(nt) > mathshim::abs_of(t) && k > 2 { break; }
        t = nt;
        if k % 2 == 1 {
            if (k / 2) % 2 == 0 { q += t; } else { q -= t; }
        } else {
            if (k / 2) % 2 == 0 { p += t; } else { p -= t; }
        }
        if mathshim::abs_of(t) < 1.0e-17 { break; }
        k += 1;
    }
    let chi = x - 2.356194490192345;
    return mathshim::root(0.6366197723675814 / x) * (p * mathshim::cosine(chi) - q * mathshim::sine(chi));
}

// ft_ball gives the response of the filter of the ellipsoid at the
// radius r. The response follows the count of the dimensions: the ratio
// of the sine in one dimension, the function of Bessel in two
// dimensions and the combination of the sine and the cosine in three
// dimensions.
fn ft_ball(r: f64, ndim: i64, tag: const u8[:]) -> f64, Fault {
    if ndim == 1 {
        if r == 0.0 { return 1.0, no_fault(); }
        return mathshim::sine(r) / r, no_fault();
    }
    if ndim == 2 {
        if r == 0.0 { return 1.0, no_fault(); }
        return 2.0 * bessel_j1(r) / r, no_fault();
    }
    if ndim == 3 {
        if r == 0.0 { return 1.0, no_fault(); }
        return 3.0 * (mathshim::sine(r) - r * mathshim::cosine(r)) / (r * r * r), no_fault();
    }
    return 0.0, fault(F_RANGE, tag, "the ellipsoid holds more dimensions than three");
}

// ft_coord gives the place of every dimension of the element at. The
// buffer out holds the coordinates from the last dimension to the
// first: the coordinate of the dimension e stands at ndim - 1 - e.
fn ft_coord(at: i64, shape: i64[:], ndim: i64, out: i64[>..]&) {
    var rest = at;
    var d = ndim - 1;
    while d >= 0 {
        out.push(rest % shape[d]);
        rest = rest / shape[d];
        d -= 1;
    }
}

// ft_response gives the multiplier of one element of the transform.
// The parameters hold one value for every dimension. The mode names the
// filter.
fn ft_response(mode: i64, params: f64[:], shape: i64[..8], ndim: i64, at: i64, tag: const u8[:]) -> f64, f64, Fault {
    var c: i64[>..] = [];
    ft_coord(at, shape, ndim, c);
    var re: f64 = 0.0;
    var im: f64 = 0.0;
    if mode == FT_SHIFT {
        // The shift of the phase. The shift of every dimension adds to
        // the angle of the turn.
        var angle: f64 = 0.0;
        var d: i64 = 0;
        while d < ndim {
            let n = shape[d];
            let k = c[ndim - 1 - d];
            var kf = k as f64;
            if k >= n - n / 2 { kf -= n as f64; }
            angle += 2.0 * 3.141592653589793 * params[d] * kf / n as f64;
            d += 1;
        }
        return mathshim::cosine(angle), -mathshim::sine(angle), no_fault();
    }
    if mode == FT_ELLIPSOID {
        // The square of the radius. The parameters stand for the count
        // of the points of the box of the filter.
        var sum: f64 = 0.0;
        var d: i64 = 0;
        while d < ndim {
            let n = shape[d];
            var ks = c[ndim - 1 - d] as f64;
            if c[ndim - 1 - d] >= n - n / 2 { ks -= n as f64; }
            let t = 3.141592653589793 * params[d] * ks / n as f64;
            if ndim > 1 { sum += t * t; } else { sum += mathshim::abs_of(t); }
            d += 1;
        }
        let r = mathshim::root(sum);
        let g, fr = ft_ball(r, ndim, tag);
        if fr.kind != F_NONE { return 0.0, 0.0, fr; }
        return g, 0.0, no_fault();
    }
    // The Gaussian and the uniform hold one factor for every dimension.
    var prod: f64 = 1.0;
    var d2: i64 = 0;
    while d2 < ndim {
        let n = shape[d2];
        var ks = c[ndim - 1 - d2] as f64;
        if c[ndim - 1 - d2] >= n - n / 2 { ks -= n as f64; }
        if mode == FT_GAUSSIAN {
            let p = 3.141592653589793 * params[d2] / n as f64;
            prod *= mathshim::power_e(-2.0 * p * p * ks * ks);
        } else {
            let t = 3.141592653589793 * params[d2] * ks / n as f64;
            if ks == 0.0 { prod *= 1.0; } else { prod *= mathshim::sine(t) / t; }
        }
        d2 += 1;
    }
    return prod, 0.0, no_fault();
}

// ft_apply transforms the array, multiplies every element of the
// transform with the response of the filter and transforms back. The
// parameters hold one value for every dimension, and the mode names the
// filter.
fn ft_apply(a, mode: i64, params: f64[:], tag: const u8[:]) -> CArr, Fault {
    let f = carr_check(a, tag);
    if f.kind != F_NONE { return carr_empty(), f; }
    if params.len != a.ndim {
        return carr_empty(), fault(F_SHAPE, tag, "the count of the parameters does not agree with the count of the dimensions");
    }
    var d: i64 = 0;
    while d < params.len {
        if params[d] != params[d] {
            return carr_empty(), fault(F_VALUE, tag, "a parameter holds the missing value");
        }
        d += 1;
    }
    let fw, f0 = cfftn(a);
    if f0.kind != F_NONE { return carr_empty(), f0; }
    let count = carr_size(fw);
    var outd: f64[>..] = [];
    var at: i64 = 0;
    while at < count {
        let g, gi, fr = ft_response(mode, params, fw.shape, fw.ndim, at, tag);
        if fr.kind != F_NONE { return carr_empty(), fr; }
        outd.push(fw.data[2 * at] * g - fw.data[2 * at + 1] * gi);
        outd.push(fw.data[2 * at] * gi + fw.data[2 * at + 1] * g);
        at += 1;
    }
    let scaled = carr_make(fw.shape, copy(outd));
    let out, f1 = cifftn(scaled);
    if f1.kind != F_NONE { return carr_empty(), f1; }
    return out, no_fault();
}

// fourier_gaussian multiplies the transform of an array with the
// response of the Gaussian. The parameter sigma holds one value for
// every dimension.
fn fourier_gaussian(a, sigma: f64[:]) -> CArr, Fault {
    return ft_apply(a, FT_GAUSSIAN, sigma, "fourier_gaussian");
}

// fourier_uniform multiplies the transform of an array with the
// response of the mean over a box. The parameter size holds one value
// for every dimension.
fn fourier_uniform(a, size: f64[:]) -> CArr, Fault {
    return ft_apply(a, FT_UNIFORM, size, "fourier_uniform");
}

// fourier_ellipsoid multiplies the transform of an array with the
// response of the mean over an ellipsoid. The response follows the
// count of the dimensions, and the count stands below four. The
// parameter size holds one value for every dimension.
fn fourier_ellipsoid(a, size: f64[:]) -> CArr, Fault {
    return ft_apply(a, FT_ELLIPSOID, size, "fourier_ellipsoid");
}

// fourier_shift shifts an array by the given count of the samples. The
// parameter shift holds one value for every dimension.
fn fourier_shift(a, shift: f64[:]) -> CArr, Fault {
    return ft_apply(a, FT_SHIFT, shift, "fourier_shift");
}

// The largest count of the nodes of the watershed.
const WS_MAX_NODES: i64 = 4194304;

// ws_node holds the state of one node of the watershed. The cost holds
// the cost of the cheapest path from a marker, the label holds the
// marker of the path, and the flags hold the state of the node in the
// queues.
struct WsNode {
    cost: i64,
    label: i64,
    done: bool,
    linked: bool,
    next: i64,
    prev: i64,
}

// ws_strides gives the step of every dimension of an array of the given
// shape. The step of the dimension e is the count of the elements that
// follow one step along it.
fn ws_strides(shape: i64[:], out: i64[>..]&) {
    var e: i64 = 0;
    while e < shape.len {
        out.push(ws_offset(e, shape.len, shape));
        e += 1;
    }
}

// ws_default_structure builds the cross of the connectivity one in the
// given count of the dimensions.
fn ws_default_structure(ndim: i64) -> Arr, Fault {
    var side: i64[>..] = [];
    var d: i64 = 0;
    while d < ndim { side.push(3); d += 1; }
    let s, f = fill(side, 0.0);
    if f.kind != F_NONE { return empty_arr(), f; }
    // The centre of the structure is the place of no neighbour. The
    // cross holds a neighbour on each side of the centre in every
    // dimension.
    let centre = (numel(side) - 1) / 2;
    var i: i64 = 0;
    while i < ndim {
        let step = ws_offset(i, ndim, side);
        s.data[centre + step] = 1.0;
        s.data[centre - step] = 1.0;
        i += 1;
    }
    return s, no_fault();
}

// ws_offset gives the stride of the dimension d of an array of the
// given shape.
fn ws_offset(d: i64, ndim: i64, shape: i64[:]) -> i64 {
    var step: i64 = 1;
    var e = d + 1;
    while e < ndim { step *= shape[e]; e += 1; }
    return step;
}

// ws_neighbours gives the step of every dimension for every neighbour
// of the structure. The structure holds a value of one for every place
// of a neighbour. The buffer holds ndim values for every neighbour, in
// the order of the dimensions. The call gives the count of the
// neighbours.
fn ws_neighbours(st: f64[:], stshape: i64[:], deltas: i64[>..]&) -> i64 {
    let ndim = stshape.len;
    let centre = st.len / 2;
    var count: i64 = 0;
    var d: i64 = 0;
    while d < st.len {
        if st[d] != 0.0 && d != centre {
            var e: i64 = 0;
            while e < ndim {
                let stride = ws_offset(e, ndim, stshape);
                let cc = (d / stride) % stshape[e];
                deltas.push(cc - stshape[e] / 2);
                e += 1;
            }
            count += 1;
        }
        d += 1;
    }
    return count;
}

// ws_dequeue takes the first node of the bucket out of the queue.
fn ws_dequeue(nodes: WsNode[>..]&, heads: i64[>..]&, tails: i64[>..]&, level: i64) -> i64 {
    let v = heads[level];
    if v < 0 { return -1; }
    heads[level] = nodes[v].next;
    if heads[level] < 0 { tails[level] = -1; } else { nodes[heads[level]].prev = -1; }
    nodes[v].next = -1;
    nodes[v].prev = -1;
    nodes[v].linked = false;
    return v;
}

// ws_unlink takes a node out of its queue. The links of the neighbours
// of the node are fixed, so the queue stays whole.
fn ws_unlink(nodes: WsNode[>..]&, heads: i64[>..]&, tails: i64[>..]&, level: i64, v: i64) {
    if !nodes[v].linked { return; }
    let p = nodes[v].prev;
    let n = nodes[v].next;
    if heads[level] == v { heads[level] = n; }
    if p >= 0 { nodes[p].next = n; }
    if tails[level] == v { tails[level] = p; }
    if n >= 0 { nodes[n].prev = p; }
    nodes[v].next = -1;
    nodes[v].prev = -1;
    nodes[v].linked = false;
}

// ws_push puts a node at the front or at the back of the bucket. A node
// of an object stands at the front, a node of the background stands at
// the back.
fn ws_push(nodes: WsNode[>..]&, heads: i64[>..]&, tails: i64[>..]&, level: i64, v: i64, at_front: bool) {
    nodes[v].linked = true;
    if at_front {
        nodes[v].prev = -1;
        nodes[v].next = heads[level];
        if heads[level] >= 0 { nodes[heads[level]].prev = v; }
        heads[level] = v;
        if tails[level] < 0 { tails[level] = v; }
    } else {
        nodes[v].next = -1;
        nodes[v].prev = tails[level];
        if tails[level] >= 0 { nodes[tails[level]].next = v; }
        tails[level] = v;
        if heads[level] < 0 { heads[level] = v; }
    }
}

// watershed_ift labels the regions of an image by the watershed of the
// distance transform. The input holds the values of the image, which
// stand from zero up. The markers hold a positive value for the marker
// of an object, a negative value for the marker of the background and
// zero for the nodes without a marker. The structure holds the shape of
// the neighbourhood, and an empty structure takes the cross of the
// connectivity one. The result holds one label for every node.
fn watershed_ift(input, markers, structure: Arr = Arr { ndim: 0, shape: [], data: [] }) -> Arr, Fault {
    if input.ndim != markers.ndim || input.data.len != markers.data.len {
        return empty_arr(), fault(F_SHAPE, "watershed_ift", "the input and the markers differ in shape");
    }
    // The buffer of every operand must agree with its shape, because the
    // walk reads the nodes by the shape of the input.
    let fs = check_stack(input.ndim, input.shape, input.data.len, 1, "watershed_ift");
    if fs.kind != F_NONE { return empty_arr(), fs; }
    let fm = check_stack(markers.ndim, markers.shape, markers.data.len, 1, "watershed_ift");
    if fm.kind != F_NONE { return empty_arr(), fm; }
    if input.data.len < 1 || input.data.len > WS_MAX_NODES {
        return empty_arr(), fault(F_LIMIT, "watershed_ift", "the count of the nodes stands outside one to the limit");
    }
    let n = input.data.len;
    let ndim = input.ndim;
    var i: i64 = 0;
    while i < n {
        if input.data[i] != input.data[i] || input.data[i] < 0.0 {
            return empty_arr(), fault(F_VALUE, "watershed_ift", "a value of the input stands below zero");
        }
        i += 1;
    }
    // The structure. An empty structure takes the cross.
    var st = structure;
    if st.ndim != ndim {
        let cross, fc = ws_default_structure(ndim);
        if fc.kind != F_NONE { return empty_arr(), fc; }
        st = copy(cross);
    }
    if st.ndim != ndim || st.data.len != numel(st.shape) {
        return empty_arr(), fault(F_SHAPE, "watershed_ift", "the structure does not agree with the count of the dimensions");
    }
    // The largest value of the input gives the count of the buckets.
    var maxval: i64 = 0;
    i = 0;
    while i < n {
        let v = mathshim::floor_of(input.data[i]) as i64;
        if v > maxval { maxval = v; }
        i += 1;
    }
    var nodes: WsNode[>..] = [];
    var heads: i64[>..] = [];
    var tails: i64[>..] = [];
    i = 0;
    while i <= maxval {
        heads.push(-1);
        tails.push(-1);
        i += 1;
    }
    // The state of every node and the queues. A marker enters the
    // bucket of the cost zero. A marker of an object stands at the
    // front, a marker of the background stands at the back.
    i = 0;
    while i < n {
        let m = mathshim::floor_of(markers.data[i]) as i64;
        if m != 0 {
            nodes.push(WsNode { cost: 0, label: m, done: false, linked: false, next: -1, prev: -1 });
        } else {
            nodes.push(WsNode { cost: maxval + 1, label: 0, done: false, linked: false, next: -1, prev: -1 });
        }
        i += 1;
    }
    i = 0;
    while i < n {
        let m = mathshim::floor_of(markers.data[i]) as i64;
        if m != 0 {
            ws_push(nodes, heads, tails, 0, i, m > 0);
        }
        i += 1;
    }
    // The steps of the neighbours. The buffer holds one step for every
    // dimension of every neighbour.
    var strides: i64[>..] = [];
    ws_strides(input.shape, strides);
    var deltas: i64[>..] = [];
    let nn = ws_neighbours(st.data, st.shape, deltas);
    // The propagation. The buckets run from the cost zero to the
    // largest value. A node of a bucket carries its label to the
    // neighbour whose cost falls.
    var level: i64 = 0;
    while level <= maxval {
        while heads[level] >= 0 {
            let v = ws_dequeue(nodes, heads, tails, level);
            nodes[v].done = true;
            // The coordinates of the node. ft_coord pushes the
            // coordinates from the last dimension to the first, so the
            // coordinate of the dimension e stands at ndim - 1 - e.
            var c: i64[>..] = [];
            ft_coord(v, input.shape, ndim, c);
            var h: i64 = 0;
            while h < nn {
                // The neighbour stands at the steps of the deltas.
                var p = v;
                var ok = true;
                var e: i64 = 0;
                while e < ndim {
                    let step = strides[e];
                    let at = c[ndim - 1 - e];
                    let nc = at + deltas[h * ndim + e];
                    if nc < 0 || nc >= input.shape[e] { ok = false; break; }
                    p += (nc - at) * step;
                    e += 1;
                }
                if ok {
                    let pc = nodes[p];
                    if !pc.done {
                        let pv = mathshim::floor_of(input.data[p]) as i64;
                        let vv = mathshim::floor_of(input.data[v]) as i64;
                        var wvp = pv - vv;
                        if wvp < 0 { wvp = -wvp; }
                        var mx = nodes[v].cost;
                        if wvp > mx { mx = wvp; }
                        if mx < nodes[p].cost {
                            let place = nodes[p].cost;
                            if place <= maxval { ws_unlink(nodes, heads, tails, place, p); }
                            nodes[p].cost = mx;
                            nodes[p].label = nodes[v].label;
                            ws_push(nodes, heads, tails, mx, p, nodes[v].label > 0);
                        }
                    }
                }
                h += 1;
            }
        }
        level += 1;
    }
    // The labels that no marker reached hold zero.
    var out: f64[>..] = [];
    i = 0;
    while i < n { out.push(nodes[i].label as f64); i += 1; }
    return line_arr(copy(out)), no_fault();
}
