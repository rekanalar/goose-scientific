// The vectorized filter of N dimensions, as in
// scipy.ndimage.vectorized_filter.
//
// The call gives a block the windowed array of the input. The windowed
// array holds one window for every place of the input. Its shape is the
// shape of the input followed by the shape of the window. The block fills
// the buffer of the result, which holds one value for every place of the
// input. The block reduces the window axes itself, as the function of
// SciPy does.
//
// The window reaches outside the input. The border follows the mode, as in
// the other calls of N dimensions. The mode "reflect" repeats the edge
// value. The mode "mirror" does not.
//
// The time of a block is the time of the caller. The package cannot
// interrupt a block that does not stop.
namespace scientific;

import mathshim;

// vectorized_filter runs the block over the windowed array of the input.
fn vectorized_filter<F>(a, size: i64[:], mode: const u8[:] = "reflect", cval: f64 = 0.0, origin: i64[:] = []) -> Arr, Fault {
    let tag = "vectorized_filter";
    let m, f = nd_check(a, mode, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let nd = a.ndim;
    if size.len != nd {
        return empty_arr(), fault(F_SHAPE, tag, "the window holds a count that is not one for each axis");
    }
    var og: i64[..8] = [];
    if !nd_list(origin, nd, 0, og) {
        return empty_arr(), fault(F_SHAPE, tag, "the origin holds a count that is not one for each axis");
    }
    var d: i64 = 0;
    while d < nd {
        if size[d] < 1 { return empty_arr(), fault(F_VALUE, tag, "the window holds a value below one"); }
        if !nd_origin_ok(og[d], size[d]) {
            return empty_arr(), fault(F_RANGE, tag, "the origin stands outside the window");
        }
        d += 1;
    }
    // The border of every axis and the counts of the window and the border.
    var bsh: i64[..8] = [];
    var bstr: i64[..8] = [];
    var prodw: i64 = 1;
    var prodb: i64 = 1;
    d = 0;
    while d < nd {
        let before = size[d] / 2 + og[d];
        let after = size[d] - 1 - before;
        bsh.push(a.shape[d] + before + after);
        if prodb > MAX_ELEMENTS / bsh[d] {
            return empty_arr(), fault(F_LIMIT, tag, "the border holds more values than the limit");
        }
        prodb *= bsh[d];
        if prodw > MAX_ELEMENTS / size[d] {
            return empty_arr(), fault(F_LIMIT, tag, "the window holds more values than the limit");
        }
        prodw *= size[d];
        d += 1;
    }
    d = 0;
    while d < nd {
        var s: i64 = 1;
        var e = nd - 1;
        while e > d { s *= bsh[e]; e -= 1; }
        bstr.push(s);
        d += 1;
    }
    let prodn = a.data.len;
    if prodw > 0 && prodn > MAX_ELEMENTS / prodw {
        return empty_arr(), fault(F_LIMIT, tag, "the windowed array holds more values than the limit");
    }
    // The border of the input.
    var bord: f64[>..] = nd_zeros(prodb);
    var ix: i64[..8] = [];
    d = 0;
    while d < nd { ix.push(0); d += 1; }
    var bi: i64 = 0;
    while bi < prodb {
        var flat: i64 = 0;
        var ok = true;
        d = 0;
        while d < nd {
            let s = nd_src(ix[d] - (size[d] / 2 + og[d]), a.shape[d], m);
            if s < 0 { ok = false; } else { flat = flat * a.shape[d] + s; }
            d += 1;
        }
        if ok { bord[bi] = a.data[flat]; } else { bord[bi] = cval; }
        gf_next(ix, bsh);
        bi += 1;
    }
    // The windowed array. The windows follow the places of the input.
    var win: f64[>..] = nd_zeros(prodn * prodw);
    var pix: i64[..8] = [];
    var wix: i64[..8] = [];
    d = 0;
    while d < nd { pix.push(0); wix.push(0); d += 1; }
    var p: i64 = 0;
    while p < prodn {
        var base: i64 = 0;
        d = 0;
        while d < nd { base = base * bsh[d] + pix[d]; d += 1; }
        d = 0;
        while d < nd { wix[d] = 0; d += 1; }
        var w: i64 = 0;
        while w < prodw {
            var flat = base;
            d = 0;
            while d < nd { flat += wix[d] * bstr[d]; d += 1; }
            win[p * prodw + w] = bord[flat];
            gf_next(wix, size);
            w += 1;
        }
        gf_next(pix, a.shape);
        p += 1;
    }
    var wsh: i64[..8] = [];
    d = 0;
    while d < nd { wsh.push(a.shape[d]); d += 1; }
    d = 0;
    while d < nd { wsh.push(size[d]); d += 1; }
    let warr = arr_of(copy(win), wsh);
    var out: f64[>..] = nd_zeros(prodn);
    var o = out;
    F(copy(warr), o);
    return arr_of(copy(out), a.shape), no_fault();
}
