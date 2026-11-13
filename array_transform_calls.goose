// The public calls map_coordinates, shift, zoom, affine_transform and rotate.
//
namespace scientific;

import mathshim;

// map_coordinates reads the spline at the coordinates. The array of the
// coordinates holds one row for each axis of the input, so its shape is the
// count of axes followed by the shape of the output.
fn map_coordinates(a, coordinates, order: i64 = 3, mode: const u8[:] = "constant", cval: f64 = 0.0, prefilter: bool = true) -> Arr, Fault {
    let m, f = tr_checked(a, order, mode, "map_coordinates");
    if f.kind != F_NONE { return empty_arr(), f; }
    let fc = check_stack(coordinates.ndim, coordinates.shape, coordinates.data.len, 1, "map_coordinates");
    if fc.kind != F_NONE { return empty_arr(), fc; }
    if coordinates.ndim < 2 || coordinates.shape[0] != a.ndim {
        return empty_arr(), fault(F_SHAPE, "map_coordinates", "the coordinates hold a first axis that is not one row for each axis of the array");
    }
    var oshape: i64[..8] = [];
    var d: i64 = 1;
    while d < coordinates.ndim { oshape.push(coordinates.shape[d]); d += 1; }
    return tr_eval(a, order, m, cval, prefilter, 0, coordinates.data, [], [], false, [], [], oshape, "map_coordinates");
}

// tr_floats reads a list of one value or one value for each axis.
fn tr_floats(v: f64[:], nd: i64, default: f64, out: f64[>..]&) -> bool {
    var d: i64 = 0;
    if v.len == 0 { while d < nd { out.push(default); d += 1; } return true; }
    if v.len == 1 { while d < nd { out.push(v[0]); d += 1; } return true; }
    if v.len != nd { return false; }
    while d < nd { out.push(v[d]); d += 1; }
    return true;
}

// shift moves the array by a shift on each axis. A positive shift moves the
// values toward higher indices.
fn shift(a, shift: f64[:], order: i64 = 3, mode: const u8[:] = "constant", cval: f64 = 0.0, prefilter: bool = true) -> Arr, Fault {
    let m, f = tr_checked(a, order, mode, "shift");
    if f.kind != F_NONE { return empty_arr(), f; }
    var sh: f64[>..] = [];
    if shift.len == 0 || !tr_floats(shift, a.ndim, 0.0, sh) {
        return empty_arr(), fault(F_SHAPE, "shift", "the shift holds a count that is not one for each axis");
    }
    var i: i64 = 0;
    while i < sh.len { sh[i] = -sh[i]; i += 1; }
    return tr_eval(a, order, m, cval, prefilter, 1, [], [], sh, false, [], [], a.shape, "shift");
}

// tr_round gives a float rounded to the nearest integer, a tie to the even one.
fn tr_round(x: f64) -> f64 {
    let r = mathshim::floor_of(x + 0.5);
    if r - 0.5 == x {
        let h = r / 2.0;
        if mathshim::floor_of(h) != h { return r - 1.0; }
    }
    return r;
}

// zoom resamples the array to the shape of round(size * zoom). A grid mode of
// true treats each value as the centre of a cell, so the edges stay in place.
fn zoom(a, zoom: f64[:], order: i64 = 3, mode: const u8[:] = "constant", cval: f64 = 0.0, prefilter: bool = true, grid_mode: bool = false) -> Arr, Fault {
    let m, f = tr_checked(a, order, mode, "zoom");
    if f.kind != F_NONE { return empty_arr(), f; }
    var zm: f64[>..] = [];
    if zoom.len == 0 || !tr_floats(zoom, a.ndim, 1.0, zm) {
        return empty_arr(), fault(F_SHAPE, "zoom", "the zoom holds a count that is not one for each axis");
    }
    var oshape: i64[..8] = [];
    var ratio: f64[>..] = [];
    var same = true;
    var d: i64 = 0;
    while d < a.ndim {
        let z = zm[d];
        if !(z > 0.0 && z <= 1.0e15) {
            return empty_arr(), fault(F_VALUE, "zoom", "the zoom holds a value that is not above zero or is too large");
        }
        if z != 1.0 { same = false; }
        let sz = tr_round((a.shape[d] as f64) * z);
        if sz < 1.0 { return empty_arr(), fault(F_SHAPE, "zoom", "the zoom gives an axis with no place"); }
        if sz > 4.0e18 { return empty_arr(), fault(F_LIMIT, "zoom", "the zoom gives an axis that is too large"); }
        let osz = sz as i64;
        oshape.push(osz);
        var nom = a.shape[d] as f64;
        var div = osz as f64;
        if !grid_mode { nom -= 1.0; div -= 1.0; }
        if div != 0.0 { ratio.push(nom / div); } else { ratio.push(1.0); }
        d += 1;
    }
    if same && prefilter { return clone(a); }
    return tr_eval(a, order, m, cval, prefilter, 1, [], ratio, [], grid_mode, [], [], oshape, "zoom");
}

// affine_transform reads the input at matrix * output + offset. A matrix of
// one dimension holds the diagonal. A matrix of two dimensions has one row for
// each axis of the input and one column for each axis of the output. A matrix
// with one more column holds the offset in the last column. The output shape
// defaults to the shape of the input.
fn affine_transform(a, matrix, offset: f64[:] = [], output_shape: i64[:] = [], order: i64 = 3, mode: const u8[:] = "constant", cval: f64 = 0.0, prefilter: bool = true) -> Arr, Fault {
    let m, f = tr_checked(a, order, mode, "affine_transform");
    if f.kind != F_NONE { return empty_arr(), f; }
    let fm = check_stack(matrix.ndim, matrix.shape, matrix.data.len, 1, "affine_transform");
    if fm.kind != F_NONE { return empty_arr(), fm; }
    var oshape: i64[..8] = [];
    var d: i64 = 0;
    if output_shape.len == 0 {
        while d < a.ndim { oshape.push(a.shape[d]); d += 1; }
    } else {
        if output_shape.len > 8 { return empty_arr(), fault(F_SHAPE, "affine_transform", "the output shape holds more than 8 axes"); }
        while d < output_shape.len { oshape.push(output_shape[d]); d += 1; }
    }
    var off: f64[>..] = [];
    if !tr_floats(offset, a.ndim, 0.0, off) {
        return empty_arr(), fault(F_SHAPE, "affine_transform", "the offset holds a count that is not one for each axis");
    }
    var fin = true;
    d = 0;
    while d < matrix.data.len { if !(matrix.data[d] - matrix.data[d] == 0.0) { fin = false; } d += 1; }
    d = 0;
    while d < off.len { if !(off[d] - off[d] == 0.0) { fin = false; } d += 1; }
    if !fin { return empty_arr(), fault(F_VALUE, "affine_transform", "the matrix or the offset holds a value that is not finite"); }
    let nd = a.ndim;
    if matrix.ndim == 1 {
        if matrix.shape[0] != nd || oshape.len != nd {
            return empty_arr(), fault(F_SHAPE, "affine_transform", "a matrix of one dimension needs one value for each axis of the input and the output");
        }
        var zs: f64[>..] = [];
        var sh: f64[>..] = [];
        d = 0;
        while d < nd {
            if matrix.data[d] == 0.0 { return empty_arr(), fault(F_VALUE, "affine_transform", "the diagonal holds a zero"); }
            zs.push(matrix.data[d]);
            sh.push(off[d] / matrix.data[d]);
            d += 1;
        }
        return tr_eval(a, order, m, cval, prefilter, 1, [], zs, sh, false, [], [], oshape, "affine_transform");
    }
    if matrix.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "affine_transform", "the matrix holds more than two dimensions");
    }
    var mat: f64[>..] = [];
    let rows = matrix.shape[0];
    let cols = matrix.shape[1];
    if cols == nd + 1 && (rows == nd || rows == nd + 1) {
        if rows == nd + 1 {
            d = 0;
            while d < nd + 1 {
                let want: f64 = if d == nd { 1.0 } else { 0.0 };
                if matrix.data[nd * cols + d] != want {
                    return empty_arr(), fault(F_VALUE, "affine_transform", "the last row of a matrix of homogeneous form is not 0 ... 0 1");
                }
                d += 1;
            }
        }
        var i: i64 = 0;
        while i < nd {
            var j: i64 = 0;
            while j < nd { mat.push(matrix.data[i * cols + j]); j += 1; }
            i += 1;
        }
        off = [];
        i = 0;
        while i < nd { off.push(matrix.data[i * cols + nd]); i += 1; }
        if oshape.len != nd { return empty_arr(), fault(F_SHAPE, "affine_transform", "the output shape does not match the matrix"); }
    } else {
        if rows != nd { return empty_arr(), fault(F_SHAPE, "affine_transform", "the matrix holds a count of rows that is not one for each axis of the input"); }
        if cols != oshape.len { return empty_arr(), fault(F_SHAPE, "affine_transform", "the matrix holds a count of columns that is not one for each axis of the output"); }
        var i: i64 = 0;
        while i < matrix.data.len { mat.push(matrix.data[i]); i += 1; }
    }
    return tr_eval(a, order, m, cval, prefilter, 2, [], [], [], false, mat, off, oshape, "affine_transform");
}

// tr_sindg and tr_cosdg give the sine and the cosine of an angle in degrees.
// They give exact values at multiples of 90, as the routines of Cephes do.
fn tr_sindg(x0: f64) -> f64 {
    var x = x0;
    var sign: f64 = 1.0;
    if x < 0.0 { x = -x; sign = -1.0; }
    var y = mathshim::floor_of(x / 45.0);
    var z = mathshim::floor_of(y / 16.0);
    z = y - z * 16.0;
    var j = z as i64;
    if j % 2 == 1 { j += 1; y += 1.0; }
    j = j % 8;
    z = (x - y * 45.0) * 0.017453292519943295;
    if j > 3 { sign = -sign; j -= 4; }
    if j == 1 || j == 2 { return sign * mathshim::cosine(z); }
    return sign * mathshim::sine(z);
}

fn tr_cosdg(x0: f64) -> f64 {
    var x = x0;
    var sign: f64 = 1.0;
    if x < 0.0 { x = -x; }
    var y = mathshim::floor_of(x / 45.0);
    var z = mathshim::floor_of(y / 16.0);
    z = y - z * 16.0;
    var j = z as i64;
    if j % 2 == 1 { j += 1; y += 1.0; }
    j = j % 8;
    z = (x - y * 45.0) * 0.017453292519943295;
    if j > 3 { j -= 4; sign = -sign; }
    if j > 1 { sign = -sign; }
    if j == 1 || j == 2 { return sign * mathshim::sine(z); }
    return sign * mathshim::cosine(z);
}

// rotate turns the array in the plane of two axes by an angle in degrees. A
// reshape of true grows the plane so that no corner leaves it.
fn rotate(a, angle: f64, axes: i64[:] = [1, 0], reshape: bool = true, order: i64 = 3, mode: const u8[:] = "constant", cval: f64 = 0.0, prefilter: bool = true) -> Arr, Fault {
    let m, f = tr_checked(a, order, mode, "rotate");
    if f.kind != F_NONE { return empty_arr(), f; }
    let nd = a.ndim;
    if nd < 2 { return empty_arr(), fault(F_SHAPE, "rotate", "the array holds less than two dimensions"); }
    if axes.len != 2 { return empty_arr(), fault(F_SHAPE, "rotate", "the axes hold a count that is not two"); }
    var a0 = axes[0];
    var a1 = axes[1];
    if a0 < 0 { a0 += nd; }
    if a1 < 0 { a1 += nd; }
    if a0 < 0 || a1 < 0 || a0 >= nd || a1 >= nd || a0 == a1 {
        return empty_arr(), fault(F_RANGE, "rotate", "the axes stand outside the array or repeat");
    }
    if a0 > a1 { let t = a0; a0 = a1; a1 = t; }
    if !(angle >= -1.0e14 && angle <= 1.0e14) {
        return empty_arr(), fault(F_VALUE, "rotate", "the angle is not finite or reaches beyond 1.0e14 degrees");
    }
    let c = tr_cosdg(angle);
    let s = tr_sindg(angle);
    let iy = a.shape[a0] as f64;
    let ix = a.shape[a1] as f64;
    var oy = a.shape[a0];
    var ox = a.shape[a1];
    if reshape {
        let r0 = [c * 0.0 + s * 0.0, c * 0.0 + s * ix, c * iy + s * 0.0, c * iy + s * ix];
        let r1 = [(-s) * 0.0 + c * 0.0, (-s) * 0.0 + c * ix, (-s) * iy + c * 0.0, (-s) * iy + c * ix];
        var lo0 = r0[0];
        var hi0 = r0[0];
        var lo1 = r1[0];
        var hi1 = r1[0];
        var k: i64 = 1;
        while k < 4 {
            if r0[k] < lo0 { lo0 = r0[k]; }
            if r0[k] > hi0 { hi0 = r0[k]; }
            if r1[k] < lo1 { lo1 = r1[k]; }
            if r1[k] > hi1 { hi1 = r1[k]; }
            k += 1;
        }
        let py = (hi0 - lo0) + 0.5;
        let px = (hi1 - lo1) + 0.5;
        if py > 4.0e18 || px > 4.0e18 { return empty_arr(), fault(F_LIMIT, "rotate", "the rotated plane is too large"); }
        oy = py as i64;
        ox = px as i64;
    }
    let cy = ((oy as f64) - 1.0) / 2.0;
    let cx = ((ox as f64) - 1.0) / 2.0;
    let occ0 = c * cy + s * cx;
    let occ1 = (-s) * cy + c * cx;
    let off = [(iy - 1.0) / 2.0 - occ0, (ix - 1.0) / 2.0 - occ1];
    let mat = [c, s, -s, c];
    var oshape: i64[..8] = [];
    var d: i64 = 0;
    while d < nd {
        if d == a0 { oshape.push(oy); } else if d == a1 { oshape.push(ox); } else { oshape.push(a.shape[d]); }
        d += 1;
    }
    let fo = check_shape(oshape);
    if fo.kind != F_NONE { return empty_arr(), fault(fo.kind, "rotate", fo.detail); }
    let total = numel(oshape);
    if nd == 2 {
        return tr_eval(a, order, m, cval, prefilter, 2, [], [], [], false, mat, off, oshape, "rotate");
    }
    // Planes: one transform for each position on the other axes.
    var planes: i64 = 1;
    d = 0;
    while d < nd {
        if d != a0 && d != a1 { planes *= a.shape[d]; }
        d += 1;
    }
    var out: f64[>..] = nd_zeros(total);
    var pix: i64[..8] = [];
    d = 0;
    while d < nd { pix.push(0); d += 1; }
    var pl: i64 = 0;
    while pl < planes {
        // Gather the input plane at the position pix of the other axes.
        var plane: f64[>..] = [];
        var u: i64 = 0;
        while u < a.shape[a0] {
            var v: i64 = 0;
            while v < a.shape[a1] {
                var flat: i64 = 0;
                d = 0;
                while d < nd {
                    var q = pix[d];
                    if d == a0 { q = u; } else if d == a1 { q = v; }
                    flat = flat * a.shape[d] + q;
                    d += 1;
                }
                plane.push(a.data[flat]);
                v += 1;
            }
            u += 1;
        }
        let pa = arr_of(copy(plane), [a.shape[a0], a.shape[a1]]);
        let r, g = tr_eval(pa, order, m, cval, prefilter, 2, [], [], [], false, mat, off, [oy, ox], "rotate");
        if g.kind != F_NONE { return empty_arr(), g; }
        u = 0;
        while u < oy {
            var v: i64 = 0;
            while v < ox {
                var flat: i64 = 0;
                d = 0;
                while d < nd {
                    var q = pix[d];
                    if d == a0 { q = u; } else if d == a1 { q = v; }
                    flat = flat * oshape[d] + q;
                    d += 1;
                }
                out[flat] = r.data[u * ox + v];
                v += 1;
            }
            u += 1;
        }
        d = nd - 1;
        while d >= 0 {
            if d == a0 || d == a1 {
                d -= 1;
            } else {
                pix[d] += 1;
                if pix[d] < a.shape[d] { d = -1; } else { pix[d] = 0; d -= 1; }
            }
        }
        pl += 1;
    }
    return arr_of(copy(out), oshape), no_fault();
}
