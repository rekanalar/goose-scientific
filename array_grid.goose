// The grids of coordinates and the cross product of vectors.
namespace scientific;

// meshgrid gives the coordinate arrays of a grid from two vectors. The
// indexing "xy" gives the shape (len(y), len(x)). The indexing "ij"
// gives the shape (len(x), len(y)). The sparse form gives the shapes
// of one row and one column, so the arrays broadcast to the grid.
fn meshgrid(x, y, indexing: const u8[:] = "xy", sparse: bool = false) -> Arr, Arr, Fault {
    if x.ndim != 1 || y.ndim != 1 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "meshgrid", "an input holds not one dimension");
    }
    var xy = true;
    if indexing == "ij" { xy = false; }
    else if indexing != "xy" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "meshgrid", "the indexing is not xy and not ij");
    }
    let nx = x.data.len;
    let ny = y.data.len;
    // The first array varies with x, the second one with y. In the
    // "ij" form the first dimension holds x.
    var rows = ny;
    var cols = nx;
    if !xy { rows = nx; cols = ny; }
    var sa: i64[..8] = [];
    var sb: i64[..8] = [];
    var da: f64[>..] = [];
    var db: f64[>..] = [];
    if sparse {
        if xy { sa.push(1); sa.push(nx); sb.push(ny); sb.push(1); }
        else { sa.push(nx); sa.push(1); sb.push(1); sb.push(ny); }
        var i: i64 = 0;
        while i < nx { da.push(x.data[i]); i += 1; }
        i = 0;
        while i < ny { db.push(y.data[i]); i += 1; }
        return Arr { ndim: 2, shape: sa, data: da }, Arr { ndim: 2, shape: sb, data: db }, no_fault();
    }
    sa.push(rows); sa.push(cols);
    sb.push(rows); sb.push(cols);
    let cf = check_shape(sa);
    if cf.kind != F_NONE { return empty_arr(), empty_arr(), cf; }
    var r: i64 = 0;
    while r < rows {
        var c: i64 = 0;
        while c < cols {
            if xy { da.push(x.data[c]); db.push(y.data[r]); }
            else { da.push(x.data[r]); db.push(y.data[c]); }
            c += 1;
        }
        r += 1;
    }
    return Arr { ndim: 2, shape: sa, data: da }, Arr { ndim: 2, shape: sb, data: db }, no_fault();
}

// cross gives the cross product of vectors of three elements, as NumPy
// 2.x gives it. The last dimension holds the vector. An array of one
// dimension broadcasts against the rows of an array of two dimensions.
fn cross(a, b) -> Arr, Fault {
    if a.ndim > 2 || b.ndim > 2 {
        return empty_arr(), fault(F_LIMIT, "cross", "an array holds more than two dimensions");
    }
    if a.shape[a.ndim - 1] != 3 || b.shape[b.ndim - 1] != 3 {
        return empty_arr(), fault(F_SHAPE, "cross", "a vector holds not three elements");
    }
    var na: i64 = 1;
    var nb: i64 = 1;
    if a.ndim == 2 { na = a.shape[0]; }
    if b.ndim == 2 { nb = b.shape[0]; }
    if na != nb && na != 1 && nb != 1 {
        return empty_arr(), fault(F_SHAPE, "cross", "the counts of vectors differ");
    }
    var n = na;
    if nb > n { n = nb; }
    if na == 0 || nb == 0 { n = 0; }
    var d: f64[>..] = [];
    var k: i64 = 0;
    while k < n {
        var ia: i64 = 0;
        var ib: i64 = 0;
        if na > 1 { ia = k * 3; }
        if nb > 1 { ib = k * 3; }
        let a0 = a.data[ia]; let a1 = a.data[ia + 1]; let a2 = a.data[ia + 2];
        let b0 = b.data[ib]; let b1 = b.data[ib + 1]; let b2 = b.data[ib + 2];
        d.push(a1 * b2 - a2 * b1);
        d.push(a2 * b0 - a0 * b2);
        d.push(a0 * b1 - a1 * b0);
        k += 1;
    }
    var s: i64[..8] = [];
    if a.ndim == 2 || b.ndim == 2 { s.push(n); }
    s.push(3);
    return Arr { ndim: s.len, shape: s, data: d }, no_fault();
}

// grid_cell finds the cell of a point along one grid axis. The grid
// starts at offset in the flat buffer and holds n values. The cell index
// stays within 0 to n - 2, so a point outside the grid uses the end cell.
fn grid_cell(grids: f64[:], offset: i64, n: i64, x: f64) -> i64 {
    var lo: i64 = 0;
    var hi: i64 = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if grids[offset + mid] <= x { lo = mid; } else { hi = mid; }
    }
    return lo;
}

// interpn interpolates a regular grid of N dimensions at many points,
// as scipy.interpolate.interpn and RegularGridInterpolator do. The
// grids come in one flat list, one axis after the other, and lens holds
// the length of each axis. The values hold one number for each grid
// point. The points are a matrix with one row for each point and one
// column for each axis. The method is "linear" or "nearest". A point
// outside the grid is a fault when bounds_error holds, and gives
// fill_value otherwise.
fn interpn(grids: f64[:], lens: i64[:], values, xi, method: const u8[:] = "linear", bounds_error: bool = true, fill_value: f64 = 0.0 / 0.0) -> Arr, Fault {
    let nd = lens.len;
    if nd < 1 || nd > MAX_DIMS || values.ndim != nd {
        return empty_arr(), fault(F_SHAPE, "interpn", "the values do not hold one axis for each grid");
    }
    if method != "linear" && method != "nearest" {
        return empty_arr(), fault(F_VALUE, "interpn", "the method holds an unknown name");
    }
    var offs: i64[>..] = [];
    var total: i64 = 0;
    var d: i64 = 0;
    while d < nd {
        if lens[d] < 2 || values.shape[d] != lens[d] {
            return empty_arr(), fault(F_SHAPE, "interpn", "a grid holds fewer than two points or does not fit the values");
        }
        offs.push(total);
        total += lens[d];
        d += 1;
    }
    if grids.len != total {
        return empty_arr(), fault(F_SHAPE, "interpn", "the grids do not hold the lengths in lens");
    }
    d = 0;
    while d < nd {
        var k: i64 = 0;
        while k < lens[d] {
            let g = grids[offs[d] + k];
            if !(g - g == 0.0) { return empty_arr(), fault(F_VALUE, "interpn", "a grid holds a value that is not finite"); }
            if k > 0 && !(g > grids[offs[d] + k - 1]) {
                return empty_arr(), fault(F_VALUE, "interpn", "a grid does not rise strictly");
            }
            k += 1;
        }
        d += 1;
    }
    var strides: i64[>..] = [];
    strides_of(values.shape, strides);
    var np: i64 = 0;
    if xi.ndim == 2 && xi.shape[1] == nd { np = xi.shape[0]; }
    else if xi.ndim == 1 && nd == xi.shape[0] { np = 1; }
    else { return empty_arr(), fault(F_SHAPE, "interpn", "the points do not hold one column for each axis"); }
    var out: f64[>..] = [];
    var p: i64 = 0;
    while p < np {
        var inside = true;
        var idx: i64[>..] = [];
        var frac: f64[>..] = [];
        d = 0;
        while d < nd {
            let x = xi.data[p * nd + d];
            let lo = grids[offs[d]];
            let hi = grids[offs[d] + lens[d] - 1];
            if x != x { inside = false; }
            else if x < lo || x > hi {
                if bounds_error { return empty_arr(), fault(F_RANGE, "interpn", "a point stands outside the grid"); }
                inside = false;
            }
            var c: i64 = 0;
            var t: f64 = 0.0;
            if x == x {
                c = grid_cell(grids, offs[d], lens[d], x);
                t = (x - grids[offs[d] + c]) / (grids[offs[d] + c + 1] - grids[offs[d] + c]);
            }
            idx.push(c);
            frac.push(t);
            d += 1;
        }
        if !inside { out.push(fill_value); p += 1; continue; }
        var v: f64 = 0.0;
        if method == "nearest" {
            var at: i64 = 0;
            d = 0;
            while d < nd {
                var c = idx[d];
                if frac[d] > 0.5 { c += 1; }
                at += c * strides[d];
                d += 1;
            }
            v = values.data[at];
        } else {
            var corner: i64 = 0;
            while corner < (1 << nd) {
                var w: f64 = 1.0;
                var at: i64 = 0;
                d = 0;
                while d < nd {
                    if (corner >> d) % 2 == 1 { w *= frac[d]; at += (idx[d] + 1) * strides[d]; }
                    else { w *= 1.0 - frac[d]; at += idx[d] * strides[d]; }
                    d += 1;
                }
                if w != 0.0 { v += w * values.data[at]; }
                corner += 1;
            }
        }
        out.push(v);
        p += 1;
    }
    return line_arr(copy(out)), no_fault();
}
