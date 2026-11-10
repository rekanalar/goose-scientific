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
