// The convolution and the correlation of two matrices.
//
// The formulas follow SciPy 1.18.1 (scipy.signal).
namespace scientific;

// convolve2d convolves two matrices, as SciPy convolve2d does. The
// mode is "full", "same" or "valid". The mode "valid" needs the
// first matrix to hold at least as many rows and columns as the
// second one, or the other way about. The call gives the matrix of the
// result.
fn convolve2d(a, b, mode: const u8[:] = "full") -> Arr, Fault {
    let tag = "convolve2d";
    if a.ndim != 2 || b.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, tag, "the matrices must hold two dimensions");
    }
    if a.data.len < 1 || b.data.len < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the matrices must hold one value or more");
    }
    // The kernel reads the matrices by their shape, so the buffer of
    // each matrix must agree with its shape.
    let sa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if sa.kind != F_NONE { return empty_arr(), sa; }
    let sb = check_stack(b.ndim, b.shape, b.data.len, 1, tag);
    if sb.kind != F_NONE { return empty_arr(), sb; }
    if mode != "full" && mode != "same" && mode != "valid" {
        return empty_arr(), fault(F_VALUE, tag, "the mode is unknown");
    }
    var fa = a;
    var fb = b;
    if mode == "valid" {
        let small = a.shape[0] < b.shape[0] && a.shape[1] < b.shape[1];
        let large = b.shape[0] < a.shape[0] && b.shape[1] < a.shape[1];
        if small {
            // The two matrices stand apart, so the copies are built in
            // a variable of their own.
            let na = copy(b);
            let nb = copy(a);
            fa = copy(na);
            fb = copy(nb);
        } else if !large {
            if !(a.shape[0] >= b.shape[0] && a.shape[1] >= b.shape[1]) {
                return empty_arr(), fault(F_SHAPE, tag, "the mode valid needs one matrix as large as the other in every direction");
            }
        }
    }
    let m1 = fa.shape[0];
    let n1 = fa.shape[1];
    let m2 = fb.shape[0];
    let n2 = fb.shape[1];
    var rows = m1 + m2 - 1;
    var cols = n1 + n2 - 1;
    var ro: i64 = 0;
    var co: i64 = 0;
    if mode == "same" {
        rows = m1;
        cols = n1;
        ro = (m2 - 1) / 2;
        co = (n2 - 1) / 2;
    } else if mode == "valid" {
        rows = m1 - m2 + 1;
        cols = n1 - n2 + 1;
        ro = m2 - 1;
        co = n2 - 1;
    }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < rows * cols { out.push(0.0); i += 1; }
    var r: i64 = 0;
    while r < rows {
        var c: i64 = 0;
        while c < cols {
            var sum: f64 = 0.0;
            var m: i64 = 0;
            while m < m2 {
                let ir = r + ro - m;
                if ir >= 0 && ir < m1 {
                    var n: i64 = 0;
                    while n < n2 {
                        let ic = c + co - n;
                        if ic >= 0 && ic < n1 {
                            sum += fb.data[m * n2 + n] * fa.data[ir * n1 + ic];
                        }
                        n += 1;
                    }
                }
                m += 1;
            }
            out[r * cols + c] = sum;
            c += 1;
        }
        r += 1;
    }
    var s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    return Arr { ndim: 2, shape: s, data: out }, no_fault();
}

// correlate2d correlates two matrices, as SciPy correlate2d does. The
// mode is "full", "same" or "valid", as in convolve2d. The mode
// "valid" needs one matrix as large as the other in every direction.
fn correlate2d(a, b, mode: const u8[:] = "full") -> Arr, Fault {
    let tag = "correlate2d";
    if a.ndim != 2 || b.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, tag, "the matrices must hold two dimensions");
    }
    if a.data.len < 1 || b.data.len < 1 {
        return empty_arr(), fault(F_VALUE, tag, "the matrices must hold one value or more");
    }
    // The kernel reads the matrices by their shape, so the buffer of
    // each matrix must agree with its shape.
    let sa = check_stack(a.ndim, a.shape, a.data.len, 1, tag);
    if sa.kind != F_NONE { return empty_arr(), sa; }
    let sb = check_stack(b.ndim, b.shape, b.data.len, 1, tag);
    if sb.kind != F_NONE { return empty_arr(), sb; }
    if mode != "full" && mode != "same" && mode != "valid" {
        return empty_arr(), fault(F_VALUE, tag, "the mode is unknown");
    }
    var fa = a;
    var fb = b;
    var flip = false;
    if mode == "valid" {
        let small = a.shape[0] < b.shape[0] && a.shape[1] < b.shape[1];
        let large = b.shape[0] < a.shape[0] && b.shape[1] < a.shape[1];
        if small {
            // The two matrices stand apart, so the copies are built in
            // a variable of their own.
            let na = copy(b);
            let nb = copy(a);
            fa = copy(na);
            fb = copy(nb);
            flip = true;
        } else if !large {
            if !(a.shape[0] >= b.shape[0] && a.shape[1] >= b.shape[1]) {
                return empty_arr(), fault(F_SHAPE, tag, "the mode valid needs one matrix as large as the other in every direction");
            }
        }
    }
    let m1 = fa.shape[0];
    let n1 = fa.shape[1];
    let m2 = fb.shape[0];
    let n2 = fb.shape[1];
    var rows = m1 + m2 - 1;
    var cols = n1 + n2 - 1;
    var ro: i64 = 0;
    var co: i64 = 0;
    if mode == "same" {
        rows = m1;
        cols = n1;
        ro = m2 / 2;
        co = n2 / 2;
    } else if mode == "valid" {
        rows = m1 - m2 + 1;
        cols = n1 - n2 + 1;
        ro = m2 - 1;
        co = n2 - 1;
    }
    var out: f64[>..] = [];
    var i: i64 = 0;
    while i < rows * cols { out.push(0.0); i += 1; }
    var r: i64 = 0;
    while r < rows {
        var c: i64 = 0;
        while c < cols {
            var sum: f64 = 0.0;
            var m: i64 = 0;
            while m < m2 {
                let ir = r + ro - m2 + 1 + m;
                if ir >= 0 && ir < m1 {
                    var n: i64 = 0;
                    while n < n2 {
                        let ic = c + co - n2 + 1 + n;
                        if ic >= 0 && ic < n1 {
                            sum += fb.data[m * n2 + n] * fa.data[ir * n1 + ic];
                        }
                        n += 1;
                    }
                }
                m += 1;
            }
            out[r * cols + c] = sum;
            c += 1;
        }
        r += 1;
    }
    if flip {
        // The swap turns the result around, as SciPy does.
        var rev: f64[>..] = [];
        i = 0;
        while i < rows * cols { rev.push(0.0); i += 1; }
        r = 0;
        while r < rows {
            var c: i64 = 0;
            while c < cols {
                rev[(rows - 1 - r) * cols + (cols - 1 - c)] = out[r * cols + c];
                c += 1;
            }
            r += 1;
        }
        out = copy(rev);
    }
    var s: i64[..8] = [];
    s.push(rows);
    s.push(cols);
    return Arr { ndim: 2, shape: s, data: out }, no_fault();
}
