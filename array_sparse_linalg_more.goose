// The sparse linear algebra: the triangular solve, the triangular check,
// the bandwidth and the Laplacian of a grid.
//
// The triangular solve.
//
// spsolve_triangular solves a system whose matrix is triangular. The
// matrix holds one triangle of the values. The lower triangle stands
// below the diagonal, the upper triangle stands above it.
//
// The triangular check and the bandwidth.
//
// is_sptriangular tells whether a matrix holds only the lower triangle
// and whether it holds only the upper triangle. spbandwidth gives the
// count of the rows that separate the values from the diagonal.
//
// The Laplacian of a grid.
//
// laplacian_nd gives the Laplacian of a grid of any count of dimensions.
// The boundary of the grid takes three shapes. The boundary of Neumann
// keeps the values at the edge. The boundary of Dirichlet sets the
// values at the edge to zero. The boundary of Periodic joins the two
// ends of every dimension. laplacian_nd_values gives the values of the
// matrix of the call. The values come from the closed form of the
// Laplacian of a grid.
namespace scientific;

import mathshim;

// The largest count of the nodes of the grid of the Laplacian.
const MAX_LAPLACE_NODES: i64 = 262144;

// spsolve_triangular solves a system whose matrix is triangular. The
// matrix holds the lower triangle when the flag is true. The matrix
// holds the upper triangle when the flag is false. The diagonal holds
// one for every row when the second flag is true. A zero on the
// diagonal gives a fault of the kind F_VALUE.
fn spsolve_triangular(s, b: f64[:], lower: bool = true, unit_diagonal: bool = false) -> Arr, Fault {
    if !csr_valid(s) {
        return empty_arr(), fault(F_SHAPE, "spsolve_triangular", "the buffer of the matrix does not agree with its sides");
    }
    if s.rows != s.cols || s.rows < 1 {
        return empty_arr(), fault(F_SHAPE, "spsolve_triangular", "the matrix is not square or holds no row");
    }
    if b.len != s.rows {
        return empty_arr(), fault(F_SHAPE, "spsolve_triangular", "the right side does not agree with the matrix");
    }
    let sorted, f0 = csr_sorted(s);
    if f0.kind != F_NONE { return empty_arr(), f0; }
    var x: f64[>..] = [];
    var i: i64 = 0;
    while i < s.rows { x.push(0.0); i += 1; }
    // A lower matrix walks its rows from the first to the last, and a
    // term below the diagonal holds a value that is already known. An
    // upper matrix walks its rows from the last to the first, and a term
    // above the diagonal holds a value that is already known.
    i = 0;
    while i < s.rows {
        var row = i;
        if !lower { row = s.rows - 1 - i; }
        var sum: f64 = 0.0;
        var diag: f64 = 0.0;
        var has_diag = false;
        var k = csr_ptr(sorted, row);
        while k < csr_ptr(sorted, row + 1) {
            let col = csr_col(sorted, k);
            let val = csr_val(sorted, k);
            if col == row {
                diag = val;
                has_diag = true;
            } else if (lower && col < row) || (!lower && col > row) {
                sum += val * x[col];
            }
            k += 1;
        }
        if unit_diagonal {
            diag = 1.0;
        } else if !has_diag || diag == 0.0 {
            return empty_arr(), fault(F_VALUE, "spsolve_triangular", "the diagonal of the matrix holds a zero");
        }
        x[row] = (b[row] - sum) / diag;
        i += 1;
    }
    return line_arr(copy(x)), no_fault();
}

// is_sptriangular tells whether a matrix holds only the lower triangle
// and whether it holds only the upper triangle. The first flag is true
// when no value stands above the diagonal.
fn is_sptriangular(s) -> bool, bool {
    if !csr_valid(s) { return false, false; }
    if s.rows != s.cols { return false, false; }
    var lower = true;
    var upper = true;
    var i: i64 = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        while k < csr_ptr(s, i + 1) {
            let col = csr_col(s, k);
            if col > i { lower = false; }
            if col < i { upper = false; }
            k += 1;
        }
        i += 1;
    }
    return lower, upper;
}

// spbandwidth gives the count of the rows that separate the values of a
// matrix from its diagonal. The first count holds the values below the
// diagonal, the second count holds the values above the diagonal. A
// matrix without values gives zero for both counts.
fn spbandwidth(s) -> i64, i64 {
    if !csr_valid(s) { return 0, 0; }
    var below: i64 = 0;
    var above: i64 = 0;
    var i: i64 = 0;
    while i < s.rows {
        var k = csr_ptr(s, i);
        while k < csr_ptr(s, i + 1) {
            let d = csr_col(s, k) - i;
            if d < 0 {
                if -d > below { below = -d; }
            } else {
                if d > above { above = d; }
            }
            k += 1;
        }
        i += 1;
    }
    return below, above;
}

// laplace_check checks the shape and the boundary of the Laplacian.
fn laplace_check(shape: i64[:], boundary: const u8[:], tag: const u8[:]) -> Fault {
    if shape.len < 1 || shape.len > MAX_DIMS {
        return fault(F_SHAPE, tag, "the shape holds more dimensions than the limit");
    }
    if boundary != "neumann" && boundary != "dirichlet" && boundary != "periodic" {
        return fault(F_VALUE, tag, "the boundary holds an unknown name");
    }
    var nodes: i64 = 1;
    var d: i64 = 0;
    while d < shape.len {
        if shape[d] < 1 {
            return fault(F_VALUE, tag, "a side holds a length below one");
        }
        nodes *= shape[d];
        d += 1;
    }
    if nodes > MAX_LAPLACE_NODES {
        return fault(F_LIMIT, tag, "the count of the nodes of the grid stands above the limit");
    }
    return no_fault();
}

// laplace_stride gives the count of the nodes that one step of a
// dimension covers.
fn laplace_stride(shape: i64[:], d: i64) -> i64 {
    var s: i64 = 1;
    var e = d + 1;
    while e < shape.len { s *= shape[e]; e += 1; }
    return s;
}

// laplace_diag gives the value of the diagonal of one dimension at the
// place c of a dimension of the length n.
fn laplace_diag(c: i64, n: i64, boundary: const u8[:]) -> f64 {
    if boundary == "dirichlet" { return -2.0; }
    if boundary == "periodic" {
        if n == 1 { return -1.0; }
        return -2.0;
    }
    var v = -2.0;
    if c == 0 { v += 1.0; }
    if c == n - 1 { v += 1.0; }
    return v;
}

// laplacian_nd gives the Laplacian of a grid of any count of
// dimensions. The matrix holds one row for every node of the grid. The
// diagonal holds the sum of the values of the dimensions. The value of
// every neighbour stands at one.
fn laplacian_nd(shape: i64[:], boundary: const u8[:] = "neumann") -> Csr, Fault {
    let f = laplace_check(shape, boundary, "laplacian_nd");
    if f.kind != F_NONE { return csr_empty(), f; }
    let ndim = shape.len;
    var nodes: i64 = 1;
    var d: i64 = 0;
    while d < ndim { nodes *= shape[d]; d += 1; }
    if nodes * (1 + 2 * ndim) > MAX_SPARSE_NNZ {
        return csr_empty(), fault(F_LIMIT, "laplacian_nd", "the count of the stored values stands above the limit");
    }
    // The strides of the dimensions and the kind of the boundary do not
    // change from node to node, so they are computed once.
    var strides: i64[>..] = [];
    d = 0;
    while d < ndim {
        strides.push(laplace_stride(shape, d));
        d += 1;
    }
    let periodic = boundary == "periodic";
    var ri: f64[>..] = [];
    var ci: f64[>..] = [];
    var vv: f64[>..] = [];
    var node: i64 = 0;
    while node < nodes {
        // The place of the node in every dimension.
        var c: i64[>..] = [];
        var rest = node;
        d = ndim - 1;
        while d >= 0 {
            c.push(rest % shape[d]);
            rest = rest / shape[d];
            d -= 1;
        }
        var diag: f64 = 0.0;
        d = 0;
        while d < ndim {
            diag += laplace_diag(c[ndim - 1 - d], shape[d], boundary);
            d += 1;
        }
        ri.push(node as f64);
        ci.push(node as f64);
        vv.push(diag);
        d = 0;
        while d < ndim {
            let n = shape[d];
            let at = c[ndim - 1 - d];
            let step = strides[d];
            var s: i64 = -1;
            while s <= 1 {
                if s != 0 {
                    var nc = at + s;
                    if periodic {
                        if nc < 0 { nc = nc + n; }
                        if nc >= n { nc = nc - n; }
                    } else if nc < 0 || nc >= n {
                        nc = at;
                    }
                    if nc != at {
                        ri.push(node as f64);
                        ci.push((node + (nc - at) * step) as f64);
                        vv.push(1.0);
                    }
                }
                s += 2;
            }
            d += 1;
        }
        node += 1;
    }
    let s, f1 = csr_from_triplets(nodes, nodes, ri, ci, vv);
    if f1.kind != F_NONE { return csr_empty(), f1; }
    return s, no_fault();
}

// laplace_values_dim gives the values of one dimension of the grid. The
// values come from the closed form of the Laplacian of a grid.
fn laplace_values_dim(n: i64, boundary: const u8[:]) -> f64[>..] {
    var out: f64[>..] = [];
    if boundary == "dirichlet" {
        var j: i64 = 0;
        while j < n {
            let a = mathshim::sine(3.141592653589793 * (j + 1) as f64 / (2.0 * (n + 1) as f64));
            out.push(-4.0 * a * a);
            j += 1;
        }
        return out;
    }
    if n == 1 {
        out.push(-1.0);
        return out;
    }
    if boundary == "neumann" {
        var j: i64 = 0;
        while j < n {
            let a = mathshim::sine(3.141592653589793 * j as f64 / (2.0 * n as f64));
            out.push(-4.0 * a * a);
            j += 1;
        }
        return out;
    }
    var j: i64 = 0;
    while j < n {
        let k = (j + 1) / 2;
        let a = mathshim::sine(3.141592653589793 * k as f64 / n as f64);
        out.push(-4.0 * a * a);
        j += 1;
    }
    return out;
}

// laplacian_nd_values gives the values of the matrix of laplacian_nd.
// The values come in the order from the lowest to the highest.
fn laplacian_nd_values(shape: i64[:], boundary: const u8[:] = "neumann") -> Arr, Fault {
    let f = laplace_check(shape, boundary, "laplacian_nd_values");
    if f.kind != F_NONE { return empty_arr(), f; }
    let ndim = shape.len;
    // The values of every dimension follow each other in one buffer.
    var ev: f64[>..] = [];
    var starts: i64[>..] = [];
    var d: i64 = 0;
    while d < ndim {
        starts.push(ev.len);
        let part = laplace_values_dim(shape[d], boundary);
        var i: i64 = 0;
        while i < part.len { ev.push(part[i]); i += 1; }
        d += 1;
    }
    var nodes: i64 = 1;
    d = 0;
    while d < ndim { nodes *= shape[d]; d += 1; }
    var out: f64[>..] = [];
    var node: i64 = 0;
    while node < nodes {
        var rest = node;
        var sum: f64 = 0.0;
        d = ndim - 1;
        while d >= 0 {
            let k = rest % shape[d];
            rest = rest / shape[d];
            sum += ev[starts[d] + k];
            d -= 1;
        }
        out.push(sum);
        node += 1;
    }
    let ord = order_places(out, 1, out.len);
    var sorted: f64[>..] = [];
    var i: i64 = 0;
    while i < ord.len { sorted.push(out[ord[i]]); i += 1; }
    return line_arr(copy(sorted)), no_fault();
}
