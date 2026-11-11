// The DFT matrix and the product with the orthogonal factor of the QR
// decomposition, like dft and qr_multiply of scipy.linalg. Every loop
// is bounded by the sizes of the operands.
namespace scientific;

import mathshim;
import array_decomp;
import array_linalg;
import array_carray;

// The largest side of the DFT matrix. The matrix holds side squared
// complex values.
const DFT_MAX_SIDE: i64 = 4096;

// dft gives the n by n matrix of the discrete Fourier transform. The
// entry (j, k) is exp(-2 pi i j k / n). The scale is "" for none,
// "sqrtn" to divide by the square root of n, or "n" to divide by n.
fn dft(n: i64, scale: const u8[:] = "") -> CArr, Fault {
    if n < 1 {
        return carr_empty(), fault(F_VALUE, "dft", "the size is below one");
    }
    if n > DFT_MAX_SIDE {
        return carr_empty(), fault(F_LIMIT, "dft", "the matrix holds more values than the limit");
    }
    var factor: f64 = 1.0;
    if scale == "sqrtn" {
        factor = 1.0 / mathshim::root(n as f64);
    } else if scale == "n" {
        factor = 1.0 / (n as f64);
    } else if scale != "" {
        return carr_empty(), fault(F_VALUE, "dft", "the scale is not empty, sqrtn or n");
    }
    var d: f64[>..] = [];
    var j: i64 = 0;
    while j < n {
        var k: i64 = 0;
        while k < n {
            // The product reduced by n keeps the angle exact.
            let t = ((j * k) % n) as f64;
            let ang = -2.0 * 3.141592653589793 * t / (n as f64);
            d.push(factor * mathshim::cosine(ang));
            d.push(factor * mathshim::sine(ang));
            k += 1;
        }
        j += 1;
    }
    return carr_make([n, n], copy(d)), no_fault();
}

// qr_multiply multiplies c by the orthogonal factor of the QR
// decomposition of a, and gives the product and the triangle. The
// factor holds min(m, n) columns. Mode "left" gives Q times c, and c
// holds min(m, n) rows. Mode "right" gives c times Q, and c holds m
// columns.
fn qr_multiply(a, c, mode: const u8[:] = "right") -> Arr, Arr, Fault {
    if mode != "left" && mode != "right" {
        return empty_arr(), empty_arr(), fault(F_VALUE, "qr_multiply", "the mode is not left or right");
    }
    if c.ndim != 2 {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_multiply", "the operand c holds a count of dimensions that is not two");
    }
    let q, r, f = qr(a);
    if f.kind != F_NONE { return empty_arr(), empty_arr(), f; }
    if mode == "left" {
        if c.shape[0] != q.shape[1] {
            return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_multiply", "c does not hold min(m, n) rows");
        }
        let p, fp = matmul(q, c);
        if fp.kind != F_NONE { return empty_arr(), empty_arr(), fp; }
        return p, r, no_fault();
    }
    if c.shape[1] != q.shape[0] {
        return empty_arr(), empty_arr(), fault(F_SHAPE, "qr_multiply", "c does not hold m columns");
    }
    let p, fp = matmul(c, q);
    if fp.kind != F_NONE { return empty_arr(), empty_arr(), fp; }
    return p, r, no_fault();
}
