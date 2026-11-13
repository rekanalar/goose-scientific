// The sort of the complex values.
//
// The sort orders the complex values by the real part, and by the
// imaginary part when the real parts are equal. Two equal values keep
// their order. A value that is not a number comes last, as in NumPy.
namespace scientific;

// cplx_greater tells whether the value at the place a comes after the
// value at the place b. The real part comes first, and the imaginary
// part comes when the real parts are equal.
fn cplx_greater(c: CArr&, a: i64, b: i64) -> bool {
    let ra = c.data[a * 2];
    let rb = c.data[b * 2];
    if ra != ra { return false; }
    if rb != rb { return true; }
    if ra != rb { return ra > rb; }
    return c.data[a * 2 + 1] > c.data[b * 2 + 1];
}

// sort_complex sorts the complex values of an array.
fn sort_complex(c: CArr&) -> CArr, Fault {
    // The array must agree with its own shape before any value is read.
    let fc = carr_check(c, "sort_complex");
    if fc.kind != F_NONE { return carr_empty(), fc; }
    if c.data.len % 2 != 0 {
        return carr_empty(), fault(F_SHAPE, "sort_complex", "the array does not hold pairs of values");
    }
    let n = c.data.len / 2;
    // An insertion sort keeps the order of two equal values.
    var pos: i64[>..] = [];
    var i: i64 = 0;
    while i < n { pos.push(i); i += 1; }
    i = 1;
    while i < n {
        let key = pos[i];
        var j = i - 1;
        while j >= 0 {
            let other = pos[j];
            if cplx_greater(c, other, key) {
                pos[j + 1] = other;
                j -= 1;
            } else { break; }
        }
        pos[j + 1] = key;
        i += 1;
    }
    var d: f64[>..] = [];
    i = 0;
    while i < n {
        let at = pos[i];
        d.push(c.data[at * 2]);
        d.push(c.data[at * 2 + 1]);
        i += 1;
    }
    return carr_make(c.shape, copy(d)), no_fault();
}
