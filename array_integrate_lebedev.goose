// The rules of Lebedev for the surface of the sphere.
//
// The rule integrates a function over the surface of the unit
// sphere. The rule of the order n holds all the spherical harmonics
// up to the order n. The weights sum to the area of the unit sphere.
// The table of the points comes from SciPy 1.18.1 and lives in the
// file that the script test/gen_lebedev.py writes. The construction
// follows the paper of Lebedev and Laikov, 1999.
//
namespace scientific;

import mathshim;

// The orders and the counts of the points of the rules.
const LEB_ORDERS: i64[32] = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131];
const LEB_DEGREES: i64[32] = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810];

// lebedev_value gives the value of one symbol of the pattern of the
// grid. The symbol zero is zero and the others are the parameters and
// their opposites.
fn lebedev_value(sym: i64, a: f64, b: f64, c: f64) -> f64 {
    if sym == 1 { return a; }
    if sym == 2 { return 0.0 - a; }
    if sym == 3 { return b; }
    if sym == 4 { return 0.0 - b; }
    if sym == 5 { return c; }
    if sym == 6 { return 0.0 - c; }
    return 0.0;
}

// lebedev_grid gives the points of one block of the grid. The kind
// picks the pattern of the points. The kinds 1, 2 and 3 take the
// value of a from the pattern itself. The kinds 4 and 5 take b from
// a, and the kind 6 takes c from a and b. Every point holds the three
// coordinates and the weight, one value after the other. The result
// goes into out.
fn lebedev_grid(kind: i64, a0: f64, b0: f64, v: f64, out: f64[>..]&) {
    var a = a0;
    if kind == 1 { a = 1.0; }
    if kind == 2 { a = mathshim::root(0.5); }
    if kind == 3 { a = mathshim::root(1.0 / 3.0); }
    var b = b0;
    var c: f64 = 0.0;
    if kind == 4 { b = mathshim::root(1.0 - 2.0 * a * a); }
    if kind == 5 { b = mathshim::root(1.0 - a * a); }
    if kind == 6 { c = mathshim::root(1.0 - a * a - b * b); }
    // The table holds the end of every block in the place of the
    // kind, so the block of the kind starts behind the one before.
    let lo = LEB_KOFF[kind - 1];
    let hi = LEB_KOFF[kind];
    let w = 4.0 * 3.141592653589793 * v;
    var i: i64 = lo;
    while i + 2 < hi {
        out.push(lebedev_value(LEB_KP[i], a, b, c));
        out.push(lebedev_value(LEB_KP[i + 1], a, b, c));
        out.push(lebedev_value(LEB_KP[i + 2], a, b, c));
        out.push(w);
        i += 3;
    }
}

// lebedev_rule gives the points and the weights of the rule of the
// order n over the surface of the unit sphere. The points come as a
// matrix with one row for x, one row for y and one row for z. The
// weights sum to the area of the unit sphere. The orders are 3, 5, 7,
// 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59,
// 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125 and 131.
fn lebedev_rule(n: i64) -> Arr, Arr, Fault {
    var di: i64 = -1;
    var i: i64 = 0;
    while i < 32 {
        if LEB_ORDERS[i] == n { di = i; break; }
        i += 1;
    }
    if di < 0 {
        return empty_arr(), empty_arr(), fault(F_VALUE, "lebedev_rule", "the order holds no rule");
    }
    var flat: f64[>..] = [];
    var c: i64 = 0;
    while c < LEB_COUNT[di] {
        let at = LEB_START[di] + c;
        lebedev_grid(LEB_CALL_K[at], LEB_CALL_A[at], LEB_CALL_B[at], LEB_CALL_V[at], flat);
        c += 1;
    }
    let m = LEB_POINTS[di];
    if flat.len != 4 * m {
        return empty_arr(), empty_arr(), fault(F_LIMIT, "lebedev_rule", "the table of the rule holds the wrong count of values");
    }
    var xs: f64[>..] = [];
    var ws: f64[>..] = [];
    var p: i64 = 0;
    while p < m {
        xs.push(flat[4 * p]);
        p += 1;
    }
    p = 0;
    while p < m {
        xs.push(flat[4 * p + 1]);
        p += 1;
    }
    p = 0;
    while p < m {
        xs.push(flat[4 * p + 2]);
        p += 1;
    }
    p = 0;
    while p < m {
        ws.push(flat[4 * p + 3]);
        p += 1;
    }
    return mat2(copy(xs), 3, m), line_arr(copy(ws)), no_fault();
}
