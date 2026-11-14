// The inspection and the conversion of an array. The program copies an
// array, reads its shape and prints the values.
import std;
import scientific from "..";

fn main() {
    let a, fa = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert(fa.kind == scientific::F_NONE);
    let s, f = scientific::shape(a);
    assert(f.kind == scientific::F_NONE);
    print("shape ");
    var i: i64 = 0;
    while i < s.data.len { print(s.data[i], " "); i += 1; }
    print("\n");
    let c, fc = scientific::asarray(a);
    assert(fc.kind == scientific::F_NONE);
    print("copy ");
    i = 0;
    while i < c.data.len { print(c.data[i], " "); i += 1; }
    print("\n");
    print("isfortran of a matrix is ", scientific::isfortran(a), "\n");
    let flat, ff = scientific::flat(a);
    assert(ff.kind == scientific::F_NONE);
    print("flat holds ", flat.data.len, " elements\n");
    let bad, fbad = scientific::astype(a, "int32");
    assert(fbad.kind == scientific::F_VALUE);
    print("astype refuses a type that the package does not hold\n");
}
