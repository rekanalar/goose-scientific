// The layout and the memory of the arrays.
//
// The example packs the bits of an array, copies the values into an
// array, assembles a grid of the blocks, rolls an axis to a place and
// sorts the complex values.
import std;
import scientific from "..";

fn main() {
    // The pack of the bits puts the first bit in the high place of the
    // byte. The array holds the values 1, 0, 1, 1, 0, 0, 1, 0 and 1.
    let bits, f1 = scientific::from_values([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
    assert(f1.kind == scientific::F_NONE);
    let packed, f2 = scientific::packbits(bits);
    assert(f2.kind == scientific::F_NONE);
    print("packed:");
    var i: i64 = 0;
    while i < packed.data.len { print(" {} ", packed.data[i]); i += 1; }
    print("\n");

    // The copy writes the values into the array of the caller. The mask
    // holds the places that the copy fills.
    var dst, f3 = scientific::zeros([2, 3]);
    assert(f3.kind == scientific::F_NONE);
    let src, f4 = scientific::from_values([1.0, 2.0, 3.0]);
    assert(f4.kind == scientific::F_NONE);
    let mask, f5 = scientific::from_values([1.0, 0.0, 1.0]);
    assert(f5.kind == scientific::F_NONE);
    let f6 = scientific::copyto_where(dst, src, mask);
    assert(f6.kind == scientific::F_NONE);
    print("copied:");
    var r: i64 = 0;
    while r < 2 {
        var c: i64 = 0;
        while c < 3 { print(" {} ", dst.data[r * 3 + c]); c += 1; }
        r += 1;
    }
    print("\n");

    // The grid of the blocks joins the blocks of a row along the last
    // axis, and the rows along the axis before it.
    var grid, f7 = scientific::block_new(2, 2);
    assert(f7.kind == scientific::F_NONE);
    let a, f8 = scientific::from_values([1.0, 2.0]);
    assert(f8.kind == scientific::F_NONE);
    let b, f9 = scientific::from_values([3.0, 4.0]);
    assert(f9.kind == scientific::F_NONE);
    let c, f10 = scientific::from_values([5.0, 6.0]);
    assert(f10.kind == scientific::F_NONE);
    let d, f11 = scientific::from_values([7.0, 8.0]);
    assert(f11.kind == scientific::F_NONE);
    let f12 = scientific::block_put(grid, 0, 0, a);
    assert(f12.kind == scientific::F_NONE);
    let f13 = scientific::block_put(grid, 0, 1, b);
    assert(f13.kind == scientific::F_NONE);
    let f14 = scientific::block_put(grid, 1, 0, c);
    assert(f14.kind == scientific::F_NONE);
    let f15 = scientific::block_put(grid, 1, 1, d);
    assert(f15.kind == scientific::F_NONE);
    let whole, f16 = scientific::block_build(grid);
    assert(f16.kind == scientific::F_NONE);
    print("grid:");
    r = 0;
    while r < whole.shape[0] {
        var c2: i64 = 0;
        while c2 < whole.shape[1] { print(" {} ", whole.data[r * whole.shape[1] + c2]); c2 += 1; }
        r += 1;
    }
    print("\n");

    // The roll moves an axis to a place. The array holds two rows of
    // three values.
    let m, f17 = scientific::zeros([2, 3]);
    assert(f17.kind == scientific::F_NONE);
    let rolled, f18 = scientific::rollaxis(m, 1, 0);
    assert(f18.kind == scientific::F_NONE);
    print("rolled shape: {} {}\n", rolled.shape[0], rolled.shape[1]);

    // The sort of the complex values orders by the real part, and by the
    // imaginary part when the real parts are equal.
    let re, f19 = scientific::from_values([3.0, 1.0, 1.0]);
    assert(f19.kind == scientific::F_NONE);
    let im, f20 = scientific::from_values([1.0, 5.0, 2.0]);
    assert(f20.kind == scientific::F_NONE);
    let z, f21 = scientific::complex_from_values(re.data, im.data);
    assert(f21.kind == scientific::F_NONE);
    let sorted, f22 = scientific::sort_complex(z);
    assert(f22.kind == scientific::F_NONE);
    print("sorted:");
    var k: i64 = 0;
    while k < 3 { print(" ({}, {})", sorted.data[k * 2], sorted.data[k * 2 + 1]); k += 1; }
    print("\n");
}
