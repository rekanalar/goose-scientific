// The text of numbers: the shortest text of a float, the exact digits,
// whole numbers in a base, and type codes.
// Run it with: goose run examples/array_text.goose
import std;
import scientific from "..";

fn main() {
    var o = scientific::float_format();
    let a, fa = scientific::format_float_positional(0.1, o);
    print("the shortest text of 0.1: ", a, "\n");
    let b, fb = scientific::format_float_scientific(123456.789, o);
    print("the same number in the scientific form: ", b, "\n");
    o.unique = false;
    o.precision = 20;
    let c, fc = scientific::format_float_positional(0.1, o);
    print("the exact digits of 0.1: ", c, "\n");
    let d, fd = scientific::base_repr(255, 16, 0);
    print("255 in base 16: ", d, "\n");
    let e, fe = scientific::binary_repr(-5, 8);
    print("-5 in 8 bits: ", e, "\n");
    let g, fg = scientific::typename("d");
    print("the type d is ", g, "\n");
    let h, fh = scientific::mintypecode("fF");
    print("the smallest code of fF is ", h as i64, "\n");
    assert(fa.kind == scientific::F_NONE && fb.kind == scientific::F_NONE);
    assert(fc.kind == scientific::F_NONE && fd.kind == scientific::F_NONE);
    assert(fe.kind == scientific::F_NONE && fg.kind == scientific::F_NONE);
    assert(fh.kind == scientific::F_NONE && h == 'F');
    return 0;
}
