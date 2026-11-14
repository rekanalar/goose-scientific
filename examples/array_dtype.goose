// The type layer: which type holds the result of an operation, and
// whether a cast keeps the values. Run it with: goose run examples/array_dtype.goose
import std;
import scientific from "..";

fn main() {
    let p, fp = scientific::promote_types("int8", "uint8");
    print("int8 with uint8 gives ", p, "\n");
    let q, fq = scientific::promote_types("uint64", "int64");
    print("uint64 with int64 gives ", q, "\n");
    let s, fs = scientific::can_cast("int32", "float32", "safe");
    print("int32 to float32 is safe: ", s, "\n");
    let k, fk = scientific::can_cast("int32", "float32", "same_kind");
    print("int32 to float32 keeps the kind: ", k, "\n");
    let c, fc = scientific::common_type("int16", "float32");
    print("the common type of int16 and float32 is ", c, "\n");
    let r, fr = scientific::isdtype("float32", "real floating");
    print("float32 is a real floating type: ", r, "\n");
    print("the smallest type of 300 is ", scientific::min_scalar_type(300), "\n");
    print("the smallest type of 1e10 is ", scientific::min_scalar_type(1.0e10), "\n");
    assert(fp.kind == scientific::F_NONE && fq.kind == scientific::F_NONE && fs.kind == scientific::F_NONE);
    assert(fk.kind == scientific::F_NONE && fc.kind == scientific::F_NONE && fr.kind == scientific::F_NONE);
    return 0;
}
