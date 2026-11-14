// The tests of the group array_dtype. Run them with SCIENTIFIC_SUITES=array_dtype.
//
// The type layer matches the pins of NumPy 2.5.4 on the whole grid of
// the 14 number types: promote_types, result_type, can_cast,
// common_type, issubdtype, isdtype and min_scalar_type.
namespace array_dtype_tests;

import scientific from "../..";
import array_dtype_pins;

fn name_at(i: i64) -> const u8[:] {
    if i == 0 { return "bool"; }
    if i == 1 { return "int8"; }
    if i == 2 { return "int16"; }
    if i == 3 { return "int32"; }
    if i == 4 { return "int64"; }
    if i == 5 { return "uint8"; }
    if i == 6 { return "uint16"; }
    if i == 7 { return "uint32"; }
    if i == 8 { return "uint64"; }
    if i == 9 { return "float16"; }
    if i == 10 { return "float32"; }
    if i == 11 { return "float64"; }
    if i == 12 { return "complex64"; }
    return "complex128";
}

fn class_at(i: i64) -> const u8[:] {
    if i == 0 { return "generic"; }
    if i == 1 { return "number"; }
    if i == 2 { return "integer"; }
    if i == 3 { return "signedinteger"; }
    if i == 4 { return "unsignedinteger"; }
    if i == 5 { return "floating"; }
    if i == 6 { return "complexfloating"; }
    return "inexact";
}

fn kind_at(i: i64) -> const u8[:] {
    if i == 0 { return "bool"; }
    if i == 1 { return "signed integer"; }
    if i == 2 { return "unsigned integer"; }
    if i == 3 { return "integral"; }
    if i == 4 { return "real floating"; }
    if i == 5 { return "complex floating"; }
    return "numeric";
}

fn test_promote_grid() {
    var a: i64 = 0;
    while a < 14 {
        var b: i64 = 0;
        while b < 14 {
            let want = name_at(array_dtype_pins::PROMOTE[a * 14 + b] as i64);
            let p, fp = scientific::promote_types(name_at(a), name_at(b));
            assert(fp.kind == scientific::F_NONE && p == want);
            let r, fr = scientific::result_type(name_at(a), name_at(b));
            assert(fr.kind == scientific::F_NONE && r == name_at(array_dtype_pins::RESULT[a * 14 + b] as i64));
            b += 1;
        }
        a += 1;
    }
}

fn test_can_cast_grid() {
    var a: i64 = 0;
    while a < 14 {
        var b: i64 = 0;
        while b < 14 {
            let at = a * 14 + b;
            let n, f1 = scientific::can_cast(name_at(a), name_at(b), "no");
            assert(f1.kind == scientific::F_NONE && n == (array_dtype_pins::CAST_NO[at] == 1.0));
            let e, f2 = scientific::can_cast(name_at(a), name_at(b), "equiv");
            assert(e == (array_dtype_pins::CAST_EQUIV[at] == 1.0));
            let s, f3 = scientific::can_cast(name_at(a), name_at(b), "safe");
            assert(s == (array_dtype_pins::CAST_SAFE[at] == 1.0));
            let k, f4 = scientific::can_cast(name_at(a), name_at(b), "same_kind");
            assert(k == (array_dtype_pins::CAST_SAME_KIND[at] == 1.0));
            let u, f5 = scientific::can_cast(name_at(a), name_at(b), "unsafe");
            assert(u == (array_dtype_pins::CAST_UNSAFE[at] == 1.0));
            b += 1;
        }
        a += 1;
    }
}

fn test_common_type_grid() {
    var a: i64 = 0;
    while a < 14 {
        var b: i64 = 0;
        while b < 14 {
            let want = array_dtype_pins::COMMON[a * 14 + b] as i64;
            let c, f = scientific::common_type(name_at(a), name_at(b));
            if want < 0 {
                assert(f.kind == scientific::F_VALUE);
            } else {
                assert(f.kind == scientific::F_NONE && c == name_at(want));
            }
            b += 1;
        }
        a += 1;
    }
}

fn test_issubdtype_and_isdtype_grid() {
    var a: i64 = 0;
    while a < 14 {
        var c: i64 = 0;
        while c < 8 {
            let r, f = scientific::issubdtype(name_at(a), class_at(c));
            assert(f.kind == scientific::F_NONE && r == (array_dtype_pins::SUB[a * 8 + c] == 1.0));
            c += 1;
        }
        var k: i64 = 0;
        while k < 7 {
            let r, f = scientific::isdtype(name_at(a), kind_at(k));
            assert(f.kind == scientific::F_NONE && r == (array_dtype_pins::KIND[a * 7 + k] == 1.0));
            k += 1;
        }
        // A type is a sub type of itself.
        let same, fs = scientific::issubdtype(name_at(a), name_at(a));
        assert(same);
        a += 1;
    }
}

fn test_min_scalar_type() {
    var i: i64 = 0;
    while i < array_dtype_pins::MIN_INT_IN.len {
        let v = array_dtype_pins::MIN_INT_IN[i] as i64;
        assert(scientific::min_scalar_type(v) == name_at(array_dtype_pins::MIN_INT_OUT[i] as i64));
        i += 1;
    }
    i = 0;
    while i < array_dtype_pins::MIN_FLOAT_IN.len {
        assert(scientific::min_scalar_type(array_dtype_pins::MIN_FLOAT_IN[i]) == name_at(array_dtype_pins::MIN_FLOAT_OUT[i] as i64));
        i += 1;
    }
    // NaN and the infinities give float16, as in NumPy.
    let zero = 0.0;
    assert(scientific::min_scalar_type(1.0 / zero) == "float16");
    assert(scientific::min_scalar_type(-1.0 / zero) == "float16");
    assert(scientific::min_scalar_type(zero / zero) == "float16");
}

fn test_dtype_names_and_faults() {
    // The names of NumPy for the same types count.
    let p, f = scientific::promote_types("half", "int16");
    assert(f.kind == scientific::F_NONE && p == "float32");
    let q, fq = scientific::promote_types("double", "csingle");
    assert(fq.kind == scientific::F_NONE && q == "complex128");
    let b1, fb1 = scientific::issubdtype("bool_", "bool");
    assert(b1);
    // A name that is no type is a fault and gives no value.
    let r1, f1 = scientific::promote_types("int8", "no such");
    assert(f1.kind == scientific::F_VALUE && r1 == "");
    let r2, f2 = scientific::result_type("", "int8");
    assert(f2.kind == scientific::F_VALUE && r2 == "");
    let r3, f3 = scientific::common_type("int8", "float128");
    assert(f3.kind == scientific::F_VALUE && r3 == "");
    let c1, fc1 = scientific::can_cast("int8", "int16", "bogus");
    assert(fc1.kind == scientific::F_VALUE && !c1);
    let c2, fc2 = scientific::can_cast("nope", "int16", "safe");
    assert(fc2.kind == scientific::F_VALUE && !c2);
    let s1, fs1 = scientific::issubdtype("int8", "no class");
    assert(fs1.kind == scientific::F_VALUE && !s1);
    let d1, fd1 = scientific::isdtype("int8", "nosuch");
    assert(fd1.kind == scientific::F_VALUE && !d1);
    let d2, fd2 = scientific::isdtype("float32", "float64");
    assert(fd2.kind == scientific::F_NONE && !d2);
    let d3, fd3 = scientific::isdtype("float32", "float32");
    assert(d3);
}
