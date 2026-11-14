// The type layer of NumPy: issubdtype, isdtype, can_cast, promote_types,
// result_type, common_type and min_scalar_type. The package holds one
// array type, so the calls work on the names of the 14 types that NumPy
// holds for numbers: bool, int8 to int64, uint8 to uint64, float16 to
// float64, complex64 and complex128. A call takes a name and gives a
// name or a flag. A name that the package does not know is a fault.
//
// The rules are the rules of NumPy 2.5. Two small tables hold the kind
// and the precision of each type. The rules work on those two numbers.
namespace scientific;

// The 14 types stand in this order: bool, int8, int16, int32, int64,
// uint8, uint16, uint32, uint64, float16, float32, float64, complex64,
// complex128. The kind of a type is 0 for bool, 1 for a signed whole
// number, 2 for an unsigned whole number, 3 for a float and 4 for a
// complex number. The precision of a whole number is its count of
// bits. The precision of a float is its count of mantissa bits. The
// precision of a complex number is the precision of its parts.
const DT_COUNT: i64 = 14;
const DT_KIND: i64[..14] = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4];
const DT_PREC: i64[..14] = [1, 8, 16, 32, 64, 8, 16, 32, 64, 11, 24, 53, 24, 53];

// dt_name gives the name of the type at an index. An index outside the
// table gives an empty text.
fn dt_name(i: i64) -> const u8[:] {
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
    if i == 13 { return "complex128"; }
    return "";
}

// dt_index gives the index of a type by its name. The names of NumPy
// for the same types count too: bool_, half, single, double, float,
// csingle, cdouble, complex, int and uint. A name that the package
// does not know gives -1.
fn dt_index(name: const u8[:]) -> i64 {
    var i: i64 = 0;
    while i < DT_COUNT {
        if name == dt_name(i) { return i; }
        i += 1;
    }
    if name == "bool_" { return 0; }
    if name == "int" { return 4; }
    if name == "uint" { return 8; }
    if name == "half" { return 9; }
    if name == "single" { return 10; }
    if name == "double" || name == "float" { return 11; }
    if name == "csingle" { return 12; }
    if name == "cdouble" || name == "complex" { return 13; }
    return -1;
}

fn dt_bad(tag: const u8[:]) -> Fault {
    return fault(F_VALUE, tag, "the name is not a type of numbers");
}

// dt_signed gives the signed whole type that holds a count of bits.
fn dt_signed(bits: i64) -> i64 {
    if bits <= 8 { return 1; }
    if bits <= 16 { return 2; }
    if bits <= 32 { return 3; }
    return 4;
}

// dt_float_of gives the smallest float that holds every value of a
// whole type. The index of a float rises with its width, so the larger
// of two floats is the one with the larger index.
fn dt_float_of(prec: i64) -> i64 {
    if prec <= 8 { return 9; }
    if prec <= 16 { return 10; }
    return 11;
}

// dt_complex_of gives the smallest complex type that holds every value
// of a whole type.
fn dt_complex_of(prec: i64) -> i64 {
    if prec <= 16 { return 12; }
    return 13;
}

// dt_promote gives the index of the type that holds both types.
fn dt_promote(a: i64, b: i64) -> i64 {
    var x = a;
    var y = b;
    let kx = DT_KIND[x];
    let ky = DT_KIND[y];
    if kx == 0 { return y; }
    if ky == 0 { return x; }
    if kx == ky && kx <= 2 {
        if DT_PREC[x] >= DT_PREC[y] { return x; }
        return y;
    }
    if kx + ky == 3 && kx <= 2 && ky <= 2 {
        // A signed and an unsigned whole type. A uint64 has no signed
        // type that holds it, so the result is a float.
        var s = x;
        var u = y;
        if kx == 2 { s = y; u = x; }
        if DT_PREC[u] == 64 { return 11; }
        return dt_signed(max_i(DT_PREC[u] + 1, DT_PREC[s]));
    }
    // Put the type of the lower kind first.
    if kx > ky { x = b; y = a; }
    let k1 = DT_KIND[x];
    let k2 = DT_KIND[y];
    if k1 <= 2 && k2 == 3 { return max_i(dt_float_of(DT_PREC[x]), y); }
    if k1 <= 2 && k2 == 4 { return max_i(dt_complex_of(DT_PREC[x]), y); }
    if k1 == 3 && k2 == 4 {
        var c: i64 = 12;
        if x == 11 { c = 13; }
        return max_i(c, y);
    }
    return max_i(x, y);
}

// dt_group sorts the kinds for the same_kind rule: bool, whole
// numbers, floats and complex numbers.
fn dt_group(i: i64) -> i64 {
    let k = DT_KIND[i];
    if k == 2 { return 1; }
    return k;
}

// dt_safe tells whether every value of one type is a value of the
// other type.
fn dt_safe(a: i64, b: i64) -> bool {
    if a == b { return true; }
    let ka = DT_KIND[a];
    let kb = DT_KIND[b];
    let pa = DT_PREC[a];
    let pb = DT_PREC[b];
    if ka == 0 { return true; }
    if ka == kb && ka <= 2 { return pb >= pa; }
    if ka == 1 && kb == 2 { return false; }
    if ka == 2 && kb == 1 { return pb >= pa + 1; }
    if (ka == 1 || ka == 2) && (kb == 3 || kb == 4) {
        // A float holds a whole number when its mantissa holds the bits.
        // NumPy holds a 64 bit whole number safe in a float64.
        return pa + 1 <= pb || (pa == 64 && pb == 53);
    }
    if ka >= 3 && kb >= ka { return pb >= pa; }
    return false;
}

// dt_same_kind tells whether a cast keeps the kind or goes up in kind.
fn dt_same_kind(a: i64, b: i64) -> bool {
    if dt_group(b) < dt_group(a) { return false; }
    if DT_KIND[a] == 1 && DT_KIND[b] == 2 { return false; }
    return true;
}

// dt_class tells whether a type belongs to a class of NumPy. The
// classes are generic, number, integer, signedinteger,
// unsignedinteger, floating, complexfloating and inexact. A name that
// is no class gives -1.
fn dt_class(i: i64, name: const u8[:]) -> i64 {
    let k = DT_KIND[i];
    var yes = false;
    if name == "generic" { yes = true; }
    else if name == "number" { yes = k != 0; }
    else if name == "integer" { yes = k == 1 || k == 2; }
    else if name == "signedinteger" { yes = k == 1; }
    else if name == "unsignedinteger" { yes = k == 2; }
    else if name == "floating" { yes = k == 3; }
    else if name == "complexfloating" { yes = k == 4; }
    else if name == "inexact" { yes = k == 3 || k == 4; }
    else { return -1; }
    if yes { return 1; }
    return 0;
}

// issubdtype tells whether a type is a sub type of a type or of a
// class of NumPy. The second name may be a type name or one of the
// classes: generic, number, integer, signedinteger, unsignedinteger,
// floating, complexfloating and inexact.
fn issubdtype(arg1: const u8[:], arg2: const u8[:]) -> bool, Fault {
    let a = dt_index(arg1);
    if a < 0 { return false, dt_bad("issubdtype"); }
    let b = dt_index(arg2);
    if b >= 0 { return a == b, no_fault(); }
    let c = dt_class(a, arg2);
    if c < 0 { return false, dt_bad("issubdtype"); }
    return c == 1, no_fault();
}

// isdtype tells whether a type is of a kind. The kind is one of the
// names bool, signed integer, unsigned integer, integral, real
// floating, complex floating and numeric, or the name of a type. A
// tuple of kinds is a caller loop over the names.
fn isdtype(dtype: const u8[:], kind: const u8[:]) -> bool, Fault {
    let a = dt_index(dtype);
    if a < 0 { return false, dt_bad("isdtype"); }
    let k = DT_KIND[a];
    if kind == "bool" { return k == 0, no_fault(); }
    if kind == "signed integer" { return k == 1, no_fault(); }
    if kind == "unsigned integer" { return k == 2, no_fault(); }
    if kind == "integral" { return k == 1 || k == 2, no_fault(); }
    if kind == "real floating" { return k == 3, no_fault(); }
    if kind == "complex floating" { return k == 4, no_fault(); }
    if kind == "numeric" { return k != 0, no_fault(); }
    let b = dt_index(kind);
    if b < 0 { return false, fault(F_VALUE, "isdtype", "the kind is not a kind name of NumPy or a type"); }
    return a == b, no_fault();
}

// can_cast tells whether a cast between two types obeys a rule of
// casting: no, equiv, safe, same_kind or unsafe. NumPy takes safe as
// the default, so a caller who wants that rule passes "safe".
fn can_cast(from_: const u8[:], to: const u8[:], casting: const u8[:]) -> bool, Fault {
    let a = dt_index(from_);
    let b = dt_index(to);
    if a < 0 || b < 0 { return false, dt_bad("can_cast"); }
    if casting == "no" || casting == "equiv" { return a == b, no_fault(); }
    if casting == "safe" { return dt_safe(a, b), no_fault(); }
    if casting == "same_kind" { return dt_same_kind(a, b), no_fault(); }
    if casting == "unsafe" { return true, no_fault(); }
    return false, fault(F_VALUE, "can_cast", "the casting is not no, equiv, safe, same_kind or unsafe");
}

// promote_types gives the smallest type that holds the values of both
// types.
fn promote_types(type1: const u8[:], type2: const u8[:]) -> const u8[:], Fault {
    let a = dt_index(type1);
    let b = dt_index(type2);
    if a < 0 || b < 0 { return "", dt_bad("promote_types"); }
    return dt_name(dt_promote(a, b)), no_fault();
}

// result_type gives the type that an operation on two arrays of the
// given types has. For two types it is promote_types.
fn result_type(type1: const u8[:], type2: const u8[:]) -> const u8[:], Fault {
    return promote_types(type1, type2);
}

// dt_common_prec gives the rank of the float that a type asks of
// common_type: 0 for float16, 1 for float32, 2 for float64. A whole
// number asks for float64.
fn dt_common_prec(i: i64) -> i64 {
    if DT_KIND[i] <= 2 { return 2; }
    if DT_PREC[i] == 11 { return 0; }
    if DT_PREC[i] == 24 { return 1; }
    return 2;
}

// common_type gives the float type that two number types share. The
// result is always a float or a complex type. A whole number gives at
// least float64. A bool has no common type and is a fault, as in NumPy.
fn common_type(type1: const u8[:], type2: const u8[:]) -> const u8[:], Fault {
    let a = dt_index(type1);
    let b = dt_index(type2);
    if a < 0 || b < 0 { return "", dt_bad("common_type"); }
    if DT_KIND[a] == 0 || DT_KIND[b] == 0 {
        return "", fault(F_VALUE, "common_type", "a bool has no common type");
    }
    let p = max_i(dt_common_prec(a), dt_common_prec(b));
    if DT_KIND[a] == 4 || DT_KIND[b] == 4 {
        if p == 1 { return "complex64", no_fault(); }
        return "complex128", no_fault();
    }
    return dt_name(9 + p), no_fault();
}

// min_scalar_type gives the smallest type that holds a whole number.
// A value of zero or more gives an unsigned type. A value below zero
// gives a signed type. The call takes an i64, so a value above the
// largest i64 is out of its reach.
fn min_scalar_type(v: i64) -> const u8[:] {
    if v >= 0 {
        if v <= 255 { return "uint8"; }
        if v <= 65535 { return "uint16"; }
        if v <= 4294967295 { return "uint32"; }
        return "uint64";
    }
    if v >= -128 { return "int8"; }
    if v >= -32768 { return "int16"; }
    if v >= -2147483648 { return "int32"; }
    return "int64";
}

// min_scalar_type for a float gives the smallest float type that holds
// the value, as NumPy does. A value of NaN or an infinity gives
// float16. A float never demotes to a whole type. The limits 65000 and
// 3.4e38 are the limits of NumPy, not the largest values of the types.
fn min_scalar_type(v: f64) -> const u8[:] {
    if v - v != 0.0 { return "float16"; }
    if v > -65000.0 && v < 65000.0 { return "float16"; }
    if v > -3.4e38 && v < 3.4e38 { return "float32"; }
    return "float64";
}
