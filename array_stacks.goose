// The stacks that join arrays along an axis.
//
// NumPy gives each stack its own rule for the rows of one dimension.
// The rules come from the reference and each function follows one.
namespace scientific;

// as_two_rows gives an array of one dimension the shape of one row.
fn as_two_rows(a) -> Arr, Fault {
    if a.ndim != 1 { return clone(a); }
    return expand_dims(a, 0);
}

// as_two_columns gives an array of one dimension the shape of one
// column.
fn as_two_columns(a) -> Arr, Fault {
    if a.ndim != 1 { return clone(a); }
    return expand_dims(a, 1);
}

// as_three gives an array the shape of three dimensions. An array of
// one dimension becomes a plane of one row. An array of two
// dimensions becomes a plane of one layer. An array of more
// dimensions stays as it is, as NumPy leaves it.
fn as_three(a) -> Arr, Fault {
    if a.ndim == 1 {
        let row, f = expand_dims(a, 0);
        if f.kind != F_NONE { return empty_arr(), f; }
        return expand_dims(row, 2);
    }
    if a.ndim == 2 { return expand_dims(a, 2); }
    return clone(a);
}

// vstack joins arrays along the first dimension. An array of one
// dimension becomes a row, as NumPy gives it.
fn vstack(a, b) -> Arr, Fault {
    let x, fx = as_two_rows(a);
    if fx.kind != F_NONE { return empty_arr(), fx; }
    let y, fy = as_two_rows(b);
    if fy.kind != F_NONE { return empty_arr(), fy; }
    let r, f = concatenate(0, x, y);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "vstack", f.detail); }
    return r, no_fault();
}

// hstack joins arrays along the second dimension. An array of one
// dimension joins along the first, as NumPy gives it.
fn hstack(a, b) -> Arr, Fault {
    if a.ndim == 1 && b.ndim == 1 {
        let r, f = concatenate(0, a, b);
        if f.kind != F_NONE { return empty_arr(), fault(f.kind, "hstack", f.detail); }
        return r, no_fault();
    }
    if a.ndim < 2 || b.ndim < 2 {
        return empty_arr(), fault(F_SHAPE, "hstack", "the arrays hold different counts of dimensions");
    }
    let r, f = concatenate(1, a, b);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "hstack", f.detail); }
    return r, no_fault();
}

// dstack joins arrays along a new third dimension. An array of one
// dimension becomes a plane, as NumPy gives it.
fn dstack(a, b) -> Arr, Fault {
    let x, fx = as_three(a);
    if fx.kind != F_NONE { return empty_arr(), fx; }
    let y, fy = as_three(b);
    if fy.kind != F_NONE { return empty_arr(), fy; }
    let r, f = concatenate(2, x, y);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "dstack", f.detail); }
    return r, no_fault();
}

// column_stack joins arrays along the second dimension. An array of one
// dimension becomes a column, as NumPy gives it.
fn column_stack(a, b) -> Arr, Fault {
    let x, fx = as_two_columns(a);
    if fx.kind != F_NONE { return empty_arr(), fx; }
    let y, fy = as_two_columns(b);
    if fy.kind != F_NONE { return empty_arr(), fy; }
    if x.ndim != y.ndim {
        return empty_arr(), fault(F_SHAPE, "column_stack", "the arrays hold different counts of dimensions");
    }
    let r, f = concatenate(1, x, y);
    if f.kind != F_NONE { return empty_arr(), fault(f.kind, "column_stack", f.detail); }
    return r, no_fault();
}
