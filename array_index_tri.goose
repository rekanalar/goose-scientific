// The index helpers that take the shape of an array: diag_indices_from,
// tril_indices_from, triu_indices_from and mask_indices.
//
// Every call gives the places as one array of two rows, the row numbers
// and then the column numbers, as tril_indices does. A diagonal of an
// array of more than two axes gives one row per axis.
namespace scientific;

// diag_indices_from gives the places of the main diagonal of an array.
// The array holds at least two axes of one length, as in NumPy.
fn diag_indices_from(a) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "diag_indices_from");
    if f.kind != F_NONE { return empty_arr(), f; }
    if a.ndim < 2 {
        return empty_arr(), fault(F_SHAPE, "diag_indices_from", "the array holds fewer than two axes");
    }
    var i: i64 = 1;
    while i < a.ndim {
        if a.shape[i] != a.shape[0] {
            return empty_arr(), fault(F_SHAPE, "diag_indices_from", "the axes hold different lengths");
        }
        i += 1;
    }
    return diag_indices(a.shape[0], a.ndim);
}

// tril_indices_from gives the places of the lower triangle of a matrix.
// The diagonal k shifts the triangle, as in tril_indices.
fn tril_indices_from(a, k: i64 = 0) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "tril_indices_from");
    if f.kind != F_NONE { return empty_arr(), f; }
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "tril_indices_from", "the array holds a count of axes that is not two");
    }
    return tri_places(a.shape[0], k, a.shape[1], true, "tril_indices_from");
}

// triu_indices_from gives the places of the upper triangle of a matrix.
fn triu_indices_from(a, k: i64 = 0) -> Arr, Fault {
    let f = check_stack(a.ndim, a.shape, a.data.len, 1, "triu_indices_from");
    if f.kind != F_NONE { return empty_arr(), f; }
    if a.ndim != 2 {
        return empty_arr(), fault(F_SHAPE, "triu_indices_from", "the array holds a count of axes that is not two");
    }
    return tri_places(a.shape[0], k, a.shape[1], false, "triu_indices_from");
}

// mask_indices gives the places that a triangle mask picks in a square
// matrix of n rows. NumPy takes the function of the mask, and the
// language holds no storable function, so the mode is the text "triu" or
// "tril". Any other text is a fault. A triangle with no place is a
// fault, as in tril_indices.
fn mask_indices(n: i64, mode: const u8[:], k: i64 = 0) -> Arr, Fault {
    if mode == "triu" { return tri_places(n, k, n, false, "mask_indices"); }
    if mode == "tril" { return tri_places(n, k, n, true, "mask_indices"); }
    return empty_arr(), fault(F_VALUE, "mask_indices", "the mode is not triu or tril");
}
