// The cut of an array at a list of places.
//
// The cut at a count of pieces gives pieces of an equal count. The cut
// at a list of places gives a piece between each pair of places, and the
// last piece runs to the end of the axis. A piece may be empty.
//
// A place that lies outside the axis gives a fault. NumPy cuts at the end
// of the axis instead.
namespace scientific;

// split cuts an array along one axis at a list of places. It is the
// second form of the cut by a count of pieces.
fn split(a, sections: i64[:], axis: i64) -> Parts, Fault {
    // The array must agree with its own shape before any value is read.
    let fs = check_stack(a.ndim, a.shape, a.data.len, 1, "split");
    if fs.kind != F_NONE { return empty_parts(), fs; }
    if axis < 0 || axis >= a.ndim {
        return empty_parts(), fault(F_RANGE, "split", "the axis stands outside the shape");
    }
    let width = a.shape[axis];
    // Every place must lie inside the axis, and the places must not fall.
    var start: i64 = 0;
    var i: i64 = 0;
    while i < sections.len {
        if sections[i] < 0 || sections[i] > width {
            return empty_parts(), fault(F_RANGE, "split", "a place of the cut lies outside the axis");
        }
        if sections[i] < start {
            return empty_parts(), fault(F_RANGE, "split", "a place of the cut falls below the one before it");
        }
        start = sections[i];
        i += 1;
    }
    // The count of the values of every piece along the axis.
    var sizes: i64[>..] = [];
    start = 0;
    i = 0;
    while i < sections.len {
        sizes.push(sections[i] - start);
        start = sections[i];
        i += 1;
    }
    sizes.push(width - start);
    return cut_sizes(a, sizes, axis);
}
