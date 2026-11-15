// The offset of the business days: the date that stands a count of
// business days away from a date, as busday_offset of NumPy.
//
// The offset moves a date by whole weeks in one step and walks the
// rest. A rest holds fewer business days than one week holds. A date
// that is not a business day moves by the roll of the calendar first.
//
// The rolls have the names of NumPy: raise, nat, forward, following,
// backward, preceding, modifiedfollowing and modifiedpreceding. The
// default roll is forward. The two modified rolls hold the month of
// the date. NumPy compares the month alone in those rolls, so a move
// over the turn of a year may cross the year. This file holds the
// same behavior.
namespace scientific;

const MAX_BUS_OFFSET: i64 = 3000000;

// busday_offset_one gives the date that stands a count of business days
// away from a date. A date that is not a time stays a date that is not
// a time, unless the roll is raise. A roll of raise on a date that is
// not a business day is a fault of kind F_VALUE.
fn busday_offset_one(day: f64, offset: i64, code: i64, cal) -> f64, Fault {
    let tag = "busday_offset";
    if cal.busdays == 0 {
        return 0.0, fault(F_VALUE, tag, "the weekmask holds no business day");
    }
    let d, nat, f = date_of(day, tag);
    if f.kind != F_NONE { return 0.0, f; }
    if nat {
        if code == BUS_ROLL_RAISE {
            return 0.0, fault(F_VALUE, tag, "a date that is not a time meets the roll raise");
        }
        return 0.0 / 0.0, no_fault();
    }
    let rolled, rnat, rf = bus_roll_day(d, code, cal, tag);
    if rf.kind != F_NONE { return 0.0, rf; }
    if rnat { return 0.0 / 0.0, no_fault(); }
    if offset > MAX_BUS_OFFSET || offset < -MAX_BUS_OFFSET {
        return 0.0, fault(F_LIMIT, tag, "the offset stands above the limit of 3000000");
    }
    let out, of = bus_step_offset(rolled, offset, cal);
    if of.kind != F_NONE { return 0.0, of; }
    if !date_in_range(out) {
        return 0.0, fault(F_RANGE, tag, "the result stands outside the years 1 to 9999");
    }
    return out as f64, no_fault();
}

// bus_step_offset moves a business day by a count of business days.
// The move jumps by whole weeks and walks the rest. The holidays that
// the jump takes add to the walk.
fn bus_step_offset(date: i64, offset: i64, cal) -> i64, Fault {
    var d = date;
    var off = offset;
    var dow = date_weekday(d);
    if off > 0 {
        // The holidays in the jumped weeks add to the walk. The jump
        // runs from the first holiday on the date or after it to the
        // first holiday after the jumped date.
        let lo = bus_hol_ge(cal, d);
        d += (off / cal.busdays) * 7;
        off = bus_c_rem(off, cal.busdays);
        let hi = bus_hol_gt(cal, d);
        off += hi - lo;
        while off > 0 {
            d += 1;
            dow += 1;
            if dow == 7 { dow = 0; }
            if cal.mask[dow] == 1 && !bus_is_holiday(cal, d) { off -= 1; }
        }
    } else if off < 0 {
        // The holidays in the jumped weeks add to the walk, in the
        // same way as the walk up.
        let hi = bus_hol_gt(cal, d);
        d += (off / cal.busdays) * 7;
        off = bus_c_rem(off, cal.busdays);
        let lo = bus_hol_ge(cal, d);
        off -= hi - lo;
        while off < 0 {
            d -= 1;
            dow -= 1;
            if dow == -1 { dow = 6; }
            if cal.mask[dow] == 1 && !bus_is_holiday(cal, d) { off += 1; }
        }
    }
    return d, no_fault();
}

// busday_offset gives the dates that stand a count of business days
// away from the dates of a list. The offsets and the dates spread over
// each other in the way of NumPy. A date that is not a time stays a
// date that is not a time, unless the roll is raise.
fn busday_offset(dates: Arr, offsets: Arr, roll: const u8[:], cal) -> Arr, Fault {
    let tag = "busday_offset";
    let f = check_stack(dates.ndim, dates.shape, dates.data.len, 1, tag);
    if f.kind != F_NONE { return empty_arr(), f; }
    let f2 = check_stack(offsets.ndim, offsets.shape, offsets.data.len, 1, tag);
    if f2.kind != F_NONE { return empty_arr(), f2; }
    let code, cf = bus_roll_code(roll, tag);
    if cf.kind != F_NONE { return empty_arr(), cf; }
    let shape, sf = broadcast_shapes(dates.shape, offsets.shape);
    if sf.kind != F_NONE { return empty_arr(), sf; }
    var ia: i64[>..] = [];
    var ib: i64[>..] = [];
    bcast_indices(shape, dates.shape, offsets.shape, ia, ib);
    var data: f64[>..] = [];
    let n = numel(shape);
    var i: i64 = 0;
    while i < n {
        let v, vf = busday_offset_one(dates.data[ia[i]], offsets.data[ib[i]] as i64, code, cal);
        if vf.kind != F_NONE { return empty_arr(), vf; }
        data.push(v);
        i += 1;
    }
    var s: i64[..8] = [];
    var k: i64 = 0;
    while k < shape.len { s.push(shape[k]); k += 1; }
    return Arr { ndim: shape.len, shape: s, data: data }, no_fault();
}
