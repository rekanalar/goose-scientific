// The tests of the group stacks. Run them with SCIENTIFIC_SUITES=stacks.
namespace stacks_tests;

import scientific from "../..";
import os;

fn test_stacks() {
    let a1, fa1 = scientific::from_values([1.0, 2.0]);
    let b1, fb1 = scientific::from_values([3.0, 4.0]);
    let a2, fa2 = scientific::from_rows(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let b2, fb2 = scientific::from_rows(2, 2, [5.0, 6.0, 7.0, 8.0]);
    assert(fa1.kind == scientific::F_NONE && fb1.kind == scientific::F_NONE);
    assert(fa2.kind == scientific::F_NONE && fb2.kind == scientific::F_NONE);
    // The rows join under one another.
    let v1, fv1 = scientific::vstack(a1, b1);
    assert(fv1.kind == scientific::F_NONE && v1.ndim == 2 && v1.data.len == 4);
    assert(v1.data[0] == 1.0 && v1.data[1] == 2.0 && v1.data[2] == 3.0 && v1.data[3] == 4.0);
    let v2, fv2 = scientific::vstack(a2, b2);
    assert(fv2.kind == scientific::F_NONE && v2.ndim == 2 && v2.data.len == 8);
    assert(v2.shape[0] == 4 && v2.shape[1] == 2 && v2.data[4] == 5.0);
    // The rows join beside each other. A row of one dimension joins
    // along its own direction.
    let h1, fh1 = scientific::hstack(a1, b1);
    assert(fh1.kind == scientific::F_NONE && h1.ndim == 1 && h1.data.len == 4);
    assert(h1.data[0] == 1.0 && h1.data[2] == 3.0 && h1.data[3] == 4.0);
    let h2, fh2 = scientific::hstack(a2, b2);
    assert(fh2.kind == scientific::F_NONE && h2.ndim == 2 && h2.shape[1] == 4);
    assert(h2.data[0] == 1.0 && h2.data[2] == 5.0);
    // The planes join in a third dimension.
    let d1, fd1 = scientific::dstack(a1, b1);
    assert(fd1.kind == scientific::F_NONE && d1.ndim == 3 && d1.data.len == 4);
    assert(d1.data[0] == 1.0 && d1.data[1] == 3.0 && d1.data[2] == 2.0 && d1.data[3] == 4.0);
    let d2, fd2 = scientific::dstack(a2, b2);
    assert(fd2.kind == scientific::F_NONE && d2.ndim == 3 && d2.shape[2] == 2);
    assert(d2.data[0] == 1.0 && d2.data[1] == 5.0);
    // The columns join beside each other.
    let c1, fc1 = scientific::column_stack(a1, b1);
    assert(fc1.kind == scientific::F_NONE && c1.ndim == 2 && c1.data.len == 4);
    assert(c1.data[0] == 1.0 && c1.data[1] == 3.0 && c1.data[2] == 2.0 && c1.data[3] == 4.0);
    let c2, fc2 = scientific::column_stack(a2, b2);
    assert(fc2.kind == scientific::F_NONE && c2.ndim == 2 && c2.shape[1] == 4);
    assert(c2.data[0] == 1.0 && c2.data[2] == 5.0);
    // The stack takes the place of the new dimension.
    let s0, fs0 = scientific::stack(a2, b2, 0);
    let s1, fs1 = scientific::stack(a2, b2, 1);
    let s2, fs2 = scientific::stack(a2, b2, 2);
    let sm, fsm = scientific::stack(a2, b2, -1);
    assert(fs0.kind == scientific::F_NONE && fs1.kind == scientific::F_NONE);
    assert(fs2.kind == scientific::F_NONE && fsm.kind == scientific::F_NONE);
    assert(s0.data[0] == 1.0 && s0.data[4] == 5.0);
    assert(s1.data[0] == 1.0 && s1.data[2] == 5.0);
    assert(s2.data[0] == 1.0 && s2.data[1] == 5.0);
    assert(sm.data[1] == 5.0);
    // The default is the first dimension.
    let sd, fsd = scientific::stack(a2, b2);
    assert(fsd.kind == scientific::F_NONE && sd.data[0] == 1.0 && sd.data[4] == 5.0);
    // An axis outside the shape is a fault.
    let sb, fsb = scientific::stack(a2, b2, 3);
    assert(fsb.kind == scientific::F_RANGE && sb.data.len == 0);
    // Rows of different lengths hold no join.
    let cb, fcb = scientific::vstack(a1, scientific::from_values([1.0, 2.0, 3.0]));
    assert(fcb.kind == scientific::F_SHAPE && cb.data.len == 0);
    // The stacks join arrays of three dimensions in their own axes.
    let p, fp = scientific::reshape(scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), [2, 2, 2]);
    let q, fq = scientific::reshape(scientific::from_values([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]), [2, 2, 2]);
    assert(fp.kind == scientific::F_NONE && fq.kind == scientific::F_NONE);
    let d3, fd3 = scientific::dstack(p, q);
    assert(fd3.kind == scientific::F_NONE && d3.ndim == 3 && d3.shape[2] == 4);
    assert(d3.data[0] == 1.0 && d3.data[2] == 9.0 && d3.data[4] == 3.0);
    let v3, fv3 = scientific::vstack(p, q);
    assert(fv3.kind == scientific::F_NONE && v3.shape[0] == 4 && v3.shape[1] == 2);
    let h3, fh3 = scientific::hstack(p, q);
    assert(fh3.kind == scientific::F_NONE && h3.shape[1] == 4 && h3.shape[2] == 2);
    let c3, fc3 = scientific::column_stack(p, q);
    assert(fc3.kind == scientific::F_NONE && c3.shape[1] == 4 && c3.shape[2] == 2);
    // An array of the most dimensions holds no new dimension.
    let e8, fe8 = scientific::expand_dims(scientific::zeros([1, 1, 1, 1, 1, 1, 1, 2]), 0);
    assert(fe8.kind == scientific::F_LIMIT && e8.data.len == 0);
    // An array of one dimension below the most takes the new one.
    let e7, fe7 = scientific::expand_dims(scientific::zeros([1, 1, 1, 1, 1, 1, 1]), 7);
    assert(fe7.kind == scientific::F_NONE && e7.ndim == 8);
}
