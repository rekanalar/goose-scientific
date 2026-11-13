// The tests of the group assemble. Run them with SCIENTIFIC_SUITES=assemble.
namespace assemble_tests;

import scientific from "../..";
import os;
import common;

// The assembly of shapes, pinned against NumPy.
fn test_assemble() {
    let a6 = common::seq(6);
    let a, g0 = scientific::reshape(a6, [2, 3]);
    assert(g0.kind == scientific::F_NONE);
    let m, g1 = scientific::reshape(common::seq(24), [2, 3, 4]);
    assert(g1.kind == scientific::F_NONE);
    // axes
    let sw, g2 = scientific::swapaxes(m, 0, -1);
    assert(g2.kind == scientific::F_NONE && sw.shape[0] == 4 && sw.shape[2] == 2);
    assert(sw.data[3 * 6 + 2 * 2 + 1] == 23.0);
    let pd, g3 = scientific::permute_dims(m, [2, 0, 1]);
    assert(g3.kind == scientific::F_NONE && pd.shape[0] == 4 && pd.shape[1] == 2 && pd.shape[2] == 3);
    let pbad, g4 = scientific::permute_dims(m, [0, 1]);
    assert(g4.kind == scientific::F_SHAPE && pbad.data.len == 0);
    let a3, g5 = scientific::atleast_3d(common::seq(3));
    assert(g5.kind == scientific::F_NONE && a3.shape[0] == 1 && a3.shape[1] == 3 && a3.shape[2] == 1);
    let b3, g6 = scientific::atleast_3d(a);
    assert(g6.kind == scientific::F_NONE && b3.shape[0] == 2 && b3.shape[2] == 1);
    let b2, g7 = scientific::atleast_2d(common::seq(3));
    assert(g7.kind == scientific::F_NONE && b2.ndim == 2 && b2.shape[0] == 1);
    // growth and cut
    let ap, g8 = scientific::append(a, common::seq(2));
    assert(g8.kind == scientific::F_NONE && common::same_vals(ap, [8], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0]));
    let apx, g9 = scientific::append_axis(a, a, 1);
    assert(g9.kind == scientific::F_NONE && apx.shape[1] == 6 && apx.data[3] == 0.0);
    let da, g10 = scientific::delete_axis(a, [0, -1], 1);
    assert(g10.kind == scientific::F_NONE && common::same_vals(da, [2, 1], [1.0, 4.0]));
    let df, g11 = scientific::delete(a, [1, 4]);
    assert(g11.kind == scientific::F_NONE && common::same_vals(df, [4], [0.0, 2.0, 3.0, 5.0]));
    let dbad, g12 = scientific::delete(a, [6]);
    assert(g12.kind == scientific::F_RANGE && dbad.data.len == 0);
    let rz, g13 = scientific::resize(common::seq(3), [2, 4]);
    assert(g13.kind == scientific::F_NONE && common::same_vals(rz, [2, 4], [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0]));
    // padding
    let p1, g14 = scientific::pad_modes(a, [1, 2], [2, 1], "reflect");
    assert(g14.kind == scientific::F_NONE);
    assert(common::same_vals(p1, [5, 6], [5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 5.0, 4.0, 3.0, 4.0, 5.0, 4.0]));
    let q = common::av([1.0, 2.0, 3.0, 4.0]);
    let pr, g15 = scientific::pad_modes(q, [5], [6], "reflect");
    assert(g15.kind == scientific::F_NONE && common::same_vals(pr, [15], [2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0]));
    let ps, g16 = scientific::pad_modes(q, [5], [6], "symmetric");
    assert(g16.kind == scientific::F_NONE && common::same_vals(ps, [15], [4.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0]));
    let pw, g17 = scientific::pad_modes(q, [5], [6], "wrap");
    assert(g17.kind == scientific::F_NONE && common::same_vals(pw, [15], [4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0]));
    let pe, g18 = scientific::pad_modes(q, [2], [3], "edge");
    assert(g18.kind == scientific::F_NONE && common::same_vals(pe, [9], [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0]));
    let pc, g19 = scientific::pad_modes(common::av([1.0, 2.0]), [1], [2], "constant", 9.0);
    assert(g19.kind == scientific::F_NONE && common::same_vals(pc, [5], [9.0, 1.0, 2.0, 9.0, 9.0]));
    let pm, g20 = scientific::pad_modes(q, [1], [1], "unknown");
    assert(g20.kind == scientific::F_VALUE && pm.data.len == 0);
    let pn, g21 = scientific::pad_modes(q, [-1], [1]);
    assert(g21.kind == scientific::F_VALUE && pn.data.len == 0);
    // diagonals
    let dg, g22 = scientific::diagonal(m, 1, 0, 2);
    assert(g22.kind == scientific::F_NONE && common::same_vals(dg, [3, 2], [1.0, 14.0, 5.0, 18.0, 9.0, 22.0]));
    let dh, g23 = scientific::diagonal(m, -1, 1, 2);
    assert(g23.kind == scientific::F_NONE && common::same_vals(dh, [2, 2], [4.0, 9.0, 16.0, 21.0]));
    let w, g24 = scientific::reshape(common::seq(12), [3, 4]);
    assert(g24.kind == scientific::F_NONE);
    let d2, g25 = scientific::diagonal(w, 2);
    assert(g25.kind == scientific::F_NONE && common::same_vals(d2, [2], [2.0, 7.0]));
    let d3, g26 = scientific::diagonal(w, 4);
    assert(g26.kind == scientific::F_VALUE && d3.data.len == 0);
    var z, gz = scientific::zeros([7, 3]);
    assert(gz.kind == scientific::F_NONE);
    assert(scientific::fill_diagonal(z, 7.0, true).kind == scientific::F_NONE);
    assert(z.data[0] == 7.0 && z.data[4] == 7.0 && z.data[8] == 7.0 && z.data[12] == 7.0 && z.data[20] == 7.0 && z.data[1] == 0.0);
    var z4, gz4 = scientific::zeros([4, 3]);
    assert(gz4.kind == scientific::F_NONE);
    assert(scientific::fill_diagonal(z4, 7.0).kind == scientific::F_NONE);
    assert(z4.data[0] == 7.0 && z4.data[4] == 7.0 && z4.data[8] == 7.0 && z4.data[9] == 0.0 && z4.data[12 - 1] == 0.0);
    var z3, gz3 = scientific::zeros([2, 2, 2]);
    assert(gz3.kind == scientific::F_NONE);
    assert(scientific::fill_diagonal(z3, 5.0).kind == scientific::F_NONE);
    assert(z3.data[0] == 5.0 && z3.data[7] == 5.0 && z3.data[3] == 0.0);
    // tables of places
    let t, g27 = scientific::tri(3, 4, 1);
    assert(g27.kind == scientific::F_NONE);
    assert(common::same_vals(t, [3, 4], [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]));
    let tl, g28 = scientific::tril_indices(3, -1, 4);
    assert(g28.kind == scientific::F_NONE && common::same_vals(tl, [2, 3], [1.0, 2.0, 2.0, 0.0, 0.0, 1.0]));
    let tu, g29 = scientific::triu_indices(3, 1, 4);
    assert(g29.kind == scientific::F_NONE && common::same_vals(tu, [2, 6], [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 3.0]));
    let di, g30 = scientific::diag_indices(3, 2);
    assert(g30.kind == scientific::F_NONE && common::same_vals(di, [2, 3], [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]));
    let ix, g31 = scientific::indices([2, 3]);
    assert(g31.kind == scientific::F_NONE);
    assert(common::same_vals(ix, [2, 2, 3], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]));
    let gc, gr, g32 = scientific::ix_([0, 2], [1, 3, 4]);
    assert(g32.kind == scientific::F_NONE && gc.shape[0] == 2 && gc.shape[1] == 1 && gr.shape[0] == 1 && gr.shape[1] == 3);
    let un, g33 = scientific::unravel_index([5, 11], [2, 3, 4]);
    assert(g33.kind == scientific::F_NONE && common::same_vals(un, [3, 2], [0.0, 0.0, 1.0, 2.0, 1.0, 3.0]));
    let ub, g34 = scientific::unravel_index([24], [2, 3, 4]);
    assert(g34.kind == scientific::F_RANGE && ub.data.len == 0);
    let rv, g35 = scientific::ravel_multi_index(un, [2, 3, 4]);
    assert(g35.kind == scientific::F_NONE && common::same_vals(rv, [2], [5.0, 11.0]));
    let mw, g36 = scientific::from_rows(2, 2, [5.0, -1.0, 0.0, 0.0]);
    assert(g36.kind == scientific::F_NONE);
    let rw, g37 = scientific::ravel_multi_index(mw, [3, 4], "wrap");
    assert(g37.kind == scientific::F_NONE && common::same_vals(rw, [2], [8.0, 8.0]));
    let rc, g38 = scientific::ravel_multi_index(mw, [3, 4], "clip");
    assert(g38.kind == scientific::F_NONE && common::same_vals(rc, [2], [8.0, 0.0]));
    let rb, g39 = scientific::ravel_multi_index(mw, [3, 4]);
    assert(g39.kind == scientific::F_RANGE && rb.data.len == 0);
    // pieces and blocks
    let us, g40 = scientific::unstack(a, 1);
    assert(g40.kind == scientific::F_NONE && us.count == 3);
    let u1, g41 = scientific::unstack_part(us, 1);
    assert(g41.kind == scientific::F_NONE && common::same_vals(u1, [2], [1.0, 4.0]));
    let bl, g42 = scientific::block_grid(a, a, a, a);
    assert(g42.kind == scientific::F_NONE && bl.shape[0] == 4 && bl.shape[1] == 6 && bl.data[9] == 3.0);
}
