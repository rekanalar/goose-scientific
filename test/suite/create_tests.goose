// The tests of the group create. Run them with SCIENTIFIC_SUITES=create.
namespace create_tests;

import scientific from "../..";
import os;
import common;

// The creation functions hold the shapes and the values they claim.
fn test_create() {
    let a, f = scientific::zeros([2, 3]);
    assert(f.kind == scientific::F_NONE);
    assert(a.ndim == 2 && a.shape[0] == 2 && a.shape[1] == 3 && a.data.len == 6);
    assert(a.data[4] == 0.0);
    let o, f2 = scientific::ones([4]);
    assert(f2.kind == scientific::F_NONE && o.data.len == 4 && o.data[3] == 1.0);
    let fu, f3 = scientific::full([2, 2], 7.5);
    assert(f3.kind == scientific::F_NONE && fu.data[2] == 7.5);
    let e, f4 = scientific::eye(3);
    assert(f4.kind == scientific::F_NONE);
    assert(e.data[0] == 1.0 && e.data[1] == 0.0 && e.data[4] == 1.0 && e.data[8] == 1.0);
    let dg, f5 = scientific::diag([2.0, 3.0]);
    assert(f5.kind == scientific::F_NONE);
    assert(dg.data[0] == 2.0 && dg.data[1] == 0.0 && dg.data[3] == 3.0);
    let ar, f6 = scientific::arange(0.0, 5.0, 1.0);
    assert(f6.kind == scientific::F_NONE);
    assert(ar.data.len == 5 && ar.data[4] == 4.0);
    let ar2, f7 = scientific::arange(5.0, 1.0, -2.0);
    assert(f7.kind == scientific::F_NONE);
    assert(ar2.data.len == 2 && ar2.data[1] == 3.0);
    let ls, f8 = scientific::linspace(0.0, 1.0, 5);
    assert(f8.kind == scientific::F_NONE);
    assert(ls.data.len == 5 && ls.data[0] == 0.0 && ls.data[4] == 1.0 && common::close(ls.data[2], 0.5));
    let ls1, f9 = scientific::linspace(2.0, 9.0, 1);
    assert(f9.kind == scientific::F_NONE && ls1.data.len == 1 && ls1.data[0] == 2.0);
    let fv, f10 = scientific::from_values([3.0, 1.0, 4.0]);
    assert(f10.kind == scientific::F_NONE && fv.ndim == 1 && fv.data[1] == 1.0);
    let c, f11 = scientific::clone(common::make23());
    assert(f11.kind == scientific::F_NONE && c.data[5] == 6.0);
    // A bad shape, a bad count and a bad step are faults.
    let b1, g1 = scientific::zeros([0, 2]);
    assert(g1.kind == scientific::F_VALUE);
    let b2, g2 = scientific::zeros([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert(g2.kind == scientific::F_LIMIT);
    let b3, g3 = scientific::from_rows(2, 3, [1.0, 2.0]);
    assert(g3.kind == scientific::F_SHAPE);
    let b4, g4 = scientific::arange(0.0, 1.0, 0.0);
    assert(g4.kind == scientific::F_VALUE);
    let b5, g5 = scientific::linspace(0.0, 1.0, 0);
    assert(g5.kind == scientific::F_VALUE);
    let b6, g6 = scientific::eye(0);
    assert(g6.kind == scientific::F_VALUE);
    // A range over the limit is a fault.
    let b7, g7 = scientific::arange(0.0, 400000000.0, 0.5);
    assert(g7.kind == scientific::F_LIMIT);
    // A fault result holds an empty array.
    assert(b7.ndim == 0 && b7.data.len == 0);
}

fn test_misc() {
    // The shape functions.
    let e, fe = scientific::empty([2, 3]);
    assert(fe.kind == scientific::F_NONE && e.data.len == 6 && e.data[0] == 0.0);
    let fl, ffl = scientific::full_like(e, 7.0);
    assert(ffl.kind == scientific::F_NONE && fl.data[0] == 7.0);
    assert(fl.shape[0] == 2 && fl.shape[1] == 3);
    let zz, fzz = scientific::zeros_like(e);
    assert(fzz.kind == scientific::F_NONE && zz.data[0] == 0.0);
    let oo, foo = scientific::ones_like(e);
    assert(foo.kind == scientific::F_NONE && oo.data[0] == 1.0);
    // The writes through a mask.
    var m = scientific::from_values([1.0, 2.0, 3.0, 4.0]);
    let fpm = scientific::putmask(m, scientific::from_values([1.0, 0.0, 1.0, 0.0]), [9.0]);
    assert(fpm.kind == scientific::F_NONE);
    assert(m.data[0] == 9.0 && m.data[1] == 2.0 && m.data[2] == 9.0 && m.data[3] == 4.0);
    var m2 = scientific::from_values([0.0, 0.0, 0.0, 0.0]);
    let fpl = scientific::place(m2, scientific::from_values([1.0, 0.0, 1.0, 1.0]), [5.0, 6.0]);
    assert(fpl.kind == scientific::F_NONE);
    assert(m2.data[0] == 5.0 && m2.data[2] == 6.0 && m2.data[3] == 5.0);
    // place cycles the values when they run out, as NumPy does.
    let fplbad = scientific::place(m2, scientific::from_values([1.0, 1.0, 1.0, 0.0]), [7.0]);
    assert(fplbad.kind == scientific::F_NONE);
    assert(m2.data[0] == 7.0 && m2.data[1] == 7.0 && m2.data[2] == 7.0);
    // values that hold no element are a fault, and the array stays.
    let fplnone = scientific::place(m2, scientific::from_values([1.0, 0.0, 0.0, 0.0]), []);
    assert(fplnone.kind == scientific::F_VALUE);
    // a mask of a different count is a fault.
    let fplshape = scientific::place(m2, scientific::from_values([1.0, 0.0]), [7.0]);
    assert(fplshape.kind == scientific::F_SHAPE);
    // The cuts.
    let a6 = scientific::from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let sp, fsp = scientific::split(a6, 2, 0);
    assert(fsp.kind == scientific::F_NONE && sp.count == 2);
    let sp0, fsp0 = scientific::part(sp, 0);
    let sp1, fsp1 = scientific::part(sp, 1);
    assert(fsp0.kind == scientific::F_NONE && fsp1.kind == scientific::F_NONE);
    assert(sp0.data.len == 3 && sp0.data[0] == 1.0 && sp0.data[2] == 3.0);
    assert(sp1.data.len == 3 && sp1.data[0] == 4.0 && sp1.data[2] == 6.0);
    let spbad, fspb = scientific::part(sp, 2);
    assert(fspb.kind == scientific::F_RANGE);
    let spcut, fspcut = scientific::split(a6, 4, 0);
    assert(fspcut.kind == scientific::F_VALUE);
    // The uneven cut gives the first pieces one element more.
    let asp, fasp = scientific::array_split(a6, 4, 0);
    assert(fasp.kind == scientific::F_NONE && asp.count == 4);
    let ap0, fap0 = scientific::part(asp, 0);
    let ap1, fap1 = scientific::part(asp, 1);
    let ap2, fap2 = scientific::part(asp, 2);
    let ap3, fap3 = scientific::part(asp, 3);
    assert(fap0.kind == scientific::F_NONE);
    assert(ap0.data.len == 2 && ap0.data[0] == 1.0 && ap0.data[1] == 2.0);
    assert(ap1.data.len == 2 && ap1.data[0] == 3.0 && ap1.data[1] == 4.0);
    assert(ap2.data.len == 1 && ap2.data[0] == 5.0);
    assert(ap3.data.len == 1 && ap3.data[0] == 6.0);
    let m23 = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let hs, fhs = scientific::hsplit(m23, 3);
    assert(fhs.kind == scientific::F_NONE && hs.count == 3);
    let hp0, fhp0 = scientific::part(hs, 0);
    let hp1, fhp1 = scientific::part(hs, 1);
    let hp2, fhp2 = scientific::part(hs, 2);
    assert(fhp0.kind == scientific::F_NONE);
    assert(hp0.data.len == 2 && hp0.data[0] == 1.0 && hp0.data[1] == 4.0);
    assert(hp0.shape[0] == 2 && hp0.shape[1] == 1);
    assert(hp1.data.len == 2 && hp1.data[0] == 2.0 && hp1.data[1] == 5.0);
    assert(hp2.data.len == 2 && hp2.data[0] == 3.0 && hp2.data[1] == 6.0);
    let vs, fvs = scientific::vsplit(m23, 2);
    assert(fvs.kind == scientific::F_NONE && vs.count == 2);
    let vp0, fvp0 = scientific::part(vs, 0);
    assert(fvp0.kind == scientific::F_NONE);
    assert(vp0.data.len == 3 && vp0.data[0] == 1.0 && vp0.data[2] == 3.0);
    // The growth.
    let pd, fpd = scientific::pad_value(a6, 0, 2, 1, 0.5);
    assert(fpd.kind == scientific::F_NONE && pd.data.len == 9);
    assert(pd.data[0] == 0.5 && pd.data[1] == 0.5 && pd.data[2] == 1.0);
    assert(pd.data[8] == 0.5);
    let pdbad, fpdbad = scientific::pad_value(a6, 0, -1, 1, 0.0);
    assert(fpdbad.kind == scientific::F_VALUE);
    // The values between points.
    let ip, fip = scientific::interp([0.0, 10.0, 20.0], [0.0, 1.0, 3.0], [0.5, 2.0, -1.0, 5.0]);
    assert(fip.kind == scientific::F_NONE);
    assert(ip.data[0] == 5.0 && ip.data[1] == 15.0 && ip.data[2] == 0.0 && ip.data[3] == 20.0);
    let ipbad, fipbad = scientific::interp([0.0, 10.0], [1.0, 0.0], [0.5]);
    assert(fipbad.kind == scientific::F_VALUE);
    // The derivative.
    let gr, fgr = scientific::gradient(scientific::from_values([1.0, 4.0, 9.0, 16.0]), 1.0);
    assert(fgr.kind == scientific::F_NONE);
    assert(gr.data[0] == 3.0 && gr.data[1] == 4.0 && gr.data[2] == 6.0 && gr.data[3] == 7.0);
    let grbad, fgrbad = scientific::gradient(a6, 0.0);
    assert(fgrbad.kind == scientific::F_VALUE);
    // The trims and the triangles.
    let tz, ftz = scientific::trim_zeros(scientific::from_values([0.0, 0.0, 3.0, 1.0, 0.0, 2.0, 0.0]));
    assert(ftz.kind == scientific::F_NONE);
    assert(tz.data[0] == 3.0 && tz.data[1] == 1.0 && tz.data[2] == 0.0 && tz.data[3] == 2.0);
    let tzbad, ftzbad = scientific::trim_zeros(scientific::from_values([0.0, 0.0]));
    assert(ftzbad.kind == scientific::F_VALUE);
    let ins, fins = scientific::insert(a6, 2, [9.0, 9.5]);
    assert(fins.kind == scientific::F_NONE && ins.data.len == 8);
    assert(ins.data[0] == 1.0 && ins.data[1] == 2.0 && ins.data[2] == 9.0 && ins.data[3] == 9.5);
    assert(ins.data[4] == 3.0 && ins.data[7] == 6.0);
    let insbad, finsbad = scientific::insert(a6, 7, [1.0]);
    assert(finsbad.kind == scientific::F_RANGE);
    let tl, ftl = scientific::tril(m23, 0);
    let tu, ftu = scientific::triu(m23, 0);
    assert(ftl.kind == scientific::F_NONE && ftu.kind == scientific::F_NONE);
    assert(tl.data[0] == 1.0 && tl.data[1] == 0.0 && tl.data[2] == 0.0);
    assert(tl.data[3] == 4.0 && tl.data[4] == 5.0 && tl.data[5] == 0.0);
    assert(tu.data[0] == 1.0 && tu.data[1] == 2.0 && tu.data[2] == 3.0);
    assert(tu.data[3] == 0.0 && tu.data[4] == 5.0 && tu.data[5] == 6.0);
}

// The scales of numbers hold the values of NumPy.
fn test_scales() {
    let id, fid = scientific::identity(3);
    assert(fid.kind == scientific::F_NONE && id.data.len == 9);
    assert(id.data[0] == 1.0 && id.data[4] == 1.0 && id.data[8] == 1.0 && id.data[1] == 0.0);
    let vals, fvals = scientific::from_values([1.0, 2.0, 3.0]);
    assert(fvals.kind == scientific::F_NONE);
    let df, fdf = scientific::diagflat(vals.data, 1);
    assert(fdf.kind == scientific::F_NONE && df.data.len == 16);
    assert(df.data[1] == 1.0 && df.data[6] == 2.0 && df.data[11] == 3.0 && df.data[0] == 0.0);
    let df0, fdf0 = scientific::diagflat(vals.data);
    assert(fdf0.kind == scientific::F_NONE && df0.data.len == 9);
    assert(df0.data[0] == 1.0 && df0.data[4] == 2.0 && df0.data[8] == 3.0);
    // The matrix of the identity takes a count of columns and an
    // offset of the diagonal.
    let ey, fey = scientific::eye(3, 5, 1);
    assert(fey.kind == scientific::F_NONE && ey.data.len == 15);
    assert(ey.data[1] == 1.0 && ey.data[7] == 1.0 && ey.data[13] == 1.0 && ey.data[0] == 0.0);
    let ey2, fey2 = scientific::eye(3, 0, -1);
    assert(fey2.kind == scientific::F_NONE && ey2.data.len == 9);
    assert(ey2.data[3] == 1.0 && ey2.data[7] == 1.0 && ey2.data[0] == 0.0);
    // A size below one is a fault.
    let eybad, feybad = scientific::eye(0);
    assert(feybad.kind == scientific::F_VALUE && eybad.data.len == 0);
    // The scale of a power holds the power of the base.
    let ls, fls = scientific::logspace(0.0, 2.0, 3);
    assert(fls.kind == scientific::F_NONE);
    assert(common::close(ls.data[0], 1.0) && common::close(ls.data[1], 10.0) && common::close(ls.data[2], 100.0));
    let ls2, fls2 = scientific::logspace(0.0, 3.0, 4, 2.0);
    assert(fls2.kind == scientific::F_NONE);
    assert(common::close(ls2.data[0], 1.0) && common::close(ls2.data[1], 2.0));
    assert(common::close(ls2.data[2], 4.0) && common::close(ls2.data[3], 8.0));
    // A count below one is a fault.
    let lsbad, flsbad = scientific::logspace(0.0, 2.0, 0);
    assert(flsbad.kind == scientific::F_VALUE && lsbad.data.len == 0);
    // The scale of a geometric sequence rises by a factor.
    let gs, fgs = scientific::geomspace(1.0, 100.0, 3);
    assert(fgs.kind == scientific::F_NONE);
    assert(common::close(gs.data[0], 1.0) && common::close(gs.data[1], 10.0) && common::close(gs.data[2], 100.0));
    let gs2, fgs2 = scientific::geomspace(-1.0, -100.0, 3);
    assert(fgs2.kind == scientific::F_NONE);
    assert(common::close(gs2.data[0], -1.0) && common::close(gs2.data[1], -10.0) && common::close(gs2.data[2], -100.0));
    // A zero holds no geometric sequence.
    let gsbad, fgsbad = scientific::geomspace(0.0, 100.0, 3);
    assert(fgsbad.kind == scientific::F_VALUE && gsbad.data.len == 0);
    let gsbad2, fgsbad2 = scientific::geomspace(1.0, 0.0, 3);
    assert(fgsbad2.kind == scientific::F_VALUE && gsbad2.data.len == 0);
}

fn test_positions() {
    // NumPy 2.5.3: argwhere and flatnonzero of a mask on a 3x3 grid.
    let a, fa = scientific::from_rows(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert(fa.kind == scientific::F_NONE);
    let m, fm = scientific::from_rows(3, 3, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    assert(fm.kind == scientific::F_NONE);
    let w, fw = scientific::argwhere(m);
    assert(fw.kind == scientific::F_NONE && w.shape[0] == 5 && w.shape[1] == 2);
    assert(w.data[0] == 0.0 && w.data[1] == 0.0 && w.data[2] == 0.0 && w.data[3] == 2.0);
    assert(w.data[4] == 1.0 && w.data[5] == 1.0 && w.data[6] == 2.0 && w.data[7] == 0.0);
    assert(w.data[8] == 2.0 && w.data[9] == 2.0);
    let flat, ff = scientific::flatnonzero(m);
    assert(ff.kind == scientific::F_NONE && flat.data.len == 5);
    assert(flat.data[0] == 0.0 && flat.data[1] == 2.0 && flat.data[2] == 4.0 && flat.data[3] == 6.0 && flat.data[4] == 8.0);
    let w2, fw2 = scientific::argwhere(a);
    assert(fw2.kind == scientific::F_NONE && w2.shape[0] == 9 && w2.shape[1] == 2);
    assert(w2.data[0] == 0.0 && w2.data[16] == 2.0 && w2.data[17] == 2.0);
    // A mask of one dimension gives one column.
    let v, fv = scientific::from_values([0.0, 3.0, 0.0, 5.0]);
    assert(fv.kind == scientific::F_NONE);
    let f2, ff2 = scientific::flatnonzero(v);
    assert(ff2.kind == scientific::F_NONE && f2.data.len == 2 && f2.data[0] == 1.0 && f2.data[1] == 3.0);
    let w3, fw3 = scientific::argwhere(v);
    assert(fw3.kind == scientific::F_NONE && w3.shape[0] == 2 && w3.shape[1] == 1);
    assert(w3.data[0] == 1.0 && w3.data[1] == 3.0);
    // No match gives zero rows.
    let z, fz = scientific::from_values([0.0, 0.0]);
    assert(fz.kind == scientific::F_NONE);
    let w4, fw4 = scientific::argwhere(z);
    assert(fw4.kind == scientific::F_NONE && w4.shape[0] == 0 && w4.shape[1] == 1 && w4.data.len == 0);
}

// eigvalsh takes a matrix of twelve rows to full accuracy.
fn test_eigvalsh_larger() {
    var d: f64[>..] = [];
    var i: i64 = 0;
    while i < 144 { d.push(0.0); i += 1; }
    i = 0;
    while i < 12 {
        d[i * 12 + i] = 2.0 + 0.1 * i;
        if i + 1 < 12 { d[i * 12 + i + 1] = -1.0; d[(i + 1) * 12 + i] = -1.0; }
        i += 1;
    }
    let a, fa = scientific::from_rows(12, 12, copy(d));
    let w, f = scientific::eigvalsh(a);
    assert(fa.kind == scientific::F_NONE && f.kind == scientific::F_NONE);
    assert(common::all_loose(w, [0.39949588055807855, 0.7684859001589346, 1.086782275967645, 1.4388282352528388, 1.8538443683280288, 2.312938653238329, 2.7870613467616696, 3.246155631671971, 3.6611717647471615, 4.013217724032353, 4.3315140998410655, 4.7005041194419235]));
    return;
}
