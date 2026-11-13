// The tests of the group schur_stacks. Run them with SCIENTIFIC_SUITES=schur_stacks.
namespace schur_stacks_tests;

import scientific from "../..";
import os;
import common;

fn test_schur_stacks() {
    // The complex Schur form of a stack of two forms.
    let ts = scientific::Arr { ndim: 3, shape: [2, 3, 3], data: [2.6589670819169946, 1.4244045756721777, -1.9293343948770278, 0.0, -0.32948354095849702, -0.49063703822841376, 0.0, 1.3117892147840797, -0.32948354095849702, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, -1.0, 4.0] };
    let zs = scientific::Arr { ndim: 3, shape: [2, 3, 3], data: [0.72711590900990863, -0.60156188058076243, 0.33079564491817354, 0.52839427726760835, 0.79801891979128614, 0.28976765072380067, 0.43829435613014156, 0.035904143037891308, -0.89811410739236175, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
    let tc, zc, ftc = scientific::rsf2csf(ts, zs);
    common::vchk12(tc, ftc, [2.6589670819169946, 0.0, -1.6459278069146803, 0.7431641870407133, -1.21516887151373, 1.0066046203353818, 0.0, 0.0, -0.329483540958497, 0.8022545575574112, -0.8211521765556657, -2.7755575615628914e-17, 0.0, 0.0, 0.0, 0.0, -0.329483540958497, -0.8022545575574112, 0.0, 0.0, -1.7499999999999996, 0.43301270189221974, -0.9999999999999997, -0.8660254037844395, 0.0, 0.0, 2.499999999999999, 0.8660254037844394, 2.4999999999999964, 2.598076211353318, 0.0, 0.0, 0.0, 0.0, 2.5000000000000004, -0.8660254037844393]);
    common::vchk12(zc, ftc, [0.7271159090099086, 0.0, 0.2822039309633517, -0.3138569291140597, 0.5131963797757693, -0.172588238433742, 0.5283942772676083, 0.0, 0.24720268043583152, 0.4163557825486023, -0.680795166465797, -0.15118242686017014, 0.43829435613014156, 0.0, -0.7661870265023007, 0.018732510220669858, -0.030630059551327294, 0.4685791185240134, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7499999999999998, 0.43301270189221974, 0.4999999999999999, 0.0, 0.0, 0.0, -0.4999999999999999, 0.0, -0.7499999999999998, -0.43301270189221974]);
    assert(tc.ndim == 3 && tc.shape[0] == 2 && tc.shape[1] == 3 && tc.shape[2] == 3);
    assert(zc.ndim == 3 && zc.shape[0] == 2 && zc.shape[1] == 3 && zc.shape[2] == 3);
    // One change of the basis over the stack of two forms.
    let pz2 = scientific::Arr { ndim: 2, shape: [3, 3], data: [0.72711590900990863, -0.60156188058076243, 0.33079564491817354, 0.52839427726760835, 0.79801891979128614, 0.28976765072380067, 0.43829435613014156, 0.035904143037891308, -0.89811410739236175] };
    let tc2, zc2, ftc2 = scientific::rsf2csf(ts, pz2);
    common::achk(tc2, tc);
    common::vchk12(zc2, ftc2, [0.7271159090099086, 0.0, 0.2822039309633517, -0.3138569291140597, 0.5131963797757693, -0.172588238433742, 0.5283942772676083, 0.0, 0.24720268043583152, 0.4163557825486023, -0.680795166465797, -0.15118242686017014, 0.43829435613014156, 0.0, -0.7661870265023007, 0.018732510220669858, -0.030630059551327294, 0.4685791185240134, 0.7271159090099086, 0.0, 0.28577358797648494, -0.2604839352656408, -0.5488776739790112, -0.14323871598019766, 0.5283942772676083, 0.0, -0.7433980152053646, 0.3455523286199354, 0.18168372185279252, -0.12547307336087396, 0.43829435613014156, 0.0, 0.42212894641776233, 0.015546949985962046, 0.6915376520632167, 0.38889481624948574]);
    assert(tc2.shape[0] == 2 && zc2.shape[0] == 2);
    // A stack of two changes of the basis over one form.
    let pt2 = scientific::Arr { ndim: 2, shape: [3, 3], data: [2.6589670819169946, 1.4244045756721777, -1.9293343948770278, 0.0, -0.32948354095849702, -0.49063703822841376, 0.0, 1.3117892147840797, -0.32948354095849702] };
    let tc3, zc3, ftc3 = scientific::rsf2csf(pt2, zs);
    common::vchk12(tc3, ftc3, [2.6589670819169946, 0.0, -1.6459278069146803, 0.7431641870407133, -1.21516887151373, 1.0066046203353818, 0.0, 0.0, -0.329483540958497, 0.8022545575574112, -0.8211521765556657, -2.7755575615628914e-17, 0.0, 0.0, 0.0, 0.0, -0.329483540958497, -0.8022545575574112, 2.6589670819169946, 0.0, -1.6459278069146803, 0.7431641870407133, -1.21516887151373, 1.0066046203353818, 0.0, 0.0, -0.329483540958497, 0.8022545575574112, -0.8211521765556657, -2.7755575615628914e-17, 0.0, 0.0, 0.0, 0.0, -0.329483540958497, -0.8022545575574112]);
    common::vchk12(zc3, ftc3, [0.7271159090099086, 0.0, 0.2822039309633517, -0.3138569291140597, 0.5131963797757693, -0.172588238433742, 0.5283942772676083, 0.0, 0.24720268043583152, 0.4163557825486023, -0.680795166465797, -0.15118242686017014, 0.43829435613014156, 0.0, -0.7661870265023007, 0.018732510220669858, -0.030630059551327294, 0.4685791185240134, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.610101844710567e-17, 0.5217367310758697, -0.853106548706705, 0.0, 0.0, 0.0, 0.853106548706705, 0.0, -3.610101844710567e-17, -0.5217367310758697]);
    assert(tc3.shape[0] == 2 && zc3.shape[0] == 2);
    // The real block form of a stack of two sets.
    let ws = scientific::CArr { ndim: 2, shape: [2, 3], data: [1.0, 0.0, 4.0, 5.0, 4.0, -5.0, 2.0, 3.0, 2.0, -3.0, 7.0, 0.0] };
    let vs = scientific::CArr { ndim: 3, shape: [2, 3, 3], data: [1.0, 0.0, -0.019055398349167637, -0.4001633653325205, -0.019055398349167637, 0.4001633653325205, 0.0, 0.0, 0.0, -0.64788354387169989, 0.0, 0.64788354387169989, 0.0, 0.0, 0.6478835438717, 0.0, 0.6478835438717, -0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0, 1.0] };
    let sw, sv, fsw = scientific::cdf2rdf(ws, vs);
    common::vchk12(sw, fsw, [1.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, -5.0, 4.0, 2.0, 3.0, 0.0, -3.0, 2.0, 0.0, 0.0, 0.0, 7.0]);
    common::vchk12(sv, fsw, [1.0, 0.4001633653325205, -0.019055398349167637, 0.0, 0.64788354387169989, 0.0, 0.0, 0.0, 0.6478835438717, 0.0, 1.5, 3.0, 2.5, 2.0, 6.0, 0.0, 7.5, 9.0]);
    assert(sw.ndim == 3 && sw.shape[0] == 2 && sw.shape[1] == 3 && sw.shape[2] == 3);
    assert(sv.ndim == 3 && sv.shape[0] == 2 && sv.shape[1] == 3 && sv.shape[2] == 3);
    // One set of eigenvalues over two sets of eigenvectors.
    let wb = scientific::CArr { ndim: 2, shape: [1, 3], data: [1.0, 0.0, 4.0, 5.0, 4.0, -5.0] };
    let tw, tv, ftw = scientific::cdf2rdf(wb, vs);
    common::vchk12(tw, ftw, [1.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, -5.0, 4.0]);
    common::vchk12(tv, ftw, [1.0, 0.4001633653325205, -0.019055398349167637, 0.0, 0.64788354387169989, 0.0, 0.0, 0.0, 0.6478835438717, 1.0, 0.0, 2.5, 4.0, -2.5, 3.0, 7.0, 0.5, 8.5]);
    assert(tw.shape[0] == 1 && tv.shape[0] == 2);
    // A real value between two complex values. The pair stays whole.
    let w2 = scientific::CArr { ndim: 1, shape: [3], data: [1.0, 1.0, 2.0, 0.0, 1.0, -1.0] };
    let v2 = scientific::CArr { ndim: 2, shape: [3, 3], data: [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0, 1.0] };
    let uw2, uv2, fuw = scientific::cdf2rdf(w2, v2);
    common::vchk12(uw2, fuw, [1.0, 0.0, 1.0, 0.0, 2.0, 0.0, -1.0, 0.0, 1.0]);
    common::vchk12(uv2, fuw, [0.0, 2.0, 2.0, 0.0, 0.0, 5.0, 0.5, 8.0, 8.0]);
    // The complex values pair in the order they appear.
    let w3 = scientific::CArr { ndim: 1, shape: [4], data: [1.0, 1.0, 2.0, 1.0, 1.0, -1.0, 2.0, -1.0] };
    let v3 = scientific::CArr { ndim: 2, shape: [4, 4], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 0.0] };
    let uw3, uv3, fuw3 = scientific::cdf2rdf(w3, v3);
    common::vchk12(uw3, fuw3, [1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 2.0]);
    common::vchk12(uv3, fuw3, [0.0, 0.5, 0.0, 0.5, -0.5, 1.0, 0.0, 0.5, 0.0, 1.5, 0.0, 2.0, 1.0, 0.0, 0.0, 2.5]);
    // The counts of the diagonals of every matrix of a stack.
    let ba = scientific::Arr { ndim: 3, shape: [2, 3, 4], data: [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0] };
    let blo, bhi, fbl = scientific::bandwidth_stack(ba);
    common::vchk12(blo, fbl, [2.0, 1.0]);
    common::vchk12(bhi, fbl, [2.0, 2.0]);
    assert(blo.ndim == 1 && blo.shape[0] == 2);
    // A matrix of zeros gives zero and zero.
    let bz2 = scientific::Arr { ndim: 3, shape: [2, 3, 3], data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };
    let zlo, zhi, fbl2 = scientific::bandwidth_stack(bz2);
    common::vchk12(zlo, fbl2, [0.0, 0.0]);
    common::vchk12(zhi, fbl2, [0.0, 0.0]);
    // A matrix without a batch gives the shape [1].
    let bm = scientific::Arr { ndim: 2, shape: [3, 4], data: [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0] };
    let mlo, mhi, fbl3 = scientific::bandwidth_stack(bm);
    common::vchk12(mlo, fbl3, [2.0]);
    common::vchk12(mhi, fbl3, [2.0]);
    assert(mlo.ndim == 1 && mlo.shape[0] == 1);
    // A value that is not finite counts as a value that is not zero.
    let bn = scientific::Arr { ndim: 3, shape: [1, 2, 2], data: [0.0, 0.0, 0.0 / 0.0, 0.0] };
    let nlo, nhi, fbl4 = scientific::bandwidth_stack(bn);
    common::vchk12(nlo, fbl4, [1.0]);
    common::vchk12(nhi, fbl4, [0.0]);
    // A batch that does not broadcast faults.
    let zbad = scientific::Arr { ndim: 3, shape: [3, 3, 3], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
    let h1, h2, fh1 = scientific::rsf2csf(ts, zbad);
    assert(fh1.kind == scientific::F_SHAPE);
    // A side above the limit faults.
    let big2 = scientific::Arr { ndim: 2, shape: [513, 513], data: [1.0] };
    let h3, h4, fh2 = scientific::rsf2csf(big2, pz2);
    assert(fh2.kind == scientific::F_LIMIT);
    // A value that is not finite faults.
    let nan2 = 0.0 / 0.0;
    let tnan = scientific::Arr { ndim: 3, shape: [1, 2, 2], data: [nan2, 0.0, 0.0, 1.0] };
    let zid = scientific::Arr { ndim: 3, shape: [1, 2, 2], data: [1.0, 0.0, 0.0, 1.0] };
    let h5, h6, fh3 = scientific::rsf2csf(tnan, zid);
    assert(fh3.kind == scientific::F_VALUE);
    // A count of dimensions that lies and a buffer of the wrong length fault.
    let lie2 = scientific::Arr { ndim: 4, shape: [2, 3, 3], data: [1.0] };
    let h7, h8, fh4 = scientific::rsf2csf(lie2, zs);
    assert(fh4.kind == scientific::F_SHAPE);
    let short2 = scientific::Arr { ndim: 3, shape: [2, 3, 3], data: [1.0] };
    let h9, h10, fh5 = scientific::rsf2csf(short2, zs);
    assert(fh5.kind == scientific::F_SHAPE);
    // The eigenvectors must hold one dimension more than the eigenvalues.
    let wshort = scientific::CArr { ndim: 2, shape: [2, 3], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] };
    let h11, h12, fh6 = scientific::cdf2rdf(ws, wshort);
    assert(fh6.kind == scientific::F_SHAPE);
    // A matrix with one dimension, a matrix that is not square and two
    // different sides fault.
    let one2 = scientific::Arr { ndim: 1, shape: [3], data: [1.0, 2.0, 3.0] };
    let h13, h14, fh7 = scientific::rsf2csf(one2, pz2);
    assert(fh7.kind == scientific::F_SHAPE);
    let nsq2 = scientific::Arr { ndim: 3, shape: [1, 2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
    let h15, h16, fh8 = scientific::rsf2csf(nsq2, zid);
    assert(fh8.kind == scientific::F_SHAPE);
    let two2 = scientific::Arr { ndim: 3, shape: [1, 2, 2], data: [1.0, 0.0, 0.0, 1.0] };
    let three2 = scientific::Arr { ndim: 3, shape: [1, 3, 3], data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
    let h17, h18, fh9 = scientific::rsf2csf(two2, three2);
    assert(fh9.kind == scientific::F_SHAPE);
    // The batches of the eigenvalues and the eigenvectors must
    // broadcast.
    var vbad2: f64[>..] = [];
    var q: i64 = 0;
    while q < 54 { vbad2.push(1.0); vbad2.push(0.0); q += 1; }
    let vbad3 = scientific::CArr { ndim: 3, shape: [3, 3, 3], data: copy(vbad2) };
    let h19, h20, fh10 = scientific::cdf2rdf(ws, vbad3);
    assert(fh10.kind == scientific::F_SHAPE);
    // A side above the limit and a value that is not finite fault.
    let wbig2 = scientific::CArr { ndim: 1, shape: [513], data: [] };
    let vbig2 = scientific::CArr { ndim: 2, shape: [513, 513], data: [] };
    let h21, h22, fh11 = scientific::cdf2rdf(wbig2, vbig2);
    assert(fh11.kind == scientific::F_LIMIT);
    let wnan = scientific::CArr { ndim: 1, shape: [3], data: [nan2, 0.0, 1.0, 0.0, 1.0, 0.0] };
    let h23, h24, fh12 = scientific::cdf2rdf(wnan, v2);
    assert(fh12.kind == scientific::F_VALUE);
    // A count of dimensions that lies, a buffer of the wrong length and
    // one dimension fault.
    let blie = scientific::Arr { ndim: 3, shape: [2, 3], data: [1.0] };
    let c1, c2, fc1 = scientific::bandwidth_stack(blie);
    assert(fc1.kind == scientific::F_SHAPE);
    let bshort = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let c3, c4, fc2 = scientific::bandwidth_stack(bshort);
    assert(fc2.kind == scientific::F_SHAPE);
    let bone = scientific::Arr { ndim: 1, shape: [3], data: [1.0, 2.0, 3.0] };
    let c5, c6, fc3 = scientific::bandwidth_stack(bone);
    assert(fc3.kind == scientific::F_SHAPE);
    // Two valid batches whose broadcast holds far more slices than the
    // limit fault before any index exists. The shapes lie about no
    // buffer: each operand holds its own batch of 1 by 1 matrices.
    var hostile_a: f64[>..] = [];
    var hostile_i: i64 = 0;
    while hostile_i < 2097152 { hostile_a.push(1.0); hostile_i += 1; }
    let hug_t = scientific::Arr { ndim: 4, shape: [2097152, 1, 1, 1], data: copy(hostile_a) };
    let hug_z = scientific::Arr { ndim: 4, shape: [1, 2097152, 1, 1], data: copy(hostile_a) };
    let h25, h26, fh13 = scientific::rsf2csf(hug_t, hug_z);
    assert(fh13.kind == scientific::F_LIMIT);
    // The matrix form of the bandwidth checks the buffer too.
    let blie2 = scientific::Arr { ndim: 2, shape: [1000000, 1000000], data: [1.0] };
    let bl1, bl2, fbl5 = scientific::bandwidth(blie2);
    assert(fbl5.kind != scientific::F_NONE);
    // A stack of many sets and many forms does not cost much.
    var wdata: f64[>..] = [];
    var vdata: f64[>..] = [];
    var i: i64 = 0;
    while i < 4096 {
        var j: i64 = 0;
        while j < 16 { wdata.push(1.0); wdata.push(0.0); j += 1; }
        j = 0;
        while j < 256 { vdata.push(1.0); vdata.push(0.0); j += 1; }
        i += 1;
    }
    let wbig = scientific::CArr { ndim: 2, shape: [4096, 16], data: copy(wdata) };
    let vbig = scientific::CArr { ndim: 3, shape: [4096, 16, 16], data: copy(vdata) };
    let bw2, bv2, fbw = scientific::cdf2rdf(wbig, vbig);
    assert(fbw.kind == scientific::F_NONE);
    assert(bw2.data.len == 4096 * 256 && bv2.data.len == 4096 * 256);
    var tdata: f64[>..] = [];
    var zdata: f64[>..] = [];
    i = 0;
    while i < 4096 {
        var j: i64 = 0;
        while j < 256 { tdata.push(1.0); zdata.push(0.0); j += 1; }
        i += 1;
    }
    let tbig = scientific::Arr { ndim: 3, shape: [4096, 16, 16], data: copy(tdata) };
    let zbig = scientific::Arr { ndim: 3, shape: [4096, 16, 16], data: copy(zdata) };
    let bt2, bz2, fbt = scientific::rsf2csf(tbig, zbig);
    assert(fbt.kind == scientific::F_NONE);
    assert(bt2.data.len == 2 * 4096 * 256 && bz2.data.len == 2 * 4096 * 256);
}
