// The tests of the group carray. Run them with SCIENTIFIC_SUITES=carray.
namespace carray_tests;

import scientific from "../..";
import os;
import common;
import common_2;
import common_3;

fn test_carray() {
    let re = common::av([1.0, -2.0, 3.5, 0.0]);
    let im = common::av([0.5, 2.0, -1.25, 4.0]);
    let z, zf = scientific::complex_array(re, im);
    assert(zf.kind == scientific::F_NONE);
    assert(z.ndim == 1 && z.shape[0] == 4 && z.data.len == 8);
    // The parts come back as they went in.
    let zr = scientific::real(z);
    let zi = scientific::imag(z);
    assert(common::all_near(zr, [1.0, -2.0, 3.5, 0.0]) && common::all_near(zi, [0.5, 2.0, -1.25, 4.0]));
    // The size and the angle of each element.
    let za = scientific::cabs(z);
    assert(common::all_near(za, [1.118033988749895, 2.8284271247461903, 3.7165171868296265, 4.0]));
    let zang = scientific::angle(z);
    assert(common::all_near(zang, [0.4636476090008061, 2.356194490192345, -0.3430239404207034, 1.5707963267948966]));
    let zdeg = scientific::angle(z, true);
    assert(common::all_near(zdeg, [26.56505117707799, 135.0, -19.65382405805331, 90.0]));
    // The conjugate turns the imaginary part around.
    let zc = scientific::conj(z);
    assert(common::all_near_c(zc, [1.0, -2.0, 3.5, 0.0], [-0.5, -2.0, 1.25, -4.0]));
    // The tests of the parts.
    let zr0, zrf = scientific::complex_from_values([1.0, 2.0], [0.0, 0.0]);
    assert(zrf.kind == scientific::F_NONE);
    let ic0 = scientific::iscomplex(zr0);
    let ir0 = scientific::isreal(zr0);
    assert(common::all_near(ic0, [0.0, 0.0]) && common::all_near(ir0, [1.0, 1.0]));
    let ic1 = scientific::iscomplex(z);
    let ir1 = scientific::isreal(z);
    assert(common::all_near(ic1, [1.0, 1.0, 1.0, 1.0]) && common::all_near(ir1, [0.0, 0.0, 0.0, 0.0]));
    // The parts of different shapes make a fault.
    let zbad, zbadf = scientific::complex_array(re, common::av([1.0, 2.0]));
    assert(zbadf.kind == scientific::F_SHAPE);
    // The elementwise operations broadcast like the real ones.
    let a = common::mx(2, 2, [1.0, 3.0, 0.5, -2.0]);
    let ai = common::mx(2, 2, [2.0, -1.0, 0.5, 0.0]);
    let b = common::av([1.0, 2.0]);
    let bi = common::av([1.0, -0.5]);
    let ca, caf = scientific::complex_array(a, ai);
    let cb, cbf = scientific::complex_array(b, bi);
    assert(caf.kind == scientific::F_NONE && cbf.kind == scientific::F_NONE);
    let sum, sf = scientific::cadd(ca, cb);
    assert(sf.kind == scientific::F_NONE);
    assert(common::all_near_c(sum, [2.0, 5.0, 1.5, 0.0], [3.0, -1.5, 1.5, -0.5]));
    let dif, df = scientific::csub(ca, cb);
    assert(df.kind == scientific::F_NONE);
    assert(common::all_near_c(dif, [0.0, 1.0, -0.5, -4.0], [1.0, -0.5, -0.5, 0.5]));
    let pro, pf = scientific::cmul(ca, cb);
    assert(pf.kind == scientific::F_NONE);
    assert(common::all_near_c(pro, [-1.0, 5.5, 0.0, -4.0], [3.0, -3.5, 1.0, 1.0]));
    let quo, qf = scientific::cdiv(ca, cb);
    assert(qf.kind == scientific::F_NONE);
    assert(common::all_near_c(quo, [1.5, 1.5294117647058822, 0.5, -0.9411764705882353], [0.5, -0.11764705882352941, 0.0, -0.23529411764705882]));
    let pw, pwf = scientific::cpower(ca, cb);
    assert(pwf.kind == scientific::F_NONE);
    assert(common::all_near_c(pw, [-0.24720004426291722, 2.932609903259764, 0.2918503793793073, 18.09782608169716], [0.6964504870825432, -7.992978025796342, 0.13697862691506052, -6.536036508550907]));
    // The functions of one operand.
    let u, uf = scientific::complex_from_values([0.3, -1.0, 2.0, -0.5], [0.4, 2.0, -1.0, -0.5]);
    assert(uf.kind == scientific::F_NONE);
    let ue = scientific::cexp(u);
    assert(common::all_near_c(ue, [1.2433022950695027, -0.1530918656742263, 3.992324048441272, 0.5322807302156708], [0.5256597791969788, 0.33451182923926226, -6.217676312367968, -0.29078628821269187]));
    let ul = scientific::clog(u);
    assert(common::all_near_c(ul, [-0.6931471805599453, 0.8047189562170503, 0.8047189562170503, -0.34657359027997264], [0.9272952180016123, 2.0344439357957027, -0.4636476090008061, -2.356194490192345]));
    let us = scientific::csqrt(u);
    assert(common::all_near_c(us, [0.6324555320336759, 0.7861513777574233, 1.455346690225355, 0.3217971264527913], [0.31622776601683794, 1.272019649514069, -0.34356074972251244, -0.7768869870150187]));
    let usn = scientific::csin(u);
    assert(common::all_near_c(usn, [0.31947873074156474, -3.165778513216168, 1.4031192506220405, -0.5406126857131534], [0.3924066848326388, 1.9596010414216063, 0.4890562590412937, -0.4573041531842493]));
    let uc = scientific::ccos(u);
    assert(common::all_near_c(uc, [1.0327878842028413, 2.0327230070196656, -0.64214812471552, 0.9895848833999199], [-0.12138561220787392, 3.0518977991518, 1.0686074213827783, -0.24982639750046154]));
    let ush = scientific::csinh(u);
    assert(common::all_near_c(ush, [0.2804817641764754, 0.4890562590412937, 1.9596010414216063, -0.4573041531842493], [0.4070739913234487, 1.4031192506220405, -3.165778513216168, -0.5406126857131534]));
    let uch = scientific::ccosh(u);
    assert(common::all_near_c(uch, [0.9628205308930272, -0.64214812471552, 2.0327230070196656, 0.9895848833999199], [0.11858578787353008, -1.0686074213827783, -3.0518977991518, 0.24982639750046154]));
    // The matrices.
    let m1, m1f = scientific::complex_array(common::mx(2, 2, [1.0, 0.0, 2.0, 1.0]), common::mx(2, 2, [2.0, -1.0, 0.0, 1.0]));
    let m2, m2f = scientific::complex_array(common::mx(2, 2, [0.5, 1.0, -1.0, 0.5]), common::mx(2, 2, [0.0, 1.0, 2.0, -0.5]));
    assert(m1f.kind == scientific::F_NONE && m2f.kind == scientific::F_NONE);
    let mm, mmf = scientific::cmatmul(m1, m2);
    assert(mmf.kind == scientific::F_NONE);
    assert(common::all_near_c(mm, [2.5, -1.5, -2.0, 3.0], [2.0, 2.5, 1.0, 2.0]));
    let tr, trf = scientific::ctranspose(m1);
    assert(trf.kind == scientific::F_NONE);
    assert(common::all_near_c(tr, [1.0, 2.0, 0.0, 1.0], [2.0, 0.0, -1.0, 1.0]));
    let ad, adf = scientific::adjoint(m1);
    assert(adf.kind == scientific::F_NONE);
    assert(common::all_near_c(ad, [1.0, 2.0, 0.0, 1.0], [-2.0, 0.0, 1.0, -1.0]));
    // The solve of a complex system.
    let rhs, rhsf = scientific::complex_from_values([1.0, 2.0], [1.0, -1.0]);
    assert(rhsf.kind == scientific::F_NONE);
    let sol, solf = scientific::csolve(m1, rhs);
    assert(solf.kind == scientific::F_NONE);
    assert(common::all_near_c(sol, [0.7307692307692308, 0.11538461538461536], [-0.34615384615384615, -0.423076923076923]));
    // The solve with a matrix on the right.
    let bm, bmf = scientific::complex_array(common::mx(2, 2, [1.0, 2.0, 0.0, 1.0]), common::mx(2, 2, [0.0, 1.0, -1.0, 0.0]));
    assert(bmf.kind == scientific::F_NONE);
    let solm, solmf = scientific::csolve(m1, bm);
    assert(solmf.kind == scientific::F_NONE);
    assert(common::all_near_c(solm, [0.11538461538461539, 0.7307692307692308, -0.19230769230769232, 0.11538461538461534], [-0.42307692307692313, -0.34615384615384615, 0.03846153846153848, 0.5769230769230771]));
    // The inverse and the determinant.
    let iv, ivf = scientific::cinv(m1);
    assert(ivf.kind == scientific::F_NONE);
    assert(common::all_near_c(iv, [0.15384615384615383, 0.19230769230769235, 0.0769230769230769, 0.3461538461538462], [-0.23076923076923075, -0.03846153846153845, 0.3846153846153847, -0.2692307692307692]));
    let dr, di, dtf = scientific::cdet(m1);
    assert(dtf.kind == scientific::F_NONE);
    assert(common::near(dr, -1.0) && common::near(di, 5.0));
    // The transforms of a complex signal.
    let sig, sigf = scientific::complex_from_values([1.0, 2.0, 0.0, -1.0], [0.0, -1.0, 0.5, 2.0]);
    assert(sigf.kind == scientific::F_NONE);
    let ft, ftf = scientific::cfft(sig);
    assert(ftf.kind == scientific::F_NONE);
    assert(common::all_near_c(ft, [2.0, -2.0, 0.0, 4.0], [1.5, -3.5, -0.5, 2.5]));
    let itf, itff = scientific::cifft(sig);
    assert(itff.kind == scientific::F_NONE);
    assert(common::all_near_c(itf, [0.5, 1.0, 0.0, -0.5], [0.375, 0.625, -0.125, -0.875]));
    // The eigenvalues of a real matrix may hold complex pairs.
    let ea, eaf = scientific::eigvals_complex(common::mx(3, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]));
    assert(eaf.kind == scientific::F_NONE);
    assert(common::all_near_c(ea, [-0.9057401795217577, 0.1982468633970094, 16.707493316124747], [0.0, 0.0, 0.0]));
    let eb, ebf = scientific::eigvals_complex(common::mx(2, 2, [0.0, -1.0, 1.0, 0.0]));
    assert(ebf.kind == scientific::F_NONE);
    assert(common::all_near_c(eb, [0.0, 0.0], [-1.0, 1.0]));
    let ec, ecf = scientific::eigvals_complex(common::mx(3, 3, [2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, -4.0, 3.0]));
    assert(ecf.kind == scientific::F_NONE);
    assert(common::all_near_c(ec, [2.0, 3.0, 3.0], [0.0, -4.0, 4.0]));
    // A matrix that is not square makes a fault.
    let c23, c23f = scientific::complex_array(common::mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), common::mx(2, 3, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(c23f.kind == scientific::F_NONE);
    let bad, badf = scientific::csolve(c23, rhs);
    assert(badf.kind == scientific::F_SHAPE);
    let bad2, bad2f = scientific::cmatmul(c23, c23);
    assert(bad2f.kind == scientific::F_SHAPE);
    let bad3, bad3f = scientific::ctranspose(rhs);
    assert(bad3f.kind == scientific::F_SHAPE);
    let bad4, bad4f = scientific::eigvals_complex(common::mx(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert(bad4f.kind == scientific::F_SHAPE);
    // The buffer of the operand is shorter than the matrix.
    let short = scientific::Arr { ndim: 2, shape: [2, 2], data: [1.0] };
    let bad5, bad5f = scientific::eigvals_complex(short);
    assert(bad5f.kind == scientific::F_SHAPE);
    return;
}

fn test_qz_c_time() {
    // A complex pair of side 256 runs inside the limit of time.
    let ra, rb = common_2::qt_cpair(256, 11);
    let a = scientific::CArr { ndim: 2, shape: [256, 256], data: copy(ra) };
    let b = scientific::CArr { ndim: 2, shape: [256, 256], data: copy(rb) };
    let t0 = clock();
    let aa, bb, q, z, f = scientific::qz_c(copy(a), copy(b));
    let t1 = clock();
    assert(f.kind == scientific::F_NONE);
    assert(t1 - t0 < 600.0);
}

// test_cdot pins the products of complex arrays against NumPy.
fn test_cdot() {
    let a, f0 = scientific::complex_from_values([1.0, 3.0], [2.0, -1.0]);
    let b, f1 = scientific::complex_from_values([2.0, -1.0], [-1.0, 0.5]);
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE);
    let p, pf = scientific::cdot(a, b);
    assert(pf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(p, [1.5], [5.5], 1.0e-12));
    let m, mf = scientific::complex_array(common::mx(2, 2, [2.0, 1.0, 1.0, 3.0]), common::mx(2, 2, [0.0, 1.0, -1.0, 0.0]));
    assert(mf.kind == scientific::F_NONE);
    let mm, mmf = scientific::cdot(m, m);
    assert(mmf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(mm, [6.0, 5.0, 5.0, 11.0], [0.0, 5.0, -5.0, 0.0], 1.0e-12));
    let v, vf = scientific::complex_from_values([1.0, 2.0], [0.0, 0.0]);
    assert(vf.kind == scientific::F_NONE);
    let mv, mvf = scientific::cdot(m, v);
    assert(mvf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(mv, [4.0, 7.0], [2.0, -1.0], 1.0e-12));
    let vm, vmf = scientific::cdot(v, m);
    assert(vmf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(vm, [4.0, 7.0], [-2.0, 1.0], 1.0e-12));
    // The vectors of one element.
    let one, f2 = scientific::complex_from_values([1.0], [2.0]);
    let two, f3 = scientific::complex_from_values([3.0], [-1.0]);
    assert(f2.kind == scientific::F_NONE && f3.kind == scientific::F_NONE);
    let p1, p1f = scientific::cdot(one, two);
    assert(p1f.kind == scientific::F_NONE);
    assert(common_2::all_close_c(p1, [5.0], [5.0], 1.0e-12));
    let q1, q2, q3 = scientific::cvdot(one, two);
    assert(q3.kind == scientific::F_NONE);
    assert(common::close(q1, 1.0, 1.0e-12) && common::close(q2, -7.0, 1.0e-12));
    // The matrix of one row and one column.
    let tiny, f4 = scientific::complex_array(common::mx(1, 1, [2.0]), common::mx(1, 1, [1.0]));
    assert(f4.kind == scientific::F_NONE);
    let p2, p2f = scientific::cdot(tiny, tiny);
    assert(p2f.kind == scientific::F_NONE);
    assert(common_2::all_close_c(p2, [3.0], [4.0], 1.0e-12));
    // The scalar product conjugates its first operand.
    let cr, ci, cf = scientific::cvdot(a, b);
    assert(cf.kind == scientific::F_NONE);
    assert(common::close(cr, -3.5, 1.0e-12) && common::close(ci, -4.5, 1.0e-12));
    let cr2, ci2, cf2 = scientific::cvdot(b, a);
    assert(cf2.kind == scientific::F_NONE);
    assert(common::close(cr2, -3.5, 1.0e-12) && common::close(ci2, 4.5, 1.0e-12));
    return;
}

// test_cdot_hostile covers the faults of the products.
fn test_cdot_hostile() {
    let a, f0 = scientific::complex_from_values([1.0, 3.0], [2.0, -1.0]);
    let c, f1 = scientific::complex_from_values([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
    let m, mf = scientific::complex_array(common::mx(2, 2, [2.0, 1.0, 1.0, 3.0]), common::mx(2, 2, [0.0, 1.0, -1.0, 0.0]));
    assert(f0.kind == scientific::F_NONE && f1.kind == scientific::F_NONE && mf.kind == scientific::F_NONE);
    // The lengths do not agree.
    let r1, r1f = scientific::cdot(a, c);
    assert(r1f.kind == scientific::F_SHAPE);
    // The inner sides do not agree.
    let r2, r2f = scientific::cdot(m, c);
    assert(r2f.kind == scientific::F_SHAPE);
    // An operand holds more than two dimensions.
    let three = scientific::CArr { ndim: 3, shape: [2, 2, 2], data: [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] };
    let r3, r3f = scientific::cdot(three, a);
    assert(r3f.kind == scientific::F_SHAPE);
    // A real array reaches the call.
    let r4, r4f = scientific::cdot(m, common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]));
    assert(r4f.kind == scientific::F_SHAPE);
    // The buffer does not match the shape.
    let short = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0] };
    let r5, r5f = scientific::cdot(short, m);
    assert(r5f.kind == scientific::F_SHAPE);
    // An axis holds a length below one.
    let empty = scientific::CArr { ndim: 1, shape: [0], data: [] };
    let r6, r6f = scientific::cdot(empty, a);
    assert(r6f.kind == scientific::F_VALUE);
    // A shape holds more elements than the limit.
    let huge = scientific::CArr { ndim: 2, shape: [268435456, 2], data: [1.0] };
    let r7, r7f = scientific::cdot(huge, m);
    assert(r7f.kind == scientific::F_LIMIT);
    // The counts of the elements do not agree.
    let s1, s2, s3 = scientific::cvdot(a, c);
    assert(s3.kind == scientific::F_SHAPE);
    // A real array reaches the call.
    let s4, s5, s6 = scientific::cvdot(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]), common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]));
    assert(s6.kind == scientific::F_SHAPE);
    return;
}

// test_ccholesky pins the factor of a complex matrix against NumPy.
fn test_ccholesky() {
    let a, f0 = scientific::complex_array(common::mx(2, 2, [2.0, 1.0, 1.0, 3.0]), common::mx(2, 2, [0.0, 1.0, -1.0, 0.0]));
    assert(f0.kind == scientific::F_NONE);
    let l, lf = scientific::ccholesky(a);
    assert(lf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(l, [1.4142135623730951, 0.0, 0.7071067811865475, 1.4142135623730951], [0.0, 0.0, -0.7071067811865475, 0.0], 1.0e-12));
    // The upper triangle of the operand stays out.
    let b, f1 = scientific::complex_array(common::mx(2, 2, [4.0, 1.0, 1.0, 2.0]), common::mx(2, 2, [0.0, 9.0, 1.0, 0.0]));
    assert(f1.kind == scientific::F_NONE);
    let lb, lbf = scientific::ccholesky(b);
    assert(lbf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(lb, [2.0, 0.0, 0.5, 1.224744871391589], [0.0, 0.0, 0.5, 0.0], 1.0e-12));
    // The imaginary part of the diagonal stays out.
    let c, f2 = scientific::complex_array(common::mx(2, 2, [4.0, 0.0, 0.0, 2.0]), common::mx(2, 2, [1.0, 0.0, 0.0, 0.0]));
    assert(f2.kind == scientific::F_NONE);
    let lc, lcf = scientific::ccholesky(c);
    assert(lcf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(lc, [2.0, 0.0, 0.0, 1.4142135623730951], [0.0, 0.0, 0.0, 0.0], 1.0e-12));
    // The factor and its conjugate transpose give the matrix back.
    let la, laf = scientific::adjoint(l);
    assert(laf.kind == scientific::F_NONE);
    let pr, prf = scientific::cmatmul(l, la);
    assert(prf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(pr, [2.0, 1.0, 1.0, 3.0], [0.0, 1.0, -1.0, 0.0], 1.0e-12));
    // The matrix of one by one.
    let one, f2 = scientific::complex_array(common::mx(1, 1, [4.0]), common::mx(1, 1, [0.0]));
    assert(f2.kind == scientific::F_NONE);
    let lo, lof = scientific::ccholesky(one);
    assert(lof.kind == scientific::F_NONE);
    assert(common_2::all_close_c(lo, [2.0], [0.0], 1.0e-12));
    // The factor of 20 random Hermitian matrices gives the matrix
    // back. The factor of a positive definite matrix is unique, so the
    // property holds the value of NumPy.
    var g = scientific::default_rng(295);
    let idm = scientific::complex_real(common::mx(4, 4, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]));
    let eight = scientific::complex_real(common::mx(4, 4, [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]));
    var t: i64 = 0;
    while t < 20 {
        let re, f2 = scientific::randn(g, [4, 4]);
        let im, f3 = scientific::randn(g, [4, 4]);
        assert(f2.kind == scientific::F_NONE && f3.kind == scientific::F_NONE);
        let mm, f4 = scientific::complex_array(re, im);
        assert(f4.kind == scientific::F_NONE);
        let ad, f5 = scientific::adjoint(mm);
        assert(f5.kind == scientific::F_NONE);
        let w, f6 = scientific::cmatmul(mm, ad);
        assert(f6.kind == scientific::F_NONE);
        let di, f7 = scientific::cmul(idm, eight);
        assert(f7.kind == scientific::F_NONE);
        let h, f8 = scientific::cadd(w, di);
        assert(f8.kind == scientific::F_NONE);
        let lf2, f9 = scientific::ccholesky(h);
        assert(f9.kind == scientific::F_NONE);
        let lf2a, f10 = scientific::adjoint(lf2);
        assert(f10.kind == scientific::F_NONE);
        let pr2, f11 = scientific::cmatmul(lf2, lf2a);
        assert(f11.kind == scientific::F_NONE);
        var i: i64 = 0;
        while i < h.data.len {
            assert(common::close(pr2.data[i], h.data[i], 1.0e-12));
            i += 1;
        }
        t += 1;
    }
    return;
}

// test_cfftn pins the transform of a complex array against NumPy.
fn test_cfftn() {
    let a, f0 = scientific::complex_array(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]), common::mx(2, 2, [1.0, 0.0, -1.0, 2.0]));
    assert(f0.kind == scientific::F_NONE);
    let f, ff = scientific::cfftn(a);
    assert(ff.kind == scientific::F_NONE);
    assert(common_2::all_close_c(f, [10.0, -2.0, -4.0, 0.0], [2.0, -2.0, 0.0, 4.0], 1.0e-12));
    let g, gf = scientific::cifftn(a);
    assert(gf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(g, [2.5, -0.5, -1.0, 0.0], [0.5, -0.5, 0.0, 1.0], 1.0e-12));
    // The matrix of three columns.
    let b, f1 = scientific::complex_array(common::mx(2, 3, [1.0, 2.0, 0.0, 3.0, 0.0, 1.0]), common::mx(2, 3, [2.0, 0.0, -1.0, -4.0, 0.0, 1.0]));
    assert(f1.kind == scientific::F_NONE);
    let fb, fbf = scientific::cfftn(b);
    assert(fbf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(fb, [7.0, 2.5, 2.5, -1.0, -0.7679491924311228, -4.232050807568877], [-2.0, -2.866025403784439, -1.1339745962155616, 4.0, 4.401923788646684, 9.598076211353316], 1.0e-12));
    // The axis of a length of one gives the array back.
    let one, f2 = scientific::complex_array(common::mx(1, 1, [1.0]), common::mx(1, 1, [2.0]));
    assert(f2.kind == scientific::F_NONE);
    let fo, fof = scientific::cfftn(one);
    assert(fof.kind == scientific::F_NONE);
    assert(common_2::all_close_c(fo, [1.0], [2.0], 1.0e-12));
    // The array of zeros gives zero in every element.
    let zero, f3 = scientific::complex_array(common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]), common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(f3.kind == scientific::F_NONE);
    let fz, fzf = scientific::cfftn(zero);
    assert(fzf.kind == scientific::F_NONE);
    assert(common_2::all_close_c(fz, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 1.0e-12));
    // The round trip of 20 random arrays of one, two and three
    // dimensions gives the array back.
    var g2 = scientific::default_rng(296);
    var t: i64 = 0;
    while t < 20 {
        var sh: i64[..8] = [];
        sh.push(3);
        var k: i64 = 1;
        while k < 1 + (t % 3) { sh.push(3); k += 1; }
        let re, f2 = scientific::randn(g2, sh);
        let im, f3 = scientific::randn(g2, sh);
        assert(f2.kind == scientific::F_NONE && f3.kind == scientific::F_NONE);
        let x, f4 = scientific::complex_array(re, im);
        assert(f4.kind == scientific::F_NONE);
        let ft, f5 = scientific::cfftn(x);
        assert(f5.kind == scientific::F_NONE);
        let iv, f6 = scientific::cifftn(ft);
        assert(f6.kind == scientific::F_NONE);
        var i: i64 = 0;
        while i < x.data.len {
            assert(common::close(iv.data[i], x.data[i], 1.0e-12));
            i += 1;
        }
        t += 1;
    }
    return;
}

// test_cfftn_hostile covers the faults of the transform.
fn test_cfftn_hostile() {
    // The buffer does not match the shape.
    let short = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0] };
    let f1, f1f = scientific::cfftn(short);
    assert(f1f.kind == scientific::F_SHAPE);
    let f2, f2f = scientific::cifftn(short);
    assert(f2f.kind == scientific::F_SHAPE);
    // A value is not finite.
    let nan = 0.0 / 0.0;
    let bad, f3 = scientific::complex_array(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]), common::mx(2, 2, [nan, 0.0, 0.0, 0.0]));
    assert(f3.kind == scientific::F_NONE);
    let f4, f4f = scientific::cfftn(bad);
    assert(f4f.kind == scientific::F_VALUE);
    let f5, f5f = scientific::cifftn(bad);
    assert(f5f.kind == scientific::F_VALUE);
    // A shape holds more elements than the limit.
    let huge = scientific::CArr { ndim: 1, shape: [268435457], data: [1.0] };
    let f6, f6f = scientific::cfftn(huge);
    assert(f6f.kind == scientific::F_LIMIT);
    return;
}

// test_cqr pins the factors against NumPy and tests their properties.
fn test_cqr() {
    let a1, f1 = scientific::complex_array(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]), common::mx(2, 2, [1.0, 0.0, -2.0, 1.0]));
    assert(f1.kind == scientific::F_NONE);
    let q, r, f = scientific::cqr(a1);
    assert(f.kind == scientific::F_NONE);
    assert(common_2::all_close_c(q, [-0.258198889747161, -0.7348469228349537, -0.7745966692414835, -0.16329931618554494], [-0.25819888974716115, 0.5715476066494081, 0.5163977794943222, -0.3265986323710902], 1.0e-12));
    assert(common_2::all_close_c(r, [-3.8729833462074166, -3.0983866769659345, 0.0, -2.4494897427831783], [0.0, -2.32379000772445, 0.0, 0.0], 1.0e-12));
    // A tall matrix. The reduced form holds two columns.
    let a2, f2 = scientific::complex_array(common::mx(3, 2, [1.0, 2.0, 0.0, 1.0, 2.0, -1.0]), common::mx(3, 2, [0.0, -1.0, 3.0, 1.0, 0.0, 2.0]));
    assert(f2.kind == scientific::F_NONE);
    let q2, r2, g2 = scientific::cqr(a2);
    assert(g2.kind == scientific::F_NONE);
    assert(q2.shape[0] == 3 && q2.shape[1] == 2 && r2.shape[0] == 2 && r2.shape[1] == 2);
    assert(common_2::all_close_c(q2, [-0.2672612419124243, -0.5298798792491219, -0.0, -0.29673273237950815, -0.5345224838248488, 0.4239039033992974], [0.0, 0.2967327323795083, -0.8017837257372732, -0.10597597584982449, -0.0, -0.5934654647590163], 1.0e-12));
    assert(common_2::all_close_c(r2, [-3.7416573867739413, -0.8017837257372729, 0.0, -3.3700360320244136], [0.0, 0.0, 0.0, 0.0], 1.0e-12));
    // A wide matrix.
    let a3, f3 = scientific::complex_array(common::mx(2, 3, [1.0, 2.0, 0.0, 3.0, 4.0, 1.0]), common::mx(2, 3, [1.0, 0.0, -1.0, -2.0, 1.0, 0.0]));
    assert(f3.kind == scientific::F_NONE);
    let q3, r3, g3 = scientific::cqr(a3);
    assert(g3.kind == scientific::F_NONE);
    assert(q3.shape[0] == 2 && q3.shape[1] == 2 && r3.shape[0] == 2 && r3.shape[1] == 3);
    assert(common_2::all_close_c(r3, [-3.8729833462074166, -3.0983866769659345, -0.5163977794943224, 0.0, -2.4494897427831783, -0.734846922834953], [0.0, -2.32379000772445, -0.2581988897471612, 0.0, 0.0, 1.0614455552060438], 1.0e-12));
    // A matrix of rank one, a matrix of zeros and a matrix of one element.
    let a4, f4 = scientific::complex_array(common::mx(2, 2, [1.0, 2.0, 2.0, 4.0]), common::mx(2, 2, [1.0, 2.0, 2.0, 4.0]));
    assert(f4.kind == scientific::F_NONE);
    let q4, r4, g4 = scientific::cqr(a4);
    assert(g4.kind == scientific::F_NONE);
    assert(common_2::all_close_c(r4, [-3.1622776601683795, -6.32455532033676, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 1.0e-12));
    common_2::check_qr(a4);
    let a5, f5 = scientific::complex_array(common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]), common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(f5.kind == scientific::F_NONE);
    let q5, r5, g5 = scientific::cqr(a5);
    assert(g5.kind == scientific::F_NONE);
    assert(common_2::all_close_c(q5, [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], 1.0e-12));
    assert(common_2::all_close_c(r5, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 1.0e-12));
    let a6, f6 = scientific::complex_array(common::mx(1, 1, [3.0]), common::mx(1, 1, [-4.0]));
    assert(f6.kind == scientific::F_NONE);
    let q6, r6, g6 = scientific::cqr(a6);
    assert(g6.kind == scientific::F_NONE);
    assert(common_2::all_close_c(q6, [-0.6000000000000001], [0.8], 1.0e-12));
    assert(common_2::all_close_c(r6, [-5.0], [0.0], 1.0e-12));
    // The properties hold for random matrices of several shapes.
    var rng = scientific::default_rng(296);
    var t: i64 = 0;
    while t < 20 {
        let m = common_2::random_complex(rng, 4, 4);
        common_2::check_qr(m);
        t += 1;
    }
    let tall = common_2::random_complex(rng, 6, 3);
    common_2::check_qr(tall);
    let wide = common_2::random_complex(rng, 3, 6);
    common_2::check_qr(wide);
    common_2::check_qr(a1);
    common_2::check_qr(a2);
    common_2::check_qr(a3);
    return;
}

// test_csvd_real_product tests a pair of columns whose inner product is a
// negative real value. The method must still turn them to zero.
fn test_csvd_real_product() {
    let a = scientific::CArr { ndim: 2, shape: [2, 2], data: [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0] };
    let u, s, vh, g = scientific::csvd(a);
    assert(g.kind == scientific::F_NONE);
    assert(common::close(s.data[0], 2.414213562373095, 1.0e-12) && common::close(s.data[1], 0.41421356237309503, 1.0e-12));
    common_2::check_svd(a);
    let b, f = scientific::complex_array(common::mx(3, 3, [1.0, -2.0, 3.0, -1.0, 4.0, 2.0, 2.0, -3.0, 1.0]), common::mx(3, 3, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    assert(f.kind == scientific::F_NONE);
    common_2::check_svd(b);
    return;
}

// test_csvd pins the singular values against NumPy and tests the factors.
fn test_csvd() {
    let a1, f1 = scientific::complex_array(common::mx(2, 2, [1.0, 2.0, 3.0, 4.0]), common::mx(2, 2, [1.0, 0.0, -2.0, 1.0]));
    assert(f1.kind == scientific::F_NONE);
    let u, s, vh, f = scientific::csvd(a1);
    assert(f.kind == scientific::F_NONE);
    assert(common::close(s.data[0], 5.770360347567416, 1.0e-12) && common::close(s.data[1], 1.6440624864103084, 1.0e-12));
    common_2::check_svd(a1);
    let a2, f2 = scientific::complex_array(common::mx(3, 2, [1.0, 2.0, 0.0, 1.0, 2.0, -1.0]), common::mx(3, 2, [0.0, -1.0, 3.0, 1.0, 0.0, 2.0]));
    assert(f2.kind == scientific::F_NONE);
    let u2, s2, vh2, g2 = scientific::csvd(a2);
    assert(g2.kind == scientific::F_NONE);
    assert(u2.shape[0] == 3 && u2.shape[1] == 3 && vh2.shape[0] == 2 && vh2.shape[1] == 2 && s2.data.len == 2);
    assert(common::close(s2.data[0], 4.020233533038644, 1.0e-12) && common::close(s2.data[1], 3.136514361489777, 1.0e-12));
    common_2::check_svd(a2);
    let a3, f3 = scientific::complex_array(common::mx(2, 3, [1.0, 2.0, 0.0, 3.0, 4.0, 1.0]), common::mx(2, 3, [1.0, 0.0, -1.0, -2.0, 1.0, 0.0]));
    assert(f3.kind == scientific::F_NONE);
    let u3, s3, vh3, g3 = scientific::csvd(a3);
    assert(g3.kind == scientific::F_NONE);
    assert(vh3.shape[0] == 3 && vh3.shape[1] == 3 && s3.data.len == 2);
    assert(common::close(s3.data[0], 5.842323677306963, 1.0e-12) && common::close(s3.data[1], 1.9665335109218063, 1.0e-12));
    common_2::check_svd(a3);
    // A matrix of rank one. The second value is near zero.
    let a4, f4 = scientific::complex_array(common::mx(2, 2, [1.0, 2.0, 2.0, 4.0]), common::mx(2, 2, [1.0, 2.0, 2.0, 4.0]));
    assert(f4.kind == scientific::F_NONE);
    let u4, s4, vh4, g4 = scientific::csvd(a4);
    assert(g4.kind == scientific::F_NONE);
    assert(common::close(s4.data[0], 7.071067811865476, 1.0e-12) && s4.data[1] < 1.0e-12);
    common_2::check_svd(a4);
    // A matrix of zeros.
    let a5, f5 = scientific::complex_array(common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]), common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(f5.kind == scientific::F_NONE);
    let u5, s5, vh5, g5 = scientific::csvd(a5);
    assert(g5.kind == scientific::F_NONE);
    assert(s5.data[0] == 0.0 && s5.data[1] == 0.0);
    assert(common_2::gram_dev(u5, true, 2) < 1.0e-12 && common_2::gram_dev(vh5, false, 2) < 1.0e-12);
    // A matrix of one element.
    let a6, f6 = scientific::complex_array(common::mx(1, 1, [3.0]), common::mx(1, 1, [-4.0]));
    assert(f6.kind == scientific::F_NONE);
    let u6, s6, vh6, g6 = scientific::csvd(a6);
    assert(g6.kind == scientific::F_NONE);
    assert(common::close(s6.data[0], 5.0, 1.0e-12));
    common_2::check_svd(a6);
    // The properties hold for random matrices.
    var rng = scientific::default_rng(2960);
    var t: i64 = 0;
    while t < 20 {
        let m = common_2::random_complex(rng, 4, 4);
        common_2::check_svd(m);
        t += 1;
    }
    let tall = common_2::random_complex(rng, 6, 3);
    common_2::check_svd(tall);
    let wide = common_2::random_complex(rng, 3, 6);
    common_2::check_svd(wide);
    // Matrices of rank two with more rows or more columns.
    let l = common_2::random_complex(rng, 5, 2);
    let rr = common_2::random_complex(rng, 2, 4);
    let low, lowf = scientific::cdot(l, rr);
    assert(lowf.kind == scientific::F_NONE);
    common_2::check_svd(low);
    let lw = common_2::random_complex(rng, 2, 5);
    let rw = common_2::random_complex(rng, 4, 2);
    let lowwide, lwf = scientific::cdot(rw, lw);
    assert(lwf.kind == scientific::F_NONE);
    common_2::check_svd(lowwide);
    return;
}

// test_ceig pins the eigenvalues and the vectors against NumPy.
fn test_ceig() {
    common_3::check_general_pin(2, [1.0, 2.0, 3.0, 4.0], [1.0, 0.0, -2.0, 1.0], [-0.4511489635772472, 5.451148963577246], [1.6777021508177925, 0.32229784918220716], [0.7805646436529173, 0.4013293303259815, -0.5663577868209877, 0.9138933768694496], [0.0, 0.06110371784313384, 0.26449516892795305, 0.0], true);
    common_3::check_general_pin(3, [1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0, -0.0, 1.0], [0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, -2.0], [0.3927309427175605, 0.6304421702494745, 3.976826887032966], [0.21202518511613053, -2.305046500316191, 0.09302131520006082], [0.7390582988548714, -0.20800049299100482, 0.47717948350295286, -0.21086268909420808, -0.38244507545935064, 0.7222179920517278, -0.07881950397052173, 0.8351271337240647, 0.018979818536928615], [0.0, -0.3185988424604842, -0.2575244998400774, -0.02708185823394555, 0.10737321487716754, 0.0, -0.6343373045865587, 0.0, -0.42897763455808363], true);
    common_3::check_general_pin(3, [2.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.13363062095621217, -0.16903085094570333, 1.0, 0.8017837257372731, 0.8451542547285166, 0.0, -0.40089186286863654, 0.0, 0.0], [0.13363062095621217, -0.50709255283711, 0.0, 0.0, 0.0, 0.0, -0.40089186286863654, 0.0, 0.0], true);
    common_3::check_general_pin(1, [3.0], [-4.0], [3.0], [-4.0], [1.0], [0.0], true);
    common_3::check_general_pin(2, [1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0, 0.0, -2.220446049250313e-16], [0.0, 0.0, 0.0, -0.0], false);
    common_3::check_general_pin(2, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], true);
    // A defective matrix gives nearly parallel vectors.
    let d, fd = scientific::complex_array(common::mx(2, 2, [1.0, 1.0, 0.0, 1.0]), common::mx(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(fd.kind == scientific::F_NONE);
    let dw, dv, df = scientific::ceig(d);
    assert(df.kind == scientific::F_NONE);
    var sine = dv.data[0] * dv.data[6] - dv.data[4] * dv.data[2];
    if sine < 0.0 { sine = -sine; }
    assert(sine < 1.0e-12);
    // The properties hold for random matrices of several sides.
    var rng = scientific::default_rng(297);
    var t: i64 = 0;
    while t < 20 {
        let m = common_2::random_complex(rng, 4, 4);
        let mw, mv, mf = scientific::ceig(m);
        assert(mf.kind == scientific::F_NONE);
        common_3::check_eig(m, mw, mv);
        t += 1;
    }
    let sides = [1, 2, 3, 6, 12];
    var si: i64 = 0;
    while si < sides.len {
        let m = common_2::random_complex(rng, sides[si], sides[si]);
        let mw, mv, mf = scientific::ceig(m);
        assert(mf.kind == scientific::F_NONE);
        common_3::check_eig(m, mw, mv);
        si += 1;
    }
    // The values of a matrix near the limits of the number format keep the scale.
    let big, fb = scientific::complex_array(common::mx(2, 2, [1.0e300, 2.0e300, 3.0e300, 4.0e300]), common::mx(2, 2, [1.0e300, 0.0, -2.0e300, 1.0e300]));
    assert(fb.kind == scientific::F_NONE);
    let bw, bv, bf = scientific::ceig(big);
    assert(bf.kind == scientific::F_NONE);
    var bs: f64[>..] = [];
    var bvs: f64[>..] = [];
    common_2::sort_eig(bw, bv, 2, bs, bvs);
    assert(common::close(bs[0] / 1.0e300, -0.4511489635772472, 1.0e-12) && common::close(bs[1] / 1.0e300, 1.6777021508177925, 1.0e-12));
    assert(common::close(bvs[0], 0.7805646436529173, 1.0e-12));
    let small, fs = scientific::complex_array(common::mx(2, 2, [1.0e-300, 2.0e-300, 3.0e-300, 4.0e-300]), common::mx(2, 2, [1.0e-300, 0.0, -2.0e-300, 1.0e-300]));
    assert(fs.kind == scientific::F_NONE);
    let sw, sv, sf = scientific::ceig(small);
    assert(sf.kind == scientific::F_NONE);
    var ss: f64[>..] = [];
    var svs: f64[>..] = [];
    common_2::sort_eig(sw, sv, 2, ss, svs);
    assert(common::close(ss[2] / 1.0e-300, 5.451148963577246, 1.0e-12));
    return;
}

// test_ceigh pins the eigenvalues and the vectors against NumPy.
fn test_ceigh() {
    let h1, fh1 = scientific::complex_array(scientific::from_rows(2, 2, [2.0, 1.0, 1.0, 3.0]), scientific::from_rows(2, 2, [0.0, -1.0, 1.0, 0.0]));
    assert(fh1.kind == scientific::F_NONE);
    let h1w, h1v, h1f = scientific::ceigh(h1);
    assert(h1f.kind == scientific::F_NONE);
    var h1i: i64 = 0;
    while h1i < 2 { assert(common::close(h1w.data[h1i], [0.9999999999999993, 3.9999999999999987][h1i], 1.0e-12)); h1i += 1; }
    let h1uw, h1uv, h1uf = scientific::ceigh(h1, "U");
    assert(h1uf.kind == scientific::F_NONE);
    h1i = 0;
    while h1i < 2 { assert(common::close(h1uw.data[h1i], [0.9999999999999992, 4.0][h1i], 1.0e-12)); h1i += 1; }
    assert(common_2::all_close_c(h1v, [0.8164965809277258, 0.40824829046386313, -0.4082482904638631, 0.8164965809277255], [-0.0, -0.40824829046386324, -0.4082482904638632, -4.543221347549799e-17], 1.0e-12));
    let h2, fh2 = scientific::complex_array(scientific::from_rows(3, 3, [2.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, -0.0, 1.0]), scientific::from_rows(3, 3, [0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0]));
    assert(fh2.kind == scientific::F_NONE);
    let h2w, h2v, h2f = scientific::ceigh(h2);
    assert(h2f.kind == scientific::F_NONE);
    var h2i: i64 = 0;
    while h2i < 3 { assert(common::close(h2w.data[h2i], [0.32486912943335366, 1.4608111271891109, 4.214319743377535][h2i], 1.0e-12)); h2i += 1; }
    let h2uw, h2uv, h2uf = scientific::ceigh(h2, "U");
    assert(h2uf.kind == scientific::F_NONE);
    h2i = 0;
    while h2i < 3 { assert(common::close(h2uw.data[h2i], [0.3248691294333537, 1.4608111271891109, 4.214319743377535][h2i], 1.0e-12)); h2i += 1; }
    assert(common_2::all_close_c(h2v, [0.30202813664790923, 0.7392387395392241, 0.3681603558983795, -2.5536906328208355e-17, -0.19929465135514815, 0.8152247447946817, 0.7493904923263159, -0.4324866297626548, -4.561233510770051e-18], [0.3020281366479092, -0.0, -0.36816035589837964, -0.5059366554786339, -0.1992946513551482, -4.561794771291678e-17, 4.93891608702895e-17, 0.4324866297626546, -0.2536227910973356], 1.0e-12));
    let h3, fh3 = scientific::complex_array(scientific::from_rows(2, 2, [2.0, 0.0, 0.0, 2.0]), scientific::from_rows(2, 2, [0.0, 0.0, 0.0, 0.0]));
    assert(fh3.kind == scientific::F_NONE);
    let h3w, h3v, h3f = scientific::ceigh(h3);
    assert(h3f.kind == scientific::F_NONE);
    var h3i: i64 = 0;
    while h3i < 2 { assert(common::close(h3w.data[h3i], [2.0, 2.0][h3i], 1.0e-12)); h3i += 1; }
    let h3uw, h3uv, h3uf = scientific::ceigh(h3, "U");
    assert(h3uf.kind == scientific::F_NONE);
    h3i = 0;
    while h3i < 2 { assert(common::close(h3uw.data[h3i], [2.0, 2.0][h3i], 1.0e-12)); h3i += 1; }
    let h4, fh4 = scientific::complex_array(scientific::from_rows(1, 1, [5.0]), scientific::from_rows(1, 1, [0.0]));
    assert(fh4.kind == scientific::F_NONE);
    let h4w, h4v, h4f = scientific::ceigh(h4);
    assert(h4f.kind == scientific::F_NONE);
    var h4i: i64 = 0;
    while h4i < 1 { assert(common::close(h4w.data[h4i], [5.0][h4i], 1.0e-12)); h4i += 1; }
    let h4uw, h4uv, h4uf = scientific::ceigh(h4, "U");
    assert(h4uf.kind == scientific::F_NONE);
    h4i = 0;
    while h4i < 1 { assert(common::close(h4uw.data[h4i], [5.0][h4i], 1.0e-12)); h4i += 1; }
    let h5, fh5 = scientific::complex_array(scientific::from_rows(3, 3, [1.0, 5.0, 5.0, 2.0, 3.0, 5.0, 4.0, 1.0, 2.0]), scientific::from_rows(3, 3, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -2.0, 0.0]));
    assert(fh5.kind == scientific::F_NONE);
    let h5w, h5v, h5f = scientific::ceigh(h5);
    assert(h5f.kind == scientific::F_NONE);
    var h5i: i64 = 0;
    while h5i < 3 { assert(common::close(h5w.data[h5i], [-2.751918833193568, 1.2035075963077457, 7.548411236885824][h5i], 1.0e-12)); h5i += 1; }
    let h5uw, h5uv, h5uf = scientific::ceigh(h5, "U");
    assert(h5uf.kind == scientific::F_NONE);
    h5i = 0;
    while h5i < 3 { assert(common::close(h5uw.data[h5i], [-3.5991143153843557, -2.4452643834117493, 12.044378698796105][h5i], 1.0e-12)); h5i += 1; }
    // The lower case of the choice of the triangle gives the same result.
    let lw, lv, lf = scientific::ceigh(h5, "u");
    assert(lf.kind == scientific::F_NONE);
    assert(lw.data[0] == h5uw.data[0] && lw.data[2] == h5uw.data[2]);
    // The properties hold for random Hermitian matrices.
    var rng = scientific::default_rng(2971);
    var t: i64 = 0;
    while t < 20 {
        let m = common_3::random_hermitian(rng, 4);
        common_3::check_eigh(m);
        t += 1;
    }
    let r1 = common_3::random_hermitian(rng, 1);
    common_3::check_eigh(r1);
    let r7 = common_3::random_hermitian(rng, 7);
    common_3::check_eigh(r7);
    // A scaled matrix keeps its scale.
    let big, fb = scientific::complex_array(common::mx(2, 2, [2.0e300, 1.0e300, 1.0e300, 3.0e300]), common::mx(2, 2, [0.0, -1.0e300, 1.0e300, 0.0]));
    assert(fb.kind == scientific::F_NONE);
    let bw, bv, bf = scientific::ceigh(big);
    assert(bf.kind == scientific::F_NONE);
    assert(common::close(bw.data[0] / 1.0e300, 1.0, 1.0e-12) && common::close(bw.data[1] / 1.0e300, 4.0, 1.0e-12));
    return;
}
