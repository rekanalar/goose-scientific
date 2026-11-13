// The tests of the group order. Run them with SCIENTIFIC_SUITES=order.
namespace order_tests;

import scientific from "../..";
import os;
import common;

// The order statistics hold the values of NumPy.
fn test_order_statistics() {
    let a, fa = scientific::from_values([1.0, 2.0, 4.0, 7.0, 11.0, 12.0]);
    assert(fa.kind == scientific::F_NONE);
    // The quantile follows the method of NumPy.
    let q, fq = scientific::quantile(a, 0.25);
    assert(fq.kind == scientific::F_NONE && q == 2.5);
    let q2, fq2 = scientific::quantile(a, 0.25, "lower");
    assert(fq2.kind == scientific::F_NONE && q2 == 2.0);
    let q3, fq3 = scientific::quantile(a, 0.25, "higher");
    assert(fq3.kind == scientific::F_NONE && q3 == 4.0);
    let q4, fq4 = scientific::quantile(a, 0.25, "midpoint");
    assert(fq4.kind == scientific::F_NONE && q4 == 3.0);
    let q5, fq5 = scientific::quantile(a, 0.5, "nearest");
    assert(fq5.kind == scientific::F_NONE && q5 == 4.0);
    let q6, fq6 = scientific::quantile(a, 0.25, "inverted_cdf");
    assert(fq6.kind == scientific::F_NONE && q6 == 2.0);
    let q7, fq7 = scientific::quantile(a, 0.5, "averaged_inverted_cdf");
    assert(fq7.kind == scientific::F_NONE && q7 == 5.5);
    let q8, fq8 = scientific::quantile(a, 0.1, "hazen");
    assert(fq8.kind == scientific::F_NONE && common::close(q8, 1.1));
    let q9, fq9 = scientific::quantile(a, 0.25, "weibull");
    assert(fq9.kind == scientific::F_NONE && common::close(q9, 1.75));
    let q10, fq10 = scientific::quantile(a, 0.25, "median_unbiased");
    assert(fq10.kind == scientific::F_NONE && common::close(q10, 1.9166666666666667));
    let q11, fq11 = scientific::quantile(a, 0.25, "normal_unbiased");
    assert(fq11.kind == scientific::F_NONE && common::close(q11, 1.9375));
    let q12, fq12 = scientific::quantile(a, 0.1, "interpolated_inverted_cdf");
    assert(fq12.kind == scientific::F_NONE && common::close(q12, 1.0));
    let q13, fq13 = scientific::quantile(a, 0.5, "closest_observation");
    assert(fq13.kind == scientific::F_NONE && q13 == 4.0);
    // A quantile outside the range and an unknown method are faults.
    let qb, fqb = scientific::quantile(a, 1.5);
    assert(fqb.kind == scientific::F_VALUE && qb == 0.0);
    let qb2, fqb2 = scientific::quantile(a, 0.5, "unknown");
    assert(fqb2.kind == scientific::F_VALUE);
    // The percentage runs from zero to one hundred.
    let p, fp = scientific::percentile(a, 50.0);
    assert(fp.kind == scientific::F_NONE && p == 5.5);
    let p2, fp2 = scientific::percentile(a, 25.0);
    assert(fp2.kind == scientific::F_NONE && p2 == 2.5);
    let pb, fpb = scientific::percentile(a, 150.0);
    assert(fpb.kind == scientific::F_VALUE);
    // The middle value of an even count is the mean of the two.
    let med, fmed = scientific::median(a);
    assert(fmed.kind == scientific::F_NONE && med == 5.5);
    let med2, fmed2 = scientific::median(scientific::from_values([1.0, 2.0, 4.0, 7.0]));
    assert(fmed2.kind == scientific::F_NONE && med2 == 3.0);
    // The mode takes the value that comes most often.
    let m1, c1, fm1 = scientific::mode(scientific::from_values([1.0, 2.0, 2.0, 3.0, 3.0]));
    assert(fm1.kind == scientific::F_NONE && m1 == 2.0 && c1 == 2);
    let m2, c2, fm2 = scientific::mode(scientific::from_values([5.0, 5.0, 5.0, 1.0, 1.0]));
    assert(fm2.kind == scientific::F_NONE && m2 == 5.0 && c2 == 3);
    let m3, c3, fm3 = scientific::mode(scientific::from_values([2.0, 1.0, 1.0, 2.0, 3.0]));
    assert(fm3.kind == scientific::F_NONE && m3 == 1.0 && c3 == 2);
    // The weighted mean follows the weights.
    let w, fw = scientific::from_values([4.0, 3.0, 2.0, 1.0]);
    let av, fav = scientific::average(scientific::from_values([1.0, 2.0, 3.0, 4.0]), w);
    assert(fav.kind == scientific::F_NONE && av == 2.0);
    // Weights of a different shape and weights of zero are faults.
    let aw, faw = scientific::average(scientific::from_values([1.0, 2.0]), scientific::from_values([1.0]));
    assert(faw.kind == scientific::F_SHAPE);
    let aw2, faw2 = scientific::average(scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                                        scientific::from_values([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    assert(faw2.kind == scientific::F_SHAPE && aw2 == 0.0);
    let az, faz = scientific::average(scientific::from_values([1.0, 2.0]), scientific::from_values([0.0, 0.0]));
    assert(faz.kind == scientific::F_VALUE);
    // The forms along an axis drop the axis.
    let m4, fm4 = scientific::from_rows(2, 3, [1.0, 2.0, 4.0, 7.0, 11.0, 12.0]);
    assert(fm4.kind == scientific::F_NONE);
    let qa, fqa = scientific::quantile_axis(m4, 0.5, 0);
    let ma, fma = scientific::median_axis(m4, 1);
    let pa, fpa = scientific::percentile_axis(m4, 50.0, 0);
    assert(fqa.kind == scientific::F_NONE && fma.kind == scientific::F_NONE && fpa.kind == scientific::F_NONE);
    assert(qa.data[0] == 4.0 && qa.data[1] == 6.5 && qa.data[2] == 8.0);
    assert(ma.data[0] == 2.0 && ma.data[1] == 11.0);
    assert(pa.data[0] == 4.0 && pa.data[1] == 6.5 && pa.data[2] == 8.0);
    // The weighted mean along an axis takes one weight for each place.
    let aa, faa = scientific::average_axis(m4, scientific::from_values([1.0, 1.0]), 0);
    assert(faa.kind == scientific::F_NONE && aa.data[0] == 4.0 && aa.data[1] == 6.5 && aa.data[2] == 8.0);
    let ab, fab = scientific::average_axis(m4, scientific::from_values([1.0, 1.0, 1.0]), 1);
    assert(fab.kind == scientific::F_NONE && common::close(ab.data[0], 2.3333333333333335) && ab.data[1] == 10.0);
    // Weights of a different length are a fault.
    let ac, fac = scientific::average_axis(m4, scientific::from_values([1.0]), 0);
    assert(fac.kind == scientific::F_SHAPE && ac.data.len == 0);
    // A bad axis is a fault.
    let qb3, fqb3 = scientific::quantile_axis(m4, 0.5, 5);
    assert(fqb3.kind == scientific::F_RANGE && qb3.data.len == 0);
}
