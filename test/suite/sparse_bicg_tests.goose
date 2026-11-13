// The tests of the group sparse_bicg. Run them with SCIENTIFIC_SUITES=sparse_bicg.
namespace sparse_bicg_tests;

import scientific from "../..";
import os;
import common;
import common_3;

fn test_sparse_bicg_pins0() {
    let qa0, qf0 = scientific::csr_from_dense(common_3::ni_nd([5, 5], [2.5164, 0.0, 0.42, 0.0, -1.398, -1.453, 1.2404, 0.0, 0.0, -0.875, 1.149, 1.456, 1.2404, 0.0, 2.344, 0.227, 2.043, 0.0, 1.2404, -2.387, -0.096, 0.0, 0.355, 0.0, -1.2376000000000003]));
    assert(qf0.kind == scientific::F_NONE);
    let qx0bicg, qc0bicg, qg0bicg = scientific::bicg(qa0, [-0.04, -1.33, 1.538, 1.09, 1.152], 1e-12);
    assert(qg0bicg.kind == scientific::F_NONE && qc0bicg == 0 && common::all_near(qx0bicg, [-0.5360018523578103, -1.6282536944202755, 3.4552167659319513, 3.854665589254586, 0.10185692447655473]));
    let qx0cgs, qc0cgs, qg0cgs = scientific::cgs(qa0, [-0.04, -1.33, 1.538, 1.09, 1.152], 1e-12);
    assert(qg0cgs.kind == scientific::F_NONE && qc0cgs == 0 && common::all_near(qx0cgs, [-0.5360018523578038, -1.628253694420268, 3.4552167659319335, 3.8546655892545876, 0.10185692447656718]));
    let qx0qmr, qc0qmr, qg0qmr = scientific::qmr(qa0, [-0.04, -1.33, 1.538, 1.09, 1.152], 1e-12);
    assert(qg0qmr.kind == scientific::F_NONE && qc0qmr == 0 && common::all_near(qx0qmr, [-0.536001852357808, -1.628253694420273, 3.4552167659319477, 3.854665589254589, 0.10185692447655756]));
    let qx0tfqmr, qc0tfqmr, qg0tfqmr = scientific::tfqmr(qa0, [-0.04, -1.33, 1.538, 1.09, 1.152], 1e-12);
    assert(qg0tfqmr.kind == scientific::F_NONE && qc0tfqmr == 0 && common::all_near(qx0tfqmr, [-0.5360018523578104, -1.6282536944202743, 3.4552167659319615, 3.854665589254587, 0.1018569244765509]));
    let qa1, qf1 = scientific::csr_from_dense(common_3::ni_nd([10, 10], [1.2742, 0.0, 1.022, 0.0, 0.1, 2.764, -1.658, -2.974, 2.795, 0.0, 0.0, -1.3927999999999998, -1.937, 0.0, 0.0, 0.0, -1.665, -0.21, -2.671, 2.237, 0.0, 0.0, 2.1812, -1.916, 0.0, -1.573, -1.691, 0.0, 0.205, -2.067, 0.0, 0.0, -0.941, -0.7608000000000001, -1.305, 0.0, 2.396, -2.423, -2.842, -2.043, 1.98, 0.0, -2.185, 0.0, -0.2198, 0.2, 0.0, -0.302, 0.12, -0.723, 1.901, -2.462, -2.758, -0.707, -2.15, 0.6242, -0.784, 2.741, -1.547, 0.0, 2.697, 2.26, -0.924, 2.043, -0.85, -0.804, 1.1572, 1.758, 1.748, 0.0, 0.0, 1.867, -0.74, 0.0, -2.356, 1.361, 2.442, 3.2992, 0.243, 0.0, 0.0, -1.283, 0.95, 0.181, 0.29, 0.816, 0.0, 0.0, 1.2742, -2.55, -2.984, 0.0, -2.761, 0.0, 0.385, -2.498, -2.194, 2.524, 0.0, 2.5632]));
    assert(qf1.kind == scientific::F_NONE);
    let qx1bicg, qc1bicg, qg1bicg = scientific::bicg(qa1, [-1.968, 1.077, -0.04, 1.462, 1.316, -0.203, -1.052, -1.072, -1.036, -0.313], 1e-12);
    assert(qg1bicg.kind == scientific::F_NONE && qc1bicg == 0 && common::all_near(qx1bicg, [0.39170726979435627, 0.2796426821380153, -0.4025126697355436, -0.8111797190166857, 1.290200384036176, 0.05381564293638457, 0.3552021285842175, 0.1008468727539701, -0.5168759217958493, -0.036283823576397714]));
    let qx1cgs, qc1cgs, qg1cgs = scientific::cgs(qa1, [-1.968, 1.077, -0.04, 1.462, 1.316, -0.203, -1.052, -1.072, -1.036, -0.313], 1e-12);
    assert(qg1cgs.kind == scientific::F_NONE && qc1cgs == 0 && common::all_near(qx1cgs, [0.39170726979403886, 0.2796426821382577, -0.40251266973638294, -0.8111797190170629, 1.2902003840355352, 0.05381564293611034, 0.3552021285843597, 0.1008468727534151, -0.5168759217952485, -0.0362838235768981]));
    let qx1qmr, qc1qmr, qg1qmr = scientific::qmr(qa1, [-1.968, 1.077, -0.04, 1.462, 1.316, -0.203, -1.052, -1.072, -1.036, -0.313], 1e-12);
    assert(qg1qmr.kind == scientific::F_NONE && qc1qmr == 0 && common::all_near(qx1qmr, [0.3917072697940997, 0.2796426821373684, -0.40251266973609456, -0.811179719017822, 1.2902003840348104, 0.05381564293554104, 0.3552021285846116, 0.10084687275359407, -0.5168759217946852, -0.036283823575945506]));
    let qx1tfqmr, qc1tfqmr, qg1tfqmr = scientific::tfqmr(qa1, [-1.968, 1.077, -0.04, 1.462, 1.316, -0.203, -1.052, -1.072, -1.036, -0.313], 1e-12);
    assert(qg1tfqmr.kind == scientific::F_NONE && qc1tfqmr == 0 && common::all_near(qx1tfqmr, [0.3917072690717445, 0.27964268185583546, -0.4025126695857786, -0.8111797182860073, 1.290200383519876, 0.053815642518144885, 0.3552021283274174, 0.10084687281623489, -0.5168759210326975, -0.036283823452906185]));
    let qa2, qf2 = scientific::csr_from_dense(common_3::ni_nd([9, 9], [1.2526666666666666, 1.98, 0.0, 0.0, 1.86, 0.0, -2.334, -2.173, 0.0, 0.761, 1.2526666666666666, 0.0, -1.628, -1.228, 0.0, -1.876, -0.385, 0.0, 2.47, 0.0, 1.2526666666666666, -2.119, 0.0, 2.83, 2.841, 0.085, -1.636, -2.115, 0.712, 2.335, 2.8426666666666667, 2.698, -2.777, 0.0, 0.0, 0.0, 2.181, 1.663, 1.795, -1.335, 0.5486666666666666, 0.0, -1.373, 0.0, -2.957, 0.0, 0.0, 0.023, -0.601, -1.093, -1.6273333333333333, 0.0, 2.475, 0.0, 0.0, 1.221, 0.099, 2.285, 0.25, -2.64, 3.1666666666666665, 2.467, 0.0, -2.1, 0.042, 0.138, -1.985, 2.874, -2.658, 0.745, 0.7096666666666666, 0.0, 0.0, 0.184, -0.153, 2.694, 2.462, 0.0, 0.0, 1.81, 4.149666666666667]));
    assert(qf2.kind == scientific::F_NONE);
    let qx2bicg, qc2bicg, qg2bicg = scientific::bicg(qa2, [0.813, -1.127, -0.765, -1.888, -0.032, 1.03, 0.487, 0.086, 1.458], 1e-12);
    assert(qg2bicg.kind == scientific::F_NONE && qc2bicg == 0 && common::all_near(qx2bicg, [2.183515375671698, -1.9799851867681553, -0.6872056327810562, 0.091832569441513, 0.48155697668002695, -1.5067409139070855, -0.1621459093185023, -0.3331841755955466, 0.21381186080399278]));
    let qx2cgs, qc2cgs, qg2cgs = scientific::cgs(qa2, [0.813, -1.127, -0.765, -1.888, -0.032, 1.03, 0.487, 0.086, 1.458], 1e-12);
    assert(qg2cgs.kind == scientific::F_NONE && qc2cgs == 0 && common::all_near(qx2cgs, [2.183515375671632, -1.9799851867681075, -0.6872056327810312, 0.09183256944143903, 0.4815569766799193, -1.5067409139070962, -0.16214590931861203, -0.33318417559546626, 0.21381186080384373]));
    let qx2qmr, qc2qmr, qg2qmr = scientific::qmr(qa2, [0.813, -1.127, -0.765, -1.888, -0.032, 1.03, 0.487, 0.086, 1.458], 1e-12);
    assert(qg2qmr.kind == scientific::F_NONE && qc2qmr == 0 && common::all_near(qx2qmr, [2.183515375671378, -1.9799851867681717, -0.6872056327811489, 0.09183256944138077, 0.48155697667943176, -1.5067409139070649, -0.16214590931854042, -0.3331841755955015, 0.21381186080407488]));
    let qx2tfqmr, qc2tfqmr, qg2tfqmr = scientific::tfqmr(qa2, [0.813, -1.127, -0.765, -1.888, -0.032, 1.03, 0.487, 0.086, 1.458], 1e-12);
    assert(qg2tfqmr.kind == scientific::F_NONE && qc2tfqmr == 0 && common::all_near(qx2tfqmr, [2.1835153756712407, -1.9799851867678269, -0.687205632781154, 0.09183256944160655, 0.48155697667995845, -1.5067409139066952, -0.16214590931860334, -0.3331841755952537, 0.21381186080382153]));
}

fn test_sparse_bicg_pins1() {
    let qa3, qf3 = scientific::csr_from_dense(common_3::ni_nd([10, 10], [1.5539, 0.955, -2.672, -1.174, 0.0, 0.0, 1.3, 1.219, 2.065, 0.0, 2.664, 1.5539, 0.0, 0.0, 0.0, 0.035, -2.646, -0.692, 1.404, 1.622, -2.442, -2.511, -1.2231, -1.189, 1.689, 0.0, 0.0, 1.248, 2.947, -0.081, -1.322, 0.035, 1.233, 0.0019000000000000128, 2.464, 0.0, 0.0, 0.0, 0.0, -1.546, 0.0, -2.767, 1.596, -1.371, 0.2579, 1.364, 2.274, 0.217, -2.52, -1.689, 0.0, -0.774, 0.0, 1.744, -2.449, 0.8669, 2.734, -0.935, 2.832, -2.146, 0.0, 0.431, -1.317, 0.0, 2.692, -0.556, 2.3269, -0.988, 2.261, -1.781, 0.381, 2.875, -0.417, -2.603, 0.0, 2.67, 0.0, 1.5539, 2.721, 0.0, 0.913, -1.466, 0.0, 0.0, 0.0, 0.0, -1.431, 2.423, 1.5539, 0.425, -0.302, 0.0, -2.257, -1.004, -0.374, 0.0, 0.0, 0.0, 0.0, 1.5539]));
    assert(qf3.kind == scientific::F_NONE);
    let qx3bicg, qc3bicg, qg3bicg = scientific::bicg(qa3, [1.583, -0.401, -1.99, 1.209, -0.511, -1.347, 1.132, 0.972, -1.597, 0.047], 1e-12);
    assert(qg3bicg.kind == scientific::F_NONE && qc3bicg == 0 && common::all_near(qx3bicg, [0.036131835750469965, 0.6293215511017058, -1.2900170248706062, -0.5921076859406958, -0.2888025460156532, -0.2816452950739169, -1.1982894987934498, -0.13387126958726664, -0.7240880525737509, -2.2885287853980674]));
    let qx3cgs, qc3cgs, qg3cgs = scientific::cgs(qa3, [1.583, -0.401, -1.99, 1.209, -0.511, -1.347, 1.132, 0.972, -1.597, 0.047], 1e-12);
    assert(qg3cgs.kind == scientific::F_NONE && qc3cgs == 0 && common::all_near(qx3cgs, [0.03613183575020873, 0.6293215511018997, -1.2900170248705976, -0.592107685940299, -0.28880254601621475, -0.28164529507417, -1.1982894987933845, -0.1338712695869561, -0.7240880525736972, -2.2885287853983933]));
    let qx3qmr, qc3qmr, qg3qmr = scientific::qmr(qa3, [1.583, -0.401, -1.99, 1.209, -0.511, -1.347, 1.132, 0.972, -1.597, 0.047], 1e-12);
    assert(qg3qmr.kind == scientific::F_NONE && qc3qmr == 0 && common::all_near(qx3qmr, [0.036131835749692275, 0.6293215511016151, -1.2900170248711857, -0.5921076859401089, -0.28880254601652233, -0.28164529507445213, -1.19828949879398, -0.13387126958686976, -0.7240880525738745, -2.2885287853987615]));
    let qx3tfqmr, qc3tfqmr, qg3tfqmr = scientific::tfqmr(qa3, [1.583, -0.401, -1.99, 1.209, -0.511, -1.347, 1.132, 0.972, -1.597, 0.047], 1e-12);
    assert(qg3tfqmr.kind == scientific::F_NONE && qc3tfqmr == 0 && common::all_near(qx3tfqmr, [0.0361318357502992, 0.6293215511018664, -1.290017024870684, -0.5921076859415267, -0.2888025460168668, -0.2816452950751879, -1.1982894987940638, -0.1338712695873903, -0.724088052573653, -2.288528785399647]));
    let qa4, qf4 = scientific::csr_from_dense(common_3::ni_nd([2, 2], [2.8435, 0.0, 0.359, 1.6145]));
    assert(qf4.kind == scientific::F_NONE);
    let qx4bicg, qc4bicg, qg4bicg = scientific::bicg(qa4, [0.689, -0.479], 1e-12);
    assert(qg4bicg.kind == scientific::F_NONE && qc4bicg == 0 && common::all_near(qx4bicg, [0.24230701600140667, -0.35056563564230725]));
    let qx4cgs, qc4cgs, qg4cgs = scientific::cgs(qa4, [0.689, -0.479], 1e-12);
    assert(qg4cgs.kind == scientific::F_NONE && qc4cgs == 0 && common::all_near(qx4cgs, [0.2423070160014067, -0.35056563564230714]));
    let qx4qmr, qc4qmr, qg4qmr = scientific::qmr(qa4, [0.689, -0.479], 1e-12);
    assert(qg4qmr.kind == scientific::F_NONE && qc4qmr == 0 && common::all_near(qx4qmr, [0.24230701600140675, -0.35056563564230714]));
    let qx4tfqmr, qc4tfqmr, qg4tfqmr = scientific::tfqmr(qa4, [0.689, -0.479], 1e-12);
    assert(qg4tfqmr.kind == scientific::F_NONE && qc4tfqmr == 0 && common::all_near(qx4tfqmr, [0.24230701600140672, -0.3505656356423072]));
    let qa5, qf5 = scientific::csr_from_dense(common_3::ni_nd([3, 3], [4.1146666666666665, 0.0, 0.0, 0.0, 2.2456666666666667, 0.0, 2.444, 2.727, -1.454333333333333]));
    assert(qf5.kind == scientific::F_NONE);
    let qx5bicg, qc5bicg, qg5bicg = scientific::bicg(qa5, [-0.451, 1.678, -0.618], 1e-12);
    assert(qg5bicg.kind == scientific::F_NONE && qc5bicg == 0 && common::all_near(qx5bicg, [-0.10960790667530787, 0.7472168621047943, 1.641837262694469]));
    let qx5cgs, qc5cgs, qg5cgs = scientific::cgs(qa5, [-0.451, 1.678, -0.618], 1e-12);
    assert(qg5cgs.kind == scientific::F_NONE && qc5cgs == 0 && common::all_near(qx5cgs, [-0.10960790667530779, 0.7472168621047943, 1.6418372626944688]));
    let qx5qmr, qc5qmr, qg5qmr = scientific::qmr(qa5, [-0.451, 1.678, -0.618], 1e-12);
    assert(qg5qmr.kind == scientific::F_NONE && qc5qmr == 0 && common::all_near(qx5qmr, [-0.10960790667530784, 0.7472168621047943, 1.6418372626944684]));
    let qx5tfqmr, qc5tfqmr, qg5tfqmr = scientific::tfqmr(qa5, [-0.451, 1.678, -0.618], 1e-12);
    assert(qg5tfqmr.kind == scientific::F_NONE && qc5tfqmr == 0 && common::all_near(qx5tfqmr, [-0.10960790667530818, 0.747216862104794, 1.6418372626944675]));
}

fn test_sparse_bicg_pins2() {
    let qa6, qf6 = scientific::csr_from_dense(common_3::ni_nd([6, 6], [1.3815000000000002, 0.0, -1.79, 0.0, 2.953, -1.786, -1.541, -0.8714999999999999, 0.0, 0.0, 0.0, -0.962, 1.998, -1.47, 1.3815000000000002, 0.0, 0.0, 2.757, 0.0, -1.41, 0.0, 1.3815000000000002, 0.0, 1.473, -2.924, -2.189, 1.628, 0.0, -0.3534999999999999, 2.663, 0.0, -1.355, -2.835, 2.921, 0.694, 3.0805000000000002]));
    assert(qf6.kind == scientific::F_NONE);
    let qx6bicg, qc6bicg, qg6bicg = scientific::bicg(qa6, [1.866, -1.443, -1.979, 0.614, 1.455, -1.401], 1e-12);
    assert(qg6bicg.kind == scientific::F_NONE && qc6bicg == 0 && common::all_near(qx6bicg, [-1.3232089715561879, 3.7548951809908973, 4.041666426083181, 4.044410210977896, 3.8326670808170116, 0.21795621095124962]));
    let qx6cgs, qc6cgs, qg6cgs = scientific::cgs(qa6, [1.866, -1.443, -1.979, 0.614, 1.455, -1.401], 1e-12);
    assert(qg6cgs.kind == scientific::F_NONE && qc6cgs == 0 && common::all_near(qx6cgs, [-1.3232089715565236, 3.7548951809909266, 4.041666426083336, 4.044410210977967, 3.832667080817233, 0.21795621095115453]));
    let qx6qmr, qc6qmr, qg6qmr = scientific::qmr(qa6, [1.866, -1.443, -1.979, 0.614, 1.455, -1.401], 1e-12);
    assert(qg6qmr.kind == scientific::F_NONE && qc6qmr == 0 && common::all_near(qx6qmr, [-1.3232089715557294, 3.7548951809909394, 4.0416664260829265, 4.044410210977799, 3.8326670808167616, 0.21795621095132245]));
    let qx6tfqmr, qc6tfqmr, qg6tfqmr = scientific::tfqmr(qa6, [1.866, -1.443, -1.979, 0.614, 1.455, -1.401], 1e-12);
    assert(qg6tfqmr.kind == scientific::F_NONE && qc6tfqmr == 0 && common::all_near(qx6tfqmr, [-1.3232089715565487, 3.7548951809909323, 4.041666426083346, 4.04441021097802, 3.832667080817228, 0.21795621095117598]));
    let qa7, qf7 = scientific::csr_from_dense(common_3::ni_nd([7, 7], [3.4842857142857144, 0.0, 0.176, -1.211, -0.051, -2.734, -1.405, 0.0, 2.2252857142857145, 0.0, 0.391, -0.152, 2.776, 0.0, 0.0, 0.0, 3.1142857142857148, -0.697, 0.0, 0.0, 0.0, -2.979, 0.0, 0.0, 4.146285714285715, -1.85, 1.645, 2.824, -1.812, 0.0, -2.732, 0.0, 3.8752857142857144, 0.0, 0.796, -1.279, 0.0, 0.0, -1.461, 0.0, 2.2252857142857145, 0.0, 0.0, 0.0, 0.226, -1.81, 0.0, 1.751, 5.083285714285715]));
    assert(qf7.kind == scientific::F_NONE);
    let qx7bicg, qc7bicg, qg7bicg = scientific::bicg(qa7, [-0.786, 1.302, 0.939, 0.24, 0.429, -1.656, 1.718], 1e-12);
    assert(qg7bicg.kind == scientific::F_NONE && qc7bicg == 0 && common::all_near(qx7bicg, [-24.666590588693115, 34.46868385520802, -3.3569431752086607, -16.346456490991553, -14.509071054900728, -25.653668618729018, 3.503473473908244]));
    let qx7cgs, qc7cgs, qg7cgs = scientific::cgs(qa7, [-0.786, 1.302, 0.939, 0.24, 0.429, -1.656, 1.718], 1e-12);
    assert(qg7cgs.kind == scientific::F_NONE && qc7cgs == 0 && common::all_near(qx7cgs, [-24.66659058869453, 34.46868385520963, -3.356943175208823, -16.346456490992445, -14.509071054901538, -25.653668618730396, 3.50347347390846]));
    let qx7qmr, qc7qmr, qg7qmr = scientific::qmr(qa7, [-0.786, 1.302, 0.939, 0.24, 0.429, -1.656, 1.718], 1e-12);
    assert(qg7qmr.kind == scientific::F_NONE && qc7qmr == 0 && common::all_near(qx7qmr, [-24.666590588690703, 34.46868385520422, -3.3569431752082943, -16.34645649098982, -14.509071054899351, -25.653668618726304, 3.503473473907909]));
    let qx7tfqmr, qc7tfqmr, qg7tfqmr = scientific::tfqmr(qa7, [-0.786, 1.302, 0.939, 0.24, 0.429, -1.656, 1.718], 1e-12);
    assert(qg7tfqmr.kind == scientific::F_NONE && qc7tfqmr == 0 && common::all_near(qx7tfqmr, [-24.666590588748566, 34.46868385529095, -3.3569431752183596, -16.346456491051427, -14.509071054967764, -25.65366861879362, 3.5034734739158564]));
    let qa8, qf8 = scientific::csr_from_dense(common_3::ni_nd([4, 4], [-0.11099999999999999, -2.733, 0.0, -1.861, -1.948, 1.641, 0.0, -2.081, 0.0, 0.0, 3.589, -0.079, 1.507, -0.313, -2.253, 4.009]));
    assert(qf8.kind == scientific::F_NONE);
    let qx8bicg, qc8bicg, qg8bicg = scientific::bicg(qa8, [1.645, 0.643, -1.135, 1.246], 1e-12);
    assert(qg8bicg.kind == scientific::F_NONE && qc8bicg == 0 && common::all_near(qx8bicg, [-2.230428223536853, -1.120800469379323, -0.29654206528507476, 0.8950699707825076]));
    let qx8cgs, qc8cgs, qg8cgs = scientific::cgs(qa8, [1.645, 0.643, -1.135, 1.246], 1e-12);
    assert(qg8cgs.kind == scientific::F_NONE && qc8cgs == 0 && common::all_near(qx8cgs, [-2.23042822353686, -1.1208004693792901, -0.2965420652850863, 0.8950699707824956]));
    let qx8qmr, qc8qmr, qg8qmr = scientific::qmr(qa8, [1.645, 0.643, -1.135, 1.246], 1e-12);
    assert(qg8qmr.kind == scientific::F_NONE && qc8qmr == 0 && common::all_near(qx8qmr, [-2.2304282235368724, -1.1208004693792544, -0.29654206528511745, 0.8950699707824686]));
    let qx8tfqmr, qc8tfqmr, qg8tfqmr = scientific::tfqmr(qa8, [1.645, 0.643, -1.135, 1.246], 1e-12);
    assert(qg8tfqmr.kind == scientific::F_NONE && qc8tfqmr == 0 && common::all_near(qx8tfqmr, [-2.2304282235368684, -1.120800469379311, -0.29654206528508864, 0.8950699707824962]));
}

fn test_sparse_bicg_pins3() {
    let qa9, qf9 = scientific::csr_from_dense(common_3::ni_nd([4, 4], [1.69375, -2.586, 0.0, 1.901, 0.0, 1.81075, 0.0, -1.988, 0.0, 2.586, 4.35175, 0.0, 0.0, -0.188, 2.125, 1.69375]));
    assert(qf9.kind == scientific::F_NONE);
    let qx9bicg, qc9bicg, qg9bicg = scientific::bicg(qa9, [0.204, 0.695, -0.448, 0.662], 1e-12);
    assert(qg9bicg.kind == scientific::F_NONE && qc9bicg == 0 && common::all_near(qx9bicg, [8.592538659753531, 16.015375207985272, -9.619982831699875, 14.237847413410115]));
    let qx9cgs, qc9cgs, qg9cgs = scientific::cgs(qa9, [0.204, 0.695, -0.448, 0.662], 1e-12);
    assert(qg9cgs.kind == scientific::F_NONE && qc9cgs == 0 && common::all_near(qx9cgs, [8.592538659752977, 16.015375207984636, -9.619982831699518, 14.237847413409646]));
    let qx9qmr, qc9qmr, qg9qmr = scientific::qmr(qa9, [0.204, 0.695, -0.448, 0.662], 1e-12);
    assert(qg9qmr.kind == scientific::F_NONE && qc9qmr == 0 && common::all_near(qx9qmr, [8.592538659754547, 16.015375207988495, -9.619982831701758, 14.237847413413053]));
    let qx9tfqmr, qc9tfqmr, qg9tfqmr = scientific::tfqmr(qa9, [0.204, 0.695, -0.448, 0.662], 1e-12);
    assert(qg9tfqmr.kind == scientific::F_NONE && qc9tfqmr == 0 && common::all_near(qx9tfqmr, [8.592538659750925, 16.015375207981982, -9.619982831697804, 14.23784741340732]));
    let qa10, qf10 = scientific::csr_from_dense(common_3::ni_nd([7, 7], [-1.66, 0.0, -0.393, -1.94, -2.332, 0.0, 0.761, 0.0, 1.9709999999999999, 0.747, 0.788, -2.247, -2.747, 2.614, 0.4, -1.417, 1.023, -0.265, 1.474, -1.256, 0.0, 2.599, 0.008, 2.289, 1.023, -0.062, -2.073, 0.0, 0.0, -0.14, -0.012, 1.321, 0.825, 0.0, 0.0, -0.13, -2.695, 1.934, 2.052, 0.0, 1.023, 0.057, -0.485, -0.124, 0.0, -0.341, 0.825, -0.908, 3.117]));
    assert(qf10.kind == scientific::F_NONE);
    let qx10bicg, qc10bicg, qg10bicg = scientific::bicg(qa10, [0.965, -0.406, 1.197, 0.155, 1.847, -0.94, 0.813], 1e-12);
    assert(qg10bicg.kind == scientific::F_NONE && qc10bicg == 0 && common::all_near(qx10bicg, [12.323715119839148, -16.83705386331473, -23.75247766102146, 4.703516873360129, -8.49522795455241, -8.341235072789685, 1.8417799243823074]));
    let qx10cgs, qc10cgs, qg10cgs = scientific::cgs(qa10, [0.965, -0.406, 1.197, 0.155, 1.847, -0.94, 0.813], 1e-12);
    assert(qg10cgs.kind == scientific::F_NONE && qc10cgs == 0 && common::all_near(qx10cgs, [12.323715119838292, -16.837053863313837, -23.75247766102088, 4.703516873360809, -8.495227954551822, -8.341235072789877, 1.8417799243819502]));
    let qx10qmr, qc10qmr, qg10qmr = scientific::qmr(qa10, [0.965, -0.406, 1.197, 0.155, 1.847, -0.94, 0.813], 1e-12);
    assert(qg10qmr.kind == scientific::F_NONE && qc10qmr == 0 && common::all_near(qx10qmr, [12.323715119838965, -16.837053863314093, -23.75247766102133, 4.703516873360842, -8.495227954552306, -8.341235072789251, 1.8417799243823687]));
    let qx10tfqmr, qc10tfqmr, qg10tfqmr = scientific::tfqmr(qa10, [0.965, -0.406, 1.197, 0.155, 1.847, -0.94, 0.813], 1e-12);
    assert(qg10tfqmr.kind == scientific::F_NONE && qc10tfqmr == 0 && common::all_near(qx10tfqmr, [12.32371511983892, -16.837053863313873, -23.752477661020798, 4.703516873360456, -8.495227954552236, -8.341235072789122, 1.8417799243823634]));
    let qa11, qf11 = scientific::csr_from_dense(common_3::ni_nd([3, 3], [1.528, 1.033, -1.928, 0.0, 3.112, 0.0, 0.28, 0.24, 1.528]));
    assert(qf11.kind == scientific::F_NONE);
    let qx11bicg, qc11bicg, qg11bicg = scientific::bicg(qa11, [-1.119, 0.787, -1.608], 1e-12);
    assert(qg11bicg.kind == scientific::F_NONE && qc11bicg == 0 && common::all_near(qx11bicg, [-1.852849294869131, 0.252892030848329, -0.7525499246336662]));
    let qx11cgs, qc11cgs, qg11cgs = scientific::cgs(qa11, [-1.119, 0.787, -1.608], 1e-12);
    assert(qg11cgs.kind == scientific::F_NONE && qc11cgs == 0 && common::all_near(qx11cgs, [-1.8528492948691313, 0.25289203084832906, -0.7525499246336663]));
    let qx11qmr, qc11qmr, qg11qmr = scientific::qmr(qa11, [-1.119, 0.787, -1.608], 1e-12);
    assert(qg11qmr.kind == scientific::F_NONE && qc11qmr == 0 && common::all_near(qx11qmr, [-1.8528492948691317, 0.2528920308483293, -0.7525499246336662]));
    let qx11tfqmr, qc11tfqmr, qg11tfqmr = scientific::tfqmr(qa11, [-1.119, 0.787, -1.608], 1e-12);
    assert(qg11tfqmr.kind == scientific::F_NONE && qc11tfqmr == 0 && common::all_near(qx11tfqmr, [-1.8528492948690685, 0.2528920308484842, -0.7525499246336324]));
}

fn test_sparse_bicg_pins4() {
    let qa12, qf12 = scientific::csr_from_dense(common_3::ni_nd([2, 2], [2.459, -1.131, -0.396, 5.097]));
    assert(qf12.kind == scientific::F_NONE);
    let qx12bicg, qc12bicg, qg12bicg = scientific::bicg(qa12, [-0.551, 1.779], 1e-12);
    assert(qg12bicg.kind == scientific::F_NONE && qc12bicg == 0 && common::all_near(qx12bicg, [-0.06589618247165428, 0.3439091841752452]));
    let qx12cgs, qc12cgs, qg12cgs = scientific::cgs(qa12, [-0.551, 1.779], 1e-12);
    assert(qg12cgs.kind == scientific::F_NONE && qc12cgs == 0 && common::all_near(qx12cgs, [-0.06589618247165417, 0.34390918417524513]));
    let qx12qmr, qc12qmr, qg12qmr = scientific::qmr(qa12, [-0.551, 1.779], 1e-12);
    assert(qg12qmr.kind == scientific::F_NONE && qc12qmr == 0 && common::all_near(qx12qmr, [-0.06589618247165421, 0.3439091841752453]));
    let qx12tfqmr, qc12tfqmr, qg12tfqmr = scientific::tfqmr(qa12, [-0.551, 1.779], 1e-12);
    assert(qg12tfqmr.kind == scientific::F_NONE && qc12tfqmr == 0 && common::all_near(qx12tfqmr, [-0.06589618247165428, 0.34390918417524513]));
    let qa13, qf13 = scientific::csr_from_dense(common_3::ni_nd([9, 9], [1.4155555555555557, 1.542, 0.819, -2.928, -0.103, 1.786, 1.618, 2.536, 0.0, -0.11, 0.4485555555555557, -0.249, -1.917, 1.06, 0.0, 0.0, 0.0, -2.704, 0.201, 0.0, 1.1665555555555556, 0.63, 2.53, 0.0, -0.782, -1.886, -2.412, -0.778, 0.0, -0.011, 1.4155555555555557, 0.408, -2.512, 2.605, -2.522, 0.779, 2.25, 0.0, -2.188, -1.988, 1.4155555555555557, 0.0, -1.81, -1.098, 0.0, 0.0, 1.565, -2.898, 2.654, -0.131, -1.0084444444444443, 1.998, 0.916, 0.0, 0.0, 0.203, -1.92, -0.811, 0.0, 0.0, 1.6695555555555557, -0.603, 2.87, 0.119, -2.999, 0.0, 0.0, 0.0, 0.223, 1.542, 1.4155555555555557, -1.266, 0.0, -2.786, 2.644, 0.0, -1.398, 1.969, 2.668, 0.0, 1.0615555555555556]));
    assert(qf13.kind == scientific::F_NONE);
    let qx13bicg, qc13bicg, qg13bicg = scientific::bicg(qa13, [-1.92, 1.298, -1.078, -1.342, -1.842, -0.803, -0.047, -0.197, -1.545], 1e-12);
    assert(qg13bicg.kind == scientific::F_NONE && qc13bicg == 0 && common::all_near(qx13bicg, [-1.8064217826565758, -0.08992646219332934, -0.4790389389937091, -0.39341888008207365, -0.19937756265546228, 0.2604266566789075, -0.424663916442248, 0.08580746214348582, -0.17659250946900157]));
    let qx13cgs, qc13cgs, qg13cgs = scientific::cgs(qa13, [-1.92, 1.298, -1.078, -1.342, -1.842, -0.803, -0.047, -0.197, -1.545], 1e-12);
    assert(qg13cgs.kind == scientific::F_NONE && qc13cgs == 0 && common::all_near(qx13cgs, [-1.8064217826568496, -0.08992646219328237, -0.47903893899362876, -0.39341888008196385, -0.19937756265551593, 0.2604266566789107, -0.4246639164423318, 0.08580746214351612, -0.17659250946905403]));
    let qx13qmr, qc13qmr, qg13qmr = scientific::qmr(qa13, [-1.92, 1.298, -1.078, -1.342, -1.842, -0.803, -0.047, -0.197, -1.545], 1e-12);
    assert(qg13qmr.kind == scientific::F_NONE && qc13qmr == 0 && common::all_near(qx13qmr, [-1.8064217826566873, -0.08992646219329131, -0.4790389389937423, -0.3934188800819531, -0.1993775626555231, 0.26042665667903675, -0.4246639164421962, 0.08580746214361494, -0.17659250946895771]));
    let qx13tfqmr, qc13tfqmr, qg13tfqmr = scientific::tfqmr(qa13, [-1.92, 1.298, -1.078, -1.342, -1.842, -0.803, -0.047, -0.197, -1.545], 1e-12);
    assert(qg13tfqmr.kind == scientific::F_NONE && qc13tfqmr == 0 && common::all_near(qx13tfqmr, [-1.8064217826568754, -0.08992646219327818, -0.479038938993614, -0.3934188800819861, -0.19937756265547948, 0.26042665667890247, -0.42466391644231616, 0.08580746214353108, -0.1765925094690494]));
    let qa14, qf14 = scientific::csr_from_dense(common_3::ni_nd([1, 1], [2.0]));
    assert(qf14.kind == scientific::F_NONE);
    let qx14bicg, qc14bicg, qg14bicg = scientific::bicg(qa14, [3.0], 1e-12);
    assert(qg14bicg.kind == scientific::F_NONE && qc14bicg == 0 && common::all_near(qx14bicg, [1.5]));
    let qx14cgs, qc14cgs, qg14cgs = scientific::cgs(qa14, [3.0], 1e-12);
    assert(qg14cgs.kind == scientific::F_NONE && qc14cgs == 0 && common::all_near(qx14cgs, [1.5]));
    let qx14qmr, qc14qmr, qg14qmr = scientific::qmr(qa14, [3.0], 1e-12);
    assert(qg14qmr.kind == scientific::F_NONE && qc14qmr == 0 && common::all_near(qx14qmr, [1.5]));
    let qx14tfqmr, qc14tfqmr, qg14tfqmr = scientific::tfqmr(qa14, [3.0], 1e-12);
    assert(qg14tfqmr.kind == scientific::F_NONE && qc14tfqmr == 0 && common::all_near(qx14tfqmr, [1.5]));
    let qa15, qf15 = scientific::csr_from_dense(common_3::ni_nd([2, 2], [4.0, 1.0, 1.0, 3.0]));
    assert(qf15.kind == scientific::F_NONE);
    let qx15bicg, qc15bicg, qg15bicg = scientific::bicg(qa15, [1.0, 2.0], 1e-12);
    assert(qg15bicg.kind == scientific::F_NONE && qc15bicg == 0 && common::all_near(qx15bicg, [0.09090909090909091, 0.6363636363636364]));
    let qx15cgs, qc15cgs, qg15cgs = scientific::cgs(qa15, [1.0, 2.0], 1e-12);
    assert(qg15cgs.kind == scientific::F_NONE && qc15cgs == 0 && common::all_near(qx15cgs, [0.09090909090909091, 0.6363636363636364]));
    let qx15qmr, qc15qmr, qg15qmr = scientific::qmr(qa15, [1.0, 2.0], 1e-12);
    assert(qg15qmr.kind == scientific::F_NONE && qc15qmr == 0 && common::all_near(qx15qmr, [0.09090909090909094, 0.6363636363636364]));
    let qx15tfqmr, qc15tfqmr, qg15tfqmr = scientific::tfqmr(qa15, [1.0, 2.0], 1e-12);
    assert(qg15tfqmr.kind == scientific::F_NONE && qc15tfqmr == 0 && common::all_near(qx15tfqmr, [0.09090909090909091, 0.6363636363636364]));
}

fn test_sparse_bicg_pins() {
    sparse_bicg_tests::test_sparse_bicg_pins0();
    sparse_bicg_tests::test_sparse_bicg_pins1();
    sparse_bicg_tests::test_sparse_bicg_pins2();
    sparse_bicg_tests::test_sparse_bicg_pins3();
    sparse_bicg_tests::test_sparse_bicg_pins4();
}

// test_sparse_bicg_hostile checks the faults of the biconjugate gradient
// solvers.
fn test_sparse_bicg_hostile() {
    let a, fa = scientific::csr_from_dense(common_3::ni_nd([3, 3], [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]));
    assert(fa.kind == scientific::F_NONE);
    let b = [1.0, 2.0, 3.0];
    // A matrix that is not square.
    let s, fs = scientific::csr_from_dense(common_3::ni_nd([2, 3], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
    assert(fs.kind == scientific::F_NONE);
    let x1, c1, g1 = scientific::bicg(s, b);
    assert(g1.kind == scientific::F_SHAPE);
    let x2, c2, g2 = scientific::cgs(s, b);
    assert(g2.kind == scientific::F_SHAPE);
    let x3, c3, g3 = scientific::qmr(s, b);
    assert(g3.kind == scientific::F_SHAPE);
    let x4, c4, g4 = scientific::tfqmr(s, b);
    assert(g4.kind == scientific::F_SHAPE);
    // A right side so small that its norm falls to zero is a fault.
    let x5, c5, g5 = scientific::bicg(a, [1.0e-170, 1.0e-170, 1.0e-170]);
    assert(g5.kind == scientific::F_VALUE);
    let x6, c6, g6 = scientific::cgs(a, [1.0e-170, 1.0e-170, 1.0e-170]);
    assert(g6.kind == scientific::F_VALUE);
    let x7, c7, g7 = scientific::qmr(a, [1.0e-170, 1.0e-170, 1.0e-170]);
    assert(g7.kind == scientific::F_VALUE);
    let x8, c8, g8 = scientific::tfqmr(a, [1.0e-170, 1.0e-170, 1.0e-170]);
    assert(g8.kind == scientific::F_VALUE);
    // A right side that does not fit.
    let x5, c5, g5 = scientific::bicg(a, [1.0, 2.0]);
    assert(g5.kind == scientific::F_SHAPE);
    let x6, c6, g6 = scientific::cgs(a, [1.0, 2.0]);
    assert(g6.kind == scientific::F_SHAPE);
    let x7, c7, g7 = scientific::qmr(a, [1.0, 2.0]);
    assert(g7.kind == scientific::F_SHAPE);
    let x8, c8, g8 = scientific::tfqmr(a, [1.0, 2.0]);
    assert(g8.kind == scientific::F_SHAPE);
    // A tolerance that holds a negative value.
    let x9, c9, g9 = scientific::bicg(a, b, -1.0);
    assert(g9.kind == scientific::F_VALUE);
    let x10, c10, g10 = scientific::cgs(a, b, -1.0);
    assert(g10.kind == scientific::F_VALUE);
    let x11, c11, g11 = scientific::qmr(a, b, -1.0);
    assert(g11.kind == scientific::F_VALUE);
    let x12, c12, g12 = scientific::tfqmr(a, b, -1.0);
    assert(g12.kind == scientific::F_VALUE);
    // A right side that is not finite, an overflow of its norm and a
    // tolerance that is not finite.
    let xa, ca, ga = scientific::bicg(a, [3.0, 1.0 / 0.0, 1.0]);
    assert(ga.kind == scientific::F_VALUE);
    let xb, cb, gb = scientific::cgs(a, [1.0e200, 1.0e200, 1.0e200]);
    assert(gb.kind == scientific::F_VALUE);
    let xc, cc, gc = scientific::qmr(a, [0.0 / 0.0, 1.0, 1.0]);
    assert(gc.kind == scientific::F_VALUE);
    let xd, cd, gd = scientific::tfqmr(a, b, 1.0 / 0.0);
    assert(gd.kind == scientific::F_VALUE);
    // A right side of zeros gives the zero solution.
    let x13, c13, g13 = scientific::bicg(a, [0.0, 0.0, 0.0]);
    assert(g13.kind == scientific::F_NONE && c13 == 0 && common_3::ms_same(x13, [0.0, 0.0, 0.0]));
    // A limit of one turn gives the count of the turns.
    let x14, c14, g14 = scientific::bicg(a, b, 1.0e-12, 1);
    assert(g14.kind == scientific::F_NONE && c14 == 1);
    // The product of the transposed matrix.
    let t, ft = scientific::csr_matvec_t(a, [1.0, 1.0, 1.0]);
    assert(ft.kind == scientific::F_NONE && common_3::ms_same(t, [2.0, 3.0, 4.0]));
    // A vector that does not fit the matrix.
    let u, fu = scientific::csr_matvec_t(a, [1.0, 1.0]);
    assert(fu.kind == scientific::F_SHAPE);
}
