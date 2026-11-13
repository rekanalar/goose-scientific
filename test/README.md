# The tests of the package

The tests live in group files. Each group holds the tests of one source file, or of a few related ones. You can run one group, one test, or all tests.

## Run the tests

Start every run in the package directory.

| Goal | Command |
|---|---|
| All tests | `goose run test/run.goose` |
| One group | `SCIENTIFIC_SUITES=qz goose run test/run.goose` |
| One test | `SCIENTIFIC_TESTS=test_qz goose run test/run.goose` |
| One test of one group | `SCIENTIFIC_SUITES=qz SCIENTIFIC_TESTS=test_qz goose run test/run.goose` |

A full run prints `ok` and exits with 0. A run of a group or of a test prints `ok` too.

A name that no group or test has stops the run. The run prints the names of the groups and exits with a status other than 0. A test name with a space or a wrong letter is an unknown name. When both variables are set, the test must belong to the group.

A test that another test calls runs inside that test in a full run. A run of its own group runs it again on its own.

## The files

| File | Content |
|---|---|
| `run.goose` | The entry. It reads the two variables and calls the index. |
| `suite/suite.goose` | The index. `gen_suite.py` writes it. Do not edit it by hand. |
| `suite/<group>_tests.goose` | The tests of one group. |
| `suite/<group>_2_tests.goose` | The next part of a group. A group of more than 1000 lines has more files. |
| `suite/common.goose` | The helpers that more than one group uses. A group calls them as `common::name`. |
| `suite/common_2.goose` | The next part of the common helpers. The first file holds the helpers that the other common files call. |
| `gen_suite.py` | Writes the index. `python3 -I test/gen_suite.py --list` shows each group with its tests. |

The group `dist` holds the files `dist_tests.goose` and `dist_2_tests.goose`. The group `sparse` holds the files of its parts. The name of a group is the name of its files without the `_tests` part.

A group file holds at most 1000 lines. The generator fails above that.

## Add a test

1. Write the test function in the group file of the source file that you test. The name starts with `test_` and no other test has it.
2. Run `python3 -I test/gen_suite.py`. This adds the test to the index.
3. Run the new test alone with `SCIENTIFIC_TESTS=<name>`.

Put a helper that two groups use into a common file. The common files hold every helper that the first split moved out of the group files.

To add a group, make the file `suite/<group>_tests.goose`. Its first lines are `namespace <group>_tests;` and `import scientific from "../..";`. Then run the generator.

The generator fails when two files hold the same test name. It fails too when a file is too long.

## Print the pins

Each `gen_*.py` script prints Goose source from NumPy or SciPy. Paste the output into the file in this table.

| Script | File |
|---|---|
| `gen_are.py` | `test/suite/dare_singular_tests.goose` |
| `gen_ceig.py` | `test/suite/ceig_tests.goose`, `test/suite/ceigh_hostile_tests.goose` |
| `gen_ceig_pair.py` | `test/suite/ceig_pair_tests.goose` |
| `gen_cfactor.py` | `test/suite/cfactor_tests.goose` |
| `gen_complex.py` | `test/suite/carray_tests.goose` |
| `gen_cossin.py` | `test/suite/cossin_tests.goose` |
| `gen_dist.py` | `test/suite/dist_pins_tests.goose` |
| `gen_generic.py` | `test/suite/generic_pins_tests.goose` |
| `gen_geometric.py` | `test/suite/geometric_pins_tests.goose` |
| `gen_grey.py` | `test/suite/grey_pins_tests.goose` |
| `gen_interp.py` | `test/suite/interp_tests.goose`, `test/suite/interp1d_kinds_tests.goose`, `test/suite/interpn_tests.goose` |
| `gen_measure.py` | `test/suite/measure_pins_tests.goose` |
| `gen_morph.py` | `test/suite/morph_pins_tests.goose` |
| `gen_ndimage.py` | `test/suite/ndimage_pins_tests.goose` |
| `gen_pad.py` | `test/suite/random_tests.goose` |
| `gen_qz.py` | `test/suite/qz_tests.goose`, `test/suite/ordqz_tests.goose`, `test/suite/qz_c_tests.goose` |
| `gen_rank.py` | `test/suite/rank_pins_tests.goose` |
| `gen_sparse_bicg.py` | `test/suite/sparse_bicg_tests.goose` |
| `gen_sparse_bmat.py` | `test/suite/sparse_bmat_tests.goose` |
| `gen_sparse_bsr.py` | `test/suite/sparse_bsr_2_tests.goose` |
| `gen_sparse_coo.py` | `test/suite/sparse_coo_2_tests.goose` |
| `gen_sparse_csc.py` | `test/suite/sparse_csc_2_tests.goose` |
| `gen_sparse_dia.py` | `test/suite/sparse_dia_2_tests.goose` |
| `gen_sparse_dok.py` | `test/suite/sparse_dok_tests.goose` |
| `gen_sparse_expm.py` | `test/suite/sparse_expm_tests.goose` |
| `gen_sparse_kronsum.py` | `test/suite/sparse_kronsum_tests.goose`, `test/suite/sparse_tests.goose` |
| `gen_sparse_lgmres.py` | `test/suite/sparse_lgmres_tests.goose` |
| `gen_sparse_lil.py` | `test/suite/sparse_lil_tests.goose` |
| `gen_sparse_lsmr.py` | `test/suite/sparse_lsmr_tests.goose` |
| `gen_sparse_lu.py` | `test/suite/sparse_lu_tests.goose` |
| `gen_stacks.py` | `test/suite/stacks_tests.goose` |
| `gen_transform.py` | `test/suite/transform_pins_tests.goose` |
| `gen_vectorized.py` | `test/suite/vectorized_pins_tests.goose` |

Each script names the Python packages and versions that it needs in its first comment lines.

## Limits

- A run of one test compiles the whole program first. The compile time of about 21 to 24 seconds stays. Only the run time of the tests drops.
- The test code takes no input from outside. It runs fixed arrays and fixed numbers.
- The path of the longest test file is below 200 characters. A deep checkout path on Windows can still reach the limit of 260 characters.
