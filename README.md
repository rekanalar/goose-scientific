# scientific

[![ci](https://github.com/rekanalar/goose-scientific/actions/workflows/ci.yml/badge.svg)](https://github.com/rekanalar/goose-scientific/actions/workflows/ci.yml)

Numeric arrays for the [Goose](https://github.com/aardappel/goose)
language. The package holds the N-dimensional array of `f64` numbers.
It holds the operations that broadcast over the array and the
reductions to one number. It uses the standard library only.

The array stores its elements in one flat buffer in row-major order.
An operation builds its result in the value of the caller:
`let c, f = scientific::add(a, b);` fills `c`. A read of an array
binds it by reference and copies nothing.

## Install

Add this to the root file of the program that uses it:

```goose
import scientific from "rekanalar/goose-scientific@v0.166.0";
```

## API

| Area | Functions |
|---|---|
| Creation | `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `identity`, `diag`, `diagflat`, `logspace`, `geomspace`, `from_rows`, `from_values`, `clone` |
| Stacks | `vstack`, `hstack`, `dstack`, `column_stack` |
| Grids | `meshgrid(x, y, indexing="xy", sparse=false)`, `cross`, `interpn` |
| Constants | `constants::pi`, `constants::c`, `constants::h` and the rest of the module of SciPy, `value`, `unit`, `precision`, `find`, `convert_temperature`, `lambda2nu`, `nu2lambda` |
| Moves | `moveaxis`, `moveaxis_many`, `flipud`, `fliplr`, `rot90(a, k=1, axis0=0, axis1=1)`, `broadcast_to`, `broadcast_arrays`, `select`, `apply_along_axis`, `apply_along_axis_scalar` |
| Positions | `argwhere`, `flatnonzero` |
| Histograms of two dimensions | `histogram2d(x, y, xbins=10, ybins=10)`, `histogramdd(points, bins=10)` |
| Buffer | `frombuffer(data, count=-1, offset=0)`, `fromfile(path, count=-1)` |
| Closeness | `isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=false)`, `allclose(...)`, `array_equal(a, b, equal_nan=false)` |
| Shape | `reshape`, `ravel`, `transpose`, `transpose_perm`, `squeeze`, `expand_dims`, `concatenate`, `stack`, `broadcast_shapes`, `index` |
| Elementwise | `add`, `sub`, `mul`, `div`, `power`, `maximum`, `minimum`, `neg`, `abs`, `sqrt`, `exp`, `log`, `log2`, `log10`, `scale`, `clip`, `sign`, `floor`, `ceil`, `round`, `trunc`, `rint`, `fix`, `square`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`, `expm1`, `log1p`, `reciprocal`, `deg2rad`, `rad2deg`, `signbit`, `isneginf`, `isposinf`, `floor_divide`, `mod`, `fmod`, `cbrt`, `hypot`, `exp2`, `nextafter`, `copysign`, `spacing` |
| Names of NumPy | `array`, `subtract`, `multiply`, `divide`, `true_divide`, `negative`, `positive`, `fabs`, `absolute`, `around` |
| Logic | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `logical_and`, `logical_or`, `logical_xor`, `logical_not`, `isnan`, `isinf`, `isfinite`, `all`, `any`, `count_nonzero`, `where` |
| Reductions | `sum`, `prod`, `min`, `max`, `mean`, `argmin`, `argmax` |
| Reductions along an axis | `sum_axis`, `prod_axis`, `min_axis`, `max_axis`, `mean_axis`, `var_axis`, `std_axis` |
| Reductions that skip NaN | `nansum`, `nanprod`, `nanmin`, `nanmax`, `nanmean`, `nanvar`, `nanstd`, `nansum_axis`, `nanprod_axis`, `nanmin_axis`, `nanmax_axis`, `nanmean_axis`, `nanvar_axis`, `nanstd_axis`, `nan_to_num` |
| Running and spread | `cumsum`, `cumprod`, `diff`, `flip`, `variance`, `std`, `ptp` |
| Indexing | `slice_axis`, `take`, `put`, `nonzero`, `mask`, `take_along_axis`, `repeat`, `tile`, `flatten`, `partition` |
| Sets | `union1d`, `intersect1d`, `setdiff1d`, `in1d`, `isin` |
| Sorting | `sort`, `argsort`, `sort_axis`, `unique`, `unique_sorted`, `roll` |
| Search | `searchsorted`, `histogram`, `histogram_edges`, `bincount`, `digitize` |
| Order statistics | `median`, `percentile`, `quantile`, `mode`, `average`, `median_axis`, `quantile_axis`, `percentile_axis`, `average_axis` |
| Products | `matmul`, `dot`, `inner`, `outer`, `kron`, `trace`, `norm` |
| Solving | `solve`, `inv`, `det`, `slogdet`, `matrix_power` |
| Decompositions | `cholesky`, `qr`, `schur`, `ldl` |
| Eigenvalues | `eigvals`, `eig`, `eigvalsh`, `eigh` |
| Singular values | `svd`, `svdvals`, `lstsq`, `matrix_rank`, `cond`, `pinv(a, rtol=-1)` |
| Polynomials | `polyval`, `polyfit`, `roots` |
| Complex numbers | `Cplx`, `c_add`, `c_sub`, `c_mul`, `c_div`, `c_abs`, `c_conj`, `c_scale`, `c_value`, `roots_of` |
| Polynomial type | `polynomial`, `polynomial_call`, `polynomial_degree`, `polynomial_add`, `polynomial_sub`, `polynomial_mul`, `polynomial_pow`, `polynomial_divmod`, `polynomial_der`, `polynomial_integ`, `polynomial_roots`, `polynomial_fromroots`, `polynomial_fit`, `polynomial_cutdeg`, `polynomial_linspace` |
| Transforms | `fft`, `ifft`, `fft2`, `fft2c`, `ifft2`, `fftn`, `ifftn`, `fftn_ex`, `ifftn_ex`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, `rfft`, `irfft` |
| Sampling | `default_rng`, `uniform`, `random_uniform`, `integers`, `normal`, `exponential`, `poisson`, `choice`, `permutation`, `shuffled`, `shuffle_rows` |
| Names of NumPy | `seed`, `shuffle`, `random_sample`, `rand`, `randn`, `randint`, `bytes`, `get_state`, `set_state` |
| Shapes of another array | `empty`, `full_like`, `zeros_like`, `ones_like`, `trim_zeros`, `insert` |
| Writes | `putmask`, `place`, `put_along_axis` |
| Cuts | `split`, `array_split`, `hsplit`, `vsplit`, `part` |
| Triangles | `tril`, `triu` |
| Between the points | `interp`, `gradient`, `pad_value` |
| Root finding | `brentq`, `brenth`, `newton`, `halley`, `secant`, `toms748`, `fsolve`, `root`, `root_scalar` |
| Minimization | `minimize_scalar`, `minimize`, `minimize_grad`, `minimize_constrained`, `differential_evolution`, `dual_annealing`, `basinhopping`, `least_squares`, `curve_fit` |
| Integration | `trapezoid`, `trapezoid_axis`, `simpson`, `simpson_axis`, `cumulative_trapezoid`, `cumulative_trapezoid_axis`, `cumulative_simpson`, `cumulative_simpson_dx`, `cumulative_simpson_axis`, `romberg` |
| Interpolation | `interp`, `spline`, `spline_value`, `spline_values` |
| Signals | `convolve`, `correlate`, `detrend`, `welch` |
| Filters | `butter`, `lfilter`, `lfilter_axis`, `lfilter_state`, `lfilter_zi`, `filtfilt`, `sosfilt`, `hilbert` |
| Covariance and NaN statistics | `cov`, `cov_weights`, `cov_with`, `corrcoef`, `corrcoef_with`, `nanmedian`, `nanpercentile`, `nanquantile`, `nanargmax`, `nanargmin`, `nancumsum`, `nancumprod`, `cumulative_sum`, `cumulative_prod`, `ediff1d`, `histogram_bin_edges` |
| The rest of the elementwise math | `fmax`, `fmin`, `heaviside`, `logaddexp`, `logaddexp2`, `ldexp`, `frexp`, `modf`, `sinc`, `i0`, `gcd`, `lcm`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `invert`, `left_shift`, `right_shift`, `bitwise_count`, `unwrap`, and the names `float_power`, `pow`, `remainder`, `radians`, `degrees`, `acos`, `asin`, `atan`, `atan2`, `acosh`, `asinh`, `atanh`, `bitwise_not`, `bitwise_invert`, `bitwise_left_shift`, `bitwise_right_shift`, `concat`, `amax`, `amin`, `ndim` |
| Assembly of shapes | `swapaxes`, `permute_dims`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `append`, `append_axis`, `delete`, `delete_axis`, `resize`, `pad_modes`, `pad_axes`, `diagonal`, `fill_diagonal`, `tri`, `tril_indices`, `triu_indices`, `diag_indices`, `indices`, `ix_`, `unravel_index`, `ravel_multi_index`, `unstack`, `unstack_part`, `block_grid` |
| Selection and order | `lexsort`, `argpartition`, `compress`, `compress_axis`, `extract`, `choose`, `piecewise`, `setxor1d`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values` |
| Products, windows and old polynomial functions | `vdot`, `vecdot`, `matvec`, `vecmat`, `vander`, `tensordot`, `tensordot_n`, `einsum`, `einsum1`, `einsum3`, `einsum_set`, `ESet`, `eset_new`, `eset_add`, `hanning`, `hamming`, `blackman`, `bartlett`, `kaiser`, `general_cosine`, `general_hamming`, `general_gaussian`, `taylor`, `chebwin`, `kaiser_bessel_derived`, `dpss`, `dpss_ratios`, `get_window`, `poly`, `poly_cplx`, `polyadd`, `polysub`, `polymul`, `polydiv`, `polyder`, `polyint` |
| Factors, special matrices and the exponential | `lu_factor`, `lu`, `lu_solve`, `solve_triangular`, `solve_banded`, `block_diag`, `toeplitz`, `hankel`, `circulant`, `companion`, `fiedler_companion`, `hadamard`, `pascal`, `invpascal`, `invhilbert`, `leslie`, `fiedler`, `helmert`, `convolution_matrix`, `khatri_rao`, `bandwidth`, `bandwidth_stack`, `issymmetric`, `ishermitian`, `solve_toeplitz`, `solve_circulant`, `matmul_toeplitz`, `diagsvd`, `orthogonal_procrustes`, `subspace_angles`, `pinvh`, `null_space`, `orth`, `expm` |
| Cosine, sine and real transforms | `dct`, `idct`, `dst`, `idst`, `dct_axis`, `idct_axis`, `dst_axis`, `idst_axis`, `dctn`, `idctn`, `dstn`, `idstn`, `hfft`, `ihfft`, `hfft_axis`, `ihfft_axis`, `next_fast_len`, `rfft2`, `rfftn`, `irfft2`, `irfftn`, `hfft2`, `ihfft2`, `hfftn`, `ihfftn`, `hfftn_ex`, `ihfftn_ex`, `fht`, `ifht`, `fhtoffset` |
| Distances, nearest points and hulls | `cdist`, `pdist`, `squareform`, `distance`, `euclidean`, `sqeuclidean`, `cityblock`, `chebyshev`, `minkowski`, `cosine_distance`, `correlation`, `hamming_distance`, `jaccard`, `canberra`, `braycurtis`, `seuclidean`, `mahalanobis`, `dice`, `russellrao`, `rogerstanimoto`, `sokalsneath`, `sokalmichener`, `yule`, `kdtree`, `kdtree_query`, `kdtree_query_ball_point`, `convex_hull`, `delaunay`, `delaunay_simplices`, `delaunay_find_simplex`, `griddata`, `nd_interpolator`, `nd_interpolate` |
| Filters of N dimensions | `correlate1d`, `convolve1d`, `uniform_filter1d`, `gaussian_filter1d`, `correlate_nd`, `convolve_nd`, `uniform_filter`, `gaussian_filter` |
| Rank and derivative filters of N dimensions | `rank_filter`, `median_filter`, `percentile_filter`, `minimum_filter`, `maximum_filter`, the forms `rank_filter_fp`, `median_filter_fp`, `percentile_filter_fp`, `minimum_filter_fp` and `maximum_filter_fp`, `minimum_filter1d`, `maximum_filter1d`, `sobel`, `prewitt`, `laplace`, `gaussian_laplace`, `gaussian_gradient_magnitude` |
| Labels, morphology and distances of N dimensions | `generate_binary_structure`, `iterate_structure`, `label`, `label_st`, `binary_erosion`, `binary_dilation`, `binary_opening`, `binary_closing`, the forms with the suffix `_st`, `binary_fill_holes`, `binary_fill_holes_st`, `distance_transform_edt`, `distance_transform_edt_indices` |
| Filters that take a block | `generic_filter`, `generic_filter_fp`, `generic_laplace`, `generic_gradient_magnitude` |
| Measurements of labeled regions | `sum_labels`, `mean_labels`, `variance_labels`, `standard_deviation_labels`, `minimum_labels`, `maximum_labels`, `median_labels`, `minimum_position`, `maximum_position`, `extrema`, `center_of_mass`, `histogram_labels`, `find_objects`, `labeled_comprehension` |
| Grey morphology, propagation and hit or miss | `grey_erosion`, `grey_dilation`, `grey_opening`, `grey_closing`, `morphological_gradient`, `morphological_laplace`, `white_tophat`, `black_tophat`, the forms with the suffix `_st`, `binary_propagation`, `binary_propagation_st`, `binary_hit_or_miss`, `binary_hit_or_miss_st` |
| Chamfer and brute force distances and value positions | `distance_transform_cdt`, `distance_transform_cdt_indices`, `distance_transform_cdt_st`, `distance_transform_cdt_st_indices`, `distance_transform_bf`, `distance_transform_bf_indices`, `value_indices` |
| Spline filters and interpolating transforms of N dimensions | `spline_filter1d`, `spline_filter`, `map_coordinates`, `affine_transform`, `shift`, `zoom`, `rotate` |
| Geometric transform and line filter of N dimensions | `geometric_transform`, `generic_filter1d` |
| Vectorized filter of N dimensions | `vectorized_filter` |
| Cubic splines and `interp1d` | `cubic_spline`, `cubic_spline_ends`, `cubic_hermite`, `bspline`, `make_interp_spline`, `make_interp_spline_bc`, `splrep`, `splrep_residual`, `splev`, `bspline_eval`, `bspline_design_matrix`, `pchip`, `akima`, `spline_eval`, `spline_integrate`, `interp1d` |
| Adaptive quadrature and ODEs | `quad`, `fixed_quad`, `dblquad`, `tplquad`, `nquad`, `quad_vec`, `solve_ivp`, `solve_ivp_jac`, `solve_ivp_events`, `solve_ivp_dense`, `dense_eval`, `odeint` |
| Programs of optimization | `linprog`, `milp`, `nnls`, `linear_sum_assignment`, `approx_fprime`, `bisect`, `ridder` |
| Signal design and analysis | `firwin`, `firwin2`, `remez`, `cheby1`, `cheby2`, `ellip`, `bessel`, `residue`, `residue_complex`, `residuez`, `invres`, `invresz`, `unique_roots`, `freqz`, `sosfreqz`, `sosfreqz_at`, `resample_poly`, `decimate`, `lfiltic`, `fftconvolve`, `medfilt`, `savgol_coeffs`, `savgol_filter`, `find_peaks`, `peak_prominences`, `periodogram`, `spectrogram`, `welch_psd`, `welch_ex`, `stft`, `istft`, `resample` |
| Sparse matrices of coordinates | `Coo`, `coo_from_triplets`, `coo_from_dense`, `coo_to_dense`, `coo_to_csr`, `coo_from_csr`, `coo_transpose`, `coo_get`, `coo_scale`, `coo_matvec` |
| Sparse matrices of diagonals | `Dia`, `dia_from_dense`, `dia_to_dense`, `dia_from_csr`, `dia_to_csr`, `dia_from_coo`, `dia_to_coo`, `dia_get`, `dia_scale`, `dia_matvec`, `dia_transpose`, `dia_sum`, `dia_nnz`, `dia_diagonals`, `dia_from_diagonals` |
| Sparse matrices of keys | `Dok`, `dok_new`, `dok_from_dense`, `dok_from_csr`, `dok_from_coo`, `dok_to_dense`, `dok_to_csr`, `dok_to_coo`, `dok_get`, `dok_set`, `dok_pop`, `dok_setdiag`, `dok_resize`, `dok_transpose`, `dok_scale`, `dok_matvec`, `dok_diagonal`, `dok_sum`, `dok_nnz`, `dok_keys` |
| Sparse matrices of blocks | `Bsr`, `bsr_from_entries`, `bsr_from_dense`, `bsr_to_dense`, `bsr_from_csr`, `bsr_to_csr`, `bsr_from_coo`, `bsr_to_coo`, `bsr_get`, `bsr_scale`, `bsr_matvec`, `bsr_transpose`, `bsr_sum`, `bsr_nnz`, `bsr_diagonal`, `bsr_blocks` |
| Sparse matrices of compressed columns | `Csc`, `csc_from_dense`, `csc_from_triplets`, `csc_to_dense`, `csc_eye`, `csc_diags`, `csc_transpose`, `csc_get`, `csc_scale`, `csc_matvec`, `csc_add`, `csc_sub`, `csc_matmul`, `csc_diagonal`, `csc_sum`, `csc_from_csr`, `csc_to_csr` |
| Sparse matrices of compressed rows | `Csr`, `csr_from_dense`, `csr_from_triplets`, `csr_to_dense`, `csr_eye`, `csr_diags`, `csr_transpose`, `csr_get`, `csr_scale`, `csr_matvec`, `csr_add`, `csr_sub`, `csr_matmul`, `csr_diagonal`, `csr_sum`, `csr_kron`, `csr_kronsum`, `csr_hstack`, `csr_vstack`, `csr_block_diag`, `csr_tril`, `csr_triu`, `csr_multiply`, `csr_slice`, `spsolve`, `cg`, `bicgstab`, `bicg`, `cgs`, `qmr`, `tfqmr`, `gmres`, `lsqr`, `lsmr`, `Lsmr`, `minres`, `eigsh`, `csr_matvec_t` |
| Functions of matrices | `sqrtm`, `logm`, `sinm`, `cosm`, `tanm`, `sinhm`, `coshm`, `tanhm`, `signm`, `fractional_matrix_power`, `polar`, `cho_factor`, `cho_solve`, `funm`, `solve_sylvester`, `solve_continuous_lyapunov`, `solve_lyapunov`, `solve_discrete_lyapunov`, `eig_banded`, `eigvals_banded` |
| Tridiagonal, banded and RQ forms | `eigvalsh_tridiagonal`, `eigh_tridiagonal`, `cholesky_banded`, `cho_solve_banded`, `solveh_banded`, `rq` |
| DFT matrix and QR product | `dft`, `qr_multiply` |
| Balance of a matrix and the exponential | `matrix_balance`, `matrix_balance_separate`, `expm_frechet`, `expm_cond` |
| Update of the QR decomposition | `qr_update`, `qr_insert`, `qr_delete` |
| Complex Schur form and real block form | `rsf2csf`, `cdf2rdf` |
| Cosine-sine decomposition | `cossin`, `cossin_separate`, `cossin_blocks` |
| Generalized Schur form of a pair | `qz`, `ordqz`, `qz_complex`, `ordqz_complex`, `qz_c`, `ordqz_c`, `ordqz_by`, `ordqz_complex_by`, `ordqz_c_by` |
| Algebraic Riccati equations | `solve_continuous_are`, `solve_discrete_are`, `solve_discrete_are_pencil` |
| Complex arrays | `CArr`, `complex_array`, `complex_real`, `complex_from_values`, `real`, `imag`, `angle`, `conj`, `iscomplex`, `isreal`, `cabs`, `cadd`, `csub`, `cmul`, `cdiv`, `cpower`, `cneg`, `cexp`, `clog`, `csqrt`, `csin`, `ccos`, `csinh`, `ccosh`, `ctranspose`, `adjoint`, `cmatmul`, `cdot`, `cvdot`, `csolve`, `cinv`, `cdet`, `ccholesky`, `cfft`, `cifft`, `cfftn`, `cifftn`, `cqr`, `csvd`, `ceig`, `ceigvals`, `ceigh`, `ceigvalsh`, `eigvals_complex`, `ceig_pair`, `ceigvals_pair`, `ceigh_pair`, `ceigvalsh_pair`, `eig_pair`, `eigvals_pair`, `eigh_pair`, `eigvalsh_pair` |
| Sections of a filter | `butter_sos`, `cheby1_sos`, `cheby2_sos`, `bessel_sos`, `zpk2sos`, `tf2sos`, `sosfilt_zi`, `sosfiltfilt` |
| Type and faults | `Arr`, `Fault`, `F_NONE`, `F_SHAPE`, `F_RANGE`, `F_LIMIT`, `F_VALUE`, `kind_name`, `numel`, `size` |

Every function that can fail gives `Arr, Fault` or `f64, Fault`.
The fault holds a kind, the name of the check and a short detail.

## The array type

`Arr` holds a count of dimensions, a shape, and the elements. The
shape holds at most 8 dimensions. Every dimension holds at least one
element. The count of elements of one array stays below `MAX_ELEMENTS`.

Assignment binds a name and copies nothing: two names of one array
share their elements. `clone` builds an independent array.

## Broadcasting

A binary operation applies the broadcast rule of NumPy. The shapes
align at the last dimension. At each position the lengths are equal,
or one side holds 1, or one side has no position left. A matrix and a
row add, subtract, multiply and divide like in NumPy:

```goose
let a, f = scientific::from_rows(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let row, f2 = scientific::linspace(10.0, 30.0, 3);
let s, f3 = scientific::add(a, row);   // shape 2 x 3
```

## Limits

- `eigh` and `eig` gave wrong eigenvectors for a symmetric matrix.
  They did not solve the eigenproblem, and they sat in rows. The old form
  held the eigenvector i in the row i, with a wrong rotation. The new
  form holds it in the column i, as NumPy does, and each column
  solves `a v = w v`. The eigenvalues did not change.
- `sqrtm` and `logm` take real matrices without an eigenvalue on the
  negative real axis. A result that needs a complex number is a fault.
  `sqrtm` takes a side up to 512.
- `sinm`, `cosm` and `tanm` use the real block form of `exp(iA)`, so
  the side stops at 128. `signm` takes a matrix with no eigenvalue on
  the imaginary axis. Else it gives a fault.
- `CArr` keeps the real and the imaginary part of each element side by
  side in one buffer. The functions of a complex array carry the
  prefix `c`, except `real`, `imag`, `angle`, `conj`, `iscomplex`,
  `isreal` and `adjoint`. The real functions do not take a `CArr`.
  `iscomplex` and `isreal` give arrays of one and zero, not booleans.
- `csolve` and `cinv` solve the real system of twice the side, so the
  side stays at 1024 or below. `cdet` takes a side up to 2048
  and gives the real and the imaginary part. The complex matrices have no `eig`.
  `eigvals_complex` takes a real matrix and gives no eigenvectors. `cfft` and
  `cifft` take one dimension.
- `cdot` takes vectors and matrices only. A count of dimensions above
  two gives a fault. NumPy takes more than two dimensions. It sums the
  products over the last axis of the first operand. It sums over the
  second to last axis of the second. The package gives that difference.
- `cvdot` flattens both operands and conjugates the first one, as
  NumPy does. It gives the real part and the imaginary part of the
  scalar. The package holds no call of a complex array and a real
  array. NumPy accepts that pair.
- `ccholesky` reads the lower triangle of the operand and takes the
  real part of the diagonal, as NumPy does. SciPy reads the upper
  triangle by default. SciPy gives a fault for a matrix that is not
  Hermitian. The package follows NumPy.
- `cfftn` and `cifftn` transform every axis. They take no choice of
  axes and no shape.
- `cqr` gives the reduced form. The diagonal of the upper triangle
  holds real values, and the sign follows LAPACK, as NumPy does. It
  takes no `mode` and no pivoting.
- `csvd` gives the full left and right
  vectors and the singular values from the largest to the smallest.
  NumPy fixes no phase for these vectors. The package makes the
  largest component of each left vector real and positive. The vectors
  of a repeated singular value differ from the vectors of NumPy. The
  singular values agree.
- `csvd` takes no `full_matrices` choice and no `compute_uv` choice.
  Both `cqr` and `csvd` take a side up to 512. The Jacobi method of
  `csvd` runs at most 60 sweeps. A matrix of random values of side 512
  took about 70 seconds. A matrix that needs all 60 sweeps takes longer.
  Cap the side of untrusted input.
- `ceig` and `ceigvals` give the eigenvalues in the order of the
  diagonal of the Schur form. NumPy calls its order unordered, so
  compare sorted values. Each vector has length one with its largest
  component real and above zero. A defective matrix gives vectors that
  are nearly parallel, as LAPACK does. A matrix that needs more than
  30 steps for one eigenvalue gives a fault.
- `ceigh` and `ceigvalsh` read one triangle, `"L"` or `"U"` in either
  case, and take the real part of the diagonal. The eigenvalues rise.
  NumPy fixes no phase for the vectors. The package makes the largest
  component of each vector real and above zero. The vectors of a
  repeated eigenvalue differ from NumPy. The Jacobi method runs at most
  60 sweeps.
- An unknown `uplo` is a fault.
- The four eigen calls take a side up to 256. A matrix of random
  values of side 256 took about 1.4 seconds for each of `ceig` and
  `ceigh`.
- `ceig_pair` gives alpha, beta and the vectors of a pair of matrices.
  The eigenvalue is alpha divided by beta. The order follows the
  generalized Schur form, and SciPy fixes no order. Each vector has
  length one with a real largest component above zero. SciPy scales
  its vectors in another way, so compare after the same rule.
- `ceigvals_pair` gives the real part infinity and the imaginary part
  zero for a zero beta. A zero alpha with a zero beta gives not a
  number, as SciPy does for a singular pair.
  An eigenvalue beyond the number range also gives infinity. The vectors of
  a singular pair stay finite.
- `ceigh_pair` needs a positive definite B and gives a fault otherwise.
  It reads one triangle of both operands. The vectors satisfy
  V^H B V equal to the unit matrix. The largest component of each
  vector is real and above zero. SciPy fixes no phase.
- The eight pair calls take a side up to 256 and no batch. A pair of
  random values of side 256 took about 1.4 seconds for `ceig_pair` and
  about 0.8 seconds for `ceigh_pair`.
- `solve_sylvester` and the two Lyapunov solvers build the Kronecker
  form. The product of the sides stays at 4096 or below.
- `schur` holds the real form. The form is not unique, so the signs
  of the columns of Z and the rows of T stay unpinned. A side above
  4096 exceeds the turn limit and gives a fault.
- `funm` takes the function of the caller as a block. A pair of
  complex eigenvalues gives a fault. Two diagonal values closer than
  1e-12 of the scale give a fault too. The caller needs a matrix with
  distinct real values.
- `ldl` holds no pivoting. A matrix that needs pivoting gives a
  fault, while SciPy returns a permutation with it. A symmetric
  matrix with a zero leading minor gives that fault.
- `eig_banded` and `eigvals_banded` expand the matrix, so the side
  stays at 2048 or below. They take `select` with `a`, `v` and `i`.
  They take no general band. A `v` range that holds no value is a
  fault, while SciPy returns an empty result. The sign of an
  eigenvector may differ from SciPy.
- `bicgstab` and `gmres` solve a square system from a start of zero.
  They give the solution and a count: zero on success. They take no
  preconditioner, no start vector and no callback. The path of the
  iteration may differ from SciPy, so only the solution matches.
- `bicg`, `cgs`, `qmr` and `tfqmr` solve a square system from a start
  of zero. They give the solution and a count: zero on success. The
  count holds the turns when the limit ran out. A breakdown gives a
  fault, while SciPy gives a negative count. The path of the iteration
  may differ from SciPy, so only the solution matches.

  A right side that is not finite is a fault. A tolerance that is not
  finite is a fault too. A right side whose norm falls to zero, or
  grows past the largest number, is a fault. SciPy returns the zero
  solution in that case.

  `bicg` and `qmr` use the product of the transposed matrix.
  `csr_matvec_t` gives that product.
- `lsqr` takes a matrix of any shape and solves the least squares
  problem. It gives the solution of the least norm when the unknowns
  outnumber the equations. It has no `damp`, no `x0` and no `atol`.
  `minres` takes a symmetric matrix and does not check the symmetry.
  It has no `shift`, no `x0` and no preconditioner. The path may differ
  from SciPy, so only the solution matches.
- `lsmr` takes a matrix of any shape and solves the least squares
  problem, with `damp`, `atol`, `btol`, `conlim` and `maxiter`. It gives
  the type `Lsmr`. The type holds `x`, `istop`, `itn`, `normr`, `normar`,
  `norma`, `conda` and `normx`, as SciPy gives them in a tuple.

  It starts from zero, so it has no `x0` and no `show`. The default
  limit is the lesser side of the matrix. A tolerance, the damping or a
  right side that is not finite is a fault. A right side whose norm
  falls to zero, or grows past the largest number, is a fault too.
  SciPy returns the zero solution in that case.
- A matrix so small that its norm falls to zero gives the zero
  solution, as SciPy gives. Scale the matrix first.
- `eigsh` takes the choices `LA`, `SA`, `LM` and `SM` and gives the
  values in rising order. It uses the Lanczos method with a full
  reorthogonalization, and a space above 256 vectors is a fault. It has
  no `M`, no `sigma`, no `v0` and no `ncv`. The vectors may differ in
  sign from SciPy. Give it a symmetric matrix.
- `csr_hstack` and `csr_vstack` join two matrices, not a list. Call them
  again for more. `csr_kron` and `csr_block_diag` take two matrices.
- `csr_kronsum` follows `scipy.sparse.kronsum`: `kron(I_m, a) + kron(b, I_n)`.
  Both operands need to be square. Any other shape gives a fault of kind
  `F_SHAPE`. It takes two matrices of the type `Csr` and gives one.
- `csr_slice` takes the row and column ranges with the end excluded and
  no step. `csr_tril` and `csr_triu` take the diagonal `k`.
  `csr_multiply` takes two matrices of one shape and does not broadcast.
- The sparse types are `Csr`, `Csc`, `Coo`, `Dia` and `Bsr`. They hold
  the compressed rows, the compressed columns, the coordinates, the
  diagonals and the blocks. The types `lil_array` and `dok_array` stay
  out.
- Build a matrix from a list of entries with `csr_from_triplets`, with
  `csc_from_triplets` or with `coo_from_triplets`. The first two add the
  entries of one place. The third keeps them. A sparse result keeps the
  stored zeros that a sum or a product makes.
- The type `Dia` holds the offsets of the diagonals and the values of the
  diagonals. The offsets rise and no two of them are the same. The values
  of one diagonal lie in the order of the row. The first place of a
  diagonal with the offset `k` is the row `max(0, -k)`.
- A diagonal of `Dia` that holds no value is not kept. A zero that lies on
  a kept diagonal stays in the diagonal, so `dia_nnz` counts the zeros.
  SciPy counts the same zeros, unless its band stops before the last
  column. The count of `dia_nnz` is then larger than the count of SciPy.
- The conversion to the compressed rows and to the coordinates drops the
  zeros of the diagonals, as SciPy drops them.
- The transpose of `Dia` negates the offsets and sorts them again. The
  transpose of SciPy lists the offsets in the other order. The values of
  one diagonal keep their order in both.
- `dia_from_diagonals` follows `scipy.sparse.diags`. The values of one
  diagonal cover its places inside the matrix. The older
  `scipy.sparse.spdiags` places the values by column, so it drops the
  first values of an upper diagonal.
- The type `Bsr` cuts the matrix into blocks of `br` rows and `bc`
  columns. The block size divides the shape. A block that holds no entry
  is not kept. A zero inside a kept block stays in the block, so
  `bsr_nnz` counts the zeros.
- The blocks of one block row keep the order of their first entry. SciPy
  keeps that order too. The block columns of a block row can therefore
  fall. `bsr_get` and `bsr_to_csr` walk the blocks of the row, so they
  need no sorted row.
- The transpose of `Bsr` swaps the block size. It sorts the blocks by the
  block row and the block column of the result, as SciPy sorts them.
- The conversion to the compressed rows keeps the zeros of the blocks, as
  SciPy keeps them. The count of the stored values of the compressed rows
  counts those zeros too.
- The coordinates of a block matrix hold every value of every block.
  The zeros stay, as the coordinates of SciPy keep them. The count of
  the entries is the count of the stored values.
- The blocks of a matrix that the coordinates build keep the order of the
  first entry of the coordinates. That order can differ from the order
  of the source. SciPy differs in the same way.
- `bsr_diagonal` gives the values on the main diagonal. The sums along an
  axis come from the compressed rows.
- The type `Dok` holds the keys in the order of their first insertion.
  The dictionary of keys of SciPy keeps the same order. An overwrite keeps the place of the
  key. A value of zero removes the key, and a later set puts the key at
  the end.
- `dok_setdiag` follows the rule of SciPy. It visits the diagonal in the
  order. A short list leaves the later places alone. A long list is cut.
  A value of zero removes its place. An offset outside the matrix is a
  fault of kind `F_RANGE`.
- `dok_resize` drops the keys that fall outside the new shape. The other
  keys keep their order.
- The transpose of `Dok` swaps the rows and the columns. The order of the
  keys stays.
- The compressed rows sort the columns of a row, as the compressed rows of
  SciPy do. The coordinates keep the order of the keys.
- The conversions to the compressed rows and to the coordinates add up the
  duplicates of one place. A zero of the source stays, as the zero of
  SciPy stays.
- SciPy counts a negative place from the end. The package faults with a
  fault of kind `F_RANGE`.
- SciPy stores a value of NaN or infinity. The package faults with a
  fault of kind `F_VALUE`.
- The lookup of a place uses a hash that holds a random number of the
  matrix. A read and a write cost a constant time on average. A removal
  rebuilds the table, so it costs one pass over the keys.
- The fields of `Dok` are public. A matrix that a caller builds by hand
  must keep the layout of the buffer. A table without an empty slot gives
  a fault of kind `F_VALUE` on a set. Other forged fields stop with an
  index error. Build a matrix with `dok_new` to avoid the risk.
- The table grows by a rebuild when it holds too many keys. A read and a
  write of a place that exists cost the same at every count of keys.
- The type `Coo` keeps the duplicate entries of one place, as SciPy does,
  and `nnz` counts them. The dense form, `coo_get` and the compressed
  rows add them up. `coo_get` and `coo_matvec` read every entry, so they
  cost one pass over the entries. Sums and products convert to the
  compressed rows first. `coo_transpose` keeps the count of the entries.
- The type `Csc` holds the same values as the type `Csr` in the other
  order. Every call that has no own rule converts to the compressed rows
  and back. The calls `csc_from_csr` and `csc_to_csr` convert a matrix
  without a copy of the values.
- `csr_diags` takes one diagonal for each call. Add the results to get
  several diagonals. `kron`, `hstack`, `vstack`, slicing and `norm`
  stay out.
- `spsolve` turns the matrix dense. The side stays at 4096 or below.
  The cost equals the cost of a dense solve. `cg` takes the
  start zero and no preconditioner. `gmres`, `bicgstab`, `splu`,
  `eigsh` and `svds` stay out.
- `solve` with a vector on the right side gave a wrong answer when
  the elimination swapped rows. The forward substitution read the
  rows in the wrong order. The answer now equals LAPACK. `inv` took
  the same path and holds the fix.
- `firwin` takes a bool for `pass_zero`, not the band names, and the
  windows of `get_window`. `freqz` gives the real and the imaginary
  parts in two arrays and takes a count of points, not an array.
  `sosfreqz` does the same for sections. `sosfreqz_at` takes the
  frequencies as an array.
- `resample_poly` takes the signal, `up` and `down`. The filter is the
  Kaiser window of SciPy with beta 5, and the edges hold zeros. The
  arguments `window`, `padtype` and `axis` stay out.
- `spectrogram` gives the frequencies, the times of the centres and
  the matrix of the power. The matrix holds one row per frequency. The
  window is a name of `get_window` and `tukey` is the default. The
  scaling is `density` or `spectrum`. The arguments `mode`,
  `return_onesided` and `axis` stay out.
- `stft` gives the frequencies, the times and the real and imaginary
  matrices. The arguments `boundary` and `padded` take a bool, so the
  zero extension is the only boundary. `detrend` stays out.
- `istft` takes the matrices of `stft` and the same window, size and
  overlap. A window that fails the overlap condition is a fault, as it
  is in SciPy. The arguments `nfft`, `input_onesided` and the axes stay
  out.
- `welch_psd` gives the mean power of the parts, as SciPy `welch` does.
  The older `welch` of the package stays as it is.
- `welch_ex` takes all the options of SciPy `welch`. The window is a
  name with its parameter, or a list of values. It takes `nfft`, the
  trend `constant`, `linear` or `none`, and the scaling. It also takes
  the one-sided or two-sided spectrum and the average `mean` or
  `median`. A signal shorter than a part is a fault, where SciPy
  shortens the part and warns. It gives the frequencies and the power
  of one signal.

- `firwin2` takes the window as a name, and `none` gives no window.
- `remez` follows the exchange routine of SciPy step by step, so the
  taps agree to 1e-9. The call takes `bands`, `desired`, an empty or
  complete `weight`, the kind `bandpass`, `differentiator` or `hilbert`,
  `maxiter`, `grid_density` and `fs`. The default `fs` is 1, as in
  SciPy. The name of the argument `type` is `kind`, because `type` is a
  keyword. When the deviation stops rising, the call gives `F_VALUE`,
  as SciPy raises an error. When `maxiter` ends first, the call gives
  the taps of the last pass without a fault, as SciPy does.
- `remez` refuses more input than SciPy does. It faults on a value that
  is not finite and on a weight at or below zero. It faults on an `fs`
  at or below zero and on more than 4096 taps.

- `remez` also faults on a grid of more than 262144 points. It faults
  on a `maxiter` above 1000 and on a `grid_density` above 1024. SciPy
  takes some of these and may fail on memory or time.
  It follows SciPy for the types one to four and for a jump in `freq`.
- `decimate` takes the kinds `iir` and `fir` with a zero phase shift.
  The argument `ftype` takes a name, not a filter object. The
  arguments `axis` and `zero_phase` stay out. The kind `iir` needs a
  signal longer than the padding of `sosfiltfilt`.
- `find_peaks` takes the lower limits `height` and `prominence` and
  the `distance`. The limits `threshold`, `width`, `plateau_size`
  and the pairs of limits stay out. A signal with no peak is a fault.
  The distance test may drop other peaks of equal height than SciPy
  drops, because the sort here is stable.
- `periodogram` takes the detrend of the mean or none, and the
  scaling `density` or `spectrum`. `nfft`, `return_onesided` and the
  detrend `linear` stay out. `resample` takes no window and no `t`.
- `savgol_filter` and `medfilt` take one dimension and no `axis`.
  `savgol_filter` takes no `cval`. The edge modes are `interp`,
  `nearest`, `mirror`, `constant` and `wrap`.
- `milp` takes the arguments of `linprog` and a list of flags for the
  integer variables. It uses branch and bound over the relaxations of
  `linprog`. The search stops with a fault after 20000 nodes. SciPy uses
  HiGHS, which also cuts and presolves. A problem with ties may return
  another optimal point with the same value.
- `linprog` takes flat lists by row for the matrices and runs the two
  phase simplex with the rule of Bland. The cost is dense, and a side
  holds up to 2048 values. A problem with no point or with no least
  value is a fault. The simplex gives a vertex. When the optimum is
  not unique, SciPy may give another vertex. The arguments `method`,
  `callback`, `options` and the integer variables stay out.
- `nnls` solves the passive system by the normal equations, so a
  matrix with a large condition number loses digits.
- `linear_sum_assignment` takes a finite cost. When several pairings
  hold the same total, the pairing may differ from SciPy.
- `approx_fprime` takes the forward difference. `bisect` and `ridder`
  stop by the absolute tolerance only.
- `quad` uses the rule of Gauss and Kronrod of 21 points and bisects
  the interval of the largest error. The ends may be infinite. The
  estimate of the error differs from QUADPACK, because the package
  holds no extrapolation. A hard integrand ends with the fault
  `F_LIMIT` when the limit of intervals is reached. The weights of
  `quad` stay out.
- `odeint` runs the RK45 method of `solve_ivp` between the times and
  gives one row per time. SciPy runs LSODA, so values agree only to the
  tolerance. It holds no Jacobian, no stiff method, no `args` and no
  `full_output`. The function takes `y` first, then `t`.
- `dblquad` takes the function as a block `{ y, x => ... }`, as SciPy
  orders the arguments. The limits of y are two named functions. They
  come last in the call, after the tolerances. A limit of x or of y
  may be infinite, as in SciPy. A missing value of a limit is a
  fault. The error is the sum of the errors of the inner integrals.

- `tplquad` takes the block `{ z, y, x => f }`. The four limit
  functions come last, in the order `g`, `h`, `q`, `r`. The functions
  `g` and `h` take x. The functions `q` and `r` take x and y.
  A limit that is not finite is a fault.

- `nquad` takes the block `{ p => f }`, which takes the point as a
  slice. The first value of the point belongs to the first pair of
  the ranges. The ranges hold the pairs of the ends, and an end may
  be infinite. The call gives the value and the error estimate of the
  outermost integral.

- The package holds no more than eight ranges for `nquad`. It holds
  no more than twenty million evaluations of the function. A fault of
  an inner integral stops the call.

- `quad_vec` takes the block `{ x, out => out.push(a); }`. The block
  fills a row with the values of the vector. Every point must give the
  same count of values. The norm is `2` or `max`. The call gives the
  vector and one error estimate.

- The estimate of the error of `nquad` and `quad_vec` differs from
  SciPy, because the package holds no extrapolation.

- `fixed_quad` takes up to 512 points.
- `solve_ivp` takes the methods `RK45`, `RK23`, `DOP853`, `BDF` and
  `Radau`. The steps
  equal the steps of SciPy. The result holds one row per time, where
  SciPy holds one column. The argument `t_eval` takes sorted times
  inside the span. It gives the values from the dense output.
- `DOP853` is the method of Dormand and Prince of the order eight. It
  holds the error estimate and the interpolation of SciPy. Its dense
  output and its events call the function three more times per step.
- `BDF` is the variable order method of the backward differentiation
  formulas, with the modified formulas of SciPy. The order runs from
  one to five. The Jacobian comes from finite differences, with the
  step control of SciPy. The count of steps equals the count of SciPy.
  The argument `jac_sparsity` stays out.
- `Radau` is the implicit method of the order five. It solves the
  three stages through one real and one complex system. The Jacobian
  comes from finite differences. The count of steps equals the count
  of SciPy. The argument `jac_sparsity` stays out.
- `solve_ivp_jac` takes the Jacobian of the caller for `BDF` and
  `Radau`. The block gets the time, the state and a mode. Mode zero
  gives the derivative. Mode one gives the matrix in row-major order.
  A growable vector holds both results. The step counts equal those of
  SciPy with the same Jacobian.
- A matrix of the wrong size or with a value that is not finite is a
  fault in `solve_ivp_jac`.
- A wrong value in the matrix of `solve_ivp_jac` is not detected. It
  shrinks the steps until the run faults. The methods `RK45`, `RK23`
  and `DOP853` ignore the Jacobian, as in SciPy. A Jacobian does not
  combine with events or dense output.
- A singular matrix in `BDF` or `Radau` shrinks the step. A step below the
  smallest step is a fault.
- The argument `first_step` sets the first step. A value of zero lets
  the solver choose, where SciPy treats zero as a fault. A negative
  value or a value beyond the span is a fault. The method `LSODA`
  stays out. A stiff system needs many steps. It may reach the
  limit of one million steps.
- `solve_ivp_events` takes the events of SciPy. The block gives the
  derivative and then the event functions. The arrays `terminal` and
  `direction` hold one value per event. It gives the event times, the
  event numbers and the event states. It finds each root by bisection
  of the polynomial of the step.
- `solve_ivp_dense` keeps the polynomial of each step. `dense_eval`
  reads it at any times. This replaces `dense_output=True`, because a
  function cannot leave its block. The two features do not combine.
- `cubic_spline` takes the conditions `not-a-knot`, `natural`,
  `clamped` and `periodic`. `clamped` takes the end slopes `left`
  and `right`, zero by default. A periodic spline needs equal first
  and last values, and it repeats outside the knots.
  `cubic_spline_ends` takes an order, 1 or 2, and a value for each end.
  It gives a slope or a second derivative at that end. The two ends
  may differ in order.
- `spline_eval` takes the
  derivative orders zero to three. A point outside the knots takes the
  end cubic for `cubic_spline` and `pchip`. A spline of `akima` gives
  NaN there, as SciPy does.
- `make_interp_spline` takes points, values and a degree from zero to
  twenty, three by default. The knots follow the not-a-knot rule of
  SciPy. `bspline` builds a spline from knots, coefficients and a
  degree. The spline is the type `BSpl`.
- `make_interp_spline_bc` is the `bc_type` argument of SciPy. It takes
  the orders and values of the left end, then of the right end. The two
  counts add up to the degree minus one. The knots repeat the end
  points. The clamped form is order 1 with value 0 at each end of a
  cubic. The natural form is order 2 with value 0.
- `bspline_design_matrix` gives the matrix of the basis functions at
  the points. It is the `design_matrix` of SciPy. The matrix is
  sparse, in the type `Csr`. The extrapolation is `no`, `yes` or
  `periodic`. A point outside the knots gives a fault when the
  extrapolation holds `no`.
- `bspline_eval` and `splev` give values and derivatives. A point
  outside the knots takes the end polynomial, as SciPy does. A
  derivative above the degree is zero.
- `splrep` gives the interpolation spline when `s` is zero. A factor
  above zero gives the smoothing spline of Dierckx, with the same
  knots and coefficients as SciPy. The weights hold one. The
  arguments `w`, `xb`, `xe`, `task`, `t`, `per`, `quiet` and the
  periodic form stay out. SciPy only warns when the search of the
  knots stops without a root. The package returns a fault there.
- `splrep_residual` gives the weighted sum of squared residuals.
- `cubic_hermite` takes the knots, the values and the slopes. It
  gives the same cubics as SciPy `CubicHermiteSpline`. A count of
  slopes that differs from the knots and a slope that is not finite
  are faults. A point outside the knots takes the end cubic.
- `interp1d` takes the kinds `linear`, `slinear`, `nearest`, `previous`,
  `zero`, `next`, `quadratic` and `cubic`. The kind `quadratic` takes
  three to 512 points and solves a dense system. A point
  outside the knots is a fault unless `extrapolate` holds true. The
  argument `fill` with `fill_low` and `fill_high` gives the values of
  SciPy `fill_value` for such a point. The older `spline` of the
  package stays a natural spline with its own type.
- The spatial functions follow `scipy.spatial`. `cosine` and `hamming`
  already name other functions here. The distances carry the names
  `cosine_distance` and `hamming_distance`.
- `kdtree_query` takes one query row and scans every point. The
  build costs nothing and a query costs one pass.
  `kdtree_query_ball_point` gives a fault when no point lies in range.
- `convex_hull` takes points of a plane. It gives the corner places,
  the area and the perimeter.
- `griddata` takes the methods `nearest` and `linear`. The method
  `nearest` works in any count of dimensions. The method `linear` works
  in one dimension and in a plane. A point outside the hull takes
  `fill_value`, and the method `nearest` never takes it. The values
  hold one number for each point, or a row of numbers for each point.
  A table of values gives a table of results.
- The method `cubic` is a fault in `griddata`. So is `linear` in three
  dimensions or more.
- `nd_interpolator` builds the triangulation once. `nd_interpolate`
  then answers many sets of points. They match `LinearNDInterpolator`
  and `NearestNDInterpolator` of SciPy. `griddata` calls them. An
  interpolator of 2000 points answered 1000 points in about 0.14
  seconds.
- The filters of N dimensions follow `scipy.ndimage`: `correlate1d`,
  `convolve1d`, `uniform_filter1d`, `gaussian_filter1d`, `uniform_filter`
  and `gaussian_filter`. SciPy names the filter of an array of weights
  `correlate` and `convolve`. Those names hold the signals of one
  dimension here, so the filters of N dimensions are `correlate_nd` and
  `convolve_nd`. The values agree with SciPy 1.18 to `1.0e-12`. This holds
  for every mode, for even and odd weights and for every valid origin. It
  holds for the derivative orders 0 to 3.
- The modes are `reflect`, `constant`, `nearest`, `mirror` and `wrap`.
  The names `grid-mirror`, `grid-constant` and `grid-wrap` are the same
  as `reflect`, `constant` and `wrap`, as in SciPy. The mode is one name
  for all axes. SciPy also takes a list of modes. `cval` fills the
  values outside the array in the mode `constant`.
- The filters take no `output`, no `axes` and no `radius` argument, and
  no complex array. The filters take `f64` arrays only. The median, rank,
  minimum, maximum, Sobel, Prewitt and Laplace filters are not in the
  package.
- A size or a sigma of one value holds for every axis. `uniform_filter`
  needs a size. A `gaussian_filter` axis with a sigma of 1.0e-15 or less
  stays as it is, as in SciPy. `gaussian_filter1d` gives a fault for such
  a sigma, where SciPy raises an error for a sigma of zero or below.
- The cost of a call is the count of values times the count of weights.
  A cost above 1.0e9 is a fault of kind `F_LIMIT`. A Gaussian radius above
  1000000 and a derivative order above 64 are the same fault. SciPy has
  no such limit. A matrix of 512 by 512 with a Gaussian of sigma 2.0 filters
  in under one second, build time included.
- `uniform_filter1d` moves a running sum along the line, as SciPy does. A
  line with a NaN or an infinity takes the direct sum. So a NaN spoils
  only the windows that hold it. SciPy spoils every value after the
  first NaN.
- The rank filters take a box size, or with the suffix `_fp` a footprint
  array. A nonzero value marks a place of the window. The rank, the
  percentile, the origin and the five modes follow SciPy 1.18 and agree to
  `1.0e-12`. A NaN counts as the largest value, so the result is defined.
  A median of 5 by 5 on a matrix of 256 by 256 takes under one second.
  That time includes the build.
- SciPy filters data of one dimension through a fast path that ignores the
  holes of a footprint. The result there differs from the definition of
  the window. This package follows the definition for every count of
  dimensions. So a footprint of one dimension with holes can differ from
  SciPy.
- The cost of a rank filter is the count of values times the count of places
  of the window. A cost above 2.0e8 is a fault of kind `F_LIMIT`. A box
  that sorts a window takes time in proportion to its size times its log.
  The calls `minimum_filter` and `maximum_filter` with a size filter one
  axis at a time. They scan each window, where SciPy uses a method that
  needs less time for a large window. Their limit is 1.0e9.
- `sobel`, `prewitt`, `laplace`, `gaussian_laplace` and
  `gaussian_gradient_magnitude` equal SciPy. They take one mode for all
  axes.
- The calls `generic_filter`, `generic_filter_fp`, `generic_laplace` and
  `generic_gradient_magnitude` take a trailing block where SciPy takes a
  Python function. Write `generic_filter(a, [3, 3]) { w => spread(w) }`.
  The window holds the values of the nonzero places in row-major order.
  The block of the other two calls takes the array, the axis, the mode and
  the fill value. It gives the derivative along that axis. A block captures
  the locals of the caller, so the extra arguments of SciPy have no use.
- The spline filters and transforms equal SciPy 1.18. They take the orders
  0 to 5. They take the modes `mirror`, `reflect` (also `grid-mirror`),
  `wrap`, `grid-wrap`, `nearest`, `constant` and `grid-constant`. The
  pinned cases agree to `1.0e-10`. The spline filter is recursive, so its rounding
  differs from SciPy in the last digits.
- For a spline of order 2 or more, the mode `wrap` of SciPy folds like
  `mirror`. For the orders 0 and 1 it uses the period `n - 1`. The mode
  `grid-wrap` uses the period `n`. The package follows SciPy in both.
  The modes `nearest` and `grid-constant` pad the array with 12 values
  before the filter, as SciPy does. So a spline there is not exact beyond
  that range.
- `spline_filter` takes the orders 2 to 5. `spline_filter1d` also takes
  0 and 1 and returns the array. `zoom` with a factor of one on every axis
  and `prefilter` true returns a copy. `rotate` takes the plane as two axes.
  It rotates every plane of an array of three or more dimensions.
- The calls take real arrays and give `f64`. They give no `output` argument
  and no complex arrays. `map_coordinates` and the transforms take
  `prefilter` and `cval`.
- The cost of a transform is the count of output values. It is
  multiplied by `(order + 1)` once for each axis. A cost above 1.0e9 is
  a fault of kind `F_LIMIT`. A 512 by 512 `rotate` and a `zoom` of order 3
  took 1.4 seconds together.
- Goose cannot pass a block on to another generic call. So each of the
  calls holds its own loop.
- The call `geometric_transform` takes a trailing block where SciPy takes a
  Python function. The block reads the coordinate of an output value and
  writes the coordinate in the input. Write
  `geometric_transform(a) { c, o => o[0] = c[0] * 2.0 }`. The buffer of the
  input coordinate holds one value for each axis and starts at zero. The
  call reads the spline at the mapped coordinates, as SciPy does.
- An empty output shape of `geometric_transform` means the shape of the
  input. The pinned cases agree with SciPy to `1.0e-10`.
- The call `vectorized_filter` takes a trailing block where SciPy takes a
  Python function. The block gets the windowed array of the input. Its
  shape is the shape of the input followed by the shape of the window. It
  fills the buffer of the result. Write
  `vectorized_filter(a, [3, 3]) { x, o => o[0] = x.data[0] }`.

- The block reduces the window axes itself, as the function of SciPy does.
  The pinned cases agree with SciPy to `1.0e-10`.
- The count of the values of the windowed array stays below the limit of
  the elements. The check divides the limit by the count of the window. A
  window below one is a fault of kind `F_VALUE`. An origin off the window
  is a fault of kind `F_RANGE`. An unknown mode is a fault of kind
  `F_VALUE`.
- The call `generic_filter1d` takes a trailing block where SciPy takes a
  Python function. The block reads one line of the array and fills the
  buffer of the result. The line holds `n + size - 1` values, where `n` is
  the length of the axis. The line starts `size / 2 + origin` values before
  the first value of the array. The border of the line follows the mode, as
  SciPy does. The pinned cases agree with SciPy to `1.0e-10`.
- The count of the coordinates of `geometric_transform` is the count of the
  output values times the count of the axes. It stays below the limit of
  the elements. The count of the lines of `generic_filter1d` times the
  length of a line stays below the cost limit of 1.0e9. A call above a
  limit is a fault of kind `F_LIMIT`.

- A size below one is a fault of kind `F_VALUE`. A bad axis, or an origin
  off the window, is a fault of kind `F_RANGE`. An unknown mode is a fault
  of kind `F_VALUE`.
- `geometric_transform` faults on an output shape with a dimension below
  one. SciPy gives an array of no values there. It faults on a coordinate
  that is not a number, or a coordinate beyond 4.0e15. SciPy gives NaN or
  an infinity there.
- The time of a block is the time of the caller. The package checks only
  the count of values times the count of places, against the limit 2.0e8.
  A block that gives NaN or an infinity passes that value on. A block of
  the derivative calls that gives an array of another size is a fault.
- The morphology calls take a connectivity, or with the suffix `_st` a
  structure array and a mask. An empty array stands for no mask. A value
  counts as true when it is not zero. The mask keeps the old value where it
  is zero. The border value is 0 or 1. The results follow SciPy 1.18
  exactly, origins included.
- A count of iterations below one repeats until no value changes. The cost
  of one round is the count of values times the count of places of the
  structure. A cost above the budget is a fault of kind `F_LIMIT`.
- `label` numbers the parts from one in the order of their first value in
  row-major order, as SciPy does. A structure for `label_st` must hold the
  size 3 on every axis and be symmetric. SciPy puts the same limits on it.
- `distance_transform_edt` runs the lower envelope of parabolas along each
  axis, as SciPy does. It faults when the array holds no background value.
  SciPy returns values without meaning there. The sampling takes none, one
  value, or one spacing for each axis. A spacing that is not finite or not
  above zero is a fault.
- `generate_binary_structure` clamps the connectivity to the range from one
  to the rank, as SciPy does. `iterate_structure` grows a structure by a
  dilation with itself. A count of iterations above the budget is a fault,
  so the time stays inside the budget.
- The measurements of labeled regions take NaN as the largest value.
  They take the first place for a tie. SciPy gives a result there that
  depends on the path of the call. The rule of this package does not
  depend on the order of the values.
- `histogram_labels` gives one row of bins for each entry of the index
  list. Without an index list it gives one row of bins. This is the
  shape of SciPy.
- `find_objects` gives one row of numbers for each label. SciPy gives
  one list of slices for each label. A label without a value gives a row
  of -1 here, and None in SciPy.
- A label without a value gives zero for the sums, the extremes and the
  median. It gives NaN for the mean, the variance and the center.
- The names of the SciPy calls `mean`, `variance`, `minimum`, `maximum`,
  `median` and `histogram` carry the suffix `_labels` here. The suffix
  keeps them apart from the statistics names of the package.
- The grey calls equal SciPy 1.18. The plain form takes a size and a flat
  box. The form with the suffix `_st` takes a footprint and a structure. An
  empty array stands for None. A structure without a footprint uses all its
  places. A dilation reflects the footprint and the structure and flips the
  origin, as SciPy does.
- An erosion takes the minimum of value minus structure. A dilation takes
  the maximum of value plus structure.
- NaN is the largest value in the grey calls, as in the rank filters. The
  result does not depend on the order of the window. The cost limit of the
  rank filters holds for them.
- `binary_propagation` is a dilation that repeats until no value changes. In
  `binary_hit_or_miss` an empty `origin2` stands for `origin1`, and an empty
  `structure2` stands for the complement of `structure1`. To get an origin of
  zeros for the second structure, pass a list of zeros.
- A structure with no true place gives an erosion of all ones. It gives a
  dilation of all zeros, as in SciPy. The binary calls gave a fault there before. The cost
  check then counts one place.
- `distance_transform_cdt` equals SciPy 1.18, distances and positions
  included. The metrics are `chessboard`, `taxicab`, `cityblock` and
  `manhattan`. The `_st` forms take a structure of side 3 on every axis.
  A value that the structure never reaches keeps the distance -1, as in
  SciPy. A chamfer transform of 512 by 512 took 0.1 seconds.
- `distance_transform_bf` equals SciPy 1.18. The metrics are `euclidean`,
  `taxicab`, `cityblock`, `manhattan` and `chessboard`. A tie goes to the
  later position in row-major order, as in SciPy. The sampling counts for
  the Euclidean metric only, as in SciPy. The count of pairs of foreground
  and edge values has the limit 2.0e8, which takes about 15 seconds.
- The indices calls give the positions in one array. Its shape is the count
  of axes followed by the shape. A background value points to itself.
- The distances calls fault when the array holds no background value. SciPy
  gives a distance of -1, of 1.34e154 or of 4294967295 there. The package
  faults, as `distance_transform_edt` does.
- `value_indices` gives three arrays. The first holds the distinct values in
  ascending order. The second holds the count of positions of each value.
  The third holds one row of coordinates for each position.
- SciPy gives a dictionary of index tuples for `value_indices`. A value that
  is not a whole number is a fault. With `has_ignore` true the call skips
  `ignore_value`.
- A measurement of an array of 512 by 512 took 0.04 seconds. The
  median sorts the values of each part, so it takes the longest.
- A tie between two nearest points in `griddata` goes to the lower
  coordinate in one dimension. In more dimensions it goes to the lower
  place. SciPy leaves such a tie to the order of its tree.
- `delaunay` triangulates points of a plane by the method of Bowyer
  and Watson. A set in general position gives the triangles of SciPy.
  Points on one circle have no unique triangulation, and Qhull and this
  package may pick different diagonals. A repeated point stays out of
  the triangles. The time grows with the square of the count of
  points. A set of more than 5000 points is a fault, so that hostile
  input cannot stall the call.
- The triangulation of three dimensions or more, the hull of three
  dimensions and the arguments `eps` and `workers` stay out. The metrics of true and false counts read
  a nonzero value as true.
- `dct`, `idct`, `dst` and `idst` take one row and the types one to
  four. They take the norms `backward`, `ortho` and `forward`. Type one
  of the cosine needs two points.
- `dctn`, `idctn`, `dstn` and `idstn` take an array and a list of axes. They run each axis in turn. An
  empty list gives the array back. No list means every axis.
- `dct_axis`, `idct_axis`, `dst_axis` and `idst_axis` run one row
  function along one axis of an array. The length `n` crops or pads
  the axis. A length of zero keeps the size of the axis.
- `hfft` and `ihfft` take and give the parts as two rows. `hfft_axis`
  and `ihfft_axis` run them along one axis of an array. `hfft_axis`
  crops or pads the axis to `n / 2 + 1` places. `ihfft_axis` gives
  `n / 2 + 1` places and takes the norms `backward`, `ortho` and
  `forward`. `rfft2`,
  `rfftn`, `irfft2` and `irfftn` take the parts as two arrays. `irfftn`
  takes the length of the last axis, because it cannot tell an even
  length from an odd one.
- `hfft2` and `hfftn` take the parts as two arrays and the length of
  the last axis. `ihfft2` and `ihfftn` take a real array and give two
  arrays. They hold no `s` list for the other axes, no `axes` and no
  `norm`.
- `hfftn_ex` and `ihfftn_ex` add the `s`, `axes` and `norm`
  arguments of SciPy. A size of `-1` in `s` keeps the size of the
  input. The list `[1000]` for `axes` means every axis, or the last
  `len(s)` axes. The half spectrum sits on the last axis of `axes`.
  The count of `s` must equal the count of `axes`. The default norm is
  `backward`.
- SciPy takes a `workers` count on its transforms. NumPy, Rust and Go
  hold no such count. The package runs every transform on one worker.
  A count of workers holds no effect on the result.
- `next_fast_len` follows SciPy. A complex transform takes the factors
  up to 11. A real one takes 2, 3 and 5.
- `fht` and `ifht` take a one-row signal, `dln`, `mu`, `offset` and
  `bias`. `fhtoffset` takes `dln`, `mu`, `initial` and `bias`. They hold
  no `axis` argument. A transform that is singular for the bias gives a
  fault, where SciPy warns and goes on. `dln` must not be zero.
- `lu_factor` and `lu` follow LAPACK. The row i swaps with the row
  `piv[i]`. A singular matrix gives no fault there. `lu_solve` gives a
  fault when it meets a zero on the diagonal. `solve` takes a matrix on the right side.
- `solve_toeplitz` builds the matrix and runs the dense solve of the
  package, where SciPy runs the recursion of Levinson. The values
  agree. The cost grows with the cube of n, where the recursion grows
  with the square. A singular matrix is a fault. `solve_circulant`
  runs through the transform, as SciPy does, and a frequency at or
  below the cutoff is a fault. The name `raise` of the argument
  `singular` is the only behavior.
- The angles of `subspace_angles` may differ from SciPy in the last
  digits. The bases of the two spaces come from different singular
  value decompositions.
- The special matrices follow `scipy.linalg` and give floating point
  values. `invpascal` stops at the order 30, where the integer entries
  still fit in 64 bits. `invhilbert` stops at 150 and `pascal` of the
  symmetric kind at 1000. Every other special matrix stops at 4096.
  A larger size is a fault, so that an input cannot ask for a result of unbounded size.
- `invhilbert` takes no `exact` flag. `pascal` and `invpascal` take no
  `exact` flag either. Each gives floating point values.
- `ishermitian` reads a real matrix. The complex form stays out.
- `eigvalsh_tridiagonal` and `eigh_tridiagonal` hold real values only.
  The values come from the bisection. The vectors come from the inverse
  iteration. SciPy runs the routines of LAPACK. The values agree to the
  last digits that the bisection reaches. The vectors hold no fixed
  sign, because the form is not unique.
- A select of `i` takes the ranks from zero. A select of `v` takes the
  values from the lower bound up to the upper bound. A select of `a`
  takes all. The diagonal holds at most 4096 values. The cost of the
  bisection grows with the square of the size.
- `cholesky_banded` expands the band into the dense matrix. It runs the
  dense Cholesky of the package. SciPy runs the banded routine of
  LAPACK. The values agree. The cost grows with the square of n, where
  the banded routine grows with the width of the band.
- `rq` follows LAPACK. The mode `r` holds back the orthogonal matrix.
- `qr` follows the signs of LAPACK. The diagonal of R is negative for a turn, and the last column of a square matrix keeps its sign. This changes the signs of the factors against earlier releases.
- `dft` gives the complex matrix with the scales none, `sqrtn` and `n`. A size above 4096 gives a limit fault.
- `matrix_balance` follows LAPACK dgebal. It gives the balanced matrix and the matrix of the change. The second form gives the scales and the permutation apart.
- `expm_frechet` uses the block matrix of the matrix and the direction. A side above 128 gives a limit fault.
- `expm_cond` builds the form of the derivative, which holds the square of the side of the matrix. A side above 8 gives a limit fault.
- `qr_update` follows scipy.linalg.qr_update. It takes the factors of a matrix, the left vector and the right vector. The right vector holds one column or many columns. The reduced form needs a square triangle.

  A side above 4096 gives a limit fault. The rank of the change must not be above the size of the factors. The full form needs the rows below the triangle to hold the rank.
- `qr_insert` follows scipy.linalg.qr_insert. It takes the factors, the new rows or columns, the place and the word `row` or `col`. A negative place counts from the end. The new rows or the new columns hold one row or one column, or many. The reduced form needs a square triangle.

  A matrix that loses rank can give other signs than SciPy in the orthogonal factor. SciPy states that the result can vary.
- `qr_delete` follows scipy.linalg.qr_delete. It takes the factors of a matrix, the place, the count and the word `row` or `col`. A negative place counts from the end. The count holds the number of the removed rows or the removed columns.

  The count must hold at least one. The count must not hold more rows or more columns than the factors hold. A matrix of one column and the reduced form give the rows that stay as one vector of unit length.
- `rsf2csf` follows scipy.linalg.rsf2csf. It takes the real Schur form and the change of the basis, and gives both in complex form. The method turns each 2 by 2 block with the root of larger imaginary part. LAPACK orders a conjugate pair the same way.

  Both operands take a stack of matrices. The last two dimensions give the matrix. The dimensions before them give the batch.

  The batch of the two operands broadcasts. The result takes the batch of the broadcast.
- `cdf2rdf` follows scipy.linalg.cdf2rdf. It takes the complex eigenvalues as a vector and the eigenvectors as a matrix. It gives the real block form and the real eigenvectors. An unpaired complex value is a fault.

  The operands take a stack. The eigenvector array holds one dimension more than the eigenvalue array.

  The complex values of a slice pair in the order they appear. The first pairs with the second. The third pairs with the fourth. A real value between two complex values does not break a pair. SciPy states that the caller must not interleave the pairs. The package does not check it.

  The version before `v0.139.0` paired a complex value with the value beside it. That version gave a wrong block for a real value between two complex values.

  The block form takes the batch of the eigenvalues. The eigenvectors take the broadcast of both batches. SciPy gives the same shapes.
- `solve_continuous_are(a, b, q, r)` and `solve_discrete_are(a, b, q, r)` follow scipy.linalg. A call with six operands solves the generalized equation of SciPy with `e` and `s`. Pass an empty array for an operand you leave out. The result is the stabilizing solution, and it is symmetric.
- The package solves the continuous equation with the matrix sign function after an exact removal of `e` and `s`. It solves the discrete equation with the doubling. When `r` is singular, the call solves the extended symplectic pencil of SciPy. The same happens when the doubling fails its residual test.
- The pencil has side `2n + m`. The call orders it with `ordqz` and the rule `iuc`. That route needs no inverse of `r`.
- `solve_discrete_are_pencil` runs the pencil alone. The stabilizing solution is unique, so the values agree with SciPy to about 1e-12 on well conditioned data.
- `solve_discrete_are` accepts a singular `r` when `r + b^T x b` is nonsingular, like SciPy. A zero `b` with a singular `r` is a fault, because the equation has no meaning. SciPy returns zeros for one such input.
- A matrix `e` needs to be nonsingular, like in SciPy. The matrices `q` and `r` need to be symmetric, like in SciPy.
- The side of every matrix stays at 128 or below. The argument `balanced` of SciPy does not exist. The package does not balance the pencil.
- On data with no stabilizing solution, SciPy sometimes returns a matrix that does not solve the equation. The package checks the residual of the answer in the original equation and gives a fault instead.

  Stacks of matrices are not supported. A side above 512 gives a limit fault.
- `qr_multiply` takes the modes `left` and `right`. It holds no pivoting and no conjugate option.
  The mode `economic` drops the zero rows and the zero columns of the
  wide matrix.
  `issymmetric` and `ishermitian` take `atol` and `rtol`, and the
  matrix with NaN is never symmetric.

  `bandwidth` takes one matrix and gives the lower and the upper
  width. `bandwidth_stack` takes a stack and gives two arrays of the
  counts, one for each matrix of the stack. The counts hold `f64`
  values. A matrix without a batch gives an array of one element.
  SciPy gives an integer for a matrix.
- `block_diag` takes two arrays. Call it again for more. `toeplitz`
  and `hankel` take an empty second row for the default of SciPy.
- `null_space` and `orth` give a basis whose signs may differ from
  SciPy. `null_space` of a matrix with no null vector is a fault,
  because the package holds no empty array.
- `expm` takes a matrix with a side up to 256. It uses the Pade degree
  13 for every matrix. SciPy picks a lower degree for a small norm.
  The last digits may differ.
- `solve_banded` takes the band form of SciPy. It factors the band with
  partial pivoting, as LAPACK does, so the cost grows with the side and
  the band. A singular matrix is a fault.
- `logm`, `sqrtm` and `eig_banded` stay out.
- `matmul`, `matvec` and `vecdot` follow the batch rule of NumPy. The
  batch dimensions broadcast. A row operand of `matmul` takes the place
  of a matrix with one row or one column.
- `matmul` of two rows and `vecdot` of two rows give an array of no
  dimension. Old code that read a shape of `[1]` there must read `[]`.
- `vecmat` takes one row and one matrix, with no batch axes.
  `tensordot_n` is `tensordot` with the count form of `axes`.
- `einsum` takes two operands, `einsum1` takes one and `einsum3` takes
  three. For more operands, build an `ESet` with `eset_new` and
  `eset_add`. Then call `einsum_set`.
- A set holds at most 8 operands of at most 8 axes each. The rule
  holds letters, a comma, an arrow and one ellipsis per operand.
- A rule holds at most 24 distinct letters. A length of one broadcasts.
- A rule with no letter in the result gives an array of no dimension.
  Old code that read `[1]` there must read `[]`.
- `get_window` knows 18 names. These are boxcar, triang, bartlett,
  hann, hamming, blackman, blackmanharris, nuttall, flattop, bohman,
  barthann, cosine, lanczos, parzen, kaiser, gaussian, tukey,
  exponential.
  `kaiser` takes beta up to 700.
- `dpss` takes the length, `nw`, `kmax` and `sym`. A `kmax` of zero
  gives one window as a vector, and a positive `kmax` gives that many
  rows. The norms are `approximate`, `subsample` and `2`.
  `dpss_ratios` gives the windows and the concentration ratios.
  It solves the tridiagonal problem by bisection, and it takes a length
  up to 100000.
- `general_cosine`, `general_hamming`, `general_gaussian`, `taylor`
  and `chebwin` take `sym`, where false gives the periodic window.
  `kaiser_bessel_derived` takes an even length and the symmetric form
  only, as SciPy does. SciPy warns for `chebwin` below 45 dB. The
  package gives no warning.
- `polyder` of a constant gives `[0]`, where NumPy gives an empty
  array, because the package holds no empty array. `polyint` takes the
  constants of the integration as a list. `poly_cplx` takes the roots
  as two lists and gives both parts, which NumPy would hold in one array.
- `choose` takes the choices stacked along the first axis, and the
  index must hold the shape of one choice. `piecewise` takes constants
  in place of functions. `compress` and `extract` give a fault when no
  element stays, because the package holds no empty array.
- `argpartition` gives the stable sort order. NumPy leaves the order
  inside each side of `kth` open, so only the side of each place is
  the same. The result holds the places inside each row of the last axis.
- The `unique_*` functions keep each NaN apart, as NumPy 2 does for
  them. `unique` and `unique_sorted` keep the old rule. `setxor1d` keeps
  a NaN that stands in both arrays twice, as NumPy does.
- `lexsort` takes the keys as the rows of a matrix. The last row is the
  main key.
- `pad` and `block` are keywords of Goose, so NumPy `pad` is `pad_modes`
  and NumPy `block` is `block_grid`. `pad_modes` takes one count per
  axis in two lists. The modes are `constant`, `edge`, `reflect`,
  `symmetric`, `wrap`, `linear_ramp`, `maximum`, `minimum`, `mean` and
  `median`.
  `value` is the end value of `linear_ramp`. `stat_length` is one count
  for every axis.

  `pad_axes` takes the full form of NumPy: `stat_length`,
  `end_values` and `constant_values` are lists of 0, 1, 2 or `2 * ndim`
  values. One value holds for both sides of every axis. Two values are
  the before and after sides of every axis. `2 * ndim` values hold one
  before and after pair per axis, in the order `b0, a0, b1, a1, ...`.
  A count above the width of an axis clips to the width, as NumPy does.

  In `pad_axes` a `stat_length` below 1 is a fault. In `pad_modes` a
  `stat_length` of 0 means the whole axis. The mode `empty` fills the
  pad with zero. NumPy leaves that pad undefined, so no value of NumPy
  is pinned. The package never returns memory that it did not set.
- `clarkson_woodruff_transform(a, sketch_size, seed)` is the CountSketch
  of scipy.linalg. Each row of `a` goes to one row of the result with a
  sign of plus or minus one. The package draws all rows first, then all
  signs, from `default_rng(seed)`. The stream differs from the stream
  of NumPy, so the values differ from SciPy. The structure, the
  norms in expectation and the faults agree. A `sketch_size` below 1, a
  value that is not finite or a result above the element limit is a
  fault.
- `append` and `delete` flatten the array, as NumPy does with no axis.
  `append_axis` and `delete_axis` take an axis. A `delete` result with no
  element is a fault, because the package holds no empty array.
- `tril_indices`, `triu_indices`, `diag_indices`, `unravel_index` and
  `indices` give one array: each row holds the places of one axis. A
  triangle with no place is a fault. `ix_` takes two lists.
  `block_grid` takes a grid of two rows and two columns.
- `unstack` gives pieces that `unstack_part` reads with the axis
  dropped. A piece of an array of one dimension keeps an axis of length
  one.
- These functions work on whole numbers: `gcd`, `lcm`, `invert`, the
  shifts, `bitwise_count` and the other `bitwise_*` functions. The shift
  count of `ldexp` follows the same rule. A whole number of at most 2^53
  in size is exact in a double. Any other element gives an `F_VALUE`
  fault. A shift result above 2^53 loses its low digits. `ldexp` may
  round twice below the smallest normal number.
- `unwrap` takes a negative `axis`, a `discont` and a `period`. The
  default `discont` is half the period, as in NumPy. `frexp` and `modf`
  give two arrays, because the package holds no tuple of arrays.
- `cov` takes `ddof` in place of `bias`: `bias=true` equals `ddof=0`.
  `cov_weights` takes the frequency weights and the analysis weights.
  The second array of NumPy goes through
  `cov_with` and `corrcoef_with`. A vector gives a matrix of one row
  and one column, because the package holds no array of zero
  dimensions.
- The NaN order statistics work on the whole array and give NaN for
  an array of only NaN. `nanargmax` and `nanargmin` refuse such an
  array with a fault, as NumPy raises an error.
- `cumulative_sum` and `cumulative_prod` take an axis. They take no
  flat form. `histogram_bin_edges` takes a count of bins.
- `histogram_rule` and `histogram_edges_rule` choose the count of bins
  by a rule: `auto`, `fd`, `sturges`, `doane`, `scott`, `rice`, `sqrt`
  or `stone`. The rule of Stone searches the count of bins and is slow
  for a long array. The count of bins stays at one million or below.
  A rule with weights holds no meaning, as in NumPy.
- The type `Polynomial` holds the coefficients from the lowest
  power. The functions carry the name of the type, because the
  namespace holds no methods. The functions of array_poly keep the
  names of NumPy and the order from the highest power.
- `polynomial_pow` takes a power of zero to 16. `polynomial_integ`
  holds the constant of zero and takes no count of turns. The
  integral and the derivative hold no domain and no window, as the
  class of NumPy holds none by default.
- `polynomial_linspace` takes the range and the count. The linspace
  of NumPy takes them from the domain with a count of 100.
- `polynomial_roots` gives the real parts and the imaginary parts of
  the roots in two arrays. The old form gave one array. A repeated
  root holds fewer digits.
- The names of NumPy take the generator as the first argument,
  because the package holds no global state. `seed` returns a
  generator. `bytes` fills a buffer of the caller and returns no
  value. `rand` and `randn` take the shape as one list.
- The state of the generator holds one word. The state of NumPy
  holds a key and a place. `get_state` and `set_state` carry one
  word. The numbers differ from the numbers of NumPy.
- `butter` gives the numerator and the denominator of the filter.
  `butter_sos` gives the sections. Neither holds an `fs` argument: the
  cut is a fraction of the Nyquist rate. The order stays between 1 and 32.
  Write the coefficients with real arithmetic, so a high order with a
  cut near zero or near one loses digits. Use a low order there.
- `sosfilt` takes a matrix of six columns. It refuses a section whose
  first denominator coefficient differs from one, and so do
  `sosfilt_zi` and `sosfiltfilt`. `sosfilt` takes no initial state.
- `zpk2sos` takes the zeros and the poles as real parts and imaginary
  parts in four arrays. It holds the pairing "nearest" of SciPy and
  no other pairing. It refuses a complex number with no conjugate
  partner and a filter of more than 64 poles or zeros.
- `tf2sos` finds the roots with `roots`. A filter of a high order
  with repeated zeros loses digits, as it does in SciPy. Use
  `butter_sos` for a Butterworth filter.
- `cheby1_sos` designs a Chebyshev filter of type I as sections. It
  takes the order, the ripple in decibels, the cuts and the type. It
  has no `fs` argument and no analog filter. The order stays at 32
  or below.
- `cheby2_sos` designs a Chebyshev filter of type II as sections. The
  cuts are the edges of the stop band. It has the same limits as
  `cheby1_sos`.
- `cheby1`, `cheby2` and `ellip` give the numerator and the
  denominator of a digital filter, as SciPy gives them. They take the
  order, the ripples in decibels, the cuts and the type. `ellip`
  takes the ripple of the pass band and the attenuation of the stop
  band. They have no `fs` argument, no analog filter and no `output`
  option. Use `zpk2sos` or the `_sos` forms for a high order.
- `bessel` and `bessel_sos` design a digital Bessel filter. They take
  the order, the cuts, the type and the norm `phase`, `delay` or
  `mag`, `phase` by default. The poles come from the roots of the
  reverse Bessel polynomial, improved by Newton steps. They have no
  `fs` argument, no analog filter and no `output` option. The order
  runs from one to 32.
- `ellip` faults when the attenuation is not above the ripple.
  SciPy gives NaN or zeros in that case.
- `residue` takes real coefficients only. It gives the residues as
  two arrays: the real parts and the imaginary parts. It gives the
  poles the same way. The direct terms stay real.
- `residue_complex` takes complex coefficients. It gives six arrays.
  The residues, the poles and the direct terms each come as a real
  part and an imaginary part. The parts are two separate lists. The
  function has the same poles and residues as SciPy.
- `residue` finds the poles with `roots`. A group of close poles
  stands for one repeated pole. The default rule takes the mean of
  the group. The package then moves that mean to the root of the
  derivative of the group. A repeated pole so keeps its digits. The
  rules `min` and `max` take a pole of the group as it stands.
- Poles of one size, such as a complex pair, may come in the other
  order than in SciPy. Each pole keeps its own residue.
- `residue` takes at most 257 coefficients in a polynomial.
- `residuez` takes real coefficients from the constant term up. The
  poles stay the roots of the denominator in z. Its limits follow
  `residue`.
- `invres` and `invresz` take the residues and the poles as real
  parts and imaginary parts. They give real coefficients. A result
  with a complex coefficient is a fault. The residues and the poles
  must hold the same count, and a group of poles must sit side by
  side.
- `unique_roots` takes the roots as real parts and imaginary parts
  and gives the same form. It takes at most 4096 roots.
- `lfilter` filters one line from a state of zero. `lfilter_axis`
  filters an array along an axis. `lfilter_state` also takes the
  initial state `zi` and gives the final state. Along the axis, the
  state holds one value less than the longer coefficient list. The
  other axes keep the shape of the signal. SciPy lets the state
  broadcast, but this package needs the exact shape.
- `filtfilt` takes the odd extension of SciPy. The pad holds three
  times the coefficient count. No other pad option exists.
- `hilbert` returns the real and the imaginary parts as two arrays,
  as `fft` does.
- The shape holds at most 8 dimensions.
- One array holds at most `MAX_ELEMENTS` elements, 268435456.
- Every dimension holds at least one element. The release refuses an
  array of length zero.
- The stacks join two arrays along an axis. `vstack` joins the rows
  under one another. `hstack` joins the rows beside each other.
  `dstack` joins them in a third dimension. `column_stack` joins them
  as columns.
- The constants of `scipy.constants` live in the namespace
  `constants`. Examples are `constants::c`, `constants::pi`,
  `constants::inch` and `constants::kilo`. The module holds 155 numbers
  under the names of SciPy. The values come from SciPy 1.18, which
  holds CODATA 2022. A program that has its own `constants` namespace
  must rename one of the two.
- `value`, `unit` and `precision` read the table of 445 physical
  constants by name. The name must match in case, as the keys of SciPy
  do. A name that the table lacks is a fault, where SciPy raises an
  error. The table holds the entries that CODATA 2022 replaced, as SciPy
  holds them. SciPy warns when it reads one of those names. This package
  does not warn.
- `precision` divides the uncertainty by the value and keeps the sign,
  as SciPy does. A constant below zero gives a result below zero.
- `find` gives the matching names as one text, one name per line, in
  sorted order. SciPy gives a list of names. The text of `find` is the
  same names in the same order.
- The package holds no `physical_constants` dictionary. The table sits
  in `constants::NAMES`, `constants::UNITS`, `constants::VALUES` and
  `constants::UNCERTAINTIES`, in the order of the names.
- `convert_temperature` takes Celsius, Kelvin, Fahrenheit and Rankine.
  The name is the long name or C, K, F or R, in any case. The steps follow
  SciPy, so the digits agree. `lambda2nu` and `nu2lambda` divide the
  speed of light by each element.
- `moveaxis` takes one axis. `moveaxis_many` takes lists of axes, as
  NumPy does. A negative axis counts from the end.
- `rot90` takes the two axes as `axis0` and `axis1`. A negative `k`
  turns the other way.
- `broadcast_arrays` takes two arrays, not a list. `broadcast_to`
  returns a copy and not a view, as the package holds no views.
- `select` takes the conditions and the choices as two arrays. The
  first axis of each array holds the list that NumPy takes as a
  Python list. The first true condition wins. The default is a number.
- `apply_along_axis` calls a function that takes a line of one
  dimension and gives a line of one dimension. The length of the
  result replaces the axis. `apply_along_axis_scalar` takes a function
  that gives one number, and the axis drops. The caller passes the
  function at the end of the call. NumPy decides between the two by
  the type of the result.
- `argwhere` gives one row of positions per element that is not zero,
  and one column per dimension. No match gives zero rows. NumPy gives
  the same shape.
- `isclose` holds the rule of NumPy. The difference sits under the
  absolute size plus the relative size times the second number. The
  infinity equals itself. The missing value matches itself only under
  the flag.
- `histogram2d` and `histogramdd` take the range from the input, as
  the histogram of one dimension does. The range runs from the
  smallest value to the largest value. NumPy takes a `range` option. A column where every value holds
  the same number is a fault. The counts hold no `weights` and no
  `density` option.
- `histogramdd` takes rows of points. Each column holds one dimension.
  The edges come back as one row per dimension. The counts sit in the
  order of the rows of the grid, so the last dimension runs fastest.
- `frombuffer` and `fromfile` decode numbers of 64 bits, least
  significant byte first, as the NumPy default does. The package holds
  no other type of number, so there is no `dtype` option. A buffer
  that holds no whole count of numbers is a fault.
- `fromfile` drops a rest of fewer than eight bytes at the end, as
  NumPy does. A count above the file takes what the file holds. A file
  that does not read is a value fault. It reads the whole file into
  memory first, so the caller bounds the size of the file.
- `array_equal` takes one shape and equal elements. It takes no
  broadcast, as NumPy does not.
- `flatnonzero` gives the flat positions of the elements that are not
  zero. It holds the same result as `nonzero`.
- `interpn` interpolates a regular grid at many points. It takes the
  grids in one flat list and the length of each axis. It takes the
  values and the points. The methods are `linear`, `nearest`,
  `slinear`, `cubic`, `quintic` and `pchip`.
- A point outside the grid is a fault in `interpn`. The caller can turn
  the check off and give a fill value. The caller can also set
  `extrapolate`. Then the point takes the value of the end cell, as
  `fill_value=None` does in SciPy.
- A grid that does not rise strictly is a fault in `interpn`.
- `interpn` takes at most eight axes. The spline methods take at most
  512 points on an axis and need more points than the degree.
- The methods `cubic` and `quintic` fold one axis at a time, like the
  methods `cubic_legacy` and `quintic_legacy` of SciPy. SciPy gives
  `cubic` and `quintic` by an iterative solve with a tolerance near
  1e-6. Its values differ from the exact tensor product spline by
  about 1e-5, and by more outside the grid. This package gives the
  exact spline. A caller who needs the digits of SciPy must expect
  that difference.
- `meshgrid` takes two vectors and returns two arrays. The indexing
  `"xy"` gives the shape (len(y), len(x)). The indexing `"ij"` gives
  (len(x), len(y)). It takes two inputs, not more.
- `cross` takes vectors of three elements, as NumPy 2 does. A vector
  of two elements is a fault. It takes arrays of at most two
  dimensions. One vector broadcasts against the rows of the other.
  It takes no `axis` option.
- A row of one dimension joins in its own direction under `vstack`
  and `dstack`. NumPy gives the same result.
- `dstack` joins an array of three or more dimensions along its third
  dimension, as NumPy does.
- `stack` takes the place of the new dimension. The axis gives that
  place. A negative axis counts from the end. The default is the
  first dimension.
- `transpose` swaps the two dimensions of a matrix. `transpose_perm`
  takes any permutation of the axes.
- An axis reduction drops the reduced axis from the shape. It has no
  `keepdims` option. Use `expand_dims` on the result instead.
- The reductions that skip NaN work on the values that are not NaN.
  `nansum` and `nanprod` give 0 and 1 for an array of only NaN. The
  other five give NaN, as NumPy does, and they raise no warning.
- `nanvar` and `nanstd` divide by the count of the values that are
  not NaN, less `ddof`. A divisor below one gives NaN. A `ddof` below
  zero is a fault.
- `nan_to_num` takes a value for NaN, one for the positive infinity
  and one for the negative infinity. The defaults are zero, the
  largest double and the least double. `nanmedian`, `nanpercentile`,
  `nanargmin` and `nanargmax` stay out.
- `argmin` and `argmax` give the flat position of the first extreme.
  They take no axis.
- The elementwise math holds the functions of this table. The
  functions of NumPy that work on another type stay out.
- A comparison gives an array of 1.0 and 0.0. The array holds one
  element type, so a truth value has no type of its own.
- `round` sends a half to the even neighbour, as NumPy does. `floor`,
  `ceil` and `trunc` send a value to a whole number.
- `round` takes a count of decimals. The count of zero keeps the
  whole number. `around` takes the same count.
- `eye` takes a count of rows and a count of columns. A count of
  columns of zero means the count of the rows. The package holds no
  dimension of length zero. The offset moves the diagonal.
- `identity` gives the square matrix of the identity. It takes no
  offset.
- `diagflat` holds a row of values on a diagonal of an offset. The
  side of the matrix holds the count of the values and the size of the
  offset.
- `logspace` takes a count of numbers and a base. `geomspace` takes a
  count of numbers and no base. A zero at the start or at the end of a
  geometric sequence is a fault.
- `array` builds a one-dimensional array from a row of values. It
  holds no nested list, so a matrix comes from `from_rows`.
- `floor_divide` follows the rule of NumPy. The remainder comes
  first, so the quotient of one and a tenth is nine.
- `isinf` and `isfinite` use the rule of the standard library. A
  number of great size holds no infinity.
- The hyperbolic functions and their arcs come from the formulas of
  the package. The standard library holds no arc of a hyperbolic
  function. They hold the digits of a double.
- `rint` sends a half to the even neighbour, as `round` does. `fix`
  cuts a value toward zero, as `trunc` does.
- `mod` holds the sign of the divisor, as Python holds it. `fmod`
  holds the sign of the dividend, as the C library holds it. A divisor
  of zero gives NaN under both.
- `cbrt` holds the root of the cube that rounds right. The C library
  behind NumPy gives the value next to it on some inputs, up to two
  steps away.
- `hypot` holds the length of two numbers. The squares carry their
  lost parts, so the result holds every digit.
- `exp2`, `cbrt`, `copysign`, `nextafter` and `spacing` come from the
  formulas of the package. The standard library of the older Goose
  version holds none of them. Both supported versions give the same
  values.
- `nextafter` gives the number of the machine next to the one of the
  caller. `spacing` gives the distance to it. The distance below a
  power of two is half the distance above it.
- `copysign` gives the size of a number with the sign of another. A
  zero of the negative side gives the sign of minus.
  A NaN holds no sign here, so a NaN of the negative side gives the
  sign of plus. NumPy reads the sign bit of the NaN.
- The names of NumPy that the package holds under another name are
  these. `var` holds the name `variance`. `pad` holds the name
  `pad_value`. `trapz` holds the name `trapezoid`.
- `cumsum` and `cumprod` take an axis. `diff` and `flip` take an axis
  too, and `diff` drops one element of it.
- `variance` and `std` take a divisor count. The divisor is the
  length minus that count.
- `slice_axis` slices one axis at a time. NumPy slices every axis in
  one call. A caller slices each axis in turn.
- A slice that holds no element is a fault. The check runs before any
  element moves.
- `take` and `put` use the flat position, the row-major position of
  an element. `put` writes into the array of the caller.
- `nonzero` gives the flat positions. It gives no position per axis.
- `take_along_axis` takes an index array with one position for each
  element of the other axes.
- `mask` keeps the elements of one array that hold true in a second
  array of the same length.
- The sets hold each value once, in the order of its value. `union1d`
  holds the values of two rows. `intersect1d` holds the values that
  two rows hold in common. `setdiff1d` holds the values of the first
  row that the second row holds no value of.
- `in1d` tells for each value of a flat row whether another row holds
  it. `isin` tells the same for the elements of an array and keeps
  the shape of the array. Both hold 1.0 and 0.0. Both take a flag
  that turns the answer around.
- A NaN holds no value in a membership test, not even another NaN.
  So `in1d` and `isin` never match a NaN. `intersect1d` holds no NaN
  and `setdiff1d` keeps a NaN of the first row, as NumPy does.
- The union holds one NaN and puts it at the end. Two NaN of the rows
  come once, as NumPy holds them in a unique array.
- The sets sort the second row and search it, so a large row takes a
  long time. The sort is an insertion sort, as the sort of the
  package is.
- `union1d` holds the values of two rows in one row. Two rows that
  hold more elements together than the limit give a fault.
- The sort is stable and gives a copy. Two equal elements keep the
  order of their positions.
- The sort is an insertion sort. It holds a small or a middle array
  well. A large array takes a long time.
- `unique` gives the values in the order of their first appearance.
  `unique_sorted` gives them in the order of their value.
- `searchsorted` takes a sorted array and gives the place before the
  equal elements. It does not check that the array holds a sorted
  order.
- The histogram holds equal bins between the smallest and the largest
  value. An array of one value is a fault, because it holds no bin.
- `median` and `percentile` sort a copy of the array.
- `quantile` takes a fraction from zero to one. `percentile` takes a
  percentage from zero to one hundred. Both take the name of a
  method. The names are `linear`, `lower`, `higher`, `nearest`,
  `midpoint`, `inverted_cdf`, `averaged_inverted_cdf`,
  `closest_observation`, `interpolated_inverted_cdf`, `hazen`,
  `weibull`, `median_unbiased` and `normal_unbiased`. The default is
  `linear`. A name outside this list is a fault.
- `mode` gives the value that comes most often and its count. Of two
  values with the same count, the smaller one wins. A NaN is no
  value, so it stays out of the count.
- `average` takes a weight for each value. The weights hold the
  shape of the values. `average_axis` takes one weight for each place
  of the axis. Weights that add up to zero are a fault.
- The order statistics along an axis drop the axis, as the other
  reductions do.
- A matrix is an array with two dimensions. The package holds one
  array type.
- `dot` of two rows gives a number. It is held in an array of no
  dimension.
- `dot` of a row and a matrix sums over the row and the first axis of
  the matrix. `dot` of a matrix and a row sums over the last axis of
  both, as NumPy does. The old form summed over the first axis of the
  matrix. The old form was wrong.
- `norm` gives the Frobenius norm of a matrix and the second norm of
  a row. Both are the square root of the sum of the squares.
- `solve`, `inv` and `det` use a Gaussian elimination with a partial
  pivot. A matrix without a pivot is a fault. The determinant of such
  a matrix is zero, so `slogdet` gives zero and no fault.
- `matrix_power` takes a whole power. A power below zero gives the
  inverse of the matrix.
- `cholesky` gives the lower factor, as NumPy does under its default.
- `qr` gives the reduced form, as NumPy does under its default. The
  orthogonal matrix holds as many columns as the smaller dimension.
- The eigenvalues of a symmetric matrix come from the Jacobi method.
  They come in the order of their value, from the smallest to the
  largest.
- The eigenvalues of a matrix that is not symmetric come from the QR
  algorithm with a shift. They come in no particular order.
- `eig` and `eigvals` take real eigenvalues only. A matrix with a
  complex pair of eigenvalues is a fault for them. `eigvals_complex`
  takes a real square matrix of side 512 or below. It gives the
  eigenvalues as a complex array. They sort by real part, then by
  imaginary part. NumPy uses another order.
- The eigenvectors of a matrix that is not symmetric come from an
  iteration. A matrix whose eigenvalues hold no vector is a fault.
- `svd` gives the full form, as NumPy gives it under its default.
  The left vectors hold as many rows and columns as the matrix has
  rows. The right vectors hold the transpose of the vectors.
- The singular values hold only the count of the smaller dimension.
- `lstsq` uses the singular values. A singular value below the
  tolerance gives a zero in the solution, as NumPy gives it.
- `matrix_rank` uses the larger side of the matrix, the machine
  epsilon and the largest singular value, as NumPy 2 does.
- `pinv` uses the same cutoff. A negative `rtol` takes that default.
  A singular value at or under the cutoff counts as zero. A matrix
  of exact rank deficiency gives the result of NumPy to about 1e-14.
- The coefficients of a polynomial come from the highest power to the
  lowest, as NumPy holds them.
- `roots` uses the rule of Durand and Kerner, so it finds real and
  complex roots. It gives the real parts and the imaginary parts in
  two arrays, as `fft` does. The old form gave one array and made a
  complex root a fault. The new form gives `re, im, fault`. The
  roots come in no fixed order, and a repeated root holds fewer
  digits.
- `roots` takes a degree of 256 or below, because the cost grows with
  the square of the degree. A larger degree is a fault.
- The transform works on two arrays: one holds the real parts and one
  holds the imaginary parts. The type `CArr` holds a complex array.
- `fft` takes every length. A power of two takes the radix-2 method.
  Every other length takes the Bluestein method, so it allocates the
  next power of two above twice the length.
- `ifft` divides by the length, as NumPy does. `fft` and `ifft` of
  the same parts give the signal back.
- `fft2` takes a matrix and transforms both axes. It takes no shape
  argument and no choice of axes.
- `fftfreq` puts the positive frequencies first, then the negative
  ones, as NumPy does. A step of zero is a fault.
- `rfft` gives the positive half of the frequencies. The count of the
  half is the length divided by two, plus one.
- `irfft` takes the parts of that half and the length of the signal.
  A count of the parts that does not hold the length is a fault.
- `fftshift` and `ifftshift` roll every axis, as NumPy does with its
  default. For an odd length the two differ. They take no `axes`
  option.
- `fftn`, `ifftn`, `fft2c` and `ifft2` take the real parts and the
  imaginary parts as two arrays of one shape. `fft2` takes a real
  matrix only. `fft2c` takes both parts. They hold no `s` option, no
  `axes` and no `norm`.
- `fftn_ex` and `ifftn_ex` add the `s`, `axes` and `norm` arguments of
  SciPy. A size of `-1` in `s` keeps the size of the input. The list
  `[1000]` for `axes` means every axis, or the last `len(s)` axes.
  The count of `s` must equal the count of `axes`. The default norm is
  `backward`.
- `rfftfreq` gives the frequencies from zero to half the rate. A step
  of zero or a length below one is a fault.
- The generator comes from the standard library. A seed gives the
  same numbers on every run and on both supported versions. The
  package holds no entropy of the machine, so the caller gives the
  seed.
- `uniform` gives a number of the half-open range. `random_uniform`
  gives a number of the range zero to one.
- `normal` uses the pair of Box and Muller. It gives one number of
  the pair and holds the other for the next call. So the numbers of
  one generator come in a fixed order.
- `poisson` uses the method of Knuth. A rate above 30 is a fault,
  because the method holds a small rate only.
- `choice` takes a row and a count. A pick without a repeat that asks
  for more elements than the row holds is a fault.
- `permutation` gives the shuffled order of a count. `shuffled` gives
  a shuffled copy of a row. `shuffle_rows` shuffles the rows of a
  matrix in place, so the array of the caller changes.
- `empty` gives an array with no element set. The package holds no
  uninitialized memory, so every element holds zero.
- `putmask` and `place` write into the array of the caller. Both read
  the values in a cycle when they run out, as NumPy does. Values that
  hold no element are a fault.
- `split` cuts an array into parts of an equal count. A count that
  does not cut the axis is a fault, as in NumPy. `array_split` cuts
  it into parts of a count as equal as the axis allows. The first
  parts hold one element more.
- The parts of a cut come back in a `Parts` value. `part` gives one
  of them as an array. A struct of the package holds one growable
  field. So the buffer of a `Parts` holds the length and the count of
  every part first. The elements come after them.
- `pad_value` grows an array along one axis with a value at the
  start, at the end, or at both. It fills with one value, so it holds
  no mode of edge or wrap.
- `interp` takes the values between the points of a rising row. A
  point below the first point takes the first value. A point above
  the last point takes the last value. Points that do not rise are a
  fault.
- `gradient` takes the difference of the one side at the two ends.
  It takes the difference of the two neighbours in the middle.
- `tril` and `triu` take a count of the diagonal. A count of zero
  holds the diagonal and one side of it.
- `trim_zeros` drops the zeros at the two ends of a row. A row of
  only zeros is a fault.
- A solver takes the function of the caller as a parameter of a
  generic type. The caller passes the function as a block at the end
  of the call. A call takes one block. So a solver that needs a
  second function takes it as a named function before the block.
- `brentq` finds a root between two points where the function changes
  its sign. The two points that hold no change of sign are a fault.
- `brenth` finds the same root with the step of the false position.
  The step holds the root between the two points at every turn.
- `toms748` finds a root in a bracket with the algorithm 748 of
  Alefeld, Potra and Shi. It takes the function as a block. The count
  of steps of Newton type goes from one to a hundred. SciPy prints a
  warning when the count lies outside those limits. The package takes
  the nearest count and does not print a warning.
- `toms748` gives a fault when the two ends of the bracket hold values
  of the same sign. SciPy gives an error in that case.
- `secant` finds a root from two points. It takes the function as a
  block. The method uses the last two points of the function. SciPy
  takes the second point as an option, and it takes one point when the
  second one is missing. The package takes both points.
- `halley` finds a root from a start point with the method of Halley.
  It takes the first and the second derivative as named functions of
  the caller, and the function as a block. SciPy gives a value that
  did not converge when the method cycles. The package gives a fault.
- `root_scalar` takes a bracket and the method `brentq`, `brenth`,
  `bisect` or `ridder`. It gives no result object. The methods that
  need a derivative stay with `newton`. A call with a start point and
  no bracket is a fault.
- `newton` takes the derivative of the function as a named function.
  A derivative of zero is a fault.
- `fsolve` finds the point where several equations hold zero. The
  function of the caller takes a vector and gives a vector. The
  method takes the derivative by a small difference, so the root
  holds fewer digits than the other methods.
- `root` finds the point where several equations hold zero. The
  methods are `hybr`, `lm`, `broyden1`, `broyden2`, `linearmixing`,
  `diagbroyden`, `excitingmixing`, `anderson`, `df-sane` and
  `krylov`. The result holds the point and the values of the
  equations.
- The method `hybr` of `root` takes damped Newton steps, not the
  trust region of MINPACK. The path differs from SciPy, but the root
  matches.
- The four simple iterations of `root` use a line search and take the descent sign for the first guess. SciPy takes
  the other sign and diverges on common systems with
  `linearmixing` and `excitingmixing`. The roots match, the paths
  differ.
- The method `df-sane` of `root` takes no derivative. It uses the
  spectral residual method of La Cruz, Martinez and Raydan. The line
  search is the non-monotone one of the same authors. The tolerance is
  relative to the first residual. The count of turns limits the
  count of evaluations to five times the count plus a thousand.
- The method `krylov` of `root` takes no derivative either. It uses
  a Newton step and a Krylov iteration for the linear part. The
  product of the derivative and a vector comes from a small
  difference. The line search of Armijo takes the step. The path
  differs from SciPy, but the root matches.
- The method `lm` that stops at a point that is not a root is a
  fault.
- Every solver takes a tolerance and a count of turns. A tolerance
  of zero and a count below one are faults. A method that does not
  reach the tolerance in the count of turns is a fault.
- `minimize_scalar` finds the smallest point between two points. The
  method cuts the bracket by the golden section, and it fits a
  parabola at the end. The two points must hold a minimum between
  them, else the method ends at one of them.
- `differential_evolution` uses the strategy `best1bin` with the
  default settings of SciPy. A seed fixes the random numbers, so the
  results differ from those of SciPy. The result holds the point, and
  the value is the last element. The arguments `strategy`, `polish`,
  `constraints`, `workers` and `init` stay out. The run has no local
  polish, so a result is a good start for `minimize` and not a final
  value. Use a large `popsize` for functions with many local minima.
- `dual_annealing` takes bounds, a seed, `maxiter` and an initial
  temperature. It anneals with a heavy-tailed step and runs a local
  search at each new best point. The step uses a Cauchy form, not the
  exact distribution of SciPy. The random numbers differ from SciPy.
  The local search is the method of Nelder and Mead, not L-BFGS-B.
- `basinhopping` takes a start, `niter`, the temperature `T`, a
  step size and a seed. Each turn takes a random step and a local
  search, then applies the rule of Metropolis. The arguments
  `minimizer_kwargs`, `take_step`, `accept_test`, `callback` and
  `interval` stay out. The result holds the point and the value.
- `minimize` finds the smallest point of a function of several
  variables. The method of Nelder and Mead holds a simplex. A start
  far from the point of the smallest value can end at a local point.
- `minimize_grad` finds the smallest point with a method that uses the
  gradient. The block of the caller takes a vector. It gives the
  value first and the gradient after it. The methods are `bfgs`,
  `cg`, `lbfgsb` and `newton-cg`. The bounds hold one value per variable. The
  start that lies outside the bounds moves to the border.
- `minimize_constrained` takes one block. It gives the value, the
  gradient and then the constraints. The first `neq` constraints equal
  zero. The others stay zero or above. Bounds are optional.
- `minimize_constrained` covers the problems of the SciPy methods
  `SLSQP` and `trust-constr`. It does not use their algorithms. It uses
  the method of the multipliers with L-BFGS-B, so the path differs. The
  optimum matches to about six digits. The gradients of the constraints
  come from central differences.
- `newton-cg` gets the product of the Hessian and a vector from a
  small difference of two gradients. SciPy does the same without `hess`. It
  takes no bounds, and a call with bounds is a fault. Its `xtol` is
  the mean step size that ends the run.
- `minimize_grad` replaces the methods `bfgs` and `cg` with `lbfgsb`
  when bounds are given, as SciPy does. SciPy prints a warning in that
  case. The package does not.
- `minimize_grad` stops on the gradient alone for the methods `bfgs`
  and `cg`, as SciPy does. The method `lbfgsb` also stops on the
  value.
- `minimize_grad` has no `hess`, no `epsilon` and no `callback`. A
  function without a gradient has no method of the gradient, so the
  caller gives the gradient.
- `minimize_grad` gives the point and the value. It holds no count of
  the turns and no status of the end.
- `curve_fit` takes the model as a block `{ x, p => ... }`, the data
  and a start. It gives the parameters and the covariance. The
  covariance uses the scale of SciPy for the default setting. The
  fifth argument `sigma` takes one value, one value per point or the
  full covariance matrix of the points. A matrix that is not positive
  definite, a value of zero and a wrong count are faults.
- The sixth and seventh arguments of `curve_fit` are `lower` and `upper`.
  They bound the parameters. An empty
  array means no bound, and a bound of infinity leaves one side open.
  A start outside the bounds is a fault. It has no `method`.
- `least_squares` takes the same `lower` and `upper` arrays. A parameter
  on a bound stays there while the step points outward. SciPy uses the
  trust region method there. The fit reaches the same minimum.
- `curve_fit` parameters match SciPy to about five digits. The
  covariance matches to about three digits, because SciPy takes it
  from a rough difference of the derivatives.
- `least_squares` fits a function to values. The method of Levenberg
  and Marquardt takes the matrix of the derivatives by a small
  difference. So the fit holds fewer digits than a fit that takes the
  derivatives from the caller.
- The integration functions take the values of a signal and a
  spacing. A caller that needs the integral of a function samples the
  function first.
- `trapezoid` takes the rule of the trapezoid. `simpson` takes the
  rule of Simpson. An even count of parts takes the correction of
  Cartwright on the last part, as SciPy 1.18 does. Before v0.110.0 the
  even count took the trapezoid on the last part. The value of an
  even count differs from that release on.
- `trapezoid` of a signal of fewer than two values gives zero.
  `simpson` of one value gives zero. `simpson` of two values takes
  the rule of the trapezoid. Before v0.110.0 these were faults.
- `trapezoid_axis` and `simpson_axis` take the integral of an array
  along one axis. The axis drops from the shape, as SciPy does. A
  signal of one dimension gives a result of no dimension. The places
  `x` are not taken; the spacing is `dx`.
- `cumulative_trapezoid_axis` takes the running total along one axis
  of an array. The axis holds one value less, as SciPy does without a
  start value. A signal of fewer than two values on the axis is a
  fault.
- `cumulative_trapezoid` takes the running total. The result holds
  one value fewer than the signal, as NumPy does.
- `cumulative_simpson` takes the signal and the places and gives the
  integral up to every point. The intervals may differ in width. The
  rule of Simpson of Cartwright gives each of them. A signal of two
  values falls back to the trapezoid. The result holds one value fewer
  than the signal.
- `cumulative_simpson_axis` runs along one axis of an array. The places
  hold one value per point of the axis, or one per element. Without
  places the spacing is `dx`.
- The argument `initial` of `cumulative_simpson` adds a start value
  and puts it first. The result then holds as many values as the
  signal.
- `cumulative_simpson_dx` takes one spacing for every interval and the
  same `initial`. Neither function takes an axis or a shape other than
  one dimension.
- `romberg` takes the extrapolation of Romberg. The count of the
  parts of the grid holds a power of two, as the method of the
  references holds. A count that holds no power of two is a fault.
- `spline` holds the coefficients of a natural cubic spline through
  the points of a signal. The second derivative of the natural spline
  holds zero at the two ends, as the spline of the references does.
- `spline_value` gives the value of a spline at a point. A point
  outside the points of the signal is a fault.
- `spline_values` gives the values of a spline at a row of points.
- `convolve` gives the overlap of two signals. The result holds a
  count that is the sum of the two counts minus one.
- `correlate` gives the match of a pattern in a signal. It is the
  overlap with the pattern turned around, as SciPy holds it under
  its full form.
- `detrend` takes the straight line under a signal out of it. The
  line comes from the least squares of the package.
- `welch` gives the size of the parts of a signal by their
  frequency. The parts overlap by half. Every part takes the window
  of Hann. The sizes come as the mean of the squares of the values of
  the transform. The result holds the positive frequencies of one
  side.
- `cossin` gives the cosine-sine decomposition of an orthogonal
  matrix. It takes the split of the rows and of the columns as two
  integers. A float split is a fault. SciPy takes the integer part of
  a float.
- `cossin` takes both parts of the split. A call with one part only
  has no default. SciPy sets the second part to one.
- `cossin` gives the factors up to the sign of each paired column and
  row. LAPACK fixes that sign by its method of bidiagonalization. The
  package fixes it by its own rule. A free block of a factor is fixed
  by its properties only.
- An angle of `cossin` below `1.0e-8` has an absolute accuracy of
  that order. A cluster of equal cosines hides the small angles.
- `cossin` takes a real matrix only. A complex unitary matrix waits
  for the complex array functions of the package.
- `cossin_separate` takes one matrix. A stack of matrices gives a
  fault of kind `F_SHAPE`. SciPy fails for a stack there too.
- The element limit of `cossin` does not bound the time of a call.
  The work of one matrix grows with the cube of its side.
- `qz` and `ordqz` give the generalized Schur form of a real pair.
  `qz_c` and `ordqz_c` take a complex pair and give the complex form.
- `ordqz` takes the four names `lhp`, `rhp`, `iuc` and `ouc`. The calls
  `ordqz_by`, `ordqz_complex_by` and `ordqz_c_by` take a block of the
  caller instead. The block takes the alpha and the beta of a diagonal
  block. It gives true for a chosen block.
- A block of sort that crashes ends the program. The package cannot
  catch a crash of the caller.
- SciPy picks the type of the result by the argument `output`. Goose
  has no result type that changes at run time. `qz_complex` and
  `ordqz_complex` therefore give the triangular complex form.
- `qz` gives the factors `Q` and `Z` up to the sign of each column.
  LAPACK fixes that sign by its own steps. The eigenvalues are equal to
  those of SciPy.
- The phase of a column of `q` and `z` can differ from the phase that
  LAPACK gives. The eigenvalues are equal.
- The order of the eigenvalues of `qz` can differ from the order of
  SciPy for the same pair. The set of the eigenvalues is equal. After
  `ordqz` the chosen eigenvalues come first in both libraries.
- A pair of complex eigenvalues gives two equal values of `beta`. They
  hold the geometric mean of the two entries of `BB`. LAPACK scales
  them in another way. The quotient `alpha / beta` is equal.
- `ordqz` swaps two blocks by the solution of a generalized Sylvester
  system. The package tests the swap by its residual. A residual above
  `1.0e-11` of the largest entry of the window gives a fault of kind
  `F_VALUE`. A pair of equal eigenvalues in two blocks that must swap
  causes that fault. LAPACK also refuses an unsafe swap.
- A pair whose values come near the limit of the number format gives a
  result. The package scales the pair by a power of two before the
  iteration. A power of two leaves every digit of a value alone.
- The iteration takes at most 30 turns for each eigenvalue. A pair that
  needs more gives a fault of kind `F_LIMIT`.
- A pair of side 512 takes about 50 seconds on a current machine. The
  work grows with the cube of the side.

## Security

- `cdist` and `pdist` check the size of the result before they build
  it. A result above `MAX_ELEMENTS` is a fault, so a large set cannot
  exhaust memory.
- Every shape passes its checks before any element is built. A shape
  with a dimension below one is a fault, so no length lies about its
  buffer.
- `arange` computes its count before it allocates. A range that
  reaches over `MAX_ELEMENTS` is a fault, so a bad step cannot grow
  memory without bound.
- A comparison never reads a value as text, so it cannot fail on a
  value. A shape that does not broadcast is a fault, and the fault
  comes before any element moves.
- The IEEE rules hold for the values. A division by zero gives an
  infinity. A log of a number below zero gives NaN. No fault comes
  from a value, only from a shape, a range or a limit. Check `is_nan`
  of the standard library when the input is hostile.
- `eye` and `diagflat` check the size of the matrix before they build
  it. A count that reaches over the limit is a fault, so a bad count
  cannot grow the memory without bound.
- The sparse calls check the sides, the count of the entries and the
  places of the entries. An entry outside the matrix is a fault, so a
  bad place cannot write outside the buffer. The four sparse formats
  share one check of the entries. The compressed columns sort the
  entries with a heap sort, so no data gives a slow case.
- The diagonal format checks its offsets before it builds the matrix. An
  offset outside the matrix is a fault. Two equal offsets are a fault. A
  count of the values that does not match the diagonals is a fault. A
  value that is not finite is a fault, so a bad value cannot spread
  through the matrix.
- The coordinates of a block matrix hold every value of every block.
  Their count grows with the count of the stored values. A count above
  `MAX_SPARSE_NNZ` is a fault of kind `F_LIMIT`. The check comes before
  the entries are built.
- The block format checks its block size before it builds the matrix. A
  block size below one is a fault of kind `F_RANGE`. A block size that
  does not divide the shape is a fault of kind `F_RANGE`. The count of
  the stored values above `MAX_SPARSE_NNZ` is a fault of kind `F_LIMIT`.
  The count is checked by a division, so no product can overflow. A value
  that is not finite is a fault, so a bad value cannot spread through
  the blocks.
- `bsr_scale` checks every product. A factor that makes a product not
  finite is a fault of kind `F_VALUE`.
- The format of keys checks a place before it turns it into a whole number.
  A place that is not a whole place inside the matrix is a fault of kind
  `F_RANGE`. The check comes first, because the cast of such a number
  gives no defined value.
- The values of a matrix of keys stay finite. A value of NaN or
  infinity gives a fault of kind `F_VALUE`. A bad value cannot spread
  through the matrix.
- `dok_sum` checks the sum of the values. A sum that is not finite is a
  fault of kind `F_VALUE`. SciPy lets the sum grow to infinity.
- `dok_scale` checks every product. A factor that makes a product not
  finite is a fault of kind `F_VALUE`.
- The hash of a place holds a random number of the matrix, taken from the
  system. A caller cannot choose places that collide. A long series of
  writes cannot make the lookup slow.
- The table of a matrix of keys holds one and a half slots for each key.
  A count of keys above `MAX_SPARSE_NNZ` is a fault of kind `F_LIMIT`.
  The check comes before the table grows.
- The diagonal format checks an offset before it turns it into a whole
  number. An offset that lies outside the range of an integer is a fault.
  The cast of such a number gives no defined value, so the check must come
  first. An offset of that size would otherwise build the wrong diagonal
  and read as a correct answer.
- `csr_kronsum` builds two products with `csr_kron`, and each product
  checks its sides and its count of stored values. `csr_add` checks the
  count of the sum, so a result above `MAX_SPARSE_NNZ` is a fault of kind
  `F_LIMIT`. The count checks the stored values of the result, so a sum
  that fits the limit is not refused.
- The duplicate entries of one place add up, as the duplicates of SciPy
  add them up. The coordinates take only whole places inside the
  matrix. A negative or fractional place is a fault.
- The coordinates keep the duplicates of one place. The dense form and
  the compressed rows add them up, as SciPy adds them up.
- The biconjugate gradient solvers and `lsmr` check the norm of the
  right side before they start. A right side with a norm that is not
  finite is a fault. A
  right side with a norm that falls to zero is a fault too. A right
  side with a norm that grows past the largest number is a fault too.
  SciPy gives the zero solution in the last two cases, and the zero
  solution reads as a correct answer.
- The formulas of the hyperbolic functions take a power of the number.
  A power that reaches over the size of a number gives an infinity,
  as NumPy gives it. No fault comes from the value.
- The arc of the hyperbolic cosine takes the root of a square that a
  number below one makes negative. The root gives NaN, as NumPy gives
  it. No fault comes from the value.
- The quotient that rounds down divides by a zero divisor. The IEEE
  rules hold, so the result holds an infinity. A caller that needs a
  finite value checks the result.
- `mod` and `fmod` divide by a zero divisor. The IEEE rules hold, so
  the result holds NaN. A caller that needs a finite value checks the
  result.
- `exp2` of a number of great size gives an infinity, as NumPy gives
  it. `cbrt` of an infinity gives the infinity. No fault comes from
  the value.
- `nextafter` and `spacing` give the smallest number of the machine
  for a value of zero. No fault comes from the value.
- `mode` and `average` read the whole array. A large array takes a
  long time.
- The sets sort a row to search it. A row of many elements takes a
  long time, as the sort of the package takes.
- A NaN reduction never faults on a value. An array of only NaN gives
  a neutral value or NaN. A caller that needs a number checks the
  result with `is_nan` of the standard library.
- A bad axis, or a permutation that repeats an axis, is a fault. The
  check runs before any element moves.
- The sort of a large array takes a long time. A caller that sorts a
  large array waits.
- The elimination is not scaled. A matrix whose rows hold values of
  different sizes loses some digits.
- The QR algorithm with a shift is slow for a large matrix. It holds
  a small and a middle matrix well.
- The singular values come from the one-sided Jacobi method. The
  method holds a small and a middle matrix well.
- A position outside the array is a fault in `take` and in `put`. The
  check of `put` runs over every position before the first write.
  A bad index leaves the array as it was.
- A fault result holds an empty array. A caller that keeps reading
  after a fault reads an array of length zero, not a torn one.
- The package takes no text input and holds no parser. The attack
  surface is the shape and the values of the caller.
- The transform checks the count of the elements and the count of the
  parts before it allocates. A signal that reaches over
  `MAX_ELEMENTS` is a fault, so a bad length cannot grow memory
  without bound.
- `irfft` refuses parts whose count does not hold the length. The
  check runs before any element moves, so a bad length cannot write
  outside its arrays.
- The Bluestein method holds the work of the next power of two above
  twice the length. A caller that transforms a large signal of a
  length that is not a power of two holds that memory.
- The generator holds no hidden state outside the `Rng` of the
  caller. Two generators of the same seed give the same numbers. A
  seed that a hostile party can guess gives numbers it can guess.
- Every write checks its mask and its values before the first write.
  A mask of a different count is a fault. `place` and `putmask` hold
  no value outside their buffer. So a bad mask cannot write outside
  the array of the caller.
- `put_along_axis` checks every position before the first write. A
  position outside the axis is a fault, and the array stays as it
  was.
- `interp` checks that the points rise before it takes a value. The
  check runs once. A row of points that does not rise cannot make the
  search read outside the buffer.
- `pad_value` computes the count of the result before it allocates.
  A growth that reaches over `MAX_ELEMENTS` is a fault.
- A solver calls the function of the caller only inside the bracket
  that it holds. A function that changes its value between the calls
  can lead the method outside the root. A caller that holds such a
  function gives a wrong root in silence.
- The count of the turns bounds the work of a solver. A function that
  needs many turns holds a long time. A caller that sets the count
  low gets a fault instead of a wrong root.
- A function whose values are close near its smallest point holds no
  difference that the method can read. The method then stops at a
  distance of about the size of the root of the machine. SciPy stops
  at the same distance.
- The integration functions hold no adaptive method that calls the
  function of the caller at every point of a grid. A caller that
  needs a fine integral samples the function on a fine grid.
- `welch` takes the size of a part from the caller. A caller that
  wants the parts of SciPy gives the same size and the same count of
  parts. The parts of SciPy start at every half of the size too.
- `residue` checks its input before it computes. It faults on a
  coefficient that is not finite and on a negative `tol`. It also
  faults on an unknown rule and on more than 257 coefficients. SciPy lets a
  negative `tol` give an empty group and an error deep in the call.
  The checks also bound the work of the root finder.
- `residue_complex` checks its input before it computes. It faults on
  a coefficient that is not finite. It faults on a zero leading
  coefficient of the denominator. It faults on a real part and an
  imaginary part list of a different length. It faults on a negative
  `tol` and on an unknown rule. The checks bound the work of the root
  finder.
- `toms748` checks its input before it computes. It faults on a
  tolerance of zero or below. It also faults on a relative tolerance
  below zero and on a count of turns below one. A value of the
  function that is not finite is a fault. The count of steps of Newton
  type is held between one and a hundred.
- `secant` and `halley` check their input before they compute. They
  fault on a tolerance of zero or below and on a count of turns below
  one. The secant method also faults on two points that hold the same
  value. The method of Halley faults on a derivative of zero.
- `minimize_grad` checks its input before it computes. It faults on
  a start that holds no element and on a tolerance of zero or below.
  It also faults on a count of turns below one and on an unknown
  method. The bounds of a wrong count are a fault. So is a lower
  bound above the upper bound.
- `invres`, `invresz`, `residuez` and `unique_roots` check their input
  as `residue` does. They fault on a value that is not finite.
  They also fault on a bad `tol`, a bad rule and a count above the
  limit.
- `cheby1`, `cheby2` and `ellip` fault on a ripple outside zero to
  one million. They also fault on an order above 32. A large value
  cannot make the design overflow into infinite coefficients.
- `bessel` and `bessel_sos` fault on an order outside one to 32. They
  also fault on a cut outside zero to one and on an unknown norm.
  The search for the 3 dB point stops after a fixed count of turns.
- The spline methods of `interpn` fault on an axis of more than 512
  points. The dense solve of each line would take too long.
- `welch_ex` faults on a window list of the wrong size or with a value
  that is not finite. It faults on a transform length above 4194304 and
  on a signal with a value that is not finite.
- `fht`, `ifht` and `fhtoffset` fault on a `dln` of zero and on a value
  that is not finite. They fault on a bias that makes the transform
  singular. The log gamma function uses a reflection and a short
  shift. A large `mu` cannot make a loop run long.
- `qr_update` checks the length of the buffer of every operand against
  its shape before it reads a value. A shape that lies about its
  buffer is a fault, so a bad shape cannot read outside the buffer.
- `qr_update` checks the side of every factor against the limit before
  it copies. A side above 4096 is a fault, so a bad side cannot grow
  the memory without bound.
- `qr_update` takes a vector of no length and leaves the factors as
  they are. SciPy divides by the length of the vector and writes a
  division by zero to its error stream. The package gives the same
  factors and writes nothing.
- `qr_insert` and `qr_delete` check the length of the buffer of every
  operand against its shape before they read a value. A shape that
  lies about its buffer is a fault, so a bad shape cannot read outside
  the buffer.
- `qr_insert` and `qr_delete` check the side of every factor against
  the limit before they copy. A side above 4096 is a fault, so a bad
  side cannot grow the memory without bound.
- `qr_insert` and `qr_delete` fault on a value that is not finite.
  SciPy takes the same check in its default. A value that is not
  finite would spread through every factor.
- `rsf2csf` and `cdf2rdf` check the length of the buffer of every operand against its shape before they read a value. A shape that lies is a fault. They check the side against the limit, so a bad side cannot grow the memory.
- `rsf2csf` and `cdf2rdf` check the count of elements of every array and of every result against the limit. They run the check on the broadcast of the two batches before they build any index. Two valid batches can broadcast to far more slices than each holds. A bad pair of shapes cannot exhaust the memory.
- `rsf2csf` and `cdf2rdf` fault on a value that is not finite. SciPy takes the same check in its default. A value that is not finite would spread through every result.
- `bandwidth` and `bandwidth_stack` check the length of the buffer of their operand against its shape before they read a value. A shape that lies is a fault, so a bad shape cannot read outside the buffer. `bandwidth_stack` also checks the count of elements against the limit.
- `solve_continuous_are` and `solve_discrete_are` check every buffer against its shape before they read a value. They check the side against the limit. A shape that lies is a fault, and so is a value that is not finite.
- `pad_axes` checks the buffer of the array against its shape. It checks every count against the element limit before it forms a product. It checks every list length. A negative count and an unknown mode are faults. The mode `empty` fills zero, so no call returns uninitialized memory.
- `clarkson_woodruff_transform` checks the buffer against the shape. It checks the size of the sketch against the element limit. It checks every value for finiteness. All checks run before it draws a number.
- Both equations check the residual of the answer in the original equation. SciPy does not. A data set with no stabilizing solution gives a fault here, and a wrong matrix in SciPy.
- `qr_delete` checks the count of the removed lines against the
  size of the factors before it removes anything.
  A count above the size is a fault, so a bad count cannot read
  outside the buffer.
- `cossin` checks the length of the buffer of its operand against its
  shape before it reads a value. A shape that lies is a fault, so a bad
  shape cannot read outside the buffer.
- `cossin` checks the side of its operand against the limit of 512
  before it copies. A side above the limit is a fault, so a bad side
  cannot grow the memory without bound.
- `cossin` checks the count of elements of every result against the
  limit. A stack of many matrices cannot exhaust the memory.
- `cossin` faults on a value that is not finite. A value that is not
  finite would spread through every result.
- `cossin` checks that its operand is orthogonal. The fault follows
  when the largest entry of `X^T X - I` is above `1.0e-8`. SciPy
  checks nothing here and gives a wrong answer.
- `cossin_blocks` checks the sides of its four blocks before it joins
  them. Blocks that do not fit give a fault, so a bad side cannot read
  outside a buffer.
- `qz` checks the length of the buffer of each operand against its
  shape before it reads a value. A shape that lies is a fault, so a bad
  shape cannot read outside the buffer.
- `qz` checks the side of each operand against the limit of 512 before
  it copies. A side above the limit is a fault, so a bad side cannot
  grow the memory without bound.
- `qz` checks the count of matrices of the broadcast against the
  element limit before it allocates. Two small batches cannot give a
  result of unbounded size.
- `qz` faults on a value that is not finite. A NaN would spread through
  the iteration, and the iteration would then never meet its test.
- `qz` counts the turns of the iteration. A pair that does not settle
  gives a fault of kind `F_LIMIT`. A hostile pair cannot hold the call
  in a loop.
- `qz` checks every result for values that are not finite. A pair with
  entries near the limit of the number format gives a fault. No result
  holds a NaN without a fault.
- `ordqz` checks the name of the rule before it reads a value.
- `qz_c` checks the length of the buffer of each operand against twice
  the count of its shape. A short buffer is a fault, so a bad shape
  cannot read outside the buffer.
- `ordqz_by` calls the block of the caller once for each diagonal
  block. A block that changes its answer cannot hold the call in a
  loop.
- `qz` and `qz_c` scale the pair by a power of two before the
  iteration. The scale keeps the values of the iteration near one. A
  pair with entries near the limit of the number format cannot
  overflow the work.
- `cdot`, `cvdot`, `ccholesky`, `cfftn` and `cifftn` check the buffer
  of each operand against twice the count of its shape. A short buffer
  is a fault, so a bad shape cannot read outside the buffer. A real
  array gives that fault too. They check the count of the entries of
  the shape against the count of dimensions before they read an
  entry. `ccholesky` and `eigvals_complex` do the same. `eigvals_complex`
  checks its buffer of one value per element before it reads a value.
- `cqr` and `csvd` check the buffer, the count of the entries of the
  shape, the side and every value. A side above 512 is a fault. A value
  that is not finite is a fault. NumPy lets a NaN into the factors. The
  package does not.
- `cqr` and `csvd` scale an operand whose largest part lies outside
  1e-100 to 1e100. The squares of the parts then cannot overflow or
  underflow. A matrix near 1e300 gives finite factors, not infinity.
- `ceig`, `ceigvals`, `ceigh` and `ceigvalsh` check the buffer, the
  shape, the side and every value. A side above 256 is a fault. A value
  that is not finite is a fault. They scale an operand whose largest
  part lies outside 1e-100 to 1e100 and scale the eigenvalues back.
  Both iterations have a fixed cap, so hostile input cannot loop
  forever.
- The eight pair calls check both buffers, both shapes, the side and
  every value. A side above 256 is a fault. Two sides that differ are a
  fault. A value that is not finite is a fault. The back substitution
  scales both triangles first and replaces a tiny denominator, so no
  division by zero occurs. The iterations have fixed caps.
- `ccholesky` checks the side of the matrix before it reads a value.
  A side above 1024 is a fault. It checks every element for a value
  that is not finite. A matrix that is not positive definite gives a
  fault. NumPy lets a NaN into the factor. The package does not.
- `cdot` checks the count of the elements of the result before it
  builds it. The check divides the limit by the other side, so no
  product overflows. A result above `MAX_ELEMENTS` is a fault. A
  complex array holds two values for each element. The largest result
  holds 536870912 values, which is 4 GiB.
- The filters of N dimensions check the shape and the buffer of the array
  before they read a value. They check the origin, the size, the axis
  `i64` is a fault, so no index wraps. The count of values times the
  count of weights has a limit. So a hostile size or sigma cannot hold
  the process for hours.
- `gaussian_filter1d` gives a fault for a sigma whose square underflows.
  SciPy builds weights of NaN for such a sigma and returns NaN without an
  error. A sigma or a truncation that is not finite is a fault.
- The rank filters sort a window with a heap sort, so no data gives a
  slow case. The cost limit holds for a footprint, for a size and for a
  size that lies by its product. A footprint with no nonzero place and a
  rank outside the window are faults.
- The transforms check the order, the mode, the shape and the cost before
  any value moves. A coordinate that is not finite, or that lies beyond
  4.0e15, is a fault, so no index wraps. A zoom that gives an axis with
  no place, or a zoom above 1.0e15, is a fault. An angle of `rotate` beyond
  1.0e14 degrees or not finite is a fault. SciPy computes values from such
  inputs that carry no meaning. The padded array of the modes `nearest` and
  `grid-constant` is checked against `MAX_ELEMENTS` first.
- The calls with a block check the array, the footprint, the origin and
  the cost before the first block runs. A block that does not stop is the
  risk of the caller. The package cannot interrupt it.
- The morphology calls check the structure, the mask and the origin before
  the first value moves. The cost of the rounds has a limit. So a hostile
  size or a large count of iterations cannot hold the process for hours.
- `label` walks with a stack of positions. The stack cannot grow above the
  count of values of the array, so no input grows memory without bound.
- The measurements check the shape of the labels against the shape of the
  data before they read a value. A shape that lies is a fault, so a bad
  shape cannot read outside the buffer.
- The measurements fault on a label that is not a whole number, or that
  lies beyond 9.0e18. A NaN would spread through every result.
- The measurements check the count of index entries against the element
  limit. They check the count of bins the same way. Both checks run before
  they build a result. The check divides the limit by the other count. No
  product overflows.
- The measurements find the place of a value with a heap sort of the
  index list. They use a binary search of the labels. No data gives a
  slow case.
- The grey calls check the array, the footprint, the structure and the
  origin before the first value moves. A structure of another shape than the
  footprint is a fault. An origin beyond one billion is a fault, so the
  flip of the origin cannot overflow. The cost of a call has a limit, so a
  hostile footprint cannot hold the process for hours.
- `distance_transform_cdt_st` reads the structure by value. SciPy reads the
  buffer of an integer or float structure as bytes, so it gives -1 or wrong
  distances there. Only a boolean structure works in SciPy.
- The distance calls check the shape, the buffer, the structure and the
  sampling before any value moves. They check the count of pairs of the
  brute force search against a limit. The size of the result is checked
  against `MAX_ELEMENTS` with a quotient, so no product overflows.
- `value_indices` faults on a value that is not a whole number. It sorts
  with a heap sort, so no data gives a slow case.
- The distance transform faults when the array holds no background value.
  SciPy returns values without meaning there. A sampling above `1.0e100`
  on an axis is a fault, so no squared distance overflows to an infinity.
- `geometric_transform` checks the shape and the count of the coordinates
  before the first block runs. The check divides the element limit by the
  count of the axes, so no product overflows. A block that writes a
  coordinate outside the array is read with the border rule, as SciPy does.
- `generic_filter1d` checks the size, the axis, the origin, the mode and
  the cost before the first block runs. The line of a filter never grows
  above the length that the size names. So a hostile size cannot hold the
  process for hours, and a block cannot read outside the line.
- `vectorized_filter` checks the window, the origin, the mode and the
  count of the values before the first block runs. The check of the
  windowed array divides the limit of the elements by the count of the
  window. No product overflows. The border of the input is checked the
  same way. A block reads the windowed array only.

## License

MIT
