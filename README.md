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
import scientific from "rekanalar/goose-scientific@v0.83.0";
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
| Transforms | `fft`, `ifft`, `fft2`, `fft2c`, `ifft2`, `fftn`, `ifftn`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, `rfft`, `irfft` |
| Sampling | `default_rng`, `uniform`, `random_uniform`, `integers`, `normal`, `exponential`, `poisson`, `choice`, `permutation`, `shuffled`, `shuffle_rows` |
| Names of NumPy | `seed`, `shuffle`, `random_sample`, `rand`, `randn`, `randint`, `bytes`, `get_state`, `set_state` |
| Shapes of another array | `empty`, `full_like`, `zeros_like`, `ones_like`, `trim_zeros`, `insert` |
| Writes | `putmask`, `place`, `put_along_axis` |
| Cuts | `split`, `array_split`, `hsplit`, `vsplit`, `part` |
| Triangles | `tril`, `triu` |
| Between the points | `interp`, `gradient`, `pad_value` |
| Root finding | `brentq`, `brenth`, `newton`, `fsolve`, `root_scalar` |
| Minimization | `minimize_scalar`, `minimize`, `differential_evolution`, `least_squares`, `curve_fit` |
| Integration | `trapezoid`, `simpson`, `cumulative_trapezoid`, `cumulative_simpson`, `cumulative_simpson_dx`, `romberg` |
| Interpolation | `interp`, `spline`, `spline_value`, `spline_values` |
| Signals | `convolve`, `correlate`, `detrend`, `welch` |
| Filters | `butter`, `lfilter`, `lfilter_zi`, `filtfilt`, `sosfilt`, `hilbert` |
| Covariance and NaN statistics | `cov`, `cov_weights`, `cov_with`, `corrcoef`, `corrcoef_with`, `nanmedian`, `nanpercentile`, `nanquantile`, `nanargmax`, `nanargmin`, `nancumsum`, `nancumprod`, `cumulative_sum`, `cumulative_prod`, `ediff1d`, `histogram_bin_edges` |
| The rest of the elementwise math | `fmax`, `fmin`, `heaviside`, `logaddexp`, `logaddexp2`, `ldexp`, `frexp`, `modf`, `sinc`, `i0`, `gcd`, `lcm`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `invert`, `left_shift`, `right_shift`, `bitwise_count`, `unwrap`, and the names `float_power`, `pow`, `remainder`, `radians`, `degrees`, `acos`, `asin`, `atan`, `atan2`, `acosh`, `asinh`, `atanh`, `bitwise_not`, `bitwise_invert`, `bitwise_left_shift`, `bitwise_right_shift`, `concat`, `amax`, `amin`, `ndim` |
| Assembly of shapes | `swapaxes`, `permute_dims`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `append`, `append_axis`, `delete`, `delete_axis`, `resize`, `pad_modes`, `diagonal`, `fill_diagonal`, `tri`, `tril_indices`, `triu_indices`, `diag_indices`, `indices`, `ix_`, `unravel_index`, `ravel_multi_index`, `unstack`, `unstack_part`, `block_grid` |
| Selection and order | `lexsort`, `argpartition`, `compress`, `compress_axis`, `extract`, `choose`, `piecewise`, `setxor1d`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values` |
| Products, windows and old polynomial functions | `vdot`, `vecdot`, `matvec`, `vecmat`, `vander`, `tensordot`, `tensordot_n`, `einsum`, `einsum1`, `hanning`, `hamming`, `blackman`, `bartlett`, `kaiser`, `get_window`, `poly`, `poly_cplx`, `polyadd`, `polysub`, `polymul`, `polydiv`, `polyder`, `polyint` |
| Factors, special matrices and the exponential | `lu_factor`, `lu`, `lu_solve`, `solve_triangular`, `solve_banded`, `block_diag`, `toeplitz`, `hankel`, `circulant`, `null_space`, `orth`, `expm` |
| Cosine, sine and real transforms | `dct`, `idct`, `dst`, `idst`, `dctn`, `idctn`, `dstn`, `idstn`, `hfft`, `ihfft`, `next_fast_len`, `rfft2`, `rfftn`, `irfft2`, `irfftn`, `hfft2`, `ihfft2`, `hfftn`, `ihfftn` |
| Distances, nearest points and hulls | `cdist`, `pdist`, `squareform`, `distance`, `euclidean`, `sqeuclidean`, `cityblock`, `chebyshev`, `minkowski`, `cosine_distance`, `correlation`, `hamming_distance`, `jaccard`, `canberra`, `braycurtis`, `seuclidean`, `mahalanobis`, `dice`, `russellrao`, `rogerstanimoto`, `sokalsneath`, `sokalmichener`, `yule`, `kdtree`, `kdtree_query`, `kdtree_query_ball_point`, `convex_hull` |
| Cubic splines and `interp1d` | `cubic_spline`, `pchip`, `akima`, `spline_eval`, `spline_integrate`, `interp1d` |
| Adaptive quadrature and ODEs | `quad`, `fixed_quad`, `dblquad`, `tplquad`, `nquad`, `quad_vec`, `solve_ivp`, `odeint` |
| Programs of optimization | `linprog`, `nnls`, `linear_sum_assignment`, `approx_fprime`, `bisect`, `ridder` |
| Signal design and analysis | `firwin`, `firwin2`, `freqz`, `sosfreqz`, `sosfreqz_at`, `resample_poly`, `decimate`, `lfiltic`, `fftconvolve`, `medfilt`, `savgol_coeffs`, `savgol_filter`, `find_peaks`, `peak_prominences`, `periodogram`, `spectrogram`, `welch_psd`, `stft`, `istft`, `resample` |
| Sparse matrices of compressed rows | `Csr`, `csr_from_dense`, `csr_from_triplets`, `csr_to_dense`, `csr_eye`, `csr_diags`, `csr_transpose`, `csr_get`, `csr_scale`, `csr_matvec`, `csr_add`, `csr_sub`, `csr_matmul`, `csr_diagonal`, `csr_sum`, `csr_kron`, `csr_hstack`, `csr_vstack`, `csr_block_diag`, `csr_tril`, `csr_triu`, `csr_multiply`, `csr_slice`, `spsolve`, `cg`, `bicgstab`, `gmres`, `lsqr`, `minres`, `eigsh` |
| Functions of matrices | `sqrtm`, `logm`, `sinm`, `cosm`, `tanm`, `sinhm`, `coshm`, `tanhm`, `signm`, `fractional_matrix_power`, `polar`, `cho_factor`, `cho_solve`, `funm`, `solve_sylvester`, `solve_continuous_lyapunov`, `solve_discrete_lyapunov`, `eig_banded`, `eigvals_banded` |
| Complex arrays | `CArr`, `complex_array`, `complex_real`, `complex_from_values`, `real`, `imag`, `angle`, `conj`, `iscomplex`, `isreal`, `cabs`, `cadd`, `csub`, `cmul`, `cdiv`, `cpower`, `cneg`, `cexp`, `clog`, `csqrt`, `csin`, `ccos`, `csinh`, `ccosh`, `ctranspose`, `adjoint`, `cmatmul`, `csolve`, `cinv`, `cdet`, `cfft`, `cifft`, `eigvals_complex` |
| Sections of a filter | `butter_sos`, `cheby1_sos`, `cheby2_sos`, `zpk2sos`, `tf2sos`, `sosfilt_zi`, `sosfiltfilt` |
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
  and gives the real and the imaginary part. The complex matrices have no `eig`, `svd`, `qr` or
  `cholesky`, and `eigvals_complex` gives no eigenvectors. `cfft` and
  `cifft` take one dimension.
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
- `lsqr` takes a matrix of any shape and solves the least squares
  problem. It gives the solution of the least norm when the unknowns
  outnumber the equations. It has no `damp`, no `x0` and no `atol`.
  `minres` takes a symmetric matrix and does not check the symmetry.
  It has no `shift`, no `x0` and no preconditioner. The path may differ
  from SciPy, so only the solution matches.
- `eigsh` takes the choices `LA`, `SA`, `LM` and `SM` and gives the
  values in rising order. It uses the Lanczos method with a full
  reorthogonalization, and a space above 256 vectors is a fault. It has
  no `M`, no `sigma`, no `v0` and no `ncv`. The vectors may differ in
  sign from SciPy. Give it a symmetric matrix.
- `csr_hstack` and `csr_vstack` join two matrices, not a list. Call them
  again for more. `csr_kron` and `csr_block_diag` take two matrices.
- `csr_slice` takes the row and column ranges with the end excluded and
  no step. `csr_tril` and `csr_triu` take the diagonal `k`.
  `csr_multiply` takes two matrices of one shape and does not broadcast.
- The sparse type holds compressed rows only, in the type `Csr`. The
  types `csc_array`, `coo_array`, `lil_array`, `dok_array`, `dia_array`
  and `bsr_array` stay out. Build a matrix from a list of entries with
  `csr_from_triplets`, which adds the entries of one place. A sparse
  result keeps the stored zeros that a sum or a product makes.
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

- `firwin2` takes the window as a name, and `none` gives no window.
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
  come last in the call, after the tolerances. Every limit is
  finite. A limit that is not finite is a fault. The error is the
  sum of the errors of the inner integrals.

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
- `solve_ivp` takes the methods `RK45` and `RK23`. The steps equal the
  steps of SciPy. The result holds one row per time, where SciPy
  holds one column. The arguments `t_eval`, `dense_output`, `events`
  and `first_step` stay out, and so do the methods `DOP853`, `Radau`,
  `BDF` and `LSODA`. A stiff system needs many steps and may reach the
  limit of one million steps.
- `cubic_spline` takes the conditions `not-a-knot`, `natural` and
  `clamped`. `clamped` gives zero slope at both ends. The condition
  `periodic` and given end values stay out. `spline_eval` takes the
  derivative orders zero to three. A point outside the knots takes the
  end cubic for `cubic_spline` and `pchip`. A spline of `akima` gives
  NaN there, as SciPy does.
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
- `Delaunay`, the hull of three dimensions and the arguments `eps`
  and `workers` stay out. The metrics of true and false counts read
  a nonzero value as true.
- `dct`, `idct`, `dst` and `idst` take one row and the types one to
  four. They take the norms `backward`, `ortho` and `forward`. Type one
  of the cosine needs two points.
- `dctn`, `idctn`, `dstn` and `idstn` take an array and a list of axes. They run each axis in turn. An
  empty list gives the array back. No list means every axis. The
  `axis` argument of the one-row functions stays out.
- `hfft` and `ihfft` take and give the parts as two rows. `rfft2`,
  `rfftn`, `irfft2` and `irfftn` take the parts as two arrays. `irfftn`
  takes the length of the last axis, because it cannot tell an even
  length from an odd one.
- `hfft2` and `hfftn` take the parts as two arrays and the length of
  the last axis. `ihfft2` and `ihfftn` take a real array and give two
  arrays. They hold no `s` list for the other axes, no `axes` and no
  `norm`. The default norm is `backward`.
- `next_fast_len` follows SciPy. A complex transform takes the factors
  up to 11. A real one takes 2, 3 and 5. The Fast Hankel transform
  `fht` stays out.
- `lu_factor` and `lu` follow LAPACK. The row i swaps with the row
  `piv[i]`. A singular matrix gives no fault there. `lu_solve` gives a
  fault when it meets a zero on the diagonal. `solve` takes a matrix on the right side.
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
- `vecdot` takes two arrays of one shape and does not broadcast.
  `matvec` and `vecmat` take one matrix and one row, with no batch axes.
  `tensordot_n` is `tensordot` with the count form of `axes`.
- `einsum` takes two operands and `einsum1` takes one. The rule holds
  letters, a comma and an arrow, but no ellipsis. It holds at most 8
  distinct letters. A length of one broadcasts.
- `get_window` knows 18 names. These are boxcar, triang, bartlett,
  hann, hamming, blackman, blackmanharris, nuttall, flattop, bohman,
  barthann, cosine, lanczos, parzen, kaiser, gaussian, tukey,
  exponential. The windows chebwin, dpss, taylor, general_cosine and
  the others stay out.
  `kaiser` takes beta up to 700.
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
  for every axis; NumPy also takes one pair per axis. A count above the
  width of an axis is a fault.
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
  gives no numerator and denominator form, no `fs` argument and no
  analog filter. The order stays at 32 or below.
- `cheby2_sos` designs a Chebyshev filter of type II as sections. The
  cuts are the edges of the stop band. It has the same limits as
  `cheby1_sos`.
- `lfilter` takes no initial state and no axis. `filtfilt` takes the
  odd extension of SciPy. The pad holds three times the coefficient
  count. No other pad option exists.
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
  values and the points. The methods are `linear` and `nearest`. A point
  outside the grid is a fault, unless the caller turns the check off
  and gives a fill value. A grid that does not rise strictly is a
  fault.
- `interpn` has no method of `cubic` and no extrapolation. It takes at
  most eight axes.
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
  matrix only. `fft2c` takes both parts. The transforms take no `s`
  option for padding or cropping, and no `axes` option.
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
- `minimize` finds the smallest point of a function of several
  variables. The method of Nelder and Mead holds a simplex. A start
  far from the point of the smallest value can end at a local point.
- `curve_fit` takes the model as a block `{ x, p => ... }`, the data
  and a start. It gives the parameters and the covariance. The
  covariance uses the scale of SciPy for the default setting. It has
  no `sigma`, no `bounds` and no `method`. Parameters match SciPy to
  about five digits.
- `least_squares` fits a function to values. The method of Levenberg
  and Marquardt takes the matrix of the derivatives by a small
  difference. So the fit holds fewer digits than a fit that takes the
  derivatives from the caller.
- The integration functions take the values of a signal and a
  spacing. A caller that needs the integral of a function samples the
  function first.
- `trapezoid` takes the rule of the trapezoid. `simpson` takes the
  rule of Simpson. An even count of parts takes the trapezoid on the
  last part, as NumPy does.
- `cumulative_trapezoid` takes the running total. The result holds
  one value fewer than the signal, as NumPy does.
- `cumulative_simpson` takes the signal and the places and gives the
  integral up to every point. The intervals may differ in width. The
  rule of Simpson of Cartwright gives each of them. A signal of two
  values falls back to the trapezoid. The result holds one value fewer
  than the signal.
- `cumulative_simpson_dx` takes one spacing for every interval. Neither
  function takes an axis, a start value or a shape other than one
  dimension.
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

## License

MIT
