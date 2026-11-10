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
import scientific from "rekanalar/goose-scientific@v0.14.0";
```

## API

| Area | Functions |
|---|---|
| Creation | `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `diag`, `from_rows`, `from_values`, `clone` |
| Shape | `reshape`, `ravel`, `transpose`, `transpose_perm`, `squeeze`, `expand_dims`, `concatenate`, `stack`, `broadcast_shapes`, `index` |
| Elementwise | `add`, `sub`, `mul`, `div`, `power`, `maximum`, `minimum`, `neg`, `abs`, `sqrt`, `exp`, `log`, `log2`, `log10`, `scale`, `clip`, `sign`, `floor`, `ceil`, `round`, `trunc`, `square` |
| Logic | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `logical_and`, `logical_or`, `logical_xor`, `logical_not`, `isnan`, `isinf`, `isfinite`, `all`, `any`, `count_nonzero`, `where` |
| Reductions | `sum`, `prod`, `min`, `max`, `mean`, `argmin`, `argmax` |
| Reductions along an axis | `sum_axis`, `prod_axis`, `min_axis`, `max_axis`, `mean_axis`, `var_axis`, `std_axis` |
| Running and spread | `cumsum`, `cumprod`, `diff`, `flip`, `variance`, `std`, `ptp` |
| Indexing | `slice_axis`, `take`, `put`, `nonzero`, `mask`, `take_along_axis`, `repeat`, `tile`, `flatten`, `partition` |
| Sorting | `sort`, `argsort`, `sort_axis`, `unique`, `unique_sorted`, `roll` |
| Search | `searchsorted`, `histogram`, `histogram_edges`, `bincount`, `digitize` |
| Order statistics | `median`, `percentile` |
| Products | `matmul`, `dot`, `inner`, `outer`, `kron`, `trace`, `norm` |
| Solving | `solve`, `inv`, `det`, `slogdet`, `matrix_power` |
| Decompositions | `cholesky`, `qr` |
| Eigenvalues | `eigvals`, `eig`, `eigvalsh`, `eigh` |
| Singular values | `svd`, `svdvals`, `lstsq`, `matrix_rank`, `cond` |
| Polynomials | `polyval`, `polyfit`, `roots` |
| Transforms | `fft`, `ifft`, `fft2`, `fftfreq`, `fftshift`, `rfft`, `irfft` |
| Sampling | `default_rng`, `uniform`, `random_uniform`, `integers`, `normal`, `exponential`, `poisson`, `choice`, `permutation`, `shuffled`, `shuffle_rows` |
| Shapes of another array | `empty`, `full_like`, `zeros_like`, `ones_like`, `trim_zeros`, `insert` |
| Writes | `putmask`, `place`, `put_along_axis` |
| Cuts | `split`, `array_split`, `hsplit`, `vsplit`, `part` |
| Triangles | `tril`, `triu` |
| Between the points | `interp`, `gradient`, `pad_value` |
| Root finding | `brentq`, `brenth`, `newton`, `fsolve` |
| Minimization | `minimize_scalar`, `minimize`, `least_squares` |
| Integration | `trapezoid`, `simpson`, `cumulative_trapezoid`, `romberg` |
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

- The shape holds at most 8 dimensions.
- One array holds at most `MAX_ELEMENTS` elements, 268435456.
- Every dimension holds at least one element. The release refuses an
  array of length zero.
- `transpose` swaps the two dimensions of a matrix. `transpose_perm`
  takes any permutation of the axes.
- An axis reduction drops the reduced axis from the shape. It has no
  `keepdims` option. Use `expand_dims` on the result instead.
- `argmin` and `argmax` give the flat position of the first extreme.
  They take no axis.
- The elementwise math holds the functions of this table. The rest of
  the scalar functions of NumPy stay out for now.
- A comparison gives an array of 1.0 and 0.0. The array holds one
  element type, so a truth value has no type of its own.
- `round` sends a half to the even neighbour, as NumPy does. `floor`,
  `ceil` and `trunc` send a value to a whole number.
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
- A matrix is an array with two dimensions. The package holds one
  array type.
- `dot` of two rows gives a number. It is held in an array of no
  dimension.
- `dot` of a row and a matrix takes the last axis of the row.
  It also takes the first axis of the matrix.
- `dot` of a matrix and a row takes the first axis of the matrix.
  It also takes the last axis of the row.
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
- The package holds real numbers only. A matrix with a complex pair of
  eigenvalues is a fault for `eig` and `eigvals`.
- The eigenvectors of a matrix that is not symmetric come from an
  iteration. A matrix whose eigenvalues hold no vector is a fault.
- `svd` gives the full form, as NumPy gives it under its default.
  The left vectors hold as many rows and columns as the matrix has
  rows. The right vectors hold the transpose of the vectors.
- The singular values hold only the count of the smaller dimension.
- `lstsq` uses the singular values. A singular value below the
  tolerance gives a zero in the solution, as NumPy gives it.
- `matrix_rank` and `cond` use the tolerance of the largest singular
  value, as NumPy does.
- The coefficients of a polynomial come from the highest power to the
  lowest, as NumPy holds them.
- `roots` uses the eigenvalues of the companion matrix. A root that
  holds a complex number is a fault.
- The transform works on two arrays: one holds the real parts and one
  holds the imaginary parts. The package holds no complex type.
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
- `fftshift` moves the zero frequency to the middle. It holds a flat
  array of frequencies, so it takes no axis.
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
- `minimize` finds the smallest point of a function of several
  variables. The method of Nelder and Mead holds a simplex. A start
  far from the point of the smallest value can end at a local point.
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
- `romberg` takes the extrapolation of Romberg. The count of the
  parts of the grid holds a power of two, as the method of the
  references holds. A count that holds no power of two is a fault.

## Security

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

## License

MIT
