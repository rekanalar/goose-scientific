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
import scientific from "rekanalar/goose-scientific@v0.4.0";
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
- A position outside the array is a fault in `take` and in `put`. The
  check of `put` runs over every position before the first write.
  A bad index leaves the array as it was.
- A fault result holds an empty array. A caller that keeps reading
  after a fault reads an array of length zero, not a torn one.
- The package takes no text input and holds no parser. The attack
  surface is the shape and the values of the caller.

## License

MIT
