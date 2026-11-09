# scientific

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
import scientific from "rekanalar/goose-scientific@v0.1.0";
```

## API

| Area | Functions |
|---|---|
| Creation | `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `diag`, `from_rows`, `from_values`, `clone` |
| Shape | `reshape`, `ravel`, `transpose`, `squeeze`, `expand_dims`, `concatenate`, `stack`, `broadcast_shapes`, `index` |
| Elementwise | `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `neg`, `abs`, `sqrt`, `exp`, `log`, `scale`, `clip` |
| Reductions | `sum`, `prod`, `min`, `max`, `mean` |
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
- `transpose` swaps the two dimensions of a matrix. The general
  permutation of N dimensions stays out for now.
- The reductions hold the whole array. The reductions along one axis
  stay out for now.
- The elementwise math holds the functions of this table. The rest of
  the scalar functions of NumPy stay out for now.

## Security

- Every shape passes its checks before any element is built. A shape
  with a dimension below one is a fault, so no length lies about its
  buffer.
- `arange` computes its count before it allocates. A range that
  reaches over `MAX_ELEMENTS` is a fault, so a bad step cannot grow
  memory without bound.
- The IEEE rules hold for the values. A division by zero gives an
  infinity. A log of a number below zero gives NaN. No fault comes
  from a value, only from a shape, a range or a limit. Check `is_nan`
  of the standard library when the input is hostile.
- A fault result holds an empty array. A caller that keeps reading
  after a fault reads an array of length zero, not a torn one.
- The package takes no text input and holds no parser. The attack
  surface is the shape and the values of the caller.

## License

MIT
