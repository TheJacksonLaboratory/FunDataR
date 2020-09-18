# Fundamentals of computational data analysis using R
## A primer on programming using R: lesson 3

### Index

- [Type tests](#type-tests)
- [Type conversions](#type-conversions)
- [Numeric limits](#numeric-limits)
- [Conditional branching](#conditional-branching)
- [User defined functions](#user-defined-functions)
- [Data import and export](#data-import-and-export)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Type tests

[Return to index](#index)

---

### Type conversions

[Return to index](#index)

---

### Numeric limits

For representations of numbers, we often have to worry about things like exceeding the
  representable range as well as rounding errors. There are also some special numeric
  values that require separate treatment.

R comes in 32-bit and 64-bit versions. However, both versions on my system use 4-byte 
  integers and double-precision floating point representations of numeric types. The 
  main difference between these R versions is the size of 'pointers', which is 4-bytes
  on my 32-bit version and 8-bytes on my 64-bit versions. This implies that the 64-bit
  version can accommodate much larger data structures (practically unlimited), while 
  the 32-bit version is limited to vectors of only a few billion (!) elements. In order
  to see the particulars of number representation, type `.Machine` at the `>` prompt
  and press `<ENTER>`. One thing to note is that the value of .Machine$integer.max is
  only around 2 billion. However, using numerics (represented by double-precision 
  floating point numbers with .Machine$double.digits (53) 'fraction' bits and 
  .Machine$double.exponent (11) 'exponent' bits, integers can be exactly represented
  (not rounded) up to 9,007,199,254,740,992. Beyond this number, at best, only every 
  other integer can be expressed. The point is that you can freely use the
  numeric type to represent integers as long as you initialize with a whole number
  only add, subtract or multiply by other whole numbers, and avoid division. For
  instance, numerics are perfectly suitable for counting and indexing.

Another issue to consider is that numeric calculations often result in some rounding
  that can make answers differ slightly from their theoretical values. We've
  already seen this once in a previous lesson when running `solve()` on a matrix.
  Therefore, it is often helpful or critical to distinguish tests of exact equality
  (what you've seen thus far) from tests of approximate equality. 

```
(x <- c(1, 2, 3, 1, 2, 1, 2, 3, 1))
(x <- matrix(x, ncol=3))
(y <- x %*% solve(x))             ## note rounding issue (should be identity)
(i <- diag(3))                    ## make an identity matrix

identical(y, i)                   ## is y identical to its theoretical value?
y == i
y[y != i]

all.equal(y, i)                   ## test for approximate equality
isTRUE(T)                         ## is.logical(x) && length(x) == 1 && !is.na(x) && x
isTRUE(all.equal(y, i))           ## the 'safe' way to test for equality

```

[Return to index](#index)

---

### Conditional branching

[Return to index](#index)

---

### User defined functions

[Return to index](#index)

---

### Data import and export

[Return to index](#index)

---

## FIN!

