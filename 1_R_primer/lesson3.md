# Fundamentals of computational data analysis using R
## A primer on programming using R: lesson 3
#### Contact: mitch.kostich@jax.org

---

### Index

- [Type tests](#type-tests)
- [Type conversions](#type-conversions)
- [Numeric limits](#numeric-limits)
- [Conditional execution](#conditional-execution)
- [User defined functions](#user-defined-functions)
- [Data import and export](#data-import-and-export)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Type tests

You will often run into a vector and need to determine its type. You've already seen
  how to do this with the `class()` function. If you need to check for a particular
  type (say 'logical') you could try to use the output of the `class()` command,
  but there is an easier way. You can use the `is.<type>()` command, where `<type>` 
  is the type you are testing for:

```
(x <- 1 : 10)
class(x)
is.integer(x)                     ## is x integer?
is.numeric(x)                     ## is x numeric?
is.logical(x)                     ## is x logical?
is.character(x)                   ## is x character?

(x <- seq(from=0, to=10, by=1))   ## should be same as 1:10, right?
class(x)                          ## surprise!
is.integer(x)
is.numeric(x)
is.logical(x)
is.character(x)

(x <- x >= 5)
class(x)
is.integer(x)
is.numeric(x)
is.logical(x)
is.character(x)

(x <- c('the', 'quick', 'brown', 'fox'))
class(x)
is.integer(x)
is.numeric(x)
is.logical(x)
is.character(x)

```

In addition to these conventional types, we have the missing value
  related types `NA` and `NULL`. The `NA` value is used as a placeholder
  for missing values. It can be included as an element in any kind of
  vector. By contrast, the `NULL` value is used as a placeholder for a 
  missing object (rather than value). It is normally not used as an 
  element of vector, but instead to indicate that an entire vector or 
  other object is missing. So it can be assigned as an element of a list.

One can detect `NA`s using the `is.na()` function, which returns a value 
  for every element in a vector (since any individual element can be set to `NA`). 
  Similarly, `NULL` values are detected using the `is.null()` function, which 
  returns a single value for any object you feed to it.

```
(x <- NA)                           ## vector of length 1
class(x)                            ## NA defaults to logical (most 'convertible' class)
length(x)                           ## length 1 (compare to NULL)
attributes(x)
is.na(x)                            ## a logical vector of the same length as x
is.null(x)                          ## a logical vector always of length 1

(x <- NULL)
class(x)
attributes(x)
length(x)                           ## compare to length(NA)
is.na(x)                            ## a logical vector of the same length as x; length(x) == 0
is.null(x)                          ## a logical vector always of length 1

(x <- c(1, NA, 3, NA, 5))           ## NA is placeholder in vector for missing values
class(x)                            ## NA 'promoted' to numeric (see type conversion section)
length(x)                           ## each NA counts for 1
attributes(x)
is.na(x)                            ## a logical vector of the same length as x
is.null(x)                          ## a logical vector always of length 1

(x <- c(1, NULL, 3, NULL, 5))       ## NULL not a missing value indicator; is a 'nothingness' indicator
class(x) 
length(x)                           ## NULLs gone, because they were 'nothing'
attributes(x)
is.na(x)
is.null(x)

rm(x)                               ## undefine x: x no longer exists
x                                   ## not the same as setting to NULL ...
class(x)
length(x)
attributes(x)
is.na(x)
is.null(x)

```

[Return to index](#index)

---

### Type conversions

You may have noticed that you can combine integer and numeric types successfully, but 
  what is the resulting type? Let's take a look:

```
x <- 3L                      ## integer
y <- 4                       ## numeric
class(x)
class(y)
(z <- x + y)
class(z)

```

So adding a numeric and integer results in the correct numeric value. What 
  about other combinations of data types? A simple way to explore this is with the
  `c()` concatenation function, which will take many different data types as arguments
  but always returns a vector with a uniform type, converting other input types into
  the whatever the final type happens to be:

```
(x <- c(F, 1L))
class(x)

(x <- c(T, 2L, 3))
class(x)

(x <- c(F, 1L, 2, "3"))
class(x)

```

So the direction of auto-conversion is `logical -> integer -> numeric -> character`. 
  If any type is combined with any second type that lies to the right of the first type 
  in the series, the type on the left in the series is converted to the type that lies 
  further to the right. The conversions are pretty sensible for the most part, as was 
  demo'd in the previous code block. 

But what about converting types in the opposite direction? Turns out that is possible 
  too, using the `as.<type>()` series of functions:

```
str(x <- list(logical=T, integer=1L, numeric=1, character="1"))
sapply(x, as.logical)             ## numeric/integer 0 is FALSE; else TRUE 
sapply(x, as.integer)             ## logical TRUE -> 1L, FALSE -> 0L
sapply(x, as.numeric)             ## logical TRUE -> 1, FALSE -> 0
sapply(x, as.character)

str(x <- list(logical=F, integer=0L, numeric=0, character="0"))
sapply(x, as.logical)             ## numeric/integer 0 is FALSE; else TRUE
sapply(x, as.integer)             ## logical TRUE -> 1L, FALSE -> 0L
sapply(x, as.numeric)             ## logical TRUE -> 1, FALSE -> 0
sapply(x, as.character)

str(x <- list(logical=F, integer=-3L, numeric=-0.1, character="-1.6e5"))
sapply(x, as.logical)             ## numeric/integer 0 is FALSE; else TRUE
sapply(x, as.integer)             ## logical TRUE -> 1L, FALSE -> 0L
sapply(x, as.numeric)             ## logical TRUE -> 1, FALSE -> 0
sapply(x, as.character)

```

So a character representation of a number can be converted into an integer or
  numeric, and an integer or numeric can be converted into a logical, but 
  a character representation of a number cannot be directly converted into
  a logical! Instead, you can always indirectly convert a character representation 
  of a numeric value into a logical by passing to `as.numeric()` first. 
  In addition, : there are several character values that can be directly
  converted to logical values. The example code below enumerates them: 

```
as.logical("0")                     ## nope!
as.logical("1")                     ## nope!
as.logical("")                      ## nope

## TRUE and FALSE character equivalents:

as.logical("TRUE")                  ## works!
as.logical("FALSE")
as.logical("T")
as.logical("F")
as.logical("true")                  ## works! even though 'true' is not keyword (unlike 'TRUE' and 'T')
as.logical("false")
as.logical("True")
as.logical("False")

as.logical(as.character(TRUE))      ## works (as expected)!
as.logical(as.character(FALSE))     ## works (as expected)!

as.logical(as.numeric("0"))         ## simple work-around for character 'numbers'
as.logical(as.numeric("-0.0"))      ## simple work-around for character 'numbers'
as.logical(as.numeric("1"))         ## simple work-around for character 'numbers'
as.logical(as.numeric("-3.2e-16"))  ## simple work-around for character 'numbers'

```

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
  the 32-bit version is limited to vectors of only a few billion(!) elements. In order
  to see the particulars of number representation, type `.Machine` at the `>` prompt
  and press `<ENTER>`. One thing to note is that the value of .Machine$integer.max is
  'only' around 2 billion. However, using numerics (represented by double-precision 
  floating point numbers with .Machine$double.digits (53) 'fraction' bits and 
  .Machine$double.exponent (11) 'exponent' bits, successive integers can be exactly 
  represented (not rounded) up to 9,007,199,254,740,992 (`2 ^ 53`). Beyond this number, 
  at best, only every other integer can be expressed. The point is that you can freely 
  use the numeric type to represent very large integers as long as you initialize with 
  a whole number only add, subtract or multiply by other whole numbers, and avoid 
  division or other operations that might result in fractional results. So numerics 
  are perfectly suitable for counting and indexing.

```
.Machine
.Machine$integer.max              ## largest representable integer

(x <- .Machine$double.exponent)   ## number of double 'exponent' bits
2 ^ (x - 1)                       ## decimal value of maximal exponent (base-10)

(x <- .Machine$double.digits)     ## number of double 'digits' () bits
2 ^ x                             ## can exactly represent all integers up to this one

```

Another issue to consider is that numeric calculations often result in some rounding
  that can make answers differ slightly from their theoretical values. We've
  already seen this once in a previous lesson when running `solve()` on a matrix.
  Therefore, it is often helpful or critical to distinguish tests of exact equality
  (what you've seen thus far) from tests of approximate equality. 

```
(x <- c(1, 2, 3, 1, 2, 1, 2, 3, 1))
(x <- matrix(x, ncol=3))
(y <- x %*% solve(x))             ## note rounding issue (should be identity)
(i <- diag(3))                    ## identity matrix; equiv: cbind(c(1,0,0), c(0,1,0), c(0,0,1))

identical(y, i)                   ## is y exactly equal to its theoretical value?
y == i                            ## which positions have values that match
y[y != i]                         ## which values don't match

all.equal(y, i)                   ## is y approximately equal to its theoretical value?
isTRUE(T)                         ## is.logical(x) && length(x) == 1 && !is.na(x) && x
isTRUE(all.equal(y, i))           ## the 'safe' way to test for equality

```

Standard floating point representations in any programming language include several special 
  values including 'not a number' (represented by `NaN` in R) and 'infinity' (`Inf` in R). 
  The `NaN` value in R turns out to be a 'subtype' of the NA type: all `NaN` are `NA`, but 
  not all `NA are `NaN`. R provides  special functions for detection of `NaN` and `Inf` 
  values. This is particularly important for `NA` and `NaN` values, as any comparisons of 
  these values, even with themselves, only yields `NA`: 

```
(x <- c(-1 / 0, 0 / 0, 1 / 0))    ## note 0/0 is NA, not Inf
is.finite(x)
is.infinite(x)
is.nan(x)
is.na(x)

(x <- c(-Inf, NA, NaN, Inf))
is.finite(x)
is.infinite(x)
is.nan(x)
is.na(x)

NA == NA                          ## nope
NaN == NA                         ## nope
NaN == NaN                        ## nope
Inf == Inf                        ## works!
-Inf == -Inf                      ## works!
Inf == -Inf                       ## makes sense!

```

[Return to index](#index)

---

### Conditional execution

R includes facilities for conditional code execution similar to those found
  in many other languages. In particular, it offers `if`, `for` and `while` 
  statements. In addition, it includes the keyword `repeat` to repeat execution
  of a loop, `break` to exit the loop prematurely, and `next` to skip the 
  execution of the current loop iteration and begin the next iteration.

One thing to keep in mind is that many R functions/operators already work with 
  vectors (like the `+` operator or the `sum()` function), or can be adapted to 
  operate on vectors or lists using the `sapply()` and `lapply()`  commands. In turn, 
  vectorized operations can be extended to matrices, arrays and data.frames using
  the `apply()` function. As a result, there is less need to explicitly code
  loops that iterate over data than in some other lower-level languages. In addition
  to speeding development by simplifying code, using these vector/array-aware 
  facilities can often result in reductions in run time of more than an order of
  magnitude. So, whenever you set out to write a `for` loop or `while` loop in R,
  first make sure that a vectorized operation, `sapply()`, `lapply()`, `tapply()`, 
  or `apply()` is not an adequate solution for your problem.

You can execute an entire block of code conditionally by placing that block within
  curly brackets `{}`. 

Here are some examples of how to use the `if` statement and blocks of code:

```
## '\n' is carriage return, or end-of-line marker:

if(T) cat("is true\n")            ## a whole expression; end-of-line executes
if(F) cat("is true\n")            ## nothing printed since false

## the right way to 'if else':

if(F) {                           ## expression won't be done till block closed w/ '}'
  cat("is true\n")                ## first line in 'if' block of code
  cat("really is true\n")         ## second/last line in 'if' block
} else {                          ## block closing '}' ON SAME LINE as 'else {' open block
  cat("is false\n")               ## first line of 'else' block
  cat("really is false\n")        ## second/last line of 'else' block
}                                 ## 'else' block complete, end-of-line executes

## the wrong way to 'if else':

if(F) { cat("is true\n") }        ## a whole expression; executed w/ end-of-line
else { cat("is false\n") }        ## oops! expression starting w/ 'else' not valid

## braces are optional:

if(T) cat("true\n") else cat("false\n")  ## one-liner
if(F) cat("true\n") else cat("false\n")  ## one-liner

## a more realistic example:

(x <- c(1, 2, 3, 1, 2, 1, 2, 3, 1))
(x <- matrix(x, ncol=3))
(y <- x %*% solve(x))             ## note rounding issue (should be identity)
(i <- diag(3))                    ## identity matrix; equiv: cbind(c(1,0,0), c(0,1,0), c(0,0,1))

## not right:

if(y == i) cat("==\n") else cat("!=\n")

## better, but often not what you want:

if(identical(y, i)) {
  cat("identical\n") 
} else cat("not identical\n")

## even better:

if(all.equal(y, i)) {
  cat("close enough\n")
} else cat("not close enough")

## the right way: more robust to potential error conditions

if(isTRUE(all.equal(y, i))) {
  cat("close enough\n") 
} else cat("not close enough\n")

```

In R, `while` loops have a fairly simple structure:

```
x <- 1
while(x < 10) x <- x + 1
x

x <- 1
while(x < 10) {
  cat("x:", x, "\n")             ## can mix literals and variables
  x <- x + 1
}
x

x <- 1
while(T) {                       ## infinite loop (TRUE is always TRUE)
  cat("x:", x, "\n")
  x <- x + 1
  if(x >= 10) break              ## conditionally 'break' out of loop
}
x

```

In R, 'for' loops cann be used to iterate over sequences of values:

```
x <- 1:10
for(x.i in x) cat("x.i:", x.i, "\n")
x

x <- 1:10
for(x.i in x) {
  cat("Initially, x.i:", x.i, "\n")
  x.i <- x.i + 1
  cat("Later, x.i:", x.i, "\n")
}
x                                 ## 'x' unchanged

x <- 1:100
for(x.i in x) {
  tmp <- x.i %% 5                 ## remainder from dividing x.i by 5
  if(!tmp) cat(x.i, "\n")         ## print out if no remainder; as.logical(0) == F
}

## example use of 'next':

x <- 1:100
for(x.i in x) {
  if(x.i %% 5) next               ## if remainder, then go to next iteration
  cat(x.i, "\n")                  ## print out if no remainder
}

```

[Return to index](#index)

---

### User defined functions

```
myfunc <- function(a, b) {
  a / b                               ## result of last statement returned by default
}

myfunc(1:5, 6:10)
(1:5) / (6:10)

myfunc <- function(a) {
  return median(abs(a - median(a)))   ## or can use 'return' keyword
}

myfunc(1:10)
mad(1:10, constant=1)

myfunc <- function(a, b) {
  tmp <- c(a, b)                      ## 'tmp' is only visible inside 'myfunc'
  list(
    mean=mean(tmp),
    mean.a=mean(a),
    mean.b=mean(b),
    sd=sd(tmp),
    sd.a=sd(a),
    sd.b=sd(b)
  )
}

myfunc(1:5, seq(from=2, to=10, length.out=5))

```

[Return to index](#index)

---

### Data import and export

Navigate directories:

```
getwd()                   ## where am I at?
setwd("C:\Users\kostim\tmp\rclass")           ## Oops! '\' is escape character: changes meaning of next char
setwd("C:/Users/kostim/tmp/rclass")           ## '/' path delimiter works on windows + linux
setwd("C:\\Users\\kostim\\tmp\\rclass")       ## this works on Windows too (escape '\' itself).
getwd()

```

Read/write data frame:

```
x1 <- data.frame(
  diet=c(rep('ad lib', 10), rep('low fat', 10), rep('high protein', 10)),
  strain=rep(c('BL/6', 'AJ'), 15),
  mass=c(rnorm(10, mean=25, sd=7), rnorm(10, 15, 5), rnorm(10, 20, 6))
)

x1
write.table(x1, file="mouse_diets.tsv", quote=T, sep="\t", row.names=F)
x2 <- read.table(file="mouse_diets.tsv", header=T, sep="\t", as.is=T)
all.equal(x1, x2)

## also: colClasses, fill, row.names, col.names
## read.csv(sep=',')
## read.delim(sep='\t')

```
Read/write vector/matrix:

```
x1 <- matrix(1:15, ncol=3)

write(x1, file="matrix1.txt", ncolumns=3, sep=",")
x2 <- scan("matrix1.txt", what=integer(), sep=",")
all.equal(x1, x2)

x1
x2
x2b <- matrix(x2, ncol=3)
all.equal(x1, x2b)

## scan(file, what, sep, quote, fill, blank.lines.skip, 
## readLines(), readBin(), readChar()

```

[Return to index](#index)

---

## FIN!

