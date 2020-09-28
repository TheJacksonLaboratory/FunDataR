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
  missing object (rather than value). It should normally not be used as 
  an element of vector (use `NA` instead), but instead to indicate that 
  an entire vector or other object is missing. In that capacity it is 
  often used as a failure return value from functions which would otherwise
  yield objects, and it can be assigned as an element of a list to indicate
  that element (which can be any type of object) is missing. If you used
  an `NA` for a missing list element, it would not be clear if the 
  corresponding element was missing or if that element was a vector of 
  length 1 whose only element was the value `NA`.

One can detect `NA`s using the `is.na()` function, which returns a value 
  for every element in a vector (since any individual element can be set to `NA`). 
  Similarly, `NULL` values are detected using the `is.null()` function, which 
  returns a single value for any object you feed to it.

```
ls()                                ## messy workplace
rm(list=ls())                       ## clean the slate
ls()                                ## start with a fresh slate

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
x                                   ## NOT THE SAME as setting to NULL ...
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
rm(list=ls())                ## fresh slate

x <- 3L                      ## integer
y <- 4                       ## numeric
class(x)
class(y)
(z <- x + y)
class(z)

```

So adding a numeric and integer results in the correct numeric value. This makes sense,
  since a numeric value can represent an integer such as `17`, but an integer cannot
  represent a numeric value such as `17.25`. So the numeric type is more general and 
  in some sense encompasses the integer type. The same logic applies when combining
  other types. A simple way to explore the effects of combining types is with the
  `c()` concatenation function, which will take many different data types as arguments
  but always returns a vector with a uniform type, converting other input types into
  the whatever the final type happens to be:

```
rm(list=ls())                ## fresh slate

## combine logical with integer:
(x <- c(F, 1L))
class(x)

## combine logical, integer, and numeric:
(x <- c(T, 2L, 3))
class(x)

## combine logical, integer, numeric, and character:
(x <- c(F, 1L, 2, "3"))
class(x)

```

So the direction of auto-conversion is `logical -> integer -> numeric -> character`. 
  This makes sense since `TRUE` can be encoded as `1L` and `FALSE` as `0L`, that is
  integers can adequately represent logical values. As mentioned earlier, numerics
  can represent integers. Furthermore, the character type can represent any numeric
  by simply quoting it. If any type is combined with any second type that lies to the 
  right of the first type in the series, the type on the left in the series is 
  converted to the type that lies further to the right.  

But what about converting types in the opposite direction? Turns out that is possible 
  too, using the `as.<type>()` series of functions:

```
rm(list=ls())                     ## fresh slate

str(x <- list(logical=T, integer=1L, numeric=1, character="1"))
sapply(x, as.logical)             ## numeric/integer 0 is FALSE; else TRUE 
sapply(x, as.integer)             ## logical TRUE -> 1L, FALSE -> 0L
sapply(x, as.numeric)             ## logical TRUE -> 1, FALSE -> 0
sapply(x, as.character)           ## mostly just quotes stuff; T -> 'TRUE'

str(x <- list(logical=F, integer=0L, numeric=0, character="0"))
sapply(x, as.logical)             ## numeric/integer 0 is FALSE; else TRUE
sapply(x, as.integer)             ## logical TRUE -> 1L, FALSE -> 0L
sapply(x, as.numeric)             ## logical TRUE -> 1, FALSE -> 0
sapply(x, as.character)           ## mostly just quotes stuff; T -> 'TRUE'

str(x <- list(logical=F, integer=-3L, numeric=-0.1, character="-1.6e5"))
sapply(x, as.logical)             ## numeric/integer 0 is FALSE; else TRUE
sapply(x, as.integer)             ## logical TRUE -> 1L, FALSE -> 0L
sapply(x, as.numeric)             ## logical TRUE -> 1, FALSE -> 0
sapply(x, as.character)           ## mostly just quotes stuff; T -> 'TRUE'

```

So a character representation of a number can be converted into an integer or
  numeric, and an integer or numeric can be converted into a logical, but 
  a character representation of a number cannot be directly converted into
  a logical! Instead, you can always indirectly convert a character representation 
  of a numeric value into a logical by passing to `as.numeric()` first. 
  However, there are several character values that can be directly
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
  division or other operations (like combining with fractional values) that might 
  result in fractional results. If you follow those rules, numerics are perfectly 
  suitable for counting and indexing.

```
rm(list=ls())                     ## fresh slate

.Machine                          ## R limits on current machine
.Machine$integer.max              ## largest representable integer in R

(x <- .Machine$double.exponent)   ## number of double 'exponent' bits
2 ^ (x - 1)                       ## decimal value of maximal exponent (base-10)

(x <- .Machine$double.digits)     ## number of double 'digits' () bits
2 ^ x                             ## can exactly represent all integers up to this one

```

Standard floating point representations in any programming language include several special 
  values including 'not a number' (represented by `NaN` in R) and 'infinity' (`Inf` in R). 
  The `NaN` value in R turns out to be a 'subtype' of the `NA` type: all `NaN` are `NA`, but 
  not all `NA` are `NaN`. R provides  special functions for detection of `NaN` and `Inf` 
  values. This is particularly important for `NA` and `NaN` values, as any comparisons of 
  these values, even with themselves, only yields `NA`: 

```
rm(list=ls())                     ## fresh slate

1 / 0
-1 / 0
0 / 0

(x <- c(ninf=-Inf, na=NA, nan=NaN, inf=Inf))
is.finite(x)
is.infinite(x)
is.nan(x)
is.na(x)

NA == NA                          ## nope
NaN == NA                         ## nope
NaN == NaN                        ## nope
Inf == Inf                        ## works!
-Inf == -Inf                      ## works!
Inf == -Inf                       ## makes sense, sign matters

```

Another issue to consider is that numeric calculations often result in some rounding
  that can make answers differ slightly from their theoretical values. We've
  already seen this once in a previous lesson when running `solve()` on a matrix.
  Therefore, it is important to distinguish tests of exact equality (what you've seen 
  thus far) from tests of approximate equality (what you should typically use
  for comparison of numerics). 

```
rm(list=ls())                     ## fresh slate

(x <- c(1, 2, 3, 1, 2, 1, 2, 3, 1))
(x <- matrix(x, ncol=3))
(y <- x %*% solve(x))             ## note rounding issue (should be identity)
(i <- diag(3))                    ## identity matrix; equiv: cbind(c(1,0,0), c(0,1,0), c(0,0,1))

## tip: a logical matrix can be used to index another matrix of the same dimensions, 
##   as is done below:

identical(y, i)                   ## is y exactly equal to its theoretical value?
y == i                            ## which positions have values that match; logical matrix
y[y != i]                         ## which values don't match; logical matrix indexing matrix 'y'
i[y != i]                         ## the corresponding values from 'i'

all.equal(y, i)                   ## is y approximately equal to its theoretical value?

is.logical(NULL)                  ## FALSE
is.logical(NA)                    ## TRUE!
as.logical(NULL)                  ## returns a 0 length logical: is that T or F?
as.logical(NA)                    ## returns NA: is that T or F? 
isTRUE(NULL)                      ## is.logical(NULL) && length(NULL) == 1 && !is.na(NULL) && NULL
isTRUE(NA)                        ## is.logical(NA) && length(NA) == 1 && !is.na(NA) && x

isTRUE(T)                         ## works as expected
isTRUE(F)                         ## works as expected
isTRUE(all.equal(y, i))           ## the robust (safe) way to test for equality

## Why go to all this trouble? See below:

a <- 1
b <- 0
(x <- b / b)                                               ## oops! numeric operation yielding broken 'x'

if(a == 0) cat("yes\n") else cat("no\n")                   ## introducing the R 'if()' statement!
if(b == 0) cat("yes\n") else cat("no\n")                   ## 'cat()' used to print to console; '\n' = 'new line'
if(x == 0) cat("yes\n") else cat("no\n")                   ## oops!
x == 0                                                     ## not a very useful index

all.equal(x, 0)                                            ## close, but not logical
if(all.equal(x, 0)) cat("yes\n") else cat("no\n")          ## no cigar!

isTRUE(all.equal(x, 0))                                    ## a logical at last!
if(isTRUE(all.equal(x, 0))) cat("yes\n") else cat("no\n")  ## finally, nice!!!

```

[Return to index](#index)

---

### Check your understanding 1

Set up your data as follows:

```
set.seed(3)                       ## make reproducible
x <- rbinom(30, 1, 0.5)           ## 30 coin flips (heads=1, tails=0)
y <- rbinom(30, 1, 0.5)           ## 30 1s and 0s
z <- x / y                        ## 1/0 is Inf; 0/0 is NaN
x.char <- as.character(x)

```

1) How many infinite values are in `z`? Hint: make a logical index
     using `is.finite()`; `table()` or `sum()` the result.

2) How many `NaN` values are in `z`? Hint: make a logical index ...

3) Return all the finite numeric values (not infinity and not `NaN`)
     in `z`; Hint: `&` two logical vectors (perhaps one returned 
     by `is.nan()`, the other by ...) to get another logical suitable
     for indexing.

4) Convert `x` into a logical vector, where `1` is replaced with `TRUE`
     and `0` with `FALSE`.

5) Convert `x.char` into a logical vector where `'1'` is replaced with
     `TRUE` and `'0'` with `FALSE`. Hint: two successive conversions.

[Return to index](#index)

---

### Conditional execution

R includes facilities for conditional code execution similar to those found
  in many other languages. In particular, it offers `if`, `for` and `while` 
  statements. In addition, it includes the keyword `repeat` to repeat execution
  of a loop, `break` to exit the loop prematurely, and `next` to skip the 
  execution of the current iteration and begin the next iteration.

One thing to keep in mind is that many R functions/operators already work with 
  vectors (like the `+` operator or the `sum()` function), or can be adapted to 
  operate on vectors or lists using the `sapply()` and `lapply()`  commands. In turn, 
  vectorized operations can be extended to matrices, arrays and data.frames using
  the `apply()` function. As a result, there is less need to explicitly code
  loops that iterate over data than in some other lower-level languages. In addition
  to speeding development and simplifying code, these vector/array-aware 
  facilities can result in run time reductions of more than an order of
  magnitude. So, whenever you set out to write a `for` loop or `while` loop in R,
  first make sure that a vectorized operation, `sapply()`, `lapply()`, `tapply()`, 
  or `apply()` is not an adequate solution for your problem.

You can execute an entire block of code conditionally by placing that block within
  curly brackets `{}`. 

Here are some examples of how to use the `if` statement and blocks of code:

```
rm(list=ls())                     ## fresh slate

## '\n' is carriage return, or end-of-line marker:

if(T) cat("yes\n")                ## a whole expression; end-of-line executes
if(F) cat("yes\n")                ## nothing printed since false

## wrong way to 'if else' across lines:

if(F) { cat("yes\n") }            ## a complete expression; executed at end-of-line
else { cat("no\n") }              ## oops! expression starting w/ 'else' not valid

## right way to 'if else' across lines:

if(F) {                           ## expression won't be done till block closed w/ '}'
  cat("yes\n")                    ## first line in 'if' block of code
  cat("is true\n")                ## second/last line in 'if' block
} else {                          ## block closing '}' ON SAME LINE as 'else {' open block
  cat("no\n")                     ## first line of 'else' block
  cat("is false\n")               ## second/last line of 'else' block
}                                 ## 'else' block complete, end-of-line executes

## braces are optional:

if(T) cat("yes\n") else cat("no\n")  ## one-liner
if(F) cat("yes\n") else cat("no\n")  ## one-liner

## chain multiple conditions like so:

x <- 0
if(x < 0) {
    cat(x, "is less than zero\n")   ## can mix variables and literals w/ cat()
} else if(x == 1) {
    cat(x, "is one\n")        
} else if(x > 0) {
    cat(x, "is greater than zero\n")
} else {
    cat(x, "is zero\n")
}

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

## the right way; more robust to potential errors:

if(isTRUE(all.equal(y, i))) {
  cat("close enough\n") 
} else cat("not close enough\n")

```

In R, `while()` loops have a fairly simple structure:

```
rm(list=ls())                    ## fresh slate

x <- 1
while(x < 10) 
  x <- x + 1
x

x <- 1
while(x < 10) {
  cat("x:", x, "\n")
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

In R, `for()` loops can be used to iterate over sequences of values:

```
rm(list=ls())                     ## fresh slate

(x <- 1:10)                       ## vector over which to iterate
for(x.i in x) 
  cat("x.i:", x.i, "\n")

## iterate over a list:

x <- list(weight=37, colors=c('red', 'green', 'blue'), is.even=F)
for(x.i in x) 
  cat("x.i:", x.i, "length(x.i):", length(x.i), "\n")

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

### Check your understanding 2

1) Use a `for()` loop to sum all the odd whole numbers from `1` to `99`. Hint: 
     initialize a variable to hold the sum to 0, iterate over the series of 
     numbers, adding each to the sum. Consider generating a sequence of odd 
     numbers w/ `seq()`, in order to make answering problem #3 easier. 
     Alternatively (you like a challenge, don't you?) check successive numbers 
     for oddness; `%%` is the modulo operator, which yields the remainder from 
     division (e.g. by `2`).

2) Do the same thing with a `while()` loop. 

3) Check your work using the `sum()` function. Hint: no loop required if you used 
     `seq()` to make a sequence of odd numbers for problem #1.

4) Use a `while()` loop to total up the whole numbers from `1` to `100`, printing 
     out (you can use `cat()`) the total after each successive addition, stopping
     (exiting the `while()` loop) when the total reaches `1000`. Hint: use an `if()`
     within your `while()`.

[Return to index](#index)

---

### User defined functions

One of the most powerful and features R offers is the ability for users to define 
  their own functions. The combination of user-defined functions and functions
  like those you've seen for applying arbitrary functions to data (such as 
  `lapply()`, `sapply()`, `tapply()`, and `apply()` allows complex
  operations to be executed on large datasets using very concise (but hard to
  decypher by newbies) code.

In R, the `function()` function is used to define new functions. The output of
  `function()` can be assigned to a variable or used directly. Like expressions 
  containing conditional execution statements, function definitions can include 
  multi-line blocks of code enclosed within `{}` or single lines of code.
  In the latter case, the use of `{}` is optional. :

```
rm(list=ls())                         ## fresh slate

x <- 1:10
myfunc <- function(a) sum(a)          ## pass-thru function; param named 'a'; executes 'sum(a)'
myfunc                                ## invoke just the name: see what's under the hood
myfunc(x)                             ## invoke with parentheses '()': execute myfunc
sum(x)                                ## quick check

myfunc2 <- function(a) {
  sum(a)
}
myfunc2(x)

myfunc3 <- function(a) {

  ## argument variables such as 'a' and internally defined variables such as 'tot' 
  ##   and 'nxt' are only visible within the function block. The variable values
  ##   only exist from the point where the variable is defined until the function 
  ##   is exited.

  tot <- 0
  for(nxt in a) tot <- tot + nxt
  tot                                 ## last expression 'tot' is returned
}
myfunc3(x)

myfunc4 <- function(a) {
  tot <- 0
  for(a.i in a) tot <- tot + a.i
  return(tot)                          ## can use explicit return() anywhere in block 
}
myfunc4(x)
myfunc4

```

A number of fancy features await you, including the ability to accept multiple arguments,
  assign default values to arguments, and return complex objects:

```
rm(list=ls())                        ## fresh slate

## x required; y and z are optional since defaults set:

myfunc <- function(x, y=20:11, z=21:30) {  

  ## validate input type:
  if(! (is.numeric(x) && is.numeric(y) && is.numeric(z)) ) 
    stop("oops! all arguments must be numeric")

  ## validate input lengths:
  if(! (length(x) == length(y) && length(x) == length(z)) )
    stop("oops! all arguments must have same length")

  out <- list(x=x, y=y, z=z)         ## save parameters for output

  ## add some more useful elements to 'out':
  out$avg.x <- mean(x, na.rm=T)      ## na.rm: ignore missing values
  out$avg.y <- mean(y, na.rm=T)
  out$avg.z <- mean(z, na.rm=T)
  out$sum <- out$x + out$y + out$z
  out$cumsum <- data.frame(x=cumsum(x), y=cumsum(y), z=cumsum(z))

  out                                ## return 'out'
}

myfunc(1:10, 10:1, 1:10)             ## first x, second y, third z (like definition)
myfunc(1:10, 10:1)                   ## missing z, use default
myfunc(1:10)                         ## missing y and z, use defaults
myfunc()                             ## oops! no default assigned to first argument
myfunc(x=1:10)                       ## explicitly assign
myfunc(y=1:10)                       ## oops! no default assigned to x
(x <- myfunc(1:10, z=rep(1, 10)))    ## get to skip y (use defaults for y)
x$cumsum$z[3:5]
x$cumsum[3:5, 3]

myfunc('abc', 10:1, 1:10)            ## first argument not numeric
myfunc(1:5, 10:1, 1:10)              ## lengths don't match

```

[Return to index](#index)

---

### Check your understanding 3

1) Write a function that takes two arguments, and yields a single vector where the values 
     from the first argument are followed by a reversed copy of the second argument. That 
     is, the output vector should have a length equal to the the sum of the lengths of 
     its arguments. Hint: use `c()` to concatenate vectors and `rev()` to reverse them.

2) Write a function that takes a single argument. If that argument is `logical`,
     the number of `TRUE` values is returned. If that argument is `numeric`, a vector
     with the square-root of each element is returned. If that argument is `character`,
     then the length of the argument is returned. Hint: use `is.<type>()` series of
     functions with `if()`.

3) Write a function that takes a numeric vector as argument, and yields a vector
     of the same length where the first element of the input vector is multiplied by `1`,
     the second element is multiplied by `2`, the third by `3`, etc. Hint: determine
     vector length with `length()`; `seq()` has a parameter `length.out`.

[Return to index](#index)

---

### Data import and export

Some commonly used functions for navigating file systems using R are demonstrated below. 
  There are also functions such as `dir.create()` for creating directories, `dir.exists()`
  for seeing if a directory exists, `file.exists()` for testing if a file exists, and
  `unlink()` for removing files as well as directories. In R, the backslash serves as an
  escape in character literals, which means that it changes the meaning of the character 
  following the escape, usually resulting in the substitution of a 'special' character.
  This complicates the use of backslashes `\` in file paths. They must be either doubled
  up (escaping the escape character, substituting a plain old backslash) or replaced with
  a forward slash. Forward slashes can be used as path delimiters in R on both Linux and
  Windows machines.

```
rm(list=ls())                              ## fresh slate

getwd()                                    ## where am I at?
list.files()                               ## what is in this folder (default path is '.')
list.files(path="..")                      ## what is in the parent folder of this folder
list.files(path="..", pattern=".dat$")     ## list files ending in ".txt" in parent folder

for(file.i in list.files()) 
  cat("next:", file.i, "\n")

cat("\t", "here\n")                        ## '\' escape, changes meaning of next char; '\t' tab
cat("\\t", "here\n")                       ## use '\\' if you want a backslash to be a backslash

## set these paths to point to legitimate locations in your file system:
getwd()                                    ## where I start out
setwd("C:\Users\kostim\tmp\rclass")        ## Oops! '\' escape: '\t' tab '\r' carriage return
setwd("C:/Users/kostim/tmp/rclass")        ## '/' path delimiter works on windows + linux
setwd("C:\\Users\\kostim\\tmp\\rclass")    ## this works on Windows too (escape '\' itself).
getwd()                                    ## where I end up

```

There are several ways to write data to the filesystem or read data from the filesystem into R.
  The most commonly used method is reading and writing tabular data in a data.frame. The basic
  functionality is demonstrated below. There are particular variants of `read.table()` and
  `write.table()` that work the same but have different defaults that may be more convenient 
  for particular situations, like `read.csv()` and `write.csv()` for comma-separated data, 
  or `read.delim()` and `write.delim()` for tab-delimited data. 

```
rm(list=ls())                       ## fresh slate

## make up a data.frame with fake data:
x1 <- data.frame(
  diet=c(rep('ad lib', 10), rep('low fat', 10), rep('high protein', 10)),
  strain=rep(c('BL/6', 'AJ'), 15),
  mass=c(rnorm(10, mean=25, sd=7), rnorm(10, 15, 5), rnorm(10, 20, 6))
)
rownames(x1) <- as.character(1:nrow(x1))
x1

## set this path to point to legitimate location in your file system:
##   write the data to file mouse_diets.tsv:
setwd("C:/Users/kostim/tmp/rclass")
write.table(x1, file="mouse_diets.tsv", quote=T, sep="\t")

## set this path to point to a legitimate location in your file system:
##   read the data from mouse_diets.tsv into variable x2:
setwd("C:/Users/kostim/tmp/rclass")
x2 <- read.table(file="mouse_diets.tsv", header=T, sep="\t", as.is=T)
all.equal(x1, x2)

```

As we mentioned when we were describing matrices and data.frames, 
  data.frames offer the flexibility of including columns of different
  data types, while matrices offer efficiency gains that can be
  realized only when all the column types are the same. This applies to
  reading and writing data as well. If you have tabular data in a 
  uniform format, you are usually better off storing that data in a
  matrix instead of a data frame. Similarly, you are better off reading
  and writing matrices to the filesystem using `write()` to write the
  matrix data to the filesystem and `scan()` for reading data from
  the filesystem into a matrix variable. These two functions are also
  used for reading or writing vector data to disk.

```
rm(list=ls())                     ## fresh slate

x1 <- matrix(1:15, ncol=3)
## set this path to point to a legitimate location in your file system:
setwd("C:/Users/kostim/tmp/rclass")
write(x1, file="matrix1.txt", ncolumns=ncol(x1), sep=",")
x2 <- scan("matrix1.txt", what=integer(), sep=",")
all.equal(x1, x2)

x1
x2
x2b <- matrix(x2, ncol=3)
all.equal(x1, x2b)

rm(list=ls())                     ## fresh slate

x1 <- seq(from=0, to=20, by=2) ^ 2
## set this path to point to a legitimate location in your file system:
setwd("C:/Users/kostim/tmp/rclass")
write(x1, file="vector1.txt", ncolumns=1, sep=",")
x1
x2
x2 <- scan("vector1.txt", what=numeric(), sep=",")
all.equal(x1, x2)

```

[Return to index](#index)

---

## FIN!

