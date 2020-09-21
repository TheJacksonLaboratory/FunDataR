# Fundamentals of computational data analysis using R
## A primer on programming using R: lesson 3
#### Contact: mitch.kostich@jax.org

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

[Return to index](#index)

---

### Type conversions

You may have noticed that you can combine integer and numeric types successfully, but 
  what is the resulting type? Let's take a look:

```
x <- 3L
y <- 4
class(x)
class(y)
(z <- x + y)
class(z)

```

So adding a numeric and integer results in a numeric answer that looks correct. What 
  about other combinations of data types? A simple way to explore this is with the
  `c()` concatenation function, which will take many different data types as arguments
  but always returns a vector with a uniform type:

```
(x <- c(F, 1L))
class(x)

(x <- c(T, 2L, 3))
class(x)

(x <- c(F, 1L, 2, "3"))
class(x)

```

So the direction of conversion is `logical -> integer -> numeric -> character`. If any
  type is combined with any second type that lies to the right of the first type in 
  the series, the first type is converted to the second type. The conversions are pretty
  sensible for the most part, as was demo'd in the previous code block. But what about
  converting types in the opposite direction? Turns out that is possible too, using 
  the `as.<type>()` series of functions:

```
str(x <- list(logical=T, integer=1L, numeric=1, character="1"))
sapply(x, as.logical)
sapply(x, as.integer)
sapply(x, as.numeric)
sapply(x, as.character)

str(x <- list(logical=F, integer=0L, numeric=0, character="0"))
sapply(x, as.logical)
sapply(x, as.integer)
sapply(x, as.numeric)
sapply(x, as.character)

str(x <- list(logical=F, integer=-3L, numeric=-0.1, character=""))
sapply(x, as.logical)
sapply(x, as.integer)
sapply(x, as.numeric)
sapply(x, as.character)

as.logical("TRUE")                ## works!
as.logical("FALSE")
as.logical("T")
as.logical("F")
as.logical("true")                ## works! even though 'true' is not keyword (unlike 'TRUE' and 'T')
as.logical("false")

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

[Return to index](#index)

---

### Conditional execution

if(cond) expr
if(cond) cons.expr  else  alt.expr

for(var in seq) expr
while(cond) expr
repeat expr
break
next

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

