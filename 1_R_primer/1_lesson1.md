# Fundamentals of computational data analysis using R
## A primer on programming using R

### Index

#### [Expressions](#expressions)
#### [Numbers](#numbers)
#### [Logicals](#logicals)
#### [Characters](#characters)

### Expressions

After starting R, an R terminal window will be launched. If you bring that window into focus, you can
  start entering commands at the `>` prompt. An 'expression' is another term for a command that the
  R interpreter can process and potentially return a value. Pressing <ENTER> after a complete expression 
  results in the execution of that expression. If the expression is clearly incomplete, pressing <ENTER> 
  will take you to the next line, where you can continue entering your expression. If the expression does
  not include an assignment of the resulting value to a variable (discussed later in this lesson), the 
  resulting value is typically printed out on the screen. The simplest expressions are values themselves.
  Each value has a data type associated with it, which can be determined using the `class()` function.
  The data type determines which operators and functions can be applied to the value.


### Numbers

The first data type we will look at are data types used to represent numbers. There are several data
  types that are used to represent numbers. The most common is the 'numeric' type, which is
  encoded in a double-precision floating point format. The numeric type is most useful for 
  representing fractional numbers, like 1.23 as well as very large or very small magnitude numbers, 
  like 1.3e36 (1.3 x 10<sup>36</sup>) or 3.97e-24 (3.97 x 10<sup>-24</sup>). Another data type
  used for representing numbers is 'integer'. Integers are used to represent whole
  numbers (positive, negative and zero), like the counting numbers. In R these are primarily used
  for counting (no surprise) and indexing locations in more complex data structures. R also has 
  good support for representing complex numbers (numbers composed of 'real' and 'imaginary' parts), 
  but we won't describe those further in this class. For most practical purposes, integers and 
  doubles can be used interchangeably, so we will gloss over the differences and conversion rules
  for now:<br>

A value by itself is a complete expression. So you can enter any acceptable value at the `>` prompt,
  press <ENTER>, and the value will be printed on the next line in the terminal. A number is composed
  of digits, an optional sign, an optional single decimal place, and either `E` or `e` followed by
  an positive or negative integer exponent (see above for some examples):

```
2
```

A function is a bit of computer code that can be called by name, given an argument (in parentheses
  following the function's name), and returns a value and/or performs some operations. One simple 
  function is `class()`, which tells you the data type of its argument:<br>

```
class(2)
```

An expression can be made by combining operators (like the `+` below) with values (like the numeric
  values `2` and `3` 
2 + 3
class(2 + 3)
?"+"
?"Syntax"
??precedence
help('+')
2 / 3
2 * 3
2 - 3
2 ^ 3

2 + 3 / 5
(2 + 3) / 5
3 ^ (1 / 2)
(3 ^ (1 / 2)) ^ 2

## QUIZ:

1) what is the third-root of 5

2) what is the sum of 500,726 and 324,781, divided by 67?

3) what is 3.14 to the 3.14 power?


### Logicals:

2 < 3
2 >= 3
2 == 3
2 == 2
2 != 3
?"<"
class(2 < 3)

TRUET
TR     ## oops
class(T)
FALSE
F

TRUE && TRUE
TRUE && FALSE
TRUE || TRUE
TRUE || FALSE
FALSE || FALSE
xor(TRUE, FALSE)
xor(TRUE, TRUE)
xor(FALSE, FALSE)
!T
!F

### characters:

"abc de"         ## 1x and 2x equivalent; 2x 'preferred'
class("abc de")
''               ## use 2x to embed 1x and vice-versa
class('')

nchar('')
nchar('abc de')
substr('abc de', 2, 4)
substr('abc de', 2, 4) <- 'yz'

tolower('AbCdE')
toupper('AbCdE')

sub('c', 'D', 'abcba')
sub('c', 'D', 'abcba')
sub('b', 'd', 'abcba')
gsub('b', 'd', 'abcba')

grepl('bc', 'abcba')
grepl('yx', 'abcba')

## variables:

x <- 3          ## avoid c, q, t
x
x <- 2 + 3
x
x <- 2
y <- 3
x + y
z <- x + y
z

x <- 'abc'
x
grepl('y', x)
grepl('bc', x)

cat("x is:", x)
cat("x is:", x, "\n")   ## \t

x <- 2 
y <- 3
z <- 2 + 3
cat(x, "+", y, "=", x, "\n")

ls()
rm(x, y, z)      ## 'rm(list=ls())' gets rid of everything
ls()

## basic data type is vector-like:

x <- 3       
x
class(x)
length(x)

x <- c(1, -10, 100)      ## concatenation operator
x
class(x)
length(x)

x <- c(T, F, T)
x
class(x)
length(x)

x <- c('abc', '', 'de')
x
class(x)
length(x)

### sequence generators:

1 : 10
x <- 5 : -5

x <- 1:10
x
seq(from=1, to=10, by=2)
x <- seq(from=10, to=1, by=-2.5)
x

seq(from=1, to=10, length.out=19)

x <- rep(5, 10)
x
x <- rep('abc', 5)
x
x <- rep(c(T, F), 5)
x
x <- rep(1:3, 5)
x
x <- rep(seq(from=-1, to=-2, by=-0.25), 3)
x

## QUIZ:

1) generate the series from 1 to 100 counting by 3.5

2) how many numbers did you generate in #1?

3) generate the series from 10 to 1 counting by 2s

4) generate a 25 element vector repeating the series: 'a', 'b', 'c', 'd', 'e'

5) generate a 25 element vector with sequential values evenly spaced between 0 and 1


### vector numeric operators:

x <- 1 : 10
x + 3
x * 3
x ^ 2

x <- 1 : 10
y <- 21 : 30
x
y
x + y

x <- 1 : 4
y <- 1 : 2
x
y
x + y

x <- 1 : 4
y <- 1 : 3
x
y
x + y

### vector logical operators:

x <- 1 : 10
sum(x)
x < 8
sum(x < 8)        ## T==1; F==0
table(x < 8)

x >= 5
sum(x >= 5)
x < 8 && x >= 5   ## oops!
x < 8 & x >= 5
x > 8 || x < 5    ## oops!
x > 8 | x < 5
sum(x > 8 | x < 5)
!(x > 8 | x < 5)

### vector character operators:

x <- c('abcder', 'cdefghi', 'e', 'fgabc', 'ghijkla')
nchar(x)
substr(x, 2, 4)
substr(x, 2, 4) <- 'yz'
x
toupper(x)
x
gsub('e', 'z', x)
x
grepl('e', x)

### vector integer indexing:

x <- (1 : 10) ^ 2
x
x[1]
x[2]
x[3]
x[1 : 5]
x[10 : 6]
x[c(1, 10, 2, 9, 3, 8)]
x[-1]
x[-(1 : 5)]
x[seq(from=2, to=10, by=2)]

### vector logical indexing:

x <- (1 : 10) ^ 2
x
x[x > 30]
x[x <= 30]
x[x > 30 & x < 70]

x <- c('abcder', 'cdefghi', 'e', 'fgabc', 'ghijkla')
x[grepl('abc', x)]
x[! grepl('abc', x)]

x[c(T, F)]        ## recycle
x[c(T, F, T)]     ## no warning!!!

### vector logical indexing:

x <- (1 : 10) ^ 2
x
x[x > 30]
x[x <= 30]
x[x > 30 & x < 70]

x <- c('abcder', 'cdefghi', 'e', 'fgabc', 'ghijkla')
x[grepl('abc', x)]
x[! grepl('abc', x)]

x[c(T, F)]        ## recycle
x[c(T, F, T)]     ## no warning!!!

### Some handy numeric functions:

x <- c(0.02345, 0.50000, 0.98765)
round(x)
?round
round(x, digits=2)      ## decimal places
signif(x, digits=2)     ## digits
round(seq(from=-3, to=10, length.out=10))

x <- 3:10
x

sum(x)
prod(x)
cumsum(x)
mean(x)
sd(x)
summary(x)
quantile(x, probs=c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1))

### More vector indexing:

x <- 1:10
length(x)

x[12] <- 12
length(x)
x
x[1000]             ## NA is 'missing value'
x[length(x) + 1] <- length(x) + 1


### More concatenation:

x <- 1:10
x <- c(x, 11)
x
x <- c(x, c(12, 13, 14))
x

x <- 1:10
y <- 21:30
z <- c(x, y, rev(x), rev(y))
z

### More vector operators:

x <- 1:10
length(x)
rev(x)
rev(x)[3]

x <- c(1:5, 4:1, 1:3, 2:1)
x
table(x)
sort(x)
sort(table(x))

x <- c('ab', 'a', 'ac', 'b', 'bd', 'b', 'ab', 'ab')
x
table(x)
sort(x)
sort(table(x), decreasing=T)

x <- 1 : 100
i <- x > 35
table(i)

## QUIZ:

1) how many positive integers, when squared, yield
     a value of more than 200 and less than 3000?

2) generate a vector containing only those positive integers
     (not their squared values! use logical vector index perhaps?)

3) trim off the first 2 values and the last two values

4) add the integers 101, 102, and 103 to the back end of the vector

5) add the integers 1, 2, 3 to the front of the vector

6) reverse the order of the vector

### Simple plotting:

tm <- 1:100
dst <- t0 ^ 2

plot(x=tm, y=dst)

plot(x=tm, y=dst, main="My default plot", xlab="time (s)", ylab="distance (m)")
plot(x=tm, y=dst, main="My dot plot", type="p", xlab="time (s)", ylab="distance (m)")
plot(x=tm, y=dst, main="My line plot", type="l", xlab="time (s)", ylab="distance (m)")
points(x=tm, y=dst)

