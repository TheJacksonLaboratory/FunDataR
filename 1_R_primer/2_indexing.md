# Fundamentals of computational data analysis using R
## A primer on programming using R: lesson 2
#### Contact: mitch.kostich@jax.org

---

### Index

- [Vector integer indexing](#vector-integer-indexing)
- [Vector logical indexing](#vector-logical-indexing)
- [Vector character indexing](#vector-character-indexing)
- [Matrices](#matrices)
- [Matrix indexing](#matrix-indexing)
- [Matrix operations](#matrix-operations)
- [Factors](#factors)
- [Lists](#lists)
- [List indexing](#list-indexing)
- [Some useful list operations](#some-useful-list-operations)
- [Data frames](#data-frames)
- [Data frame indexing](#data-frame-indexing)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Vector integer indexing

Individual values or arbitrary subsets of values can be retrieved from
  a vector by specifying the position of the desired values using an 
  'index'. The simplest method uses an integer index to specify the 
  position(s) of the desired value(s) in the vector being indexed. The 
  first value in a vector is retrieved with an index of 1 (not 0!!!).
  The index can be of any length.

```
(x <- (1 : 10) ^ 2)            ## outer parentheses prints result of assignment

x[1]                           ## value in first position
x[2]                           ## value in second position
x[3]                           ## value in third position

x[length(x)]                   ## value in last position
x[length(x) - 1]               ## value in next to last position

x[1 : 5]                       ## first five values
x[10 : 6]                      ## similar idea
x[c(1, 10, 3, 5, 3, 10, 1)]    ## arbitrary order, can duplicate

x[-1]                          ## only works for integer index: everything except first
x[-(1 : 5)]                    ## everything except first five

x[seq(from=2, to=10, by=2)]    ## sometimes helps to be creative

```

Indexed vector values are 'lvalues' ('left-values', something that can be on the left-hand
  side of an assignment expression). Simply put, an 'lvalue' is something that a 
  new value can be assigned to. If you assign a value to a position beyond the last position
  of the current vector, that vector will be automatically extended to a length sufficient 
  to accommodate the last position indexed. Retrieving a position beyond the current last 
  position of the vector will return the missing value indicator `NA`.

```
x <- 1 : 10
length(x)

x[12] <- 12
length(x)                            ## automatically extends
x                                    ## NA is 'missing value'

x[1000]                              ## NA is 'missing value'/unknown
x[length(x) + 1] <- length(x) + 1    ## any expression that yields needed index works
x                                    ## NA used to fill in missing values

```

[Return to index](#index)

---

### Vector logical indexing

Another useful indexing approach is to return values meeting some sort of logical 
  criterion. In these cases, we can use a test which returns a logical value for every 
  element in the vector, indicating whether the criterion has been met for that value.
  So a logical index should usually be the same length as the vector being indexed. A 
  logical index that is shorter than the vector being indexed will be recycled without 
  warning.

```
(x <- (1 : 10) ^ 2)
x[x > 30]                            ## get values greater than 30
x[x <= 30]                           ## get values less than or equal to 30
x[x > 30 & x < 70]                   ## combine conditions; '&' NOT '&&'!!!

x <- c('abcder', 'cdefghi', 'e', 'fgabc', 'ghijkla')
x[grepl('abc', x)]                   ## get values that contain 'abc'
x[! grepl('abc', x)]                 ## get values that do not contain 'abc'

x[c(T, F)]                           ## recycle
x[c(T, F, T)]                        ## no warning!!!

```

[Return to index](#index)

---

### Vector character indexing

The concatenation command `c()` optionally takes two pre-defined parameters: `recursive` and `use.names`.
  In addition, the command can take user-defined names for each value, where the user-defined names are
  passed in as parameter names and the value is passed as a parameter value. Alternatively, the `names()`
  command can be used to assign names for each value in an already initialized vector. These names provide
  annotation of each value, and can also be used to form an index to retrieve arbitrary subsets of values 
  from the vector.

```
(x <- c(length=3.7, height=23.2, width=16.3))           ## note: names are not quoted here!
names(x)                                                ## get a character vector of names
x["length"]                                             ## get the element from the position labeled 'length'
x['height']                                             ## note: here the names must be quoted
x['width'] 
x[2]                                                    ## integer indexing still works
x[c(T, F, T)]                                           ## logical indexing still works

x <- seq(from=2, to=10, by=2)
names(x) <- c("duo", "quad", "hex", "oct", "dec")       ## can change names like this
x
x['hex']
x[c('quad', 'single', 'oct')]                           ## can use a vector of names to index

x <- 1:5
names(x) <- c('a', 'b', 'c', 'd', 'e')                  ## can create names for a vector w/o names
x
x[c('b', 'e')]

attributes(x)                                           ## names are stored as 'attributes'
attr(x, "names") <- c("1st", "2d", "3d", "4th", "5th")  ## can change the names by working w/ attributes
attributes(x)
x

x['2d']
x[c('1st', '4th', '5th')]

```

[Return to index](#index)

---

### Matrices

In addition to handling linear one-dimensional arrays of data (vectors), R has good native facilities
  for working with multi-dimensional arrays. For our purposes, the most important multi-dimensional
  arrays are two-dimensional arrays, also known as 'matrices'. In many ways, these look like
  the tables you might manipulate in Excel. One restriction that matrices have that Excel tables do
  not is that R matrices require every value in the matrix to be of the same type. If you need to mix
  heterogenous types of data (like numeric and character), your first choice will probably be 
  data.frames, which are discussed further in the next few sections. When dealing with large data 
  tables containing a single type, however, matrices should be preferred over data.frames, since 
  matrices offer substantial time/space efficiency improvements compared with data.frame operations. 
  Interestingly (at least to geeks like me), matrices in R are implemented by simply adding a 'dim' 
  attribute to a vector. That is, the internal representation continues to be linear, and the 
  'dim' attribute, which holds the number of rows and columns intended for the matrix are used to 
  simulate two-dimensional behavior from the underlying vector. 

For matrices, remember, the order is always `row, column`. The `dim()` command returns the size of 
  the matrix or array. For matrices, this is always the number of rows followed by the number of 
  columns. For indexing matrices, the row index always comes first and the column index comes 
  second (see next section).

You can create matrices from a vector of a compatible length by assigning one of the `nrow` or
  `ncol` parameters to the `matrix()` function. The other parameter is calculated based on the
  vector length and the given parameter:

```
x <- matrix(1:12, nrow=3)        ## matrices typically numeric; but do not have to be
x                                ## fill by col; number of columns inferred
class(x)
dim(x)                           ## list number of rows, then number of columns
nrow(x)
ncol(x)
attributes(x)                    ## 'dim' is all it takes

(x <- matrix(1:12, ncol=4))      ## number of rows inferred
dim(x)
attributes(x)

```

Alternatively, you can simply assign dimensionality to a vector to turn it into a vector:

```
x <- 1 : 12                      ## just a vector
dim(x) <- c(3, 4)                ## easy way to convert vector to matrix
x
class(x)
dim(x)
attributes(x)

```

You can generate matrices with a single initialization value in all cells by specifying the
  initialization value along with the `nrow` and `ncol` parameters. The initialization value
  is 'recycled' to fill a matrix of the specified dimensions:

```
matrix(1, nrow=3, ncol=4)        ## make a matrix with 3 rows and 4 columns, filled with 1s
matrix(0, nrow=4, ncol=3)        ## make a matrix with 4 rows and 3 columns, filled with 0s

x <- matrix(1:12, ncol=4)        ## number of rows inferred
x
x <- c(x)                        ## 'c()' strips attributes ...
x                                ## ... revealing the vector beneath
class(x)
dim(x)                           ## NULL is 'nothing' (different from NA)

```

Here we play with attributes to show how a matrix is 'created' from a vector underneath the hood:

```
x <- 1:12                        ## just a vector
class(x)
dim(x)                           ## NULL is 'nothing' (different from NA)
attr(x, "dim") <- c(3, 4)        ## what if we add the 'dim' attribute manually to a vector?
x
class(x)                         ## works! all it takes is the 'dim' attribute
dim(x)
attributes(x)

```

[Return to index](#index)

---

### Matrix indexing

Just like vectors, matrices (and higher-order arrays) can be indexed using integers. The 
  only difference is that instead of having a single integer between the `[]` brackets,
  one places two integers (for a matrix; the number of integers should equal the number
  of dimensions for higher order arrays) between the `[]` brackets. For matrices, the
  first integer is the row number of the value to be retrieved, and the second integer is
  the column number of that value (rows then columns!). As with vectors, integer indexing 
  is 1-based (the first value is at index `1`, not `0`).

```
x <- matrix(1:12, nrow=4)
x
x[1, 3]                           ## fetch value from first row, column 3
x[4, 2]                           ## fetch value from row 4, column 2 
y <- x[1, ]                       ## missing column index: get the whole row (as vector)
class(x)
class(y)
dim(x)
dim(y)                            ## vectors don't have 'dim' attribute

x[, 2]                            ## missing row index: get whole column (as vector)
x[, -2]                           ## everything except column 2
x[1:2, ]                          ## first two rows
x[1:2, 2:3]                       ## first two rows from columns 2 and 3

y <- x[1, , drop=F]               ## first row, as a one-dimensional matrix (not vector!)
y
class(y)
dim(y)
attributes(y)                     ## keeps 'dim' attribute

y <- x[, 1, drop=F]               ## first column, as 1-D matrix (not vector!)
y
dim(y)                            ## keeps 'dim' attribute

```

Also like vectors, matrices and arrays can be indexed using logical vectors, except 
  instead of using a single indexing vector, you use the same number of vectors as 
  the matrix/array has dimensions. That is, for a matrix (2-D) we would use two logical
  indexing vectors, with the first indicating which rows to select, and the second 
  indicating which columns to select. Keep in mind the normal logical index recycling
  rules: it is usually best to match the logical vector lengths to the corresponding
  dimension size to avoid confusion.

```
(x <- matrix(1:12, nrow=4))       ## enclosing assignment in quotes prints result of the assignment
dim(x)

x[c(T, F, T, F), c(F, T, T)]      ## match index lengths to dimension sizes
x[c(1, 3), c(2, 3)]               ## same
x[c(F, T, F, T), ]                ## omit one index: selects everything in corresponding dimension
x[c(2, 4), ]                      ## same

```

Character indices can also be used with matrices and arrays. Instead of using the
  names attribute, however, the labels for each position along each dimension are
  stored in a 'dimnames' attribute. For matrices, these dimnames are best manipulated 
  using the `rownames()` and `colnames()` functions.

```
(x <- matrix(1:12, nrow=4))
dim(x)

rownames(x) <- c('a', 'b', 'c', 'd')
colnames(x) <- c('1st', '2nd', '3rd')
rownames(x)
colnames(x)

class(x)
dim(x)
attributes(x)                    ## 'dimnames' added to 'dim'
x

x['a', '2nd']
x[c('b', 'd'), c('1st', '2nd')]
x[, '3rd', drop=F]               ## a 1-D array, not a vector

```

[Return to index](#index)

---

### Matrix operations

When matrices are mentioned in the context of mathematics, linear algebra may come to
  mind. In order to make this course more approachable, we are steering away from linear 
  algebra. However, you should be aware that R was originally developed specifically with
  linear algebra in mind, and the base package has many functions for linear algebra. 
  R is handy for statistics and machine learning in large part because R is good at 
  performing the linear algebra calculations that underly many statistical and machine
  learning procedures. Fortunately, this is about all we'll say about linear algebra in 
  this lesson.

```
(x <- c(1, 2, 3, 1, 2, 1, 2, 3, 1))    ## vector
dim(x) <- c(3, 3)                      ## convert to square matrix
x

(y <- x %*% x)                         ## dot-product
crossprod(x, y)                        ## cross-product
solve(x)                               ## inversion of square non-singular matrix

(y <- x %*% solve(x))                  ## note rounding issue (should be identity)

```

More typically, the average user will use operators and functions that operate on single
  matrices or multiple matrices in ways that are similar to the way vector operators and
  functions operate on vectors.

```
(x <- matrix(1 : 12, ncol=3))
(y <- matrix(2 * (12 : 1), ncol=3))
x + y
x * y
x / y
x - y

(x <- matrix(1 : 12, ncol=3))
sum(x)                                 ## treats it like a vector (yields one number)
apply(x, 1, sum)                       ## sum rows (yields vector with one element per row)
apply(x, 2, sum)                       ## sum columns (yields one element per column)

```

There are also a few functions that help you build up matrices from component matrices or
  vectors:

```
(x <- cbind(1:10, 101:110))            ## bind vectors together as columns of a matrix
dim(x)

(x <- rbind(1:10, 101:110, 201:210))   ## bind vectors together as rows of a matrix
dim(x)

(x <- t(x))                            ## transpose: rows to columns; columns to rows
dim(x)

(x <- cbind(x, 301:310))               ## bind vector as column on right side of matrix 'x'
## x <- cbind(301:31)                  ## would bind vector to left side of 'x'

(v <- seq(from=11, to=311, by=100))
(x <- rbind(x, v))                     ## bind vector as row on bottom of matrix 'x'
## x <- rbind(v, x)                    ## would bind vector as row on top of 'x'

dim(x)

```

[Return to index](#index)

---

### Check your understanding 1

1) Build a matrix with the first column the integers from 1 to 10,
     the second column the first 10 even integers, and
     the third column the first 10 odd integers.

2) Generate a vector with one value per row in the matrix from (1), where 
    the value is the product of the all the values in the corresponding row.

3) Generate a vector with the sum of each column of the matrix from (1)

4) Return the second and third columns of the matrix from (1) as a matrix

5) Return the second and third rows of the matrix from (1) as a matrix

6) Return the second row of the matrix from (1) as a vector

7) Return the second row of the matrix from (1) as a (one row) matrix

[Return to index](#index)

---

### Factors

Another data type that is derived from an integer vector by adding a couple attributes
  ('levels' and 'class') is the factor type. Factors are easily mistaken for the character
  data type, and they can both be used to represent categorical variables. However, 
  factors use an internal integer representation for group membership which is often 
  more efficient than using character representations of the same information. Factors 
  are usually created from a vector of character group labels. By default, the mapping 
  of character labels to integer values is based on alphabetic ordering of the labels, 
  with the first label being assigned to the integer 1, the second to 2, etc. This mapping 
  can be changed by explicitly passing a `levels` parameter to the `factor()` command, or 
  by using the `levels()` function.

We use the `set.seed()` function to seed any pseudo-random process in R, if we want to make that
  process reproducible. The underlying generators are not truly random in R, but utilize
  algorithms that operate on one number (the seed) that is used to generate another number
  (the random number for this round) which is used as the seed for the next call of the 
  generator. Therefore, if you initially seed a random number generator and get 1000 values
  from it, then repeat the experiment days later with the same seed, you will get exactly
  the same 1000 values in exactly the same order. If you want your work to be reproducible,
  seed every series of pseudo-random steps, document the seed used in your notebook and 
  report that seed along with versions of the softare used for pseudo-random processes 
  within the methods section of any associated scientific publications. R has a series of
  functions to generate random values from a specified distribution:

```
sessionInfo()                     ## get version information for current R setup

## perform your 'experiment':
set.seed(7)                       ## 'seed' pseudo-random number generator to make reproducible
rnorm(1, mean=1, sd=1)            ## get 1 random value from normal distribution
runif(2, min=0, max=1)            ## get 2 random value from a uniform distribution on interval [0, 1)
rbinom(3, size=1, prob=0.5)       ## get 3 coin flips (1=heads, 0=tails)
rpois(4, lambda=10)               ## get 4 random count from Poisson process with avg rate of 10 per

## repeat your 'experiment':
set.seed(7)                       ## 'seed' pseudo-random number generator to make reproducible
rnorm(1, mean=1, sd=1)            ## get 1 random value from normal distribution
runif(2, min=0, max=1)            ## get 2 random value from a uniform distribution on interval [0, 1)
rbinom(3, size=1, prob=0.5)       ## get 3 coin flips (1=heads, 0=tails)
rpois(4, lambda=10)               ## get 4 random count from Poisson process with avg rate of 10 per

```

Below we simulate a dataset with two groups ('control' and 'treated'), each with 
  30 observations. For the 'control' group, we simulate values by (pseudo-)random
  draws from a normal (a.k.a Guassian) distribution with a mean of 10 and a 
  standard deviation of 3. For the 'treated' group, we simulate values by drawing
  from a normal distribution with a mean of 15 and a standard deviation of 5.
  Normal distributions can be fully specified using just two parameters, 
  `mean` and `sd`. The `rnorm()` function is used to generate random values from
  a normal distribution.

```
set.seed(21)                      ## make reproducible
(x1 <- rep('control', 30))        ## make up one group of 30
(y1 <- rnorm(30, mean=10, sd=3))  ## simulate values for each member of the first group
(x2 <- rep('treated', 30))        ## make up a second group of 30
(y2 <- rnorm(30, mean=15, sd=5))  ## simulate values for the second group

(x <- c(x1, x2))                  ## group 1, followed by group 2
(x <- factor(x))                  ## default level mapping: alphabetical order
table(x)
class(x)
levels(x)                         ## first maps to 1, second to 2, etc.
attributes(x)                     ## 'levels' and 'class'
unclass(x)                        ## drops $class attribute; result is an integer vector!

## can specify the order of mapping of labels to integer values:
x <- c(x1, x2)
(x <- factor(x, levels=c('treated', 'control')))
attributes(x)                     ## 'levels' and 'class'
levels(x)                         ## first maps to 1, second to 2, etc; user prescribed order
unclass(x)                        ## drops $class attribute; result is an integer vector!

## some superficial group comparisons:
(y <- c(y1, y2))
tapply(y, x, mean)
tapply(y, x, sd)
tapply(y, x, summary)             ## what are those '$' thingies?
tapply(y, x, quantile, probs=c(0.1, 0.25, 0.5, 0.75, 0.9))

```

[Return to index](#index)

---

### Lists

Lists are the most flexible basic structure provided by R. Lists are similar to
  dictionaries, maps, or structures used in other languages. Lists are collections of 
  vectors, each with a potentially different length and type from other list member
  vectors.

To get a quick idea of what is within a potentially more complicated data type (like list),
  it is often convenient to use the `str()` command, which works on all the data types 
  we will discuss in this course.

```
str(1)                     ## works on numeric 'scalar'
str('a')                   ## works on character
str(c(T, F, T))            ## works on logical
str(1 : 10)                ## works on longer vectors
str(1 : 10000)             ## as vector gets even longer, shows synopsis

x <- list(
  fname='mitch', 
  lname='kostich', 
  major='undecided',
  year.grad=2023, 
  year.birth=1906,
  classes=c('basket weaving', 'chiromancy', 'chainsaw juggling', 'cat psychology'),
  grades=c('B+', 'C-', 'D+', 'F'),
  favorite.foods=c('mango pickle', 'century egg', 'camembert')
)

x
class(x)
attributes(x)              ## just names; 'list' because recursive vector, or vector of vector
str(x)                     ## works on lists

```

[Return to index](#index)

---

### List indexing

Like the other data types we've discussed, lists can be indexed using integers,
  character values, or logical vectors.

```
x <- list(
  fname='mitch', 
  lname='kostich', 
  major='undecided',
  year.grad=2023, 
  year.birth=1906,
  classes=c('basket weaving', 'chiromancy', 'chainsaw juggling', 'cat psychology'),
  grades=c('C+', 'C-', 'B-', 'D'),
  favorite.foods=c('mango pickle', 'century egg', 'camembert')
)

str(x)
x['lname']
x[c(T, T, F, F, T, T, F, F)]      ## logical indexing of lists works too

x['grades']                       ## what is that '$' thingy? indicates element of a list!
class(x['grades'])                ## got a list back with single bracket
x['grades'][2]                    ## oops! cannot use a single bracket indexing on result

x[['grades']]                     ## no '$' thingy
class(x[['grades']])              ## just a vector!
x[['grades']][2]                  ## can index the result with single bracket (since vector)

x[7]                              ## integer indexing with 1x bracket: same issue
class(x[7])                       ## list
x[7][2]                           ## oops! cannot use 1x bracket indexing on result

x[[7]]                            ## double bracket saves the day again!
class(x[[7]])                     ## vector
x[[7]][2]                         ## no problem

x[[c('fname', 'grades')]]         ## oops! can only use one name with 2x bracket

x[c('fname', 'grades')]           ## 1x brackets useful for selecting >1 element
class(x[c('fname', 'grades')])    ## still a list, but only subset of elements
x[c('fname', 'grades')][[2]]      ## index result like a list

## an easy way to specify a list element (no quotes, unless embedded spaces):
x$grades                          ## only 1 value at a time, but easy typing!
x$"grades"                        ## quotes optional
class(x$grades)                   ## vector
x$grades[2]                       ## normal vector indexing of result works

```

[Return to index](#index)

---

### Some useful list operations

The most useful operations specifically intended for lists are the commands
  `lapply()` and `sapply()`. They both apply a user-specified function to each 
  element of the list. They differ in that `lapply(x)` always returns a list with 
  one element per element in the list `x`, while `sapply()` tries to return
  a vector (with one element per list element), if possible. For functions which 
  predictably return values of the same length for any valid input, `sapply()` 
  will usually succeed in returning a vector, while functions with more complicated 
  return values may result in a list to be returned by `sapply()`, identical to 
  that which would be returned by `lapply()`.

```
x <- list(
  fname='mitch', 
  lname='kostich', 
  major='undecided',
  year.grad=2023, 
  year.birth=1906,
  classes=c('basket weaving', 'chiromancy', 'chainsaw juggling', 'cat psychology'),
  grades=c('C+', 'C-', 'B-', 'D'),
  favorite.foods=c('mango pickle', 'century egg', 'camembert')
)

str(x)
length(x)                         ## length of list (how many elements at outer level)
lapply(x, length)                 ## length of each element of list
class(lapply(x, length))          ## a list!

sapply(x, length)                 ## try to make the result a vector instead
class(sapply(x, length))          ## yields vector for length()
sapply(x, length)[3]              ## so can index result like vector

sapply(x[c('classes', 'favorite.foods')], length)

```

[Return to index](#index)

---

### Check your understanding 2

1) Make a list with four member elements, including one logical (length=1), 
     one integer (length=1), one numeric (length=3), and one character element (length=5).

2) Retrieve the second element from your list as a list.

3) Retrieve the third element from your list as a vector.

4) Generate a vector with the class of each element of your list.

5) Generate a list with the lengths of the 2d (integer) and 4th (character) elements of your list.

[Return to index](#index)

---

### Data frames

The data.frame type is a particularly important type for data analysis. This type is similar 
  to a list in that each element can be of a different type. However, it differs in that each 
  element of a data.frame is required to be of the same length as every other element. This 
  means that the data.frame can be conveniently displayed as a two-dimensional table with a 
  value in every cell. Each column (data.frame element) of this table can be of a different 
  type, but all the values within any one column must be of the same type. This is very 
  similar to how you might lay out a data table in Excel: you can have each column represent 
  a variable of interest (e.g. group, treatment intensity, outcome measurent), and each row 
  represent an individual observation of these variables (e.g. a subject in an experiment).

```
dat <- data.frame(
  treatment=factor(c(rep('ctl', 10), rep('trt', 10))),
  weight=c(rnorm(10, mean=10, sd=3), rnorm(10, mean=20, sd=5)),
  operator=factor(rep(c('weichun', 'mitch'), 10))
)

dat                               ## rows are observations, columns are variables
class(dat)                        ## names, row.names, class (implies equal lengths)
attributes(dat)                   ## dimensions from lengths of 'names' and 'row.names'
unclass(dat)                      ## drop $class attribute
class(unclass(dat))               ## show your true self, list!

str(dat)

## don't try if more rows than letters! 
##   usually use meaningful row.names, such as observation labels:

rownames(dat) <- letters[1 : nrow(dat)]
rownames(dat)
colnames(dat)
dat

dim(dat)                         ## just like matrix (figured out from names and row.names)
nrow(dat)                        ## just like matrix
ncol(dat)                        ## just like matrix
length(dat)                      ## just like list

apply(dat, 1, length)            ## just like matrix
apply(dat, 2, length)            ## just like matrix
sapply(dat, length)              ## just like list

```

[Return to index](#index)

---

### Data frame indexing

Data frames can be indexed using either the methods used for lists or the methods 
  used for matrices. Character indexing of rows depends on assignment of 
  `row.names` attribute (e.g. using the `rownames()` function). Rownames are 
  not required to be unique, but your code will be clearer if 
  you always make sure `names` and `row.names` elements are unique.

```
dat <- data.frame(
  treatment=factor(c(rep('ctl', 10), rep('trt', 10))),
  weight=c(rnorm(10, mean=10, sd=3), rnorm(10, mean=20, sd=5)),
  operator=factor(rep(c('weichun', 'mitch'), 10))
)
rownames(dat) <- letters[1 : nrow(dat)]
dat

dat[2, 3]                         ## index like a matrix (integers)
dat[2, 'operator']                ## index like a matrix (integer + character)
dat['b', 'operator']              ## index like a matrix
dat['b', 3]                       ## index like a matrix
dat[c('a', 'c'), 2:3]             ## index like a matrix
dat[c('a', 'c'), c(F, T, T)]      ## index like a matrix

## when pick one row (values of potentially different types):
dat[3, ]                          ## index like a matrix
attributes(dat[3, ])              ## BUT: still a data.frame
class(dat[3, ])
dim(dat[3, ])                     ## so still treated as 2-dimensions

## however, when pick one column (values all same type):
dat[, 2]                          ## index like a matrix
attributes(dat[, 2])
class(dat[, 2])
dim(dat[, 2])

## use 'drop=F' to preserve attributes:
dat[, 2, drop=F]
attributes(dat[, 2, drop=F])
class(dat[, 2, drop=F])

## another way to pick a column by name:
dat$weight                        ## index like a list: yields vector
attributes(dat$weight)            ## attributes gone
class(dat$weight)                 ## vector indeed

```

[Return to index](#index)

---

### Check your understanding 3

1) Make a `data.frame` with 5 rows and 4 columns, where the first column
     is `logical`, second is `integer`, third is `numeric` and last is
     `character`. Name the columns as you like.

2) Add some `row.names` to the `data.frame` from (1).

3) Pull the 2d and 3d columns from the 1st and 3d row with a 
     single indexing operation using a character index and 
     a logical index.

4) Using `sapply()`, determine the data type of each column.

5) Using `apply()`, determine the data type of each column.

[Return to index](#index)

---

## FIN!


