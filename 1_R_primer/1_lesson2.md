# Fundamentals of computational data analysis using R
## A primer on programming using R: lesson 2

### Index

- [Names and character indexing](#names-and-character-indexing)
- [Matrices](#matrices)
- [Matrix indexing](#matrix-indexing)
- [Matrix operations](#matrix-operations)
- [Factors](#factors)
- [Lists](#lists)
- [List indexing](#list-indexing)
- [Some useful list operations](#some-useful-list-operations)
- [Data frames](#data-frames)
- [Data frame indexing](#data-frame-indexing)
- [Formulas for plotting and fitting](#formulas-for-plotting-and-fitting)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Names and character indexing

The concatenation command `c()` potentially takes two pre-defined parameters: `recursive` and `use.names`.
  In addition, the command can take user-defined names for each value, where the user-defined names are
  passed in as parameter names and the value is passed as a parameter value. Alternatively, the `names()`
  command can be used to assign names for each value in an already initialized vector. These names provide
  annotation of each value, and can also be used to form an index to retrieve arbitrary subsets of values 
  from the vector.

```
x <- c(a=2, b=4, c=6, d=8, e=10))                       ## 'parameters' a, b, c, d, e; values 2, 4, 6, 8, 10
x
names(x)                                                ## get a character vector of names

x['a']                                                  ## get the element from the position labeled 'a'
x['c']
x['f']

names(x) <- c("duo", "quad", "hex", "oct", "dec")       ## can change the names
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
  arrays are two-dimensional arrays, also known as 'matrices'. As you may imagine, these look like
  the tables you might manipulate in excel. One restriction that matrices have that Excel tables do
  not is that R matrices require every value in the matrix to be of the same type. If you need to mix
  heterogenous types of data, your first choice will probably be data.frames, which are discussed 
  further in the next few sections. When dealing with large data tables containing a single type, 
  however, matrices should be preferred over data.frames, since matrices offer substantial time/space 
  efficiency improvements compared with data.frame operations. Interestingly, matrices in R are 
  implemented by simply adding a 'dim' attribute to a vector. That is, the internal representation 
  continues to be linear, and the 'dim' attribute, which holds the number of rows and columns
  intended for the matrix are used to treat the data as if it were truly two-dimensional. For 
  matrices, remember, the order is always `row, column`. The `dim()` command returns the size of the
  matrix or array. For matrices, this is always the number of rows followed by the number of 
  columns. For indexing matrices, the row index always comes first and the column index comes 
  second (see next section).

```
x <- matrix(1:12, nrow=3)        ## matrices typically numeric; but do not have to be
x                                ## fill by col; number of columns inferred
class(x)
dim(x)                           ## list number of rows, then number of columns
nrow(x)
ncol(x)
attributes(x)                    ## 'dim' is all it takes

x <- matrix(1:12, ncol=4)        ## number of rows inferred
x
dim(x)
attributes(x)

x <- c(x)                        ## 'c()' strips attributes ...
x                                ## ... revealing the vector beneath
class(x)
dim(x)                           ## NULL is 'nothing' (different from NA)

x <- 1:12                        ## just a vector
x
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
  the column number of that value. As with vectors, integer indexing is 1-based (the first 
  value is at index `1`, not `0`).

```
x <- matrix(1:12, nrow=4)
x
x[1, 3]                           ## fetch value from row 1, column 3
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
x <- matrix(1:12, nrow=4)
x
dim(x)

x[c(T, F, T, F), c(F, T, T)]      ## match index lengths to dimension sizes
x[c(F, T, F, T), ]                ## omit one index: selects everything in corresponding dimension
```

Character indices can also be used with matrices and arrays. Instead of using the
  names attribute, however, the labels for each position along each dimension are
  stored in a 'dimnames' attribute. For matrices, these dimnames are best manipulated 
  using the `rownames()` and `colnames()` functions.

```
x <- matrix(1:12, nrow=4)
x
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

```
(x <- matrix(c(1, 2, 3, 1, 2, 1, 2, 3, 1), ncol=3))
x %*% x
solve(x)                 ## invert matrix
(y <- x %*% solve(x))    ## note rounding issue (should be identity)

(x <- matrix(1 : 12, ncol=3))
(y <- matrix(2 * (12 : 1), ncol=3))
x + y
x * y
x / y
x - y

(x <- matrix(1 : 12, ncol=3))
sum(x)
apply(x, 1, sum)
apply(x, 2, sum)

(x <- cbind(1:10, 101:110))
dim(x)

(x <- rbind(1:10, 101:110, 201:210))
dim(x)

(x <- t(x))
dim(x)

(x <- cbind(x, 301:310))
(x <- rbind(x, seq(from=11, to=311, by=100)))
dim(x)
```

[Return to index](#index)

---

### Check your understanding 1

1) build a matrix with the first column the integers from 1 to 10,
     the second column the first 10 even integers, and
     the third column the first 10 odd integers.

2) generate a vector with the product of each row

3) generate a vector with the sum of each column

4) return the second and third columns as a matrix

5) return the second and third rows as a matrix

6) return the second row as a vector

7) return the second row as a (one row) matrix

[Return to index](#index)

---

### Factors

```
(x1 <- c(rep('control', 30)))
(y1 <- rnorm(30, mean=10, sd=3))
(x2 <- c(rep('treated', 30)))
(y2 <- rnorm(30, mean=15, sd=5))

(x <- c(x1, x2))
(x <- factor(x))       ## default level mapping: alphabetical order
class(x)
attributes(x)          ## levels and class
levels(x)
unclass(x)             ## drops $class attribute; result an integer vector!

x <- c(x1, x2)
(x <- factor(x, levels=c('treated', 'control')))
attributes(x)          ## levels and class
levels(x)
unclass(x)             ## drops $class attribute; result an integer vector!

(y <- c(y1, y2))
tapply(y, x, mean)
tapply(y, x, summary)  ## what are those '$' thingies?
tapply(y, x, quantile, probs=c(0.1, 0.25, 0.5, 0.75, 0.9))
```

[Return to index](#index)

---

### Lists

Lists hold vectors of potentially different lengths and types; like structures, dictionaries, and maps.

```
x <- list(
  fname='mitch', 
  lname='kostich', 
  major='body building,
  year.grad=2023, 
  year.birth=1906,
  classes=c('basket weaving', 'chiromancy', 'chainsaw juggling', 'cat psychology'),
  grades=c('C+', 'C-', 'B-', 'D'),
  favorite.foods=c('mango pickle', 'century egg', 'camembert')
)

x
class(x)
attributes(x)              ## just names; 'list' because recursive vector, or vector of vector

str(x)                     ## can use for every structure we have/will discussed
str(1)
str('a')
str(1 : 10)
str(1 : 10000)
```

[Return to index](#index)

---

### List indexing

```
str(x)
x['lname']

x['grades']
class(x['grades'])
x['grades'][2]

x[['grades']]
class(x[['grades']])
x[['grades']][2]

x[7]
class(x[7])
x[7][2]

x[[7]]
class(x[[7]])
x[[7]][2]

x[c('fname', 'grades')]
class(x[c('fname', 'grades')])
x[c('fname', 'grades')][[2]]

x$grades
class(x$grades)
x$grades[2]
```

[Return to index](#index)

---

### Some useful list operations

```
str(x)
length(x)
lapply(x, length)
class(lapply(x, length))

sapply(x, length)
class(sapply(x, length))

sapply(x, length)[3]
sapply(x[c('classes', 'favorite.foods')], length)
```

[Return to index](#index)

---

### Check your understanding 2

[Return to index](#index)

---

### Data frames

Like a list where all elements have same length

```
dat <- data.frame(
  treatment=factor(c(rep('ctl', 10), rep('trt', 10))),
  weight=c(rnorm(10, mean=10, sd=3), rnorm(10, mean=20, sd=5)),
  operator=factor(rep(c('weichun', 'mitch'), 10))
)

dat
class(dat)                ## names, row.names, class (implies equal lengths)
attributes(dat)
unclass(dat)              ## drop $class attribute
class(unclass(dat))       ## show your true self, list!

str(dat)
rownames(dat) <- letters[1 : nrow(dat)]
rownames(dat)
colnames(dat)
dat

dim(dat)
nrow(dat)
ncol(dat)
length(dat)
apply(dat, 1, length)
apply(dat, 2, length)
sapply(dat, length)
```

[Return to index](#index)

---

### Data frame indexing

```
dat
dat[2, 3]
dat[2, 'operator']
dat['b', 'operator']
dat['b', 3]
dat[c('a', 'c'), 2:3]
dat$weight
```

[Return to index](#index)

---

### Check your understanding 3

[Return to index](#index)

---

### Formulas for plotting and fitting

Plot:

```
plot(dat)      ## what do you see?
plot(rock)

par(mfrow=c(1, 2))
plot(weight ~ operator, data=dat)
plot(weight ~ treatment, data=dat)
```

Fit a statistical model to a data.frame:

```
fit1 <- lm(weight ~ treatment, data=dat)
summary(fit1)
```

[Return to index](#index)

---

FIN!


