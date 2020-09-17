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
- [Data frames](#data frames)
- [Data frame indexing](#data-frame-indexing)
- [Formulas for plotting and fitting](#formulas-for-plotting-and-fitting)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Names and character indexing

```
x <- c(a=2, b=4, c=6, d=8, e=10))
x
names(x)
x['a']
x['c']
x['f']

names(x) <- c("duo", "quad", "hex", "oct", "dec")
x
x['hex']
x[c('quad', 'single', 'oct')]

x <- 1:5
names(x) <- c('a', 'b', 'c', 'd', 'e')
x
x[c('b', 'e')]

attributes(x)
attr(x, "names") <- c("1st", "2d", "3d", "4th", "5th")
attributes(x)
x

x['2d']
x[c('1st', '4th', '5th')]
```

[Return to index](#index)

---

### Matrices

```
x <- matrix(1:12, nrow=3)        ## typically numeric
x                                ## fill by col
class(x)
dim(x)
nrow(x)
ncol(x)
attributes(x)

x <- matrix(1:12, ncol=4)
x
dim(x)                           ## NULL is 'nothing' (different from NA)

x <- c(x)                        ## strips attributes
x
class(x)
dim(x)
attributes(x)

x <- 1:12
class(x)
dim(x)
x
attr(x, "dim") <- c(3, 4)
x
class(x)
dim(x)
attributes(x)
```

[Return to index](#index)

---

### Matrix indexing

Matrix integer vector indexing:

```
x <- matrix(1:12, nrow=4)
x
x[1, 3]
x[4, 2]
y <- x[1, ]
class(x)
class(y)
dim(x)
dim(y)

x[, 2]
x[, -2]
x[1:2, ]
x[1:2, 2:3]

y <- x[1, , drop=F]
y
class(y)
dim(y)
attributes(y)

y <- x[, 1, drop=F]
y
dim(y)
```

Matrix logical vector indexing:

```
x[c(T, F, T, F), c(F, T, T)]
x[c(F, T, F, T), ]
```

Matrix character vector indexing:

```
rownames(x) <- c('a', 'b', 'c', 'd')
colnames(x) <- c('1st', '2nd', '3rd')
rownames(x)
colnames(x)

class(x)
dim(x)
attributes(x)
x

x['a', '2nd']
x[c('b', 'd'), c('1st', '2nd')]
x[, '3rd', drop=F]
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

### Factors:

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

### Lists: hold vectors of potentially different lengths and types; like structures, dictionaries, and maps.

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

### List indexing:

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


