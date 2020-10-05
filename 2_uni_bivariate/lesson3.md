# Fundamentals of computational data analysis using R
## Univariate and bivariate statistics: bivariate statistics
#### Contact: mitch.kostich@jax.org

---

### Index

- [Comparing two population proportions](#comparing-two-population-proportions)
- [Comparing two population means](#comparing-two-population-means)
- [Comparing three or more means](#comparing-three-or-more-means)
- [Association between two variables](#association-between-two-variables)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Comparing population proportions

intro here

```
rm(list=ls())

dat <- USArrests                    ## arrests per 100,000 in 1973; by state
class(dat)                          ## data frame
dim(dat)                            ## one row per state
summary(dat)
head(dat)
tail(dat)

dat <- dat * 10                      ## convert to whole numbers; now arrests per 10,000

## total up arrests:
dat$Total <- apply(dat[, c(1, 2, 4)], 1, sum)
summary(dat)
head(dat)

## set up comparison of Vermont to New York:
x <- dat[c('Vermont', 'New York'), c('Murder', 'Total')]
x
x$Murder
x$Total

## did arrest rates (proportion per 10,000 or 1e4) differ?
(rslt <- prop.test(x=x$Total, n=c(1e4, 1e4)))
rslt$conf.int                     ## confidence interval on difference in proportion

## did murders as a proportion of arrests differ?
(rslt <- prop.test(x=x$Murder, n=x$Total))
rslt$conf.int                     ## confidence interval on difference in proportion

## how about when there are more than 2 groups?:

x <- dat[c('Vermont', 'New York', 'California'), c('Murder', 'Total')]
x
x$Murder
x$Total

## did arrest rates (proportion per 10,000 or 1e4) differ AMONG ANY OF THE THREE?
(rslt <- prop.test(x=x$Total, n=c(1e4, 1e4, 1e4)))
str(rslt)                         ## no confidence interval when >2 groups

## did murders as a proportion of arrests differ AMONG ANY OF THE THREE?
(rslt <- prop.test(x=x$Murder, x$Total))
str(rslt)                         ## no confidence interval when >2 groups

```

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Comparing two population means

intro here: also show repeated measures; power trade-off w/ repeated measures

```
rm(list=ls())

dat <- mtcars
dat
table(dat$cyl)

(x <- dat$mpg[dat$cyl == 4])
(y <- dat$mpg[dat$cyl == 8])

t.test(x=x, y=y)
t.test(x=x, y=y, alternative='greater')

```

[Return to index](#index)

---

### Comparing three or more means

intro here

```
code here

```

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Association between two variables

intro here

```
code here

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
