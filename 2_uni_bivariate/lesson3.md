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
## prep our data for analysis:
rm(list=ls())

?USArrests                          ## find out about a built-in dataset
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
tail(dat)

## set up comparison of Vermont to New York:
x <- dat[c('Vermont', 'New York'), c('Murder', 'Total')]
x
x$Murder
x$Total

```

Now we can do the actual test (easy part):

```
## did arrest rates (proportion per 10,000 or 1e4) differ?
(rslt <- prop.test(x=x$Total, n=c(1e4, 1e4)))
rslt$conf.int                     ## confidence interval on difference in proportion

## did murders as a proportion of arrests differ?
(rslt <- prop.test(x=x$Murder, n=x$Total))
rslt$conf.int                     ## confidence interval on difference in proportion

```

Extend the proportion test to more than 2 groups:

```
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

Some mention of chisq.test().

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Comparing two population means

intro here: also show repeated measures; power trade-off w/ repeated measures

```
## prep our data for analysis:

rm(list=ls())

dat <- mtcars
dat
table(dat$cyl)

(x <- dat$mpg[dat$cyl == 4])
(y <- dat$mpg[dat$cyl == 8])

## two-sided test: confidence interval is on difference in group means;
##   p-value is of h0: difference is zero.

t.test(x=x, y=y)

## one-sided test: h0: difference is less than or equal to zero:
rslt <- t.test(x=x, y=y, alternative='greater')

class(rslt)                       ## h(ypothesis)test
is.list(rslt)                     ## why we can use '$' to index elements
names(rslt)                       ## same old same old
attributes(rslt)

```

Background on repeated measures. Trade-off between degrees of freedom and effect size.

```
## set up data:
rm(list=ls())

dat <- sleep
dat
table(dat$group)

(x <- dat$extra[dat$group == 1])
(y <- dat$extra[dat$group == 2])

## do an unpaired test for comparison:

rslt1 <- t.test(x=x, y=y)

## conduct the paired test:

rslt2 <- t.test(x=x, y=y, paired=T)

## can do one-sided as well:

rslt3 <- t.test(x=x, y=y, paired=T, alternative='less')

c(p1=rslt1$p.value, p2=rslt2$p.value, p3=rslt3$p.value)
cbind(ci1=rslt1$conf.int, ci2=rslt2$conf.int, ci3=rslt3$conf.int)

```

A note on picking tests before examining the data.

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
