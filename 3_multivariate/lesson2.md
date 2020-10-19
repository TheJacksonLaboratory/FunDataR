# Fundamentals of computational data analysis using R
## Multivariate statistics: computational alternatives to parametric statistics
#### Contact: mitch.kostich@jax.org

---

### Index

- [Permutation testing](#permutation-testing)
- [Empirical boostrap](#empirical-bootstrap)
- [Cross-validation](#cross-validation)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Permutation testing

intro here; null tied to exchangeability; randomization vs. permutation testing;
  randomization: outcomes for one observation not dependent on treatment or
  ourcome of other observations
  permutation: 

random assignment to treatment groups leads to one set of assumptions
non-random assignment requires further justification for exchangeability assumption.

t-test

sampling without replacement...

```
## two-sample equal-variances t-test:

rm(list=ls())
set.seed(1)

dat <- mtcars
dat <- dat[dat$cyl %in% c(4, 8), c('mpg', 'cyl')]
(x <- dat$mpg[dat$cyl == 4])
(y <- dat$mpg[dat$cyl == 8])

(fit <- t.test(x=x, y=y, var.equal=T))
fit$statistic

## but neither dataset is large or normally distributed:
par(mfrow=c(1, 2))
qqnorm(x, main='x')
qqline(x)
qqnorm(y, main='y')
qqline(y)

## however homogeneous enough to try permutation as check on parametrics:
bartlett.test(mpg ~ cyl, data=dat)

## set up permutation:

R <- 9999
rslts <- rep(as.numeric(NA), R)

for(i in 1 : R) {
  dat.i <- dat
  dat.i$cyl <- sample(dat.i$cyl, nrow(dat.i), replace=F)
  x.i <- dat.i$mpg[dat.i$cyl == 4]
  y.i <- dat.i$mpg[dat.i$cyl == 8]
  fit.i <- t.test(x=x.i, y=y.i, var.equal=T)
  rslts[i] <- fit.i$statistic
}

x                                 ## original values for 1st group
y                                 ## original values for 2d group
x.i                               ## same length as x, but values from both grps
y.i                               ## same length as y, but values from both grps

fit$statistic
summary(rslts)
(n.exceed <- sum(abs(rslts) >= abs(fit$statistic)))
(n.exceed + 1) / (R + 1)          ## formula for p-value (should never yield 0!!!)
fit$p.value                       ## compare to parametric (finer grained)

```

Simple linear regression or other models with a single predictor (including polynomial
  terms and multiple levels for a factor predictor):

Can use lmperm() or coin().

```
## p-value on coefficient from lm():

rm(list=ls())
set.seed(1)

fit <- lm(dist ~ speed, data=cars)
par(mfrow=c(2, 3))
plot(fit, which=1:6)
## not very normal, but looks reasonably homoskedastic, so try permutation
##   for checking parametric p-value

R <- 9999
rslts <- rep(as.numeric(NA), R)

for(i in 1 : R) {
  dat.i <- cars
  dat.i$speed <- sample(dat.i$speed, nrow(dat.i), replace=F)
  fit.i <- lm(dist ~ speed, data=dat.i)
  rslts[i] <- coef(summary(fit.i))['speed', 't value']
}

## p-value on coefficient from original fit:
(stat <- coef(summary(fit))['speed', 't value'])

summary(rslts)
(n.exceed <- sum(abs(rslts) >= abs(stat)))
(n.exceed + 1) / (R + 1)                           ## permutation p-value
(stat <- coef(summary(fit))['speed', 'Pr(>|t|)'])  ## compare to parametric

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Empirical boostrap

Confidence interval on single mean.

```
## CI on single mean
library(boot)
sessionInfo()

rm(list=ls())
set.seed(1)

dat <- iris[iris$Species == 'virginica', ]
dat
par(mfrow=c(1, 1))
qqnorm(dat$Sepal.Length)
qqline(dat$Sepal.Length)

(fit1 <- t.test(dat$Sepal.Length))

## function needs to take original data as first argument,
##   and integer index of observations in bootstrap sample
##   (generated and passed by boot()) as the second argument.
##   It then needs to split the data based on the index and
##   compute + return the statistic of interest:

f <- function(dat, idx) {
  mean(dat[idx, 'Sepal.Length'], na.rm=T)
}

R <- 9999
out <- boot(dat, f, R)
class(out)
is.list(out)
attributes(out)

out                               ## note estimated bias about zero
plot(out)
jack.after.boot(out)

out$t0
f(dat, T)
length(out$t)
summary(out$t)
head(out$t)

ci <- boot.ci(out)
class(ci)
is.list(ci)
attributes(ci)

ci
fit1

```

CI and bias for variance.

```
## CI and bias for variance.

library('boot')
rm(list=ls())
set.seed(1)

n <- 15
x <- iris[iris$Species == 'virginica', 'Sepal.Length']
x <- sample(x, n, replace=F)

f.var.pop <- function(x) {
  m <- mean(x, na.rm=T)
  mean((x - m) ^ 2, na.rm=T)
}

var(x)                            ## sample formula for variance (unbiased)
f.var.pop(x)                      ## population formula (biased)

f <- function(x, i) {
  f.var.pop(x[i])
}

R <- 9999
out <- boot(x, f, R)

out                               ## note bias
plot(out)
jack.after.boot(out)

(bias <- out$t0 - mean(out$t, na.rm=T))
(est <- out$t0 + bias)
var(x)
f.var.pop(x)

(ci <- boot.ci(out))

```

CI for lm() coefficient.

```
## CI for lm() coefficient.

libary(boot)
rm(list=ls())
set.seed(1)

par(mfrow=c(1, 1))
plot(cars)

f <- function(dat, i) {
  fit.i <- lm(dist ~ speed, data=dat[i, ])
  coef(fit.i)['speed']
}

fit <- lm(dist ~ speed, data=cars)
coef(fit)['speed']
f(cars, T)

R <- 999
out <- boot(cars, f, R)
plot(out)
jack.after.boot(out)
(ci <- boot.ci(out))

confint(fit)['speed', ]

```

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Cross-validation

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
