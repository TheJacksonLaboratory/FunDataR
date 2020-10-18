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

intro here; null tied to exchangeability

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

linear model

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

```

CI and bias for variance.

```
## CI and bias for variance.

```

CI for lm() coefficient.

```
## CI for lm() coefficient.

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
