# Fundamentals of computational data analysis using R
## Multivariate statistics: more on simple regression
#### Contact: mitch.kostich@jax.org

---

### Index

- [Transforming the response](#transforming-the-response)
- [Transforming predictors](#transforming-predictors)
- [Local regression](#local-regression)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### title 1

Transforming the response

To achieve normality of the response (and therefore residuals in small 
  samples) and homogeneity of variances.

Variance stabilizing transformations. 
  for strictly positive data: log(y); sqrt(y)
  for positive data: log(y+1); sqrt(y)

For strictly positive continuous data: box-cox: finds exponent of `y` 
  `lambda` such that makes `y ^ lambda` look as close as possible to 
  what we would expect for random draws from a normal distribution 
  with a zero mean and constant standard deviation. By default, the
  R `MASS::boxcox()` function searches for an optimal `lambda` exponent in
  the range of `-2` to `2` by default, where a `lambda` exponent of 
  `0` is treated as `log(y)`. Typically, if `1` falls within the given
  95% confidence interval, there is not much point to transformation.
  If `1` is not within the confidence interval, try to pick a whole 
  number that falls within the interval, as it makes interpretation 
  simpler. That is, we know that `y ^ 1` means no transformation,
  `y ^ -1` means taking the reciprocal, `lambda == 0` means the same
  thing as `log(y)`, `y ^ 2` means squaring, and `y ^ -2` means 
  squaring then taking the reciprocal. By contrast, `y ^ 1.87`
  is a far less familiar/meaningful transformation, even though it
  has a precise mathematical definition.

```
library('MASS')                   ## included in most R distros; provides boxcox()
sessionInfo()                     ## version info

rm(list=ls())

fit1 <- lm(Volume ~ Height, data=trees)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

par(mfrow=c(1, 1))
boxcox(fit1, plotit=T)

## since the confidence interval does not include 1, but does include 0, which 
##   corresponds to the simple to interpret log(y),:

fit2 <- lm(log(Volume) ~ Height, data=trees)
par(mfrow=c(2, 3))
plot(fit2, which=1:6)

dev.new()                         ## open a new plotting window
par(mfrow=c(2, 3))
plot(fit1, which=1:6)
dev.off()                         ## close the extra plotting window

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Transforming predictors

intro here

polynomials/sqrt/log are common. Often motivated by theory.

```
rm(list=ls())
set.seed(1)

## configure simulation:
n <- 100                          ## sample size
p.tst <- 0.2                      ## proportion for test set
n.tst <- round(p.tst * n)         ## test set size

## simulate a sample:

tm <- runif(n, min=0, max=n)      ## time
accel <- 3                        ## acceleration
err <- rnorm(length(tm), mean=0, sd=150)
y.init <- 200                     ## initial velocity
y <- y.init + 0.5 * accel * (tm ^ 2) + err
dat <- data.frame(y=y, tm=tm)

## split sample in test and training sets:

idx.tst <- sample(1 : n, n.tst, replace=F)
i.tst <- rep(F, n)
i.tst[idx.tst] <- T
i.trn <- ! i.tst

dat.tst <- dat[i.tst, ]
dat.trn <- dat[i.trn, ]

nrow(dat.tst)
nrow(dat.trn)

## plot the training data:
par(mfrow=c(1, 1))
plot(y ~ tm, data=dat.trn)

## fit a line to the training data:
fit1 <- lm(y ~ tm, data=dat.trn)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

## residuals vs. fitted clearly suggests second order (squared)
##   but residuals look homoskedastic, so don't want to mess 
##   with Weight.

fit2 <- lm(y ~ I(tm ^ 2), data=dat.trn)
par(mfrow=c(2, 3))
plot(fit2, which=1:6)

## lets do a more objective comparison; predict values
##   for test set using both models:

pred1 <- predict(fit1, newdata=dat.tst)
pred2 <- predict(fit2, newdata=dat.tst)

## our mean-squared error function:

f.mse <- function(y, y.hat) {

  if(! (is.numeric(y) && is.numeric(y.hat)) )
    stop("y and y.hat must be numeric")

  if(length(y) != length(y.hat))
    stop("y and y.hat must be same length")

  if(length(y) == 0) return(NaN)

  mean((y - y.hat) ^ 2)
}

## second model appears to have WAY less error:

f.mse(dat.tst$y, pred1)
f.mse(dat.tst$y, pred2)

```

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Local regression

intro here

```
rm(list=ls())
set.seed(1)

summary(DNase)
plot(DNase)

dat <- DNase


```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
