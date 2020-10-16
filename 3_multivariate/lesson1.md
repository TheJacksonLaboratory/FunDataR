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
  samples) and homogeneity of variances. Also, can improve linearity
  of the relationship. 

Variance stabilizing transformations. 
  for strictly positive data: log(y); sqrt(y)
  for positive data: log(y+1); sqrt(y)

arcsine proportions; square-root counts: better yet, use GLMs!!!

log(x) is particularly popular as it tends to both stabilize the variance
  and improve the linearity of causal processes where the independent
  variable has an effect on the dependent variable that is proportional
  to the size of the dependent variable. For instance, if increasing
  `x` one unit increases the corresponding value of `y` by `1%`, the 
  size of the increase will depend on the initial value of `y`, rather
  than simply being a constant.

For strictly positive continuous data: box-cox: finds exponent of `y` 
  `lambda` such that makes `y ^ lambda` look as close as possible to 
  what we would expect for a variable that is linearly related to some
  other variable and has normally distributed errors with a mean of 
  zero and constant standard deviation. There are several criteria here
  being improved. In many cases, the main effect of the change is to
  make distributions of residuals more homoskedastic and symmetrically
  distributed, but often will still not appear quite normal. By default, 
  the R `MASS::boxcox()` function searches for an optimal `lambda` 
  exponent in the range of `-2` to `2` by default, where a `lambda` 
  exponent of `0` is treated as the natural log transformation `log(y)`. 
  Typically, if `1` falls within the given 95% confidence interval, 
  there is not much point to transformation. If `1` is not within the 
  confidence interval, try to pick one of the values {-2, -1, -1/2, 0, 
  1/2, 1, 2} if it falls within the interval, as it makes interpretation 
  simpler. That is, we know that `y ^ 1` means no transformation, 
  `y ^ -1` means taking the reciprocal, `y ^ (1/2)` is a square-root 
  transformation, `lambda == 0` means the same thing as `log(y)`, 
  `y ^ 2` means squaring, and `y ^ -2` means squaring then taking the 
  reciprocal. By contrast, `y ^ 1.87` is a far less familiar/meaningful 
  transformation, even though it has a precise mathematical definition.
  You may gain some slight improvement in the appearance of the residual
  plots, but at the expense of substantially complicating model 
  intepretation.

Even when result is not exactly normal, will tend to at least be 
  symmetric and homoskedastic, which reduces the sample sizes needed for CLT 
  approximations to kick in.

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

to linearize the relationship; other than log(y), rarely transform 
  y for linearization, more for meeting assumptions about residual
  distribution. the most popular 
  way to linearize the relationship is transformation of x.
  Often an iterative process guided by residual plots. In one
  dimension x vs y plot can be very helpful, but in multiple 
  regression, really depend on residual plots.

polynomials/sqrt/log/exp/reciprocal are common. Often motivated by theory.

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
table(DNase$Run)

## hold out two whole runs (independent experiments) for testing:

unique(DNase$Run)
(run.tst <- sample(unique(DNase$Run), 2, replace=F))
i.tst <- DNase$Run %in% run.tst
dat.tst <- DNase[i.tst, ]
dat.trn <- DNase[! i.tst, ]
summary(dat.tst)
summary(dat.trn)

## plot the observations:

plot(density ~ conc, data=dat.trn, pch='+', col='black')
abline(h=mean(dat.trn$density), lty=4, col='black')
points(density ~ conc, data=dat.tst, pch='o', col='orangered')

## fit a simple linear regression model to training data:

(fit1 <- lm(density ~ conc, data=dat.trn))
(smry1 <- summary(fit1))
abline(fit1, lty=2, col='magenta')

## fit a local regression model to training data:

(fit2 <- loess(density ~ conc, data=dat.trn))
class(fit2)
is.list(fit2)
names(fit2)
attributes(fit2)

dat.plot <- data.frame(conc=seq(from=0, to=13, by=0.01))
dat.plot$density <- predict(fit2, newdata=dat.plot)
lines(density ~ conc, data=dat.plot, lty=3, col='cyan')

## point estimates of relative performance:

pred0 <- rep(mean(dat.trn$density), nrow(dat.tst))
pred1 <- predict(fit1, newdata=dat.tst)
pred2 <- predict(fit2, newdata=dat.tst)

f.mse <- function(obs, pred) {
  if(! (is.numeric(obs) && is.numeric(pred)) )
    stop("obs and pred must be numeric")

  if(length(obs) != length(pred))
    stop("obs and pred must be same length")

  if(length(obs) == 0) return(NaN)

  mean((obs - pred) ^ 2)
}

f.mse(dat.tst$density, pred0)
f.mse(dat.tst$density, pred1)
f.mse(dat.tst$density, pred2)

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
