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

### Transforming the response

The assumptions behind a linear model can be divided into assumptions
  about the form of the relationship (a line) and assumptions about the
  error term. Assumptions behind the error term justify use of parametric
  distributions for estimating p-values and confidence intervals on
  coefficients, as well as prediction intervals for the response variable
  for new observations. The error model assumptions in small sample sizes 
  are that the errors are randomly drawn from a normal distribution with 
  a mean of zero and constant standard deviation. For larger samples,
  we make the more relaxed assumpton that errors are randomly drawn from
  the same distribution with constant finite standard deviation. The
  size of the sample required for the relaxed assumption to kick in is
  dependent on the actual distribution: the closer to normal it is, 
  the smaller samples are sufficient. For very skewed distributions, 
  larger samples are required. The rule-of-thumb is that about 30 
  observations are required in the case of simple linear regression.

One way to address deviations from the assumptions of the linear model
  is to transform the variables. We can transform either the response
  variable or the predictor variables, or both. This is often an iterative
  process that is guided by the appearance of the residual plots. 
  Some rules of thumb may be helpful: if the fit appears to be non-linear
  (usually signalled by a non-linear systematic trend in the 'Residuals
  vs fitted' plot), but the error assumptions appear reasonable, then
  first try to transform the predictors. On the other hand, if the 
  shape of the fit looks reasonable, but the error assumptions appear to
  be violated, try to transform the response variable. If both sets of
  assumptions appear to be incorrect, try first transforming the response. 
  If this does not fix the issue try transforming the predictors (one at 
  a time if there are several) as well. It is worth keeping in mind that 
  for some data sets, no monotonic transformation will result in both
  sets of assumptions holding. Sometimes you will find that as you improve
  linearity, you make residuals look worse in other respects, and 
  vice-versa. In these cases alternatives to the linear model with constant
  variance should be considered. We will discuss some options later in
  the course.

The transformations used are always monotonic. This makes the transformation
  reversible, which is critical for making predictions on the original
  scale of the response variable. The transformations are usually some type 
  of 'power function', that is they can be expressed as putting an exponent 
  `lambda` on the original variable `y.trans == y ^ lambda`. Common 
  transformations include the square-root `y ^ 0.5`, square `y ^ 2`, 
  reciprocal `y ^ -1`, the natural log `log(y)`, the inverse of the natural 
  log `exp(y)`, and the arcsine transformation `asin(sqrt(y))`. In the past, 
  it was common for analysts to transform count data (which tend to be 
  otherwise heteroskedastic: the standard deviation increases with the
  magnitude of the count) with the square-root transformation in order 
  to make the variance more homogeneous (reduce the heteroskedasticity)
  which made it possible to analyze these data with linear models. Similarly,
  it was common to transform proportion data using the arcsine 
  transformation to reduce heteroskedasticity and help linearize the
  relationship with predictors. However, in the modern era we typically
  use generalized linear models (presented later in this course) to deal
  with counts and proportions.

The `log(y)` transformation is still widely used however, as it seems
  to attenuate both heteroskedasticity and non-linearity seen in the
  relationships between variables in many natural and complex man-made
  systems. In particular, it is often used when data are strictly
  positive (no zeros or negative values), and the variant `log(y + 1)`
  is often used when data are non-negative (restricted to values
  that are positive or zero). The reasons why this transformation 
  works well on so many data sets is because it changes multiplicative
  relationships into additive ones. That is, 
  `log(x * y) == log(x) + log(y)`. If there is a causal relationship
  between `x` and `y` that is additive, that is a one unit change
  in `x` always results in the same change in `y`, regardless of the
  initial value of `y`, is equivalent to a linear relationship 
  between the two variables. However, if a one unit change in `x` 
  has an effect on `y` that is proportional to `y`, this means that
  the same size change in `x` will have a smaller impact on small
  `y` than on large `y`. This will cause the relationship to be 
  non-linear, and typically also cause larger dispersion around 
  the prediction curve at larger values of `y` than for smaller
  values. These proportional multiplicative effects are common in 
  nature: adding a fertilizer might make all the plants grow by an
  additional 10% within a given test period. This means that in
  absolute terms, large plants will see more gain in size than 
  small ones. The relationship is multiplicative/non-linear, and we
  should not be surprised that the spread in size of large plants
  is larger than for small ones. Transforming these data by taking 
  the log of the size will transform the relationship into a more
  linear one as well as attenuate much of the heteroskedasticity.

For strictly positive continuous data (there are less popular extensions
  that are more general), there is a systematic method for exploring
  the effects of a continuous range of power transformations on the 
  response variable. Within the `MASS` package (included in most R
  distributions) is the `boxcox()` function, which finds an exponent 
  `lambda` of the response `y` such that `y ^ lambda` looks as close as 
  possible to what we would expect for a variable that is linearly 
  related to some other variable and has normally distributed errors 
  with a mean of zero and constant standard deviation. There are 
  several criteria here being improved, and it is not always clear 
  which aspect of the distribution is being improved. In many cases, 
  the main effect of the transformatoin is to make distributions of 
  residuals more homoskedastic and symmetrically distributed, but 
  often, even after transformation, the residuals will still not 
  appear quite normally distributed. However, the improvement in the
  homoskedasticity and reduced skewness may still be very useful when
  working with smaller samples, in order to improve the applicability
  of CLT assumptions.

By default, the R `MASS::boxcox()` function searches for an optimal 
  `lambda` exponent in the range of `-2` to `2`, where a `lambda` 
  exponent of `0` is treated as the natural log transformation `log(y)`.
  It also returns a 95% confidence interval for the best `lambda` value. 
  Typically, if `1` (an exponent of one is equivalent to no 
  transformation) falls within the given 95% confidence interval, 
  there is not much point to transformation. If `1` is not within the 
  confidence interval, try to pick one of the values {-2, -1, -1/2, 0, 
  1/2, 1, 2} if it falls within the interval, as it makes interpretation 
  much simpler. That is, we know that `y ^ 1` means no transformation, 
  `y ^ -1` means taking the reciprocal, `y ^ (1/2)` is a square-root 
  transformation, `lambda == 0` means the same thing as `log(y)`, 
  `y ^ 2` means squaring, and `y ^ -2` means squaring then taking the 
  reciprocal. By contrast, `y ^ 1.87` is a far less familiar/meaningful 
  transformation, even though it has a precise mathematical definition.
  You may gain some slight improvement in the appearance of the residual
  plots by making a more granular choice, but at the expense of 
  substantially complicating model intepretation. This is rarely worth it
  unless model interpretation is not important (as in the case of 
  calibration curves).

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
