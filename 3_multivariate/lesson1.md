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
  error term. Deviations from either set of assumptions are reflected
  in the distribution of residuals from the model. Assumptions behind 
  the error term justify use of parametric distributions for estimating 
  p-values and confidence intervals on coefficients, as well as 
  prediction intervals for the response variable value for new 
  observations. The error model assumptions in small sample sizes are 
  that the errors are randomly drawn from a normal distribution with 
  a mean of zero and constant standard deviation. For larger samples,
  we make the more relaxed assumpton that errors are randomly drawn from
  the same distribution with constant finite standard deviation. The
  size of the sample required for the relaxed assumption to kick in is
  dependent on the actual distribution: the closer to normal it is, 
  the smaller the required sample size. For very skewed distributions, 
  larger samples are needed. The rule-of-thumb is that about 30 
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
  linearity, you cause departure from the error assumptions, and 
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
  which made it more acceptable to analyze these data with linear models. 
  Similarly, it was common to transform proportion data using the arcsine 
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
  works well on so many data sets is because it can be used to 
  transform multiplicative relationships into additive ones. That is, 
  `log(x * y) == log(x) + log(y)`. If there is a causal relationship
  between `x` and `y` that is additive, that is a one unit change
  in `x` always results in the same change in `y`, regardless of the
  initial value of `y`, this will result in a linear dependence of
  `y` on `x`. However, if a one unit change in `x` has an effect on 
  `y` that is proportional to `y`, this means that the same size 
  change in `x` will have a smaller impact on small `y` than on large 
  `y`. This will cause the relationship to be multiplicative rather
  than linear, and also usually results in larger dispersions around 
  the prediction curve at larger values of `y` than at smaller
  values. These proportional or multiplicative effects are common in 
  nature: adding a fertilizer might make all the plants grow by an
  additional 10% within a given test period. This means that in
  absolute terms, large plants will see more gain in size than 
  small ones. The relationship is multiplicative/non-linear, and we
  should not be surprised that the spread about the mean of the 
  sizes of large plants is greater than for small ones. Transforming 
  these data by taking the log of the size will transform the 
  relationship into a more linear one as well as attenuate much of 
  the heteroskedasticity.

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
  which aspect of the distribution is changing the most. In many cases, 
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
nrow(coef(fit1)) / nrow(trees)    ## expected average leverage: p / n
d1 <- cooks.distance(fit1)        ## calculate Cook's distances
mean(d1) + 3 * sd(d1)             ## a reasonable Cook's distance cutoff
max(d1)

par(mfrow=c(2, 3))
plot(fit1, which=1:6)

## The scale-location plot shows a clear trend in the size of residuals with 
##   changing response value. Let's see if power transforming the response
##   will help:

par(mfrow=c(1, 1))
out <- boxcox(fit1, plotit=T)
class(out)
names(out)
(idx.best <- which.max(out$y))    ## max in log-likelihood ('best fit')
out$x[idx.best]                   ## corresponding exponent

## since the confidence interval does not include 1, but does include 0, which 
##   corresponds to the simple to interpret log(y), let's try log(y):

fit2 <- lm(log(Volume) ~ Height, data=trees)
nrow(coef(fit2)) / nrow(trees)    ## expected average leverage: p / n
d2 <- cooks.distance(fit2)        ## calculate Cook's distances
mean(d2) + 3 * sd(d2)             ## a reasonable Cook's distance cutoff
max(d2)                           ## 'outlier' influence has been attenuated

par(mfrow=c(2, 3))
plot(fit2, which=1:6)

dev.new()                         ## open a new plotting window
par(mfrow=c(2, 3))
plot(fit1, which=1:6)
dev.off()                         ## close the extra plotting window

```

In the previous example, we see that log transformation resulted in substantial
  attenuation of the heteroskedasticity of residuals, BUT the 'Normal Q-Q' plot
  still looks questionable. The log transformation could nevertheless help make
  the CLT assumptions kick in quicker (at smaller sample sizes) and it can be 
  seen that the influence of potential 'outlier' observations has been 
  substantially reduced as well.

[Return to index](#index)

---

### Check your understanding 1

1) Fit a third linear model using `lm()` with `trees$Volume` as response and
   `trees$Height` using the 'optimal' `lambda` value suggested above (instead
   of rounding `lambda` to `0` and using the `log()`) to transform the 
   response. How do the residual plots differ from using the `log()`?

2) Fit another linear model with `lm()` with `trees$Volume` as response and
   `trees$Height` trying a `lambda` value of `-1` (corresponding to the 
   reciprocal), to transform the response. How do the residual plots 
   differ from using the `log()`?

[Return to index](#index)

---

### Transforming predictors

If assumptions about the error term are met reasonably well, but 
  non-linearity in the relationship between the response and predictors
  is suggested by the residual plots (especially the 'Residuals
  vs Fitted' plot), consider transforming the predictors. If there is 
  more than one predictor, we can try to transform one predictor at a
  time. The most commonly employed transformations are `log(x)`, 
  `exp(x)`, `x ^ 2`, `x ^ 3`, `sqrt(x)`, and `1 / x`. For mechanistic
  models, the choices are largely driven by, or inspire, the theory
  about the process being studied. For empirical models, the 'best'
  transformation is often found through trial and error.

Sometimes a transformation of the predictors can linearize the 
  relationship, but at the expense of introducing violations of the
  assumptions about the error distribution. In these cases it may
  be worth trying to transform the response variable as well. 
  If that does not work or is impractical, weighted regression can
  be useful for addressing heteroskedasticit. The `lm()` function 
  accepts observation weights which can be used to adjust for effects 
  of heteroskedasticity on the error model by adjusting the sum-of-squared
  deviations calculation to weight the contribution of individual
  observations by their estimated variance. Due to time restrictions,
  we won't cover that method here. In addition, generalized linear 
  models (covered later) can be specified with more flexible error 
  modeling, including heteroskedasticity. However, you should be 
  aware that even if a linear model is fit to heteroskedastic data, 
  the coefficient estimates will still be unbiased: that is, if you 
  repeated the experiment many, many times, the average coefficient 
  estimates would converge on the true coefficient values. This means 
  the coefficient estimates can be trusted, despite the presence of 
  heteroskedasticity, though parametric p-values and confidence 
  cannot be trusted unless the heteroskedasticity is addressed. 
  Under these circumstances, non-parametric methods for calculating 
  p-values and intervals can be considered (introduced in the next
  lesson). 

In the following case, we will construct an example based on the
  known physical relationship between position, time and acceleration: 
  `position = position.initial + 0.5 * acceleration * (time ^ 2)`, 
  where acceleration is assumed constant. In a real experiment of this 
  type, position measurement errors are likely to increase with velocity, 
  so we expect errors to grow with velocity. Since the system is 
  accelerating, velocity grows over time which means that measurement
  errors are likely to rise over time as well, introducing 
  heteroskedasticity. For demonstration purposes, we will ignore this 
  possibility and model a constant dispersion error term.

First we will simulate some data and split it into a test-set and
  training-set:

```
rm(list=ls())
set.seed(1)

## configure simulation:

n <- 100                          ## sample size
p.tst <- 0.2                      ## proportion for test set
n.tst <- round(p.tst * n)         ## test set size

accel <- 3                        ## acceleration
y.init <- 1000                    ## initial position
tm <- runif(n, min=0, max=n)      ## time

## the modeled trajectory:
y.mdl <- y.init + 0.5 * accel * (tm ^ 2)

## a 'measurement error' term:
err <- rnorm(length(tm), mean=0, sd=150)

## response and predictor values loaded into dataframe:
y <- y.mdl + err
dat <- data.frame(y=y, tm=tm)

## split sample into test and training sets:

idx.tst <- sample(1 : n, n.tst, replace=F)
i.tst <- rep(F, n)
i.tst[idx.tst] <- T
i.trn <- ! i.tst

dat.tst <- dat[i.tst, ]
dat.trn <- dat[i.trn, ]

nrow(dat.tst)                     ## 1/5th of data
nrow(dat.trn)                     ## 4/5ths of data

```

Next we will take a peak at the training data, fit a linear model
  to the training data and examine the residual plots:

```
## plot the training data:

par(mfrow=c(1, 1))
plot(y ~ tm, data=dat.trn)

## fit a line to the training data:

fit1 <- lm(y ~ tm, data=dat.trn)

## very significant regression in all respects:
summary(fit1)

## but check out the residual plots:
par(mfrow=c(2, 3))
plot(fit1, which=1:6)
par(mfrow=c(1, 1))

```

In the above case, the 'Residuals vs Fitted' plot suggests a second 
  order (squared) relationship, due to the unidirectional curvature.
  However, residuals look homoskedastic, and we want to preserve 
  that, so we may want to avoid transorming Weight. Instead, we will
  try to transform the `tm` time variable. In the model formula we use
  specify the transformation to `lm()`, we protect the exponentiation
  operator `^` from being interpreted as a model formula operator 
  (`^` is used for specifying interactions between variables in model 
  formulas) by enclosing it in the `I()` function. Any operators 
  enclosed in an `I()` function are treated as normal R mathematical 
  functions, even if they have a special meaning in the context of
  model formulas.

```
## transform the predictor, protecting the '^' operator from 
##   interpretation as a formula operator by enclosing in 'I()':

fit2 <- lm(y ~ I(tm ^ 2), data=dat.trn)
par(mfrow=c(2, 3))
plot(fit2, which=1:6)

## lets do a more objective comparison; predict values
##   for test set using both models as well as an intercept-only
##   (global mean) model:

pred0 <- rep(mean(dat.trn$y), n.tst)
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

## The purely linear model has much less error than the intercept-only
##   (global mean model), but much more error than the model with
##   squared predictor:

f.mse(dat.tst$y, pred0)
f.mse(dat.tst$y, pred1)
f.mse(dat.tst$y, pred2)

```

### Check your understanding 2

1. Repeat the simulation of position vs time of an accelerating system from above (copy and
   paste). Then split the data into 20% test set and 80% training set (copy and paste). 
   Then fit a model of `y ~ tm` to the training data. Then fit another model of `sqrt(y) ~ tm` to 
   the training data. Finally, fit a model of `y ~ I(tm ^ 2)`.

2. Generate residual plots for all three fits and rank the fits with regard to how well
   you feel they meed assumptions for the linear model.

3. Make predictions of the response values for the observations in the test set. Generate
   estimates of the mean-squared-error for each fit based on the test set.

4. Plot the predicted response values on the y/left/vertical axis and observed response values on the
   x/bottom/horizontal axis for all three fits.

[Return to index](#index)

---

### Local regression

Linear regression has the virtues of conceptual simplicity and 
  relatively straightforward interpretation: the intercept 
  coefficient corresponds to the mean value of the response when
  the predictor is zero, and the slope coefficient corresponds
  to the ratio of the expected size of the change in the response 
  after a change in the predictor. The simplicity of the model 
  makes it easy imagine a consistent process tying together the 
  response and predictor variables. This facilitates development and 
  testing of theories underlying the association between response 
  and predictor. In addition, the model assumptions about the 
  error term justify our use of parametric distributions to conduct
  hypothesis tests and estimate intervals. The trade-offs of using
  a linear model include the relative difficulty (or inability) in 
  discovering the right set of transformations to achieve both a 
  linear relationship between response and predictors, while also
  maintaining normally distributed homoskedastic residuals. In 
  addition, the fitting process, because it involves minimization 
  of a sum-of-squared residuals penalty, is relatively sensitive 
  to the presence of outliers.

In some circumstances, the interpretability of the linear model
  is not nearly as important as accurately estimating the shape 
  of the relationship of the mean of the response variable with the 
  predictors. For instance, we sometimes want to calibrate some 
  measurements, for instance removing baseline drift from a mass 
  spectrum, or translating optical density measurements into 
  chemical concentration based on a calibration curve. In these
  cases, understanding the underlying causal processes underlying
  the observed relationships among variables may be interesting, 
  but not necessary for correcting the baseline drift or 
  translating optical measurements into estimates of chemical
  concentration. 

When interpretability is a secondary concern, one commonly employed 
  alternative to traditional linear modeling is local linear 
  regression. One popular alrogithm is implemented by the R 
  `loess()` function. This algorithm takes a training-set of 
  observations on both the response and predictor variables. At each 
  predictor value `x.i`, a line is fit using the data near that 
  predictor value, weighting the influence of surrounding observations
  by a function of their `x` distance from `x.i`. The default
  weighting function decreases an observation's weight rapidly as 
  the `x` distance from the observation to `x.i` rises. Model 
  fitting can be done using either by minimization of the (weighted) 
  sum-of-squared residuals penalty, or by using a 'robust' (to
  outliers) penalty function that is similar to the sum-of-squared 
  penalty for small residuals, but tapers off for larger residuals, 
  thereby reducing the influence of outliers.

The advantage of using the `loess()` function over `lm()` is that
  it is relatively easy to get a good fit to the data without 
  variable transformation. There is a parameter called `span`
  that may need to be optimized in order to achieve the desired
  level of smoothness in the prediction curve. In addition, the
  `degree` parameter specifies the order of the locally fitted 
  polynomial. The default is `2`, which is a curve. Specifying
  `1` will still result in a flexible fit, but the flexibility will
  tend to be less. The main disadvantages of using the `loess()` 
  function are that it is nearly impossible to interpret: the 
  coefficients do not have very intuitive meanings. In addition, 
  there are no p-values or intervals being returned, so we have 
  to rely on methods like use of hold-out test-set if we want 
  to get an estimate of performance. Another more technical issue
  is that once a loess model is fit to one dataset, communicating
  the details of that model to other researchers so it can be
  reused is not as straightforward as for a linear model where 
  simply knowing the estimated coefficient values is sufficient
  to recreate the analysis. For loess, one might need to share
  the training data and ensure exactly the same version of the 
  fitting algorithm is used to re-fit the model.

We will demonstrate the use of the `loess()` function with a dataset
  on weight-loss over time of obesity patients at a weight-loss 
  clinic. We begin by splitting our sample into a test-set and 
  training-set:

```
rm(list=ls())
set.seed(3)

summary(wtloss)

n <- nrow(wtloss)
p.tst <- 0.5
n.tst <- round(p.tst * n)

idx.tst <- sample(1 : n, n.tst, replace=F)
i.tst <- rep(F, n)
i.tst[idx.tst] <- T
dat.tst <- wtloss[i.tst, ]
dat.trn <- wtloss[! i.tst, ]

```

Now we will plot the training-set and test-set with distinctive
  symbols and colors, fit a simple linear regression to the 
  training-set, and fit a local regression to the same data. We
  then add the prediction lines for each fit to the plot:

```
## plot the observations:

par(mfrow=c(1, 1))
plot(Weight ~ Days, data=dat.trn, pch='+', col='black')
abline(h=mean(dat.trn$Weight), lty=4, col='black')
points(Weight ~ Days, data=dat.tst, pch='o', col='orangered')

## fit a simple linear regression model to training data:

(fit1 <- lm(Weight ~ Days, data=dat.trn))
(smry1 <- summary(fit1))
abline(fit1, lty=2, col='magenta')

## fit a local regression model to training data:

(fit2 <- loess(Weight ~ Days, data=dat.trn))
class(fit2)
is.list(fit2)
names(fit2)
attributes(fit2)

## generate some regularly spaced predictor values for plotting:
dat.plot <- data.frame(Days=seq(from=0, to=246, by=0.01))

## add the corresonding predictions and plot them:
dat.plot$Weight <- predict(fit2, newdata=dat.plot)
lines(Weight ~ Days, data=dat.plot, lty=3, col='cyan')

legend('topright', legend=c('global mean', 'lm', 'loess'), 
  lty=c(4, 2, 3), col=c('black', 'magenta', 'cyan'))

```

Now we will make predictions for our test-set using our two
  fitted models and make point estimates for the performance
  of each:

```
## predictions (global mean, lm() and loess()):

pred0 <- rep(mean(dat.trn$Weight), nrow(dat.tst))
pred1 <- predict(fit1, newdata=dat.tst)
pred2 <- predict(fit2, newdata=dat.tst)

## performance metric function:

f.mse <- function(obs, pred) {
  if(! (is.numeric(obs) && is.numeric(pred)) )
    stop("obs and pred must be numeric")

  if(length(obs) != length(pred))
    stop("obs and pred must be same length")

  if(length(obs) == 0) return(NaN)

  mean((obs - pred) ^ 2, na.rm=T)
}

## performance (global mean, lm() and loess()):

f.mse(dat.tst$Weight, pred0)
f.mse(dat.tst$Weight, pred1)
f.mse(dat.tst$Weight, pred2)

```

Finally, we will fiddle with some of the parameters to `loess()` to see 
  what difference they make:

```
f.fit <- function(span, degree, family, lty, col) {

  ## fit the model using training-set:
  suppressWarnings(fit <- loess(Weight ~ Days, data=dat.trn, span=span, 
    degree=degree, family=family))

  dat.plot <- data.frame(Days=seq(from=0, to=246, by=0.01))

  ## add the corresonding predictions and plot them:
  dat.plot$Weight <- predict(fit, newdata=dat.plot)
  lines(Weight ~ Days, data=dat.plot, lty=lty, col=col)

  ## return mse estimated from test-set:
  pred <- predict(fit, newdata=dat.tst)
  f.mse(dat.tst$Weight, pred)
}

f.plot <- function(main, degree, family) {

  plot(Weight ~ Days, data=dat.trn, pch='+', col='black', main=main, type='n')

  ltys <- c(2, 3, 2, 3, 2)
  cols <- c('cyan', 'magenta', 'orangered', 'cyan', 'magenta')
  spans <- c(0.2, 0.4, 0.6, 0.8, 1)

  for(i in 1 : length(spans)) {
    mse <- f.fit(spans[i], degree, family, ltys[i], cols[i])
    cat("span", spans[i], "mse", mse, "\n")
    flush.console()
  }

  legend('topright', legend=spans, col=cols, lty=ltys, cex=0.75)
}

par(mfrow=c(2, 2))

f.plot("Gaussian degree 2", 2, 'gaussian')
f.plot("Gaussian degree 1", 1, 'gaussian')
f.plot("Symmetric degree 2", 2, 'symmetric')
f.plot("Symmetric degree 1", 1, 'symmetric')

```

[Return to index](#index)

---

### Check your understanding 3

1. Split the built-in 'DNase' dataset into a test-set consisting of `Run == 3` and `Run == 6` and a 
   training-set consisting of the other nine `Run` values.

2. Fit a linear model of `density ~ conc` to the training data. Make predictions on the held-out test-set.

3. Fit a loess model to the same formula, using the default parameter settings. Make predictions on the
   held-out test-set.

4. Calculate the mean-squared error for the two models based on the test-set predictions.

[Return to index](#index)

---

## FIN!
