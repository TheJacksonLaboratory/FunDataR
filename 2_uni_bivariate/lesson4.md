# Fundamentals of computational data analysis using R
## Univariate and bivariate statistics: linear regression
#### Contact: mitch.kostich@jax.org

---

### Index

- [The linear model](#the-linear-model)
- [Equivalence to t-test and ANOVA](#equivalence-to-t-test-and-ANOVA)
- [Analysis of residuals](#analysis-of-residuals)
- [Prediction](#prediction)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### The linear model

In the previous lesson we saw how when two variables are correlated,
  if you know something about the values of one variable, it tells
  you something about the values of the other variable. For instance,
  if two variables `x` and `y` are positively correlated, if `x` goes
  up, you know that chances of `y` going up are increased. But we 
  don't really have a way of making a more precise prediction about 
  the behavior of `y`. Previously we also learned that if we know 
  the population mean for `y` and are asked to predict the `y` value 
  of a single observation randomly drawn from that population, the 
  population mean would be our best predictor, in the least-squares
  sense. That is, the mean has a lower **average** squared deviation 
  (difference) to to the `y` values of the population members than 
  any other value.

Independent and dependent variable. Typically used to model a causal
  relationship, which means that changing the 'independent' variable
  `x` would cause the 'dependent' variable `y` to tend to change as
  well. This is a type of 'mechanistic' model. But we 
  can also use a linear model to predict `y` based on 
  the values of `x`, even when changing `x` directly might have no
  effect on `y`. For instance, perhaps changes in `z` drive changes
  in both `x` and `y`. Then the data might suggest a linear 
  relationship between `x` and `y`, even though though one is not
  strictly dependent on the other. Nevertheless, predicting `y` based
  on `x` might work quite well as long as there were no other 
  substantial influences on `x` and `y` other than `z`. This latter
  type of model might arise as an 'empirical' model, where we observe
  an association between `x` and `y` without really understanding 
  why the relationship exists. Perhaps we don't even know `z` exists.
  It is generally considered far better to have a mechanistic model
  than an empirical model, since mechanistic models impart understanding
  about the system being studied and are usually more reliable than
  empirical models. Nevertheless, mechanistic modeling is not always
  possible given our current understanding of a system, but we may 
  nevertheless be able to predict the system behavior to a useful 
  extent using empirical models.

The model:

```
y = m * x + b                     ## notation you may have seen in high school
y = b1 * x + b0                   ## use 'b#' for 'constant coefficients'
y = b0 + b1 * x                   ## for bivariate case: b0 (intercept) and b1 (slope)

## sneak peak at multivariate case:
y = b0 + b1 * x1 + b2 * x2 + b3 * x3 + ...

## the statistical model:
y.i = b0 + b1 * x.i + e.i
e.i ~ N(0, s)

```

With e.i independent (reflects random sampling) and come from the 
  same normal distribution N(0, s). Fitting sensitive to outliers.

```
rm(list=ls())
set.seed(1)

n <- 100
b0 <- 5                           ## intercept (where line hits vertical y-axis)
b1 <- 3                           ## slope

x <- runif(n, -10, 10)

e1 <- rnorm(n, 0, 1)
e2 <- rnorm(n, 0, 2)
e4 <- rnorm(n, 0, 4)
e8 <- rnorm(n, 0, 8)

y1 <- b0 + b1 * x + e1
y2 <- b0 + b1 * x + e2
y4 <- b0 + b1 * x + e4
y8 <- b0 + b1 * x + e8

par(mfrow=c(2, 2))

plot(x=x, y=y1, main="s == 1")
abline(a=b0, b=b1, col='cyan', lty=2)

plot(x=x, y=y2, main="s == 2")
abline(a=b0, b=b1, col='cyan', lty=2)

plot(x=x, y=y4, main="s == 4")
abline(a=b0, b=b1, col='cyan', lty=2)

plot(x=x, y=y8, main="s == 8")
abline(a=b0, b=b1, col='cyan', lty=2)

par(mfrow=c(1, 1))

```

Fitting, coeficients, fitted values, residuals, coefficient of 
  determination. F-statistic. After fitting, we are often in looking at the distribution of 
  'residuals' to ensure assumptions are met. Residuals are the 
  difference between the 'fitted' (predicted) value of the 

Now fit first model and see what it yields. 

```
(fit1 <- lm(y1 ~ x))
class(fit1)
is.list(fit1)
names(fit1)
attributes(fit1)
str(fit1)

coef(fit1)
fitted(fit1)
residuals(fit1)
fit1$df.residual
fit1$call

```

But typically work with the summary, which adds confidence intervals and p-values.
  The confidence intervals are for the model estimates of the two coefficients 
  `b0` and `b1`. The null hypothesis for all the coefficients is that they are 
  zero. For `b0`, this is equivalent to the hypothesis that the line passes 
  through the origin (x=0, y=0) of the plot. Test uses the t-distribution, like 
  the t-test. Also get the F-statistic for the entire model, just like for ANOVA. 
  We will further discuss the close relationship between all three of these 
  procedures shortly. For the F-test, the null hypothesis is that the overall
  slope of the line is not zero. In the case of a single
  'independent' variable `x` we are looking at now, this is equivalent to the
  test on `b1`. 

```
(smry1 <- summary(fit1))          ## this is the main output you are interested in
class(smry1)
is.list(smry1)
names(smry1)
attributes(smry1)
str(smry1)

(coefs <- coef(smry1))            ## much more detail than coef(fit1)
class(coefs)
coefs['x', 'Pr(>|t|)']

all(residuals(smry1) == residuals(fit1))
## no 'fitted(smry1)'

smry1$adj.r.squared
(fstat <- smry1$fstatistic)       ## F-statistic + F-distrib params: numerator df, denominator df
pf(fstat[1], fstat[2], fstat[3], lower.tail=FALSE)

```

Let's see how changing the error distribution changes results:

```
fit2 <- lm(y2 ~ x)
fit4 <- lm(y4 ~ x)
fit8 <- lm(y8 ~ x)

smry2 <- summary(fit2)
smry4 <- summary(fit4)
smry8 <- summary(fit8)

smry1$fstatistic
smry2$fstatistic
smry4$fstatistic
smry8$fstatistic

smry1$adj.r.squared
smry2$adj.r.squared
smry4$adj.r.squared
smry8$adj.r.squared

coef(smry1)
coef(smry2)
coef(smry4)
coef(smry8)

```

Let's try this on some real data:

```
rm(list=ls())

dat <- mtcars

fit <- lm(mpg ~ wt, data=dat)     ## do the initial fit
smry <- summary(fit)              ## compute p-values and CIs on b#
coef(smry)                        ## the main table of interest
smry$adj.r.squared                ## coefficient of determination

## p-value for overall model; usually less interesting 
##   than p-values on coefficients:

fstat <- smry$fstatistic          ## F-statistic
pf(fstat[1], fstat[2], fstat[3], lower.tail=FALSE)

```

[Return to index](#index)

---

### Equivalence to t-test and ANOVA

Intro here. Dependent can be a categorical variable.
  In this case, null hypothesis becomes that the group
  means are all the same, just like t-test and ANOVA.
  Coefficients same as for ANOVA. Encode group means the
  same way too. 

Here we will look at the equivalency of all three procedures 
  in the two-samples case:

```
rm(list=ls())

dat <- mtcars
dat
table(dat$cyl)
dat <- dat[dat$cyl %in% c(4, 8), c('mpg', 'cyl')]
dat

(x <- dat$mpg[dat$cyl == 4])
(y <- dat$mpg[dat$cyl == 8])

dat$cyl <- factor(dat$cyl)             ## make sure interpreted as category, not number!!!

fit1 <- t.test(x=x, y=y, var.equal=T)  ## must be var.equal=T for equivalence
fit2 <- aov(mpg ~ cyl, data=dat)
fit3 <- lm(mpg ~ cyl, data=dat)

smry2 <- summary(fit2)[[1]]
smry3 <- summary(fit3)

## equivalent estimates; note same coef encoding for aov() and lm():

fit1$estimate
coef(fit2)
coef(fit3)

coef(fit3)[1]                     ## mean of first group
coef(fit3)[1] + coef(fit3)[2]     ## mean of second group

## equivalent p-values:

fit1$p.value
smry2$Pr
fstat <- smry3$fstatistic          ## F-statistic
pf(fstat[1], fstat[2], fstat[3], lower.tail=FALSE)

```

3-level ANOVA and linear model:

```
rm(list=ls())

dat <- mtcars
sapply(dat, class)
dat$cyl <- factor(dat$cyl)
sapply(dat, class)
summary(dat)

fit1 <- aov(mpg ~ cyl, data=dat)
fit2 <- lm(mpg ~ cyl, data=dat)

smry1 <- summary(fit1)[[1]]
smry2 <- summary(fit2)

## same coefficients:
coef(fit1)
coef(fit2)

## same p-value from F-test for h0: all population means are same:
smry1$Pr
fstat <- smry2$fstatistic          ## F-statistic
pf(fstat[1], fstat[2], fstat[3], lower.tail=FALSE)

## p-values for t-tests for h0s: b# is zero.
coef(smry2)

## displays are different, but can make lm() output sums-of-squares, etc:

smry2
smry1
summary(aov(fit2))                ## print aov() summary for lm() fit

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Analysis of residuals

intro here; review assumptions; sensitivity to outliers. 

What is an 'outlier': something that does not seem to fit the current 
  model well. When looking at using a t-test to generate a confidence 
  interval for a population mean based on a sample, we might look for 
  data points that are more than 3 standard deviations from the mean.
  In this case, the equivalent linear model is an 'intercept-only' model,
  and we are looking for 'residuals' from the model that are unusually
  large, indicating the model fit is relatively poor for these data points.
  We can extend this idea to more complicated models. If a data point does
  not fit the model well, it may indicate that the data point represents
  an error of some sort: a measurement error perhaps, or maybe a sampling
  error (like you meant to sample maple tree circumference, but accidentally
  included an oak tree in your sample of measurements). In this case, 
  it makes good sense to remove the offending observation from the 
  sample and repeat the analysis. However, the fault may well lie in the
  model, rather than the observation. In particular, perhaps the model
  lacks an important explanatory term that would greatly improve the 
  correspondence between the expanded model and the observation. When
  outliers are identified, these possibilities need to be carefully 
  distinguished.

When `plot()` called on the fit returned by `lm()` (which is an object of 
  class `lm`), the call is redirected to the specialized function 
  `plot.lm()`, that knows how to generate a variety of diagnostic plots
  for a linear fit. Just like calling `summary()` on an object of class
  `lm` will redirect the call to the specialized function `summary.lm()`
  that knows how to calculate summary statistics for a linear fit. In 
  general, code writers developing classes of their own can specify 
  class-specific versions for a number of 'generic' functions, perhaps
  most notably 'plot()' and 'summary()'.

Leverage: based solely on the explanatory/independent variables (the single
  variable `x` here). It is a measure of how far the `x` value for an 
  observation is from the mean `x` value for the sample, normalized by the
  variability of `x` in the sample. In general, leverage greater than twice 
  the average leverage of `(p + 1) / n` is considered 'high', where `p` is 
  the number of coeffients other than the intercept (here, `p == 2`, since there
  are 3 groups and one is modeled as the coefficient) and `n` is sample size. 
  Therefore, in the present case, leverage more than twice the expected average 
  of `3 / n` would be considered high leverage.

Influence: influential observations are those which, if removed from the sample,
  would result in a large change in the fitted values for the remaining
  observations. That means that if you dropped the influential observation, 
  the coefficients of the fit would change to a relatively large degree. 
  Influence reflects both leverage (how far explanatory variables are from 
  their respective means) but also how far the `y` value for the observation is 
  from the regression line you would get by dropping this observation. The 
  further the `y` value of the omitted observation is from the regression line, 
  and the larger the influence of the observation, the higher the observations 
  influence will be. Cook's distance is a measure of influence which reflects 
  the average sum-of-squared changes in fitted values for the remaining 
  observations after dropping the observation of interest, normalized by the
  variability of residuals from the original model. Cook's distance values 
  greater than `0.5` are considered large and distances greater than `1.0` 
  are considered very large.

Leverages constant for balanced ANOVA design (lm() w/ categorical x).

```
rm(list=ls())

dat <- mtcars
3 / nrow(dat)                     ## expected average leverage

fit <- lm(mpg ~ wt, data=dat)     ## do the initial fit
smry <- summary(fit)              ## compute p-values and CIs on b#
coef(smry)                        ## the main table of interest

par(mfrow=c(2, 3))                ## split figure area into 2 rows, 3 cols

plot(fit, which=1:6)              ## default plot.lm() only plots c(1, 2, 3, 5)

par(mfrow=c(1, 1))                ## reset figure area to 1x1

```

Residuals vs. fitted: trend may suggest relationship not linear. 

Normal Q-Q: are the residuals normally distributed, per error term assumption.
  Potential outliers.

Scale-location: are residuals homoskedastic? or does residual magnitude depend  
  on fitted value. Potential outliers. sqrt(abs(residuals)) less skewed than 
  abs(residuals) for normally distributed. Should bounce around 1.

Cook's distance: identifies 'influential outliers': identified by jackknifing:
  how much do fitted values for other points change when this point is dropped from 
  the fitting procedure? Average sum-of-squared change in fitted values,
  normalized by dividing by original residual standard deviation.

Residuals vs. leverage: outliers with large leverage; disassembles Cook's distance
  into residual (`y` component) and leverage (`x` component). Look for points outside
  dashed line where Cook's distance > `0.5`. Spread should not
  change with leverage: suggests heteroskedasticity. 

Cook's distance vs. leverage: another way of projecting these properties.

Here we will try with some categorical data. Since the design is exactly balanced
  (equal number of observations in each group) each data point has exactly the
  same leverage. Three categories, so `p` (number of returned coefficients, not
  counting the intercept) is once again '2':

```
dat <- iris
summary(dat)
head(dat)
3 / nrow(dat)                     ## expected mean leverage

fit <- lm(Sepal.Length ~ Species, data=dat)
smry <- summary(fit)              ## compute p-values and CIs on b#
coef(smry)                        ## the main table of interest
summary(aov(fit))

par(mfrow=c(2, 3))
plot(fit, which=1:6)
par(mfrow=c(1, 1))

```

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Prediction

Text here; fitted are predicted for 'training set'. Here just return the
  corresponding `y` value for the fitted line at the input value `x`.

More interested
  in accuracy of predictions for future data. Different evaluation 
  sets of varying worth.

Continuous prediction:

```
###########################################################
## Cars: speed vs. stopping distance:

rm(list=ls())
set.seed(1)                       ## random seed; for 'sample()'

dat <- cars                       ## speed vs. stopping distance
class(dat)
nrow(dat)                         ## how many observations
summary(dat)
head(dat)

###########################################################
## split the data into a training set and test set:

## 1/5th for testing; 4/5ths for training:
n.test <- round(nrow(cars) / 5)

## sample n.test unique integers from 1 to nrow(cars):
idx.test <- sample(1 : nrow(dat), size=n.test, replace=F)
nrow(dat)
length(idx.test)

## make logical index from integer index:
i.test <- rep(F, nrow(dat))
i.test[idx.test] <- T
i.train <- ! i.test
cbind(i.train, i.test)            ## i.train is T when i.test is F & vice versa

## split the data into traning (trn) and test (tst) set:
dat.trn <- dat[i.train, ]
dat.tst <- dat[i.test, ]
nrow(dat.trn)
nrow(dat.tst)

###########################################################
## fit the model with the training data:

fit <- lm(dist ~ speed, data=dat.trn)
summary(fit)

## since intercept is non-significant, refit w/o intercept:

fit <- lm(dist ~ speed - 1, data=dat.trn)
summary(fit)                      ## note improved R-squared
 
## check residual plots:

par(mfrow=c(2, 3))
plot(fit, which=1:6)

## look at the fit directly:

par(mfrow=c(1, 1))
plot(dist ~ speed - 1, data=dat.trn)
abline(fit)

```

Some stuff about prediction of trn is fitted; mse as an error 
  function.

```
###########################################################
## predict values:

## 'predicted' values for the training data:
y.hat.trn <- predict(fit, newdata=dat.trn)

## are just the fitted values:
all.equal(y.hat.trn, fitted(fit))

## predict stopping distances for the test data:
y.hat.tst <- predict(fit, newdata=dat.tst)

###########################################################
## evaluate the predictions:

## mean-squared error:

f.mse <- function(y, y.hat) {

  if(! (is.numeric(y) && is.numeric(y.hat)) )
    stop("y and y.hat must be numeric")

  if(length(y) != length(y.hat))
    stop("y and y.hat must be same length")

  if(length(y) == 0) return(NaN)

  mean((y - y.hat) ^ 2)
}

## how good are the 'predictions' of training data?:
f.mse(y=dat.trn$dist, y.hat=y.hat.trn)

## how good are the 'predictions' for test data?:
f.mse(y=dat.tst$dist, y.hat=y.hat.tst)

## what if we just used the unconditional mean to predict?:
mu <- mean(dat.trn$dist)
y.hat.mu <- rep(mu, nrow(dat.tst))
f.mse(y=dat.tst$dist, y.hat=y.hat.mu)

```

Confidence intervals capture the uncertainty in the prediction line
  (the conditional mean). Since we have no intercept in the current model,
  only the slope can change, and the confidence interval captures the
  uncertainty in the slope. If there were an intercept term in the model,
  the model would not be constrained to pass thru the origin, and the 
  confidence interval would be expected to be two parallel lines, rather than
  tapering to a point at the origin. If you want to know how far off the
  prediction line calculated using your training data sample is from the 'true'
  prediction line for the population, use the confidence interval.

Prediction intervals capture the uncertainty in the predicted values for new
  observations. They include the uncertainty of the conditional mean expressed by
  the confidence interval, but add to it the uncertainty due to the variation
  represented by the error term in the model. As a reminder: this error term 
  captures the (assumed) random, independent, normally distributed 'noise' in 
  the dependent variable that cannot be accounted for by a linear relationship 
  with the independent/explanatory variable. Therefore, prediction intervals are
  always at least as large as confidence intervals. If you want to know how
  far new observations are likely to fall from the prediction line, use the 
  prediction interval.

```
## needs same type (numeric) and name 'speed' as origin independent:
dat.new <- data.frame(speed=seq(from=0, to=30, by=0.01))

## corresponding predictions: fits will be same, prediction intervals wider:
y.hat.ci <- predict(fit, newdata=dat.new, interval='confidence')
y.hat.pi <- predict(fit, newdata=dat.new, interval='prediction')
head(y.hat.ci)
head(y.hat.pi)
all.equal(y.hat.ci[, 'fit'], y.hat.pi[, 'fit'])

(xlim <- range(dat.new$speed))
(ylim <- range(c(y.hat.ci, y.hat.pi)))

plot(x=xlim, y=ylim, type='n', xlab='speed', ylab='stopping distance')
lines(x=dat.new$speed, y.hat.ci[, 'fit'], col='orangered', lty=1)
lines(x=dat.new$speed, y.hat.ci[, 'lwr'], col='cyan', lty=2)
lines(x=dat.new$speed, y.hat.ci[, 'upr'], col='cyan', lty=2)
lines(x=dat.new$speed, y.hat.pi[, 'lwr'], col='magenta', lty=3)
lines(x=dat.new$speed, y.hat.pi[, 'upr'], col='magenta', lty=3)
points(x=dat.tst$speed, y=dat.tst$dist)

legend(
  'topleft',
  legend=c('Fit line', 'Confidence interval', 'Prediction interval'),
  lty=c(1, 2, 3),
  col=c('orangered', 'cyan', 'magenta')
)

```

Prediction with a categorical model: just the corresponding group mean
  in the training se.

```
rm(list=ls())

## flower phenotypic measurements by strain:

(dat <- iris)                     ## Species clumped into blocks
summary(dat)                      ## 50 of each of 3 species
head(dat)

## split 4/5 (N=120) for training, 1/5 (N=30) for testing; since balanced
##   experiment, and blocks in order, can systematically sample to not 
##   disrupt balance (would probably sample randomly within blocks to
##   accomplish the same if we were intending to publish):

(idx.trn <- seq(from=1, to=nrow(dat), by=5))
i.trn <- rep(T, nrow(dat))
i.trn[idx.trn] <- F
i.tst <- ! i.trn
cbind(i.tst, i.trn)

dat.trn <- dat[i.trn, ]
dat.tst <- dat[i.tst, ]

summary(dat.trn)
summary(dat.tst)

## fit the model with the training data:

fit <- lm(Sepal.Length ~ Species, data=dat.trn)
summary(fit)
par(mfrow=c(2, 3))
plot(fit, which=1:6)

## how good are the 'predictions' for test data?:

f.mse <- function(y, y.hat) {

  if(! (is.numeric(y) && is.numeric(y.hat)) )
    stop("y and y.hat must be numeric")

  if(length(y) != length(y.hat))
    stop("y and y.hat must be same length")

  if(length(y) == 0) return(NaN)

  mean((y - y.hat) ^ 2)
}

y.hat.trn <- fitted(fit)
y.hat.tst <- predict(fit, newdata=dat.tst)
y.hat.mu <- rep(mean(dat.trn$Sepal.Length), nrow(dat.tst))

## for categorical explanatory variable, predicted values are group means:
tapply(dat.trn$Sepal.Length, dat.trn$Species, mean)
table(round(y.hat.trn, 6))
table(round(y.hat.tst, 6))
table(round(y.hat.mu, 6))

f.mse(y=dat.trn$Sepal.Length, y.hat=y.hat.trn)   ## evaluate fit w/ training data
f.mse(y=dat.tst$Sepal.Length, y.hat=y.hat.tst)   ## evaluate fit w/ hold-out test data
f.mse(y=dat.tst$Sepal.Length, y.hat=y.hat.mu)    ## compare to global mean prediction

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
