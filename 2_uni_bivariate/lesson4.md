# Fundamentals of computational data analysis using R
## Univariate and bivariate statistics: linear regression
#### Contact: mitch.kostich@jax.org

---

### Index

- [The linear model](#the-linear-model)
- [Fitting a simple linear regression](#fitting-a-simple-linear-regression)
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
  don't yet know how to make a more precise prediction about 
  the behavior of `y`. Previously we also learned that if we know 
  the population mean for `y` and are asked to predict the `y` value 
  of a single observation randomly drawn from that population, the 
  population mean would be our best predictor, in the least-squares
  sense. That is, the mean has a lower **average** squared deviation 
  (difference) to to the `y` values of the population members than 
  any other value. 

Here we will put these two ideas together: we will model the mean value
  of `y` as being conditional on `x`. That is, the mean of `y` values
  sampled at one value of `x` is different than at another value of `x`. 
  The 'conditional' mean of `y` in this case will be represented as a 
  straight line on the plot of `x` vs. `y`. This is the model assumed
  by Pearson's correlation. If `x` and `y` are positively
  correlated, this line will rise from left to right. Conversely,
  if they are negatively correlated, this line will decline from left
  to right. If there is no correlation between the variables, the
  line will be level (slope of zero) and pass through the mean of the
  `y` values (since in this case, the mean is not 'conditional' on
  `x`, so the global mean of `y` is the best 'fit' to the `y` values of
  the data). 

In order to do so, we will need to estimate the two parameters of a 
  straight line: the intercept (where the line crosses the `y` axis, 
  which corresponds to the value of `y` at `x == 0`), and the slope 
  (representing the rate at which `y` changes with changing `x`). We 
  will estimate these parameters of our linear equation for the 
  conditional mean in a way to minimize a sum-of-squared deviations 
  penalty, just like we implicitly do when calculating a global mean. 
  That is, the line we come up with will reduce the average squared 
  vertical `y` distances from the observations to the line. Note that
  these distances, or 'residuals', measure how far off observed `y` values are
  from the corresponding `y` value of the fitted line. These distances are 
  NOT necessarily perpendicular to the fitted line (as the shortest 
  graph distance from the observation to the line would be), but 
  perpendicular to the `x` horizontal/bottom axis, so there is no 
  `x` component to the distance, only the `y` component. This 
  minimization of the sum-of-squared deviations is conceptually very 
  similar to working with the mean, except now the 'conditional 
  mean' of `y` changes with changing `x` along the line we are 
  estimating. Just as when working with the mean, or any other 
  procedure minimizing a sum-of-squared deviations penatly function,
  the fitting procedure can be relatively sensitive to outliers.

You will often hear the terms 'independent variable' and 'dependent 
  variable', especially in the context of regression. Often, we
  try to use regression to model a causal mechanistic relationship, 
  which means that changing the 'independent' variable `x` would cause 
  the 'dependent' variable `y` to tend to change as well. However, we
  typically do not expect changing `y` to necessarily have a 
  corresponding effect on `x`. This asymmetry is reflected in labeling
  the `y` variable as 'dependent' on the nominally 'indpendent' `x`. 
  The choice of `x` in a 'mechanistic model' is ideally based on a 
  theory about the mechanism by which changes of `x` tend to lead to
  changes in `y`.

Sometimes we know very little about how the `x` are mechanistically 
  related to our 'dependent' variable of interest `y`. Nevertheless, 
  we may empirically discover an association between `x` and `y` based 
  on measurements of both variables in a random sample. We could look 
  into whether `x` causes `y` (either directly or indirectly -- in many
  cases, like some genetic interactions, we won't necessarily know) by 
  altering `x` and measuring the resulting effect on `y`. But in many 
  cases, that experiment is not practical, either because of technological 
  limits, cost or ethical restraints. We may nevertheless be interested in 
  trying to 'predict' the value of `y` based on measurements of `x`. For 
  example, in gene activity profiling experiment (e.g. RNA-seq) a lab may 
  find an association between the transcriptional activity of a gene called 
  'GeneA' and dementia risk. However, they may have no idea how the two are
  related. In principle, one might try to experimentally alter 'GeneA' 
  activity to see if it really changes dementia risk, but the lack of good
  dementia animal models, ethical restraints on human experimentation, and
  the long lag time between birth and dementia onset may make testing the
  causal nature of the relationship impractical. In fact, the gene 
  activity may not cause the risk: both may have a common antecedent, like a
  transcription factor that regulates GeneA as well as GeneB, which is the 
  true 'culprit'. Nevertheless, if the association between gene activity 
  level and risk is reliable, a statistical model of the relationship may 
  have substantial practical clinical utility. In this type of situation, 
  you may well see the terms 'independent' and 'dependent' still being 
  used, but only with the implication that the 'dependent' variable `y` 
  is being predicted from the 'independent' variable. In these cases, it 
  is often clearer if you refer to `x` the 'explanatory variable' or 
  'predictor variable' (depending on context -- are you explaining, or 
  predicting?) and `y` the 'response  variable'.

Now, we will build up to the linear model equation from the formula for a 
  line that may be familiar from high-school. Initially, `m` is the slope of 
  line, and `b` is the intercept. As we build up to a more general case 
  where there may be more than one explanatory variable, it will be 
  convenient to refer to all the constant variable coefficients using the 
  series `b0, b1, b2, b3, ...` (instead of `b`, `m` and whatever you would
  tend to use for the next coefficient), and the explanatory variables as 
  `x1, x2, x3, ...` (instead of `x`, `z`, and whatever comes next). In this
  scheme, the old intercept `b` becomes `b0`, and the old slope `m` 
  becomes `b1`, which is the coefficient for the first variable `x1`. The 
  formula for the line defines the variation in `y` that we can explain by 
  a linear relationship with `x`. We also need an expression for the variation 
  in `y` that is not captured by that linear relationship. This is the 
  'random error' term in the model, which models the population from which
  'residuals' we discussed above (differences between predicted and observed 
  `y` values) are assumed to be drawn. For calculation of the customary 
  parametric p-values, confidence intervals, and prediction intervals, we 
  further assume that the residuals share a common `N(0, s)` distribution, 
  where `s` is estimated from our sample of observations. When sample sizes 
  are large (N >= 30 in the case of a single explanatory variable is typically 
  considered adequate), we can invoke the CLT to justify use of the usual
  parametric p-values and intervals while relaxing the requirement for 
  normal distribution of residuals to a requirement that the residuals have 
  a common distribution with a mean of zero and constant (does not depend on 
  the `x` value) variance, which is finite (some distributions, e.g. Cauchy, 
  have infinite variance and would not work here). Even when the CLT cannot be 
  invoked, residual distributions are non-normal and heteroskedasticity 
  (residual spread changes with `x`) is present, as long as the residual
  distribution is not skewed and has a mean of zero, the coefficient estimates
  will be unbiased (and therefore useful), but parametric p-values and parametric 
  intervals (estimates of uncertainty in the coefficient estimates and predicted 
  values) will not be reliable.

```
y = m * x + b                     ## notation you may have seen in high school
y = b1 * x + b0                   ## use 'b#' for 'constant coefficients' instead
y = b0 + b1 * x                   ## more conventional ordering: b0 (intercept) and b1 (slope)

## sneak peak at multivariate case: explains the ordering and the numbering better:
y = b0 + (b1 * x1) + (b2 * x2) + (b3 * x3) + ...

## the statistical model for individual observations includes explanatory and error parts:
y.i = b0 + b1 * x.i + e.i
e.i ~ N(0, s)                     ## errors: model for residuals is normal w/ mean 0 and constant sd

```

In general, hypothesis testing will be less powerful and intervals wider as 
  the variance of the residual distribution rises. Another aspect of the residual
  distribution that is important for linear modeling is that the residuals
  be 'independent' of one another. This means that the direction (positive or 
  negative) and magnitude of one residual have no effect on any other residuals.
  This assumption is violated in many circumstances, particularly when working
  with temporal (time-series) or spatial (2D or 3D coordinate) data. For 
  instance, cloudy weather comes in clumps, as does sunny weather: if it rained
  yesterday, chances are higher that it will rain today. If it was sunny yesterday
  it is less likely to rain today. Similary, precipitation patterns follow 
  seasonal patterns that makes similar weather tend to clump together. When
  observations close together in time are correlated with one another, 
  this implies correlation in their residuals, violating this independence assumption. 
  Similarly, if we were to measure the basal circumference of trees in the forest, 
  we would probably find that trees near each other tend to share a similar 
  circumference distribution, different from trees further from each other: ground 
  moisture will tend to accelerate the growth of trees near the moisture, and 
  trees near each other will be more likely to be of the same species and 
  share a similar genetic allele distribution. Therefore observations that are spatially 
  closer together will tend to be more strongly correlated, introducing 
  a corresponding correlation in 
  their residuals. Both the temporal and spatial cases described exemplify 
  'auto-correlation', that is correlation between residuals that are 'close by'
  one another. There are special statistical models for autocorrelated data
  that are beyond the scope of this lesson.

Another assumption is that all the explanatory variable values are known exactly
  (without error). This is called a 'fixed effects' model. Strictly speaking, this
  assumption is typically not true when `x` is measured (as opposed to being a fixed 
  treatment of some type, where the value may be precisely known). Nevertheless, 
  you will often see linear modeling being used even when `x` is an inexactly 
  measured value. If the measurements are fairly tight, this may be an acceptable 
  practice, but depending on how important the details of the model are, using an 
  alternative formulation, called the 'random effects' model may be advisable, which 
  can explicitly model the uncertainty in the explanatory variable values, leading
  to more reliable p-values and interval estimates. Random
  effects models, which have broader applicability than the case of imprecisely
  measured `x`, are beyond the scope of this lesson, but this is a common issue you 
  should be aware of.

Here we will look at what data simulated to meet the model assumptions looks like 
  when we vary the the standard deviation of the error term:

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

[Return to index](#index)

---

### Fitting a simple linear regression

Simple linear regression refers to fitting a linear model with a single explanatory 
  variable. As we mentioned, fitting a linear model requires calculating the 
  coefficients (intercept and slope for simple regression) for the best fitting line. 
  For the slope, this involves a straightforward algebraic calculation in the case 
  of a single explanatory variable, which is closely related for the calculation of 
  Pearson's correlation:

```
## Returns intercept, slope and Pearson's correlation for y ~ x;
##   Arguments: 
##     x: explanatory variable; a numeric vector w/ length(x) == length(y) 
##     y: response variable; a numeric vector w/ length(y) == length(x)
##   Value: a numeric vector with the following named elements:
##     b0: estimated intercept of the straight line fit of 'y ~ x'.
##     b1: estimated slope of the straight line fit of 'y ~ x'.
##     r: Pearson's correlation (r) between 'y' and 'x'.

f.fit <- function(x, y) {

  x.dev <- x - mean(x)              ## x deviations from mean
  y.dev <- y - mean(y)              ## y deviations from mean

  s.xy <- sum(x.dev * y.dev)        ## n * ('covariance' of x and y)
  s.xx <- sum(x.dev ^ 2)            ## n * var.pop(x)
  s.yy <- sum(y.dev ^ 2)            ## n * var.pop(y)

  ## formula for slope of simple linear regression:
  b1 <- s.xy / s.xx

  ## formula for Pearson's correlation:
  r <- s.xy / (sqrt(s.xx) * sqrt(s.yy))

  ## formula for the intercept:
  b0 <- mean(y) - b1 * mean(x)

  return(c(b0=b0, b1=b1, r=r))
}

```

The fit produced by the `lm()` (linear model) function will include, in addition
  to the model coefficients, the 'fitted' `y` values (`y` values of the fitted
  line representing the conditional mean at values of `x` corresponding to the
  `x` value of the corresponding observation) for the data used to train the 
  model, the 'residuals' (vertical `y` distances between observed `y` values in 
  the training set and the corresponding `y` values of the fitted line 
  representing the conditional mean of `y`). The fit also includes a number of 
  elements used by the `summary()` function to generate the p-values 
  and confidence intervals we want. After fitting, for any particularly important 
  fit, we will be interested in looking at the distribution of 'residuals' to 
  identify outliers and ensure other assumptions associated with the fit are met. 

Here we will fit a simple linear regression model to the first simulated dataset
  from the last example:

```
(fit1 <- lm(y1 ~ x))
class(fit1)
is.list(fit1)
names(fit1)
attributes(fit1)
str(fit1)

## use purpose-built functions to extract coefficients, fitted values and residuals:
coef(fit1)
fitted(fit1)
residuals(fit1)

## compare to the calculations described earlier:
f.fit(x, y1)
coef(fit1)
cor(x, y1)

```

After we fit the model, we typically call `summary()` on the model, which 
  adds standard errors for the coefficient estimates and p-values for the null 
  hypothesis that the respective coefficient is equal to zero. For the slope 
  `b1`, this would correspond to no correlation between `x` and `y`). For the
  intercept `b0`, this is equivalent to the hypothesis that the line passes 
  through the origin (`x=0, y=0`) of the plot. The p-values and confidence intervals
  for the coefficients are calculated using the parametric t-distribution, just
  like the t-test. The summary also returns the F-statistic for the entire model, 
  just like for ANOVA. We will further discuss the close relationship between all 
  three of these procedures shortly. For the F-test, the null hypothesis is that 
  the overall slope of the line is not zero. In the case of a single 'independent' 
  variable `x` we are looking at now, this is equivalent to the t-test on `b1` 
  being equal to zero. We can get parametric confidence intervals for the 
  coefficients using the `confint()` function.

```
smry1 <- summary(fit1)           ## this is the main output you are interested in
class(smry1)
is.list(smry1)
names(smry1)
attributes(smry1)
str(smry1)

smry1

(coefs <- coef(smry1))            ## much more detail than coef(fit1)
class(coefs)                      ## array
coefs['x', 'Pr(>|t|)']            ## can use character indexing

confint(fit1)                     ## use t-distribution to calculate CIs for coefficients

all(residuals(smry1) == residuals(fit1))
## no 'fitted(smry1)'

cor(x, y1) ^ 2
smry1$r.squared                   ## same as squared correlation
smry1$adj.r.squared               ## adjusted downward to better reflect 'chance' fit
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

confint(fit1)
confint(fit2)
confint(fit4)
confint(fit8)

```

Let's try to model some real data:

```
rm(list=ls())

dat <- mtcars

fit <- lm(mpg ~ wt, data=dat)     ## do the initial fit
smry <- summary(fit)              ## compute p-values and CIs on b#
coef(smry)                        ## the main table of interest
confint(fit)                      ## confidence intervals on coefficients
smry$adj.r.squared                ## coefficient of determination

## p-value for overall model; usually less interesting 
##   than p-values on coefficients:

fstat <- smry$fstatistic          ## F-statistic
pf(fstat[1], fstat[2], fstat[3], lower.tail=FALSE)

```

[Return to index](#index)

---

### Equivalence to t-test and ANOVA

So far we have shown how to fit a linear model with a 
  continuous explanatory variable. However, the explanatory
  variable can also be categorical variable, like group 
  membership. In this case, null hypothesis becomes that the group
  means are all the same, just like t-test and ANOVA. The 
  coefficients returned will be the same as for ANOVA, and the
  group means will be encoded just like with ANOVA. In the
  two-sample case, the equal-variances unpaired samples version
  of `t.test()`, `aov()` and `lm()` will produce identical 
  estimates on differences in group means and p-values for the
  null hypothesis. 

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

When we have more than two groups, the `aov()` and `lm()` functions 
  return the same coefficients (and therefore same estimates of group
  means) and the same p-values:

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

Using the `mtcars` dataset:

1) Plot `mpg` vs. `disp` using the formula syntax.

2) Fit a linear model with `mpg` as the response and `disp` as the 
   explanatory variable. Extract the intercept estimate. Extract the 
   slope estimate. Get the fitted values. Get the residuals.

3) Summarize the fit. Extract the p-value for the null hypothesis that
   the regression line passes through the origin (`or y==0 when x==0`).
   Extract the p-value for the null hypothesis that the conditional 
   mean of `y` does not depend on `x`. Hint: think about the meaning of
   the coefficients returned.

4) Generate 95% confidence intervals for the two coefficients returned by
   `lm()`.

[Return to index](#index)

---

### Analysis of residuals

When we examine the adequacy of a linear model fit to the data, we are
  often most interested in looking at the residuals. This is because 
  outliers are often relatively obvious in plots of residuals, as are 
  deviations from the model assumptions of linearity, homoskedasticity
  and (particularly important for smaller samples) normality of the residuals.

So, what is an 'outlier'? An outlier is simply something that does not seem to fit 
  he current model well. When looking at using a t-test to generate a confidence 
  interval for a population mean based on a sample, we might look for 
  data points that are more than 3 standard deviations from the mean.
  In this case, the equivalent linear model is an 'intercept-only' model,
  where the intercept represents the global mean of `y`. We are looking for 
  residuals from the model that are unusually large, indicating the model 
  fit is relatively poor for these data points. We can extend this idea to 
  our linear models of a conditional mean, looking for residuals that are more
  than 3 standard deviations (of the residual distribution) from the prediction 
  line. 

When a data point does not fit the model well, it may indicate that the data 
  point represents an error of some sort: a measurement error perhaps, or maybe 
  a sampling error (like you meant to sample maple tree circumference, but 
  accidentally included an oak tree in your sample of measurements). In this 
  case, it makes good sense to remove the offending observation from the sample 
  and repeat the analysis. However, the fault may well lie in the model, rather 
  than the observation. In particular, perhaps the model lacks an important 
  explanatory term (like an additional explanatory variable or a non-linear 
  relationship to the current explanatory variable) that would greatly improve 
  the correspondence between the expanded model and the observation. When 
  outliers are identified, these possibilities need to be carefully 
  distinguished. Often this distinction is hard to make with certainty, in
  which case we should act with care, only removing apparent outliers if they
  have a significant impact on the estimates of the model coefficients. In any 
  case, if any data are removed, it should be documented (specifying which data
  were removed and why) in the methods section of subsequent scientific 
  publications. It may also be useful to compare the fits with and without the
  outliers in order to quantify the 'sensitivity' of the fit to the choice about
  whether the outliers are included or not.

In addition to having `y` response variable values that do not fit the model 
  developed from the rest of the data well, which is signalled by relatively 
  large residuals, outliers can also have `x` explanatory variable values 
  that are unusually far from the rest of the data. Below are two terms that
  are often used when discussing the impact of outliers on a fit:

**Leverage**: is based solely on the explanatory/independent variables (the single
  variable `x` here). It is a measure of how far the `x` value for an 
  observation is from the mean `x` value for the sample, normalized by the
  variability of `x` in the sample. In general, leverage greater than twice 
  the average leverage of `p / n` is considered 'high', where `p` is 
  the number of coeffients in the model and `n` is sample size. The higher the 
  leverage of an observation, the more potential differences in the `y` value 
  of that observation will tend to affect the coefficient estimates. For balanced
  ANOVA designs (all groups have equal sample size) the leverage of each 
  observation is always the same.

**Influence**: influential observations are those which, if removed from the 
  sample, would result in a large change in the fitted values for the remaining
  observations. That means that if you dropped the influential observation, 
  the coefficients of the fit would change to a relatively large degree. 
  Influence reflects both leverage (how far explanatory variables are from 
  their respective means) but also how far the `y` value for the observation is 
  from the regression line you would get by dropping this observation. The 
  further the `y` value of the omitted observation is from the regression line
  (the larger the 'residual'), and the larger the influence of the observation, 
  the higher the observations influence will be. **Cook's distance** is a measure 
  of influence which reflects the average sum-of-squared changes in fitted values 
  for the remaining observations after dropping the observation of interest, 
  normalized by the variability of residuals from the original model. Cook's 
  distance values greater than `0.5` suggest the corresponding observation has 
  high influence on the fit, and observations with Cook's distances greater than 
  `1.0` are considered to have very high influence.

When `plot()` is called on the fit returned by `lm()` (which is an object of 
  class `lm`), the call is redirected to the specialized function 
  `plot.lm()`, that knows how to generate a variety of diagnostic plots
  for a linear fit. This is just like calling `summary()` on an object of class
  `lm` will redirect the call to the specialized function `summary.lm()`
  that knows how to calculate summary statistics for a linear fit. In 
  general, code writers developing classes of their own can specify 
  class-specific versions for a number of 'generic' functions, perhaps
  most notably 'plot()' and 'summary()'.

```
rm(list=ls())

dat <- mtcars

fit <- lm(mpg ~ wt, data=dat)     ## do the initial fit
smry <- summary(fit)              ## compute p-values and CIs on b#
coef(smry)                        ## the main table of interest

nrow(coef(smry)) / nrow(dat)      ## expected average leverage: p / n

par(mfrow=c(2, 3))                ## split figure area into 2 rows, 3 cols

plot(fit, which=1:6)              ## default plot.lm() only plots c(1, 2, 3, 5)

par(mfrow=c(1, 1))                ## reset figure area to 1x1

```

Here is a list of the six residual plots and what they represent:

**Residuals vs. fitted**: a trend in the residual mean suggests the relationship 
  is not linear. Changing spread of the residuals suggests heteroskedasticity, though 
  this may be easier to see on Scale-location plot. Outliers.

**Normal Q-Q**: are the residuals normally distributed, per error term assumption in 
  the case of smaller sample sizes (if large sample, you may not care unless 
  deviations are really large). Outliers.

**Scale-location**: are residuals homoskedastic? or does residual magnitude change
  w/ fitted value. Potential outliers. sqrt(abs(residuals)) less skewed than 
  abs(residuals) for normally distributed. The scale values should bounce around 1.

**Cook's distance**: identifies 'influential' outliers by jackknifing: measure
  how much fitted values for other points change when this point is dropped from 
  the fitting procedure? Average sum-of-squared change in fitted values, normalized 
  by dividing by original residual standard deviation. Values greater than `0.5` 
  indicate high influence.

**Residuals vs. leverage**: outliers with large leverage; disassembles Cook's distance
  into residual (`y` component) and leverage (`x` component). Look for points outside
  dashed line where Cook's distance > `0.5`. Spread should not change with leverage: 
  suggests heteroskedasticity. 

**Cook's distance vs. leverage**: another way of projecting these properties.

Now we will try the same thing with some categorical data. Since the design is 
  exactly balanced (equal number of observations in each group) each data point has 
  exactly the same leverage. There are three categories, so `p` (number of returned 
  coefficients, not counting the intercept) is once again '2':

```
rm(list=ls())

dat <- iris
summary(dat)
head(dat)

fit <- lm(Sepal.Length ~ Species, data=dat)
smry <- summary(fit)              ## compute p-values and CIs on b#
coef(smry)                        ## the main table of interest
summary(aov(fit))
nrow(coef(smry)) / nrow(dat)                     ## expected mean leverage

par(mfrow=c(2, 3))
plot(fit, which=1:6)
par(mfrow=c(1, 1))

```

[Return to index](#index)

---

### Check your understanding 2

Generate a linear fit of the `mtcars` data with `mpg` as response variable
  and `disp` as a continuous explanatory variable. Generate a single figure area
  containing plots 1 thru 6 from `plot.lm()` called on your model.

1) Does the 'residuals vs. fitted' plot suggest the relationship is truly linear?

2) What does the 'scale vs. location' plot suggest about the relationship between
   residual variance and larger predicted values? Is that consistent with assumptions
   behind the the p-values `summary()` will generate for your coefficients?

3) Are there any 'highly influential' data points?

4) Does the 'residuals vs. leverage' plot suggest that 'Toyota Corolla' is influential
   more because of how far the `mpg` value is from the modeled conditional mean or how 
   far the `disp` value is from the sample average?

[Return to index](#index)

---

### Prediction

Mechanistic statistical models, which explicitly model the associations between
  variables as causal connections, are a great way to gain and test our understanding 
  about the components of the system being studied. In addition, both mechanistic 
  and empirical (where we may be ignorant of the mechanisms behind associations
  between variables) models can have value for making predictions about the value
  of the response `y` variable of a new observation based on the the 'predictor' 
  (explanatory) variable `x` value for that observation. In the examples we've 
  discussed thus far, the predicted value for the new observation will be the 
  `y` value of the line from our fit at the observation's value of `x`. This 
  `y` value is the 'fitted' or 'predicted' value for the new observation.

For any model, the model is initially developed with a finite sample from a 
  presumably much larger population. This sample used to initially fit, or 
  'train' the model can be referred to as a 'training set'. The methods we've
  shown for scrutinizing linear models has focused on looking at the residuals
  in the training set. Here we are looking for consistency with model 
  assumptions that are necessary for making parametric inferences (via p-values
  and confidence intervals) about the model coefficients. However, just like
  a sample mean fits the sample better than it would another random sample
  from that population, we should expect that our training set will fit our
  linear model of the conditional mean better than another random sample from 
  the same population. Therefore, any evaluation of our model that we do
  with the training set is expected to be somewhat overly optimistic of the
  quality of the fit to the entire population of interest. In order to get 
  a fairer evaluation of the model, it is best to use another independent
  random sample from that population. This second sample can be referred to
  as the 'test set'.

When evaluating a model, it is worth carefully thinking about the relationship
  of the samples used for training and testing with the population we wish
  to make inferences about. We may find that it is actually quite difficult 
  to get random samples from that population and get unbiased estimates of the
  performance of our model. We often want to be able to make inferences about
  what will happen if other labs try to repeat our experiment. If those 
  inferences prove correct, it means that our results are 'repeatable'. We
  could (randomly or randomly within treatment groups) hold out some of the 
  observations from an experiment conducted in our lab to use as a test set, 
  then use the remaining observations as a training set. The test set will
  give us a better estimate of model performance we should expect in someone
  else's lab than the training set, but the estimate is still probably too
  optimistic, because all the experimental parameters are constant. For 
  instance, we are not capturing variation in day-to-day parameters, such as 
  temperature and humidity. If we conducted the same experiment in our lab
  again, we would expect some variation in experimental parameters, which
  would cause the observated variable values to be systematically slightly 
  different from those in the previous experiment. So using observations 
  from the new experiment to evaluate the original model is expected to 
  result in somewhat more pessimistic, but also more realistic estimates 
  of model performance to be expected when other labs try to repeat the 
  experiment. However, even a repeat experiment in our lab will not capture
  expected additional lab-to-lab variation due to differences in reagent
  lots, experimental material (their *C. elegans* 'N2' strain colony is 
  likely genetically different from yours, due to genetic drift; their
  rearing conditions are likely somewhat different as well) equipment, 
  protocols, inter-operator variation, etc. These differences introduce
  more systematic effects that we expect will cause the model performance
  to be worse than what would be estimated by repeating the experiment
  in our own lab. As a practical matter, we want to always hold out some
  randomly selected observations for a test set which must not be used for 
  any aspect of training the model. The results from the evaluation using
  this test set provide some very preliminary estimates of model 
  performance. We can repeat the experiment later to provide somewhat more
  independent, and therefore better, test sets, refining the performance
  estimate. However, the actual model performance we care about will not 
  be known until several different labs have tried to repeat the 
  experiment.

Something about sample() function.

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
  in the training set.

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
