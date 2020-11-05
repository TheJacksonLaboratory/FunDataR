# Fundamentals of computational data analysis using R
## Multivariate statistics: generalized linear models
#### Contact: mitch.kostich@jax.org

---

### Index

- [Generalized linear modeling](#generalized-linear-models)
- [Logistic regression](#logistic-regression)
- [Poisson regression](#poisson-regression)
- [Negative-binomial regression](#negative-binomial-regression)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Generalized linear models

So far we have been working with linear models where the response variable `y` was 
  modeled as function of individual predictor variables, `x1`, `x2`, `x3`, etc.
  along with polynomial series such as `x1 + x1^2 + x1^3`, and interactions such 
  as `x1:x3`. The assumptions are that the right transformation of `y` (which may
  be no transformation at all) and inclusion of the right predictor terms will 
  result in a model where the conditional mean of `y` is linearly related to a sum 
  of these types of predictor terms, and individual random observations from the 
  population are expected to be normally distributed around the line representing 
  the conditional mean and have a constant variance.

We can extend this modeling to systems where the variance of `y` has a relatively
  predictable relationship to the conditional mean of `y` by using **weighted 
  linear regression**. Here, instead of tuning the prediction line coefficients to
  minimize the sum-of-squared deviations of observations from the prediction line,
  we weight each observations contribution to the sum-of-squares by dividing by
  the expected variance at the observations predicted response value. That is,
  we estimate a functional relationship between the variance of the residuals
  and the corresponding fitted `y` value from an unweighted linear regression. 
  If the residuals vs. fitted plot looks like it spreads out, but does not slope,
  then the absolute values of the residuals can be modeled as a linear function 
  of the fitted values: `abs(residuals(fit)) ~ fitted(fit)`. If the residuals 
  vs. fitted plot looks like it has an upwardly curving slope, we can try to 
  model the squared residuals as a function of the fitted values: 
  `residuals(fit)^2 ~ fitted(fit)`. Sometimes a clear relationship to fitted 
  values cannot be found, but relationship to the predictors can be found instead.
  In these cases, a similar approach is followed, but with predictors instead 
  of fitted values being used as explanatory variables.

We then use the predicted variance of observations around the conditional mean 
  to weight each observation. The weight is the inverse of the expected variance. 
  That is, observations are weighted by the expected variability of the observation's
  `y` value. Where variance is high, precision is low, so the squared-residuals are 
  down-weighted relative to where variance is expected to be low. This process works 
  in part because the **coefficient estimates from an unweighted regression are unbiased**. 
  What really changes the most with weighted linear regression is the estimated stardard 
  errors. This often leads to very different (more correct) results of parametric tests 
  (e.g. overall F-test, or t-tests on individual coefficients).

One thing you will notice in the Scale-Location residual plot of the weighted fit, `fit2`
  in the example below, is that the vertical axis is plotting the (square root of the)
  **standardized residuals**. These are the residuals divided by their expected standard 
  deviations (inferred from the observation weights). Therefore, even if the residuals
  have a spreading pattern (rising variance with fitted value), the standardized residuals
  are expected to all be around `1`. So, if our variance estimates are correct, we expect
  this plot to have a flat trend with a mean near `1`.

```
rm(list=ls())

dat <- warpbreaks

fit.lo <- lm(breaks ~ 1, data=warpbreaks)
fit.up <- lm(breaks ~ .^2, data=warpbreaks)
fit1 <- step(fit.lo, scope=list(lower=fit.lo, upper=fit.up), direction='both', trace=1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

## note scale vs. location plot upward, though residuals vs fitted flat. 
##   so model abs(residuals(fit)) as function of fitted(fit):

res.abs <- abs(residuals(fit1))
fit.sd <- lm(res.abs ~ fitted(fit1))

## have to square fitted values, since they are sd estimates, not variance estimates:
wts <- 1 / (fitted(fit.sd) ^ 2)
fit2 <- lm(formula(fit1), data=warpbreaks, weights=wts)
plot(fit2, which=1:6)             ## note scale-location plot flat now (which is good)

summary(fit1)
summary(fit2)                     ## p-values different than for fit1
anova(fit1)
anova(fit2)                       ## p-value adjustments trickles down into tests of terms

```

**Generalized linear models** (**GLMs**) are a more general way to extend linear models to 
  not only non-constant residual variances, but also a wide variety of possible transformations 
  of the response `y`. The transformations of the response can be used in order to 
  accommodate various non-linear relationships, as well as restrict predicted values for
  the response to a particular range. We've seen that we can accommodate some types of
  non-linear relationships as well as non-constant residual variance by transforming the
  response. However, we often cannot find a response transformation that both linearizes
  the relationship between the response and predictor terms, but also results in normally
  distributed homoskedastic residuals. Generalized linear modeling allows for specification 
  (via a **'link function'**) of an invertible (reversible) transformation for the response 
  variable as well specification (via a **'variance function'**) of nearly arbitrary error 
  models. This means we can depart from the assumptions of normal distributions as well as
  homoskedasticity for residuals. Having separate functions for linearization and for 
  describing the error model greatly increases the flexibility over the least-squares linear
  model.

For instance, a **binary response variable** `y` is a random variable that can only take on 
  one of two values, typically encoded as `1` or `0`, where the parameter `p` is the probability 
  that it will take on the value `1`, or the proportion of observations where `y == 1`. These two 
  values may be used to represent states/categories of interest, such as 'heads' or 'tails', 
  if modeling coin flips. Or they could be used to represent 'affected' or 'non-affected' 
  subjects in a health-related experiment. We could model the mean value of `p` as a function 
  of some predictors, `x1`, `x2`, etc. When doing this, we want the conditional mean `p` to 
  always lie within the interval `[0, 1]`. The variance of a independent random binary variable 
  with a conditional mean of `p` is not constant, but equals `p * (1 - p)`: 

```
rm(list=ls())
set.seed(1)

p <- seq(from=0, to=1, by=0.01)
f <- function(p.i) var(rbinom(1e5, 1, p.i))
s2 <- sapply(p, f)

par(mfrow=c(1, 1))
plot(x=p, y=s2, ylab='variance(p)')

y <- p * (1 - p)
lines(x=p, y=y)

```

The conditional mean being modeled in this case will be the proportion `p` of observations which are
  expected to have a response `y` value of `1`. Since a proportion is generally restricted to the
  range from `0` to `1`, we need to ensure the conditional mean does not fall outside this range.
  There are several transformations in common use, but the most common these days is probably the
  **logit transformation** of `p`, `logit(p) = log(p / (1 - p))` as the response. Here `p` is the 
  probability of observing `y == 1`, and `1 - p` is the probability of observing `y == 0` (since
  there are only two possibilities). The ratio of these two probabilities is also known as the
  '**odds**' of `y == 1`. So `logit(p)` is the log of the odds, or **log-odds** of `y == 1`. The 
  function `logit(y)` is only defined in closed (does not include endpoints) interval `(0, 1)`. 
  Although individual observations are always exactly `0` or `1`, but we assume there is always a 
  non-zero probability of getting either class. That is, nowhere is the range of the predictors does 
  the model predict a zero probability of observing a response value of `0`. Even if `y == 1` is 
  modeled as far, far more likely than observing `y == 0`, there will always be some chance of 
  observing `y == 1`. 

```
rm(list=ls())

p <- seq(from=0.0001, to=0.9999, by=0.0001)
y <- log(p / (1 - p))
p2 <- exp(y) / (1 + exp(y))
all.equal(p, p2)

par(mfrow=c(1, 2))
plot(x=p, y=y, type='l', ylab='logit(p)', main='The logit function')

x3 <- seq(from=-30, to=30, by=1)
p3 <- exp(x3) / (1 + exp(x3))
plot(x=x3, y=p3, type='l', xlab='logit(p)', ylab='p', main='Invert: p between 0 and 1') 

```

Minimizing the sum-of-squared deviations makes sense as a model fitting criterion when there
  are normally distributed residuals and homoskedasticity. When these conditions do not hold,
  this criterion may not make much sense. For GLMs, we instead find model coefficients that
  maximize the likelihood of the coefficients given the training-set observations. Likelihoods
  are very closely related to probabilities. For instance, if our model is a very simple one
  where we use the global mean (an intercept-only linear model) for prediction and assume that
  the data are normally distributed around this mean (as is the assumption for the single-sample
  t-test, when the CLT cannot be invoked). Given this model, we can estimate the probability of 
  observing a response value a certain distance from the mean. For instance, if we estimate the 
  global mean and standard deviation from the sample, we can predict the probability of observing 
  any particular value:

```
rm(list=ls())
set.seed(1)

x <- rnorm(30, mean=10, sd=2)     ## randomly sample 30 observations from population N(10, 2)
(m <- mean(x))                    ## estimate population mean from sample (a bit off)
(s <- sd(x))                      ## estimate population sd from sample  (a bit off)

## plot probability vs value:
x1 <- seq(from=0, to=20, by=0.001)
p <- dnorm(x1, mean=m, sd=s)
par(mfrow=c(1, 1))
plot(x=x1, y=p, type='l', xlab='x', ylab='probability', main='p(x) for N(10.2, 1.8)')

## generate table of probabilities for series of equally spaced values:
x2 <- seq(from=0, to=20, by=1)
cbind(x=x2, p=dnorm(x2, mean=m, sd=s))

```

We can flip this idea on its head to estimate the mean based on the estimated sd in such 
  a way as to maximize the probability of the observations given the model. This yields
  the **maximum likelihood** estimate of the single coefficient (the intercept, which,
  when the lone predictor, represents the global mean). Since here we have assumed a
  normally distributed error term, the maximum likelihood coefficient matches the
  estimates from `mean(x)` and `t.test(x)`. The advantage of this way of estimating 
  coefficients is that it can be extended to a much broader range of error models than
  what can be accomplished using least-squares fitting. Furthermore, when the assumptions
  about the error model are the same (normally distributed), the maximum likelihood and
  least-squares estimates coincide.

```
(s <- sd(x))

## returns model log likelihood of observations 'x' given model N(m, s):
f.loglik <- function(m) {
  p.obs <- dnorm(x, mean=m, sd=s)
  sum(log(p.obs))
}

## log-likelihood for a range of mean estimates:
m.try <- seq(from=0, to=20, by=0.001)
loglik <- sapply(m.try, f.loglik)

## which estimate of the mean maximizes the log-likelihood?
(m.best <- m.try[which.max(loglik)])
mean(x)                           ## compare to 'regular' mean
t.test(x)                         ## compare to t-test estimate of mean

## plot coefficient (estimate of mean) log-likelihood vs x:
par(mfrow=c(1, 1))
plot(x=m.try, y=loglik, type='l')
abline(v=m.best, lty=2)           ## vertical line at maximum likelihood estimate
rug(x)                            ## tick marks for individual observation x values

```

When using maximum likelihood for estimating coefficients, for many non-normal error models, it 
  does not make much sense to describe the quality of the fit in terms of squared-errors. Instead,
  we use the concept of **deviance**. For one observation, the **observation deviance** is defined as: 
  `deviance(y.i) = 2 * log(p.saturated(y.i)) - 2 * log(p.fit(y.i))`. Here `p.saturated(y.i)` is
  the probability of observing `y == y.i` given a fully overfit model with one coefficient 
  per observation (e.g. categorical term with one level per observation). This term is a 
  constant that depends only on the observations, and not on the actual model you are interested
  in. The term `p.fit(y.i)` is the probability of observing `y == y.i` given the actual model you 
  are working with. This probability will vary depending on which formula you use to fit the
  model. This allows you to compare the relative quality of two models based on the same 
  training-set: we just compare the `log(p.fit(y.i))` from each model, since the saturated model
  is the same in each case. The overall **model deviance** is the sum of the deviances for each
  individual observation. Maximum likelihood fitting estimates model coefficients which 
  minimize the overall model deviance.

In the last lesson, we worked with the **Akaike Information Criterion** or **AIC**. This metric 
  is defined in terms of the log-likelihood of the model given the observations in the training-set:
  `AIC = 2 * k - 2 * log(likelihood(fit))`, where `k` is the number of model coefficients. We
  can see that the AIC rewards models with higher likelihoods, but penalizes models with more
  coefficients. The balance between these two terms determines is used to improve model fit, while
  providing some resistance to overfitting. The term `2 * k` can be changed to make the penalty
  more or less stringent. For instance, setting this term to `1 * k` will tend to increase the
  complexity (and potential for overfitting) of models selected, while setting this term to
  `3 * k` or `10 * k` will tend to decrease the complexity (and potential for overfitting) of
  selected models. The R `step()` function has a parameter `k` that can be used to modify the
  multiplier for the coefficient number penalty term.

[Return to index](#index)

---

### Logistic regression

We discussed modeling of binary responses using GLMs above. This type of regression model is called a
  **logistic regression** model. We saw that the conditional mean we end up modeling is the expected 
  proportion of observations where `y == 1`. The underlying process being modeled can be viewed as 
  repeated coin flips or dice rolls. For any single observation, the response variable can take on one 
  of two possible values, coded as `1` and `0`. These values can be used to represent any binary states 
  of interest, such as 'heads' and 'tails', or 'success' and 'failure' or 'treated' and 'control', or 
  'affected' and 'unaffected'. The probability `p` with which any random observation from the population 
  will have `y == 1` is assumed to be constant and independent of which observations have previously been 
  sampled. Since there are only two possible values, the probability of drawing an observation where 
  `y == 0` is `1 - p`.

The GLM error distribution for logistic regression is assumed to be the **binomial distribution**. This 
  distribution has a single parameter `p`, which is the probability of a random observation (or proportion 
  of observations) having a response value of `1`. As we've seen, for binomially distributed data, the 
  variance is expected to be `p * (1 - p)`. The GLM link function is `logit(y) = log(y / (1 - y))`, where 
  `y` is a conditional mean proportion of `1`s. 

We can perform logistic regression by calling the R `glm()` generalized linear modeling function with the
  argument setting `family='binomial'`. This will automatically invoke the appropriate logit link function 
  and binomial error model. There are several ways of representing the observations. One way is to provide
  a matrix with the first column being the number of observations where `y == 1` and the second being the 
  number of observations where `y == 0`. This form is convenient when you have designed experiments with
  fixed treatments you are using as predictors. Then there are typically many more observations than there
  are treatment combinations, so this form will be more compact than having a separate row for each 
  observation with the corresponding response and predictor values. However, encoding your entire dataset 
  with this format makes it more difficult to conduct procedures such as cross-validation, which require a 
  separate record for each observation. In the built-in `esoph` dataset, the data are presented in the 
  summarized form, where `ncases` indicates the number of observations where `y == 1` and `ncontrols`
  indicates the number of observations where `y == 0`. 

Below we fit a logistic regression model to the `esoph` data, generate a summary returning p-values on 
  coefficients, model deviance, and model AIC. We show how to extract the model deviance and AIC. We 
  conduct a parametric test on the terms of the model using the `anova()` function, which in this case 
  uses the deviances instead of sums-of-squares to test each term. The p-values returned by `summary()` 
  and `anova()` are based on assumptions of a sufficiently large samples for CLT-based approximations 
  to be 'close enough'. We also conduct familiar-looking residual plots of the fit. In this case, 
  for some plots you will see the **Pearson's residuals** being plotted. These are the residuals divided
  by the expected residual variance for that observation under the GLM error model being used. If this
  error model is correct, the absolute value of these residuals should have a mean around one and should
  appear as a flat trend line in the **Scale-Location** plot, just like the standardized residuals after
  weighted regression. Just like in the case of ordinary least-squares linear regression, the relationship
  between the transformed (via the link function) response and the predictors to be linear. This means that
  the **Residuals vs Fitted** plot should be flat. If it is not, it suggests the need to include other 
  predictor terms, such as transformations, polynomial terms or interactions. Or it could indicate that the 
  link function is not correct. Finally, we show how to make predictions using the fit. The added 
  complication here is that the default predictions are made on the link-transformed scale (not returning
  the conditional mean proportion/probability `p`, but the logit of `p`), so if we want predictions on 
  the original scale, we need to specify the `type` to the `predict()` function:

```
rm(list=ls())

## esophageal cancer rates vs. various predictors:
summary(esoph)
nrow(esoph)
head(esoph)
par(mfrow=c(1, 1))
plot(esoph)

## fit a logistic regression model to these data:
(fit <- glm(cbind(ncases, ncontrols) ~ ., data=esoph, family='binomial'))
(smry <- summary(fit))
smry$aic
deviance(fit)
anova(fit, test='Chisq')

par(mfrow=c(2, 3))
plot(fit, which=1:6)

p1 <- fitted(fit)
summary(p1)

## default predictions on logit scale, not probability scale:
p2 <- predict(fit, newdata=esoph) 
all.equal(p1, p2)
summary(p2)                       ## outside allowed range of [0, 1]

## get predictions on probability scale:
p3 <- predict(fit, newdata=esoph, type='response')
all.equal(p1, p3)
summary(p3)                       ## within allowed range of [0, 1]

```

It is easier to work with if give one row per observation, instead of the
  more compact format used above:


```
rm(list=ls())

dat <- iris[, c('Species', 'Sepal.Length', 'Sepal.Width', 'Petal.Width')]
dat$Grp <- 'negative'
dat$Grp[dat$Species == 'virginica'] <- 'virginica'
dat$Grp <- factor(dat$Grp)
dat$Species <- NULL
summary(dat)
plot(dat)

fit1 <- glm(Grp ~ ., data=dat, family='binomial')
(smry1 <- summary(fit1))
smry1$aic
anova(fit1)
deviance(fit1)

fit2 <- glm(Grp ~ Sepal.Width + Petal.Width, data=dat, family='binomial')
(smry2 <- summary(fit2))
smry2$aic

## for logistic regression models, assume difference in deviance between two nested models chisquare-distributed:
anova(fit1, fit2, test='Chisq')

```

Can use lots of familiar methods. Loss function can be deviance instead of MSE:

```
rm(list=ls())
set.seed(1)

dat <- iris[, c('Species', 'Sepal.Length', 'Sepal.Width', 'Petal.Width')]
dat$Grp <- 'negative'
dat$Grp[dat$Species == 'virginica'] <- 'virginica'
dat$Grp <- factor(dat$Grp)
dat$Species <- NULL

idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=10, times=7)
idx.trn  <- folds[[1]]

dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

fit.lo <- glm(Grp ~ 1, data=dat.trn, family='binomial')
fit.up <- glm(Grp ~ .^2, data=dat.trn, family='binomial')
fit <- step(fit.lo, scope=list(lower=fit.lo, upper=fit.up), direction='both')

formula(fit)

pred.prob <- predict(fit, newdata=dat.tst, type='response')
pROC::roc(dat.tst$Grp, pred.prob, direction='<')
pred.grp <- c('negative', 'virginica')[(pred.prob > 0.5) + 1]
pred.grp <- factor(pred.grp)
caret::confusionMatrix(dat.tst$Grp, pred.grp)

## https://topepo.github.io/caret/measuring-performance.html

```

Let's turn this into a full-fledged cross-validation:

```
f.cv <- function(idx.trn) {

  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]

  fit.lo <- glm(Grp ~ 1, data=dat.trn, family='binomial')
  fit.up <- glm(Grp ~ .^2, data=dat.trn, family='binomial')
  fit <- step(fit.lo, scope=list(lower=fit.lo, upper=fit.up), direction='both', trace=0)

  pred.prob <- predict(fit, newdata=dat.tst, type='response')
  tmp <- pROC::roc(dat.tst$Grp, pred.prob, direction='<')

  as.numeric(tmp$auc)
}

set.seed(1)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=10, times=7)
rslts <- sapply(folds, f.cv)
mean(rslts)
sd(rslts)

```

Now that we have our performance estimate, let's fit the final model to the complete dataset:

```
fit.lo <- glm(Grp ~ 1, data=dat, family='binomial')
fit.up <- glm(Grp ~ .^2, data=dat, family='binomial')
fit <- step(fit.lo, scope=list(lower=fit.lo, upper=fit.up), direction='both', trace=0)

(smry <- summary(fit))
anova(fit)
deviance(fit)
smry$aic

## how about a goodness-of-fit test? deviance / df.residual should be about 1;
##   if >> 1, suggests possible overdispersion, maybe due to missing predictors:
1 - pchisq(deviance(fit), fit$df.residual)

## pearson residuals vs. fitted should be flat and not have points >3 from 0. If curved, maybe
##   needs covariates.

## abs(pearson residuals) should be flat or error model not right.

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Poisson regression

The underlying process is assumed to be one where events occur randomly, but with
  a constant average frequency. The likelihood of an event occurring is assumed to be independent
  of how long ago the last event was. An example of this type of process is nuclear decay:
  the rate of radioactive emissions is assumed to be constant (at least over short periods), but 
  the timing is still random. In this case, the response variable can take on non-negative integer 
  values (including `0`), representing the number of events observed within a given period.

The distribution has a single parameter `lambda`, which represents the average rate with which
  events occur. The link function is `log(y)`. where `y` is a count. The variance of the data
  at any value of `lambda` is also `lambda`. So these data are not homoskedastic either:

```
rm(list=ls())
set.seed(1)

lambdas <- seq(from=0, to=100, by=1)
f <- function(lambda.i) var(rpois(1e5, lambda=lambda.i))
s2 <- sapply(lambdas, f)

par(mfrow=c(1, 1))
plot(x=lambdas, y=s2)
lines(x=lambdas, y=lambdas, col='cyan')

```

Overdispersion diagnosed by ...; may be caused by missing variables; or maybe the error model
  is not right, and the negative binomial (which has greater dispersion) is better. Another
  problem is excess zeros. A different 'zero-inflated' model (beyond this course) can be 
  considered. Also, sometimes zero truncated, so a different model also needed there.

Fit a model to some count data:

```
library('caret')

rm(list=ls())

## the data
summary(warpbreaks)
head(warpbreaks)
par(mfrow=c(1, 1))
plot(warpbreaks)

## make folds (list of training-set indices):
set.seed(1)
idx <- 1 : nrow(warpbreaks)
folds <- caret::createMultiFolds(idx, k=10, times=7)

## use first fold to make training and test sets:
idx.trn <- folds[[1]]               ## folds is a list, so double bracket index helpful
dat.trn <- warpbreaks[idx.trn, ]
dat.tst <- warpbreaks[-idx.trn, ]   ## since idx.trn is integer index, use '-' to negate

## fit upper and lower models; select working model by stepping guided by AIC:
fit.lo <- glm(breaks ~ 1, data=dat.trn, family='poisson')
fit.up <- glm(breaks ~ .^2, data=dat.trn, family='poisson')
fit <- step(fit.lo, scope=list(lower=fit.lo, upper=fit.up), direction='both')

formula(fit)                        ## the selected model

## some parametric performance estimates based solely on training-set:
summary(fit.lo)                     ## initial model + lower model of scope searched
summary(fit.up)                     ## upper model of scope searched
(smry <- summary(fit))              ## working model
anova(fit, test='Chisq')            ## significance of terms
deviance(fit)                       ## how to get model deviance (relative to training-set)
smry$aic                            ## how to get model AIC (also only uses training-set)

## evaluation using our test-set:

prd.trn <- predict(fit, newdata=dat.trn, type='response')
prd.tst <- predict(fit, newdata=dat.tst, type='response')
prd.int <- predict(fit.lo, newdata=dat.tst, type='response')

(mse.trn <- mean((dat.trn$breaks - prd.trn) ^ 2))
(mse.tst <- mean((dat.tst$breaks - prd.tst) ^ 2))
(mse.int <- mean((dat.tst$breaks - prd.int) ^ 2))

```

Do a full-fledged cross-validation:

```
library('caret')

rm(list=ls())

## make folds (list of training-set indices):
set.seed(1)
idx <- 1 : nrow(warpbreaks)
folds <- caret::createMultiFolds(idx, k=10, times=7)

## function to run for each fold:

f.cv <- function(idx.trn) {

  dat.trn <- warpbreaks[idx.trn, ]
  dat.tst <- warpbreaks[-idx.trn, ]   ## since idx.trn is integer index, use '-' to negate

  ## fit upper and lower models; select working model by stepping guided by AIC:
  fit.lo <- glm(breaks ~ 1, data=dat.trn, family='poisson')
  fit.up <- glm(breaks ~ .^2, data=dat.trn, family='poisson')
  fit <- step(fit.lo, scope=list(lower=fit.lo, upper=fit.up), direction='both', trace=0)

  prd.trn <- predict(fit, newdata=dat.trn, type='response')
  prd.tst <- predict(fit, newdata=dat.tst, type='response')
  prd.int <- predict(fit.lo, newdata=dat.tst, type='response')

  mse.trn <- mean((dat.trn$breaks - prd.trn) ^ 2)
  mse.tst <- mean((dat.tst$breaks - prd.tst) ^ 2)
  mse.int <- mean((dat.tst$breaks - prd.int) ^ 2)

  c(mse.trn=mse.trn, mse.tst=mse.tst, mse.int=mse.int)
}

rslts <- sapply(folds, f.cv)
apply(rslts, 1, mean)
apply(rslts, 1, sd)

## fit final model to whole dataset:

fit.lo <- glm(breaks ~ 1, data=warpbreaks, family='poisson')
fit.up <- glm(breaks ~ .^2, data=warpbreaks, family='poisson')
fit <- step(fit.lo, scope=list(lower=fit.lo, upper=fit.up), direction='both', trace=0)

deviance(fit)
(smry <- summary(fit))
smry$aic
anova(fit, test='Chisq')

par(mfrow=c(1, 2))

plot(warpbreaks$breaks, fitted(fit))
abline(a=0, b=1, col='cyan', lty=2)

plot(fitted(fit), abs(residuals(fit)))

```

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Negative-binomial regression

intro here; rate parameter no longer constant, but varies randomly from observation
  to observation! Introduces additional variation. Makes dispersion around the 
  mean larger.

```
library('caret')
library('MASS')

rm(list=ls())

## make folds (list of training-set indices):
set.seed(1)
idx <- 1 : nrow(warpbreaks)
folds <- caret::createMultiFolds(idx, k=10, times=7)

## use first fold to make training and test sets:
idx.trn <- folds[[1]]               ## folds is a list, so double bracket index helpful
dat.trn <- warpbreaks[idx.trn, ]
dat.tst <- warpbreaks[-idx.trn, ]   ## since idx.trn is integer index, use '-' to negate

## no longer need 'family' argument:

fit.lo <- glm.nb(breaks ~ 1, data=dat.trn)
fit.up <- glm.nb(breaks ~ .^2, data=dat.trn)
fit <- step(fit.lo, scope=list(lower=fit.lo, upper=fit.up), direction='both', trace=1)

fit
formula(fit)
deviance(fit)
(smry <- summary(fit))
smry$aic
anova(fit, test='Chisq')

## evaluation using our test-set:

prd.trn <- predict(fit, newdata=dat.trn, type='response')
prd.tst <- predict(fit, newdata=dat.tst, type='response')
prd.int <- predict(fit.lo, newdata=dat.tst, type='response')

(mse.trn <- mean((dat.trn$breaks - prd.trn) ^ 2))
(mse.tst <- mean((dat.tst$breaks - prd.tst) ^ 2))
(mse.int <- mean((dat.tst$breaks - prd.int) ^ 2))

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
