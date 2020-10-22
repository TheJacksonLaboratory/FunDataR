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

We are often interested in knowing whether two variables (e.g. `x` and `y`) are associated with
  one another. We do this when we conduct a proportion test where we want to know if two
  group membership variables are associated with one another, the two-sample t-tests or ANOVA 
  (where we want to know if the variable for group membership is associated with the continuous 
  variable being investigated). We also did it for two continuous variables when we looked at 
  correlation measures as well as when we fit linear models and tested whether a coefficient 
  for an explanatory variable was significant or not. The latter amounts to a test for linear 
  association between the explanatory variable and the outcome variable. In the more general 
  case, we are interested whether the value of one variable is in any way (linear or not) 
  determined by the value of another variable. The null hypothesis would be that there is no 
  relationship, which would be the same thing as saying that the value of the first variable 
  has no influence on the second variable. 

Even if there is no association between variables, due to the use of finite samples, there
  will always be some error in the estimation of population parameters. Therefore, even if
  the true population correlation of `x` and `y` is zero, in any finite sample, it is likely
  that the correlation estimated with the sample will be non-zero due to random variation.
  What we would like to know is whether the non-zero correlation (say `est1`) is non-zero 
  due to this random finite sampling effect, or due to a true association between variables. 
  In order to figure out what the chances are that the non-zero correlation was observed 
  simply by random chance, we can shuffle the values of `y` and re-estimate the correlation 
  based on the 'permuted' values. If we repeat this say 1000 times, and we observe a 
  correlation estimate at least as extreme as `est1` only 3 times, then we can say that 
  there is only about a 3 in 1000 chance that a correlation of size `est1` will be observed
  by chance when there is no true correlation between `x` and `y`. This idea is behind
  how we can use permutation testing to put p-values on hypotheses about associations 
  between variables.

The main assumption behind permutation tests is that the observations be 'exchangeable' 
  under the null hypothesis. This means that we are assuming that under the null, all the 
  observations are drawn from a single common population where `x` has no influence on 
  on the distribution of `y` and vice-versa. This imposes a more stringent null hypothesis
  than we might be used to. For instance, in the case of the two-sample t-test, this means
  that not only are the means of the two groups assumed to be equal, but every other 
  aspect of the distributions as well, such as the variance. Therefore, permutation would
  be a valid way to non-parametrically to a two-sample equal variances t-test, but not
  a two-sample unequal variances (Welch's) t-test. Fortunately, in the case of designed 
  experiments, random assignment of experimental units (e.g. test mice) to treatments 
  ensures the assumption of exchangeability is met. Permutation testing in this context
  is sometimes called 'randomization testing'. If units have not been randomly assigned
  to treatment groups (e.g. in the case of observational studies), establishing 
  exchangeability is more complicated. For complex experimental designs, permutation 
  strategies can be similarly complicated in order to ensure the exchangeability 
  condition is met.

In the examples below, we will use the R `sample()` function again, with `replace=F` 
  to conduct the permutations. This simply takes all the values in one variable, 
  scrambles them, and returns them. Each value will appear exactly as many times 
  in the permutation as it did in the original list: only the order changes. We
  will first demonstrate permutation testing with the two-sample equal variances 
  t-test. We will conduct a Bartlett test to ensure that obvious departures from
  exchangeability are observed:

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

We can employ the same strategy used for the two-sample t-test to apply 
  permutation testing to proportion tests and correlation tests. Resampling
  the group variable makes the one-factor ANOVAs we've used thus far also
  amenable to permutation testing. For simple linear regression or other 
  models with a single predictor we can just scramble the values for that
  predictor: 

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

### Empirical bootstrap

When we conduct a parametric test, we typically use estimates of e.g. the mean
  and variance of a sample to 'parameterize' a known family of distributions,
  such as parameterizing a normal distribution with a mean and variance estimated
  from the sample. That is we assume a normal distribution is the reality and then
  use some summary measures from the sample to specify the center and spread of 
  the distribution. However, we are often wrestling with the question of whether
  the distribution we are interested really is normal or not. We may do tests for
  normality for small samples, or assume the CLT applies in larger samples making
  estimates normally distributed even when the variable being estimated is not
  normally distributed. This way of doing business has been around for a long time
  and has resulted in many successes. However, for small and intermediate sized
  samples, we are often unsure if the distribution of our estimates is really 
  normal enough to justify invoking the CLT.

One alternative approach that can be particularly useful for generating confidence
  intervals in the case of small and intermediate sized samples is the 'empirical
  bootstrap'. The idea here is that instead of measuring e.g. two summary statistics
  (mean and sd) to parameterize an assumed underlying distribution (e.g. normal), 
  why don't we abandon any preconceptions about the population distribution and use 
  the sample to estimate not only the mean and sd, but the entire shape of the 
  population distribution. If the sample distribution is skewed, assume the 
  population distribution is similarly skewed. If the sample seems to have two
  peaks, then assume the population distribution has two peaks, etc. The only 
  assumption behind this method is that the sample has been drawn randomly (and
  independently) from the population of interest. In practice, what we do is 
  resample the sample observations with replacement, that is individual observations
  can appear more than once or not at all in the resampled dataset. The idea
  here is that we are using the sample as the 'surrogate' population, and 
  drawing from it should be like sampling a population without altering the 
  composition of that population (wording that we've seen several times before
  when discussing the assumptions behind statistical inference in general).

Sample with replacement: mimics sampling the population.

Assumptions. Idea like sampling population. Sample is best estimate of population
  distribution.

Confidence interval on single mean. 

Percentile: whatever the percentile of `t.star`; may be sensitive to unusual distribution 
  tails, but otherwise not too bad, and super-simple to interpret/implement.
  lo: theta((1 - alpha) / 2) 
  hi: theta(1 - (1 - alpha) / 2)

Normal: uses the z-distribution (semi-parametric) and estimated se to get percentiles.
  Assumes your plot of `t.star` is normal.
  b <- t0 - mean(t.star)
  (t0 - b) +/- z(alpha) * se ==
  (2 * t0 - mean(t.star)) +/- z(alpha) * se

Basic: uses distribution of difference between `t0` and `t.star`; more robust than percentile 
  to strange tails, but can give values out of range.
  lo: 2 * t0 - theta((1 - alpha) / 2) 
  hi: 2 * t0 - theta(1 - (1 - alpha) / 2)

BCa: adjusts both bias and skewness in the distribution of *t. may be best in larger 
  samples; unstable (high variance) when used with smaller samples. Computationally
  expensive, since adds jackknifing to the process to in order to 'accelerate' the
  bias adjustment.

If get agreement between Percentile and BCa, good to go. If BCa blows up, ...

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

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Cross-validation

Previously, we split data into a training-set and test-set, then used the
  training-set to fit a model and the test-set to evaluate the resulting
  model. If we select one fifth of the data for our test-set we can get
  a performance estimate from those data. However, if we were to repeat
  the experiment and pick a different fifth of the data for our test-set,
  we expect we would get a similar, but somewhat different result. That 
  is, our results are somewhat unstable, and depend on exactly how we 
  split the data into a training-set and test-set. The idea behind
  cross-validation (CV) is that if we average the results over the different
  possible held-out test-sets, the final result will be more stable
  and therefore a more reliable (less noisy) estimate of model
  performance. The data can be split into fifths in a way in which each
  observation appears exactly once in a test set, and other ways in 
  which individual observations can appear in more than one test set.
  In the first case, the observation order is randomized and the first
  fifth used for test-set and the rest for training-set. In the next 
  iteration, the second fifth of observations are reserved for the 
  test-set, and in the fifth iteration, the last fifth of observations 
  are used for testing. However, this procedure depends on the 
  original randomization order. By repeating the entire process
  several times, randomizing observation order at the start of each
  repetition, we can get a much larger assortment of test-sets 
  containing 20% of the observations:

```
rm(list=ls())
set.seed(1)

x <- 1:20
x <- sample(x, length(x), replace=F)
x[1:5]
x[6:10]
x[11:15]
x[16:20]

x <- sample(x, length(x), replace=F)
x[1:5]
x[6:10]
x[11:15]
x[16:20]

x <- sample(x, length(x), replace=F)
x[1:5]
x[6:10]
x[11:15]
x[16:20]

```

The size of the held-out test-set for each iteration of cross-validation
  has to be carefully considered. In general, assuming that observations 
  are drawn at random from the population of interest and that the model 
  has some predictive value (it need not be perfect; no model is), the 
  larger the sample used to train the model, the better the model will 
  fit the population and therefore the better it will tend to perform. 
  So when we only use a subset of the sample for training, as in the 
  case of cross-validation, the model will tend to be somewhat worse 
  than it would have been had we used all the available data to train 
  the model. The larger the test-set, the less data is available for
  training the model, and the worse we expect the model to perform. This
  means that performance estimates from a CV are downwardly biased, that 
  is, they will tend to underestimate the performance one would expect 
  if the full dataset was used to fit the model.

On the other hand, making the test-sets too small will tend to make the
  training-sets too similar to one another, which makes the performance
  estimate more dependent on the particular sample one began with. That
  is, if the experiment were repeated with another random sample, the
  result might be quite different, since that result will also be highly
  dependent on the particular observations in that sample. An extreme
  case of this is seen when the test-set size is one, which is sometimes
  called 'Leave-one-out cross-validation', or 'LOOCV'. In this case 
  there is one iteration per observation, with that single observation
  being the test-set and the rest of the observations in the training-set.
  This means that the training-set is as large as can be for CV, 
  minimizing the downward bias in performance estimates mentioned in the
  previous paragraph. On the other hand, the test-sets are very similar
  to each other and the original sample: they only differ by one 
  observation. That means that the models produced are all very similar 
  to one another, and strongly reflect the composition of the original 
  sample. That is, if we started with a different sample, we would
  expect a different result reflective of that particular sample. This
  means that there is a lot of variation in results that would be 
  expected if the experiment was repeated many times with new random
  samples. Therefore the precision of estimates based on CV with small 
  test-sets tends to be low.

The general advice is to use a smaller portion of observations for your
  test-sets when your initial sample size is small and a larger portion
  when initial samples sizes are large. In cases other than LOOCV (where
  it is not meaningful), it is best to repeat the cross-validation process
  several times, randomizing the observation order for each repetition, 
  and averaging the results. This further improves precisions by reducing 
  dependency of the final result on the original randomization order.
  For very small samples (n <= 10), LOOCV may well be the best option. In 
  medium sized samples (n <= 50) ten-fold CV (where training-set size
  is about 10% of the initial sample size) with at least three repetitions
  after re-randomization appears to be a good strategy. For larger 
  samples, as much as 50% of the data (2-fold CV) can be in the test-set, 
  again with some repetitions after re-randomization to smooth-out the 
  results. 

Two-fold CV also opens up the possibility of doing paired-sample t-tests 
  to estimate p-values (including non-parametric permutation-based 
  p-values) and (parametric) confidence intervals on performance estimates. 
  In order to do something similar in a general way for CV, one can wrap 
  the entire CV process within an empirical bootstrap procedure, which 
  allows estimation of confidence intervals. We can then do the equivalent
  of hypothesis testing by seeing if hypothetical values corresponding to 
  whatever our null hypothesis is (e.g. testing slope vs. hypothetical 
  value of 0) lie within the confidence interval. Bootstrapping also 
  allows us to estimate and correct for the bias in the CV performance
  measure. However, enclosing CV within a bootstrap process can require 
  many tens of thousands of model fits to be performed, which can be 
  prohibitive for computationally expensive model fits or where many 
  individual models must be evaluated.

```
library('caret')
sessionInfo()

rm(list=ls())
set.seed(1)

k <- 5
times <- 3
dat <- trees
frm1 <- Volume ~ Girth
frm2 <- Volume ~ 1
fit1 <- lm(frm1, data=dat)
fit2 <- lm(frm2, data=dat)
summary(fit1)
summary(fit2)
mean(dat$Volume)

plot(Volume ~ Girth, data=trees)
abline(fit1, lty=2, col='cyan')
abline(fit2, lty=3, col='magenta')

f <- function(idx.trn) {

  ## split into training and testing:
  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]

  ## fit traditional linear model:
  fit1 <- lm(frm1, data=dat.trn)
  pred1 <- predict(fit1, newdata=dat.tst)

  ## fit loess model:
  fit2 <- lm(frm2, data=dat.trn)
  pred2 <- predict(fit2, newdata=dat.tst)

  ## estimate error for each model:
  mse1 <- mean((dat.tst$Volume - pred1) ^ 2, na.rm=T)
  mse2 <- mean((dat.tst$Volume - pred2) ^ 2, na.rm=T)

  ## return error estimates:
  c(mse1=mse1, mse2=mse2)
}

idx <- 1 : nrow(dat)
(folds <- createMultiFolds(idx, k=k, times=times))
(rslt <- sapply(folds, f))
apply(rslt, 1, mean)
apply(rslt, 1, sd)

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
