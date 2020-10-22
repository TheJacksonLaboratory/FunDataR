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

1) From the built-in `warpbreaks` dataset, split out the `breaks` values where `wool == 'A'` 
   and `tension == 'L'` as one variable (say `x`). Split out the `breaks` values where 
   `wool == 'B'` and `tension == 'M'` as another variable (say `y`). 

2) Rearrange the data to be in a data-frame with `length(x) + length(y)` rows and two columns. The 
   first column should be the `breaks` and the second column should be a new indicator, of type
   `character` or `factor`, representing group membership. The first group (maybe labeled 'A') is
   the data with `wool == 'A'` and `tension == 'L'`, and the other (maybe 'B') is the data with
   `wool == 'B'` and `tension == 'M'`. Hint: concatenate `x` and `y` to make the first column; 
   use their lengths to specify the group variable.

3) Conduct a Bartlett test to see if the variances between the two groups is different.

4) We will assume the Bartlett test showed no evidence of unequal variances, so conduct a 
   two-sample equal variances t-test on the null hypothesis that the mean `breaks` of 
   group `A` and group `B` (or equivalently, `x` and `y`) are different. Return the 
   statistic (t-statistic in this case) and p-value from the test.

5) Conduct 9999 permutations of group labels, conducting the same `t.test()` described 
   above, saving the statistic for each iteration. Compare the distribution of permutation
   results to the original unpermuted result in order to calculate a permutation p-value. 
   Does it corroborate or contradict the parametric test? Hint: this will likely be 
   easier if you use the data.frame you created above and permute the group labels.

[Return to index](#index)

---

### Empirical bootstrap

When we conduct a parametric test, we typically use estimates of e.g. the mean
  and variance of a sample to 'parameterize' a known family of distributions,
  such the family of normal distributions, `N(mean, sd)`. That is we assume a 
  normal distribution is the real shape of the distribution of the parameter of
  interest, and then use some summary measures from the sample to specify the 
  center and spread of that normal shape. However, we are often wrestling with 
  the question of whether the distribution we are interested really has the shape
  of a normal distribution or not. In the case of small samples, we may do tests 
  for normality to justify parametric estimation. For largers samples, we might
  assume the CLT applies, making estimates normally distributed even when the 
  data used for estimation is itself not normally distributed. This way of doing 
  business has been around for a long time and has resulted in many successful
  discoveries and general advance of human knowledge about the universe. However, 
  for small and intermediate sized samples, as you may have noticed, it can be 
  hard to be sure if the distribution of our estimates is really normal enough 
  to justify invoking the CLT.

One alternative approach that can be particularly useful for generating confidence
  intervals in the case of small and intermediate sized samples is the 'empirical
  bootstrap'. The idea here is that instead of measuring e.g. two summary statistics
  (mean and sd) to parameterize an assumed underlying distribution (e.g. normal)
  of our data, why don't we abandon any preconceptions about the population
  distribution and instead use the sample to estimate not only the mean and sd
  of an assumed distribution shape, but instead estimate the entire shape of the 
  distribution from that sample. If the sample distribution is skewed, assume the 
  population distribution is similarly skewed. If the sample seems to have two
  peaks, then assume the population distribution has two peaks, etc. The only 
  assumption behind this method is that the sample has been drawn randomly (and
  independently) from the population of interest. 

In order to carry out a bootstrap analysis, we treat the sample as the specification
  of the population distribution. Then we take random samples (let's call them
  'resamples') from the original sample, as if the resamples were drawn from the 
  original population. Because sampling is not supposed to change the population 
  being sampled, the resampling is done 'with replacement'. That is, each time we 
  draw an observation from the original sample, the observation is replaced, so 
  the original sample remains unchanged. This means that subsequent draws from the
  sample can result in individual observations occurring once, more than once, or 
  not at all in the resample. One can use the R `sample()` function to do this 
  resampling, setting `replace=T`, but we will demonstrate an easier way using 
  the R `boot` package.

For each bootstrap resample, we perform the entire analysis we are interested in
  evaluating. We are seeing how well that analysis process is at describing the
  parameter of interest in the original population. To get at this question, we use
  the set of bootstrap results `t.star` (one result per bootstrap iteration) as 
  a surrogate for the set of results we would have gotten had we repeated the 
  original non-bootstrap analysis with many different identically sized samples 
  from the original population. This set of bootstrap results allows estimation 
  of the shape of the distribution of parameter estimate `t0` we make when using 
  the whole original sample. This is exactly the shape we were trying to assume 
  was normal in the case of parametric statistical inference about e.g. group 
  means or linear model coefficients. Now we can use the bootstrap-based estimate 
  of the shape in order to estimate confidence intervals for our parameters. 
  Also, we can compare the mean of `t.star` with `t0` to see if `t0` is a biased
  estimate of the corresponding population parameter. That is, if `mean(t.star)`
  is 5, but `t0` is 10, this suggests that estimates are downwardly biased, since
  estimates `t.star` of the parameter in the original sample `t0`, based on 
  resamples appear downwardly biased. This suggests that `t0` is itself a biased
  estimate of the true population parameter, since it was arrived at using a 
  virtually identical process of sampling and calculation as was carried out 
  with the resamples. 

There are many methods for generating confidence intervals from the distribution 
  of bootstrap results `t.star` and the estimate using the whole sample `t0`. The 
  default ones provided by the `boot::boot.ci()` function are briefly described 
  below. Unless otherwise indicated, all are 95% CIs (the default):

Percentile: this is the most straight-forward method. We estimate the 95% confidence
  interval for `t0` by taking the 2.5th percentile value and 97.5th percentile value
  of `t.star`. That is, 95% of the values of `t.star` lie between the 2.5th and 97.5th
  percentiles, so we use those as confidence bounds. This method has no adjustment
  for bias, and tends to underestimate confidence bound width when used with small 
  samples. The calculated confidence limits tend to converge to their nominal coverage
  approximately with the square-root of the sample size, similar to the speed of
  convergence toward the asymptotic results described by the CLT. Therefore some may 
  question the value over simply using asymptotically correct parametric methods,
  when the latter are available (they often are not).

```
## Percentile intervals:
ci95.lo <- quantile(t.star, prob=0.025)
ci95.hi <- quantile(t.star, prob=0.975)

```

Normal: this method is a hybrid of sorts, in that it uses `t.star` to estimate the 
  standard error of the estimates and their bias. But then it uses the standard normal 
  distribution `N(0, 1)` to calculate the the confidence bounds. This is essentially
  assuming that `t.star` is normally distributed, which can be checked either graphically,
  or by conducting a formal test for normality.

```
## Normal intervals:
bias <- t0 - mean(t.star)
ci95.lo <- (t0 - bias) - qnorm(0.025, mean=0, sd=1) * sd(t.star)
ci95.hi <- (t0 - bias) + qnorm(0.975, mean=0, sd=1) * sd(t.star)

```

Basic: this method uses the distribution of differences between `t0` and `t.star` to 
  calculate confidence bounds. It can be more robust than the percentile method to 
  skewness in the tails of the distribution of `t.star`, and incorporates a bias
  adjustment, but can produce confidence bounds that are out of the possible range 
  of the parameter being estimated (for instance, it can give a bound less than zero 
  for a quantity that is always greater than equal to zero). The speed of convergence 
  of CIs to their nominal coverage is similar to the percentile method.

```
## Basic intervals:
ci95.lo <- 2 * t0 - quantile(t.star, prob=0.025)
ci95.hi <- 2 * t0 - quantile(t.star, prob=0.975)

```

BCa: adjusts `t0` for both the bias and skewness observed in the distribution of 
  `t.star`. Confidence bounds approach their nominal coverage faster than the other
  bootstrap methods described above. However, in small samples, the results can
  be very unstable (are not precise) because the bias and particularly the skewness
  estimates tend to be unstable with small samples. This method may be the best
  choice for medium and larger samples, though it is computationally expensive 
  compared to the other methods, because the skewness estimates are made using
  jackknifing, which is another resampling method. In addition, the computations
  may simply fail for some datasets. With small samples, if the distribution
  of `t.star` appears normal, the normal method may be a good choice. The basic 
  interval can be a good choice if `t.star` does not appear quite normal and
  if the potential range of `t0` is not bounded. Otherwise the percentile method
  can be used. More typically, all four methods will produce fairly comparable
  results, supporting the robustness of the overall final result. We will not
  review the formulas as they are relatively complicated and not necessarily
  intuitive.

First we will conduct a bootstrap analysis to generate a confidence interval
  for the population mean based on a single sample. In particular, we'll 
  estimate the mean sepal length in the `virginica` species in the `iris`
  dataset. The `boot::boot()` function requires a function as an argument.
  Functions can be passed by simply passing the names of the functions, 
  without any quotes. The function passed needs to accept the original
  dataset as the first argument, and an integer index of observations in
  the bootstrap resample as the second argument. The index of bootstrap
  observations is created and passed to your function by the `boot::boot()`
  function. Your function is supposed to take the index, subset the data
  using that index, then conduct the analysis on that subset and return 
  the result. The `boot::boot()` function repeats this process, passing in
  a new index, with every bootstrap iteration:

```
## CI on population mean
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
##   and integer index of observations in the bootstrap sample
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
plot(out)                         ## bias, skew; normal enough for normal intervals?
jack.after.boot(out)              ## are their outliers affecting point estimate or CIs?

out$t0                            ## the estimate from the original sample
f(dat, T)                         ## test our function (should produce out$t0)
length(out$t)                     ## out$t is t.star (from the discussion above)
summary(out$t)
head(out$t)

ci <- boot.ci(out)
class(ci)
is.list(ci)
attributes(ci)

ci
fit1

```

As we mentioned previously, bootstrap analysis can also detect and adjust for biases
  in the original parameter estimator formula used. For instance, we have discussed
  how applying the formula for a population variance to a sample will result in a 
  downwardly biased estimate of the variance of the population from which the sample
  was drawn. We know this because many mathematicians have carefully studied the
  properties of the variance formula, and cleverly figured out how to adjust that
  formula to generate unbiased estimates of the population variance by replacing
  the sample size used in the denominator when calculating an average, with the
  samples size minus one. However, you may be interested in estimating a population 
  parameter based on a sample and not know if the sample-based calculation will 
  be a biased estimator or how to adjust it if it is biased. Bootstrapping provides
  a convenient way to estimate and adjust for the bias. We will demonstrate using
  the population formula for variance to estimate population variance based on a 
  sample randomly drawn from the population: 

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
plot(out)                         ## check for skew, bias, and normality
jack.after.boot(out)              ## check for outlier effects on point estimate and CIs

(bias <- out$t0 - mean(out$t, na.rm=T))
(est <- out$t0 + bias)
var(x)
f.var.pop(x)

(ci <- boot.ci(out))

```

Finally, we will show how to estimate confidence intervals for a coefficient
  from a linear model:

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

1) Fit a linear model with the formula `sqrt(Volume) ~ Girth` to the built-in trees dataset.

2) Generate a parametric 95% confidence interval for the for the Girth coefficient.

3) Generate a bootstrap 95% confidence interval (BCa if it works, otherwise Percentile) on the
   Girth coefficient. Does it generally corroborate or contradict the parametric interval?

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

1) Fit a linear model to the formula `Volume ~ Girth` for the built-in `trees` dataset. Fit a 
   second linear model to the formula `sqrt(Volume) ~ Girth`. Generate summaries for each fit.

2) Use 5-fold cross-validation with 3 repetitions (use the `caret::createMultiFolds()` parameter 
   `times` to set the repetitions) to compare the two formulas above in terms of mean-squared 
   error of the resulting models. Consider both the mean and standard deviations of the results 
   for each formula to decide if the `sqrt()` transformation is worthwhile.

[Return to index](#index)

---

## FIN!
