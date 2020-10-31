# Fundamentals of computational data analysis using R
## Multivariate statistics: model building
#### Contact: mitch.kostich@jax.org

---

### Index

- [Multiple testing](#multiple-testing)
- [Overfitting](#overfitting)
- [Model selection](#model-selection)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Multiple testing

We often test many hypotheses using a single dataset. For instance, when constructing a 
  multiple regression model, we saw that the summary contains the results of an F-test
  of the overall model, as well as results of t-tests on individual model coefficients.
  If the F-test is significant, it suggests the model fits the data better than can be 
  expected by chance, or, equivalently, that at least one of the coefficients (not 
  counting the intercept) really is not zero. In the case of simple linear regression 
  (only one explanatory variable), the this F-test is equivalent to the t-test on the 
  coefficient on the explanatory variable: the two tests should return identical 
  results. In the case of multiple regression, the t-tests must be considered
  individually. As in the case of post-hoc testing following a significant omnibus
  F-test on an ANOVA model, this testing should control for the effects of multiple
  testing. For instance, in the example below, there are six separate t-tests being 
  conducted, and the p-values returned are (unfortunately) not adjusted to reflect 
  that multiple tests are being performed using the same dataset:

```
rm(list=ls())

dat <- mtcars
dat$gear <- factor(dat$gear, ordered=F)

fit <- lm(mpg ~ wt * gear, data=dat)
summary(fit)

```

We sometimes know from theory what terms should be included in a model, and in this case,
  we should not reject terms simply because the corresponding coefficient test returned 
  a non-significant p-value. Nevertheless, there are many times when there is no data to 
  guide us and we need to empirically determine a good model. In principle, this can be 
  guided by t-tests on individual coefficients, but the effects of multiple testing must 
  be accounted for. We mentioned that in the case of variables with significant interaction 
  coefficients or polynomial coefficients, the coefficients for lower-order terms should 
  not be excluded from the model, even if the lower-order coefficients were themselves not 
  significant. Sometimes one or more interaction terms involving particular levels of a 
  categorical variable will be non-significant, while terms involving other levels are highly 
  significant. In these case, it is advisable to leave the bare variable and the interaction 
  term (including the coefficients for all the levels of the categorical variable) in the 
  model.

However, in all these cases, we must properly adjust the p-values returned to account for 
  the effects of multiple testing. Multiple testing issues arise from a basic property
  of p-values: when the null hypothesis is true, p-values will be uniformly
  distributed on the interval between zero and one. That is, a p-value of 0.5 is just as 
  likely as a p-value of 0.99 or a p-value of 0.0001. Therefore, in any one test where the 
  null hypothesis is true, the p-value itself is randomly distributed, with a 95% chance
  of landing at or above the value 0.05 (since 95% of the space between 0 and 1 is 
  occupied by the region between 0.05 and 1) and a 5% chance of landing below 0.05 (since
  only 5% of the space between 0 and 1 lies below 0.05). That is, even when the null 
  hypothesis is true, there is a one-in-twenty chance that the p-value will be below 0.05
  anyway. When testing multiple hypotheses using the same dataset, this also means that
  even if all the null hypotheses are all true, on average, one in twenty tests will 
  reject the null hypothesis with a p-value < 0.05. For instance, in the example below,
  we will draw two samples (`x` and `y`) from the same population (a normal distribution 
  with `mean=0` and `sd=1`) and conduct a t-test to see if the populations each sample
  was drawn from had different means. Since they were drawn from the same population, the
  null hypothesis is always true, and significant test results (very close to the 
  expected proportion of 0.05) are false positives reflecting the distribution of p-values 
  under the null hypothesis: 

```
rm(list=ls())

set.seed(3)

R <- 50000
p.values <- rep(as.numeric(NA), R)

for(i in 1:R) {
  x <- rnorm(10, mean=0, sd=1)
  y <- rnorm(10, mean=0, sd=1)
  p.values[i] <- t.test(x, y, var.equal=T)$p.value
}

par(mfrow=c(1, 1))
hist(p.values)
summary(p.values)
sum(p.values < 0.05) / R

```

In the broadest sense, there are two approaches to controlling for the
  effects of multiple testing by 'adjusting' p-values. One approach attempts 
  to control the **family-wise error rate** or **FWER**, while the other 
  approach attempts to control the **false discovery rate** or **FDR**. These 
  approaches differ not only in the algorithms employed, but more importantly, 
  in the interpretation of the results of the corresponding adjustment. When
  adjusting for `m` tests, FWER control involves adjusting p-values so they 
  have the following interpretation: the adjusted p-value represents the 
  probability that ANY of the `m` tests will reject a null hypothesis by 
  chance. By contrast, FDR adjustment results in p-values representing the
  probability that ANY ONE of the `m` tests will reject a null hypothesis
  by chance. In the case that 1000 tests were being conducted simultaneously
  (in the case of differential expression analysis of whole transcriptome 
  data, there will often be 10s of thousands of simultaneous tests, and 
  in genome-wide association studies, the number of hypotheses can reach into
  the millions), an FWER cutoff of 0.05 means there is a 5% chance that one or
  more positive result in the set of positive results will turn out to be 
  wrong. By contrast, an FDR cutoff of 0.05 means that 5% of all the returned 
  positive results are expected to be wrong by chance. Therefore, the FWER
  is the right choice when each individual hypothesis test is considered 
  equally critical, while the FDR is a better choice in cases where you want
  to reduce the false positive rate, but are still willing to accept a small
  proportion of false posivites. The advantage of using FDR is that the 
  tests retain more power: that is they can detect smaller deviations from
  the null hypothesis for a given sample size and noise level.

Both approaches have several algorithms associated with them. Two common 
  algorithms for FWER control are the **Bonferroni** and the **Holm-Bonferroni** 
  adjustments. The Bonferroni adjustment is particularly easy to perform
  and understand: one simply multiplies each p-value by the number of tests
  being performed, and resets all values greater than `1` back to `1` (to
  keep everything within the interval from zero to one). That is:
  `p.adjust.bonferroni <- p.values * number.of.tests`. However, in practice,
  it is better to use the Holm-Bonferroni procedure for FWER control, as it 
  will always be at least as powerful, but is often more powerful as the 
  original Bonferroni procedure.

Two popular methods for achieving FDR control are the **Benjamini-Hochberg**
  or **BH** method, and the **Benjamini-Yekutieli** or **BY** method. The
  BH method is preferred when the tests are independent of each other, since 
  it retains more power under those circumstances. However, when tests are
  not independent (for instance, in an RNA-seq experiment, genes often move
  up and down together, introducing correlations between the corresponding
  test results; or in regression, when variables are correlated, their
  coefficient t-tests are also correlated), the BH method may fail to provide
  the nominal level of control (e.g. more than 5% of the positive tests may
  turn out to be wrong), so the BY method should be used, since it controls
  the FDR at the nominal rate even when tests are positively or negatively
  correlated.

In the following example, we will take the set of p-values from the last
  example, where the null hypothesis was always true, but we found about
  5% of the p-values falling below 0.05, and see what happens when we 
  apply the adjustments discussed above, using the R `p.adjust()` 
  function. When all the null hypotheses are true, the procedures tend to 
  produce fairly comparable results:

```
p.adj <- p.values * length(p.values)
p.adj[p.adj > 1] <- 1
p.bonf <- p.adjust(p.values, method='bonferroni')
all.equal(p.adj, p.bonf)

p.holm <- p.adjust(p.values, method='holm')
p.bh <- p.adjust(p.values, method='BH')
p.by <- p.adjust(p.values, method='BY')

par(mfrow=c(2, 2))
hist(p.bonf, main='Bonferroni (FWER)')
hist(p.holm, main='Holm (FWER)')
hist(p.bh, main='BH (FDR)')
hist(p.by, main='BY (FDR)')

min(p.bonf)
min(p.holm)
min(p.bh)
min(p.by)

```

One more example with about 5% of experiments null is not true. 
  The differences between procedures are more evident in this
  case:

```
rm(list=ls())
set.seed(1)

R <- 1000
p.pos <- 0.05
p.values <- rep(as.numeric(NA), R)
cnt <- 0

for(i in 1 : R) {
  val <- rbinom(1, 1, prob=p.pos)
  if(val == 1) mu = 1 else mu = 0
  x <- rnorm(10, mean=mu, sd=0.5)
  y <- rnorm(10, mean=0, sd=0.5)
  p.values[i] <- t.test(x, y, var.equal=T)$p.value
  cnt <- cnt + mu
}

p.bonf <- p.adjust(p.values, method='bonferroni')
p.holm <- p.adjust(p.values, method='holm')
p.bh <- p.adjust(p.values, method='BH')
p.by <- p.adjust(p.values, method='BY')

cnt
sum(p.values < 0.05)
sum(p.bonf < 0.05)
sum(p.holm < 0.05)
sum(p.bh < 0.05)
sum(p.by < 0.05)

par(mfrow=c(2, 2))
hist(p.bonf, main='Bonferroni (FWER)')
hist(p.holm, main='Holm (FWER)')
hist(p.bh, main='BH (FDR)')
hist(p.by, main='BY (FDR)')

```

When working with the p-values returned by t-tests on regression coefficients,
  the context should be reflected in the approach. Sometimes only a single 
  coefficient is of interest. This is typically true in the case of simple
  linear regression, since there is only one explanatory variable. However, it
  can also be true in the case of multiple regression. Frequently, we are 
  interested in whether one particular variable or interaction term is useful 
  for explaining or predicting the response variable, and we include other
  predictor variables because we already know they are important. For instance,
  we may be interested in whether there are mortality rate differences between
  women. So we may have a categorical variable for `Gender` in our model, but
  we might also include the continuous variable `Age`, because we already know
  that mortality rate (e.g. proportion of individuals who die within a year)
  is strongly associated with `Age`. Therefore, we include `Age` because it 
  reduces the sum-of-squared residuals when fitting the data, which in turn
  reduces the standard error of all the coefficient estimates, which will make
  our test on the `Gender` coefficient (in this case) much more powerful.
  Here, no adjustment of the result from the `Gender` coefficient test is 
  required. If it is significant, it suggests that `Gender` impacts the 
  mortality rate. There is no need to pay any attention to the `Age` test in
  this case. The term can be included regardless, since theory strongly suggests
  a relationship, and a non-significant test can simply reflect a lack of
  power due to too small a sample size or too much noise in the data.

If you wish to test multiple variables or terms in a single model, you can focus on
  the corresponding coefficients alone (e.g. you can still exclude tests on
  variables introduced to reduce standard errors), however you should adjust
  for multiplicity of testing. Whether you use FWER or FDR control will 
  depend on how tolerant you are of false results. If the number of hypotheses
  is relatively small, FWER often makes more sense, while FDR control may make
  more sense as the number of tests becomes larger.

If you are trying to select terms for inclusion in your model, FWER control should 
  typically be applied when the number of coefficient tests is small, and FDR 
  control used when the number of tests rises. When constructing the model, whenever 
  possible (sometimes there are too many models required to make this practical), 
  you should still also take careful account of residual plots as you build the model 
  and especially when evaluating the final model. If you are working with a linear
  model with categorical explanatory variables, if you want to achieve similar 
  interpretability of 'significance' as a post-hoc test from an ANOVA, FWER control
  should be used on the tests of the corresponding coefficients. It is curious that 
  virtually every statistics textbook will advise multiple testing control when 
  conducting post-hoc testing after ANOVA, but rarely mention this in the context 
  of what are essentially equivalent tests on coefficients of categorical variables 
  included in a linear regression.  

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Overfitting

intro here; pick the most parsimonious model consistent with the data; check with independent data.

```
rm(list=ls())

f.draw <- function(fit, lty, col) {
  x.plot <- 1:10000
  y.plot <- predict(fit, newdata=data.frame(x=x.plot))
  lines(x.plot, y.plot, lty=lty, col=col)
}

par(mfrow=c(2, 3))

n <- 2
x <- seq(from=1, to=10, length.out=n)
y <- rnorm(n, mean=0, sd=1)
fit <- lm(y ~ x)
plot(x, y, main='y ~ x')
f.draw(fit, 2, 'cyan')

n <- 3
x <- seq(from=1, to=10, length.out=n)
y <- rnorm(n, mean=0, sd=1)
fit <- lm(y ~ x)
plot(x, y, main='y ~ x')
f.draw(fit, 2, 'cyan')

fit <- lm(y ~ x + I(x^2))
plot(x, y, main='y ~ x + x^2')
f.draw(fit, 2, 'cyan')

n <- 4
x <- seq(from=1, to=10, length.out=n)
y <- rnorm(n, mean=0, sd=1)
fit <- lm(y ~ x)
plot(x, y, main='y ~ x')
f.draw(fit, 2, 'cyan')

fit <- lm(y ~ x + I(x^2))
plot(x, y, main='y ~ x + x^2')
f.draw(fit, 2, 'cyan')

fit <- lm(y ~ x + I(x^2) + I(x^3))
plot(x, y, main='y ~ x + x^2 + x^3')
f.draw(fit, 2, 'cyan')

summary(fit)

```

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Model selection

Variables vs. terms. Some factor levels may be non-significant alone or in interactions.
  For polynomials and interactions, when building, add best lowest degree terms first; when
  pruning, cut worst highest degree terms first.

For instance, `y ~ x + x^2` allows horizontal adjustment; `y ~ x^2` forces minimum to be at x == 0.
  An unwarranted assumption under most circumstances, even if not significant -- allows for a 
  more fine-tuned fit. If theory suggests at `x==0`, `y==minimum` (e.g. growth of a seed perhaps), then
  makes sense to drop `x^1`.

Can use parametric methods to compare two models, when one is nested within the other. Generally
  compare how well different models fit the training data, and have a real danger of over-fitting.

```
rm(list=ls())

dat <- mtcars[, c('mpg', 'wt', 'disp', 'gear')]
dat$gear <- factor(dat$gear, ordered=F)

fit1 <- lm(mpg ~ 1, data=dat)
fit2 <- lm(mpg ~ wt, data=dat)              ## implicitly mpg ~ 1 + hp, so fit1 nested w/i fit2
anova(fit1, fit2)
anova(fit2, fit1)
summary(fit2)                               ## anova p-value same as for t-test on coefficient

fit3 <- lm(mpg ~ wt + disp, data=dat)       ## fit2 nested w/i fit3
anova(fit2, fit3)
summary(fit3)                               ## anova p-value same as for t-test on coefficient

fit4 <- lm(mpg ~ wt * gear, data=dat)
anova(fit2, fit4)
summary(fit4)

```

AIC extends to non-nested models evaluated using the same sample. Allows arbitrary model comparisons 
  within the same family (e.g. linear models vs. other linear models). Assumes normally distributed 
  residuals.

step function order/outcome for non-orthogonal terms depends on order of terms in formula. may
  need to permute them a bit to see what effect order has -- should stick to the more stable/frequently
  appearing model. 

AIC: -2 * log-likelihood + 2 * p
 
Parametric model selection. 

```
rm(list=ls())

(fit1 <- lm(mpg ~ 1, data=mtcars))
(fit2 <- lm(mpg ~ .^2, data=mtcars))

## add/delete (since direction='both') one 'term' at time, within the bounds 
##   specified by fit1 (intercept only) and fit2 (all predictors and two-way interactions)
##   keeping model that drops AIC most; stops when no single step further improves model;
##   here start with smaller model, since larger model has too many coefficients to 
##   estimate with so few data points:

(fit3 <- step(fit1, scope=list(lower=fit1, upper=fit2), direction='both'))

summary(fit3)
par(mfrow=c(2, 3))
plot(fit3, which=1:6)

```

Evaluation procedures: CV; for continuous, try to keep size of training 
  sets `n.trn` sufficient for error degrees-of-freedom of at least 30. So, w/ 
  `n / p >= 10` or `n - p >= 30`. k for AIC.

Let's try to evaluate with a single hold-out test-set consisting of about 10%
  of the observations, like a single fold out of a 10-fold cross-validation:

```
rm(list=ls())
set.seed(1)

dat <- mtcars
mult <- 2                         ## 'k' for AIC

(n <- nrow(dat))                  ## sample size; not a ton of data; try 10-fold CV
(n.tst <- round(n / 10))          ## test set size
(n.trn <- n - n.tst)              ## training set size

## integer index of training samples:
idx.trn <- sample(1:n, size=n.trn, replace=F)

## code below will end up in a function taking idx.train, dat, and mult:

dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

fit1 <- lm(mpg ~ 1, data=dat.trn)
fit2 <- lm(mpg ~ .^2, data=dat.trn)

## trace=1 so we can see the details of the process:
fit3 <- step(fit1, scope=list(lower=fit1, upper=fit2), k=mult, direction='both', trace=1)

mpg.trn <- predict(fit3, newdata=dat.trn)
mpg.tst <- predict(fit3, newdata=dat.tst)
mpg.int <- predict(fit1, newdata=dat.tst)

(mse.trn <- mean((dat.trn$mpg - mpg.trn) ^ 2))
(mse.tst <- mean((dat.tst$mpg - mpg.tst) ^ 2))
(mse.int <- mean((dat.tst$mpg - mpg.int) ^ 2))

```

Make a function for CV, taking `idx.trn`, `dat`, and `mult`:

```
rm(list=ls())
set.seed(1)

dat <- mtcars
mult <- 2                         ## 'k' for AIC

(n <- nrow(dat))                  ## sample size; not a ton of data; try 10-fold CV
(n.tst <- round(n / 10))          ## test set size
(n.trn <- n - n.tst)              ## training set size

idx <- 1:n                        ## the integer row (observation) indices for mtcars

## grab n.trn observation indices at random as test-set:
idx.trn <- sample(idx, size=n.trn, replace=F)

## must take idx.trn as first positional argument; order of other parameters arbitrary:
f.mse <- function(idx.trn, dat, mult=mult) {

  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]

  fit1 <- lm(mpg ~ 1, data=dat.trn)
  fit2 <- lm(mpg ~ .^2, data=dat.trn)

  ## trace=0, so printout not too long:
  fit3 <- step(fit1, scope=list(lower=fit1, upper=fit2), k=mult, direction='both', trace=0)

  mpg.trn <- predict(fit3, newdata=dat.trn)
  mpg.tst <- predict(fit3, newdata=dat.tst)
  mpg.int <- predict(fit1, newdata=dat.tst)

  (mse.trn <- mean((dat.trn$mpg - mpg.trn) ^ 2))
  (mse.tst <- mean((dat.tst$mpg - mpg.tst) ^ 2))
  (mse.int <- mean((dat.tst$mpg - mpg.int) ^ 2))

  c(mse.trn=mse.trn, mse.tst=mse.tst, mse.int=mse.int)
}

f.mse(idx.trn, dat, mult=mult)

```

Now we can use f.mse to cross-validate:

```
library('caret')
set.seed(1)

dat <- mtcars
mult <- 2                         ## 'k' for AIC

n <- nrow(dat)
idx <- 1 : n

## folds for 10-fold CV: each fold is integer index of training observations:
folds <- createMultiFolds(idx, k=10, times=3)

## note how dat and mult get passed to f.mse:
rslt <- sapply(folds, f.mse, dat=dat, mult=mult)

apply(rslt, 1, mean)
apply(rslt, 1, sd)

```

Let's see what happens for different values of mult:

```
set.seed(1)

dat <- mtcars

f.mult <- function(dat, mult) {
  n <- nrow(dat)
  idx <- 1 : n
  folds <- createMultiFolds(idx, k=10, times=3)
  rslt <- sapply(folds, f.mse, dat=dat, mult=mult)
  apply(rslt, 1, mean)
}

rslts <- NULL
mults <- 1:15

for(mult.i in mults) {

  cat(mult.i, "\n")
  flush.console()                 ## for windows!

  rslt.i <- f.mult(dat, mult.i)
  rslts <- rbind(rslts, c(mult=mult.i, rslt.i))
}

rslts
(idx.best <- which.min(rslts[, 'mse.tst']))
rslts[idx.best, 'mult']

```

But the effect of the tuning procedure itself is not reflected in
  the evaluation: it is outside the cross-validation. 
  Do nested CV. The cross-validation we did so far serves as part of
  the procedure being tested, and we can refer to it as the inner
  loop of the cross-validation. The outer cross-validation is used
  for evaluation of the whole procedure.

Need another function, taking idx.trn, but this time calling
  f.mult():

```
## functionalize code from last example; search for 'best'
##   value of 'mult' using an inner cross-validation loop:

f.try <- function(dat) {

  rslts <- NULL
  mults <- 1:15

  for(mult.i in mults) {
    cat(mult.i, "\n")
    flush.console()                 ## for windows!
    rslt.i <- f.mult(dat, mult.i)
    rslts <- rbind(rslts, c(mult=mult.i, rslt.i))
  }

  rslts
  (idx.best <- which.min(rslts[, 'mse.tst']))
  rslts[idx.best, 'mult']
}

## evaluate the model you get using the tuned parameter:

f.eval <- function(idx.trn, dat) {

  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]
  mult.best <- f.try(dat.trn)

  fit1 <- lm(mpg ~ 1, data=dat.trn)
  fit2 <- lm(mpg ~ .^2, data=dat.trn)
  fit3 <- step(fit1, scope=list(lower=fit1, upper=fit2), k=mult.best, direction='both', trace=0)

  mpg.trn <- predict(fit3, newdata=dat.trn)
  mpg.tst <- predict(fit3, newdata=dat.tst)
  mpg.int <- predict(fit1, newdata=dat.tst)

  (mse.trn <- mean((dat.trn$mpg - mpg.trn) ^ 2))
  (mse.tst <- mean((dat.tst$mpg - mpg.tst) ^ 2))
  (mse.int <- mean((dat.tst$mpg - mpg.int) ^ 2))

  c(mse.trn=mse.trn, mse.tst=mse.tst, mse.int=mse.int)
}

## let's execute the outer cross-validation:

set.seed(1)

dat <- mtcars
n <- nrow(dat)
idx <- 1 : n

folds <- createMultiFolds(idx, k=10, times=3)
rslt <- sapply(folds, f.eval, dat=dat)

## note the disagreement btwn mse.trn and mse.tst, which emphasizes the
##   need for the outer CV loop:

apply(rslt, 1, mean)
apply(rslt, 1, sd)

```

Perform on whole data-set:

```
f.mse <- function(idx.trn, dat, mult=mult) {

  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]

  fit1 <- lm(mpg ~ 1, data=dat.trn)
  fit2 <- lm(mpg ~ .^2, data=dat.trn)

  ## trace=0, so printout not too long:
  fit3 <- step(fit1, scope=list(lower=fit1, upper=fit2), k=mult, direction='both', trace=0)

  mpg.trn <- predict(fit3, newdata=dat.trn)
  mpg.tst <- predict(fit3, newdata=dat.tst)
  mpg.int <- predict(fit1, newdata=dat.tst)

  (mse.trn <- mean((dat.trn$mpg - mpg.trn) ^ 2))
  (mse.tst <- mean((dat.tst$mpg - mpg.tst) ^ 2))
  (mse.int <- mean((dat.tst$mpg - mpg.int) ^ 2))

  c(mse.trn=mse.trn, mse.tst=mse.tst, mse.int=mse.int)
}

f.mult <- function(dat, mult) {
  n <- nrow(dat)
  idx <- 1 : n
  folds <- createMultiFolds(idx, k=10, times=3)
  rslt <- sapply(folds, f.mse, dat=dat, mult=mult)
  apply(rslt, 1, mean)
}

f.try <- function(dat) {

  rslts <- NULL
  mults <- 1:15

  for(mult.i in mults) {
    cat(mult.i, "\n")
    flush.console()                 ## for windows!
    rslt.i <- f.mult(dat, mult.i)
    rslts <- rbind(rslts, c(mult=mult.i, rslt.i))
  }

  rslts
  (idx.best <- which.min(rslts[, 'mse.tst']))
  rslts[idx.best, 'mult']
}

## the final fitting:

set.seed(1)
dat <- mtcars

(mult.best <- f.try(dat))

fit1 <- lm(mpg ~ 1, data=dat)
fit2 <- lm(mpg ~ .^2, data=dat)
fit.final <- step(fit1, scope=list(lower=fit1, upper=fit2), k=mult.best, direction='both', trace=0)

summary(fit.final)
par(mfrow=c(2, 3))
plot(fit.final, which=1:6)

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
