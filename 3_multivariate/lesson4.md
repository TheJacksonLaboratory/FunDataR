# Fundamentals of computational data analysis using R
## Multivariate statistics: more about multiple regression
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

intro here

p-values uniformly distributed; dangers of false positives when running hypothesis tests:

```
rm(list=ls())

set.seed(1)

R <- 10000
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

family-wise error rate (Bonferroni and Holm); vs false discovery rates:
  proportion of false positives among all positives.

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

One more example with about 5% of experiments null is not true:

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

```

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
