# Fundamentals of computational data analysis using R
## Machine learning
#### Contact: mitch.kostich@jax.org

---

### Index

- [Lesson 1 : Check 1](#lesson-1--check-1)
- [Lesson 1 : Check 1](#lesson-1--check-2)
- [Lesson 1 : Check 1](#lesson-1--check-3)

---

### Lesson 1 : Check 1 

Use the code above as a guide to use the 1-se rule and performance estimates (Cohen's Kappa) based
  on 7-fold cross-validation repeated 3 times to choose the value of `k` (number of nearest neighbors) 
  to use when predicting `Species` based only on the two features `Sepal.Width` and `Sepal.Length`. Try
  the following values for `k`: `c(1, 3, 5, 11, 15, 21, 31, 51, 75)`.

```
library('caret')
library('class')

rm(list=ls())

## prep data:
dat <- iris[, c('Species', 'Sepal.Width', 'Sepal.Length')]
summary(dat)

## inner cross-validation function; used to evaluate kappa for k neighbors using 
##   one fold defined by idx.trn:

f.cv <- function(idx.trn, dat, k) {
  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]
  ## NOTE: change in index of Species column from 5 to 1:
  prd.tst <- class::knn(train=dat.trn[, -1], test=dat.tst[, -1], cl=dat.trn[, 1], k=k)
  kap.tst <- caret::confusionMatrix(dat.tst[, 1], prd.tst)$overall[['Kappa']]
  kap.tst
}

## calls the f.cv() inner cross-validation function for different numbers of neighbors 
##   specified in ks and fold specified by idx.trn:

f.cv.cmp <- function(idx.trn, dat, ks) {
  rslt <- rep(as.numeric(NA), length(ks))
  for(i in 1:length(ks)) {
    rslt[i] <- f.cv(idx.trn, dat=dat, k=ks[i])
  }
  names(rslt) <- ks
  rslt
}

## generate folds:
set.seed(1)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=7, times=3)

## evaluate different values for k:

ks <- c(1, 3, 5, 11, 15, 21, 31, 51, 75)
rslt <- sapply(folds, f.cv.cmp, dat=dat, ks=ks)

## process the results:

m <- apply(rslt, 1, mean)
s <- apply(rslt, 1, sd)
se <- s / sqrt(nrow(rslt))
(rslt <- data.frame(k=ks, kap.mean=m, kap.se=se))

## the 'best' result:

(idx.max <- which.max(m))
rslt[idx.max, ]

## apply 1-se rule to attenuate potential overfitting:

(cutoff <- m[idx.max] - se[idx.max])
i.good <- m >= cutoff
rslt[i.good, ]
max(ks[i.good])
i.pick <- ks == max(ks[i.good])
rslt[i.pick, ]

```

---

### Lesson 1 : Check 2

Use the code from the above example as a template to conduct a nested cross-validation 
  of the knn-classification of the `iris` data `Species` based only on the two features 
  `Sepal.Width` and `Sepal.Length`. Use 7-fold CV, repeated 3-times at both levels, using 
  the inner-CV to select a value for `k` and the outer-CV to evaluate the entire procedure.
  Express performance in terms of Cohen's Kappa. Calculate the standard error of the 
  final performance estimate.

```
library('caret')
library('class')

rm(list=ls())

## prep data:
dat <- iris[, c('Species', 'Sepal.Width', 'Sepal.Length')]
summary(dat)

## inner cross-validation function; used to evaluate kappa for k neighbors using 
##   one fold defined by idx.trn:

f.cv <- function(idx.trn, dat, k) {
  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]
  prd.tst <- class::knn(train=dat.trn[, -1], test=dat.tst[, -1], cl=dat.trn[, 1], k=k)
  kap.tst <- caret::confusionMatrix(dat.tst[, 1], prd.tst)$overall[['Kappa']]
  kap.tst
}

## calls the f.cv() inner cross-validation function for different numbers of neighbors 
##   specified in ks and fold specified by idx.trn:

f.cv.cmp <- function(idx.trn, dat, ks) {
  rslt <- rep(as.numeric(NA), length(ks))
  for(i in 1:length(ks)) {
    rslt[i] <- f.cv(idx.trn, dat=dat, k=ks[i])
  }
  names(rslt) <- ks
  rslt
}

## uses the inner cross-validation to pick the value for the number of neighbors 
##   from the selection specified in ks:

f.pick.k <- function(dat, ks) {

  ## folds for inner cross-validation (used for tuning parameter)
  idx <- 1:nrow(dat)
  folds <- caret::createMultiFolds(idx, k=7, times=3)
  rslt <- sapply(folds, f.cv.cmp, dat=dat, ks=ks)

  m <- apply(rslt, 1, mean)
  s <- apply(rslt, 1, sd)
  se <- s / sqrt(nrow(rslt))

  ## the 'best' result:
  idx.max <- which.max(m)

  ## apply 1se rule to attenuate potential overfitting:
  cutoff <- m[idx.max] - se[idx.max]
  i.good <- m >= cutoff
  max(ks[i.good])
}

f.cv.outer <- function(idx.trn, dat) {
  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]
  ks <- c(1, 3, 5, 10, 15, 20, 30, 50, 75)
  k.pick <- f.pick.k(dat=dat.trn, ks=ks)
  prd.tst <- class::knn(train=dat.trn[, -1], test=dat.tst[, -1], cl=dat.trn[, 1], k=k.pick)
  caret::confusionMatrix(dat.tst[, 1], prd.tst)$overall[['Kappa']]
}

set.seed(1)
idx <- 1:nrow(iris)
folds <- caret::createMultiFolds(idx, k=7, times=3)
rslt <- sapply(folds, f.cv.outer, dat=dat)
mean(rslt)                       ## the final performance estimate
sd(rslt)                         ## standard deviation of performance results
sd(rslt) / sqrt(length(rslt))    ## standard error of performance estimate

---

### Lesson 2 : Check 1

```
library(glmnet)
library(caret)

rm(list=ls())

set.seed(1)
dat <- mtcars
for(i in 1:100) {
  nom <- paste('s', i, sep='')
  dat[[nom]] <- rnorm(nrow(dat), 0, 10)
}

set.seed(1)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=3)

f.cv <- function(idx.trn) {

  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]

  fit.lm.lo <- lm(mpg ~ 1, data=dat.trn)
  fit.lm.hi <- lm(mpg ~ ., data=dat.trn)
  fit.lm <- step(fit.lm.lo, scope=list(lower=fit.lm.lo, upper=fit.lm.hi), direction='both', trace=1)

  cv.ridge <- cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=0)
  cv.lasso <- cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=1)

  fit.ridge <- cv.ridge$glmnet.fit
  fit.lasso <- cv.lasso$glmnet.fit

  y.lm <- predict(fit.lm, newdata=dat.tst[, -1]) 
  y.ridge <- predict(fit.ridge, newx=as.matrix(dat.tst[, -1]), s=cv.ridge$lambda.min)
  y.lasso <- predict(fit.lasso, newx=as.matrix(dat.tst[, -1]), s=cv.lasso$lambda.min)

  mse.lm <- mean((y.lm - dat.tst[, 1]) ^ 2)
  mse.ridge <- mean((y.ridge - dat.tst[, 1]) ^ 2)
  mse.lasso <- mean((y.lasso - dat.tst[, 1]) ^ 2)

  c(lm=mse.lm, ridge=mse.ridge, lasso=mse.lasso)
}

idx.trn <- folds[[1]]
f.cv(idx.trn)

rslts <- sapply(folds, f.cv)
apply(rslts, 1, mean)
apply(rslts, 1, sd) / sqrt(length(folds))

```

---

### Lesson 2 : Check 2

```
library(glmnet)
library(caret)

rm(list=ls())

data(tecator)
dat <- absorp

dat <- t(apply(dat, 1, function(v) (v - mean(v)) / sd(v)))
dat <- t(apply(dat, 1, function(v) v + rnorm(length(v), 0, 0.5)))
dat <- cbind(endpoints[, 1], dat)
dat <- data.frame(dat)
names(dat) <- c('y', paste('x', 1:ncol(absorp), sep=''))

set.seed(1)
idx <- 1 : nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=3)

f.cv <- function(idx.trn) {

  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]

  fit.lm.lo <- lm(y ~ 1, data=data.frame(dat.trn))
  fit.lm.hi <- lm(y ~ ., data=data.frame(dat.trn))
  fit.lm <- step(fit.lm.lo, scope=list(lower=fit.lm.lo, upper=fit.lm.hi), direction='both', trace=1)

  f.alpha <- function(alpha.i) {
    cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=alpha.i)
  }

  alphas <- seq(from=0, to=1, by=0.2)
  cvs <- lapply(alphas, f.alpha)
  names(cvs) <- paste('s', alphas, sep='')

  f.fit <- function(cv.i) cv.i$glmnet.fit
  fits <- lapply(cvs, f.fit)

  f.prd <- function(idx) {
    predict(fits[[idx]], newx=as.matrix(dat.tst[, -1]), s=cvs[[idx]]$lambda.min)
  }
  ys <- sapply(1:length(fits), f.prd)
  y.lm <- predict(fit.lm, newdata=dat.tst[, -1]) 

  f.mse <- function(y.i) mean((y.i - dat.tst[, 1]) ^ 2)
  mses <- apply(ys, 2, f.mse)
  names(mses) <- names(cvs)
  c(mses, lm=f.mse(y.lm))
}

rslts <- sapply(folds, f.cv)
apply(rslts, 1, mean)
apply(rslts, 1, sd) / sqrt(nrow(rslts))

```

---

### Lesson 2 : Check 3


