# Fundamentals of computational data analysis using R
## Machine learning: model tuning and evaluation
#### Contact: mitch.kostich@jax.org

---

### Index

- [Title 1](#title-1)
- [Title 2](#title-2)
- [Title 3](#title-3)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### title 1

intro here; 

kappa: (observed.accuracy - expected.accuracy) / (1 - expected.accuracy)
  expected.accuracy: depends on composition of training set ...
  interpretation
  can be used to compare classifiers and parameterizations

```
library('caret')
library('class')

rm(list=ls())
set.seed(1)

dat <- iris
plot(dat)
nrow(dat)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)

idx.trn <- folds[[1]]
dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

class(dat.trn)
summary(dat.trn)

prd.trn <- class::knn(train=dat.trn[, -5], test=dat.trn[, -5], cl=dat.trn[, 5], k=7)
prd.tst <- class::knn(train=dat.trn[, -5], test=dat.tst[, -5], cl=dat.trn[, 5], k=7)

(cnf.trn <- caret::confusionMatrix(dat.trn[, 5], prd.trn))
(cnf.tst <- caret::confusionMatrix(dat.tst[, 5], prd.tst))

## double bracket indexing 'Kappa' drops the name:
cnf.trn$overall[['Kappa']]
cnf.tst$overall[['Kappa']]

```

Turn into a cross-validation:

```
library('caret')
library('class')

rm(list=ls())

f.cv <- function(idx.trn, dat, k) {
  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]
  prd.tst <- class::knn(train=dat.trn[, -5], test=dat.tst[, -5], cl=dat.trn[, 5], k=k)
  kap.tst <- caret::confusionMatrix(dat.tst[, 5], prd.tst)$overall[['Kappa']]
  kap.tst
}

set.seed(1)
dat <- iris
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)
rslt <- sapply(folds, f.cv, dat=dat, k=7)
mean(rslt)
sd(rslt)

```

Try a series of ks for knn:

```
set.seed(1)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=3)
idx.trn <- folds[[1]]

ks <- c(1, 3, 5, 10, 15, 20, 30, 50, 75)
rslt <- rep(as.numeric(NA), length(ks))

for(i in 1:length(ks)) {
  rslt[i] <- f.cv(idx.trn, dat=dat, k=ks[i])
}
names(rslt) <- ks
signif(rslt, 3)

```

Turn into a function, and cross-validate; the 1-se rule-of-thumb:

```
f.cv.cmp <- function(idx.trn, dat, ks) {
  rslt <- rep(as.numeric(NA), length(ks))
  for(i in 1:length(ks)) {
    rslt[i] <- f.cv(idx.trn, dat=dat, k=ks[i])
  }
  names(rslt) <- ks
  rslt
}

set.seed(1)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=7, times=3)

ks <- c(1, 3, 5, 10, 15, 20, 30, 50, 75)
(rslt <- sapply(folds, f.cv.cmp, dat=dat, ks=ks))

m <- apply(rslt, 1, mean)
s <- apply(rslt, 1, sd)
se <- s / sqrt(nrow(rslt))
(rslt <- data.frame(k=ks, kap.mean=m, kap.se=se))

## the 'best' result:
(idx.max <- which.max(m))
rslt[idx.max, ]

## apply 1se rule to attenuate potential overfitting:
(cutoff <- m[idx.max] - se[idx.max])
i.good <- m >= cutoff
rslt[i.good, ]
max(ks[i.good])

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### title 2

intro here

```
library('caret')
library('class')

rm(list=ls())

f.cv <- function(idx.trn, dat, k) {
  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]
  prd.tst <- class::knn(train=dat.trn[, -5], test=dat.tst[, -5], cl=dat.trn[, 5], k=k)
  kap.tst <- caret::confusionMatrix(dat.tst[, 5], prd.tst)$overall[['Kappa']]
  kap.tst
}

f.cv.cmp <- function(idx.trn, dat, ks) {
  rslt <- rep(as.numeric(NA), length(ks))
  for(i in 1:length(ks)) {
    rslt[i] <- f.cv(idx.trn, dat=dat, k=ks[i])
  }
  names(rslt) <- ks
  rslt
}

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

set.seed(1)
idx <- 1:nrow(iris)
folds <- caret::createMultiFolds(idx, k=7, times=3)
idx.trn <- folds[[1]]

dat.trn <- iris[idx.trn, ]
dat.tst <- iris[-idx.trn, ]
nrow(dat.tst)
nrow(dat.trn)

ks <- c(1, 3, 5, 10, 15, 20, 30, 50, 75, 100)
(k.pick <- f.pick.k(dat=dat.trn, ks=ks))
prd.tst <- class::knn(train=dat.trn[, -5], test=dat.tst[, -5], cl=dat.trn[, 5], k=k.pick)
caret::confusionMatrix(dat.tst[, 5], prd.tst)
caret::confusionMatrix(dat.tst[, 5], prd.tst)$overall[['Kappa']]

```

Once again, but this time, nested cross-validation:

```
f.cv.outer <- function(idx.trn) {
  dat.trn <- iris[idx.trn, ]
  dat.tst <- iris[-idx.trn, ]
  ks <- c(1, 3, 5, 10, 15, 20, 30, 50, 75)
  k.pick <- f.pick.k(dat=dat.trn, ks=ks)
  prd.tst <- class::knn(train=dat.trn[, -5], test=dat.tst[, -5], cl=dat.trn[, 5], k=k.pick)
  caret::confusionMatrix(dat.tst[, 5], prd.tst)$overall[['Kappa']]
}

set.seed(1)
idx <- 1:nrow(iris)
folds <- caret::createMultiFolds(idx, k=7, times=3)
rslt <- sapply(folds, f.cv.outer)
mean(rslt)
sd(rslt)
sd(rslt) / sqrt(length(rslt))

```

### Check your understanding 2

1) question here

[Return to index](#index)

---

### title 3

intro here; what are we tuning? knn-regression helps us develop intuition;
  knn is a smoother of sorts, with bandwidth defined by k; the more it is
  smoothed, the more predictions tend to be pulled towards the global mean:


```
library('caret')

rm(list=ls())
set.seed(1)

dat <- cars
dat$speed <- dat$speed + rnorm(nrow(dat), mean=0, sd=1/2)
plot(dat)

set.seed(1)
nrow(dat)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)

idx.trn <- folds[[1]]
dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

fit.1 <- caret::knnreg(dist ~ speed, data=dat.trn, k=1)
fit.4 <- caret::knnreg(dist ~ speed, data=dat.trn, k=4)
fit.16 <- caret::knnreg(dist ~ speed, data=dat.trn, k=16)
fit.all <- caret::knnreg(dist ~ speed, data=dat.trn, k=nrow(dat.trn))

speed <- seq(from=min(dat$speed), to=max(dat$speed), length.out=10000)
newdata <- data.frame(speed=speed)

prd.1 <- predict(fit.1, newdata=newdata)
prd.4 <- predict(fit.4, newdata=newdata)
prd.16 <- predict(fit.16, newdata=newdata)
prd.all <- predict(fit.all, newdata=newdata)

ylim <- range(c(prd.1, prd.4, prd.16, prd.all, dat$dist))
plot(x=speed, y=prd.1, ylab='dist', ylim=ylim, type='n')
points(x=dat.trn$speed, y=dat.trn$dist, pch='x', col='orangered', cex=0.5)
points(x=dat.tst$speed, y=dat.tst$dist, pch='o', col='magenta', cex=0.5)

lines(x=speed, y=prd.1, lty=2, col='cyan')
lines(x=speed, y=prd.4, lty=3, col='magenta')
lines(x=speed, y=prd.16, lty=2, col='orangered')
lines(x=speed, y=prd.all, lty=3, col='cyan')

legend(
  'topleft',
  legend=c('k=1', 'k=4', 'k=16', 'k=all'),
  lty=c(2, 3, 2, 3),
  col=c('cyan', 'magenta', 'orangered', 'cyan')
)

prd.1 <- predict(fit.1, newdata=dat.tst)
prd.4 <- predict(fit.4, newdata=dat.tst)
prd.16 <- predict(fit.16, newdata=dat.tst)
prd.all <- predict(fit.all, newdata=dat.tst)

res.1 <- dat.tst$dist - prd.1
res.4 <- dat.tst$dist - prd.4
res.16 <- dat.tst$dist - prd.16
res.all <- dat.tst$dist - prd.all

mean(res.1^2)
mean(res.4^2)
mean(res.16^2)
mean(res.all^2)

```

CV it:

```
library('caret')

rm(list=ls())

set.seed(1)
dat <- cars
dat$speed <- dat$speed + rnorm(nrow(dat), mean=0, sd=1/2)

f.test.k <- function(k, idx.trn) {

  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]

  fit <- caret::knnreg(dist ~ speed, data=dat.trn, k=k)
  prd <- predict(fit, newdata=dat.tst)
  res <- dat.tst$dist - prd

  mean(res^2)
}

f.cv <- function(idx.trn, ks) {
  rslt <- sapply(ks, f.test.k, idx.trn=idx.trn)
  names(rslt) <- ks
  rslt
}

set.seed(1)
nrow(dat)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)
idx.trn <- folds[[1]]

ks <- 1:25
rslt <- f.cv(idx.trn, ks)
rslt

rslt <- sapply(folds, f.cv, ks)
m <- apply(rslt, 1, mean)
se <- apply(rslt, 1, sd) / ncol(rslt)
mx <- apply(rslt, 1, max)
idx.min <- which.min(m)
rslt <- data.frame(k=ks, mean=m, se=se, max=mx)
rslt[idx.min, ]
cutoff <- rslt[idx.min, 'mean'] + rslt[idx.min, 'se']
i.good <- rslt[, 'mean'] <= cutoff
rslt[i.good, ]
max(rslt[i.good, 'k'])

par(mfrow=c(1, 1))
plot(mean ~ ks, data=rslt, xlab='number of nearest neighbors', ylab='MSE')
abline(h=cutoff, lty=2, col='orangered')

```

```
library('caret')
library('MASS')

rm(list=ls())

dat <- c(WWWusage)
dat <- data.frame(minute=1:length(dat), users=dat)
plot(dat)
nrow(dat)

fit1 <- caret::knnreg(users ~ minute, data=dat, k=1)
fit10 <- caret::knnreg(users ~ minute, data=dat, k=10)
fit25 <- caret::knnreg(users ~ minute, data=dat, k=25)
fit100 <- caret::knnreg(users ~ minute, data=dat, k=100)

minute <- seq(from=min(dat$minute), to=max(dat$minute), length.out=10000)
newdata <- data.frame(minute=minute)


prd10 <- predict(fit10, newdata=newdata)
prd25 <- predict(fit25, newdata=newdata)
prd100 <- predict(fit100, newdata=newdata)

ylim <- range(c(prd1, prd10, prd25, prd100, dat$users))
plot(x=minute, y=prd1, ylab='users', ylim=ylim, type='n')
points(x=dat$minute, y=dat$users)
lines(x=minute, y=prd1, lty=2, col='cyan')
lines(x=minute, y=prd10, lty=3, col='magenta')
lines(x=minute, y=prd25, lty=2, col='orangered')
lines(x=minute, y=prd100, lty=3, col='cyan')

legend(
  'topleft',
  legend=c('k=1', 'k=10', 'k=25', 'k=100'),
  lty=c(2, 3, 2, 3),
  col=c('cyan', 'magenta', 'orangered', 'cyan')
) 

```

```
library('caret')

rm(list=ls())

dat <- c(WWWusage)
dat <- data.frame(minute=1:length(dat), users=dat)

f.try.k <- function(k, idx.trn) {

  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]

  fit <- caret::knnreg(users ~ minute, data=dat.trn, k=k)
  prd.tst <- predict(fit, newdata=dat.tst)
  res <- prd.tst - dat.tst$users

  c(k=k, bias=mean(res), s2=var(res), mse=mean(res^2))
}

f.cv <- function(idx.trn) {
  ks <- c(1, 2, 4, 8, 16, 32, 64)
  rslt <- sapply(ks, f.try.k, idx.trn=idx.trn)
  colnames(rslt) <- ks
  rslt
}

set.seed(1)
idx <- 1:nrow(iris)
folds <- caret::createMultiFolds(idx, k=7, times=3)
idx.trn <- folds[[1]]
f.cv(idx.trn)

rslt <- sapply(folds, f.cv)

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
