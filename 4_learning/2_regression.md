# Fundamentals of computational data analysis using R
## Machine learning: regression
#### Contact: mitch.kostich@jax.org

---

### Index

- [Lasso and ridge regression](#lasso-and-ridge-regression)
- [Elastic net regularization](#elastic-net-regularization)
- [Support vector machines](#support-vector-machines)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Lasso and ridge regression

intro here

```
library(glmnet)
library(caret)

rm(list=ls())

dat <- mtcars

set.seed(1)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)
idx.trn <- folds[[1]]

dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

plot(dat.trn)
summary(dat.trn)
round(cor(dat.trn), 3)

fit.lm.lo <- lm(mpg ~ 1, data=dat.trn)
fit.lm.hi <- lm(mpg ~ ., data=dat.trn)
fit.lm <- step(fit.lm.lo, scope=list(lower=fit.lm.lo, upper=fit.lm.hi), direction='both', trace=1)

set.seed(1)                                      ## for CV randomness
cv.ridge <- cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=0)
cv.lasso <- cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=1)

cv.ridge
is.list(cv.ridge)
names(cv.ridge)
all(names(cv.lasso) == names(cv.ridge))
str(cv.ridge)

par(mfrow=c(1, 2))
plot(cv.ridge)
plot(cv.lasso)

fit.ridge <- cv.ridge$glmnet.fit
fit.ridge                                   ## note Df (number of non-zero coefs) always same (no feature selection)
is.list(fit.ridge)
names(fit.ridge)
str(fit.ridge)

fit.lasso <- cv.lasso$glmnet.fit
all(names(fit.lasso) == names(fit.ridge))
fit.lasso                                   ## note Df changing (feature selection

par(mfrow=c(1, 2))
plot(fit.ridge, xvar='lambda', label=T, main='ridge')
plot(fit.lasso, xvar='lambda', label=T, main='lasso')

## smooth, so pick min; can also specify cutoff as s="lambda.1se" or s=1.5
coef(fit.ridge, s=cv.ridge$lambda.min)
coef(fit.lasso, s=cv.lasso$lambda.min)
coef(fit.lm)

y.lm <- predict(fit.lm, newdata=dat.tst[, -1]) 
y.ridge <- predict(fit.ridge, newx=as.matrix(dat.tst[, -1]), s=cv.ridge$lambda.min)
y.lasso <- predict(fit.lasso, newx=as.matrix(dat.tst[, -1]), s=cv.lasso$lambda.min)

mean((y.lm - dat.tst[, 1]) ^ 2)
mean((y.ridge - dat.tst[, 1]) ^ 2)
mean((y.lasso - dat.tst[, 1]) ^ 2)

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Elastic net regularization

intro here; finds coefficients for linear model that minimizes the sum of two terms. 
  the first term is the negative of the log-likelihood of the model given
  the training set observations; the second is the **elastic-net penalty** 
  `lambda * ((1 - alpha) * sum(beta^2) / 2 + alpha * sum(abs(beta))`, where `lambda` 
  is the a tunable parameter to control the relative strength of the coefficient penalty,
  and alpha is the tunable 'mixing parameter' controlling the proportion of the penalty 
  that is contributed by the ridge-like component `sum(beta^2)` versus the lasso-like
  component `sum(abs(beta))`. That is, `alpha == 1` corresponds to a pure lasso regression,
  while `alpha == 0` corresponds to a pure ridge regression. Elastic net regularization
  allows ...

```
library(glmnet)
library(caret)

rm(list=ls())

data(tecator)
dim(absorp)               ## IR absorbance at 100 wavelengths
dim(endpoints)            ## moisture, fat, protein

dat <- cbind(endpoints[, 1], absorp)
dat <- data.frame(dat)
names(dat) <- c('y', paste('x', 1:ncol(absorp), sep=''))
dat[1:5, 1:5]

set.seed(1)
idx <- 1 : nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)
idx.trn <- folds[[1]]

dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

dim(dat.trn)

fit.lm.lo <- lm(y ~ 1, data=dat.trn)
fit.lm.hi <- lm(y ~ ., data=dat.trn)
fit.lm <- step(fit.lm.lo, scope=list(lower=fit.lm.lo, upper=fit.lm.hi), direction='both', trace=1)
summary(fit.lm)

f.alpha <- function(alpha.i) {
  cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=alpha.i)
}

(alphas <- seq(from=0, to=1, by=0.2))
set.seed(1)                                      ## for tuning CV randomness
cvs <- lapply(alphas, f.alpha)
names(cvs) <- paste('s', alphas, sep='')

par(mfrow=c(2, 3))
sapply(cvs, plot)

f.fit <- function(cv.i) cv.i$glmnet.fit
fits <- lapply(cvs, f.fit)
sapply(fits, plot, xvar='lambda', label=T)

f.coef <- function(idx) {
  fit.i <- fits[[idx]]
  cv.i <- cvs[[idx]]
  coef.i <- coef(fit.i, s=cv.i$lambda.min)
  as.numeric(coef.i)
}
coefs <- sapply(1:length(fits), f.coef)
colnames(coefs) <- names(cvs)
round(coefs, 3)
coef(fit.lm)

apply(coefs, 2, summary)
summary(coef(fit.lm))

f.prd <- function(idx) {
  predict(fits[[idx]], newx=as.matrix(dat.tst[, -1]), s=cvs[[idx]]$lambda.min)
}
ys <- sapply(1:length(fits), f.prd)
y.lm <- predict(fit.lm, newdata=dat.tst[, -1]) 

f.mse <- function(y.i) {
  mean((y.i - dat.tst[, 1]) ^ 2)
}
mses <- apply(ys, 2, f.mse)
mse.lm <- f.mse(y.lm)

```

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Support vector machines

intro here; margin maximization; support vectors are observations on margin; 
  hinge loss maximizes margin; hinge loss zero for correctly classified; 
  only misclassified points and the support vectors matter. Score cutoffs are
  -1 and 1, not 0.5; the margin extends from -1 to 1.

kernel trick example:

```
library(caret)
library(glmnet)
library(e1071)
library(pROC)

rm(list=ls())

set.seed(1)
n <- 60
s1 <- 10
s2 <- 1
x1 <- rnorm(n, mean=0, sd=s1)
x2 <- rnorm(n, mean=0, sd=s1)

r <- x1^2 + x2^2
y <- rep('A', n)
y[r > median(r)] <- 'B'
y <- factor(y)
x1 <- x1 + rnorm(n, mean=0, sd=s2)
x2 <- x2 + rnorm(n, mean=0, sd=s2)
i.A <- y == 'A'
table(y)

dat <- data.frame(y=y, x1=x1, x2=x2)
par(mfrow=c(1, 1))
plot(x=x1, y=x2, type='n')
points(x1[i.A], x2[i.A], pch='o', col='orangered')
points(x1[!i.A], x2[!i.A], pch='x', col='magenta')

set.seed(1)
idx <- 1 : nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=3)
idx.trn <- folds[[1]]

dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

cv.net <- glmnet::cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=0.5, family='binomial')
fit.net <- cv.net$glmnet.fit
(prd.net <- predict(fit.net, newx=as.matrix(dat.tst[, -1]), s=cv.net$lambda.min, type='response'))
(prd.net <- prd.net[, 1])
par(mfrow=c(1, 2))
plot(cv.net)
plot(fit.net, xvar='lambda', label=T)

fit.svm1 <- e1071::svm(y ~ ., data=dat.trn, probability=T)
(prd.svm1 <- predict(fit.svm1, newdata=dat.tst[, -1], probability=T, decision.values=T))
(prd.svm1 <- attr(prd.svm1, 'probabilities')[, 'A'])
par(mfrow=c(1, 1))
plot(fit.svm1, data=dat.trn)      ## color is class; 'x' is support vector

cv.svm <- e1071::tune.svm(y ~ ., data=dat.trn, gamma=2^(-2:2), cost=2^(1:5), probability=T)
summary(cv.svm)
plot(cv.svm)
fit.svm2 <- cv.svm$best.model
(prd.svm2 <- predict(fit.svm2, newdata=dat.tst[, -1], probability=T, decision.values=T))
(prd.svm2 <- attr(prd.svm2, 'probabilities')[, 'A'])
par(mfrow=c(1, 1))
plot(fit.svm2, data=dat.trn)      ## color is class; 'x' is support vector

pROC::roc(dat.tst$y == 'A', prd.net, direction='<')$auc
pROC::roc(dat.tst$y == 'A', prd.svm1, direction='<')$auc
pROC::roc(dat.tst$y == 'A', prd.svm2, direction='<')$auc

prd.net                           ## all predictions identical (similar to the global mean)

```

```
library(caret)
library(pROC)
library(glmnet)
library(e1071)

data(dhfr)                        ## from caret
dat <- dhfr
table(dat$Y)
class(dat$Y)

set.seed(1)
idx <- 1 : nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=3)
idx.trn <- folds[[1]]

dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

cv.net <- glmnet::cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=0.5, family='binomial')
fit.net <- cv.net$glmnet.fit
(prd.net <- predict(fit.net, newx=as.matrix(dat.tst[, -1]), s=cv.net$lambda.min, type='response'))
(prd.net <- prd.net[, 1])
par(mfrow=c(1, 2))
plot(cv.net)
plot(fit.net, xvar='lambda', label=T)

cv.svm <- e1071::tune.svm(Y ~ ., data=dat.trn, gamma=2^(-2:2), cost=2^(1:5), probability=T)
summary(cv.svm)
plot(cv.svm)
fit.svm <- cv.svm$best.model

(prd.svm <- predict(fit.svm, newdata=dat.tst[, -1], probability=T, decision.values=T))
(prd.svm <- attr(prd.svm, 'probabilities')[, 'active'])

pROC::roc(dat.tst$Y == 'active', prd.net, direction='>')
pROC::roc(dat.tst$Y == 'active', prd.svm, direction='<')

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
