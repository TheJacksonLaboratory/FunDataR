# Fundamentals of computational data analysis using R
## Multivariate statistics: multiple regression
#### Contact: mitch.kostich@jax.org

---

### Index

- [Multiple regression](#multiple-regression)
- [Correlated predictors](#correlated-predictors)
- [Interactions](#interactions)
- [Hierarchical models](#hierarchical-models)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Multiple regression

intro here

```
library('caret')
rm(list=ls())
set.seed(1)

summary(wtloss)

par(mfrow=c(1, 1))
plot(wtloss)

fit1 <- lm(Weight ~ Days, data=wtloss)
summary(fit1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

## residuals vs. fitted clearly suggests second order (squared)
##   but residuals look homoskedastic, so don't want to mess 
##   with Weight.

fit2 <- lm(Weight ~ I(Days ^ 2), data=wtloss)
summary(fit2)
par(mfrow=c(2, 3))
plot(fit2, which=1:6)

fit3 <- lm(Weight ~ Days + I(Days ^ 2), data=wtloss)
summary(fit3)
par(mfrow=c(2, 3))
plot(fit3, which=1:6)

f <- function(idx.trn) {
  dat.trn <- wtloss[idx.trn, ]
  dat.tst <- wtloss[-idx.trn, ]
  fit1 <- lm(Weight ~ Days, data=dat.trn)
  fit2 <- lm(Weight ~ I(Days ^ 2), data=dat.trn)
  fit3 <- lm(Weight ~ Days + I(Days ^ 2), data=dat.trn)
  prd1 <- predict(fit1, newdata=dat.tst)
  prd2 <- predict(fit2, newdata=dat.tst)
  prd3 <- predict(fit3, newdata=dat.tst)
  mse1 <- mean((dat.tst$Weight - prd1) ^ 2, na.rm=T)
  mse2 <- mean((dat.tst$Weight - prd2) ^ 2, na.rm=T)
  mse3 <- mean((dat.tst$Weight - prd3) ^ 2, na.rm=T)
  c(mse1=mse1, mse2=mse2, mse3=mse3)
}

k <- 5
times <- 3
idx <- 1:nrow(wtloss)
folds <- createMultiFolds(idx, k=k, times=times)
rslt <- sapply(folds, f)
apply(rslt, 1, mean)
apply(rslt, 1, sd)

```

Another example:

```
rm(list=ls())
set.seed(1)

fit0 <- lm(Volume ~ 1, data=trees)
fit1 <- lm(Volume ~ Height, data=trees)
fit2 <- lm(Volume ~ Girth, data=trees)
fit3 <- lm(Volume ~ Girth + I(Girth ^ 2), data=trees)
fit4 <- lm(Volume ~ Girth + I(Girth ^ 2) + Height, data=trees)

summary(fit0)
summary(fit1)
summary(fit2)
summary(fit3)
summary(fit4)

par(mfrow=c(2, 3))
plot(fit0, which=1:6)
plot(fit1, which=1:6)
plot(fit2, which=1:6)
plot(fit3, which=1:6)
plot(fit4, which=1:6)

f <- function(idx.trn) {
  dat.trn <- trees[idx.trn, ]
  dat.tst <- trees[-idx.trn, ]
  fit0 <- lm(Volume ~ 1, data=dat.trn)
  fit1 <- lm(Volume ~ Height, data=dat.trn)
  fit2 <- lm(Volume ~ Girth, data=dat.trn)
  fit3 <- lm(Volume ~ Girth + I(Girth ^ 2), data=dat.trn)
  fit4 <- lm(Volume ~ Girth + I(Girth ^ 2) + Height, data=dat.trn)
  prd0 <- predict(fit0, newdata=dat.tst)
  prd1 <- predict(fit1, newdata=dat.tst)
  prd2 <- predict(fit2, newdata=dat.tst)
  prd3 <- predict(fit3, newdata=dat.tst)
  prd4 <- predict(fit4, newdata=dat.tst)
  mse0 <- mean((dat.tst$Volume - prd0) ^ 2, na.rm=T)
  mse1 <- mean((dat.tst$Volume - prd1) ^ 2, na.rm=T)
  mse2 <- mean((dat.tst$Volume - prd2) ^ 2, na.rm=T)
  mse3 <- mean((dat.tst$Volume - prd3) ^ 2, na.rm=T)
  mse4 <- mean((dat.tst$Volume - prd4) ^ 2, na.rm=T)
  c(mse0=mse0, mse1=mse1, mse2=mse2, mse3=mse3, mse4=mse4)
}

k <- 7
times <- 5
idx <- 1:nrow(wtloss)
folds <- createMultiFolds(idx, k=k, times=times)
rslt <- sapply(folds, f)
apply(rslt, 1, mean)
apply(rslt, 1, sd)

```

[Return to index](#index)

---

### Correlated predictors

intro here

```
code here

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Interactions

intro here

```
library('caret')
rm(list=ls())

k <- 5
times <- 3

f <- function(idx.trn) {

  dat.trn <- ChickWeight[idx.trn, ]
  dat.tst <- ChickWeight[-idx.trn, ]

  fit1 <- lm(weight ~ Time, data=dat.trn)
  fit2 <- lm(log(weight) ~ Time, data=dat.trn)
  fit3 <- lm(log(weight) ~ Time * Diet, data=dat.trn)
  fit4 <- lm(log(weight) ~ Time / Diet, data=dat.trn)

  prd1 <- predict(fit1, newdata=dat.tst)
  prd2 <- exp(predict(fit2, newdata=dat.tst))
  prd3 <- exp(predict(fit3, newdata=dat.tst))
  prd4 <- exp(predict(fit4, newdata=dat.tst))

  mse1 <- mean((dat.tst$weight - prd1) ^ 2, na.rm=T)
  mse2 <- mean((dat.tst$weight - prd2) ^ 2, na.rm=T)
  mse3 <- mean((dat.tst$weight - prd3) ^ 2, na.rm=T)
  mse4 <- mean((dat.tst$weight - prd4) ^ 2, na.rm=T)

  c(mse1=mse1, mse2=mse2, mse3=mse3, mse4=mse4)
}

idx <- 1 : nrow(ChickWeight)
folds <- createMultiFolds(idx, k=k, times=times)
mse <- sapply(folds, f)
apply(mse, 1, mean)
apply(mse, 1, sd)

fit1 <- lm(weight ~ Time, data=ChickWeight)
summary(fit1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

fit2 <- lm(log(weight) ~ Time, data=ChickWeight)
summary(fit2)
par(mfrow=c(2, 3))
plot(fit2, which=1:6)

fit3 <- lm(log(weight) ~ Time * Diet, data=ChickWeight)
summary(fit3)
par(mfrow=c(2, 3))
plot(fit3, which=1:6)

fit4 <- lm(log(weight) ~ Time / Diet, data=ChickWeight)
summary(fit4)
par(mfrow=c(2, 3))
plot(fit4, which=1:6)

```

### Hierarchical models

intro here

```
code here

```

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Mixed effects models

intro here

```
library('nlme')
library('caret')
rm(list=ls())

k <- 5
times <- 3

f <- function(idx.trn) {

  dat.trn <- ChickWeight[idx.trn, ]
  dat.tst <- ChickWeight[-idx.trn, ]

  fit1 <- lm(weight ~ Time, data=dat.trn)
  fit2 <- lm(log(weight) ~ Time, data=dat.trn)
  fit3 <- lm(log(weight) ~ Time * Diet, data=dat.trn)
  fit4 <- lm(log(weight) ~ Time / Diet, data=dat.trn)

  prd1 <- predict(fit1, newdata=dat.tst)
  prd2 <- exp(predict(fit2, newdata=dat.tst))
  prd3 <- exp(predict(fit3, newdata=dat.tst))
  prd4 <- exp(predict(fit4, newdata=dat.tst))

  mse1 <- mean((dat.tst$weight - prd1) ^ 2, na.rm=T)
  mse2 <- mean((dat.tst$weight - prd2) ^ 2, na.rm=T)
  mse3 <- mean((dat.tst$weight - prd3) ^ 2, na.rm=T)
  mse4 <- mean((dat.tst$weight - prd4) ^ 2, na.rm=T)

  c(mse1=mse1, mse2=mse2, mse3=mse3, mse4=mse4)
}

idx <- 1 : nrow(ChickWeight)
folds <- createMultiFolds(idx, k=k, times=times)
mse <- sapply(folds, f)
apply(mse, 1, mean)
apply(mse, 1, sd)

fit1 <- lm(weight ~ Time, data=ChickWeight)
summary(fit1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

fit2 <- lm(log(weight) ~ Time, data=ChickWeight)
summary(fit2)
par(mfrow=c(2, 3))
plot(fit2, which=1:6)

fit3 <- lm(log(weight) ~ Time * Diet, data=ChickWeight)
summary(fit3)
par(mfrow=c(2, 3))
plot(fit3, which=1:6)

fit4 <- lm(log(weight) ~ Time / Diet, data=ChickWeight)
summary(fit4)
par(mfrow=c(2, 3))
plot(fit4, which=1:6)

```

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
