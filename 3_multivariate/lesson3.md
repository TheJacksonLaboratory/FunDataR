# Fundamentals of computational data analysis using R
## Multivariate statistics: multiple regression
#### Contact: mitch.kostich@jax.org

---

### Index

- [Multiple regression](#multiple-regression)
- [Correlated predictors](#correlated-predictors)
- [Interactions](#interactions)
- [Hierarchical models](#hierarchical-models)
- [Mixed models](#mixed-models)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Multiple regression

intro here

```
library('caret')
library('MASS')
rm(list=ls())
set.seed(1)

summary(wtloss)

par(mfrow=c(1, 1))
plot(wtloss)

fit1 <- lm(Weight ~ Days, data=wtloss)
smry1 <- summary(fit1)
coef(smry1)
smry1$adj.r.squared
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

## residuals vs. fitted clearly suggests second order (squared)
##   but residuals look homoskedastic, so don't want to mess 
##   with Weight.

fit2 <- lm(Weight ~ I(Days ^ 2), data=wtloss)
smry2 <- summary(fit2)
coef(smry2)
smry2$adj.r.squared
par(mfrow=c(2, 3))
plot(fit2, which=1:6)

## Residuals vs Fitted plot suggests model still has wrong shape

fit3 <- lm(Weight ~ Days + I(Days ^ 2), data=wtloss)
smry3 <- summary(fit3)
coef(smry3)
smry3$adj.r.squared
par(mfrow=c(2, 3))
plot(fit3, which=1:6)             ## Much better!

## plot the fits: 

## for plotting: predictions on dense grid of Days:
days <- seq(from=min(wtloss$Days), to=max(wtloss$Days), length.out=10000)
newdata <- data.frame(Days=days)
prd1 <- predict(fit1, newdata=newdata)
prd2 <- predict(fit2, newdata=newdata)
prd3 <- predict(fit3, newdata=newdata)

par(mfrow=c(1, 1))
plot(Weight ~ Days, data=wtloss)
lines(x=days, y=prd1, lty=2, col='cyan')
lines(x=days, y=prd2, lty=3, col='magenta')
lines(x=days, y=prd3, lty=4, col='orangered')
legend(
  'topright', 
  legend=c('linear', 'quadratic', 'both'), 
  col=c('cyan', 'magenta', 'orangered'),
  lty=c(2, 3, 4)
)

## cross-validate the models:

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

k <- 10                           ## number of folds
times <- 7                        ## number of repetitions
idx <- 1:nrow(wtloss)             ## vector of index positions of observations
folds <- createMultiFolds(idx, k=k, times=times)
rslt <- sapply(folds, f)          ## folds is list of indices to training-set observations
apply(rslt, 1, mean)              ## average mse over folds
apply(rslt, 1, sd)                ## spread of results; stability of performance

```

Another example:

```
rm(list=ls())

par(mfrow=c(1, 1))
plot(mtcars)

## break out some interesting looking variables to predict mpg:
plot(mtcars[, c('mpg', 'wt', 'disp', 'hp')])

## correlation between variables:
cor(mtcars[, c('mpg', 'wt', 'disp', 'hp')])

fit1 <- lm(mpg ~ wt, data=mtcars)
summary(fit1)                     ## wt coefficient significant
par(mfrow=c(2, 3))
plot(fit1, which=1:6)             ## Residuals vs Fitted suggests quadratic

## add wt^2 quadratic component:
fit2 <- lm(mpg ~ wt + I(wt ^ 2), data=mtcars)
summary(fit2)                     ## adj.r.squared improves, wt^2 coef significant
par(mfrow=c(2, 3))
plot(fit2, which=1:6)             ## residuals much better, though not normal and influence so so

fit3 <- lm(mpg ~ wt + I(wt ^ 2) + disp, data=mtcars)
summary(fit3)                     ## slightly better adj.r.squared, but disp barely signif
par(mfrow=c(2, 3))
plot(fit3, which=1:6)             ## residuals about the same as for fit2

fit4 <- lm(mpg ~ wt + I(wt ^ 2) + hp, data=mtcars)
summary(fit4)                     ## bigger jump in adj.r.squared; hp coef significant
par(mfrow=c(2, 3))
plot(fit4, which=1:6)             ## plot still looks ok

fit5 <- lm(mpg ~ wt + I(wt ^ 2) + hp + disp, data=mtcars)
summary(fit5)                     ## disp coef still not significant; fit4 is better
par(mfrow=c(2, 3))
plot(fit5, which=1:6)             ## plot still looks ok

```

So what happened? disp had higher correlation with mpg than hp, but hp ended up 
  being a more useful predictor.

Since residual plots seem weird, what do CV have to say about this? 

```
library(caret)
set.seed(1)

f.mse <- function(frm, dat.trn, dat.tst) {
  fit <- lm(frm, data=dat.trn)
  prd <- predict(fit, newdata=dat.tst)
  mse <- mean((dat.tst$mpg - prd) ^ 2, na.rm=T)
}

f <- function(idx.trn) {
  dat.trn <- mtcars[idx.trn, ]
  dat.tst <- mtcars[-idx.trn, ]
  mse0 <- f.mse(mpg ~ 1, dat.trn, dat.tst)
  mse1 <- f.mse(mpg ~ wt, dat.trn, dat.tst)
  mse2 <- f.mse(mpg ~ wt + I(wt ^ 2), dat.trn, dat.tst)
  mse3 <- f.mse(mpg ~ wt + I(wt ^ 2) + disp, dat.trn, dat.tst)
  mse4 <- f.mse(mpg ~ wt + I(wt ^ 2) + hp, dat.trn, dat.tst)
  mse5 <- f.mse(mpg ~ wt + I(wt ^ 2) + hp + disp, dat.trn, dat.tst)
  c(mse0=mse0, mse1=mse1, mse2=mse2, mse3=mse3, mse4=mse4, mse=mse5)
}

k <- 10
times <- 7
idx <- 1:nrow(dat)

## make list of k * times folds; elements are training-set indices:
folds <- createMultiFolds(idx, k=k, times=times)

rslt <- sapply(folds, f)          ## run f() on each training-set index in folds
apply(rslt, 1, mean)              ## best: mpg ~ wt + I(wt ^ 2) + hp
apply(rslt, 1, sd)                ## best: mpg ~ wt + I(wt ^ 2) + hp; but SEs are large!

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Correlated predictors

intro here

```
rm(list=ls())

## some fits we've seen:
fit1 <- lm(mpg ~ wt, data=mtcars)
fit2 <- lm(mpg ~ wt + I(wt ^ 2), data=mtcars)
fit3 <- lm(mpg ~ wt + I(wt ^ 2) + disp, data=mtcars)
fit4 <- lm(mpg ~ wt + I(wt ^ 2) + hp, data=mtcars)
fit5 <- lm(mpg ~ wt + I(wt ^ 2) + hp + disp, data=mtcars)

## correlation between variables:
dat <- mtcars[, c('wt', 'disp', 'hp')]
dat$wt2 <- dat$wt ^ 2
dat <- dat[, c('wt', 'wt2', 'disp', 'hp')]
cor(dat)

coef(summary(fit1))
coef(summary(fit2))
coef(summary(fit3))
coef(summary(fit4))
coef(summary(fit5))

```

A synthetic example:

```
rm(list=ls())
set.seed(1)

x1 <- runif(100, 0, 1)
x2 <- runif(100, 0, 1)
e <- rnorm(100, 0, 0.1)
y <- x1 + x2 + e
cor(cbind(y, x1, x2, e))          ## x1, x2 both correlated w/ y, but not each other

fit1 <- lm(y ~ x1)
fit2 <- lm(y ~ x2)
fit3 <- lm(y ~ x1 + x2) 
coef(summary(fit1))               ## coefficient estimate about right
coef(summary(fit2))               ## coefficient estimate about right
coef(summary(fit3))               ## improved coef estimates, std errors, and p-values!!!

x1 <- runif(100, 0, 1)
x2 <- 0.95 * x1 + 0.05 * runif(100, 0, 1)
y <- x1 + x2 + e
cor(cbind(y, x1, x2, y))          ## x1 and x2 highly correlated; info redundant!

fit1 <- lm(y ~ x1)
fit2 <- lm(y ~ x2)
fit3 <- lm(y ~ x1 + x2)
coef(summary(fit1))               ## coefficient estimate way off; 
coef(summary(fit2))               ## coefficient estimate way off; 
coef(summary(fit3))               ## standard errors, t-values, p-values much worse

```

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Interactions

intro here

synthetic example:

```
rm(list=ls())
set.seed(1)

h <- runif(100, min=1, max=10)    ## heights
w <- runif(100, min=1, max=10)    ## widths
e <- rnorm(100, 0, 0.2)           ## 'measurement error'
a <- h * w + e                    ## measurements of rectangle areas
dat <- data.frame(area=a, h=h, w=w)

fit1 <- lm(area ~ h + w, data=dat)
summary(fit1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)             ## strong quadratic component

fit2 <- lm(area ~ h + w + I(h^2) + I(w^2), data=dat)
summary(fit2)                     ## quadratic terms non-significant
plot(fit2, which=1:6)             ## did not fix the issue w/ fit1

## we know this works; note only 2 coeficients
fit3 <- lm(area ~ I(h * w), data=dat)
summary(fit3)                     ## only intrcpt + I(h*w) coefs
plot(fit3, which=1:6)             ## pretty good

## adds h, w and interaction between h and w (4 coefficients total):
fit4 <- lm(area ~ h * w, data=dat)
summary(fit4)                     ## intrcpt, h, w and h:w interaction
plot(fit4, which=1:6)             ## pretty good

## add all 2-way interactions (since only h and w, same as fit4):
fit5 <- lm(area ~ . ^ 2, data=dat)
summary(fit5)
plot(fit5, which=1:6)

## explicitly include interaction term (same as fit4):
fit6 <- lm(area ~ h + w + h:w, data=dat)
summary(fit6)
plot(fit6, which=1:6)

```

a real example with a smaller but still significant interaction 
  between a numeric and categorical variable:

```
rm(list=ls())

dat <- mtcars[, c('mpg', 'wt', 'gear')]
dat$gear <- factor(dat$gear)
summary(dat)
par(mfrow=c(1, 1))
plot(dat)

fit1 <- lm(mpg ~ wt, data=dat)
summary(fit1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

fit2 <- lm(mpg ~ wt + gear, data=dat)
summary(fit2)
par(mfrow=c(2, 3))
plot(fit2, which=1:6)

fit3 <- lm(mpg ~ wt * gear, data=dat)
summary(fit3)
par(mfrow=c(2, 3))
plot(fit3, which=1:6)

fit4 <- lm(mpg ~ wt / gear, data=dat)
summary(fit4)
par(mfrow=c(2, 3))
plot(fit4, which=1:6)

```

Since non-normal residuals, lets look to CV for corroboration:

```
library('caret')
set.seed(1)

f <- function(idx.trn) {

  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]

  fit1 <- lm(mpg ~ wt, data=dat.trn)
  fit2 <- lm(mpg ~ wt + gear, data=dat.trn)
  fit3 <- lm(mpg ~ wt * gear, data=dat.trn)

  prd1 <- predict(fit1, newdata=dat.tst)
  prd2 <- predict(fit2, newdata=dat.tst)
  prd3 <- predict(fit3, newdata=dat.tst)

  mse1 <- mean((dat.tst$mpg - prd1) ^ 2, na.rm=T)
  mse2 <- mean((dat.tst$mpg - prd2) ^ 2, na.rm=T)
  mse3 <- mean((dat.tst$mpg - prd3) ^ 2, na.rm=T)

  c(mse1=mse1, mse2=mse2, mse3=mse3)
}

k <- 10                           ## since n >> p, bias likely low
times <- 7                        ## repeat k-fold CV this many times

idx <- 1 : nrow(dat)
folds <- createMultiFolds(idx, k=k, times=times)
mse <- sapply(folds, f)
apply(mse, 1, mean)               ## improvement small, but appears real
apply(mse, 1, sd)                 ## also corroborates improvement

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
