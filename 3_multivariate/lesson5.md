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

Linear relationship of `y` or some transformation of `y`. Normal homoskedastic errors.
  Can add polynomials and interactions on predictors. Can transform response, but
  this leads to analysis on unnatural scale which complicates interpretation. We are
  also relying on a single transformation being capable of both linearizing a relationship
  as well as making the residuals normal and homoskedastic. That is often not possible.
  Generalized linear modeling allows for specification (via a **'link function'**) 
  of an invertible (reversible) transformation for the response as well specification 
  (via a **'variance function'**) of nearly arbitrary error models. Having separate 
  functions for linearization and for describing the error model greatly increases
  the flexibility.

For instance, a binary response variable `y` is a random variable that can only take on 
  the values `1` or `0`, where the parameter `p` is the probability that it will take on 
  the value `1`. The two values may be used to encode states/categories of interest, 
  such as 'heads' or 'tails', if modeling coin flips. Or they could be used to represent 
  'affected' or 'non-affected' subjects in a health-related experiment. We could model
  the mean value of `p` as a function of some predictors, `x1`, `x2`, etc. When doing this,
  we want the conditional mean `p` to always lie within the interval `[0, 1]`. The 
  variance of a binary variable with a conditional mean of `p` will be `p * (1 - p)`: 

```
rm(list=ls())
set.seed(1)

p <- seq(from=0, to=1, by=0.01)
f <- function(p.i) var(rbinom(1e5, 1, p.i))
s2 <- sapply(p, f)

par(mfrow=c(1, 1))
plot(x=p, y=s2)

y <- p * (1 - p)
lines(x=p, y=y)

```

In order to restrict ... model logit(y) = log(p / (1 - p)) as the response; only defined in 
  closed (does not include endpoints) interval `(0, 1)`. Remember, we are modeling the conditional
  mean probability of getting a `1`. Individual observations are always exactly `0` or `1`, but
  we assume there is always a non-zero probability of getting either class. That is, nowhere is
  the range of the predictors does the model predict a zero probability of observing a `0`. Even
  if a `1` is modeled as far, far more likely than observing a `0`, there will always be some
  chance of observing a `1`. 

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
  this criterion may not make much sense. 

Likelihood of an observation given a model: if our model is a global mean (intercept-only linear model),
  and we assume that the data are normally distributed (as in the case of a single-sample
  t-test), then if we assume the model is correct, we can estimate the probability of observing
  a value a certain distance from the mean. For instance, if we estimate that the global mean
  and standard deviation from the sample, and we assume the data are normally distributed, we
  can predict the probability of observing any particular value:

```
rm(list=ls())
set.seed(1)

x <- rnorm(30, mean=10, sd=2)
(m <- mean(x))
(s <- sd(x))

x1 <- seq(from=0, to=20, by=0.001)
p <- dnorm(x1, mean=m, sd=s)
par(mfrow=c(1, 1))
plot(x=x1, y=p, type='l', xlab='x', ylab='probability', main='p(x) for N(10.2, 1.8)')

x2 <- seq(from=0, to=20, by=1)
cbind(x=x2, p=dnorm(x2, mean=m, sd=s))

```

We can flip this idea on its head to estimate the mean based on the estimated sd in such 
  a way as to maximize the probability of the observations given the model:

```
(s <- sd(x))

f.loglik <- function(m) {
  p.obs <- dnorm(x, mean=m, sd=s)
  sum(log(p.obs))
}

m.try <- seq(from=0, to=20, by=0.001)
loglik <- sapply(m.try, f.loglik)

(m.best <- m.try[which.max(loglik)])
mean(x)
t.test(x)

par(mfrow=c(1, 1))
plot(x=m.try, y=loglik, type='l')
abline(v=m.best, lty=2)
rug(x)

```

Deviance for one observation: dev(y, pred) = 2 * log(p(y | fit.saturated)) - log(p(y | fit))
For comparing fits, drop constant part: dev(y, pred) = -log(p(y | fit)) == -loglik(y | fit)
Model deviance is sum of dev(y, pred) for all y: -sum(loglik(y | fit))
Instead of minimizing the sums-of-squared deviations, we find model coefficients that 
  minimize the deviance.

AIC in terms of likelihoods.

[Return to index](#index)

---

### Logistic regression

intro here;

can specify the response several ways.

```
rm(list=ls())

summary(esoph)
nrow(esoph)
head(esoph)

par(mfrow=c(1, 1))
plot(esoph)

(fit <- glm(cbind(ncases, ncontrols) ~ ., data=esoph, family='binomial'))
summary(fit1)
anova(fit1)
deviance(fit1)

p1 <- fitted(fit1)

## on logit scale, not probability scale:
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

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Poisson regression

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

### Negative-binomial regression

intro here

```
code here

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
