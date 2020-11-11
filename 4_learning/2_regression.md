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

In the previous lesson, we saw how tuning the modeling parameter `k` specifying the number of
  nearest-neighbors used for nearest-neighbor regression changed the ability of the model
  to fit the data. At one extreme, setting `k=1` resulted in a very flexible model that perfectly
  fit the training-set observations response values, inclusing the random 'noise' in the response
  variable. This extreme resembles what happens when introducing the same number of coefficients 
  as training-set observations in a linear regression. At the other extreme, setting `k` to the 
  number of training set observations resulted in a very inflexible model that makes the same 
  prediction, corresponding to the global mean of the response in the training observations, for 
  any observations (new or old). This extreme resembles the 'intercept-only' linear model, which 
  also yields predictions equal to the global mean in the training-set for any observations. In 
  the context of linear modeling, using the global mean for prediction is the same as setting all 
  coefficients other than the intercept to zero, which corresponds to the assertion that the 
  features (predictors) are extraneous and not helpful for predicting the response.

Setting the number of neighbors `k` too low allows the model enough flexibility to fit the noise in 
  the training set. This results in low reproducibility, since the noise varies between different 
  randomly drawn potential training sets. So we say that this results in a model with high variance: 
  fitting the model to different training sets will result in relatively widely varying predictions. 
  Setting `k` too high makes the model less likely to fit the noise in the training set, but also 
  makes it more difficult for the model to follow the true underlying pattern in the relationship 
  between the response and the predictors. The 'stiffer' we make the model, the more we force the 
  model to depart the true relationship in a systematic way (it will tend to 'straighten' and 
  'flatten'). Since the departure is systematic, a model that is too stiff is biased. So a major 
  focus of parameter tuning in the context of machine learning is controlling a trade-off between 
  a model whose variance is too high because too much flexibility was allowed, and a model that is 
  too biased because not enough flexibility was allowed. 

Traditional statistics has focused on the development of unbiased estimators, particularly the unbiased 
  estimators with the lowest variance. This is a satisfying approach, because it assures us that on average, 
  we are getting the right answer, and the dispersion around the right answer will be smaller than any other
  estimator which on average gets the right answer. As the number of predictors becomes large relative to 
  the number of training observations, model variance becomes a growing problem. One innovation introduced 
  to deal with this issue was the popularization of certain biased estimators which can have much lower 
  variance than their unbiased counterparts. The resulting models, although biased, could still have a much 
  lower overall error because of greatly reduce model variance, and that variance is often the dominant 
  component of the total error when the number of coefficients (or model parameters) is large relative to 
  the number of training-set observations.

In the context of linear regression, we can 'stiffen' the model either by reducing the number of coefficients
  by setting them equal to zero (straightening), or (less dramatically) by 'shrinking' the coefficients towards 
  zero (flattening/reducing slope). In general, this process is referred to as **regularization** or
  **shrinkage**. The extremely regularized 'intercept-only' model and k-nearest neighbors model with
  `k=length(training.set)` both result in a straight, totally flat prediction line at the level of the global 
  mean of the training-set responses. Regularization of linear regression models involves biasing the
  coefficient estimates towards zero in order to model estimates across different potential training-sets
  more consistent. The two most popular approaches for doing this are the closely related **lasso regression** 
  and **ridge regression** methods. Both methods work by altering the fitting criterion for a linear model.
  For instance, ordinary unweighted linear regression selects coefficient values that minimizes the MSE 
  `mean((y.trn - y.prd) ^ 2`, `y.trn` are the actual values for the response in a training set observations,
  and `y.prd` are the correspondng values predicted by the model. The lasso instead minimizes the term 
  `mean((y.trn - y.prd) ^ 2) + lambda * sum(abs(beta))`, where `lambda` is a tunable parameter controlling the
  strength of the regularization, and `beta` is the vector of (non-intercept) model coefficients. So the
  second term is penalizing coefficients by how far they are from zero. Letting the model be more flexible 
  results in a smaller `mean((y.trn - y.prd) ^ 2)` component, but a larger `sum(abs(beta))` component. The
  tunable parameter `lambda` controls the balance between these two components. Larger `lambda` results in
  stronger regularization, which results in a stiffer model by biasing coefficient estimates towards zero.
  Ridge regression does something very similar to lasso, minimizing the term 
  `mean((y.trn - y.prd) ^ 2) + lambda * sum(abs(beta ^ 2))`. For GLMs, the process is similar, except the term
  corresponding to the MSE is replaced by then negative of the model likelihood. The main difference between 
  ridge regression and the lasso is that the lasso penalty drives some individual coefficients to zero while 
  leaving others non-zero. So lasso regression can be thought of as a feature selection procedure, since 
  features with zero coefficients have no effect on model predictions. By contrast, ridge regression tends 
  to reduce coefficient magnitudes, but does not make them zero. So all the features you start with play some 
  role in the final predictions, though some features will have much more influence than others.

Ridge regression was originally developed, and is most often touted for addressing issues that arise from
  **multicollinearity**, which is the condition where predictors are linearly related to one another
  (correlated to other individual predictors or weighted sums of those other predictors). Multicollinearity
  can result in very similar MSEs for models with wildly varying coefficient estimates and very high
  standard errors. Ridge regression, by penalizing the fit by the square of the coefficient estimates
  makes fits with smaller coefficient estimates score better than those with large coefficient estimates
  as long as the MSEs are similar. If there are two highly correlated predictors `x1` and `x2`, ridge
  regression will tend to make the coefficients for both smaller, but leave their relative influence on
  the final prediction similar. By contrast, lasso also allows us to handle multicollinear predictors, 
  but will do so by driving the coefficients for one of the two variables (the one with the lower correlation 
  to the response) to zero, essentially eliminating those predictors, thereby purging collinearity 
  from the predictor set. The lasso favors simpler models with better interpretability (since the variables
  with non-zero coefficients are few and have relatively independent influence on the response), while
  ridge models remain complex and include many variables with redundant influences on the response. Leaving
  redundant information in the predictor variable set is not necessarily a bad idea though. In particular,
  where the predictor variable measurements themselves are noisy, having redundant information in other 
  variables allows the model to average out the noise of the individual predictors. 

In the example below, we will use the R `glmnet` package to fit both lasso and ridge regression models. We
  use the same `glmnet()` function for both purposes, setting the argument `alpha=0` for ridge regression
  and `alpha=1` for lasso regression. We will discuss other potential settings for this argument in the
  next section.

```
library(glmnet)
library(caret)

rm(list=ls())

dat <- mtcars

## generate folds:
set.seed(1)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)
idx.trn <- folds[[1]]

## split into training and test-sets:
dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

## peak at training data (only -- leave your test data exclusively for testing):
plot(dat.trn)
summary(dat.trn)
round(cor(dat.trn), 3)

## stepwise regression model (training-set only):
fit.lm.lo <- lm(mpg ~ 1, data=dat.trn)
fit.lm.hi <- lm(mpg ~ ., data=dat.trn)
fit.lm <- step(fit.lm.lo, scope=list(lower=fit.lm.lo, upper=fit.lm.hi), direction='both', trace=1)

## cross-validation to determine best lambda for a ridge model and for a lasso model (training-set only):
set.seed(1)                                      ## for CV randomness
cv.ridge <- cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=0)
cv.lasso <- cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=1)

## beneath the hood:
cv.ridge
is.list(cv.ridge)
names(cv.ridge)
all(names(cv.lasso) == names(cv.ridge))
str(cv.ridge)

## what the cross-validation results look like:
par(mfrow=c(1, 2))
plot(cv.ridge)
plot(cv.lasso)

## extract the ridge fit and look under the hood:
fit.ridge <- cv.ridge$glmnet.fit
fit.ridge                                   ## note Df (number of non-zero coefs) always same (no feature selection)
is.list(fit.ridge)
names(fit.ridge)
str(fit.ridge)

## extract the lasso fit:
fit.lasso <- cv.lasso$glmnet.fit
all(names(fit.lasso) == names(fit.ridge))
fit.lasso                                   ## note Df changing (feature selection

## how the coefficients 'shrink' across values of lambda:
par(mfrow=c(1, 2))
plot(fit.ridge, xvar='lambda', label=T, main='ridge')
plot(fit.lasso, xvar='lambda', label=T, main='lasso')

## smooth, so pick min; can also specify cutoff as s="lambda.1se" or s=1.5
coef(fit.ridge, s=cv.ridge$lambda.min)
coef(fit.lasso, s=cv.lasso$lambda.min)
coef(fit.lm)

## make predictions on test-set:
y.lm <- predict(fit.lm, newdata=dat.tst[, -1]) 
y.ridge <- predict(fit.ridge, newx=as.matrix(dat.tst[, -1]), s=cv.ridge$lambda.min)
y.lasso <- predict(fit.lasso, newx=as.matrix(dat.tst[, -1]), s=cv.lasso$lambda.min)

## mse:
mean((y.lm - dat.tst[, 1]) ^ 2)
mean((y.ridge - dat.tst[, 1]) ^ 2)
mean((y.lasso - dat.tst[, 1]) ^ 2)

```

[Return to index](#index)

---

### Check your understanding 1

Start with the following dataset, which has 100 extraneous variables added:

```
rm(list=ls())

set.seed(1)
dat <- mtcars
for(i in 1:100) {
  nom <- paste('s', i, sep='')
  dat[[nom]] <- rnorm(nrow(dat), 0, 10)
}

```

Use the example from the end of the last section to conduct a 5-fold cross-validation, 
  repeated 3-times to compare the MSEs for:

1) a model derived using `lm()` and `step()`, with the scope between `mpg ~ 1` and `mpg ~ .`;

2) a ridge regression model (with lambda tuning per the example) with `mpg` as response and
   all other variables as predictors;

3) a lasso regression model (with lambda tuning per the example) with `mpg` as response and
   all other variables as predictors;

Make sure to use the same folds for all three procedures.

[Return to index](#index)

---

### Elastic net regularization

Elastic-net regularization bridges the gap between a ridge regression and a lasso regression.
  Here, we minimize the sum of one term representing how well the model fits the training data
  (the MSE or the negative log likelihood of the model) and a penalty on the coefficients:
  `lambda * ((1 - alpha) * sum(beta^2) / 2 + alpha * sum(abs(beta))`. This penalty is a mixture
  of the lasso and ridge penalties, with the balance controlled by the new tunable parameter
  `alpha`, which ranges between `0` (corresponding to a pure ridge penalty) and `1` (corresponding
  to a pure lasso penalty).

```
library(glmnet)
library(caret)

rm(list=ls())

data(tecator)
dim(absorp)               ## IR absorbance at 100 wavelengths
dim(endpoints)            ## moisture, fat, protein

## moisture as response, absorbances as predictors:
dat <- cbind(endpoints[, 1], absorp)
dat <- data.frame(dat)
names(dat) <- c('y', paste('x', 1:ncol(absorp), sep=''))
dat[1:5, 1:5]

## generate folds:
set.seed(1)
idx <- 1 : nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)
idx.trn <- folds[[1]]

## split out training and test-sets:
dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

dim(dat.trn)

## stepwise model selection of lm() model using training-set only:
fit.lm.lo <- lm(y ~ 1, data=dat.trn)
fit.lm.hi <- lm(y ~ ., data=dat.trn)
fit.lm <- step(fit.lm.lo, scope=list(lower=fit.lm.lo, upper=fit.lm.hi), direction='both', trace=1)
summary(fit.lm)

## a function that takes an alpha value and fits an elasticnet model on training-set, with lambda tuning:
f.alpha <- function(alpha.i) {
  cv.glmnet(x=as.matrix(dat.trn[, -1]), y=dat.trn[, 1], alpha=alpha.i)
}

## generate models with different values of alpha using training-set only:
(alphas <- seq(from=0, to=1, by=0.2))            ## alphas to try
set.seed(1)                                      ## for tuning CV randomness
cvs <- lapply(alphas, f.alpha)
names(cvs) <- paste('s', alphas, sep='')

## plot tunes:
par(mfrow=c(2, 3))
sapply(cvs, plot)

## note that as alpha approaches 1, more and more feature selection:
f.fit <- function(cv.i) cv.i$glmnet.fit          ## function to extract fit
fits <- lapply(cvs, f.fit)
sapply(fits, plot, xvar='lambda', label=T)

## function to extract coefficients:
f.coef <- function(idx) {
  fit.i <- fits[[idx]]
  cv.i <- cvs[[idx]]
  coef.i <- coef(fit.i, s=cv.i$lambda.min)
  as.numeric(coef.i)
}

## compare coefficients:
coefs <- sapply(1:length(fits), f.coef)
colnames(coefs) <- names(cvs)
round(coefs, 3)
coef(fit.lm)

apply(coefs, 2, summary)
summary(coef(fit.lm))

## making predictions on test-set:
f.prd <- function(idx) {
  predict(fits[[idx]], newx=as.matrix(dat.tst[, -1]), s=cvs[[idx]]$lambda.min)
}
ys <- sapply(1:length(fits), f.prd)
y.lm <- predict(fit.lm, newdata=dat.tst[, -1]) 

## for scoring performance:
f.mse <- function(y.i) {
  mean((y.i - dat.tst[, 1]) ^ 2)
}
mses <- apply(ys, 2, f.mse)
mse.lm <- f.mse(y.lm)

```

### Check your understanding 2

Starting with the following dataset (where noise has been added to the predictors):

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

```

Use the example from the end of the last section to set up a 5-fold cross-validation, 
  repeated 3-times, to compare the MSEs from:

1) a model derived using `lm()` and `step()`, with the scope between `y ~ 1` and `mpg ~ .`;

2) elastic-net models with alpha set to the following values `0, 0.2, 0.4, 0.6, 0.8, 1`, 
   tuning `lambda` in each case.

Make sure to use the same folds for each model.


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

An example in high dimensional space:

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

Starting with the following dataset:

```
library(caret)
rm(list=ls())
data(dhfr)                        ## from caret
dat <- dhfr

```

Use the example from the end of the last section to set up a 5-fold cross-validation, 
  repeated 1-time (because SVM tuning using one thread on a laptop is slow), to compare 
  the AUCs (make sure to get back class probabilities instead of assignments from `predict()`:

1) an elastic-net logistic regression model with `alpha=0.5` and `lambda` tuning, with 
   `dhfr[, 1]` as the response and all other columns as the predictors. 

2) an SVM logistic regression model with tuning of gamma within the range `2^(-2:2)`, 
   and tuning of cost within the range `2^(1:5)`.

Make sure to use the same folds for each model.

[Return to index](#index)

---

## FIN!
