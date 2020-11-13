# Fundamentals of computational data analysis using R
## Machine learning: partitioning methods
#### Contact: mitch.kostich@jax.org

---

### Index

- [Trees](#trees)
- [Boosting](#boosting)
- [Bagging](#bagging)
- [Random forest](#random-forest)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Trees

intro here; trees are great for explaining, but tend to perform relatively poorly for
  prediction. input data successively split in a way that decreases **impurity**,
  `f(p)`, `f(0) == f(1) == 0`; 
  a commonly used metric is the **Gini Index**, `f(p.i) = p.i * (1 - p.i)`, where `p.i` is 
  proportion of observations input to the node that belong to class `i`. Observations w/ 
  missing values for the split variable are not counted in the impurity calculation. Node impurity is
  `sum.over.i(f(p.i))`, or the sum of class impurities. Find variable and split value that
  minimize the impurity in each of the two output branches. 

The training-set observations are used to build a tree a set of binary decisions is organized into a tree. data are input at the base of the tree 
  and percolate 

a sequence of binary decisions is made, with each decision equivalent to a hierarchy of 'if, then, else' statements. for example, if we 
  were trying to identify a 

rpart.control(`cp=0.01`): primary complexity parameter; split must decrease lack of fit by `0.01` to be considered; `min.split=20`: only for 
  nodes w/ `n > min.split` are further splits attempted; `min.bucket=round(minsplit/3)`: minimum size for a leaf; `xval=10`: number of CV folds
  used for tuning `cp`; `maxdepth=30`: maximum height (number of nodes from base to leaf) of the final tree.


```
library(rpart)
library(caret)

rm(list=ls())

dat <- rpart::stagec
dat$pgstat <- c('no', 'yes')[dat$pgstat + 1]
dat$pgtime <- NULL
summary(dat)                      ## note the NAs (missing values)
head(dat)
nrow(dat)
par(mfrow=c(1, 1))
plot(dat)

set.seed(1)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)

idx.trn <- folds[[1]]
dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

set.seed(1)
fit <- rpart(pgstat ~ ., data=dat.trn, method='class')
fit
summary(fit)
plot(fit)                         ## plot the tree
text(fit)                         ## add node labels
plotcp(fit)                       ## plot cross-validation
printcp(fit)                      ## print cross-validation
(prd <- predict(fit, newdata=dat.tst, type='prob'))
data.frame(obs=dat.tst$Kyphosis, prd=prd[, 'absent'])

```

Simpler alternative:

```
library(rpart)
library(caret)

rm(list=ls())

summary(kyphosis)
head(kyphosis)
nrow(kyphosis)
par(mfrow=c(1, 1))
plot(kyphosis)

set.seed(1)
idx <- 1:nrow(rpart::kyphosis)
folds <- caret::createMultiFolds(idx, k=5, times=12)

idx.trn <- folds[[1]]
dat.trn <- kyphosis[idx.trn, ]
dat.tst <- kyphosis[-idx.trn, ]

set.seed(1)
fit <- rpart(Kyphosis ~ ., data=dat.trn, method='class')
fit
summary(fit)
plot(fit)                         ## plot the tree
text(fit)                         ## add node labels
plotcp(fit)                       ## plot cross-validation
printcp(fit)                      ## print cross-validation
(prd <- predict(fit, newdata=dat.tst, type='prob'))
data.frame(obs=dat.tst$Kyphosis, prd=prd[, 'absent'])

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Boosting

intro here

  boost: 
    for classifier AdaBoost.M1: in adabag, fastadaboost
      step1: wts.obs <- 1/n
      step2: repeat M times:
        fit model
        compute model error err as weighted average of observation errors
        compute wt adjustment adj <- log((1 - err)/err)
        wts <- wts * exp(adj * as.numeric(y == f(x))
      step3: output sum.m(adj.m * f.m(x)) for all M models
    gradient boosting:
    for regression gradient boosting in gbm

```
code here

```

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Bagging

intro here

  bag: bootstrap-aggregation: in adabag, ipred
    reduce variance (like if sd(rslts) after cv high)
    estimate 'out-of-bag' error
    fit set of otherwise identical models to (30-200) bootstrap samples
    regression: average results from the set of models 
    classification: assign to class with most votes; better to average predicted probabilities!
    loses model interpretability

```
code here

```

[Return to index](#index)

---

### Random forest

intro here

  bag: bootstrap-aggregation: in adabag, ipred
    reduce variance (like if sd(rslts) after cv high)
    estimate 'out-of-bag' error
    fit set of otherwise identical models to (30-200) bootstrap samples
    regression: average results from the set of models 
    classification: assign to class with most votes; better to average predicted probabilities!
    loses model interpretability

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
