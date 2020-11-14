# Fundamentals of computational data analysis using R
## Machine learning: partitioning methods
#### Contact: mitch.kostich@jax.org

---

### Index

- [Trees](#trees)
- [Bagging](#bagging)
- [Random forest](#random-forest)
- [Boosting](#boosting)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Trees

intro here; trees are great for explaining, but tend to perform relatively poorly for
  prediction. Although have low bias (as long as they are grown sufficiently tall), they 
  have high variance.

input data successively split in a way that decreases **impurity**,
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

For regression, splitting criterion becomes maximizing reduction in total sums-of-squared residuals in sons -- pick variable, cutoff, and 
  conditional mean so as to minimize sums-of-squares. Typically has much lower granularity than regression methods. 


```
library(rpart)
library(caret)

rm(list=ls())

## reformat prostate cancer recurrence dataset:
?rpart::stagec
dat <- rpart::stagec
dat$pgstat <- c('no', 'yes')[dat$pgstat + 1]
dat$pgstat <- factor(dat$pgstat)  ## character -> factor
dat$pgtime <- NULL                ## drop column
summary(dat)                      ## note the NAs (missing values)
head(dat)
nrow(dat)

## split into training and test-set in way can functionalize for CV:
set.seed(1)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)
idx.trn <- folds[[1]]
dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]
summary(dat.trn)
summary(dat.tst)

## take a peak:
par(mfrow=c(1, 1))
plot(dat.trn)

## fit the model, tuning 'cp' complexity parameter by (10-fold) CV:
fit0 <- rpart(pgstat ~ ., data=dat.trn, method='class')
fit0
class(fit0)
is.list(fit0)
names(fit0)
summary(fit0)
plot(fit0)                        ## plot the tree
text(fit0)                        ## add node labels
plotcp(fit0)                      ## plot cross-validation
printcp(fit0)                     ## print cross-validation (xerror is CV error)
(tbl <- fit0$cptable)

## prune tree using 'cp' value chosen from tuning:
(idx.bst <- which.min(tbl[, 'CP']))
tbl[idx.bst, ]
(fit <- prune(fit0, cp=tbl[idx.bst, 'CP']))
plot(fit)
text(fit, cex=0.75)

## make (probabilistic) predictions and take a look:
(prd.tst <- predict(fit, newdata=dat.tst, type='prob'))
prd.tst <- prd.tst[, 'yes']
roc.tst <- pROC::roc(dat.tst$pgstat == 'yes', prd.tst, direction='<')
roc.tst$auc
pROC::ci.auc(roc.tst)
(prd.class.tst <- c('no', 'yes')[(prd.tst > 0.5) + 1])
## mcnemar's test compares sensitivity and specificity:
(cnf.tst <- caret::confusionMatrix(dat.tst$pgstat, factor(prd.class.tst)))

## do predictions work better on training set?
prd.trn <- predict(fit, newdata=dat.trn, type='prob')
prd.trn <- prd.trn[, 'yes']
roc.trn <- pROC::roc(dat.trn$pgstat == 'yes', prd.trn, direction='<')
roc.trn$auc
pROC::ci.auc(roc.trn)
prd.class.trn <- c('no', 'yes')[(prd.trn > 0.5) + 1]
(cnf.trn <- caret::confusionMatrix(dat.trn$pgstat, factor(prd.class.trn)))

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Bagging

intro here; ensemble or committee method; ensemble/committee of similar models are built.
  predictions from each are averaged in some way. for continous, can be simple average. for 
  classification can be plurality vote.

  bag: bootstrap-aggregation: in adabag, ipred
    reduce variance (like if sd(rslts) after cv high)
      bias on average unchanged
      works best for high variance, low bias non-linear modeling methods
      limited by correlations between models in the ensemble
      linear estimates on bootstrap samples tend to be correlated, so work less well
    estimate 'out-of-bag' error: tends to be very similar to CV results; good for tuning
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

intro here; is related to knn. 

plurality voting for classification; averaging for regression. 
bagging of observations to increase precision (reduces variance); 
random selection of features to consider at each node, decorrelates trees so they provide 
  more independent assessments, which improves effectiveness of bagging in reducing variance.
fit each tree to a separate bootstrap sample (with replacement) of observations 
separately for each tree, at each node, select m out of p features at random; 
  only these m features are considered for splitting. 
magnitude of imrovement in loss (purity or sums-of-squares) resulting from splitting a variable 
  in that tree contributes to the importance of the variable. Sum importances at all nodes 
  where variable is split in order to get total importance for that variable in that tree. Sum 
  the variable importance across trees in order to get the final variable importance estimate. 
also can estimate variable importance based on out-of-bag error: for each tree and each variable,
  calculate out-of-bag error using original data, then with that variable randomly permuted,
  and the difference is used as an estimate of variable importance in that tree. Average over
  all trees is final variable importance estimate.

with many extraneous variables, need large m to ensure an informative variable is available for
  selection at most splits; for many correlated features, small m is better. for classification,
  default m.try is sqrt(p), or p/3 for regression. 

smaller m.try leads to more decorrelated trees. also leads to variable importances being more 
  similar, as lower-importance variables have an improved chance of being selected.

```
library(caret)
library(pROC)
library(glmnet)
library(randomForest)

## data:

rm(list=ls())
data(dhfr)                        ## from caret
dat <- dhfr
class(dat)
table(dat$Y)
class(dat$Y)

## generate hold-out test-set:

set.seed(1)
idx <- 1 : nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=3)
idx.trn <- folds[[1]]

dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]
names(dat.trn)[1]

## mtry tuning using out-of-bag error:

(tune.fit <- tuneRF(x=dat.trn[, -1], y=dat.trn$Y, stepFactor=0.5, improve=0.01, ntreeTry=1000, trace=T, plot=T, doBest=F, replace=T))
class(tune.fit)
(score.best <- min(tune.fit[, 'OOBError']))
(i.best <- tune.fit[, 'OOBError'] == score.best)
(mtry.best <- min(tune.fit[i.best, 'mtry']))

## fit with selected mtry:

fit <- randomForest(x=dat.trn[, -1], y=dat.trn$Y, mtry=mtry.best, ntree=1000, importance=T, replace=T)
par(mfrow=c(1, 1))
plot(fit, log='y')

## probabilistic predictions:

prd.tst <- predict(fit, newdata=dat.tst[, -1], type='prob')
prd.trn <- predict(fit, newdata=dat.trn[, -1], type='prob')

head(prd.tst)
prd.tst <- prd.tst[, 'active']
prd.trn <- prd.trn[, 'active']

## evaluation: 

roc.tst <- pROC::roc(dat.tst$Y == 'active', prd.tst, direction='<')
roc.tst$auc
pROC::ci.auc(roc.tst)
prd.class.tst <- c('inactive', 'active')[(prd.tst > 0.5) + 1]
(cnf.tst <- caret::confusionMatrix(dat.tst$Y, factor(prd.class.tst)))

roc.trn <- pROC::roc(dat.trn$Y == 'active', prd.trn, direction='<')
roc.trn$auc
pROC::ci.auc(roc.trn)
prd.class.trn <- c('inactive', 'active')[(prd.trn > 0.5) + 1]
(cnf.trn <- caret::confusionMatrix(dat.trn$Y, factor(prd.class.trn)))

## importance: type=2 is average (over trees) decrease in node impurity achieved by splitting on variable

imp.trn <- randomForest::importance(fit, type=2)
imp.trn[1:30, , drop=F]
imp.trn <- imp.trn[order(imp.trn, decreasing=T), , drop=F]
imp.trn[1:5, ]
par(mfrow=c(1, 1))
randomForest::varImpPlot(fit, type=2, mex=0.75, cex=0.75)

## partial dependence plot:

rownames(imp.trn)[1:6]
par(mfrow=c(2, 3))
for(idx in 1:6) {
  (x.var1 <- rownames(imp.trn)[idx])
  randomForest::partialPlot(fit, pred.data=dat.trn, x.var=c(x.var1), which.class='active', main=x.var1)
}

```

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Boosting

intro here;

Boosted trees are harder to tune, but sometimes have better performance than random forest.
  At the cost of increased complexity of tuning and computational load.

Weights observations based on how hard they are to classify (for categorical response) or 
  by how big the residual is (for continuous response). Initially all 1/n. Iteratively 
  fit model, then reweight observations based on results. Repeat a user-specified number 
  of times. The differences in approaches are largely related to how 
  the weights are updated. The simpler to understand AdaBoost.M1 algorithm is described
  below:

for classifier AdaBoost.M1: in adabag, fastadaboost
  step1: wts.obs <- 1/n
  step2: repeat M times:
    a) fit model
    b) compute model error err as weighted (wts.obs) average of observation errors
    c) compute wt adjustment adj <- log((1 - err)/err); this controls the 'step-size' for
         the change in weighting for this iteration
    d) update weights: wts <- wts * exp(adj * as.numeric(y == f(x))
  step3: output sum.m(adj.m * f.m(x)) for all M models

The effect of including all M models in the final committee is equivalent to fitting
  the first model to the data, then the second to the residuals from the first, and the
  third to the residuals of the second, etc.

gradient boosting: adjusts the weights in a more complex and effective way; 
  for regression, gradient boosting in gbm. Find direction in the weights space (where
  each variable weight defines an orthogonal dimension) which most decreases loss
  function (estimated from training-set). For instance, for regression with squared-error
  loss function (not the only choice), find direction in weight space that most reduces the 
  MSE mean(fitted - observed) for training set.
  Take a step in that direction of a size determined by a 'learning rate' parameter.
  Setting learning-rate < 1 is like shrinkage of step size, a regularization of sorts that
  reduces chance of overfitting, but also increases number of steps needed to converge. 
  Tune learning-rate vs. number of iterations. Can pick learning-rate that is as small as
  tolerable given time/compute bandwidth/ram, then tune number of iterations by CV. Can
  sample observations w/ replacement at each step in order to reduce overfitting.  
  Also lower learning-rate benefits from more trees.

```
library(caret)
library(pROC)
library(gbm)

## data:

data(dhfr)                        ## from caret
dat <- dhfr
table(dat$Y)
class(dat$Y)

## generate hold-out test-set:

set.seed(1)
idx <- 1 : nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=3)
idx.trn <- folds[[1]]

dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

fit <- gbm::gbm(y ~ ., data=dat.trn, distribution="bernoulli", 
  n.trees=500, shrinkage=0.01, interaction.depth=3, n.minobsinnode=10, 
  cv.folds=5, keep.data=F, verbose=T, n.cores=2)

## plot cv:

## best stopping point:
(n.trees.best <- gbm.perf(fit, method="cv"))
summary(fit, n.trees=1)                ## first tree
summary(fit, n.trees=n.trees.best)     ## final series of trees

## probabilistic predictions:
prd <- predict(fit, newdata=dat.tst, n.trees=n.trees.best, type="response")

## importance plots, can be multivariate; can specify variables by integer index or name:
plot(fit, i.var="x1", n.trees=n.trees.best)  
plot(fit, i.var=1:2, n.trees=n.trees.best)
plot(fit, i.var=1:3, n.trees=n.trees.best)

## predict:
predict(fit, newdata=dat.tst, n.trees=n.trees.best, 

## evaluate:

roc.trn <- pROC::roc(dat.trn$pgstat == 'yes', prd.trn, direction='<')
roc.trn$auc
pROC::ci.auc(roc.trn)
prd.class.trn <- c('no', 'yes')[(prd.trn > 0.5) + 1]
(cnf.trn <- caret::confusionMatrix(dat.trn$pgstat, factor(prd.class.trn)))

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
