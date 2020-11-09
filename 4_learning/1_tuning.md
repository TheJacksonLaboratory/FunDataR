# Fundamentals of computational data analysis using R
## Machine learning: model tuning and evaluation
#### Contact: mitch.kostich@jax.org

---

### Index

- [Introduction to model parameter tuning](#introduction-to-model-parameter-tuning)
- [Nested cross-validation](#nested-cross-validation)
- [Tuning model flexibility](#tuning-model-flexibility)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)

---

### Introduction to model parameter tuning

One major focus of traditional statistics has been providing quantitative expressions
  of uncertainty about hypotheses. That is, it usually either tested a null hypothesis
  of interest by returning a p-value, or provided estimates of population parameters 
  in the form of confidence intervals. This activity was largely performed in the 
  service of scientists or engineers who were building mechanistic models about the 
  natural or man-made world. Elaborating these models involves making hypotheses about 
  which explanatory variables are involved and about the form of the relationship 
  between the explanatory variables and the response variable. Since these hypotheses 
  are generally tested with finite sized samples from the population of interest, there
  is always some uncertainty in the answer. That is, for any finite sized random sample, 
  it is always possible that we picked a very extreme and misleading sample by chance. 
  Statistics provides a disciplined quantitative way of describing the uncertainties 
  introduced during inferrence about population parameters based on finite random samples. 
  It recognizes that when we reject the null hypothesis at a p-value cutoff of 0.05, we
  are implicitly accepting a one in twenty chance of encountering a false positive result. 
  Similarly, a 95% confidence interval amounts to an admission that one in twenty replicate 
  experiments would generate 95% confidence intervals that do not include the true 
  population value.

The other major focus of traditional statistics has been on predictive modeling. When 
  a mechanistic model is specified, it can not only be used to explain our previous
  observations, but also to make predictions about future observations. Using statistical
  modeling techniques, we can not only make predictions, but also quantitatively, 
  objectively, and reproducibly express the degree of uncertainty in the predictions 
  (e.g. with prediction intervals). This approach can be extended from prediction using
  theoretically driven mechanistic models to predictive modeling using empirically (rather 
  than theoretically) developed models. Empirical modeling focuses on empirical 
  associations between variables that can be identified from the data without speculating 
  about or specifying the nature of any underlying causal relationships. The balance of 
  hypothesis testing and prediction in traditional statistics very much depends on the 
  subject being studied, but in general, it is probably fair to say that most traditional 
  statistical work has focused on evaluating mechanistic models and estimating populaton 
  parameters more than predictive modeling.

Computers have gradually come to have a major effect on statistical practice. Over time,
  traditional statistical computation have been ported to computers. In addition, the
  general availability of computers changed the approaches statisticians were using.
  For instance, fitting generalized linear models (GLMs) to a training-set requires
  several iterations of carying out very tedious calculations. When computers were not
  widely available, statisticians modeling e.g. binomially distributed data would 
  transform the response in order to come 'close enough' to the assumptions to justify
  use of ordinary linear models. Now that computers are ubiquitous, we have no qualms
  about modeling binomially distributed data using GLMs, because that framework allows
  us to change the modeling assumptions to match the type of response being modeled,
  and because invoking the computational fitting procedures for GLMs is not 
  substantively more difficult than fitting an orderinary linear model. Similarly, some
  techniques that we mostly associate with machine learning, rather than traditional
  statistics, such as the empirical bootstrapping procedure, were first described by
  renowned traditional statisticians (Ronald Fisher) nearly a century ago, but never
  became popular because they required an immense number of calculations that were
  simply not practical to carry out by hand. Now, thanks to computers, we can easily 
  apply empirical boostrapping with equal validity to traditional statistical models
  as well as for machine learning models.

The automation of statistical tasks required statisticians to learn computer programming,
  and also attracted computer scientists to the field of statistics, where they could
  contribute their expertise in algorithm and software development. At the same time,
  as computing power grew, so did information availability. There were many interested
  in understanding the relationships between variables (e.g. patterns) in the large
  datasets that were being collected. Many of these datasets were not generated using
  controlled experiments, but instead represented observational data. The sampling schemes
  often were not random, and so many of the assumptions behind statistical inferrence were
  probably violated. The number of variables was often huge, and the mechanistic 
  relationships between variables were unknown. Nevertheless, there was an interest in
  being able to discover patterns in the data (e.g. clusters of observations and associations
  between variables) as well as make predictions about future observations. So people
  tried, initially by applying traditional statistical approaches, but then supplementing
  them with more heuristic methods, whose theoretical properties are often unknown,
  which means we lack the proofs that would justify use of parametric p-values and 
  confidence intervals. Furthermore, some of these newer **machine learning**
  methods, like the **k-nearest neighbor** algorithm described below, do not have
  any coefficients to test or make confidence interval estimates on anyway. All we can 
  do is make estimates of the model's predictive accuracy. We can estimate the performance
  based on cross-validation. We can calculate confidence intervals for the performance 
  estimates using bootstrapping. We can also see if performance is better than would be
  expected by chance using permutation. We can examine the influence of individual 
  observations in the training set using jackknifing. We can examine the importance of 
  various predictors to the prediction process either by comparing models with and without 
  the predictor, or by permuting predictor values. However, we can apply all these
  methods in the context of traditional statistical procedures as well.

Over time statisticians have come to rely more and more on computational methods from the 
  computer science world. Meanwhile, computer scientists trying to understand very large 
  datasets, have been trying to adopt the traditional statistical emphasis on expressing 
  of estimate uncertainties in a quantitatively rigourous way. You can easily find 
  discussions on the internet (and elsewhere) about the differences between machine
  learning and traditional statistics. Many try to portray them as completely different
  fields. However, there is an easy case to make that machine learning is part of the 
  natural adaptation of emerging computational power to good old fashioned statistics. 
  Statisticians have to learn to exploit the power of computers, and computer scientists 
  interested in data (data scientists) have to take into account the statistical emphasis
  on estimating and expressing uncertainty. Traditional statistics does tend to work with 
  smaller models, focusing on understanding the mechanistic relationships between variables, 
  while machine learning tends to involve larger models where predictive accuracy is a 
  higher priority than mechanistic understanding. Nevertheless, there are many cases where 
  traditional statistics has been used for prediction and machine learning methods have 
  been used to make inferrences about populations. Furthermore, statistics departments, 
  such as the Applied Statistics Department at Stanford University, continue to be 
  major contributors of 'machine learning' algorithms and the ideas behind them. In fact,
  about half the methods descibed in this course on machine learning were developed by
  members of this Stanford group.

The differences between the disciplines is reflected loosely in the terminology used for
  the explanatory variables. Statisticians are more likely to talk about **independent 
  variables** or **explanatory variables** or **regressors**. Machine learning literature 
  often refers to variables as either **predictors** or **features**. We will adopt the 
  term 'feature' for most of the rest of this lesson. In this course, either all the 
  variables are termed 'features', or one or more variables will be separately designated
  as 'response' variables, and the rest termed 'features'.

A good example of a machine-learning algorithm with no obvious counterpart in traditional 
  statistics is the **k-nearest neighbor classification** or **knn** method. In order to 
  predict the class of a new observation `obs.i`, based on a training sample observations 
  in `dat.trn`, distances (by default, **Euclidean distance**) in the feature space (treating 
  each feature variable as a **dimension**) are computed between `obs.i` and each of the 
  observations in `dat.trn`. Assume `obs.j` is an observation in `dat.trn`. Let `x1` and 
  `x2` be two features. Then `d1` and `d2` are the distances between `obs.i` and `obs.j` 
  along the corresponding variable axis. That is, `d1 <- x1.i - x1.j`, where `x1.i` 
  is the `x1` value for `obs.i` and `x1.j` is the corresponding value for `obs.j`. 
  Similarly, `d2 <- x2.i - x2.j`. Then the distance `d.ij` between `obs.i` and `obs.j` is 
  `sqrt(d1^2 + d2^2)`. That is, `d.ij^2 = d1^2 + d2^2`, which is an expression of the 
  **Pythagorean theorem** about **right angles**. This distance metric can be extended to `p` 
  features: `d.ij = sqrt(d1^2 + d2^2 + d3^2 + ... + dp^2)`. The knn procedure identifies 
  the `k` obervations (neighbors) in `dat.trn` that are 'closest' to `obs.i` in the feature 
  space, then assigns the dominant class (the response variable is categorical) among the `k` 
  closest observations in `dat.trn` as the predicted class for `obs.i`. If there is a tie 
  (e.g. two equally frequent classes among the `k` closest observations in `dat.trn`), it can 
  be broken using various implementation-dependent heuristics (like assigning randomly among 
  the tied classes, or assigning to the class of the single closest observation when there is 
  a tie). In the two-class case, it is often preferrable to only try values of `k` which are 
  odd in order to avoid this potentially ambiguous situation.

Because the Pythagorean theorem applies to right triangles, the Euclidean distance metric implies 
  that the predictors are **orthogonal** (at right angles) to one another, which implies that there 
  is no correlation between predictors. Depending on the variables involved, this may very well not 
  be true, which can result in correlated features having a disproportionately high influence on
  the distance estimate. Another concern about using the Euclidean distance is that the distances 
  between different pairs of observations become more and more similar as the number of dimensions 
  goes up. This is an unintuitive result, because people are used to visualizing 2, 3, or at most 
  4 dimensions. We can demonstrate this phenomenon using the following synthetic example, where we 
  look at the ratio between the closest pairwise distances and longest pairwise distances (in the 
  feature space) between observations with increasing numbers of randomly generated feature values. 
  In the low-dimensional space, we see that there are fairly obvious differences in distances between 
  observations (the ratio is many-fold), while in the higher-dimensional space, all the observation 
  pairwise distances become large and extremely similar to one another (the ratio becomes very 
  close to one). Because all the distances between observations become so similar, ordering the 
  distances confidently becomes challenging, especially if there is any noise in the feature data, 
  because then the random pattern of the noise will tend to drive a more or less random ordering of 
  distances. This makes the knn predictions less reliable, since that algorithm depends on being
  able to reliably compare distances.

```
rm(list=ls())
set.seed(1)

diff.prop <- NULL                                ## will hold ratios of largest to smallest obs distance
(p.features <- 2^(1:14))                         ## vector of feature numbers to try

for(p.i in p.features) {                         ## take each feature number in turn
  dat.p <- NULL                                  ## will become a matrix w/ rows=obs, cols=features
  for(i in 1:p.i) {                              ## for the next feature
    x.i <- seq(from=0, to=1, length.out=60)      ## generate an evenly spaced series of values
    x.i <- sample(x.i, length(x.i), replace=F)   ## randomize the order so successive features are uncorrelated
    dat.p <- cbind(dat.p, x.i)                   ## add the feature to the data-set
  }
  dist.p <- dist(dat.p)                          ## Euclidean distances between each pair of observations
  ## ratio of largest pairwise observation distance to smallest pairwise distance:
  diff.prop.p <- max(dist.p) / min(dist.p)
  diff.prop <- c(diff.prop, diff.prop.p)         ## save the result
  cat("number of dimensions:", p.i, ", maximum proportional difference:", diff.prop.p, "\n")
  flush.console()
}

## plot results:
par(mfrow=c(1, 1))
plot(x=p.features, y=diff.prop, main='Curse of (Euclidean) dimensionality',  
  xlab='Number of features', ylab='Maximum proportional difference', log='x')

```

Concerns about feature correlations and large feature number can sometimes be addressed by 
  **dimensional reduction** techniques, such as PCA, which we will describe in a later lesson. 
  Another consideration when using knn is that it can be strongly affected by the scales of the 
  features. That is, if the feature `x1` varies between `0` and `1000`, while `x2` only varies
  between `0` and `10`, then any distances computed in the space defined by these two features
  `d.ij <- sqrt((x1.i - x1.j)^2 - (x2.i - x2.j)^2)` will tend to be much more strongly influenced 
  by the `x1` value than by the `x2` value. Therefore, it is a good idea to normalize the variables 
  (divide them by the standard deviation in the training set) before using them for computing 
  distances, unless the features are expressed on naturally comparable scales to begin with. 
  Normalization will make any variable have a standard deviation of one, so any normalized variable
  has a similar potential influence as any other normalized variable. Most common forms of dimension 
  reduction take the input features and return a potentially much smaller set of new uncorrelated, 
  normalized features.

Distance-based approaches like knn are also adversely affected by **extraneous features**, because 
  extraneous features make the distance estimates vary in a way that has no relation to the response 
  variable. That is, they introduce noise into the determination of closest neighbors, which tends to 
  reduce predictive performance. So it is often a good idea to do some form of **feature selection** 
  prior to employing knn. Feature selection for numeric features can be as simple as conducting an 
  ANOVA omnibus F-test on a model with the feature as response and class as a categorical predictor. 
  We can do this separately for each feature, adjust the p-values for multiple testing, then only use 
  the features with significant F-tests as input into the knn process. This approach suffers from not
  taking feature correlations into account, so it may include features contributing information largely
  redundant to that already included with other features. It is also likely that important
  interactions between features will be missed if the individual features involved in the 
  interaction do not have a direct effect (e.g. they only have an effect through the interaction). 
  There are various alternative approaches that try to evaluate varying subsets of features instead
  of individual features, using predictive performance of the final model as the selection criterion.
  These **wrapper** methods for feature selection offer far more thorough exploration of the predictive
  potential of the feature space, but are often computationally intractable, as the number of feature
  combinations that can be explored grows faster than exponentially with the number of features. 
  Furthermore, in many systems, significant interaction effects are unlikely to be encountered in the 
  absence of significant invididual effects, so the simpler one feature at-a-time selection scheme 
  suffices.

The training-set used for knn should resemble the composition of the the population you are making 
  predictions for. If the composition of the training-set differs from the population, the knn 
  class assignments will tend to be biased toward the over-represented classes in the training-set.

Variations of the knn algorithm are available in R add-on packages allow arbitrary (non-Euclidean) 
  distance metrics. Several packages also provide for distance weighting of observations, where the 
  influence of the k nearest neighbors on the final decision are weighted by the inverse of the 
  distances involved, which can lead to more stable performance across training-sets. Some packages 
  also proved for automated selection of k by cross-validation (we'll 'manually' code this process 
  during this lesson), as well as other resampling approaches for evaluation and improving classifier 
  performance.

In the example below, we will use the **Cohen's Kappa** statistic to express the accuracy of classification. 
  Like raw accuracy proportion, kappa ranges between `0` and `1`, with `1` being 'perfect' performance. 
  Unlike raw accuracy, kappa expresses performance in a way that reflects the composition of the data, 
  so that it automatically compensates for differences in test-set composition as well as the effects of 
  chance on apparent predictive accuracy. Kappa can be used for comparing different parameterizations of 
  the same type of classifier, or for comparing completely different classifiers, as long as the variants 
  are developed using the same training-sets and evaluated with the same test-sets. Nevertheless, kappa 
  values are not nearly as easy to interpret as the AUC and, unlike the AUC, kappa results are sensitive 
  to the tuning of the cutoff point of the classifier score used for class assignments. 

```
library('caret')
library('class')

rm(list=ls())
set.seed(1)

## the data:
dat <- iris
plot(dat)

## split data into training-set and test-set:
nrow(dat)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)
idx.trn <- folds[[1]]
dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

class(dat.trn)
summary(dat.trn)

## predictions for training set and test-set; 7 nearest neighbors:
prd.trn <- class::knn(train=dat.trn[, -5], test=dat.trn[, -5], cl=dat.trn[, 5], k=7)
prd.tst <- class::knn(train=dat.trn[, -5], test=dat.tst[, -5], cl=dat.trn[, 5], k=7)

## whole confusion matrix-related output:
(cnf.trn <- caret::confusionMatrix(dat.trn[, 5], prd.trn))
(cnf.tst <- caret::confusionMatrix(dat.tst[, 5], prd.tst))

## double bracket indexing 'Kappa' drops the name:
cnf.trn$overall[['Kappa']]
cnf.tst$overall[['Kappa']]

```

We can take the code above and turn it into a function that we can apply to each fold, 
  where each fold is an integer index of training-set observations for that iteration, 
  and get back Kappa for that fold:

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

We wrote the function `f.cv()` above in a way that it would accept the number of neighbors
  to use for classification as the argument `k`. We'll exploit this argument here to see 
  what a hold-out test-set based estimate of performance (kappa) is when using different 
  numbers of neighbors:

```
## make a hold-out set (20% of the data); but do it in a way we can extend to 5-fold CV:
set.seed(1)
idx <- 1:nrow(dat)                          ## indices of all observations
folds <- caret::createMultiFolds(idx, k=5, times=3)
idx.trn <- folds[[1]]                       ## integer index of training-set observations for 1st fold

ks <- c(1, 3, 5, 11, 15, 21, 31, 51, 75)    ## values of 'k' to try for knn
rslt <- rep(as.numeric(NA), length(ks))     ## pre-extend so can hold 1 kappa per k

for(i in 1:length(ks)) {                    ## iterate over ks, saving kappa for each k
  rslt[i] <- f.cv(idx.trn, dat=dat, k=ks[i])
}
names(rslt) <- ks                           ## label the results with the corresponding k
signif(rslt, 3)

par(mfrow=c(1, 1))
plot(x=ks, y=rslt, xlab='number of nearest neighbors', ylab='kappa')

```

We can further functionalize the code above to produce another function that tries different
  values of k within each fold. Comparing results within a fold ensures that variations in
  training-sets and test-sets do not introduce noise into the process. That is, we know we
  typically get a slightly different performance estimate for each fold, so comparing performance
  across folds involves not only the true performance difference between models, but also variation
  due to the difference in training-sets and test-sets. Therefore, our comparisons will be more
  precise if we record performance differences for each fold using the same training-sets and 
  test-sets for all the models being compared. Then we can average the within-fold performance 
  differences for each fold to get our final performance estimate. 

After scoring all the values of `k` (number of nearest neighbors in the training set) to use for 
  classification, we may be tempted to just choose the best scoring value. However, this often 
  leads to some degree of overfitting. A popular rule of thumb for helping to attenuate potential 
  overfitting is to choose the simplest model (least flexible; for knn, this means the largest 
  value of `k`) within one standard error of the 'best' result. This rule is often referred to as 
  the **1-SE** rule. You may be thinking that we will likely use something like cross-validation to
  evaluate the final model (you would be right!) so we will have an 'independent' test-set with
  which we can detect any overfitting, so why not just pick the best scoring value of `k` anyway.
  But having your final evaluation tell you that you built a model with poor performance because
  it was likely overfit to the training data is not much solace: you are more or less stuck with 
  that poor model at that point, or you need to generate some new data for evaluation of other 
  possible models, since you already 'spent' all the available independent test data. Therefore, 
  it is important to avoid overfitting in any case, while also using an independent test-set for 
  evaluation of any final model.

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

ks <- c(1, 3, 5, 11, 15, 21, 31, 51, 75)
(rslt <- sapply(folds, f.cv.cmp, dat=dat, ks=ks))

## let's process the results:
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

## plot results:
par(mfrow=c(1, 1))
plot(kap.mean ~ k, data=rslt, type='n', 
  xlab='Number of nearest neighbors', ylab='Kappa')
points(x=ks[idx.max], y=m[idx.max], pch='x', col='magenta')
points(x=ks[i.pick], y=m[i.pick], pch='+', col='orangered')
i.other <- !i.pick
i.other[idx.max] <- F
points(x=ks[i.other], y=m[i.other], pch='o', col='cyan')
abline(h=cutoff, lty=2, col='orangered')
legend(
  'bottomleft',
  legend=c('max kappa', 'picked', 'other points', 'cutoff'),
  pch=c('x', '+', 'o', NA),
  lty=c(NA, NA, NA, 2),
  col=c('magenta', 'orangered', 'cyan', 'orangered')
)

```

[Return to index](#index)

---

### Check your understanding 1

Use the code above as a guide to use the 1-se rule and performance estimates (Cohen's Kappa) based
  on 7-fold cross-validation repeated 3 times to choose the value of `k` (number of nearest neighbors) 
  to use when predicting `Species` based only on the two features `Sepal.Width` and `Sepal.Length`. Try
  the following values for `k`: `c(1, 3, 5, 11, 15, 21, 31, 51, 75)`.

[Return to index](#index)

---

### Nested cross-validation

The model we built in the previous section is likely to overfit the training data to some
  (hopefully small) extent. That is, it is likely modeling some of the noise in those data
  that are completely unrelated to class membership. In order to evaluate the final model,
  we should once again rely on an independent test-set that was not used in any way for 
  model development. We can facilitate this process by turning some of the code
  from the end of the previous section into a function we can use to perform another level
  of cross-validation to be used for evaluating the final model. To make the process easier 
  to follow, we will reiterate all the required code:

```
library('caret')
library('class')

rm(list=ls())

## inner cross-validation function; used to evaluate kappa for k neighbors using 
##   one fold defined by idx.trn:
f.cv <- function(idx.trn, dat, k) {
  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]
  prd.tst <- class::knn(train=dat.trn[, -5], test=dat.tst[, -5], cl=dat.trn[, 5], k=k)
  kap.tst <- caret::confusionMatrix(dat.tst[, 5], prd.tst)$overall[['Kappa']]
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

## set up one fold for outer cross-validation:
set.seed(1)
idx <- 1:nrow(iris)
folds <- caret::createMultiFolds(idx, k=7, times=3)
idx.trn.out <- folds[[1]]

## split whole data-set into training-set and test-set:
dat.trn.out <- iris[idx.trn.out, ]
dat.tst.out <- iris[-idx.trn.out, ]
nrow(dat.tst.out)
nrow(dat.trn.out)

## use inner cross-validation to pick how many nearest neighbors to use k:
ks <- c(1, 3, 5, 11, 15, 21, 31, 51, 75, 101)
(k.pick <- f.pick.k(dat=dat.trn.out, ks=ks))

## make predictions for the test-set using the picked value of k:
prd.tst <- class::knn(train=dat.trn.out[, -5], test=dat.tst.out[, -5], cl=dat.trn.out[, 5], k=k.pick)

## evaluate the predictions on the test-set:
caret::confusionMatrix(dat.tst.out[, 5], prd.tst)
caret::confusionMatrix(dat.tst.out[, 5], prd.tst)$overall[['Kappa']]

```

Note that the `dat.tst.out` observations in the last example are not passed to the inner 
  cross-validation function `f.pick.k()`. So the `dat.tst` as well as the `dat.trn` that appears 
  in the inner cross-validation function `f.cv()` are split out of `dat.trn.out`, and never
  include any observations in `dat.tst.out`. So `dat.tst.out` is an independent test-set that
  we can use to evaluate the final model coming out of the process of tuning the parameter `k`,
  which specifies the number of nearest neighbors in the training set to use for classifying 
  new observations.

With a bit more work, we can turn the code above into a function we can use to do the 
  full-fledged outer cross-validation:

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
mean(rslt)                       ## the final performance estimate
sd(rslt)                         ## standard deviation of performance results
sd(rslt) / sqrt(length(rslt))    ## standard error of performance estimate

```

[Return to index](#index)

### Check your understanding 2

Use the code from the above example as a template to conduct a nested cross-validation 
  of the knn-classification of the `iris` data `Species` based only on the two features 
  `Sepal.Width` and `Sepal.Length`. Use 7-fold CV, repeated 3-times at both levels, using 
  the inner-CV to select a value for `k` and the outer-CV to evaluate the entire procedure.
  Express performance in terms of Cohen's Kappa. Calculate the standard error of the 
  final performance estimate. 

[Return to index](#index)

---

### Tuning model flexibility

When we built linear models in the last section of this series (Multivariate statistics),
  we saw that as we added coefficients to a model, the model became more and more flexible.
  Adding flexibility to a model is helpful when the underlying relationship is complex,
  as the added flexibility allows the model to follow the relationship more closely. However,
  the more flexible a model is allowed to become, the more likely it is to to start capturing
  all the variability in the training-set, including not only the systematic pattern which 
  persists across different samples randomly drawn from the same population, but also the 
  noise which varies from sample to sample. This leads to overestimates of model accuracy when
  evaluated with the training-set, and a tendency toward much worse model accuracy when 
  evaluated using an independent test-set. The same thing can happen with k-nearest neighbors
  methods.

The potential flexibility knn to adapt to the training-set data is regulated by the number of 
  neighbors in the training set used to predict the response value (class) of a new observation. 
  One extreme is represented by using 1-nearest neighbor. Here, all the variation in the 
  data, including the noise, is captured by the model. This is very similar to the extreme of 
  using one coefficient per observation (the saturated model) during ordinary or generalized
  linear modeling. This results in a super-flexible prediction line that passes exactly 
  through each training set observation, but tends to perform relatively poorly on new data. 
  The other extreme is represented by setting the number of nearest neighbors k to 
  `nrow(dat.trn)`, that is make k equal to the number of training-set observations. In this 
  case, the same prediction is made for every new observation, with that prediction being the 
  most highly represented class in the training-set. This is very similar to the extreme of 
  using the intercept-only model for linear regression, where the global mean of the response 
  in the training-set is used as the predicted value for any new observations. So the 
  intercept-only regression model and knn where `k == nrow(dat.trn)` both correspond to the 
  assertion that the predictors actually have no relationship to the response. As we mentioned
  in the univariate statistics portion of this course, when no other guide/predictor is 
  available, the mean response (which is a constant) is the best predictor (in the sense of
  minimizing the MSE) for future observations, assuming training and test-sets are both drawn 
  at random from the same population. 

Many of the similarities between the linear modeling and knn approaches are easier to visualize
  in the context of **knn-regression**. Knn-regression works much like knn-classification, except 
  instead of predicting class membership (response is categorical) using a majority-voting scheme, 
  we are predicting the value of a continuous response variable, like with ordinary linear
  regression. In this case, we use the predictor space to estimate distances of a new observation 
  `obs.i` to each of the training-set observations in `dat.trn`. Then if `k` specifies the number 
  of nearest neighbors to use, the response values of the `k` closest training-set observations 
  to `obs.i` are averaged to generate a response value prediction for `obs.i`.

We'll use the built-in `cars` dataset for demonstration. In this dataset, stopping distance 
  is measured for different speeds. Unfortunately, the data values were rounded, so there are 
  duplicate `speed` values, which makes it harder to demo the properties of knn when k is small. 
  So we are randomly perturbing the values slightly to mimic reversing the rounding process. This 
  doesn't really change the results in any substantive way, but makes it easier to show the 
  differences in the prediction lines for different values of `k`:

```
library('caret')

rm(list=ls())

## prep the data ('undo' the rounding):
set.seed(1)                       ## seed the random part
dat <- cars
length(dat$speed)                 ## how many values?
length(unique(dat$speed))         ## some duplicates
dat$speed <- dat$speed + rnorm(nrow(dat), mean=0, sd=1/2)
length(unique(dat$speed))         ## all unique now
summary(dat)
plot(dat)

## split data into training-set and test-set:
set.seed(1)                       ## new seed to compartmentalize code block
nrow(dat)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)
idx.trn <- folds[[1]]
dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

## knn regression models for different values of k:
fit.1 <- caret::knnreg(dist ~ speed, data=dat.trn, k=1)
fit.4 <- caret::knnreg(dist ~ speed, data=dat.trn, k=4)
fit.16 <- caret::knnreg(dist ~ speed, data=dat.trn, k=16)
fit.all <- caret::knnreg(dist ~ speed, data=dat.trn, k=nrow(dat.trn))

## evenly spaced 'speed' values for plotting prediction lines:
speed <- seq(from=min(dat$speed), to=max(dat$speed), length.out=10000)
newdata <- data.frame(speed=speed)

## prediction lines for different values of 'k':
prd.1 <- predict(fit.1, newdata=newdata)
prd.4 <- predict(fit.4, newdata=newdata)
prd.16 <- predict(fit.16, newdata=newdata)
prd.all <- predict(fit.all, newdata=newdata)

## range of y-values (for plotting):
ylim <- range(c(prd.1, prd.4, prd.16, prd.all, dat$dist))

## plot observations:
plot(x=speed, y=prd.1, ylab='dist', ylim=ylim, type='n')
points(x=dat.trn$speed, y=dat.trn$dist, pch='x', col='black', cex=0.5)
points(x=dat.tst$speed, y=dat.tst$dist, pch='o', col='magenta', cex=0.5)

## add prediction lines:
lines(x=speed, y=prd.1, lty=2, col='cyan')
lines(x=speed, y=prd.4, lty=3, col='magenta')
lines(x=speed, y=prd.16, lty=4, col='orangered')
lines(x=speed, y=prd.all, lty=1, col='black')

## add a legend to the plot:
legend(
  'topleft',
  legend=c('k=1', 'k=4', 'k=16', 'k=all', 'training set', 'test set'),
  pch=c(NA, NA, NA, NA, 'x', 'o'),
  lty=c(2, 3, 4, 1, NA, NA),
  col=c('cyan', 'magenta', 'orangered', 'black', 'black', 'magenta')
)

## make point predictions for held-out test-set:
prd.1 <- predict(fit.1, newdata=dat.tst)
prd.4 <- predict(fit.4, newdata=dat.tst)
prd.16 <- predict(fit.16, newdata=dat.tst)
prd.all <- predict(fit.all, newdata=dat.tst)

## errors for test-set:
res.1 <- dat.tst$dist - prd.1
res.4 <- dat.tst$dist - prd.4
res.16 <- dat.tst$dist - prd.16
res.all <- dat.tst$dist - prd.all

## mean-squared errors for different values of 'k':
mean(res.1^2)
mean(res.4^2)
mean(res.16^2)
mean(res.all^2)

```

We can copy most of the pattern from the end of the second section (nested cross-validation)
  of this lesson and adapt the knn-classification code to select and evaluate k for knn-regression 
  using nested cross-validation:

```
library('caret')

rm(list=ls())

## prep the data ('undo' the rounding):
set.seed(1)
dat <- cars
dat$speed <- dat$speed + rnorm(nrow(dat), mean=0, sd=1/2)

## try specified value for k using fold defined by idx.trn, return MSE:
f.test.k <- function(k, idx.trn) {
  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]
  fit <- caret::knnreg(dist ~ speed, data=dat.trn, k=k)
  prd <- predict(fit, newdata=dat.tst)
  res <- dat.tst$dist - prd
  mean(res^2)
}

## try different values for k specified by ks using fold defined by idx.trn, return MSEs:
f.cv <- function(idx.trn, ks) {
  rslt <- sapply(ks, f.test.k, idx.trn=idx.trn)
  names(rslt) <- ks
  rslt
}

## generate one fold:
set.seed(1)
nrow(dat)
idx <- 1:nrow(dat)
folds <- caret::createMultiFolds(idx, k=5, times=12)
idx.trn <- folds[[1]]

## try every k from 1 to 25 on a single fold:
ks <- 1:25
rslt <- f.cv(idx.trn, ks)
rslt

## do the full cross-validation (5-fold, repeated 12 times):
rslt <- sapply(folds, f.cv, ks)
m <- apply(rslt, 1, mean)
se <- apply(rslt, 1, sd) / ncol(rslt)
mx <- apply(rslt, 1, max)

## use 1-se rule-of-thumb to pick 'k':
idx.min <- which.min(m)
rslt <- data.frame(k=ks, mean=m, se=se, max=mx)
rslt[idx.min, ]
cutoff <- rslt[idx.min, 'mean'] + rslt[idx.min, 'se']
i.good <- rslt[, 'mean'] <= cutoff
rslt[i.good, ]
max(rslt[i.good, 'k'])

## plot results:
par(mfrow=c(1, 1))
plot(mean ~ ks, data=rslt, xlab='Number of nearest neighbors', ylab='MSE')
abline(h=cutoff, lty=2, col='orangered')

```

[Return to index](#index)

---

## FIN!
