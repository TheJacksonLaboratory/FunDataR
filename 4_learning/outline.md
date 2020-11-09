#######################################################################

machine learning vs. statistics
  p > n
  very flexible fits
  least-squares and nearest neighbors: 
    smoothness/inflexibility/bias vs. variance
    can plot test-set + training-set error curves vs. complexity
  the bias-variance decomposition of the MSE
  curse of dimensionality

miscellaneious methods:
  knn: class
  naive bayes: e1071

strategy for model development
  explore algorithms w/ very different approaches
  tune w/ nested CV
  final selection + tuning must use independent data!!!
    what about inclusion during normalization/calibration?
    how to handle calibration curves?

model tuning
  grid vs. random
  nested CV

ensemble methods and similar
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
  bag: bootstrap-aggregation: in adabag, ipred
    reduce variance (like if sd(rslts) after cv high)
    estimate 'out-of-bag' error
    fit set of otherwise identical models to (30-200) bootstrap samples
    regression: average results from the set of models 
    classification: assign to class with most votes; better to average predicted probabilities!
    loses model interpretability
  stack?: 
    improve average performance
    reduce variance
    loocv-based mse or deviance used to weight arbitrarily different models
  bump?: Bootstrap Umbrella of Model Parameters
    find better models; don't get stuck in local minima
    fit set of different models to all? the bootstrap samples
    pick model with lowest error against original data
    will tend to overfit
    conserves interpretability

partitioning methods: 
  trees: rpart
  random forest: randomForest
  bagged trees: ipred
  boosted trees: 
  for regression: randomForest

linear methods:
  for classification
    logistic for binary
    for multi-class
    performance metrics
  shrinkage: shrink coefficients towards zero
    L2 penalty: ridge
    L1 penalty: lasso
  elastic: 
    mixing parameter
  svm - should it be separate? e1071
    loss function
    kernel
  pls
    vs PCA regression (PCR)
  LDA
    vs logistic regression
    diagonal
    regularized

neural networks

unsupervised methods:
  k-means
    distance metrics
    k-mediods: install.packages("cluster"); cluster::pam()
  hierarchical clustering
    agglomerative
    divisive
  SOMs?
  PCA
    principle components
    principle curves
    for display
    for pre-processing
  MDS
    with Euclidean distances == PCA up to rotation
    non-liner dimensional reduction
  t-SNE: install.packages("Rtsne")
    KL-divergence
    SNE vs t-SNE
    for clustering
    for dimension reduction preprocessing
  UMAP: install.packages("umap")


#######################################################################

1: knn and tuning
2: linear regression/regularization
3: partitioning/ensembles
4: unsupervised/dimension reduction
5: 


#######################################################################

1:
- [Transforming the response](#transforming-the-response)
- [Transforming predictors](#transforming-predictors)
- [Local regression](#multiple-regression)

2:
- [Permutation testing](#permutation-testing)
- [Empirical boostrap](#empirical-bootstrap)
- [Cross-validation](#cross-validation)

3:
- [Multiple regression](#multiple-regression)
- [Correlated predictors](#correlated-predictors)
- [Interactions](#interactions)
- [Hierarchical models](#hierarchical-models)

4:
- [Multiple testing](#multile-testing)
- [Overfitting](#overfitting)
- [Feature selection](#feature-selection)
- [Model selection](#model-selection)

5:
- [Generalized linear modeling](#generalized-linear-modeling)
- [Logistic regression](#logistic-regression)
- [Poisson regression](#poisson-regression)
- [Negative-binomial regression](#negative-binomial-regression)

