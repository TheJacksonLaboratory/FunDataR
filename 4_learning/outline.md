#######################################################################

machine learning vs. statistics
  p > n
  very flexible fits
  least-squares vs nearest neighbors: 
    smoothness/inflexibility/bias vs. variance
    can plot test-set + training-set error curves vs. complexity
  the bias-variance decomposition of the MSE
  curse of dimensionality

strategy for model development
  explore algorithms w/ very different approaches
  tune w/ nested CV
  final selection + tuning must use independent data!!!
    what about inclusion during normalization/calibration?
    how to handle calibration curves?

ensemble methods and similar
  boost
  bag
  stack?
  bump?

model tuning
  grid vs. random
  nested CV

partitioning methods: 
  trees
  random forest
  boosted trees
  for regression

linear methods:
  for classification
    logistic for binary
    for multi-class
    performance metrics
  shrinkage: shrink coefficients towards zero
    L2 penalty: ridge
    L1 penalty: lasso
  elastic
    mixing parameter
  svm - should it be separate?
    loss function
    kernel
  pls
    vs PCA regression (PCR)
  LDA
    vs logistic regression
    diagonal
    regularized

miscellaneious methods:
  knn ...
  naive bayes
  neural networks

unsupervised methods:
  k-means
    distance metrics
    k-mediods
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


#######################################################################

1:
2:
3:
4:
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

