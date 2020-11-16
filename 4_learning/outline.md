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
  svm: e1071
    loss function
    kernel
  LDA
    vs logistic regression
    diagonal
    regularized

neural networks

dimension methods:
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
  pls
    vs PCA regression (PCR)
  t-SNE: install.packages("Rtsne")
    KL-divergence
    SNE vs t-SNE
    for clustering
    for dimension reduction preprocessing
  UMAP: install.packages("umap")
  random forest proximity


Clustering:

-1) comparability of variables: standardization is not always the answer; not all variables necessarily strongly
      associated with what we would consider good clustering. The latter is hard to objectively evaluate. Not 
      like supervised methods with a known response value. Often we don't know how many clusters there should
      be and have no way to know if cluster assignments are really being made in a way that reflects the functional
      differences we may ultimately be interested in. Must be driven by subject matter expertise, and will likely
      involve subjective judgement.

0) loss function based on w/i cluster distances and btwn cluster distances; sum is constant; minimize w/i, maximize btwn
     but again, this implies that all variables equally important.

1) combinatorial clustering: exhaustive search; not feasible for >2 or 3 clusters or more than about 30 observations

2) k-means: non-exhaustive: all variables numeric and (potentially weighted) distances euclidean; sensitive to starting
     centers, so try a bunch of different ones and settle on best scoring solution; also, since distances are euclidean,
     (involving squared features) sensitive to outliers; the squared term is implied by putting the centers at the 
     mean (minimizes squared distances). Changing k can result in clusters that are very different. If want to change
     granularity of distinctions without changing clustering pattern, hierarchical may be better. Sometimes k is driven
     by the problem formulation. Other times it is estimated directly from the data. Can calculate w/i group sums of 
     distances across different values of k. If this works, will see a 'kink' at the right value, as teh w/i group
     sums drop till right number, then level off. Can also use **gap statistic** to identify the kink more quantitatively:
     randomly distribute same number of observations and calculate difference in w/i group distance between uniformly
     spread data and actual data w/ changing k. Pick k where the difference in log(sum(distances.wi.cluster)) is maximal. 

   a) assign all points to nearest center
   b) recompute center based on assigned points
   c) iterate till converges

3) k-medioids: arbitrary distance metric, including ones for non-numeric variables; main change is instead of using the 
     mean of the feature values of the observations in the cluster as the cluster center, uses one of the cluster observations
     (the one least total distance to other cluster members) as the cluster center. This opens up the potential to use
     non-euclidean distances. The cost is greatly increased computational burden, as starts to look like combinatorial.

   a) for each cluster, find observation with minimal sum of distances (need not be euclidean) to rest of points in cluster.
   b) assign all observations to closest cluster

4) hierarchical clustering: does not require specification of k; but still needs distance/similarity measure. Can use gap
     statistic to estimate a 'cut point' if want distinct groups. Small changes in data can induce major hierarchical
     reorganizations. Can see how well tree recapitulates original distances by measuring correlation between original
     pairwise distances with the cophenetic distances (assuming a dendrogram). But there is no clear cutoff to use to 
     judge the result.

   a) agglomerative: start w/ individual observations as clusters; successively merge closest clusters till done;
        distances are all additive, so can make dendrogram; single-linkage (nearest neighbor): closest two points between
        clusters are used to estimate intercluster distance; can lead to too few diffuse clusters; furthest 
        neighbor another approach, produces more compact clusters but tends to produce too many; group average tries to 
        balance these two, but very sensitive to monotonic transformations of distances: neighbor approaches only depend
        on ordering of distances; 
   b) divisive: start w/ one big cluster; find split that results in two clusters w/ largest dissimilarity; 
        result is not always additive, so no promise on dendrogram. Not nearly as common or well studied as agglomerative.
        Can be computionally very expensive, depending on approach.

Principle components: loadings; PCA regression; PLS; 

MDS: Kruskal-Shephard metric scaling: `stress <- sum((d.ij - f(z.i - z.j))^2)`, where `i`, and `j` are different 
  observations and we are looking for dimensions `z` where stress is minimized; `f()` is some sort of distance metric; 
  Sammon mapping: better preserves smaller pairwise distances; `stress <- sum((d.ij - f(z.i - z.j))^2 / d.ij)`.
  Non-metric scaling: minimize stress `sum((f(z.i - z.j) - g(d.ij))^2) / sum(f(z.i - z.j)^2)`

tsne and umap; 

randomforest proximities.

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

