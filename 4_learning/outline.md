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
    start w/ distances, not coordinates (PCA needs coordinates) so variable can be subjective distance
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
  PCA minimizes sum-of-squared differences between distances in high dimension and corresponding ones in low dimension.

What if relationships of interest involve polynomial terms and interactions? What if there are many irrelevant variables?
  Hard to do 'feature selection' w/o a designated response variable. Trying to discover groupings. Linear methods are 
  more influenced by large variations than small ones (why they are sensitive to outliers). Tend to preserve long-distance
  relationships better than shorter-distance relationships. 

MDS: Kruskal-Shephard metric scaling: `stress <- sum((d.ij - f(z.i - z.j))^2)`, where `i`, and `j` are different 
  observations and we are looking for dimensions `z` where stress is minimized; `f()` is some sort of distance metric; 
  **Sammon mapping**: better preserves smaller pairwise distances; `stress <- sum((d.ij - f(z.i - z.j))^2 / d.ij)`.
  Classic scaling: based on inner products; `stress <- sum((s.ij - <z.i-z.bar, z.j-z.bar>)^2)`, where
    `s.ij <- <x.i - x.bar, x.j - x.bar>` inner product; classic scaling w/ Euclidean equivalent to PCA.
  Kruskal-Shephard non-metric scaling: minimize stress `sum((f(z.i - z.j) - g(d.ij))^2) / sum(f(z.i - z.j)^2)`

entropy: `H <- -sum(p(x.i) * log2(p(x.i)))`; calculated across all possible values of `x.i`; 
  yields number of bits needed to encode information.

asymmetric sne:  tends to preserve local structure well; 
  for 2 obs x.i and x.j: similarity using Gaussian kernal:
    `v.j|i = exp(-b.i * r.ij^2)`, where `b.i` is is fitted coefficient, `r.ij` is distance from `x.i` to `x.j`. 
    note `v.j|i` not necessarily equal to `v.i|j`, since `b.i` not necessarily equal to `b.j`. 
  Then generate probabilities:
    `p.j|i = v.j|i / sum(v.k|i)`
  Find `b.i` to achieve a specified **perplexity** (similar to a continuous version of `k` in knn). 
  Meaning of `p.j|i` is probability that, starting with `x.i`, you would pick `x.j` as nearby.
  In the new coordinate system (`x.i` -> `y.i`, etc): 
    `w.ij <- exp(-d.ij^2)`
    where `d.ij` is Euclidean distance. No coef so symmetric. 
    `q.j|i = w.ij / sum(w.ik)`; 
  minimize **Kullback-Leibler divergence** or **KL divergence** of N distributions, 
    one for each observation: `C.sne <- sum.ij(p.j|i * log(p.j|i / q.j|i))`.

`p.j|i`: conditional probability that `x.j` would be picked as a neighbor of `x.i` if probability of being picked
  was proportional to the height of a Gaussian distribution centered at `x.i` given the distance in the original
  high-dimensional space. `q.j|i` is corresponding probability in low-dimensional space. 

`b.i`: because need less bandwidth in data dense regions than in less data dense regions.

KL divergence: `D <- sum(p(x.i) * log2(p(x.i) / q(x.i)))`; same as `sum(p(x.i) * (log2(p(x.i)) - log2(q(x.i))))`;
  which is same as `sum(p(x.i) * log2(p(x.i))) - sum(p(x.i) * log2(q(x.i)))`; represents how much info is lost when
  we use `q(x.i)` instead of `p(x.i)` to approximate the distribution. Is not a metric, since (for one thing) not 
  symmetric. This means different types of mapping deficiencies are treated differently. Large cost for separating
  nearby points (e.g. `q.j|i` is small when `p.j|i` is large). But only small cost of representing far-off points
  as nearby. So tends to preserve local structure better than global structure.

symmetric sne:  https://jlmelville.github.io/uwot/umap-for-tsne.html
  tends to preserve local structure well; more stable positioning of outliers: in asymmetric, outliers do not influence fit.
  `p.ij <- (p.i|j + p.j|i) / (2 * N)`
  `q.ij <- w.ij / sum(w.kl)`
  `C.ssne <- sum.ij(p.ij * log(p.ij / q.ij))`  ## KL divergence of joint distributions in high D and low D

t-sne (t-Distributed Stochastic Neighbor Embedding): 
  still uses normal distribution for high dim space.
  centers a t-distribution (with one degree-of-freedom; equivalent to Cauchy)
    instead of normal at `y.i` (in low dim space); in order to alleviate **crowding problem**
  when mapping from higher dimensional space to lower, 
    either over-collapse nearby distances or over-expand far away distances.
    nearby distances crushed by lots of longer distances trying to map far away
  t(1) spreads out the middle distance in the map, alleviating overcrowding
  `q.ji > p.ji` results in 'repulsion'
  `w.ij <- 1 / (1 + d.ij)`                     ## heavier tail, alleviates crowding
  `q.ij <- w.ij / sum(w.kl)`
  attractive component is `p.ij`; repulsive component is `q.ij`;
  1) calibrate `p.ij` to needed perplexity
  2) iterate: 
     * calculate all `d.ij` between `y.i` and `y.j` for all `i` and `j` where `i != j`.
     * calculate `w.ij <- 1 / (1 + d.ij)`.
     * calculate `q.ij`.
     * update `y.i` and `y.j` by gradient descent.
  Author suggests that for reduction to >3 dimensions, should use t(df > 1);
    also says still sensitive to curse of dimensionality: preprocess w/ PCA so input dimensionality <50.
    tune local vs. global (or maybe middle) by perplexity; more global or more noisy or more data 
      benefit from higher perplexity; susceptible to noise in the data.
    relative cluster sizes are not reproduces, since emphasis on local distance reconstruction;
      similarly spacing between clusters has little meaning; 
      too low perplexity tends to lead to false clustering in map; too high pushes data to edges of plot;
        explore between 2 and 100.

LargeVis: each observation is a vertex or node; edges connect them; `p.ij` and `w.ij` per t-sne;
  E: set of edges w/ non-zero weight (defines number of nearest neighbors); 
    sum over members: sum.e; sum over non-members: sum.ebar
  loss: `L.lv = sum.e(p.ij * log(w.ij)) + gamma * sum.ebar(log(1 - w.ij))`;
    does not require calculation of `q.ij`
    gamma: tunes attractive first term vs. repulsive second term.

umap (Uniform Manifold Approximation and Projection):
  tends to preserve global structure better than t-sne.
  `v.j|i <- exp(-(r.ij * rho.i) / b * sigma.i)`; 
    `r.ij` is input distance; 
    `rho.i` is distance to nearest neighbor;
    `sigma.i` is fitted to achieve `sum.j(v.j|i) == log2(k)`, 
      where `k` is number of nearest neighbors;
      fits `sigma.i` w/ `v.j|i` calculated w/ `b == 1`, even if set otherwise
    `b` is `bandwidth` (defaults to `1`):
      set to one for fitting `sigma.i`, then `v.j|i` recalculated w/ `b` and fitted `sigma.i`;
  symmetric 'input affinities': `v.ij <- (v.j|i + v.i|j) - v.j|i * v.i|j`;
    this is a 'fuzzy set union'
    `v.ij` weights attractive component
  `w.ij <- 1 / (1 + a * d.ij ^ (2 * b))`; 
    `a` and `b` fit based on `min_dist` and `spread` arguments; 
    `a == 1; b == 1` corresponds to t-sne
  cost: `C.umap = sum.ij(v.ij * log(v.ij / w.ij) + (1 - v.ij) * log((1 - v.ij) / (1 - w.ij)))`

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

