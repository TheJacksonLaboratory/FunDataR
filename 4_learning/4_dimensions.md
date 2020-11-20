# Fundamentals of computational data analysis using R
## Machine learning: dimension reduction
#### Contact: mitch.kostich@jax.org

---

### Index

- [Basic clustering](#basic-clustering)
- [Principal components](#principal-components)
- [t-sne and umap](#t-sne-and-umap)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Basic clustering

The methods we have studied thus far have focused on explaining or predicting response 
  variable values based on a set of predictor/explanatory variables. Our approach was to
  use a training-set of observations to fit a predictive model to the data and then evaluate
  that model using a test-set. This process is called **supervised learning**, because the 
  training-set and test-set response values provide a guide that can be used as ground
  truth for 'supervising' (in a sense similar to a teacher, who knows the answer, supervising 
  a student's learning process) the model fitting process. However, in many scenarios, there 
  may be no particular response variable of interest. Instead, we might be interested in 
  describing the relative similarity of observations to one another, given a set of variables 
  (we will call these variables **features**) measured on each observation. This type of task 
  is called **unsupervised learning** because the ground truth for the task cannot be known
  for certain based on the features. For instance, if we were to gather morphometric data on 
  iris plants, we might not know how many species are really represented in our data set. But 
  we might be able to use the measurement data to describe the **similarity** (or its 
  converse, the **distance**) between any two observations. By considering the relative 
  similarities between observations in the data-set, we may be able to split all the 
  observations into **clusters** where each cluster contains observations with higher 
  similarity to one another than to observations in other clusters. However, the true cluster 
  assignments and even the true number of clusters is not encoded in the data, since 
  there is no response variable with species information. Instead of being driven to match
  up species labels, the clustering process minimizes some sort of measure of clustering 
  effectiveness. 

One simple loss function (measure of cluster ineffectiveness) for clustering is the sum of 
  distances between observations within each cluster. The total data variance can be split 
  into two components, with the first being variance within each cluster (describes how 
  diffuse the cluster is) and variance between clusters (describes how far the clusters are
  from each other. Ideally, we want compact, well-separated clusters, which implies that we 
  want to partition as much of the variance into the between-cluster component, which means
  that the within-cluster variance (which is based on the sum of distances between observations 
  within each cluster) will be minimized. We can in principle optimize this metric for any 
  particular cluster number by exhaustively trying all possible assignments of observations to 
  clusters and picking the arrangement that minimizes this loss. This approach, which is 
  known as **combinatorial clustering** is only feasible for assigning a small number of 
  observations to a small number of clusters. For instance, assigning 19 observations to 
  4 clusters actually involves about `10^10` distinct arrangements. For larger tasks, we 
  instead use approximate non-exhaustive methods to search the potential solution space. 

Note that we are not even sure if our dataset contains the right features for separating the 
  of iris plants into species. Perhaps features of the leaf (only flower-related features are
  included) are required for making distinctions between species. Perhaps using the flower 
  features will result in some other legitimate grouping (like the position of plants within
  the greenhouse) that affects flower growth but does not correspond to an inter-species 
  difference. 

One of the best known algorithms for assigning observations to a user-selected number `k` 
  of clusters is the **k-means** algorithm. It begins by picking `k` points (typically not
  corresponding to an observation location) in the feature space as cluster centers. Then,
  we:

1) assign all points to the nearest cluster center.

2) recompute cluster centers as the mean of all observations assigned to the cluster

3) repeat steps 1 and 2 till the process converges.

This process proceeds in a step-wise manner, where the history of steps determines the 
  likely paths for subsequent steps. So the final solution often depend on the initial random
  assignment of cluster centers. This means the process may well find a solution
  that represents a **local minimum** in the loss function, rather than the **global 
  minimum**. We stand a better chance of finding the global minimum, or at least a decent
  local minimum by starting the algorithm with different randomly chosen cluster centers,
  then choosing the solution which does the best job of reducing the loss function.

Another issue with k-means clustering is the need to specify a cluster number `k` to the 
  algorithm. Since this is often not known for certain, we try a range of values. Often 
  this will result in steady improvement in the loss function with rising `k` until the 
  real number of clusters is reached, at which point the loss function will begin to level 
  out. If we plot the loss function for successive values of `k`, we sometimes see a 'kink' 
  in the plot at the correct value of `k`, corresponding to a sudden relative leveling of 
  the loss function. In the iris flower example below, since there are only three underlying 
  groups, this method does not work very well, since we don't have enough preceding values 
  of `k` to estimate a baseline trend. Another approach to selecting `k` is to use a 
  **gap statistic** to identify the kink more quantitatively: we can randomly distribute 
  the same number of observations and calculate differences in the loss when clustering the 
  randomly spread data and the actual data as a function of changing `k`. We then pick the 
  `k` where this difference is maximal. In this example, we use the R `scale()` function to 
  center the variables (subtract the variable mean from all the variable values, resulting in 
  a transformed variable with mean of zero) and rescale (divide by the standard deviation of 
  the variable values, resulting in a transformed variable with a standard deviation of one) 
  the variables so that each has a similar influence on the distance calculations. If we do 
  not rescale the variables, differences in a variable like `Petal.Length`, which has a 
  standard deviation of about `1.76`, will influence distance estimates more than variables 
  like `Sepal.Width`, whose standard deviation is only about `0.44`.

```
library(caret)

## the data:
dat <- iris
summary(dat)
dat[, 1:4] <- scale(dat[, 1:4])   ## standardize variables
## name the rows/observations with first 3 letters of species followed by observation number:
rownames(dat) <- paste(substr(dat[, 5], 1, 3), 1:nrow(dat), sep='')
summary(dat)

## let's fit three groups/centers (since we happen to know there are 3 species)
set.seed(123)
fit <- kmeans(dat[, 1:4], centers=3, nstart=10)

fit
class(fit)
is.list(fit)
names(fit)
str(fit)

## decomposition of sums-of-squares into between group and within group:
fit$totss
fit$withinss
fit$betweenss
all.equal(fit$totss, sum(fit$withinss) + fit$betweenss)

## does the clustering correspond to the species?:
(rslt <- cbind(as.numeric(dat[, 5]), fit$cluster))
caret::confusionMatrix(factor(rslt[, 1]), factor(rslt[, 2]))

f.fit <- function(k) {
  fit <- kmeans(dat[, 1:4], centers=k, nstart=10)
  sum(fit$withinss)
}

set.seed(123)
ks <- 2:10
(withinss <- sapply(ks, f.fit))

par(mfrow=c(1, 1))
plot(x=ks, y=withinss, xlab='number of clusters', ylab='sum-of-squares (within)', type='l')

```

The k-means algorithm requires recomputing cluster centers at each step in the process
  as the average position across all observations currently assigned to that cluster. 
  This is equivalent to finding the center that minimizes the mean-squared distances 
  between observations and the cluster center. This squaring of values makes the 
  process sensitive to outliers. In addition, the k-means distance metric is assumed 
  to be Euclidean and is not easily adapted to some other distance measures. The 
  **k-mediods** algorithm (sometimes called **partitioning around medioids** or 
  **pam**) is similar to the k-means algorithm, except that each cluster center is 
  restricted to coincide with one of the observations assigned to the cluster, which 
  makes incorporating non-Euclidean distances and non-numeric variables more natural. 
  The process starts by randomly selecting `k` observations as cluster centers, assigns 
  all observations to the closest center, then recomputes each cluster center so it 
  corresponds to the observation in the cluster that has the smallest sum of distances 
  to the other points currently assigned to the cluster. The user can employ any type 
  of distance or similarity metric they like. In the example below, we use a 
  **Manhattan** distance, which is defined as `d <- sum(abs(x.i - x.j))`, where `x.i` 
  is a vector of feature values for observation `i`, and `x.j` is a vector with the 
  corresponding feature values of observation `j`. So this distance is the sum of the 
  absolute value of the differences in individual feature values for the two observations. 
  A major difference from the Euclidean distance is the absence of squared terms, which 
  moderates the influence of large difference in individual variables as well as the 
  influence of outliers.

Another way to try and determine a good value of `k` (number of groups) for any clustering
  algorithm that assigns observations to distinct groups (k-means and k-mediods qualify,
  hierarchical clustering, which we describe next does not) is to use a silhouette plot. 
  Let `d.ij` be the average distance between observation `i` and the other observations in 
  the same cluster `j`. Let `d.ih` be the average distance between observation `i` 
  and the observations assigned to a different cluster `h`, where `h` is the cluster with the
  smallest such average distance to observation `i`. Then the **silhouette width** is 
  defined as `sil <- (d.ih - d.ij) / max(d.ih, d.ij)`. A compact well separated cluster will
  correspond to `d.ih` being considerably larger than `d.ij`. In the extreme, this will 
  result in `sil <- (d.ih - 0) / max(d.ih, 0) == d.ih / d.ih == 1`. At the other extreme,
  if clustering is completely wrong: `sil <- (0 - d.ij) / max(0, d.ij) == -d.ij / d.ij == -1.
  In general, this value will be close to one for observations that are well separated from 
  other clusters. For observations closer to the boundary between clusters, the value will 
  approach zero. A negative value suggests the observation was placed in the wrong cluster. 
  Ideally, we want the silhoette width to be high and uniform (close the average silhoette 
  width) across all observations in each cluster. 

```
library(cluster)

set.seed(123)
## generate pairwise 'manhattan' distances:
d <- dist(dat[, 1:4], method='manhattan')
## feed the distance to 'partitioning around medioids' function, specifying 3 clusters:
fit <- pam(d, k=3)
si <- silhouette(fit)             ## silhouette values for each observation
head(si)
tail(si)
summary(si)
mean(si[, 'sil_width'])           ## we want all observations to have about this value

(rslt <- data.frame(actual=as.numeric(dat[, 5]), predict=fit$clustering))

## cluster labels are arbitrary: let's line them up with most frequent Species match:
i2 <- fit$cluster == 2
i3 <- fit$cluster == 3
rslt[i2, 2] <- 3
rslt[i3, 2] <- 2

## does clustering reflect species?:
caret::confusionMatrix(factor(rslt[, 1]), factor(rslt[, 2]))

par(mfrow=c(1, 1))
plot(fit, which.plot=1)           ## how the data clustered in PCA 2D space (see next section)
plot(fit, which.plot=2)           ## silhouette plot

f.plot <- function(k) {
  fit <- pam(d, k=k)
  si <- silhouette(fit)
  plot(fit, which=2, main=k, cex=0.5, mex=0.5)
  mean(si[, 'sil_width'])
}

set.seed(123)
par(mfrow=c(1, 1))
f.plot(2)
f.plot(3)
f.plot(4)
f.plot(5)

```

A different approach to describing the relative similarities between observations is to
  assign observations as nodes (leafs) of a tree-like structure, where the branching 
  pattern describes the relative similarities between observations. This is called 
  **hierarchical clustering**. This approach does not require a specification of the number 
  of clusters `k`, but it does still require a distance or similarity measure. Ideally, the 
  branch lengths are constructed so that the sum of branch lengths between any two 
  observations is equal to the distance between the observations in the original feature 
  space. This representation is called a **dendrogram**. We can estimate how well a 
  dendrogram recapitulates the original pairwise observation distances by measuring the 
  correlation between the original pairwise distances and pairwise distances reconstructed 
  from the branch lengths. The latter are called **cophenetic distances**. However, there 
  is no clear cutoff to use to judge the result.

Tree construction approaches can be broadly categorized into two groups. The first is 
  **agglomerative clustering**. In this case, we start with one cluster per observation.
  We then successively merge the two closest clusters by placing them as adjacent nodes 
  in a single tree. We repeat this process until all the observations are incorporated 
  into the resulting tree. This approach generally produces a dendrogram. There are 
  several different ways of measuring the distance between clusters. One approach is 
  called **single-linkage**, (also known as **nearest-neighbor**). Here we find the pair 
  of observations, one in each cluster, with the smallest pairwise distance. This distance 
  is used as the distance between those two clusters. This approach can lead to a small 
  number of relatively diffuse clusters. Sometimes instead of a tree suggesting clear 
  distinct groups, single-linkage results in a 'chaining' of observations to one another,
  with no clear groupings suggested. Another approach is known as **complete-linkage** or
  **furthest neighbor** linkage, in which case we use the largest pairwise distance 
  between observations in the two clusters as an estimate of distance between clusters. 
  This tends to result in more compact clusters, but often tends to produce the impression 
  of more clusters than are actually present. One approach that tends to produce 
  intermediate results is **average** linkage otherwise known as the **unweighted pair 
  group method with arithmetic mean** or **UPGMA**, where we take the average distance 
  between all observations in one cluster and all the observations in the other cluster as 
  the distance between groups. One drawback of this approach is that while 
  furthest-neighbor and nearest-neighbor methods only depend on the ordering of distances, 
  and are therefore invariant under monotonic transformations of the variables, the UPGMA 
  method can give very different results after variable transformation.

The second approach to building a tree is called **divisive clustering**. Here we initially
  assign all observations to a single cluster. We then look for potential splits that result
  in two clusters with the largest distance between each other, placing the two resulting
  clusters at the two largest branches of the tree. Then we repeat the process, splitting
  clusters until each branch terminates in a single observation. This process does not 
  necessarily result in a dendrogram. This approach is much less used, much less studied,
  and (depending on algorithm) potentially much more computationally intensive than 
  agglomerative clustering.

Every approach described thus far, with the exception of combinatorial clustering, is an 
  approximate non-exhaustive search method. For k-means and k-medioids, one should generally
  try the procedure with many different initial random assignments of cluster centers and
  select the solution that most minimizes the within-cluster sum-of-distances. For hierarchical
  clustering, we can repeat the process with different bootstrap samples of the observations
  (sampling with replacement) and see how consistent the branching patterns are across 
  bootstraps in order to examine how reliable the branching patterns are likely to be.

```
## single linkage clustering based on manhattan distances:
d1 <- dist(dat[, 1:4], method='manhattan')
fit <- hclust(d1, method='single')
d2 <- cophenetic(fit)
cor(d1, d2)
plot(fit, cex=0.75, labels=substr(rownames(dat), 1, 3))

## complete linkage clustering based on manhattan distances:
d1 <- dist(dat[, 1:4], method='manhattan')
fit <- hclust(d1, method='complete')
d2 <- cophenetic(fit)
cor(d1, d2)
plot(fit, cex=0.75, labels=substr(rownames(dat), 1, 3))

## average linkage clustering based on manhattan distances:
d1 <- dist(dat[, 1:4], method='manhattan')
fit <- hclust(d1, method='average')
d2 <- cophenetic(fit)
cor(d1, d2)
plot(fit, cex=0.75, labels=substr(rownames(dat), 1, 3))

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Principal components

Dimension reduction refers to a process in which we try to reproduce relationships between 
  observations present in the original feature space using a representation involving a 
  smaller number of features. This is similar to the process of lossy data compression, 
  where we try to reduce a size of a dataset in a way that retains as much of the salient
  information in the original represenation as possible. **Principal components analysis**, 
  or **PCA** is probably the best known form of dimension reduction. PCA creates constructs
  new features as linear combinations of the original features. For instance, if we start 
  with three features `x.1`, `x.2` and `x.3`, PCA generates three new features (lets call 
  them `y.1`, `y.2` and `y.3`) that are linear combinations of the original features:

```
y.1 <- b.11*x.1 + b.12*x.2 + b.13*x.3
y.2 <- b.21*x.1 + b.22*x.2 + b.23*x.3
y.3 <- b.31*x.1 + b.32*x.2 + b.33*x.3

```

Here `b.ij` are constant coefficients. The coefficients `b.11`, `b.12` and `b.13` (or `b.1j`) 
  are selected so as to maximize the variance of `y.1`. The resulting formula for `y.1` describes
  a direction (ray from the origin) in the original feature space (`x.j` space). This is the 
  direction of maximum variance of the observations. So if your data cloud looks like an 
  American football, and is centered so the football center is at the origin, the first 
  direction will lay along the long axis of the football. If we extend this ray in both directions
  to form a line, the sum of squared distances from the observations to this line will be less than
  for any other possible line. Thus, just like we can think of the mean as being the best point for 
  fitting the data in the MSE sense, we can think of the first principal component as being the 
  line that best fits the data in the MSE sense. This is a very similar concept to that employed 
  when fitting linear models, except there is no designated response variable. For `y.2`, the 
  `b.2j` are then selected so as to maximize the variance of `y.2` while ensuring that the 
  correlation between `y.1` and `y.2` is zero. This is equivalent to repeating the earlier process, 
  but using the residuals between the observations and the first PCA as input to this step. Finally, 
  the `b.3j` are selected so as to maximize the variance of `y.3` while keeping its correlation 
  with the previous `y.i` (`y.1` and `y.2`) zero.

This process results in the same number of derived features `y.i` as there are original features 
  `x.j`. It also results in assignment of new coordinates in the derived feature space for each
  observation in the original feature space. If we want to use only one PCA derived feature to 
  reconstruct the position of observations in the original feature space, using the `y.1` 
  coordinates for each observation (which define the point along the line defined by `y.1` 
  in the original feature space onto which the original observation is projected) will 
  produce the closest possible reconstruction, in the sense of minimizing the sum of squared 
  distances (in the original feature space) from the observations to their corresponding 
  positions on the `y.1` line. That is, each reconstructed observation is a point on the `y.1` 
  line defined in the original feature space. If we are willing to put up with a second feature 
  (we usually are, especially when trying to generate a two-dimensional plot of the observations 
  positions in the new feature space), adding the information about each observations position 
  along the second principal component `y.2` will decrease the sum of squared distances from 
  observations to their reconstructed positions in the original feature space more than any 
  other choice. Now the reconstructed observation positions represented in the original feature 
  space will be points in the plane uniquely defined by (and containing) the two intersecting 
  lines `y.1` and `y.2`. Adding the last feature, `y.3`, will allow the reconstructed 
  observation positions in the original feature space to exactly match their original positions. 
  That is, we can reconstruct the original feature space without any information loss when we 
  use all the components from PCA.

Sometimes analysts might be tempted to use PCA to reduce dimensionality or multicollinearity 
  before applying a classification algorithm. PCA may improve multicollinearity in this case,
  but there are other methods that are better suited for this situation: both **partial-least
  squares analysis** or **PLS** and **linear discriminant analysis** or **LDA** are designed to 
  find directions (as linear combinations of the original features) in the original space along 
  which class separations are greatest. This is made possible by provision of explicit class labels 
  for the observations (PLS and LDA are supervised methods), which is something that PCA was not 
  designed to take into account. PCA chooses the directions of maximum variance in the feature data, 
  which do not necessarily coincide with the direction along which the classes are most easily 
  separated. PCA is better suited for data exploration and for preprocessing the data in order to
  reduce dimensionality before applying other unsupervised methods that would otherwise be 
  negatively affected by the curse of dimensionality.

A basic R installation provides two methods for computing principal components, `princomp()` and 
  `prcomp()`. Their algorithm, output and interpretation differ somewhat. Here, we demonstrate the 
  use of `prcomp()` because the computations uses are more numerically stable. This function returns 
  **eigenvalues** and **eigenvectors** that can be used to derive the principal components. The
  eigenvalues tell us how much of the variance in the original dataset is accounted for by each
  successive principal component, and the eigenvectors define the directions of each component. The
  PCA fitting process is reproducible, so seeding and repeated fitting is not required.

```
rm(list=ls())

dat <- iris
## we aren't scaling this time so the calculations are more clear
rownames(dat) <- paste(substr(dat[, 5], 1, 3), 1:nrow(dat))
summary(dat)

## PCA:
(fit <- prcomp(dat[, 1:4]))
summary(fit)                     ## cumulative variance: indicator of information loss
class(fit)
is.list(fit)
names(fit)

fit$sdev                         ## the eigenvalues define component magnitudes
fit$sdev^2                       ## squared eigenvalues are variances of successive components
sum(fit$sdev^2)                  ## if features scaled, sums to number of features
fit$sdev^2 / sum(fit$sdev^2)     ## proportion of variance accounted for

fit$rotation                     ## columns are eigenvectors, which define component directions
fit$center                       ## variables centered by subtracting these means
dat[1, 1:4]                      ## original feature coordinates for first observation
dat[1, 1:4] - fit$center         ## centered feature coordinates for first observation

## how observation coordinates along principle components are calculated:
(dat[1, 1:4] - fit$center) * fit$rotation[, 'PC1']
sum((dat[1, 1:4] - fit$center) * fit$rotation[, 'PC1'])
sum((dat[1, 1:4] - fit$center) * fit$rotation[, 'PC2'])
sum((dat[1, 1:4] - fit$center) * fit$rotation[, 'PC3'])
sum((dat[1, 1:4] - fit$center) * fit$rotation[, 'PC4'])
fit$x[1, ]

par(mfrow=c(1, 1))
plot(fit)                        ## variances of PCs (squared eigenvalues)

plot(fit$x[, 1], fit$x[, 2], type='n')
i <- dat$Species == 'setosa'
points(fit$x[i, 1], fit$x[i, 2], pch='x', col='cyan')
i <- dat$Species == 'versicolor'
points(fit$x[i, 1], fit$x[i, 2], pch='o', col='orangered')
i <- dat$Species == 'virginica'
points(fit$x[i, 1], fit$x[i, 2], pch='+', col='magenta')
legend(
  'topright',
  legend=c('setosa', 'versicolor', 'virginica'),
  col=c('cyan', 'orangered', 'magenta'),
  pch=c('x', 'o', '+')
)

```

### Check your understanding 2

1) question here

[Return to index](#index)

---

### t-sne and umap

One problem with PCA is that because it attempts to preserve the variance in the original
  feature space (e.g. the first component is the direction of greatest variance in the 
  original space), and because variances are based on squared distances, large distances
  play a larger role in determining the components than do small distances. As a result,
  lower-dimensional representations using PCA tend to reconstruct long-distances between
  observations (**global structure**) much better than short-distances (**local structure**). 
  Another issue with PCA is that sometimes the relationships between what we might consider 
  meaningful groupings of observations are better defined in terms of non-linear 
  relationships and interactions between the original variables that tend to not be well 
  preserved in lower dimensional PCA reconstructions. 

A different approach is based on trying to find a non-linear lower-dimensional representation 
  which best matches up probabilistically described neighborhoods between the original and 
  reduced feature spaces. For instance, we can probabilistically define the neighbors of
  an observation `x.i` by centering a Gaussian (normal) curve, or **kernel** (the shape of 
  the probability density of a normal distribution) on top of the observation. We then assign
  a probability of observation `x.j` (whose distance from `x.i` is `d.ij`) being a neighbor 
  of `x.i` with a probability proportional to the height of the Gaussian curve at a distance 
  `d.ij` from `x.i`. Another curve used to probabilistically describe neighborhoods around 
  individual observations is a t-distribution with a single degree of freedom (equivalent to 
  a Cauchy distribution), which has a much heavier tail, and therefore tends to include 
  observations further from `x.i` in the neighborhood of `x.i` with a higher probability
  than when using a Gaussian kernel. We draw both curves below for comparison:

```
rm(list=ls())

x <- seq(from=0, to=7, by=0.01)
y.gauss <- exp(-(x^2))
y.cauch <- 1 / (1 + x)
y.g2 <- exp(-(x * x / 2))

par(mfrow=c(1, 1))
plot(range(x), range(c(y.gauss, y.cauch)), xlab='distance', ylab='relative probability', type='n')
abline(h=0)
lines(x=x, y=y.gauss, col='magenta', lty=2)
lines(x=x, y=y.cauch, col='orangered', lty=3)
lines(x=x, y=y.g2, col='cyan', lty=4)

legend(
  'topright',
  legend=c('Gaussian', 't1'),
  col=c('magenta', 'orangered'),
  lty=c(2, 3)
)

```

We can use this approach to probabilistically define neighborhoods around each observation in
  not only the original feature space, but also in any lower-dimensional space. After we 
  define neighborhoods in the original feature space, we try to find a set of observation 
  coordinates in a lower-dimensional space such that the average probability of including 
  `x.j` within the neighborhood of `x.i` matches as closely as possible for all `i` and 
  `j` between the higher dimensional representation and the lower dimensional one. We can 
  express the degree of divergence between these two probability distributions with the 
  **Kullback-Leibler divergence** or **KL-divergence**. This metric is based on the information 
  theory concept of **entropy**. Entropy describes the amount of information present in a data 
  representation. For a given dataset and probability distribution, the entropy is a function 
  of the **surprisal** of each observation, which is inversely related to the probability of 
  the observation under the specified distribution: `surp.i <- -log2(p(x.i))`. Given a set of 
  observations and a corresponding probability distribution, the entropy of the observation 
  measurement data is `H <- sum(p(x.i) * surp.i) == -sum(p(x.i) * log2(p(x.i))`. Given a set 
  of observations and two probability distributions `p(x.i)` and `q(x.i)`, the KL-divergence 
  between the distributions is the difference in entropy between the two distributions: 
  `D <- sum(p(x.i) * log2(p(x.i))) - sum(q(x.i) * log2(q(x.i)))`. Thus the KL-divergence 
  expresses the information loss when using distribution `q(x.i)` to represent `p(x.i)`. We 
  apply this idea in the current context to define the loss of information resulting from 
  defining neighborhoods around points using the lower-dimensional representation instead of 
  the higher dimensional original representation of the observations. We select the observation 
  coordinates in the lower-dimensional space so as to minimize the KL-divergence. In practice,
  we cannot reliably find a global minimum solution so instead we use heuristic approaches to
  get a reasonable solution (though it may well turn out to only be a local minimum).

One problem that results when using a Gaussian kernel for generating neighborhood probabilities
  in both the original and derived feature spaces is that we can end up with situations where
  clusters near the center of the feature space are too compact (collapsed) or observations 
  further from the center are pushed too far away (their distance from the center is inflated).
  This is often referred to as the **crowding problem**. The **t-Distributed Stochastic Neighbor 
  Embedding** or **t-sne** algorithm approaches the crowding problem by using a Gaussian 
  distribution for defining neighborhoods in the orginal higher dimensional space, but then uses 
  a t-distribution with a single degree of freedom (has a much heavier tail) for defining 
  neighborhoods in the lower dimensional space. This tends to result in a more even representation 
  of distances across the feature space. However, it is often difficult to simultaneously preserve 
  both local structure and global structure. The balance between these two can be tuned using the 
  **perplexity** setting, which roughly corresponds to the number of neighbors used to define a 
  neighborhood. Larger values of perplexity tend to favor preservation of global structure over 
  the preservation of local structure. Too low a value for the perplexity will tend to result in 
  many small false clusters. Too high a value will tend to result in many points being driven to 
  the edges of the plot. Large datasets (many observations) and noisy data tend to benefit from 
  larger perplexity. The t-sne algorithm tends to do a better job than PCA of preserving local
  structure.

The plots generated by t-sne must be interpreted with caution, they do not attempt to preserve
  pairwise distances between observations. As a result, the sizes of clusters and the relative 
  spacing between clusters has very little meaning. Most implementations of the t-sne algorithm 
  only provide two-dimensional representations, though the method could in principle be extended 
  to three or more dimensions by increasing the degrees-of-freedom of the t-distribution used in 
  the reduced feature space. The t-sne algorithm is also not immune to the curse of dimensionality. 
  If you have more than 50 features, you may need to either do some feature selection or reduce 
  the dimensionality using a method like PCA, before attempting t-sne. The t-sne algorithm tends
  to be quite slow, which is made worse by the need to try multiple iterations in order to avoid 
  shallow local minima in the loss, as well as the need to explore the effects of varying the 
  perplexity.

```
library(tsne)

rm(list=ls())

dat <- iris
summary(dat)
dat[, 1:4] <- scale(dat[, 1:4])
rownames(dat) <- paste(substr(dat[, 5], 1, 2), 1:nrow(dat))

set.seed(123)
## note minimum error around iteration 600:
fit <- tsne(dat[, 1:4], perplexity=10)
class(fit)
dim(fit)
head(fit)

plot(fit[, 1], fit[, 2], type='n')
i <- dat$Species == 'setosa'
points(fit[i, 1], fit[i, 2], pch='x', col='cyan')
i <- dat$Species == 'versicolor'
points(fit[i, 1], fit[i, 2], pch='o', col='orangered')
i <- dat$Species == 'virginica'
points(fit[i, 1], fit[i, 2], pch='+', col='magenta')
legend(
  'bottomleft',
  legend=c('setosa', 'versicolor', 'virginica'),
  col=c('cyan', 'orangered', 'magenta'),
  pch=c('x', 'o', '+')
)

set.seed(123)
fit <- tsne(dat[, 1:4], perplexity=10, max_iter=600)

plot(fit[, 1], fit[, 2], type='n')
i <- dat$Species == 'setosa'
points(fit[i, 1], fit[i, 2], pch='x', col='cyan')
i <- dat$Species == 'versicolor'
points(fit[i, 1], fit[i, 2], pch='o', col='orangered')
i <- dat$Species == 'virginica'
points(fit[i, 1], fit[i, 2], pch='+', col='magenta')
legend(
  'topleft',
  legend=c('setosa', 'versicolor', 'virginica'),
  col=c('cyan', 'orangered', 'magenta'),
  pch=c('x', 'o', '+')
)

set.seed(321)
f.fit <- function(perp) tsne(dat[, 1:4], perplexity=perp)
perps <- c(2, 4, 8, 16, 32, 64)
fits <- lapply(perps, f.fit)

f.plot <- function(fit) {
  plot(fit[, 1], fit[, 2], type='n')
  i <- dat$Species == 'setosa'
  points(fit[i, 1], fit[i, 2], pch='x', col='cyan')
  i <- dat$Species == 'versicolor'
  points(fit[i, 1], fit[i, 2], pch='o', col='orangered')
  i <- dat$Species == 'virginica'
  points(fit[i, 1], fit[i, 2], pch='+', col='magenta')
}

par(mfrow=c(2, 3))
sapply(fits, f.plot)

```

umap: https://pair-code.github.io/understanding-umap/

The **Uniform Manifold Approximation and Projection** or **UMAP** method is conceptually 
  similar to t-sne, but is intended to strike a better balance between preserving local
  and global structure, so that global structure is better preserved than with t-sne. So
  both methods may show similar clustering of observations (local structure), but UMAP 
  is supposed to do a better job of reproducing the relationships between clusters (global
  structure). Instead of using a Gaussian curve `exp(-(d.ij^2))`, where `d.ij` is the 
  distance from observation `i` to observation `j`, for defining probabilistic 
  neighborhoods in the original feature space, UMAP uses the function `exp(-(d.ij * rho.i)`, 
  where `rho.i` is the distance from observation `i` to its nearest neighbor. This results 
  in a curve with a tail intermediate between the Gaussian and Cauchy curves. So this
  will tend to include more observations in the neighborhood of a given observation than
  the Gaussian, but will tend to include less observations in the neighborhood than the
  Cauchy curve.

In addition to exploring the effects of the `n_neighbors` parameter (similar to perplexity),
  UMAP is also fairly strongly affected by a `min_dist`, which determines how closely points
  in the reduced space tend to be. Fortunately, the UMAP algorithm tends to be much faster 
  than t-sne, so the extra tuning is not too time consuming.

Regardless of what methods we use for dimensional reduction, we should keep in mind that 
  there are certain arrangements of observations in higher dimensional spaces that 
  simply cannot be reproduced in a lower dimensional space. For instance, although we
  can represent three equidistant observations in two dimensions (an equilateral triangle), 
  this arrangement is impossible to represent in one dimension (along a line). Similarly, 
  although we cannot represent four equidistant points in two dimensions, we can represent 
  that relationship in three dimensions (think about a four-sided die, where each face is 
  an equilateral triangle).

```
library(umap)
rm(list=ls())

dat <- iris
summary(dat)
dat[, 1:4] <- scale(dat[, 1:4])
rownames(dat) <- paste(substr(dat[, 5], 1, 2), 1:nrow(dat))

set.seed(1)
fit <- umap(dat[, 1:4])
fit
class(fit)
is.list(fit)
str(fit)
summary(fit)

dim(fit$layout)

par(mfrow=c(1, 1))
plot(fit$layout[, 1], fit$layout[, 2], type='n')
i <- dat$Species == 'setosa'
points(fit$layout[i, 1], fit$layout[i, 2], pch='x', col='cyan')
i <- dat$Species == 'versicolor'
points(fit$layout[i, 1], fit$layout[i, 2], pch='o', col='orangered')
i <- dat$Species == 'virginica'
points(fit$layout[i, 1], fit$layout[i, 2], pch='+', col='magenta')
legend(
  'topleft',
  legend=c('setosa', 'versicolor', 'virginica'),
  col=c('cyan', 'orangered', 'magenta'),
  pch=c('x', 'o', '+')
)

umap.defaults

f.fit <- function(obj) {
  config <- umap.defaults          ## start with default parameterization
  config$n_neighbors <- obj['k']   ## number of nearest neighbors
  config$min_dist <- obj['d']      ## minimum distance between points in reconstruction
  config$random_state <- 123       ## internal seed setting
  umap(dat[, 1:4], config=config)  ## return umap fit using specified n_neighbors and min_dist
}

## parameter values to try:
ks <- c(2, 4, 8, 16, 32, 64)       
ds <- c(0.001, 0.01, 0.1, 0.5, 0.9)
x <- expand.grid(ks, ds)
colnames(x) <- c('k', 'd')
x

## try those parameter values:
set.seed(123)
fits <- apply(x, 1, f.fit)

## a plotting function for each fit:

f.plot <- function(fit) {
  plot(fit$layout[, 1], fit$layout[, 2], type='n')
  i <- dat$Species == 'setosa'
  points(fit$layout[i, 1], fit$layout[i, 2], pch='x', col='cyan')
  i <- dat$Species == 'versicolor'
  points(fit$layout[i, 1], fit$layout[i, 2], pch='o', col='orangered')
  i <- dat$Species == 'virginica'
  points(fit$layout[i, 1], fit$layout[i, 2], pch='+', col='magenta')
}

## plot min_dist (from top=ds[1] to bottom=ds[length(ds)]) 
##   vs n_neighbors (from left=ks[1] to right=ks[length(ks)]):

par.old <- par()                  ## save graph settings
par(mfrow=c(5, 6), mar=c(1, 1, 1, 1), cex=0.5, mex=0.5)
sapply(fits, f.plot)

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
