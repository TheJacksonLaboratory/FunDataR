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
  it using a test-set. This process is called **supervised learning**, because the 
  training-set and test-set response values provide a guide that can be used as ground
  truth for supervising the model fitting process. However, in many scenarios, there may be 
  no particular response variable of interest. Instead, we might be interested in describing 
  the relative similarity of observations to one another, given a set of variables (we can
  call **features** in this context) measured on each observation. This type of task is 
  called **unsupervised learning** because the ground truth for the task does not lie in
  any feature values. For instance, if we were to gather morphometric data on iris plants,
  we might not know how many species are really represented in our data set. But we might be
  able to use the measurement data to describe the **similarity** (or its converse, the 
  **distance**) between any two observations. By considering the relative similarities 
  between observations in the data-set, we may be able to split all the observations into 
  **clusters** where each cluster contains observations with higher similarity to one 
  another than to observations in other clusters. However, the true cluster assignments 
  and even the true number of clusters cannot be determined from the data, since there is
  no response variable with species information. Instead, the clustering process minimizes
  some sort of measure of clustering effectiveness.

One simple loss function for clustering is the sum of distances between observations 
  within each cluster. We can in principle optimize this metric for any particular cluster 
  number by exhaustively trying all possible assignments of observations to clusters. This 
  **combinatorial clustering** approach is only feasible for assigning a small number of 
  observations to a small number of clusters. For instance, assigning 19 observations to 
  4 clusters actually involves about 10^10 distinct arrangements. For larger tasks, we 
  instead use approximate non-exhaustive methods to search the solution space. 

One of the best known algorithms for assigning observations to a user-selected number `k` 
  of clusters is the **k-means** algorithm. It begins by picking `k` points (not necessarily
  corresponding to an observation location) in the feature space as cluster centers. Then,
  we:

1) assign all points to the nearest cluster center.

2) recompute cluster centers as the mean of all observations assigned to the cluster

3) repeat steps 1 and 2 till the process converges.

This process proceeds in a step-wise manner, where the history of steps determines the 
  available paths for subsequent steps. This can result in the process finding a solution
  that represents a **local minimum** in the loss function, rather than the **global 
  minimum**. We stand a better chance of finding the global minimum, or at least a decent
  local minimum by starting the algorithm with different randomly chosen cluster centers,
  the choosing the solution which does the best job of reducing the loss function.

Another issue with k-means clustering is the need to specify a cluster number `k` to the algorithm.
  Since this is often not known for certain, we try a range of values. Often this will result
  in steady improvement in the loss function with rising `k` until the real number of clusters is
  reached, at which point the loss function will begin to level out. If we plot the loss function
  for successive values of `k`, we often see a 'kink' in the plot at the correct value of `k`, 
  corresponding to a sudden relative leveling of the loss function. Another approach to selecting
  `k` is to use a **gap statistic** to identify the kink more quantitatively: we can randomly 
  distribute the same number of observations and calculate differences in within group distance 
  between the randomly spread data and actual data as a function of changing `k`. We then pick 
  the `k` where the difference is maximal. 

```
library(caret)

dat <- iris
summary(dat)
dat[, 1:4] <- scale(dat[, 1:4])
rownames(dat) <- paste(substr(dat[, 5], 1, 3), 1:nrow(dat), sep='')
summary(dat)

set.seed(123)
fit <- kmeans(dat[, 1:4], centers=3, nstart=10)

fit
class(fit)
is.list(fit)
names(fit)
str(fit)

(rslt <- cbind(as.numeric(dat[, 5]), fit$cluster))
caret::confusionMatrix(factor(rslt[, 1]), factor(rslt[, 2]))

i1 <- fit$cluster == 1
i2 <- fit$cluster == 2
i3 <- fit$cluster == 3

d1 <- dist(dat[i1, 1:4])
d2 <- dist(dat[i2, 1:4])
d3 <- dist(dat[i3, 1:4])

class(d1)
is.list(d1)
attributes(d1)

sum(c(d1, d2, d3))

f.loss <- function(idx) {
  d <- dist(dat[idx, 1:4])
  sum(d)
}

f.fit <- function(k) {
  fit <- kmeans(dat[, 1:4], centers=k, nstart=10)
  sum(tapply(1:nrow(dat), factor(fit$cluster), f.loss))
}

set.seed(123)
ks <- 2:10
score <- sapply(ks, f.fit)

par(mfrow=c(1, 1))
plot(x=ks, y=score, type='l')

```

The k-means process requires recomputing cluster centers at each step in the process
  as the average position across all observations in the cluster. This is equivalent
  to finding the center that minimizes the mean-squared distances between observations 
  and the cluster center, so the process is sensitive to outliers. In addition, the 
  distance metric is assumed to be Euclidean. If we have measures of distance or 
  similarity that are ordered categories (like subjective impressions of 'bad', 'ok', 
  'good', and 'excellent') or yes/no categories, k-means can be an unnatural choice
  since taking averages of such variables is not a clearly defined operation. The
  **k-mediods** algorithm is similar to the k-means algorithm, except that each 
  cluster center is restricted to coincide with one of the observations assigned to
  the cluster, which makes incorporating non-Euclidean distances and non-numeric
  variables more natural. The process starts by randomly selecting `k` observations
  as cluster centers, then assigns all observations to the closest center, then
  recompute each cluster center so it corresponds to the observation in the cluster
  that has the smallest sum of distances (need not be Euclidean) to the other points
  in the cluster.

We can also use a silhouette plot to help pick the optimal value of `k`. Let `d.ij` be 
  the average distance between observation `i` and the other observations in the same 
  cluster `j`. Let `d.ik` be the average distance between the same observation `i` and the 
  observations assigned to a different cluster `k`, where `k` is the cluster with the
  smallest such average distance to observation `i`. Then the **silhouette width** is 
  defined as `sil <- (d.ik - d.ij) / max(d.ik, d.ij)`. This value will be close to one for 
  observations that are well separated from other clusters. For observations closer to
  the boundary between clusters, the value will approach zero. A negative value 
  suggests the observation was placed in the wrong cluster. Ideally, we want the 
  silhoette width uniformly high (close the average silhoette width) across all 
  observations in all clusters. 

```
library(cluster)

set.seed(123)
d <- dist(dat[, 1:4], method='manhattan')
fit <- pam(d, k=3)
si <- silhouette(fit)
head(si)
tail(si)
summary(si)
mean(si[, 'sil_width'])

(rslt <- data.frame(actual=as.numeric(dat[, 5]), predict=fit$clustering))

i2 <- fit$cluster == 2
i3 <- fit$cluster == 3
rslt[i2, 2] <- 3
rslt[i3, 2] <- 2

caret::confusionMatrix(factor(rslt[, 1]), factor(rslt[, 2]))

par(mfrow=c(1, 1))
plot(fit, which.plot=1)
plot(fit, which.plot=2)

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
  pattern and branch lengths describe the relative similarities between observations.
  This is called **hierarchical clustering**. Ideally, the branch lengths are chosen so 
  that the sum of branch lengths between any two observations is equal to the distance 
  between the observations in the feature space.

Hierarchical greedy.

```
d1 <- dist(dat[, 1:4], method='manhattan')
fit <- hclust(d1, method='single')
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

Principal components analysis, or **PCA** is probably the most well known form of dimension 
  reduction. PCA creates new features constructed as linear combinations of the old 
  features. For instance, if we start with three features `x.1`, `x.2` and `x.3`, PCA generates 
  three new features (lets call them `y.1`, `y.2` and `y.3`) that are linear combinations of 
  the original features:

```
y.1 <- b.11*x.1 + b.12*x.2 + b.13*x.3
y.2 <- b.21*x.1 + b.22*x.2 + b.23*x.3
y.3 <- b.31*x.1 + b.32*x.2 + b.33*x.3
```

Here `b.ij` are constant coefficients. The coefficients `b.11`, `b.12` and `b.13` (or `b.1j`) 
  are selected so as to maximize the variance of `y.1`. The resulting formula for `y.1` describes
  a direction (ray from the origin) in the original feature space (`x.j` space). This is the 
  direction of maximum variance of the observations. The line defined by extending this direction 
  ray minimizes the mean of the squared deviations of the observations from the line more than 
  any other line that can be drawn in the original `x.j` space. Thus, we can think of the mean as 
  being the best point for fitting the data in the MSE sense, we can think of the first principal 
  component as being the line that best fits the data in the MSE sense. This is a very similar 
  procedure to that employed when fitting linear models, except there is no designated response 
  variable. For `y.2`, the `b.2j` are then selected so as to maximize the variance of `y.2` while 
  ensuring that the correlation between `y.1` and `y.2` is zero. This is equivalent to repeating 
  the earlier process, but using the residuals between the observations and the first PCA as input 
  to this step. Finally, the `b.3j` are selected so as to maximize the variance of `y.3` while 
  keeping its correlation with the previous `y.i` (`y.1` and `y.2`) zero.

This process results in the same number of derived features `y.i` as there are original features 
  `x.j`. It also results in assignment of new coordinates in the derived feature space for each
  observation in the original feature space. If we want to use one PCA derived feature to 
  reconstruct the position of observations in the original feature space, using the `y.1` 
  coordinates for each observation (which define the point along the line defined by `y.1` 
  in the original feature space onto which the original observation is projected) will 
  produce the closest possible reconstruction, in the sense of minimizing the sum of squared 
  distances (in the original feature space) from the observations to their corresponding positions 
  on the `y.1` line. That is, each reconstructed observation 
  is a point on the `y.1` line defined in the original feature space. If we are willing to put up 
  with a second feature (we usually are, especially when trying to plot the observations positions 
  in the new feature space), adding the information about each observations position along the 
  second principal component `y.2` will decrease the sum of squared distances in the original 
  feature space from observations to their reconstructed positions. Now the reconstructed 
  observation positions represented in the original feature space will be points in the plane 
  uniquely defined by (and containing) the two intersecting lines `y.1` and `y.2`. Adding the last 
  feature, `y.3`, will allow the reconstructed observation positions in the original feature space 
  to exactly match their original positions. That is, we can reconstruct the original feature 
  space without any information loss when we use all the components from PCA.

Sometimes analysts might be tempted to use PCA to reduce dimensionality or multicollinearity 
  before applying a classification algorithm. PCA may improve multicollinearity in this case,
  but there are other methods that are better suited for this situation: both **partial-least
  squares analysis** or **PLS** and **linear discriminant analysis** or **LDA** are designed to 
  find directions (as linear combinations of the original features) in the original space along 
  which class separations are greatest. This is made possible by provision of explicit class labels 
  for the observations (PLS and LDA are supervised methods), which is something that PCA was not 
  designed to take into account. PCA chooses the directions of maximum variance in the feature data, 
  which do not necessarily coincide with the direction along which the classes are most easily 
  separated.

A basic R installation provides two methods for computing principal components, `princomp()` and 
  `prcomp()`. Their algorithm, output and interpretation differ somewhat. Here, we demonstrate the 
  use of `prcomp()` because the computations uses are more numerically stable. This function returns 
  **eigenvalues** and **eigenvectors** that can be used to derive the principal components. The
  eigenvalues tell us how much of the variance in the original dataset is accounted for by each
  successive principal component, and the eigenvectors define the directions of each component.

```
rm(list=ls())

dat <- iris
summary(dat)
## we aren't scaling this time so the calculations are more clear
rownames(dat) <- paste(substr(dat[, 5], 1, 3), 1:nrow(dat))
summary(dat)

fit <- prcomp(dat[, 1:4])
class(fit)
is.list(fit)
names(fit)
str(fit)

fit
summary(fit)

fit$sdev                         ## the eigenvalues define component magnitudes
fit$sdev^2                       ## squared eigenvalues are variances of successive components
sum(fit$sdev^2)                  ## if features scaled, sums to number of features
fit$sdev^2 / sum(fit$sdev^2)     ## proportion of variance accounted for

fit$rotation                     ## columns are eigenvectors, which define component directions
fit$center                       ## variables centered by subtracting these means
dat[1, 1:4]
dat[1, 1:4] - fit$center
(dat[1, 1:4] - fit$center) * fit$rotation[, 'PC1']
sum((dat[1, 1:4] - fit$center) * fit$rotation[, 'PC1'])
sum((dat[1, 1:4] - fit$center) * fit$rotation[, 'PC2'])
sum((dat[1, 1:4] - fit$center) * fit$rotation[, 'PC3'])
sum((dat[1, 1:4] - fit$center) * fit$rotation[, 'PC4'])
fit$x[1, ]

fit$rotation^2
apply(fit$rotation^2, 1, sum)
apply(fit$rotation^2, 2, sum)

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
  data (e.g. the first component is the direction of greatest variance in the original 
  feature space), and because variances are based on squared distances, large distances
  play a larger role in determining the components than do small distances. As a result,
  lower-dimensional representations using PCA tend to preserve long-distances between
  observations (**global structure**) much better than short-distances (**local structure**). 
  Another issue with PCA is that sometimes the relationships between what we might consider 
  meaningful groupings of observations are better defined in terms of non-linear 
  relationships and interactions between the original variables that are not well preserved
  by PCA reconstructions. 

A different general approach is based on trying to find a non-linear lower-dimensional 
  representation which best matches up probabilistically described neighborhoods between 
  the original and derived feature spaces. For instance, we can probabilistically define 
  the neighborhood around an observation `x.i` by overlaying a Gaussian (normal) curve,
  or **kernel** (the shape of the probability density of a normal distribution) on top of 
  the observation. We then choose observation `x.j` (whose distance from `x.i` is `d.ij`) 
  as a neighbor of `x.i` with a probability proportional to the height of the Gaussian 
  curve at a distance `d.ij` from `x.i`. Another curve used to probabilistically describe 
  neighborhoods is a t-distribution with a single degree of freedom (equivalent to a 
  Cauchy distribution), which has a much heavier tail and therefore tends to include 
  observations further from `x.i` in the neighborhood of `x.i` with a higher probability
  than when using a Gaussian kernel. We draw both curves below for comparison:

```
rm(list=ls())

x <- seq(from=0, to=7, by=0.01)
y.gauss <- exp(-(x^2))
y.cauch <- 1 / (1 + x)

par(mfrow=c(1, 1))
plot(range(x), range(c(y.gauss, y.cauch)), xlab='distance', ylab='relative probability', type='n')
abline(h=0)
lines(x=x, y=y.gauss, col='magenta', lty=2)
lines(x=x, y=y.cauch, col='orangered', lty=3)

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
  express the degree of divergence between two probability distributions with the 
  **Kullback-Leibler divergence** or **KL-divergence**. This metric is based on the information 
  theory concept of **entropy**. Entropy describes the amount of information present in a data 
  representation. For a given dataset and probability distribution, the entropy is a function 
  of the **surprisal** of each observation, which is inversely related to the probability of 
  the observation under the distribution: `surp.i <- -log2(p(x.i))`. Given a set of 
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
  get a reasonable solution (though maybe not the best one).

One problem that results when using a Gaussian kernel for generating neighborhood probabilities
  in both the original and derived feature spaces is that we can end up with situations where
  clusters near the center of the feature space are too compact (collapsed) or observations 
  further from the center are pushed too far away (their distance from the center is inflated).
  This is often referred to as the crowding problem. The **t-Distributed Stochastic Neighbor 
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
  larger perplexity.
  
The plots generated by t-sne must be interpreted with caution, as they do not attempt to preserve
  pairwise distances between observations. As a result, the sizes of clusters and the relative 
  spacing between clusters has very little meaning. Most implementations of the t-sne algorithm 
  only provide two-dimensional representations, though the method could in principle be extended 
  to three or more dimensions by changing the t-distribution. The t-sne algorithm is also not 
  immune to the curse of dimensionality. If you have more than 50 features, you may need to 
  either do some feature selection or reduce the dimensionality to about 50 using a method like
  PCA before attempting t-sne.

```
library(tsne)

rm(list=ls())

dat <- iris
summary(dat)
dat[, 1:4] <- scale(dat[, 1:4])
rownames(dat) <- paste(substr(dat[, 5], 1, 2), 1:nrow(dat))

set.seed(123)
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
  'topleft',
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

f.fit <- function(k) {
  config <- umap.defaults
  config$n_neighbors <- k
  config$random_state <- 123
  umap(dat[, 1:4], config=config)
}
  
ks <- c(2, 4, 8, 16, 32, 64)
fits <- lapply(ks, f.fit)

f.plot <- function(fit) {
  plot(fit$layout[, 1], fit$layout[, 2], type='n')
  i <- dat$Species == 'setosa'
  points(fit$layout[i, 1], fit$layout[i, 2], pch='x', col='cyan')
  i <- dat$Species == 'versicolor'
  points(fit$layout[i, 1], fit$layout[i, 2], pch='o', col='orangered')
  i <- dat$Species == 'virginica'
  points(fit$layout[i, 1], fit$layout[i, 2], pch='+', col='magenta')
}

par(mfrow=c(2, 3))
sapply(fits, f.plot)

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
