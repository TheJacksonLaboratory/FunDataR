# Fundamentals of computational data analysis using R
## Machine learning: dimension reduction
#### Contact: mitch.kostich@jax.org

---

### Index

- [Clustering basics](#clustering-basics)
- [PCA and PLS](#pca-and-pls)
- [t-sne and umap](#t-sne-and-umap)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Clustering basics

k-means

```
library(caret)

dat <- iris
summary(dat)
dat[, 1:4] <- scale(dat[, 1:4])
rownames(dat) <- paste(substr(dat[, 5], 2), 1:nrow(dat))
summary(dat)

set.seed(1)
fit <- kmeans(dat[, 1:4], centers=3)

fit
class(fit)
is.list(fit)
names(fit)
str(fit)

(rslt <- cbind(as.numeric(dat[, 5]), fit$cluster))

i1 <- fit$cluster == 1
i2 <- fit$cluster == 2
i3 <- fit$cluster == 3

rslt[i1, 2] <- 2
rslt[i2, 2] <- 3
rslt[i3, 2] <- 1

caret::confusionMatrix(factor(rslt[, 1]), factor(rslt[, 2]))

```

k-medioids

```
library(cluster)

set.seed(1)
fit <- pam(dat[, 1:4], k=3)
(rslt <- cbind(as.numeric(dat[, 5]), fit$clustering))

i1 <- fit$cluster == 1
i2 <- fit$cluster == 2
i3 <- fit$cluster == 3

rslt[i1, 2] <- 1
rslt[i2, 2] <- 3
rslt[i3, 2] <- 2

caret::confusionMatrix(factor(rslt[, 1]), factor(rslt[, 2]))

```

hierarchical clustering

```
d1 <- dist(dat[, 1:4])
fit <- hclust(d1, method='single')
d2 <- cophenetic(fit)
cor(d1, d2)

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### PCA and PLS

PCA; loadings etc.

```
rm(list=ls())

dat <- iris
summary(dat)
dat[, 1:4] <- scale(dat[, 1:4])
rownames(dat) <- paste(substr(dat[, 5], 2), 1:nrow(dat))
summary(dat)

fit <- prcomp(dat[, 1:4])
class(fit)
is.list(fit)
names(fit)
str(fit)

fit
summary(fit)

fit$rotation                     ## variable loadings onto components
fit$rotation['Sepal.Length', ] 
fit$x[1, ]
sum(fit$rotation['Sepal.Length', ]  * fit$x[1, ])
dat$Sepal.Length[1] 

dat[1, 1:4]
fit$rotation[, 'PC1'] 
sum(dat[1, 1:4], fit$rotation[, 'PC1'])

apply(fit$rotation^2, 1, sum)
apply(fit$rotation^2, 2, sum)

plot(fit)                        ## variance proportions

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

PLS

```
code here

```

### Check your understanding 2

1) question here

[Return to index](#index)

---

### t-sne and umap

tsne

```
library(tsne)

rm(list=ls())

dat <- iris
summary(dat)
dat[, 1:4] <- scale(dat[, 1:4])
rownames(dat) <- paste(substr(dat[, 5], 2), 1:nrow(dat))

set.seed(1)
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

umap

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
