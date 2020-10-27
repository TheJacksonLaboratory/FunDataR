# Fundamentals of computational data analysis using R
## Multivariate statistics: more about multiple regression
#### Contact: mitch.kostich@jax.org

---

### Index

- [Multiple testing](#multiple-testing)
- [Overfitting](#overfitting)
- [Feature selection](#feature-selection)
- [Model selection](#model-selection)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Multiple testing

intro here

p-values uniformly distributed; dangers of false positives when running hypothesis tests:

```
rm(list=ls())

set.seed(1)

R <- 10000
p.values <- rep(as.numeric(NA), R)

for(i in 1:R) {
  x <- rnorm(10, mean=0, sd=1)
  y <- rnorm(10, mean=0, sd=1)
  p.values[i] <- t.test(x, y, var.equal=T)$p.value
}

par(mfrow=c(1, 1))
hist(p.values)
summary(p.values)
sum(p.values < 0.05) / R

```

family-wise error rate (Bonferroni and Holm); vs false discovery rates:
  proportion of false positives among all positives.

```
p.adj <- p.values * length(p.values)
p.adj[p.adj > 1] <- 1
p.bonf <- p.adjust(p.values, method='bonferroni')
all.equal(p.adj, p.bonf)

p.holm <- p.adjust(p.values, method='holm')
p.bh <- p.adjust(p.values, method='BH')
p.by <- p.adjust(p.values, method='BY')

par(mfrow=c(2, 2))
hist(p.bonf, main='Bonferroni (FWER)')
hist(p.holm, main='Holm (FWER)')
hist(p.bh, main='BH (FDR)')
hist(p.by, main='BY (FDR)')

min(p.bonf)
min(p.holm)
min(p.bh)
min(p.by)

```

One more example with about 5% of experiments null is not true:

```
rm(list=ls())
set.seed(1)

R <- 1000
p.pos <- 0.05
p.values <- rep(as.numeric(NA), R)
cnt <- 0

for(i in 1 : R) {
  val <- rbinom(1, 1, prob=p.pos)
  if(val == 1) mu = 1 else mu = 0
  x <- rnorm(10, mean=mu, sd=0.5)
  y <- rnorm(10, mean=0, sd=0.5)
  p.values[i] <- t.test(x, y, var.equal=T)$p.value
  cnt <- cnt + mu
}

p.bonf <- p.adjust(p.values, method='bonferroni')
p.holm <- p.adjust(p.values, method='holm')
p.bh <- p.adjust(p.values, method='BH')
p.by <- p.adjust(p.values, method='BY')

cnt
sum(p.values < 0.05)
sum(p.bonf < 0.05)
sum(p.holm < 0.05)
sum(p.bh < 0.05)
sum(p.by < 0.05)

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Overfitting

intro here

```
rm(list=ls())

f.draw <- function(fit, lty, col) {
  x.plot <- 1:10000
  y.plot <- predict(fit, newdata=data.frame(x=x.plot))
  lines(x.plot, y.plot, lty=lty, col=col)
}

par(mfrow=c(2, 3))

n <- 2
x <- seq(from=1, to=10, length.out=n)
y <- rnorm(n, mean=0, sd=1)
fit <- lm(y ~ x)
plot(x, y, main='y ~ x')
f.draw(fit, 2, 'cyan')

n <- 3
x <- seq(from=1, to=10, length.out=n)
y <- rnorm(n, mean=0, sd=1)
fit <- lm(y ~ x)
plot(x, y, main='y ~ x')
f.draw(fit, 2, 'cyan')

fit <- lm(y ~ x + I(x^2))
plot(x, y, main='y ~ x + x^2')
f.draw(fit, 2, 'cyan')

n <- 4
x <- seq(from=1, to=10, length.out=n)
y <- rnorm(n, mean=0, sd=1)
fit <- lm(y ~ x)
plot(x, y, main='y ~ x')
f.draw(fit, 2, 'cyan')

fit <- lm(y ~ x + I(x^2))
plot(x, y, main='y ~ x')
f.draw(fit, 2, 'cyan')

fit <- lm(y ~ x + I(x^2) + I(x^3))
plot(x, y, main='y ~ x + x^2 + x^3')
f.draw(fit, 2, 'cyan')

```

[Return to index](#index)

---

### Feature selection

intro here

```
code here

```

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Model selection

intro here

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
