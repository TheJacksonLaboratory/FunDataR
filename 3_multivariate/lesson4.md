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

fit <- lm(y ~ x + x^2)
plot(x, y, main='y ~ x + I(x^2)')
f.draw(fit, 2, 'cyan')

fit <- lm(y ~ x + I(x^2) + I(x^3))
plot(x, y, main='y ~ x + x^2 + x^3')
f.draw(fit, 2, 'cyan')

summary(fit)

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

Variables vs. terms. Some factor levels may be non-significant alone or in interactions.
  For polynomials and interactions, when building, add best lowest degree terms first; when
  pruning, cut worst highest degree terms first.

For instance, y ~ x + x^2 allows horizontal adjustment; y ~ x^2 forces minimum to be at x == 0.
  An unwarranted assumption under most circumstances, even if not significant -- allows for a 
  more fine-tuned fit. If theory suggests at x==0, y==minimum (e.g. growth of a seed perhaps), then
  makes sense to drop x^1.

Can use parametric methods to compare two models, when one is nested within the other:

```
rm(list=ls())

dat <- mtcars[, c('mpg', 'wt', 'disp', 'gear')]
dat$gear <- factor(dat$gear, ordered=F)

fit1 <- lm(mpg ~ 1, data=dat)
fit2 <- lm(mpg ~ wt, data=dat)              ## implicitly mpg ~ 1 + hp, so fit1 nested w/i fit2
anova(fit1, fit2)
anova(fit2, fit1)
summary(fit2)                               ## anova p-value same as for t-test on coefficient

fit3 <- lm(mpg ~ wt + disp, data=dat)       ## fit2 nested w/i fit3
anova(fit2, fit3)
summary(fit3)                               ## anova p-value same as for t-test on coefficient

fit4 <- lm(mpg ~ wt * gear, data=dat)
anova(fit2, fit4)
summary(fit4)

```

AIC extends to non-nested models. Allows arbitrary model comparisons within the same family 
  (e.g. linear models vs. linear models; not linear model vs. GLM).

step function order/outcome for non-orthogonal terms depends on order of terms in formula. may
  need to permute them a bit to see what effect order has -- should stick to the more stable/frequently
  appearing model. 

AIC: -2 * log-likelihood + 2 * p
 
Parametric model selection. 

```
rm(list=ls())

par(mfrow=c(2, 3))

fit1a <- lm(mpg ~ ., data=mtcars)
summary(fit1a)
plot(fit1a, which=1:6)

fit2a <- step(fit1a)
summary(fit2a)
plot(fit2a, which=1:6)

fit1b <- lm(mpg ~ 1, data=mtcars)
summary(fit1b)
plot(fit1b, which=1:6)

fit2b <- step(fit1b, scope=list(lower=~1, upper=~.^2))
summary(fit2b)
plot(fit2b, which=1:6)

```

Evaluation procedures: CV

```
code here

```

Tuning procedures: tune k in AIC, using nested CV.

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
