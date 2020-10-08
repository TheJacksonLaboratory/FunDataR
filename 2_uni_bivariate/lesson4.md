# Fundamentals of computational data analysis using R
## Univariate and bivariate statistics: linear regression
#### Contact: mitch.kostich@jax.org

---

### Index

- [The linear model](#the-linear-model)
- [Equivalence to t-test and ANOVA](#equivalence-to-t-test-and-ANOVA)
- [Analysis of residuals](#analysis-of-residuals)
- [Prediction](#prediction)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### The linear model

In the previous lesson we saw how when two variables are correlated,
  if you know something about the values of one variable, it tells
  you something about the values of the other variable. For instance,
  if two variables `x` and `y` are positively correlated, if `x` goes
  up, you know that chances of `y` going up are increased. But we 
  don't really have a way of making a more precise prediction about 
  the behavior of `y`. Previously we also learned that if we know 
  the population mean for `y` and are asked to predict the `y` value 
  of a single observation randomly drawn from that population, the 
  population mean would be our best predictor, in the least-squares
  sense. That is, the mean has a lower **average** squared deviation 
  (difference) to to the `y` values of the population members than 
  any other value.

The model:

y = m * x + b
y = b1 * x + b0
y = b0 + b1 * x

y = b0 + b1 * x1 + b2 * x2 + b3 * x3 + ...

y.i = b0 + b1 * x.i + e.i
e.i ~ N(0, s)

With e.i independent (reflects random sampling) and come from the 
  same normal distribution N(0, s).

```
rm(list=ls())
set.seed(1)

n <- 100
b0 <- 5                           ## intercept (where line hits vertical y-axis)
b1 <- 3                           ## slope

x <- runif(n, -10, 10)

e1 <- rnorm(n, 0, 1)
e2 <- rnorm(n, 0, 2)
e4 <- rnorm(n, 0, 4)
e8 <- rnorm(n, 0, 8)

y1 <- b0 + b1 * x + e1
y2 <- b0 + b1 * x + e2
y4 <- b0 + b1 * x + e4
y8 <- b0 + b1 * x + e8

par(mfrow=c(2, 2))

plot(x=x, y=y1, main="s == 1")
abline(a=b0, b=b1, col='cyan', lty=2)

plot(x=x, y=y2, main="s == 2")
abline(a=b0, b=b1, col='cyan', lty=2)

plot(x=x, y=y4, main="s == 4")
abline(a=b0, b=b1, col='cyan', lty=2)

plot(x=x, y=y8, main="s == 8")
abline(a=b0, b=b1, col='cyan', lty=2)

par(mfrow=c(1, 1))

```

Now fit first model and see what it yields. 

```
(fit1 <- lm(y1 ~ x))
class(fit1)
is.list(fit1)
names(fit1)
attributes(fit1)
str(fit1)

coef(fit1)
fitted(fit1)
residuals(fit1)
fit1$df.residual
fit1$call

```

But typically work with the summary, which adds confidence intervals and p-values.
  The confidence intervals are for the model estimates of the two coefficients 
  `b0` and `b1`. The null hypothesis for all the coefficients is that they are 
  zero. For `b0`, this means the hypothesis that the line passes through the 
  origin instead of hitting 

```
(smry1 <- summary(fit1))
class(smry1)
is.list(smry1)
names(smry1)
attributes(smry1)
str(smry1)

(coefs <- coef(smry1))            ## much more detail than coef(fit1)
class(coefs)

all(residuals(smry1) == residuals(fit1))
## no 'fitted(smry1)'

smry1$adj.r.squared
smry1$fstatistic

```

Let's see how changing the error distribution changes results:

```
fit2 <- lm(y2 ~ x)
fit4 <- lm(y4 ~ x)
fit8 <- lm(y8 ~ x)

smry2 <- summary(fit2)
smry4 <- summary(fit4)
smry8 <- summary(fit8)

smry1$fstatistic
smry2$fstatistic
smry4$fstatistic
smry8$fstatistic

smry1$adj.r.squared
smry2$adj.r.squared
smry4$adj.r.squared
smry8$adj.r.squared

coef(smry1)
coef(smry2)
coef(smry4)
coef(smry8)

```

[Return to index](#index)

---

### Equivalence to t-test and ANOVA

intro here

```
code here

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Analysis of residuals

intro here

```
code here

```

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Prediction

Text here

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
