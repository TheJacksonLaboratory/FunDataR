# Fundamentals of computational data analysis using R
## Multivariate statistics: multiple regression
#### Contact: mitch.kostich@jax.org

---

### Index

- [Multiple regression](#multiple-regression)
- [Correlated predictors](#correlated-predictors)
- [Interactions](#interactions)
- [Hierarchical models](#hierarchical-models)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Multiple regression

intro here

```
rm(list=ls())
summary(wtloss)

par(mfrow=c(1, 1))
plot(wtloss)

fit1 <- lm(Weight ~ Days, data=wtloss)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

## residuals vs. fitted clearly suggests second order (squared)
##   but residuals look homoskedastic, so don't want to mess 
##   with Weight.

fit2 <- lm(Weight ~ I(Days ^ 2), data=wtloss)
par(mfrow=c(2, 3))
plot(fit2, which=1:6)

fit3 <- lm(Weight ~ Days + I(Days ^ 2), data=wtloss)
par(mfrow=c(2, 3))
plot(fit3, which=1:6)

rm(list=ls())
summary(steam)

par(mfrow=c(1, 1))
plot(steam)

fit1 <- lm(Press ~ Temp, data=steam)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

## residuals vs. fitted clearly suggests second order (squared)
##   but residuals look homoskedastic, so don't want to mess 
##   with Weight.

fit1 <- lm(Press ~ I(Temp ^ 2), data=steam)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)


```

[Return to index](#index)

---

### Correlated predictors

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

### Interactions

intro here

```
code here

```

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Hierarchical models

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
