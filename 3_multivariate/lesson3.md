# Fundamentals of computational data analysis using R
## Multivariate statistics: multiple regression
#### Contact: mitch.kostich@jax.org

---

### Index

- [Multiple regression](#multiple-regression)
- [Correlated predictors](#correlated-predictors)
- [Interactions](#interactions)
- [Hierarchical models](#hierarchical-models)
- [Mixed models](#mixed-models)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Multiple regression

So far we have looked at estimating population parameters based on random
  samples, where those parameters have described either the distribution of
  a single variable considered in isolation (e.g. the one-sample t-test
  and one-sample proportion test), or how the distribution of one variable 
  depends on the value of another. When two variables are independent,
  the value of one variable has no impact on the distribution of the other. 
  When they are not independent, then the distribution of one of the variables
  is conditional on the value of the other. For instance, the t-test tests 
  whether the value of a categorical variable (designating which one of the 
  two groups being compared an observation belongs to) affects some aspect
  (in particular, the mean) of the distribution of the continuous variable.
  The ANOVA extends this analysis to the case of a categorical variable with
  more than two categories. We also used the proportion test for two 
  categorical variables to see if the group proportions of one variable
  was independent of the group value of the other variable. We used
  correlation tests to see if the distribution of two continuous variables
  were linearly independent of one another. We extended the ideas of the 
  correlation test and the t-test to use linear regression to estimate the 
  conditional mean of one continuous variable based on the value of another 
  continuous variable. We saw that linear regression could also incorporate 
  categorical variables, in which case it could be shown to be equivalent to 
  the equal variances t-test and ANOVA procedures.

When fitting a simple linear regression model with a **continuous predictor** `x`, 
  we saw that two coefficients would be estimated. One was the coefficient for the 
  continuous predictor `b1`, which is equivalent to the slope of the prediction line.
  The other coefficient was the intercept `b0`, which is a constant term in the
  prediction formula that describes the value of the response `y` when `x`
  is zero. This value sets the vertical position of the prediction line. That is,
  increasing `b0` shifts the regression line upward without changing its slope,
  while decreasing `b0` shifts the regression line downward. Knowing these two 
  coefficients alone allows us to make predictions for future values:
  `y = b0 + b1 * x1`. The accuracy of predictions would depend on the accuracy
  of the assumption that the true relationship was linear, the accuracy with
  which the coefficients were estimated, and the magnitude of the error term,
  which describes the spread of data points around the prediction line.

When fitting a simple linear regression model with a **categorical predictor** `x`,
  we saw that the number of coefficients estimated for `x` would depend on how
  many categories or 'levels' `x` contains. In general, the number of coefficients
  would be the number of categories minus one, plus one coefficient for the 
  intercept. For instance, if there were two categories, say 'A' and 'B', we would 
  estimate the intercept `b0` and one group coefficient `b1`. There are many 
  possible ways of specifying the same relationship using two coefficients that 
  result in identical predictions, even though each way results in different
  values for the coefficients. The different ways of encoding the relationship
  with the coefficients is called the 'contrast' used during the regression
  (or ANOVA). One simple to interpret (and the default for unordered factor 
  categories) contrast is the **treatment contrast**. In this case, `b0` is the 
  mean of a 'reference' group (say 'A', though we can designate any group), and 
  `b1` is the difference between the mean of group 'B' and the group mean of 'A'. 
  Predictions for any observation with group membership 'A' is simply: `y = b0`; 
  while predictions for observations with group membership 'B' will be 
  `y = b0 + b1`. In the case of three groups ('A', 'B' and 'C'), we will end up 
  with an estimate of the intercept `b0`, which represents the mean for the 
  **'reference' group** (for instance group 'A', though we can specify which group), 
  and two additional coefficients, `b1` representing the difference in means between 
  group 'A' and group 'B', as well as `b2`, representing the difference between the 
  mean of group 'C' and the mean of group 'A'. Now prediction for observations 
  belonging to group 'A' are `y = b0`, predictions for group 'B' are `y =  b0 + b1` 
  and for group 'C', predictions are `y = b0 + b2`. That is, each prediction is a 
  horizontal line with an intercept that depends on the group. Therefore, the 
  prediction for each group is simply the mean of that group in the training data.

We are often dealing with response variables whose value depends on more than one 
  explanatory variable. For instance, the area of a rectangle depends on not only
  the height, but also the width. We can extend the linear model to multiple 
  explanatory variables, including combinations of continuous and categorical 
  variables. It can also include the addition of polynomial terms for explanatory
  variables. We will begin with this latter case:

```
library('caret')
library('MASS')
rm(list=ls())
set.seed(1)

summary(wtloss)

par(mfrow=c(1, 1))
plot(wtloss)

fit1 <- lm(Weight ~ Days, data=wtloss)
smry1 <- summary(fit1)
coef(smry1)
smry1$adj.r.squared
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

## residuals vs. fitted clearly suggests second order (squared)
##   but residuals look homoskedastic, so don't want to mess 
##   with Weight.

fit2 <- lm(Weight ~ I(Days ^ 2), data=wtloss)
smry2 <- summary(fit2)
coef(smry2)
smry2$adj.r.squared
par(mfrow=c(2, 3))
plot(fit2, which=1:6)

## Residuals vs Fitted plot suggests model still has wrong shape.
##   Normally, unless theory dictates otherwise, we include
##   all the lower powers of the variable as well:

fit3 <- lm(Weight ~ Days + I(Days ^ 2), data=wtloss)
smry3 <- summary(fit3)
coef(smry3)
smry3$adj.r.squared
par(mfrow=c(2, 3))
plot(fit3, which=1:6)             ## Much better!

```

Now we see our list of coefficients has grown to three, with one `b0` (we'll 
  call it that here for notational convenience, but it is actually labeled
  `(Intercept)` in the output) for the intercept, another `b1` for `Days` 
  (labeled `Days`) and a last one `b2` for `Days ^ 2` (labeled `I(Days^2)`).
  Predictions for new observations would then be made as: 
  `y = b0 + b1 * Days + b2 * (Days ^ 2)`:

```
b0 <- coef(smry3)['(Intercept)', 'Estimate']
b1 <- coef(smry3)['Days', 'Estimate']
b2 <- coef(smry3)['I(Days^2)', 'Estimate']

days <- seq(from=min(wtloss$Days), to=max(wtloss$Days), length.out=100)

y1 <- predict(fit3, newdata=data.frame(Days=days))
y2 <- b0 + b1 * days + b2 * (days ^ 2)
all(y1 == y2)

```

Let's plot the three fits to see the differences graphically:

```
## for plotting: predictions on dense grid of Days:
days <- seq(from=min(wtloss$Days), to=max(wtloss$Days), length.out=10000)
newdata <- data.frame(Days=days)
prd1 <- predict(fit1, newdata=newdata)
prd2 <- predict(fit2, newdata=newdata)
prd3 <- predict(fit3, newdata=newdata)

par(mfrow=c(1, 1))
plot(Weight ~ Days, data=wtloss)
lines(x=days, y=prd1, lty=2, col='cyan')
lines(x=days, y=prd2, lty=3, col='magenta')
lines(x=days, y=prd3, lty=4, col='orangered')
legend(
  'topright', 
  legend=c('linear', 'quadratic', 'both'), 
  col=c('cyan', 'magenta', 'orangered'),
  lty=c(2, 3, 4)
)

```

The addition of both linear and quadratic terms looks to be better than either
  alone. It is interesting that the quadratic only fit `fit3` is curved the 
  wrong way! In general, unless theory suggests the lower order polynomial 
  terms are not needed, we usually include them if any higher order term is 
  significant.

We'll now look at an example where we'll try including different variables, as
  well as polynomial terms for one of them:

```
rm(list=ls())

par(mfrow=c(1, 1))
plot(mtcars)

## break out some interesting looking variables to predict mpg:
plot(mtcars[, c('mpg', 'wt', 'disp', 'hp')])

## correlation between variables:
cor(mtcars[, c('mpg', 'wt', 'disp', 'hp')])

## let's start w/ variable w/ highest cor to mpg:
fit1 <- lm(mpg ~ wt, data=mtcars)
summary(fit1)                     ## wt coefficient significant
par(mfrow=c(2, 3))
plot(fit1, which=1:6)             ## Residuals vs Fitted suggests quadratic

## maybe add wt^2 quadratic component:
fit2 <- lm(mpg ~ wt + I(wt ^ 2), data=mtcars)
summary(fit2)                     ## adj.r.squared improves, wt^2 coef significant
par(mfrow=c(2, 3))
plot(fit2, which=1:6)             ## residuals much better, though not normal and influence so so

fit3 <- lm(mpg ~ wt + I(wt ^ 2) + disp, data=mtcars)
summary(fit3)                     ## slightly better adj.r.squared, but disp barely signif
par(mfrow=c(2, 3))
plot(fit3, which=1:6)             ## residuals about the same as for fit2

fit4 <- lm(mpg ~ wt + I(wt ^ 2) + hp, data=mtcars)
summary(fit4)                     ## bigger jump in adj.r.squared; hp coef significant
par(mfrow=c(2, 3))
plot(fit4, which=1:6)             ## plot still looks ok

fit5 <- lm(mpg ~ wt + I(wt ^ 2) + hp + disp, data=mtcars)
summary(fit5)                     ## disp coef still not significant; fit4 is better
par(mfrow=c(2, 3))
plot(fit5, which=1:6)             ## plot still looks ok

```

Interestingly, even though `disp` has a higher correlation with `mpg` than `hp`, the latter 
  appears to be a more useful predictor. We will return to this phenomena shortly, but
  first, since residual plots seem weird, let's see if CV supports our choice of `fit3`: 

```
library(caret)
set.seed(1)

dat <- mtcars[, c('mpg', 'wt', 'disp', 'hp')]

f.mse <- function(frm, dat.trn, dat.tst) {
  fit <- lm(frm, data=dat.trn)
  prd <- predict(fit, newdata=dat.tst)
  mse <- mean((dat.tst$mpg - prd) ^ 2, na.rm=T)
}

f <- function(idx.trn) {
  dat.trn <- mtcars[idx.trn, ]
  dat.tst <- mtcars[-idx.trn, ]
  mse0 <- f.mse(mpg ~ 1, dat.trn, dat.tst)
  mse1 <- f.mse(mpg ~ wt, dat.trn, dat.tst)
  mse2 <- f.mse(mpg ~ wt + I(wt ^ 2), dat.trn, dat.tst)
  mse3 <- f.mse(mpg ~ wt + I(wt ^ 2) + disp, dat.trn, dat.tst)
  mse4 <- f.mse(mpg ~ wt + I(wt ^ 2) + hp, dat.trn, dat.tst)
  mse5 <- f.mse(mpg ~ wt + I(wt ^ 2) + hp + disp, dat.trn, dat.tst)
  c(mse0=mse0, mse1=mse1, mse2=mse2, mse3=mse3, mse4=mse4, mse=mse5)
}

k <- 10
times <- 7
idx <- 1:nrow(dat)

## make list of k * times folds; elements are training-set indices:
folds <- createMultiFolds(idx, k=k, times=times)

rslt <- sapply(folds, f)          ## run f() on each training-set index in folds
apply(rslt, 1, mean)              ## best: mpg ~ wt + I(wt ^ 2) + hp
apply(rslt, 1, sd)                ## best: mpg ~ wt + I(wt ^ 2) + hp; but SEs are large!

```

Finally, we'll look at an example of using both a continuous and categorical 
  predictor. In this case, the number of coefficients estimated for the categorical
  vector will be the number of categories minus one. The 'reference' category, will
  by default be the first categorical label alphabetically, but this can be changed by
  the user. In this case, the intercept term represents the reference category and
  all the other category coefficients represent constant vertical (along the response 
  variable axis) displacements of the prediction line for each group. That is, the
  slope of the response vs the continuous variable remains constant, but the intercept
  becomes group-specific, so there is one line per group, with all lines parallel, having
  the same slope, but vertically displaced by the group coefficient, so each group 
  effectively ends up with a different intercept.

```
rm(list=ls())

dat <- iris[iris$Species %in% c('virginica', 'setosa'), ]
fit1 <- lm(Sepal.Length ~ Sepal.Width + Species, data=dat)
summary(fit1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

## intercepts (Sepal.Length where Sepal.Width==0) differ, 
##   but slope is the same (prediction lines parallel):

coefs <- coef(fit1)
predict(fit1, newdata=data.frame(Sepal.Width=0, Species='setosa'))
coefs['(Intercept)']

predict(fit1, newdata=data.frame(Sepal.Width=0, Species='virginica'))
coefs['(Intercept)'] + coefs['Speciesvirginica']

## let's compare the prediction lines graphically:

w <- seq(from=min(dat$Sepal.Width), to=max(dat$Sepal.Width), length.out=10000)
pred1 <- predict(fit1, newdata=data.frame(Sepal.Width=w, Species='virginica'))
pred2 <- predict(fit1, newdata=data.frame(Sepal.Width=w, Species='setosa'))

par(mfrow=c(1, 1))
plot(x=range(w), y=range(c(pred1, pred2)), type='n')
i.v <- dat$Species == 'virginica'
i.s <- dat$Species == 'setosa'
points(Sepal.Length ~ Sepal.Width, data=dat[i.v, ], pch='+', col='cyan')
points(Sepal.Length ~ Sepal.Width, data=dat[i.s, ], pch='o', col='magenta')
lines(x=w, y=pred1, lty=2, col='cyan')
lines(x=w, y=pred2, lty=3, col='magenta')

```

[Return to index](#index)

---

### Check your understanding 1

1) Use 10-fold cross-validation repeated 7 times to see which of the following formula best fit
     the `MASS::wtloss` training set: `Weight ~ Days`, `Weight ~ I(Days ^ 2)`, or 
     `Weight ~ Days + I(Days ^ 2)`? Examine both the mean prediction error as well as its spread.

2) How does the improvement in performance offered by the best fit compare to the standard 
     deviation of the performance of the best fit?

[Return to index](#index)

---

### Correlated predictors

When we include multiple explanatory variables in a conventional linear
  regression, correlations between the explanatory variables can cause problems
  with the calculation of the model fit. You will typically see that the
  inclusion of correlated variables tends to increase the standard errors of
  coefficient estimates, sometimes to the point of making otherwise very
  significant coefficients non-significant. In addition, the correlation means
  the information in each variable is at least partially redundant, and therefore 
  the effects of both variables on the response are not easy to tease apart: it 
  is hard to know how much of the change in `y` to assign to changes in `x1` 
  instead of `x2` when `x1` and `x2` are highly correlated, because every time 
  `x1` changes, `x2` is changing in a related pattern. This can lead to nearly 
  all the 'responsibility' for changes in `y` being assigned to `x1` and nearly 
  none to `x2`, or vice-versa, which means that the coefficient estimates can be 
  quite far from their true population values (in this case, the `x1` 
  coefficient will be inflated and the `x2` coefficient deflated). Or the 
  two coefficients can both end up being inflated in a way that they offset 
  each other. That is, if `x1` is approximately the same as `x2`, then the
  predictions made by the formula `y = x1 + x2` will be approximatly the same
  as `y = 3 * x1 - x2` or `y = 4 * x1 - 2 * x2`,  or even 
  `y = 300 * x1 - 298 * x2`, etc. That is, very similar predictions can be 
  produced with wildly different coefficients, which makes the fitting process 
  extremely unstable (partially reflected in the standard errors rising). We 
  will discuss methods for dealing with highly correlated predictors in the 
  machine learning section of this course.

We saw some hint of this going on in the previous example. When we included 
  the `I(wt ^ 2)` term, which is correlated with `wt` itself, it increased the
  standard error for the `wt` coefficient estimate, decreasing the corresponding
  t-statistic and coefficient significance. Although `hp` had a lower pairwise
  correlation with `mpg` than `disp`, it was a more useful predictor, because
  it added more 'new' information to the fit, since `hp` had a lower correlation
  with `wt` than did `disp`. That is, much of the information `disp` had to 
  offer was already included with the inclusion of the `wt` variable.

```
rm(list=ls())

## some fits we've seen:
fit1 <- lm(mpg ~ wt, data=mtcars)
fit2 <- lm(mpg ~ wt + I(wt ^ 2), data=mtcars)
fit3 <- lm(mpg ~ wt + I(wt ^ 2) + disp, data=mtcars)
fit4 <- lm(mpg ~ wt + I(wt ^ 2) + hp, data=mtcars)
fit5 <- lm(mpg ~ wt + I(wt ^ 2) + hp + disp, data=mtcars)

## correlation between variables:
dat <- mtcars[, c('wt', 'disp', 'hp')]
dat$wt2 <- dat$wt ^ 2
dat <- dat[, c('wt', 'wt2', 'disp', 'hp')]
cor(dat)

coef(summary(fit1))
coef(summary(fit2))
coef(summary(fit3))
coef(summary(fit4))
coef(summary(fit5))

```

We can use a synthetic example to demonstrate this more dramatically. In the first
  case we will use an example with two uncorrelated predictors, and we will see that
  inclusion of the second predictor greatly improves the standard error and 
  significance of the first predictor coefficient. This is because inclusion of the 
  second predictor adds unique explanatory value that improves prediction and thereby 
  decreases the deviations of the observations from the conditional mean, resulting
  in a reduction in the magnitude of the estimated error term for the model. Decreasing
  the magnitude of the error term reduces the standard error for all individual 
  coefficients. In the second example, we use two highly correlated predictors and 
  show how adding the second predictor to the model actually can make the model worse:

```
rm(list=ls())
set.seed(1)

x1 <- runif(100, 0, 1)
x2 <- runif(100, 0, 1)
e <- rnorm(100, 0, 0.1)
y <- x1 + x2 + e
cor(cbind(y, x1, x2, e))          ## x1, x2 both correlated w/ y, but not each other

fit1 <- lm(y ~ x1)
fit2 <- lm(y ~ x2)
fit3 <- lm(y ~ x1 + x2) 
coef(summary(fit1))               ## coefficient estimate about right
coef(summary(fit2))               ## coefficient estimate about right
coef(summary(fit3))               ## improved coef estimates, std errors, and p-values!!!

x1 <- runif(100, 0, 1)
x2 <- 0.99 * x1 + 0.01 * runif(100, 0, 1)
y <- x1 + x2 + e
cor(cbind(y, x1, x2, y))          ## x1 and x2 highly correlated; info redundant!

fit1 <- lm(y ~ x1)
fit2 <- lm(y ~ x2)
fit3 <- lm(y ~ x1 + x2)
coef(summary(fit1))               ## coefficient estimate way off; 
coef(summary(fit2))               ## coefficient estimate way off; 
coef(summary(fit3))               ## standard errors, t-values, p-values much worse

```

[Return to index](#index)

---

### Check your understanding 2

1) Peform the following variable generation pattern and two fits 1000 times. Return the 
     standard deviation of the estimates of the coefficient for `x1` separately for `fit1` 
     and for `fit2`:

```
rm(list=ls())

e <- rnorm(100, mean=0, sd=0.1)   ## error
x1 <- runif(100, min=0, max=1)    ## predictor 1
x2 <- runif(100, min=0, max=1)    ## uncorrelated second predictor

## correlated third predictor:
x3 <- 0.999 * x1 + 0.001 * runif(100, min=0, max=1)

## rescale to ensure w/i interval [0, 1]:
x3 <- (x3 - min(x3)) / max(x3)

y1 <- x1 + x2 + e
y2 <- x1 + x3 + e
fit1 <- lm(y1 ~ x1 + x2)
fit2 <- lm(y2 ~ x1 + x3)

```

[Return to index](#index)

---

### Interactions

So far, we have looked at adding polynomial terms for a single explanatory
  variable, as well as adding additional continuous or categorical 
  variables to a linear regression. However, we have only looked at how to 
  include terms in an additive way. That is, we considered cases where
  the contribution of a term to the net effect was additive relative to
  other terms in the model. Therefore, the effect of a variable depended
  only on the value of that variable, but not any other variable. But 
  sometimes we want to model a multiplicative relationship between variables, 
  or a relationship where the value of a categorical variable changes not 
  only the intercept, but also the slope of the response vs. the continuous
  variable. 

We mentioned earlier in the course that we can convert multiplicative 
  relationships into additive ones by taking the log of the multiplicative 
  formula. However, this can result in a non-linear relationship in some 
  cases, or problems with the residual distribution, or simply may make 
  the relationship harder to interpret because of the tranformation. In 
  cases of a multiplicative relationship between explanatory variables, 
  we can introduce 'interactions' between those variables. In the case of 
  two continuous variables, this essentially amounts to including the 
  product of the two variables as a new variable in the regression formula. 

Here we will use the synthetic example of rectangle areas, where the area
  is the product of the height and width of the rectangle. We know this
  formula from primary school, so we could pick the right formula a priori,
  but if we were looking at some new data where the underlying phenomena
  and relationships were not understood, we might have to arrive at the right
  formula through some trial and error. In general, we would want to do
  this with some pilot data, rather than with the main experiment we would 
  like to ultimately analyze. Once the choice of model is made based on the
  pilot data, we should still with that model for the primary analysis of 
  the main experiment: hunting for a good fitting model during the main 
  experiment drastically increases the likelihood that we will find a model 
  that fits our sample well based on chance, but actually fits the underlying
  population poorly, leading to poor reproducibility of our conclusions.
  That is, you are likely to find models that fit the noise in the sample 
  more than the signal. In formulas and coefficient naming, the `:` symbol 
  between variables indicates an interaction term. In formulas, the `*` 
  operator, when not protected by the `I()` function, serves to include both 
  terms plus their interaction. That is `y ~ x1 * x2` is equivalent to 
  `y ~ x1 + x2 + x1:x2`, where `x1:x2` represents the interaction. However, 
  the number of possible interactions grows very rapidly with the number of 
  variables. For instance, `y ~ x1 * x2 * x3` is equivalent to 
  `y ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + x1:x2:x3`. With many variables, 
  this can result in trying to estimate too many coefficients for the available
  data, decreasing the stability/reliability of all the estimates. In order 
  to limit or otherwise specify the maximal degree of interactions to include,
  we can use the `^` formula operator. For instance, `y ~ (x1 + x2 + x3) ^ 2` 
  is equivalent to `y ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3`. The three-way 
  interaction `x1:x2:x3` was excluded because it is of degree three, while
  the `^ 2` notation indicated to only include interactions of degree two or 
  less. 

One can also nest variables using `y ~ x1 / x2`, where `x2` is a categorical 
  variable, which results in a different `y ~ x1` slope for each category in 
  `x2` without introducing an additive effect for `x2`. This makes sense when 
  the meaning of `x1` in some sense differs between the different groups 
  specified by `x2`. For instance, some schools might grade their Geometry 
  class based on a curve, with the average being a `85`, others using a curve 
  with an average of `75` while others report unadjusted scores. Therefore, 
  the grade may only make sense within the context of the school in which the 
  grade was assigned. So we can have a formula like: `days_absent ~ grade / school`, 
  which would generate a different `days_absent` vs `grade` slope for each 
  `school`, but all the schools have the same intercept (so `school` does not 
  have an effect except to qualify the impact of `grade`).

```
rm(list=ls())
set.seed(1)

h <- runif(100, min=1, max=10)    ## heights
w <- runif(100, min=1, max=10)    ## widths
e <- rnorm(100, 0, 0.2)           ## 'measurement error'
a <- h * w + e                    ## measurements of rectangle areas
dat <- data.frame(area=a, h=h, w=w)

fit1 <- lm(area ~ h + w, data=dat)
summary(fit1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)             ## strong quadratic component

fit2 <- lm(area ~ h + w + I(h^2) + I(w^2), data=dat)
summary(fit2)                     ## quadratic terms non-significant
plot(fit2, which=1:6)             ## did not fix the issue w/ fit1

## we know this works; note only 2 coeficients
fit3 <- lm(area ~ I(h * w), data=dat)
summary(fit3)                     ## only intrcpt + I(h*w) coefs
plot(fit3, which=1:6)             ## pretty good

## adds h, w and interaction between h and w (4 coefficients total):
fit4 <- lm(area ~ h * w, data=dat)
summary(fit4)                     ## intrcpt, h, w and h:w interaction
plot(fit4, which=1:6)             ## pretty good

## add all 2-way interactions (since only h and w, same as fit4):
fit5 <- lm(area ~ . ^ 2, data=dat)
summary(fit5)
plot(fit5, which=1:6)

## explicitly include interaction term (same as fit4):
fit6 <- lm(area ~ h + w + h:w, data=dat)
summary(fit6)
plot(fit6, which=1:6)

```

An interaction between **two continuous variables** is modeled as the product
  of the two variables times the coefficient for the interaction term. 
  Therefore, predictions for `fit4` proceed from the coefficients as 
  follows:

```
smry <- summary(fit4)
(coefs <- coef(smry))
b0 <- coefs['(Intercept)', 'Estimate']
b1 <- coefs['h', 'Estimate']
b2 <- coefs['w', 'Estimate']
b12 <- coefs['h:w', 'Estimate']

pred1 <- b0 + b1 * h + b2 * w + b12 * h * w
pred2 <- predict(fit4, newdata=dat)
names(pred2) <- NULL
all.equal(pred1, pred2)

```

An interaction between **categorical and continuous** variables results in the
  slope of the response vs. the continuous variable changing from group
  to group. If the categorical variables are also included as individual terms
  (they normally would be, if the interaction is included and theory does not
  provide justification for removing the individual terms), the intercepts
  will also be different in different groups. In the following example, we will
  look at how the slope of `Sepal.Length` vs `Sepal.Width` varies between two
  iris species:

```
rm(list=ls())

dat <- iris[iris$Species %in% c('virginica', 'setosa'), ]
fit1 <- lm(Sepal.Length ~ Sepal.Width * Species, data=dat)
summary(fit1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

w <- seq(from=min(dat$Sepal.Width), to=max(dat$Sepal.Width), length.out=10000)
pred1 <- predict(fit1, newdata=data.frame(Sepal.Width=w, Species='virginica'))
pred2 <- predict(fit1, newdata=data.frame(Sepal.Width=w, Species='setosa'))

par(mfrow=c(1, 1))
plot(x=range(w), y=range(c(pred1, pred2)), type='n')
i.v <- dat$Species == 'virginica'
i.s <- dat$Species == 'setosa'
points(Sepal.Length ~ Sepal.Width, data=dat[i.v, ], pch='+', col='cyan')
points(Sepal.Length ~ Sepal.Width, data=dat[i.s, ], pch='o', col='magenta')
lines(x=w, y=pred1, lty=2, col='cyan')
lines(x=w, y=pred2, lty=3, col='magenta')

## notice lines are more spread out at the end of the range than at the
##   beginning, reflecting the differing slopes (lines are not parallel!!!):

pred1[1] - pred2[1]
pred1[length(pred1)] - pred2[length(pred2)]

```

Now we will take a look at an example in the `mtcars` dataset with a smaller 
  but still apparently significant interaction, this time between numeric 
  variable `wt` (weight of the vehicle) and the categorical variable `gear`,
  which specifies the number of gears the vehicle has. Since we have no idea
  if the response variable `mpg` (miles-per-gallon) should have a monotonic,
  let alone linear, relationship with `gear`, and since there are only three 
  distinct values for `gear`, we will model it as a categorical variable. 
  This will allow the effect of gear to follow an arbitrary pattern, rather
  than assuming linarity or even monotonicity.

```
rm(list=ls())

## get the data together:
dat <- mtcars[, c('mpg', 'wt', 'gear')]
table(dat$gear)
dat$gear <- factor(dat$gear, ordered=F)
summary(dat)
par(mfrow=c(1, 1))
plot(dat)

fit1 <- lm(mpg ~ wt, data=dat)
summary(fit1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)             ## maybe quadratic?

## lets try adding other variables:
fit2 <- lm(mpg ~ wt + gear, data=dat)
summary(fit2)
par(mfrow=c(2, 3))
plot(fit2, which=1:6)             ## similar residuals vs fitted as fit1

## allow for an interaction:
fit3 <- lm(mpg ~ wt * gear, data=dat)
summary(fit3)
par(mfrow=c(2, 3))
plot(fit3, which=1:6)             ## residuals look better 

## here is an example of a nested relationship: gear has no direct
##   effect, but only an effect on the slope of mpg vs wt:
fit4 <- lm(mpg ~ wt / gear, data=dat)
summary(fit4)
par(mfrow=c(2, 3))
plot(fit4, which=1:6)

```
In the above example, the coefficient for `wt` captures the slope of the 
  relationship between `mpg` and `wt` for the 'reference' group, which was
  selected by default to be the observations where `dat$gear == '3'`. For
  this group, the intercept and this slope are all that is required for 
  making predictions. For observations where `dat$gear == '4'`, we add
  a constant additive effect, corresponding to `coefs['gear4', 'Estimate']`,
  plus an interaction effect between `gear` and `wt`, which has the net 
  effect of changing the slope of `mpg` vs `wt` for the `gear4` group.
  A constant additive effect and effect on slope of `mpg` vs `wt` are also
  modeled for the `gear5` group. 

This case allows us to demonstrate how predictions are made when there are
  interactions between a continuous and categorical variable:

```
prd2 <- predict(fit3, data=dat)

coefs <- coef(summary(fit3))
b0 <- coefs['(Intercept)', 'Estimate']
b1 <- coefs['wt', 'Estimate']
b24 <- coefs['gear4', 'Estimate']
b25 <- coefs['gear5', 'Estimate']
b34 <- coefs['wt:gear4', 'Estimate']
b35 <- coefs['wt:gear5', 'Estimate']

wt <- dat$wt
gr <- dat$gear

i4 <- gr == '4'                    ## only true for gear4
i5 <- gr == '5'                    ## only true for gear5

effect.wt <- b1 * wt

## constant additive effects of gear:
##   0 for gear3; b24 for gear4; b25 for gear5:
effect.gear <- i4 * b24 + i5 * b25

## interaction between gear and wt; equiv to a change in slope;
##   0 for gear3; (b34 * wt) for gear4; (b35 * wt) for gear5
effect.inter <- i4 * b34 * wt + i5 * b35 * wt

prd1 <- b0 + effect.wt + effect.gear + effect.inter
prd2 <- predict(fit3, data=dat)
names(prd2) <- NULL
all.equal(prd1, prd2)

```

Now we will look at a dataset with a possible interaction between two categorical 
  variables. In this case, a significant interaction coefficient would suggest that
  the contribution of each categorical variable is conditional on the value
  of the other variable. Therefore, the effects of the variables involved in the
  interaction are modeled as constants whose value depends on the other variable, 
  rather than as constants whose values only depend on the first variable.

```
rm(list=ls())
library('caret')

## subset some data:
dat <- diamonds
dat <- as.data.frame(dat)
dat <- dat[dat$carat == 1, c('price', 'color', 'clarity')] 
dat <- dat[dat$color %in% c('E', 'G'), ]
dat <- dat[dat$clarity %in% c('SI1', 'VS2'), ]

## make unordered, so default contrasts are 'treatment':
dat$color <- factor(dat$color, ordered=F)
dat$clarity <- factor(dat$clarity, ordered=F)

summary(dat)
par(mfrow=c(1, 1))
plot(dat)
nrow(dat)

fit1 <- lm(price ~ color * clarity, data=dat)
summary(fit1)
par(mfrow=c(2, 3))
plot(fit1, which=1:6)

## how coefficients become predictions:

coefs <- coef(fit1)
b0 <- coefs['(Intercept)']
b1 <- coefs['colorG']
b2 <- coefs['clarityVS2']
b3 <- coefs['colorG:clarityVS2']

i.G <- dat$color == 'G'
i.VS2 <- dat$clarity == 'VS2'

table(i.G)
table(i.VS2)
summary(dat)

prd1 <- b0 + i.G * b1 + i.VS2 * b2 + (i.G & i.VS2) * b3
prd2 <- predict(fit1, newdata=dat)
names(prd2) <- NULL
all.equal(prd1, prd2)

```

[Return to index](#index)

---

### Check your understanding 3

Using the following dataset:

```
dat <- mtcars[, c('mpg', 'wt', 'gear')]
table(dat$gear)
dat$gear <- factor(dat$gear, ordered=F)
summary(dat)

```

1) Plot the variables against one another.

2) Fit the linear model with formula `mpg ~ wt * gear` to the data. Are the individual variable
     coefficients significantly different from zero? Are the interaction coefficients significantly
     different from zero?

3) Plot separate regression lines for three values of `gear`.

[Return to index](#index)

---

## FIN!
