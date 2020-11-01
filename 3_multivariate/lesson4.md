# Fundamentals of computational data analysis using R
## Multivariate statistics: model selection
#### Contact: mitch.kostich@jax.org

---

### Index

- [Multiple testing](#multiple-testing)
- [Overfitting](#overfitting)
- [Model selection](#model-selection)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Multiple testing

We often test many hypotheses using a single dataset. For instance, when constructing a 
  multiple regression model, we saw that the summary contains the results of an F-test
  of the overall model, as well as results of t-tests on individual model coefficients.
  If the F-test is significant, it suggests the model fits the data better than can be 
  expected by chance, or, equivalently, that at least one of the coefficients (not 
  counting the intercept) really is not zero. In the case of simple linear regression 
  (only one explanatory variable), this F-test is equivalent to the t-test on the 
  coefficient on the explanatory variable in the sense that the two tests should return 
  identical p-values. In the case of multiple regression, the t-tests must be considered
  individually. As in the case of post-hoc testing following a significant omnibus
  F-test on an ANOVA model, this testing should control for the effects of multiple
  testing. For instance, in the example below, there are six separate t-tests being 
  conducted, and the p-values returned are not adjusted to reflect that multiple tests 
  are being performed using the same dataset:

```
## A model demonstrating overall model F-test and multiple coefficient t-tests:

rm(list=ls())

dat <- mtcars

## make gear an unordered category:
dat$gear <- factor(dat$gear, ordered=F)

## this is equivalent to mpg ~ wt + gear + gear:wt
fit <- lm(mpg ~ wt * gear, data=dat)

## note there are six coefficients being tested:
summary(fit)

```

Whenever you are performing simultaneous hypotheses tests based on a common dataset you should 
  properly adjust the p-values returned to account for the effects of multiple testing. Multiple 
  testing issues arise from the fact that when the null hypothesis is true, p-values will be 
  uniformly distributed on the interval between zero and one. That is, a p-value of `0.5` is just 
  as likely as a p-value of `0.99` or a p-value of `0.0001`. Therefore, in any one test where the 
  null hypothesis is true, the p-value itself is randomly distributed, with a 95% chance of 
  landing at or above the value `0.05` (since 95% of the space between `0` and `1` is occupied by the 
  region between `0.05` and `1`) and a 5% chance of landing below `0.05` (since only 5% of the space 
  between `0` and `1` lies below `0.05`). That is, even if all the null hypotheses are true, on 
  average, one in twenty tests will reject the corresonding null hypothesis with a p-value 
  less than `0.05`. For instance, in the example below, we will draw two 
  samples (`x` and `y`) from the same population (a normal distribution with `mean=0` and 
  `sd=1`) and conduct a t-test on the null hypothesis that the two samples were drawn from
  populations with the same mean. Since we drew both samples from the same population, 
  the null hypothesis is always true, and significant test results (very close to the 
  expected proportion of `0.05`) are false positives reflecting the random distribution of 
  p-values under the null hypothesis: 

```
## Under the null hypothesis, p-values are uniformly distributed:

rm(list=ls())

set.seed(3)

R <- 50000
p.values <- rep(as.numeric(NA), R)

for(i in 1:R) {
  x <- rnorm(10, mean=0, sd=1)
  y <- rnorm(10, mean=0, sd=1)
  p.values[i] <- t.test(x, y, var.equal=F)$p.value
}

par(mfrow=c(1, 1))
hist(p.values)
summary(p.values)
sum(p.values < 0.05) / R
probs <- c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
round(quantile(p.values, probs=probs), 3)

```

In the broadest sense, there are two approaches to controlling for the
  effects of multiple testing by 'adjusting' p-values. One approach attempts 
  to control the **family-wise error rate** or **FWER**, while the other 
  approach attempts to control the **false discovery rate** or **FDR**. These 
  approaches differ not only in the algorithms employed, but more importantly, 
  in the interpretation of the results of the corresponding adjustment. When
  adjusting for `m` tests, FWER control involves adjusting p-values so they 
  have the following interpretation: the adjusted p-value represents the 
  probability that ANY of the `m` tests will reject a null hypothesis by 
  chance. By contrast, FDR adjustment results in p-values representing the
  probability that ANY ONE of the `m` tests will reject a null hypothesis
  by chance. In the case that 1000 tests were being conducted simultaneously
  (in the case of differential expression analysis of whole transcriptome 
  data, there will often be 10s of thousands of simultaneous tests, and 
  in genome-wide association studies, the number of hypotheses can reach into
  the millions), an FWER cutoff of 0.05 means there is a 5% chance that one or
  more positive results in the set of positive results will turn out to be 
  wrong. By contrast, an FDR cutoff of 0.05 means that 5% of all the returned 
  positive results are expected to be wrong by chance. Therefore, the FWER
  is the right choice when each individual hypothesis test is considered 
  equally critical, while the FDR is a better choice in cases where you want
  to reduce the false positive rate, but are still willing to accept a small
  proportion of false posivites. The advantage of using FDR is that the 
  tests retain more power: that is they can detect smaller deviations from
  the null hypothesis for a given sample size and noise level.

Both approaches have several algorithms associated with them. Two common 
  algorithms for FWER control are the **Bonferroni** and the **Holm-Bonferroni** 
  adjustments. The Bonferroni adjustment is particularly easy to perform
  and understand: one simply multiplies each p-value by the number of tests
  being performed, and resets all values greater than `1` back to `1` (to
  keep everything within the interval from zero to one). That is:
  `p.adjust.bonferroni <- max(p.values * number.of.tests, 1)`. However, in 
  practice, it is better to use the closely related, but slightly more complicated,
  Holm-Bonferroni procedure for FWER control, as it will always be at least as 
  powerful, but is often more powerful as the original Bonferroni procedure.

Two popular methods for achieving FDR control are the **Benjamini-Hochberg**
  or **BH** method, and the **Benjamini-Yekutieli** or **BY** method. The
  BH method is preferred when the tests are independent of each other, since 
  it retains more power under those circumstances. However, when tests are
  not independent (for instance, in an RNA-seq experiment, genes often move
  up and down together, introducing correlations between the corresponding
  test results; or in regression, when variables are correlated, their
  coefficient t-tests are also correlated), the BH method may fail to provide
  the nominal level of control (e.g. more than 5% of the positive tests may
  turn out to be wrong), so the BY method should be preferred, since it controls
  the FDR at the nominal rate even when tests are positively or negatively
  correlated.

In the following example, we will take the set of uniformly distributed 
  p-values from the last example. In that example, where the null hypothesis was 
  always true, not surprisingly, we found about 5% of the p-values falling below 
  0.05. Let's see what the distributions of ajusted p-values are after the four
  adjustments described above. We'll use the R `p.adjust()` function to perform 
  each procedure. When all the null hypotheses are true, as in this case, the 
  four procedures tend to produce fairly comparable results:

```
## let's adjust the p-values from tests where null hypothesis was always true:

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

Now we'll look at an example where the null hypothesis is not true in about 10% of 
  the similuted experiments. The differences between procedures are more evident in 
  this case:

```
## let's adjust the p-values from tests where null hypothesis is sometimes false:

rm(list=ls())
set.seed(1)

R <- 10000
p.pos <- 0.1
p.values <- rep(as.numeric(NA), R)
i.diff <- rep(as.logical(NA), R)

for(i in 1 : R) {
  val <- rbinom(n=1, size=1, prob=p.pos)
  if(val == 1) { 
    mu = 1 
    i.diff[i] <- T
  } else {
    mu = 0
    i.diff[i] <- F
  }

  x <- rnorm(10, mean=mu, sd=0.5)
  y <- rnorm(10, mean=0, sd=0.5)

  p.values[i] <- t.test(x, y, var.equal=T)$p.value
}

p.bonf <- p.adjust(p.values, method='bonferroni')
p.holm <- p.adjust(p.values, method='holm')
p.bh <- p.adjust(p.values, method='BH')
p.by <- p.adjust(p.values, method='BY')

f1 <- function(p) {
  nfp <- sum(p[!i.diff] <= 0.05)
  nfn <- sum(p[i.diff] > 0.05)
  c(nfp=nfp, nfn=nfn)

}

f2 <- function(p) {
  fpf <- sum(p[!i.diff] <= 0.05) / sum(!i.diff)
  fnf <- sum(p[i.diff] > 0.05) / sum(i.diff)
  c(fpf=fpf, fnf=fnf)
}

p <- data.frame(p.values, p.bonf, p.holm, p.bh, p.by)
sapply(p, f1)
sapply(p, f2)

sum(i.diff)
sum(!i.diff)

par(mfrow=c(2, 2))
hist(p.bonf, main='Bonferroni (FWER)')
hist(p.holm, main='Holm (FWER)')
hist(p.bh, main='BH (FDR)')
hist(p.by, main='BY (FDR)')

```

[Return to index](#index)

---

### Check your understanding 1

1) 

[Return to index](#index)

---

### Overfitting

Overfitting describes the phenomena of models being fitted to the 'noise' in the
  training-set more than the 'signal'. That is, for a linear relationship, the linear
  trend in the relationship between variables is the signal, while the dispersion of 
  observations around the prediction line (the deviations) represents the noise. 
  Overfitting leads to poor predictive performance on new samples, because the noise 
  in each sample is different. The more coefficients we add to a model, the more flexible 
  the fit of the model becomes, which can be very helpful when the true form of the 
  relationship between variables is complex, but also provides more opportunities to 
  achieve a very good fit to the training data by adapting to the noise in those data. 
  This is particularly likely when sample sizes are small or noise is large compared to 
  the signal in the data. However, for any sample of size `n`, a linear model with `p` 
  coefficients will fit the data perfectly (noise and all) when `p >= n`.

Therefore, it is important to try to use larger samples to train more complex models, 
  to pick the most parsimonious (fewest coefficients) model consistent with the training 
  data, and only rely on evaluations performed with an independent test-set.

In the example below, we will generate samples of varying sizes, with each observation
  having `x` and `y` variables that are independent of each other. We will then see how
  adding terms increases the flexibility of the model, allowing 'perfect' regression 
  fits (curves which pass exactly thru each data point in training set) to be found even 
  when there is no relationship between the variables. In these cases, since the fit is 
  perfect, the residuals are all zero, so the standard error is zero, which means that 
  none of the usual statistics can be calculated (you would be dividing by zero) and NA 
  is returned instead:  

```
rm(list=ls())
set.seed(1)

f.draw <- function(fit, lty, col) {
  x.plot <- 1:10000
  y.plot <- predict(fit, newdata=data.frame(x=x.plot))
  lines(x.plot, y.plot, lty=lty, col=col)
}

f.fit <- function(n, frm, lty=2, col='cyan') {
  x <- seq(from=1, to=10, length.out=n)
  y <- rnorm(n, mean=0, sd=1)
  fit <- lm(as.formula(frm), data=data.frame(x=x, y=y))
  plot(x, y, main=frm)
  f.draw(fit, lty=2, col=col)
  summary(fit)
}

par(mfrow=c(2, 3))

f.fit(2, 'y ~ x')
f.fit(3, 'y ~ x')
f.fit(3, 'y ~ x + I(x^2)')
f.fit(4, 'y ~ x')
f.fit(4, 'y ~ x + I(x^2)')
f.fit(4, 'y ~ x + I(x^2) + I(x^3)')

```

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Model selection

We sometimes know from theory what terms should be included in a model, and in this case,
  we should not reject any of those terms simply because the corresponding coefficient t-test 
  returned a non-significant p-value. Nevertheless, there are many times when there is no 
  theory to guide us and we need to empirically determine a good model. Deciding which terms
  to include in the model can be guided by t-tests on individual coefficients, but the 
  effects of multiple testing must be accounted for.

As we build a model empirically, certain rules of thumb should be kept in mind. Usually, 
  lower-order terms should be included if one is including any higher order terms (e.g. polynomial
  terms or interactions) that encompass the lower order terms. For instance, if one wanted to 
  include the polynomial term `I(x^3)` in a model because the corresponding coefficient was 
  significant, we should also include the lower order polynomial terms `I(x^2)`, and `x`, even 
  if the corresponding coefficients are not significant. One reason for this is that omitting the 
  lower order terms enforces certain assumptions about the form of the model that we have not 
  established simply by having a coefficient test fail. For instance, if the model `y ~ x + I(x^2)` 
  was found to have a significant coefficient test for `I(x^2)`, but not `x`, then we might be 
  tempted to drop `x` from the model. However, this would imply that the model curve (a parabola) 
  minimum value MUST be at `x == 0`. This would be a strong constraint that is not really warranted 
  just because the coefficient for `x` is not significant. This non-significant coefficient test
  just means we cannot assert that the minimum is not at zero, but that is not the same thing as
  justification to claim it must be at zero. It is better to let the quadratic curve minimum be 
  estimated from the data itself. This allows the model to follow the data more closely, while 
  not changing the 'flexibility' and general shape associated with a quadratic model. Similary, 
  we normally don't drop an intercept term (the lowest order term in the model) from a model just 
  because the coefficient estimate is not significant, because doing so forces a linear model to 
  pass through the origin. Letting the intercept remain, allows the axis intercepts to be fitted 
  from the data without changing the intrinsic flexibility of the shape being modeled. On the
  other hand, sometimes theory suggests that `y == 0` when `x == 0`, or that a minimum of
  the curve is at `x == 0`. For instance, if we were growing trees from seed, it might not
  be unreasonable to assume that the size at time zero is a minimum or in fact essentially 
  zero. In that sort of case you should remove the lower order terms from the model.

When working with the p-values returned by t-tests on regression coefficients,
  the context should be reflected upon. Sometimes only a **single coefficient 
  is of interest**. This is typically true in the case of simple linear 
  regression, since there is only one explanatory variable. However, it can 
  also be true in the case of multiple regression. Frequently, we are interested 
  in whether one particular variable or interaction term is useful for 
  explaining or predicting the response variable, and we include other predictor 
  variables because we already know they are important. For instance,
  we may be interested in whether there are mortality rate differences between
  women and men. So we may have a categorical variable for `Gender` in our model, 
  but we might also include the continuous variable `Age`, because we already 
  know that mortality rate (e.g. proportion of individuals who die within a year)
  is strongly associated with `Age`. Therefore, we include `Age` because it 
  reduces the sum-of-squared residuals when fitting the data, which in turn
  reduces the standard error of all the coefficient estimates, which will make
  our test on the `Gender` coefficient (in this case) more powerful. Here, no 
  adjustment of the result from the `Gender` coefficient test is required, since 
  only a single coefficient is of interest (even though `lm()` will return t-test 
  results for the `Age` coefficient as well, it is not of interest to us: `Age` 
  was introduced as a `covariate` to control for a known effect). If the `Gender` 
  coefficient is significant, it suggests that `Gender` impacts the mortality rate. 
  A non-significant test on the `Age` coefficient may simply reflect a lack of 
  power due to too small a sample size or too much noise in the data.

If you wish to **test multiple coefficients** in a single model, you can focus on
  the corresponding coefficients alone (e.g. you can still exclude tests on
  variables introduced to reduce standard errors), however you should adjust
  for multiplicity of coefficient tests you are interested in. Whether you use 
  FWER or FDR control will depend on how tolerant you are of false positive or false
  negative results. If the number of hypotheses is relatively small, FWER often makes 
  more sense, while FDR control may make more sense as the number of tests becomes 
  larger.

If you are trying to **select terms for inclusion** in your model, FWER control should 
  typically be applied when the number of coefficient tests is small, and FDR 
  control used when the number of tests rises. When constructing the model, whenever 
  possible (sometimes there are too many models required to make this practical), 
  you should still also take careful account of residual plots as you build the model 
  and especially when evaluating the final model. If you are working with a linear
  model with categorical explanatory variables, if you want to achieve similar 
  interpretability of 'significance' as a **post-hoc test from an ANOVA**, FWER control
  should be used on the tests of the corresponding coefficients. It is curious that 
  most statistics textbook will advise multiple testing control when conducting post-hoc 
  testing after an ANOVA, but rarely mention this advice in the context of what are 
  essentially equivalent tests on coefficients of categorical variables included in a 
  linear regression.  

As you may have noticed, manually building a model requires either a solid theoretically-driven
  mathematical model as a starting point, or tedious and tricky exploration of the possible
  variables and their interactions. This is a difficult largely subjective process, facilitated 
  by experience building models and subject-matter knowledge. However, there are some parametric
  methods we can use that will help us objectively compare two models. As with all parametric methods,
  the p-values returned are dependent on certain assumptions being made about either the data or 
  residuals being normally distributed, or there being enough data that asymptotic approximations
  (e.g. invoking the CLT) will provide results that are 'close enough'.

We can use an **ANOVA** to compare two linear models, when one is nested within the other. Nesting 
  means that all the terms in the smaller model are included in the larger model. That is, the models 
  `y ~ 1`, `y ~ x1`, `y ~ x2`, and `y ~ x1 + x2`, are nested within the model `y ~ x1 + x2 + x1:x2`, 
  but the model `y ~ x3` is not nested within the larger model, since it includes the new variable 
  `x3`. We can use the R `anova()` function to compare two nested linear models. The `anova()` function 
  compares the sum-of-squared residuals (**SSR**) from the smaller model to the SSR from the larger 
  model. The proportion reduction in the SSR with the larger model is compared to the number of 
  coefficients added by the larger model in order to determine if the larger model is fitting better 
  solely due to increased flexibility (due to inclusion of more coefficients). This procedure makes the 
  same assumptions about random sampling, normally distributed residuals or 'large enough' a sample 
  size as the parametric F-tests and t-tests of a linear regression. In addition, although it provides 
  some control of overfitting, since it only uses the training-set for evaluation, it still retains 
  substantial potential for overfitting. 

One big advantage of this procedure is that it can provide p-values for terms, not just individual 
  coefficients. This makes testing for inclusion of interaction terms involving categorical variables 
  more straightforward, since a single p-value is returned for the interaction, instead of a separate
  p-value of each coefficient, where some might be significant and others not. We can also compare 
  simultaneous inclusion of multiple terms.

```
rm(list=ls())

dat <- mtcars[, c('mpg', 'wt', 'disp', 'gear')]
dat$gear <- factor(dat$gear, ordered=F)     ## turn gear into an unordered categorical variable

fit1 <- lm(mpg ~ 1, data=dat)               ## intercept only (global mean) model
fit2 <- lm(mpg ~ wt, data=dat)              ## implicitly, for fit2: mpg ~ 1 + hp, so fit1 nested w/i fit2
anova(fit1, fit2)
anova(fit2, fit1)                           ## result order changes with input order, but conclusions same
summary(fit2)                               ## for 1 numeric term: anova p-value same as for t-test on coefficient 

fit3 <- lm(mpg ~ wt + gear, data=dat)       ## fit2 nested w/i fit3
anova(fit2, fit3)
summary(fit3)

fit4 <- lm(mpg ~ wt * gear, data=dat)       ## same as mpg ~ wt + gear + wt:gear; makes mpg vs. wt slopes gear-specific
anova(fit2, fit4)
summary(fit4)
par(mfrow=c(2, 3))
plot(fit4, which=1:6)

```

The **Akaike Information Criterion** or **AIC** extends the parametric model selection approach to 
  non-nested models evaluated solely using the training-set. One can in principle use the AIC to 
  compare arbitrary models within the same parametric family (e.g. linear models with the usually 
  assumed normal errors to other linear models with assumed normal errors). This method is also 
  parametric, with normal distributions assumed at many points in the derivation of the criterion, 
  so for most real-world datasets, the p-values produced are only asymptotically correct. 
  Nevertheless, many models have successfully been built using AIC as a guide. For nested models, 
  using the AIC criterion is equivalent to the ANOVA approach, except with an adjusted p-value 
  cutoff. Both methods balance the improvement in model fit to the training data against the 
  difference in the number of coefficients included in each model. As with ANOVA, AIC testing is 
  designed to resist overfitting the data, but since only the training-set is used for evaluation, 
  risks of overfitting remain.

Manual model building is very tedious and has a substantial subjective component to it. This makes
  it impractical for building very large models, building very large numbers of models, or for 
  objective/reproducible model generation. One popular alternative approach is to use an automated 
  procedure to build the model in a step-wise process. In R, we can use the `step()` function to 
  do this for linear models. In this case, an initial model is picked, a range of model 
  complexity to explore is specified, and the direction to explore relative to the initial model
  (e.g. `direction=backward` to try smaller models only; `direction=forward` to try only larger
  models, and `direction=both` to try both larger and smaller models). Assuming the `step()` 
  function is called with `direction=both`, then the `step()` function will try to drop one
  term at a time from the current model and note the resulting change in AIC. Then it will try 
  to add one term at a time from the set of terms in the maximal model not included in the 
  present model, and note the resulting change in the AIC. It always adds bare variables before 
  adding higher order polynomial and interaction terms involving the variable (as you should) 
  and always drops higher order terms before dropping the corresponding bare variables. If any 
  of these model deletions or additions improve on the AIC of the current model, the change 
  resulting in the largest drop in AIC is made to the model. Then the procedure is repeated, 
  until no further improvement in the model is achieved through any one-step change. The resulting 
  model can depend on the initial model selected (especially when terms are correlated), so 
  sometimes it makes sense to try several starting points and pursue the best overall result. 
  Since the process is step-wise, it can miss models which might reduce the AIC substantially, 
  but only by adding several terms simultaneously. For instance, in the last example, we saw
  that the model `mpg ~ wt * gear` scored much better than `mpg ~ wt`, however, getting to 
  the first model from the second in a step-wise process would require passing through the
  step `mpg ~ wt + gear` first, and that model did not look like an improvement, so that path
  would have not been further explored, and `mpg ~ wt * gear` would never have been found.

When predictors are correlated, introducing one predictor into the model can make the rest 
  look less important, because much of the information they had to share was included with 
  the first predictor. Under these circumstances, the variables included in the initial model 
  can have a big effect on what other variables get included and the final model chosen. 
  When dealing with correlated predictors, it is often a good idea to try several different 
  initial models, and compare results, looking for models that score well and are arrived at 
  from a variety of starting points. When predictors are not correlated, the choice of the 
  initial model is less important, as long as the process converges, but it still may be 
  advisable to try several initial models.

In any case, the final model from any step-wise selection procedure will tend to have a strong
  optimistic bias (R-squared too high, p-values too low) and should be evaluated with an 
  independent test-set.

```
rm(list=ls())

(fit1 <- lm(mpg ~ 1, data=mtcars))
(fit2 <- lm(mpg ~ .^2, data=mtcars))

## add/delete (since direction='both') one 'term' at time, within the bounds 
##   specified by fit1 (intercept only) and fit2 (all predictors and two-way interactions)
##   keeping model that drops AIC most; stops when no single step further improves model;
##   here start with smaller model, since larger model has too many coefficients to 
##   estimate with so few data points:

(fit3 <- step(fit1, scope=list(lower=fit1, upper=fit2), direction='both'))

summary(fit3)
par(mfrow=c(2, 3))
plot(fit3, which=1:6)

```

We can use an independent test set to evaluate the entire procedure. Often the most 
  'independent' data we can muster are a hold-out test-set from the same experiment. 
  Although performance estimates using a hold-out set from the same experiment (or 
  cross-validation, which produces more precise results) are optimistically biased, they 
  tend to be far less so than similar performance estimates made with the training-set.

Let's try to evaluate our process with a single hold-out test-set consisting of about 10%
  of the observations, emulating a single fold out of a 10-fold cross-validation:

```
rm(list=ls())
set.seed(1)

dat <- mtcars
mult <- 2                         ## 'k' for AIC

(n <- nrow(dat))                  ## sample size; not a ton of data; try 10-fold CV
(n.tst <- round(n / 10))          ## test set size
(n.trn <- n - n.tst)              ## training set size

## integer index of training samples:
idx.trn <- sample(1:n, size=n.trn, replace=F)

## code below will end up in a function taking idx.train, dat, and mult:

dat.trn <- dat[idx.trn, ]
dat.tst <- dat[-idx.trn, ]

fit1 <- lm(mpg ~ 1, data=dat.trn)
fit2 <- lm(mpg ~ .^2, data=dat.trn)

## trace=1 so we can see the details of the process:
fit3 <- step(fit1, scope=list(lower=fit1, upper=fit2), k=mult, direction='both', trace=1)

mpg.trn <- predict(fit3, newdata=dat.trn)
mpg.tst <- predict(fit3, newdata=dat.tst)
mpg.int <- predict(fit1, newdata=dat.tst)

(mse.trn <- mean((dat.trn$mpg - mpg.trn) ^ 2))
(mse.tst <- mean((dat.tst$mpg - mpg.tst) ^ 2))
(mse.int <- mean((dat.tst$mpg - mpg.int) ^ 2))

```

Now we'll make a function out of the code above, so that we can more easily perform a
  cross-validation to evaluate our procedure. The function will take the arguments
  `idx.trn`

use to facilitateMake a function for CV, taking `idx.trn`, `dat`, and `mult`:

```
library('caret')                  ## where createMultiFolds() is found

rm(list=ls())
set.seed(1)

(n <- nrow(mtcars))               ## number of observations in mtcars data set
idx <- 1 : n                      ## integer indices of all observations

## folds for 10-fold CV: each fold is integer index of training observations:
folds <- createMultiFolds(idx, k=10, times=3)

## must take idx.trn as first positional argument:

f.mse <- function(idx.trn, dat) {

  dat.trn <- dat[idx.trn, ]
  dat.tst <- dat[-idx.trn, ]

  fit1 <- lm(mpg ~ 1, data=dat.trn)     ## minimal model: intercept-only; global mean
  fit2 <- lm(mpg ~ .^2, data=dat.trn)   ## maximal model: all predictors and 2-way interactions

  ## trace=0, so printout not too long:
  fit3 <- step(fit1, scope=list(lower=fit1, upper=fit2), direction='both', trace=0)

  mpg.trn <- predict(fit3, newdata=dat.trn)
  mpg.tst <- predict(fit3, newdata=dat.tst)
  mpg.int <- predict(fit1, newdata=dat.tst)

  (mse.trn <- mean((dat.trn$mpg - mpg.trn) ^ 2))  ## mse on training-set
  (mse.tst <- mean((dat.tst$mpg - mpg.tst) ^ 2))  ## mse on test-set
  (mse.int <- mean((dat.tst$mpg - mpg.int) ^ 2))  ## mse on intercept-only/global-mean model

  c(mse.trn=mse.trn, mse.tst=mse.tst, mse.int=mse.int)
}

idx.trn <- folds[[1]]
f.mse(idx.trn, dat=mtcars)

## note how dat parameter gets passed as 'extra' parameter to sapply():
rslt <- sapply(folds, f.mse, dat=mtcars)

apply(rslt, 1, mean)
apply(rslt, 1, sd)
t(rslt)

```

[Return to index](#index)

---

### Check your understanding 3

1) question here

[Return to index](#index)

---

## FIN!
