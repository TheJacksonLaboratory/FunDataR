# Fundamentals of computational data analysis using R
## Univariate and bivariate statistics: bivariate statistics
#### Contact: mitch.kostich@jax.org

---

### Index

- [Comparing two population proportions](#comparing-two-population-proportions)
- [Comparing two population means](#comparing-two-population-means)
- [Comparing three or more means](#comparing-three-or-more-means)
- [Association between two variables](#association-between-two-variables)

### Check your understanding

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Comparing population proportions

intro here

assumption: random sampling implies independence of observations; 

brief description of dataset

```
## prep our data for analysis:
rm(list=ls())

?USArrests                          ## find out about a built-in dataset
dat <- USArrests                    ## arrests per 100,000 in 1973; by state
class(dat)                          ## data frame
dim(dat)                            ## one row per state
summary(dat)
head(dat)
tail(dat)

dat <- dat * 10                      ## convert to whole numbers; now arrests per 10,000

## total up arrests:
dat$Total <- apply(dat[, c(1, 2, 4)], 1, sum)
summary(dat)
head(dat)
tail(dat)

## set up comparison of Vermont to New York:
x <- dat[c('Vermont', 'New York'), c('Murder', 'Total')]
x
x$Murder
x$Total

```

Now we can do the actual test (easy part):

```
## did arrest rates (proportion per 10,000 or 1e4) differ?
(rslt <- prop.test(x=x$Total, n=c(1e4, 1e4)))
rslt$conf.int                     ## confidence interval on difference in proportion

## did murders as a proportion of arrests differ?
(rslt <- prop.test(x=x$Murder, n=x$Total))
rslt$conf.int                     ## confidence interval on difference in proportion

```

Extend the proportion test to more than 2 groups:

```
x <- dat[c('Vermont', 'New York', 'California'), c('Murder', 'Total')]
x
x$Murder
x$Total

## did arrest rates (proportion per 10,000 or 1e4) differ AMONG ANY OF THE THREE?
(rslt <- prop.test(x=x$Total, n=c(1e4, 1e4, 1e4)))
str(rslt)                         ## no confidence interval when >2 groups

## did murders as a proportion of arrests differ AMONG ANY OF THE THREE?
(rslt <- prop.test(x=x$Murder, x$Total))
str(rslt)                         ## no confidence interval when >2 groups

```

Mention the chisq.test() for contingency tables and goodness-of-fit. Conceptual
  difference from prop.test(). In R, prop.test() calls chisq.test() then prints
  the results differently.

Binomial test not right for >1 groups. Can use Fisher's Exact Test `fisher.test()`
  to do non-parametric test. Some object to an implicit assumption of the test
  that marginal totals are fixed. Creates some issues defining the population
  the estimates apply to. Calculation of an exact p-value can  become computationally 
  very costly for more than two groups.

```
maybe some code here

```

[Return to index](#index)

---

### Check your understanding 1

1) question here

[Return to index](#index)

---

### Comparing two population means

intro here; what about doing repeated 2-way tests. Description of multiple
  testing issue. 

h0 is difference in group means, `h0: (m1 - m2) == 0`. ci is on 
  the difference in group means, so if 95% CI does not include 0, h0 will be 
  rejected w/ p < 0.05. One-sided h0s: `(m1 - m2) >= 0` or `(m1 - m2) <= 0`

assumption: random sampling implies independence of observations, except for
  optional 'pairing'. Can assume a homogeneous variance, which can improve power
  if the assumption is met. Technically, also assumes normal distribution of data
  w/i groups, but actually ok as long as counts are large enough (>30) to 
  invoke CLT: means of different samples of same group will be normally 
  distributed, which is what actually matters for construction of confidence 
  intervals. Sensitive to outliers because of squared-error 'penalty' function 
  minimized by mean. Plot data to detect outliers.

Brief description of dataset.

```
## prep our data for analysis:

rm(list=ls())

dat <- mtcars
dat
table(dat$cyl)

(x <- dat$mpg[dat$cyl == 4])
(y <- dat$mpg[dat$cyl == 8])

## two-sided test: confidence interval is on difference in group means;
##   default version: upaired, Welch's (no assumption of group variances
##   being equal). p-value is of h0: difference between group means is zero.

(rslt <- t.test(x=x, y=y))

## one-sided test: h0: difference is less than or equal to zero:
(rslt <- t.test(x=x, y=y, alternative='greater'))

## test assuming both groups have same variance:
(rslt <- t.test(x=x, y=y, var.equal=T))

(rslt <- t.test(x=x, y=y, var.equal=T, alternative='greater'))

class(rslt)                       ## h(ypothesis)test
is.list(rslt)                     ## why we can use '$' to index elements
names(rslt)                       ## same old same old
attributes(rslt)

```

A note on picking tests before examining the data.

Background on repeated measures. Trade-off between degrees of freedom and effect size.

```
## set up data:
rm(list=ls())

dat <- sleep
dat
table(dat$group)

(x <- dat$extra[dat$group == 1])
(y <- dat$extra[dat$group == 2])

## do an unpaired test for comparison:

rslt1 <- t.test(x=x, y=y)

## conduct the paired test:

rslt2 <- t.test(x=x, y=y, paired=T)

## can do one-sided as well:

rslt3 <- t.test(x=x, y=y, paired=T, alternative='less')

c(p1=rslt1$p.value, p2=rslt2$p.value, p3=rslt3$p.value)
cbind(ci1=rslt1$conf.int, ci2=rslt2$conf.int, ci3=rslt3$conf.int)

```

[Return to index](#index)

---

### Comparing three or more means

intro here. compares w/i group variance to between group variance, which is 
  implicitly what the t-test does as well. F-test hypothesis. If you do an
  ANOVA on two groups, the 2-sample unpaired t-test w/ homogenous variances 
  and ANOVA will give exactly the same p-value. 

assumptions: random sampling w/i groups implies independence of observations, 
  except for optional 'pairing'. Assumes variance of data within each group 
  is the same (homogenous variances), and that the data is normally distributed
  within each group. In practice, as long as counts are fairly high (degrees 
  of freedom of 30 or more) the model is robust to fairly dramatic departures
  from these assumptions. Like t-test, sensitive to outliers. Outliers best
  detected by plotting residuals from model. Formal tests for normality are
  sometimes recommended or seen in the literature, but these tests can detect
  very small departures from normality that are unlikely to substantively 
  affect the reliability of ANOVA results. The `aov()` function assumes a 
  'balanced' design (equal numbers in each group).

```
## load data:

rm(list=ls())
(dat <- warpbreaks)
class(dat)
sapply(dat, class)
summary(dat)                      ## tabulates factors (nicer than for character)

## make a box-plot:
boxplot(breaks ~ tension, data=dat)

```

Something about how do test, then mostly work with output of `summary()`. 
  Can access elements of result of `aov()` using either '$' or purpose-built
  functions. Better practice to use the functions than to directly access,
  as sometimes what is under the hood needs to be transformed before it yields
  what you might expect. The accessor functions take care of that. Common
  accessor functions for getting residuals `residuals()`, fitted values
  `fitted()`, coefficients `coef()`.

```
## F-test h0: all group means are equal:
rslt <- aov(breaks ~ tension, data=dat)
class(rslt)
is.list(rslt)
names(rslt)
attributes(rslt)
rslt

## can access directly: BUT DON'T UNLESS YOU UNDERSTAND (per docs) WHAT YOU WILL GET
rslt$coeffients
rslt$xlevels
rslt$df.residual

## HOWEVER: best practice to use accessor functions or summary object:
coef(rslt)
f <- fitted(rslt)
table(round(f, 5))                ## three means
r <- residuals(rslt)
r2 <- dat$breaks - fitted(rslt)   ## where residuals came from
table(r == r2)                    ## exact equality is often NOT what you want
all.equal(r, r2)                  ## right way to test for equality here

```

Some stuff about how summary is essential for this:

```
smry <- summary(rslt)[[1]]        ## for 1-way anova, get first list element
class(smry)
is.list(smry)
names(smry)
attributes(smry)
length(smry)
smry

smry$Df                           ## 2 'model df' = 2 means; 3d is 'intercept'; n-2 residual df
smry$Sum                          ## $Mean more important; calculated from this
smry$Mean                         ## average spread between groups > average spread within groups
smry$Sum / smry$Df                ## where the mean came from
smry$F                            ## the statistic to compare against a F distribution with 2 and n-2 df
smry$Mean[1] / smry$Mean[2]       ## where the F-statistic came from
smry$P                            ## omnibus F-test p-value: h0: all group means are equal
1 - pf(smry$F, 2, 51)             ## where the p-value came from

```

How means are represented:

```
## so unmentioned group 'L' mean is 'intercept'; add 'M' or 'H' values to 
##   intercept in order to get respective group means:

(cf <- coef(rslt))
table(dat$tension)
cf[1]                             ## mean of group L
cf[1] + cf[2]                     ## mean of group M
cf[1] + cf[3]                     ## mean of group H

## calculate means by hand:
tapply(dat$breaks, dat$tension, mean)

```

Assumptions checked by looking at distributions of residuals. should be 
  normally distributed with similar variance within each group. Quantile-
  quantile normal plot `qqnorm()` plots the corresponding percentile values from
  a variable (here the residuals from the ANOVA fit) against the expected
  percentile values from a normal distribution with the same mean and sd.
  The related function `qqplot()` allows you to graphically compare the variable 
  against any distribution or another variable/dataset.

```
## plot residuals:

nrow(dat)
res <- residuals(rslt)
summary(res)
length(res)

## some departure from N() in tails evident:

qqnorm(res)                       ## quantiles of data vs. quantiles of N() w/ same mean and sd
qqline(res)                       ## line thru 1st and 3d quantiles of corresponding N()

## do a formal test for normality:

shapiro.test(res)                 ## not normal, but how much does it matter?

```

If there appear to be worrisome departures from assumptions, one can opt to use the R 
  function `kruskall.test()` to perform a 'rank' test (or rank transform the data and do 
  a regular `aov()`) instead, but both of these approaches are also plagued by 
  assumptions that can be hard to strictly meet. Furthermore, the Kruskall test null 
  hypothesis is about medians, not means, so is not strictly comparable with ANOVA. 
  Fortunately, as was mentioned earlier, the ANOVA is fairly robust to violations of
  assumptions. 

In general, if the data within each group are symetrically distributed 
  (can plot residuals of each group separately to look at this) and each group is
  represented by a similar number of observations, results from ANOVA is usually fairly 
  robust to minor departures from normality of residuals or even two-fold differences 
  in variarnce between groups. The linear modeling framework and resampling methods that
  will be introduced in the next two lessons provide ways to further examine the
  robustness of our conclusions from ANOVA to deviations from the underlying assumptions.

If an ANOVA 'omnibus' F-test returns a significant p-value, it suggests that at least one 
  of the groups has a mean different from the others. However, it does not tell us which groups
  are different from one another. If (and usually only if) the omnibus test rejects the null
  hypothesis that all group means are the same, we need to conduct a 'post-hoc' (after the fact)
  test to determine which means are different. We can use t-tests for this, comparing each pair
  of group means to one another. However, this introduces the issue of 'multiple testing' 
  that we will show you how to explicitly address later in this course. The main thing to know
  right now is when you conduct multiple hypothesis tests in a single experiment, you need to
  'adjust' the p-values to account for Fortunately, there
  are several purpose-built methods for conducting post-hoc tests that implicitly account for
  multiple testing. One commonly used test is called Tukey's HSD (honest significant difference)
  test. This compares all groups to one another. In the case of 3 groups, this would result
  in (1v2, 1v3, 2v3). For Dunnett's test, only compare groups to a negative control (1v2, 1v3, 
  assuming group 1 is the negative control). More powerful when those are the only contrasts 
  of interest.

In order to identify the significant differences we
  need Only do if 'omnibus' F-test is significant.
  Tukey: all-vs-all. Same assumptions as ANOVA, plus equal-sample sizes;
  Dunnett's: all-vs-negative_control.

```
## post-hoc test code here

```

[Return to index](#index)

---

### Check your understanding 2

1) question here

[Return to index](#index)

---

### Association between two variables

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
