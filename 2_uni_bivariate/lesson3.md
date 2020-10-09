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

In the previous lesson we estimated the proportion of two mutually exclusive 
  groups within a population based on a random sample of observations drawn from 
  that population. We used the R `prop.test()` function to generate a confidence 
  interval for the population proportion and test the hypothesis that the 
  proportion was 0.5. We also learned that we could test against other 
  hypothetical proportions by including those proportions as the value for the 
  `p` parameter in our call to `prop.test()`.

Now we will look at how to use the same `prop.test()` function to compare group 
  proportions between two populations. This is different than the case in which 
  we compare a single population to a hypothetical value because a hypothetical
  value can be specified precisely, while population means are only approximately
  estimated by sample means. This extra uncertainty is reflected in wider intervals
  and higher p-values. As usual, we assume that each sample is randomly drawn 
  from its respective population. This ensures that each observation is 
  independently distributed from other observations in either sample. The values 
  of prior observations have no effect on the value of subsequent observations. 
  In most textbooks, you will see that the p-values and confidence intervals 
  for the 'z-test of proportion' is base on the standard normal distribution 
  N(0, 1), but for two-sided tests, R uses the chi-square distribution with one 
  degree of freedom, which is equivalent. In any case, the use of either 
  parametric distribution to describe uncertainty is justified by invoking the 
  CLT. As mentioned in the one-sample case, for proportions, we usually want a 
  count of at least 5 in each group in order to invoke the CLT with confidence.

The p-value returned is for the null hypothesis that the group proportions in 
  the first population being sampled are the same as the group proportions in 
  the second population. The confidence intervals returned by the R `prop.test()`
  is a confidence interval on the difference in the proportions in the 
  two populations.

Here we will use a dataset built into R in order to demonstrate. R has many
  different datasets included in the base distribution. They can typically
  be loaded by simply invoking their name. Descriptions of the datasets
  can be found in the same way as for function or operators, but using
  the '?' prefix operator:

```
## prep our data for analysis:
rm(list=ls())

data()                              ## view available datasets
help(package='datasets')            ## another view
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

As is often the case in data analysis, setting up this dataset for analysis is 
  more work than actually performing the statistical tests:

```
## did arrest rates (proportion per 10,000 or 1e4) differ?
(rslt <- prop.test(x=x$Total, n=c(1e4, 1e4)))
rslt$conf.int                     ## confidence interval on difference in proportion

## did murders as a proportion of arrests differ?
(rslt <- prop.test(x=x$Murder, n=x$Total))
rslt$conf.int                     ## confidence interval on difference in proportion

```

The `prop.test()` function can be used to extend the proportion test to more than 
  2 populations (or more than 2 groups). The null hypothesis in this case is that 
  all of the population proportions are equal. No confidence intervals are 
  returned in this case:

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

In the single sample case, we saw that we could use the binomial exact test when
  group counts were too low to justify invoking the CLT and using `prop.test()`. 
  However, the binomial test is not suitable for use with more than one sample.
  Instead, you can consider using Fisher's Exact Test, which is implemented by
  the `fisher.test()` function in R. The null hypothesis is once again that all
  group proportions are the same. It is expressed in terms of 'odds ratios'. An
  odds ratio of an event is the probability that the event will occur divided by
  the probability it will not occur: `odds <- p / (1 - p)`. In the case of 
  group proportions, the odds are the odds that a random observation drawn from
  the population will belong to a particular group. The null hypothesis is 
  stated in terms of the 'odds ratios' for group membership being the same for
  all groups and all populations tested. If the odds in both groups are the same,
  the odds ratio should be one. Confidence intervals for the actual odds ratio
  are only returned for 2x2 tables (where there is only one ratio to consider).
  In addition, the test is only 'exact' for 2x2 tables and an approximation is
  used for larger tables. Fisher's exact test has an unusual assumption (not 
  made by `prop.test()` that the marginal totals (the column sums and row sums for 
  the table being analyzed) are 'fixed'. That is, the p-value returned is 
  conditioned on the marginal total: if the experiment yielded the observed 
  marginal totals, the chance you would see this distribution of group counts
  among your two samples given the null hypothesis is p. This assumption has
  led to a number of controversies over application, but you will often see 
  published work where the Fisher's test was used and interpreted without 
  regard to marginal totals.

```
(x <- dat[c('Vermont', 'New York'), c('Murder', 'Total')])
t(x)
fisher.test(x)
fisher.test(t(x))                 ## orientation of your table irrelevant

(x <- dat[c('Vermont', 'New York', 'California'), c('Murder', 'Total')])
fisher.test(x)
fisher.test(t(x))

```

[Return to index](#index)

---

### Check your understanding 1

1) Compare the arrest rates for assault in Alaska and Maine using `prop.test()`.

2) Compare the arrest rates for assault in Alaska, Maine, and North Dakota using
   `prop.test()`.

3) Repeat #1, but using `fisher.test()`. Are the results similar?

4) What is the null hypothesis of these tests?

[Return to index](#index)

---

### Comparing two population means

intro here; comparing different groups; e.g. yields of different strains of wheat,
  outcome of different drug treatments, output of different production lines.
  what about doing repeated 2-way tests. Description of multiple
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
  of freedom of 30 or more) the model is robust to fairly large departures
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

Some stuff about how `summary()` is essential for this:

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
  against any distribution or against another dataset.

If there appear to be worrisome departures from assumptions, one can opt to use the R 
  function `kruskall.test()` to perform a 'rank' test (or rank transform the data and do 
  a regular `aov()`) instead, but both of these approaches are also plagued by 
  assumptions that can be hard to strictly meet. Furthermore, the Kruskall test null 
  hypothesis is about medians, not means, so is not strictly comparable with ANOVA. 
  Fortunately, as was mentioned earlier, the ANOVA is fairly robust to violations of
  assumptions. 

In general, if the data within each group are symmetrically distributed about the group 
  mean (you can plot residuals of each group separately to look at this) and each group is
  represented by a very similar number of observations (ideally should design them to
  be identical, but a missing value here or there should be ok), results from ANOVA is 
  usually fairly robust to minor departures from normality of residuals or even two-fold 
  differences in standard deviation (4-fold for the variance) within different groups. 
  Outliers are a bigger worry. One simple way to look for outliers is in the residual 
  distribution for observations more than 3-standard deviations away from the residual 
  mean (the residual mean is always zero).

The linear modeling framework and resampling methods that will be introduced in the next 
  two lessons provide additional ways to examine the robustness of our conclusions from ANOVA 
  to deviations from the underlying assumptions.

Testing for normality and homogeneity: tests get more powerful as the number of samples
  increases, which is when the assumptions matter least.

```
## plot residuals:

nrow(dat)
res <- residuals(rslt)
summary(res)
length(res)
plot(res / sd(res))                    ## no residuals more than 3 * sd from 0

par(mfrow=c(2, 2))

## some departure from N() in tails evident:

qqnorm(res, main='ANOVA residuals')    ## quantiles of res vs. quantiles of N() w/ same mean and sd
qqline(res)                            ## line thru 1st and 3d quantiles of corresponding N()

set.seed(1)
tmp1 <- rnorm(length(res), 0, sd(res))
qqnorm(tmp1, main='Normal distribution')
qqline(tmp1)

tmp2 <- rt(length(res), df=length(res) - 2)
qqnorm(tmp2, main='t-distribution')
qqline(tmp2)

tmp3 <- rt(length(res), df=1)
qqnorm(tmp3, main='t-distribution, df=1')
qqline(tmp3)

par(mfrow=c(1, 1))

## do a formal test for normality:

shapiro.test(res)                 ## not normal, but how much does it matter?
shapiro.test(tmp1)                ## random sample from normal
shapiro.test(tmp2)                ## random sample from t w/ df like for rslt
shapiro.test(tmp3)                ## random sample from t w/ df=5

## do a formal test for homogeneity of variances; can use the same 'formula' as
##   was used when doing aov(); strongly rejects homogeneity, but so what?:

bartlett.test(breaks ~ tension, data=dat)

```

If an ANOVA 'omnibus' F-test returns a significant p-value, it suggests that at least one 
  of the groups has a mean different from the others. However, it does not tell us which groups
  are different from one another. If (and usually only if) the omnibus test rejects the null
  hypothesis that all group means are the same, we need to conduct a 'post-hoc' (after the fact)
  test to determine which means are different. We can use t-tests for this, comparing each pair
  of group means to one another. However, this introduces the issue of 'multiple testing' 
  that we mentioned earlier. Fortunately, there are several purpose-built methods for conducting 
  post-hoc tests that implicitly account for multiple testing. One commonly used test is called 
  Tukey's HSD (honest significant difference) test. This compares all groups to one another, 
  using the same approach used for constructing an equal-variance t-test, except adjusting the 
  returned p-values to properly account (returning the FWER or 'family-wise error rate') for 
  the multiplicity of tests performed. For instance,
  in the case of 3 groups, Tukey's HSD will perform three comparisons, corresponding to each 
  possible group pairing: 1v2, 1v3, and 2v3. The assumptions behind Tukey's HSD are essentially 
  identical to those for the ANOVA itself. 

What if a particular contrast of prior interest: jump to post-hoc test. Why use the omnibus at 
  all? Conduct fewer overall tests. Less 'adjustment' required, more powerful individual tests.
  Non-significant omnibus does not rule out significant post-hoc. Similarly, significant omnibus
  does not guarantee you will have a significant post-hoc. You might be able to assert that some
  of the means are different without being able to assert which ones are different.

```
## the data:
summary(dat)

## the aov() fit:
rslt
summary(rslt)

## Tukey's HSD:

(hsd <- TukeyHSD(rslt))
class(hsd)
is.list(hsd)
names(hsd)

hsd$tension                       ## name depends on starting variable name (dat$tension)
class(hsd$tension)                ## familiar; use [row, col] indexing
is.list(hsd$tension)              ## do not try to use '$' to index!

rownames(hsd$tension)             ## means being compared
hsd$tension[, 'diff']             ## difference between respective means
hsd$tension[, 'lwr']              ## 95% CI lower bound on difference
hsd$tension[, 'upr']              ## 95% CI upper bound on difference
hsd$tension[, 'p adj']            ## adjusted p-value (FWER) for difference

plot(hsd)                         ## see results graphically (plot ci's vs. 0)

```

Another commonly employed post-hoc test, particularly 
  in the sciences is Dunnett's test, which only compares groups to a negative control. This 
  means that fewer tests are conducted, resulting in each test being more powerful to detect a
  difference than if we had used Tukey's HSD. For instance, in the three group case, Dunnett's
  test only conducts two tests, corresponding to all possible pairings of the negative control
  group to each other group: 1v2 and 1v3, assuming group 1 is the negative control. 

In order to identify the significant differences we
  need Only do if 'omnibus' F-test is significant.
  Tukey: all-vs-all. Same assumptions as ANOVA, plus equal-sample sizes;
  Dunnett's: all-vs-negative_control.

```
## Dunnett's test:

libary()
## install.packages('multcomp')   ## if you don't already have it
library('multcomp')               ## load the library
sessionInfo()                     ## see the version

## sorry, multcomp can be ugly:
(dun <- glht(rslt, linfct=mcp(tension="Dunnett")))
class(dun)
is.list(dun)
names(dun)
(smry.dun <- summary(dun))
class(smry.dun)
is.list(smry.dun)                 ## can use '$' to index
names(smry.dun)                   ## not clear e.g. where is p-value?
str(smry.dun)
smry.dun$test
class(smry.dun$test)
is.list(smry.dun$test)
names(smry.dun$test)
smry.dun$test$coefficients        ## numeric vector
smry.dun$test$pvalues             ## numeric vector

```

[Return to index](#index)

---

### Check your understanding 2

1) ANOVA

2) TukeyHSD

) What is the null hypothesis of a 2-sample t-test?

) What is the null hypothesis of a 1-factor ANOVA?

) Should you try post-hoc tests without conducting an omnibus test first?

[Return to index](#index)

---

### Association between two variables

intro here; prop.test() already can test for categorical associations. Give
  example or interpret past problem in this light.

Observe two numeric variables on same subjects. For example, can measure the
  both height and weight of N individuals, resulting in two vectors of the 
  same lengths (say `ht` and `wt`) with measurements for a given individual at 
  the same index position (so the height for the i-th individual would be 
  `ht[i]` and that person's weight would be `wt[i]`). We may think that there
  may be a relationship between height and weight (e.g. that the taller someone
  is, the heavier they tend to be), and wish to test this idea. To do so,
  we could look for a 'correlation' between our variables. The best known 
  measure of correlation is Pearson's correlation. This is a value ranging 
  between 1 (signaling a perfect positive correlation: whenever `ht` goes up, 
  `wt` goes up) and -1 (signaling perfect negative correlation: whenever `ht` 
  goes up, `wt` goes down). For two variables that are not associated with one
  another, Pearson's correlation will be near zero.

Pearson tests for (and assumes) a linear relationship between the two 
  variables. It further assumes that residuals (distances from the observations 
  to the prediction line) are homoskedastic. Pearson's test is defined in terms
  of 'sums-of-squares' so the results are relatively sensitive to outliers. 
  However, for continuous variables without obvious outliers, and where the
  assumptions hold, Pearson's test of correlation is the most powerful test of 
  association. For discontinuous variables, where outliers are suspected, or 
  where the assumptions are suspect, Spearman's rho test or Kendall's tau test 
  are better choices than Pearson's correlation. Spearman's rho and Kendall's 
  tau approaches make relatively few assumptions outside of random sampling
  of populations and a monotonic relationship between variables. Non-monotonic
  (as well as non-linear) relationships can often be discerned by plotting one 
  variable against the other. If a non-monotonic relationship is discovered,
  other methods should be used to characterize the association between the
  variables.

Spearman's test amounts to taking a rank transformation of the data (for instance, 
  substitute all the `ht` values with their ranks from smallest to largest; then 
  do the same thing for the `wt` values), then conducting Pearson's test on the 
  transformed data. Kendall's tau test also ranks the data for each variable, but 
  the theoretical approach and p-value calculation differs from Pearson's and 
  Spearman's correlations. Nevertheless, the p-values returned by Kendall's tau
  test tend to be pretty similar to those returned by Spearman's procedure, and 
  neither method makes many assumptions outside of random sampling and a 
  monotonic association between variables (the plot of one variable against the 
  other is always rising or always falling: it does not change directions). In 
  cases where there is a non-monotonic association between variables, other 
  methods should be considered for characterizing the relationship. Kendall's 
  tau test is sometimes preferred over Spearman's rho because the tau statistic 
  has a fairly straight-forward interpretation. Like the Pearson correlation 
  (and therefore Spearman's rho), tau varies between -1 (perfect negative 
  association) and 1 (perfect positive association) with 0 indicating no 
  association. However, for intermediate values, tau actually represents a 
  readily interpretable probability, while Spearman's rho statistic is not. 
  Kendall's tau estimates the probability that if one variable goes up, the 
  other variable will go up as well (for positive tau), or the probability that 
  if one variable goes up, the other will go down (for negative tau).

```
rm(list=ls())
set.seed(3)
n <- 100

## strong positive linear association:

x <- runif(n, 0, 10)
e <- rnorm(n, 0, 1)
y <- 3 * x + e

plot(y ~ x)                                             ## can use formula notation here too
rslt <- cor.test(x, y)                                  ## method='pearson' by default
class(rslt)                                             ## our old friend, htest
is.list(rslt)                                           ## can use '$' indexing
names(rslt)                                             ## many familiar faces

cor.test(x, y, method='pearson')                        ## same thing, explicit method
cor.test(x, y, method='pearson', alternative='greater') ## h0: r <= 0
cor.test(x, y, method='spearman')                       ## pearson on ranked data
cor.test(rank(x), rank(y), method='pearson')            ## equivalent to spearman
cor.test(x, y, method='kendall')                        ## different rank-based

## weak negative linear association:

e <- rnorm(n, 0, 10)
y <- -3 * x + e

plot(y ~ x)
cor.test(x, y, method='pearson')
cor.test(x, y, method='spearman')
cor.test(x, y, method='kendall')

## no association:

x <- rnorm(n, 0, 1)
y <- rnorm(n, 0, 1)

plot(x, y)
cor.test(x, y, method='pearson')
cor.test(x, y, method='spearman')
cor.test(x, y, method='kendall')

## no association, with one big outlier:

x <- rnorm(n, 0, 1)
y <- rnorm(n, 0, 1)
x[n] <- 10
y[n] <- 10

plot(x, y)
cor.test(x, y, method='pearson')    ## pearson very sensitive
cor.test(x, y, method='spearman')   ## spearman only slightly sensitive
cor.test(x, y, method='kendall')    ## kendall only slightly sensitive

## perfect (x 100% predicts y) non-monotonic association:

x <- runif(n, -1, 1)
y <- x ^ 2

plot(x, y)                          ## very strong interdependence
cor.test(x, y, method='pearson')    ## undetectable
cor.test(x, y, method='spearman')   ## undetectable
cor.test(x, y, method='kendall')    ## undetectable

```

[Return to index](#index)

---

### Check your understanding 3

Using the `mtcars` dataset:

1) In a 1 row, 3 column plot layout (use `par(mfrow=c(?, ?))`, substituting the right
   values for the ?s), plot the relationship between `mpg` (miles per gallon) and `disp` 
   (engine displacement in cubic inches); do the same between `mpg` and `wt` (car weight 
   in 1000s of lbs); finally, plot `disp` vs `wt`.

2) What is the Pearson's correlation between `mpg` and `disp`. Is it significant? What 
   is the confidence interval?

3) Conduct a Kendall's test on the correlation between `mpg` and `wt`. What is the p-value?
    Is there a confidence interval?

4) Conduct a one-sided Spearman's test on the correlation between `mpg` and `wt`. What is
   the p-value? Is there a confidence interval?

5) What is the null hypothesis for all three of these correlation tests?

[Return to index](#index)

---

## FIN!
