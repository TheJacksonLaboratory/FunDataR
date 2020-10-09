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

Similar to the proportion test, we can extend the t-test to comparison of a
  continuous variable based on random samples from two different populations.
  This is often used to test for the effect of group status (e.g. treated or
  untreated; strain one vs strain two; etc) on a continuous outcome variable
  (like height, weight, or gene expression level). The two-sample t-test
  conducted with `t.test()` tests the null hypothesis that the two population
  means are identical. The confidence interval returned is for the difference
  in group means, so if the CI includes 0, the p-value will not be significant.
  One can also construct one-sided tests as in the one-sample case.

As for the one-sample t-test, we assume both samples were randomly drawn from
  their respective population (except in the 'paired' case described below). 
  By default, the `t.test()` function assumes that the variance in each 
  population can be different, so estimates a separate variance for each. 
  This implies that under the null group membership may have an effect on the 
  variance of the variable of interest, without affecting the mean. In that
  case a non-signifant p-value should be expected. We can also assume that 
  under the null hypothesis, group membership has no effect whatsoever on the 
  variable of interest, in which case we can instruct `t.test()` to use a 
  single 'pooled' estimate of the variances of both populations. If the 
  common variances assumption truly holds, using the single pooled variance 
  estimate for both populations will tend to produce a more powerful test 
  than using two unpooled variance estimates. However, if the pooling 
  assumption does not hold, using the pooled estimate can result in false 
  positive results (rejection of the null hypothesis when it is in fact true) 
  at a higher rate than the p-value cutoff implies. That is, you may only
  call results significant if the p-value is below 0.05, implying that 
  there is only a 5% chance of a false positive for any single test, but
  because the common variances assumption did not hold, the chances of a 
  false positive test result are actually 8%. On the other hand, if 
  one uses the unpooled estimate when the common variances assumption does 
  hold, the resulting test may result in some borderline cases being 
  negative when the pooled test would have called them positive, but false 
  positive rates will be no higher than their nominal level (e.g. a p-value 
  of 0.05 means that a false positive really will only occur at most 5% of 
  the time). For this reason, the unpooled choice is considered 'safer' and
  is set as the default option in R.

As in the one-sample case, confidence intervals and p-values are based on the 
  parametric t-distribution. Usage of this parametric distribution is again
  justified based on the CLT. In the two-sample case, the 'rule-of-thumb' is
  that each sample should have a size of at least 30. In practice, if the
  populations being sampled are roughly normally distributed, the parametric
  approximation will be 'close enough' at sample sizes considerably smaller
  than 30 (on the order of 10 will usually be sufficient). Conversely, for
  very skewed (unsymmetric) population distributions, even a sample size of
  30 may not be enough for the returned p-values to be accurate. The test
  is more resistant to departures from assumptions when the group sizes are
  equal or nearly so. 

This test is quite sensitive to outliers, due to the implied square-error 
  penalty function being minimized by the means being compared. We can plot 
  the data to detect outliers (a simple 'heuristic' approach is to look for 
  values more than 3 standard deviations from the mean) and then remove 
  them prior to conducting the test.

```
## prep our data for analysis:

rm(list=ls())

dat <- mtcars
dat
table(dat$cyl)

(x <- dat$mpg[dat$cyl == 4])
(y <- dat$mpg[dat$cyl == 8])

## look for outliers (none evident):

par(mfrow=c(1, 2))                          ## figure area: 1 row, 2 columns (1x2)
(c.x <- sd(x) * c(-3, 3) + mean(x))         ## 3 sd outside of mean
plot(x, ylim=range(x, c.x), main='x')       ## make sure ylim accommodates data and cutoffs
abline(h=mean(x), lty=2, col='orangered')   ## horizontal line at mean
abline(h=c.x, lty=3, col='cyan')            ## horizontal lines at mean +/- 3 * sd

(c.y <- sd(y) * c(-3, 3) + mean(y))
plot(y, ylim=range(y, c.y), main='y')
abline(h=mean(y), lty=2, col='orangered')
abline(h=c.y, lty=3, col='cyan')
par(mfrow=c(1, 1))                          ## switch back to 1x1 figure area

## two-sided test: confidence interval is on difference in group means;
##   default version: upaired, Welch's (no assumption of group variances
##   being equal). p-value is of h0: difference between group means is zero.

(rslt <- t.test(x=x, y=y))

## one-sided test: h0: difference is less than or equal to zero:
(rslt <- t.test(x=x, y=y, alternative='greater'))

## test assuming both groups have same variance:
(rslt <- t.test(x=x, y=y, var.equal=T))

## one-sided, equal variances:
(rslt <- t.test(x=x, y=y, var.equal=T, alternative='greater'))

class(rslt)                       ## h(ypothesis)test
is.list(rslt)                     ## why we can use '$' to index elements
names(rslt)                       ## same old same old
attributes(rslt)

```

Above we tried several 'flavors' of the t-test and compared their results. When you
  are conducting a real experiment, you should not fish for a method that gives you
  the result you are looking for, as it will severely affect the legitimacy of your
  conclusions. You should always decide on which tests you will conduct (including
  specifying particulars, such as whether the common variances assumption will 
  be made), nominal coverage for confidence intervals and p-value cutoffs prior to
  looking at your data. After the main analysis is complete, it is legitimate to 
  conduct a 'post-hoc' analysis looking for better approaches to try 'next time', 
  but should not be mixed in with the main results of the study.

Sometimes the two samples you are comparing are not really independent, but observations
  in one sample are logically 'paired' with observations in the other sample. An
  example of this is a before-and-after study. You may take a sample of individuals
  and make measurements on a continuous variable of interest before subjecting the
  individuals to a treatment. You are interested in seeing if the variable is affected
  by the treatment. For instance, we might want to see if vitamin B12 supplements 
  affect GPA of college students. As you might suspect, at the start of the study, 
  a randomly selected sample of college students will have a higher GPA coming into
  the study than other students. You might also suspect that students who had 
  higher GPAs than average before the study might tend to have higher than average
  GPAs after the treatment. Therefore, the 'after' value for the variable of interest
  after treatment will tend to be higher if the 'before' value for that individual
  was higher. The observations in the two samples are not independent of one another,
  so the original assumption of random sampling will not hold. Under these 
  circumstances, we can use the 'paired' version of the two-sample t-test, which
  tests the null hypothesis that the average difference between before and after
  values for each individual is zero. The confidence interval returned is for the
  average size of this before vs. after difference for each individual. It is worth
  noting that even in the 'paired' sampling design, the individuals tested are
  drawn at random from the population to which you want to make inferences about
  treatment effects, like all college students, in the example above. Random 
  sampling is always a necessary step in any statistical inference about populations
  based on samples.

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

## can make this test one-sided as well:

rslt3 <- t.test(x=x, y=y, paired=T, alternative='less')

c(p1=rslt1$p.value, p2=rslt2$p.value, p3=rslt3$p.value)
cbind(ci1=rslt1$conf.int, ci2=rslt2$conf.int, ci3=rslt3$conf.int)

```

[Return to index](#index)

---

### Comparing three or more means

Often we are interested in comparing a continuous variable across more than 
  two populations. For instance in a single experiment, we may wish to 
  compare untreated individuals with those treated with drug 1 and those 
  treated with drug 2. To do so, we can do an 'analysis of variance' (ANOVA) 
  test. The null hypothesis for this test is that all the group means are
  equal. If the null hypothesis is rejected, a follow-on 'post-hoc' test
  is needed in order to determine which population means are different.

The assumptions of this test includes that each sample is randomly drawn
  from its respective population. In addition, it is assumed that under the 
  null hypothesis, each population has the same variance (like 
  `t.test(x, y, var.equal=T)`). The parametric F-distribution is used to 
  generate p-values and calculate confidence intervals. This is justified
  by invoking the CLT, which will be valid if values are normally distributed
  within each population, or for non-normally distributed populations when
  the sample size is 'large enough'. A sample size of 30 for each population
  is typically considered adequate, but then again, this may be more than
  what is needed for nearly normal populations and not enough for heavily 
  skewed population distributions. In practice, it has been found that the
  ANOVA test is fairly robust to departures from normality as well as 
  differences in population standard deviations of two-fold or more as long
  as sample sizes are around 30 or more and the design is nearly 'balanced' 
  (the number of observations from each population are equal).

Formal tests for population normality and equal population variances are 
  often recommended. These are particularly relevant if your sample sizes
  are well below 30. When sample sizes are larger, it is likely that the
  tests for departures from normality or equal variances will be powerful
  enough to detect even small departures, which at these sample sizes are
  likely to be irrelevant. In this case more cursory checks for dramatic
  departures are more appropriate.

Like the t-test, the ANOVA test is sensitive to outliers, which are most
  easily detected by plotting residuals (see below) from the model.

Technically, the ANOVA test compares the variances within each group (assumed
  to be the same in each group, so a 'pooled' estimate is made) to the
  variance between groups. The two-sample unpaired t-test with common 
  population variances can be shown to be making an equivalent comparison,
  and in fact applying the ANOVA to the two-sample case yields exactly the
  same results as this t-test.

```
## load data:

rm(list=ls())
(dat <- warpbreaks)
class(dat)
sapply(dat, class)
summary(dat)                      ## tabulates factors (nicer than for character)

## make a box-plot using formula notation: dat$breaks is a function of dat$tension.
boxplot(breaks ~ tension, data=dat)

```
The ANOVA test can be performed using the R `aov()` function. The result from 
  the call to `aov()` then fed to `summary()` which does the calculation of
  confidence intervals and p-values. 

Some components of the result can be retrieved using purpose built functions, 
  particularly the `coef()` function for getting the model coefficients 
  (described below), `fitted()` for getting the fitted values (the sample 
  means in this case), and `residuals()` which returns the residuals from the 
  fit, which is the difference in the value of the observations from their 
  respective fitted values (the sample means). Whenever accessing a component 
  for which there is a specialized retrieval function, you should get into the 
  habit of using the specialized retrieval function instead of directly indexing 
  the element, since the raw element may need to be transformed in some way to 
  yield a readily interpretable result.

```
## F-test h0: all group means are equal;
##   dat$tension is group; dat$breaks is variable of interest:

fit <- aov(breaks ~ tension, data=dat)
class(fit)
is.list(fit)
names(fit)
attributes(fit)
fit

## can access directly:
fit$coeffients
fit$xlevels
fit$df.residual

## but best practice is to use accessor functions when available:
coef(rslt)
f <- fitted(rslt)
table(round(f, 5))                ## three means
r <- residuals(rslt)
r2 <- dat$breaks - fitted(rslt)   ## where residuals came from
table(r == r2)                    ## exact equality is often NOT what you want
all.equal(r, r2)                  ## right way to test for equality here

```

Here we show how to use `summary()` to get the confidence intervals and p-value
  we are interested in:

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

3) What is the null hypothesis of a 2-sample t-test?

4) What is the null hypothesis of a 1-factor ANOVA?

5) Should you try post-hoc tests without conducting an omnibus test first?

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
