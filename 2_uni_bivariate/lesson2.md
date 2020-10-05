# Fundamentals of computational data analysis using R
## Univariate and bivariate statistics: univariate statistics
#### Contact: mitch.kostich@jax.org

---

### Index

- [Lesson goals](#lesson-goals)
- [The Central Limit Theorem](#the-central-limit-theorem)
- [Estimating means with a t-test](#estimating-means-with-a-t-test)
- [Comparing means with hypothetical values](#comparing-means-with-hypothetical-values)
- [Estimating proportions with a chi-square test](#estimating-proportions-with-a-chi-square-test)
- [Estimating proportions with exact tests](#estimating-proportions-with-exact-tests)

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

---

### Lesson goals

1) Follow the logic connecting the CLT assumption to confidence intervals for the 
   population mean based on a random sample drawn from that population. 

2) Be comfortable interpreting a 'frequentist' confidence interval or p-value.
   
3) Understand the concepts of a null hypothesis, the implications of rejecting
   a null hypothesis, as well as the difference between a one-sided test and 
   two-sided test.

4) Know how to parameterize the normal, uniform and t-distributions. Understand
   how the shape of the t-distribution compares to the normal distribution.

5) In R, be able to use the `t.test()` function to estimate a confidence interval
   for the population mean using a random sample drawn from that population. Be
   able to test null hypotheses that the population mean is the same as, above
   or below a hypothetical value.

6) Be able to use the `prop.test()` function to estimate a confidence interval for 
   a population proportion using a random sample. Be able to test null hypotheses
   that the null proportion is the same as, above or below a hypothetical value.

[Return to index](#index)

---

### The Central Limit Theorem

In our previous lesson, we used the means of samples to estimate the mean
  of a population (a simulated normal distribution or uniform distribution). 
  We learned that the sample estimates are centered on the population 
  mean, and are therefore unbiased estimators of the population mean. The
  precision of the sample estimates was described in terms of the 
  'standard error', which was directly proportional to the population 
  variance and inversely proportional to the square-root of the sample
  size:

`standard_error_mean = population_variance / square_root(sample_size)`

Furthermore, we learned that regardless of the distribution of the 
  population (e.g. for a uniformly distributed population), if the 
  sample estimates were made over and over, the distribution of sample
  estimates would be bell-shaped. The Central Limit Theorem (CLT) tells 
  us that for populations that are very large compared to the sample 
  size (whose composition would remain essentially unaffected by the 
  sampling process), as the sample sizes increases, this bell-shaped 
  distribution of sample estimates approaches a normal distribution with 
  a mean at the population mean and a standard deviation equal to the 
  standard error of the mean. That is, the distribution of sample estimates 
  of the population mean approaches:

`N(mean=population_mean, sd=standard_error_mean)`

In this lesson, we will discuss statistical inference, which means
  making inferences about population parameters based on measurements of 
  samples in a way that explicitly describes the uncertainty involved in
  extrapolating from samples to populations. If we use the CLT to justify 
  assuming a particular distribution for the sample estimates, we can then
  estimate how far from the population value any particular sample 
  estimate is likely to be. For instance, if the population has a mean
  of `100` and a standard deviation of `10`, according to the CLT, the 
  means of samples of size 30 will have the distribution:

`N(100, 10 / sqrt(30))`

Assuming this distribution for the sample means, we can calculate probability 
  of a sample mean having any particular value using the `dnorm()` function. 

```
rm(list=ls())
## no random values drawn, so no set.seed() needed

x <- seq(from=60, to=140, by=0.01)
p <- dnorm(x, mean=100, sd=10)                     ## 'density' function for normal

length(x)                                          ## evenly spaced rising series of values
length(p)                                          ## one probability per value

plot(x=x, y=p, main="N(100, 10) distribution", xlab="value", ylab="probability", type="l", lty=2)
abline(v=100, lty=3)                               ## line at population mean
abline(h=0, lty=3)                                 ## h(orizontal) line at y==0

```

Conversely, we can calculate a 'critical value' for any probability from this 
  distribution using the R function `qnorm()`. The probability of any sample
  mean having a value further from the population mean than the critical value
  is no more than the specified probability. This idea leads to the calculation
  of confidence intervals for the sample means. If we want to know where the 
  the mean of a random sample will end up 95% of the time, we can split the 
  remaining 5% (how often a sample mean is expected to fall outside the interval) 
  between the two 'tails' (left rising tail and right falling tail) of 
  the distribution. An approximation frequently employed is that about 95% of 
  the time a normally distributed value will fall within two standard deviations 
  of the mean. Therefore the sample mean should fall within two standard errors 
  of the population mean about 95% of the time. We will calculate the 'critical 
  values' more exactly here:

```
## a quantile function like 'qnorm' takes a probability as first argument, and returns
##   the value at that probability:

critical_95_left <- qnorm(0.025, mean=100, sd=10)   ## 'quantile' function for N(100, 10)
critical_95_right <- qnorm(0.975, mean=100, sd=10)  ## 1/2 of 5% == 0.025 on each side
critical_99_left <- qnorm(0.005, mean=100, sd=10)   ## 1/2 of 1% == 0.005 on left side
critical_99_right <- qnorm(0.995, mean=100, sd=10)  ## 1/2 of 1% == 0.005 on right side

## draw v(ertical) lines at the critical values:
abline(v=critical_95_left, lty=2, col='orangered')
abline(v=critical_95_right, lty=2, col='orangered')
abline(v=critical_99_left, lty=3, col='magenta')
abline(v=critical_99_right, lty=3, col='magenta')

legend('topright', legend=c('95% CI', '99% CI'), lty=c(2, 3), col=c('orangered', 'magenta'), cex=0.6)

c(critical_95_left, critical_95_right)              ## 95% chance sample mean btwn 80 and 120
c(critical_99_left, critical_99_right)              ## 99% chance between 74 and 126

## critical value -> p-value cutoff w/ 'pnorm()':

pnorm(c(critical_95_left, critical_95_right), mean=100, sd=10)
pnorm(c(critical_99_left, critical_99_right), mean=100, sd=10)

```

Invoking the CLT allows us to use a 'parametric' distribution (a distribution 
  defined by defining the parameters of a family of distribution, like 
  specifying the mean and standard deviation for a normal distribution family) to
  make various 'parametric' estimates of uncertainty, like the parametric 
  confidence intervals for the mean above. Because such estimates of uncertainty 
  asymptotically approach the 'truth' as the sample size approaches infinity, 
  these estimates are sometimes termed 'asymptotically correct'. 

How large a sample does one require in order to be able to invoke the CLT to 
  justify using parametric confidence intervals for a mean? The speed with 
  which the sample estimate distributions approach their theoretical parametric
  values as sample size increases depends on the population distribution, so 
  can depend from case-to-case. If the sample size is too low, the 'coverage' 
  of confidence intervals will tend to be lower than 'nominal'. That is, a 
  parametric confidence interval that is supposed to be 95% (nominally 95%) 
  may actually correspond to only a 91.3% interval. So the right answer depends 
  both on the shape of the population (the closer to normal, the better the 
  parametric confidence intervals will tend to cover the nominal interval) and 
  how good you need the coverage to actually be. If the main focus of a study is
  a particular mean, you should ensure you have robust sample sizes. If you
  are using parametric estimates to filter very large numbers of largely redundant 
  variables into a smaller set to use in a machine learning procedure that can
  accommodate some extraneous variables, the exact coverage of the confidence 
  interval might be less important than computationally efficiency of the filtering. 
  With small samples, bootstrapping (a non-parametric 'resampling' procedure 
  covered later in this course) can often provide confidence intervals with 
  better coverage than those based on parametric methods. Bootstrapping also 
  provides good opportunities to evaluate how well the CLT assumption holds in 
  a particular situation. 


[Return to index](#index)

---

### Estimating means with a t-test

Above we learned that if our sample sizes are large enough to justify invoking 
  the CLT, we can use the **population** mean, **population** standard deviation 
  and **sample** size to calculate confidence intervals for the **sample** mean. 

However, in statistical inference we are usually interested in moving in the 
  opposite direction: we want to take a **sample** mean, **sample** standard deviation 
  and **sample** size in order to calculate a confidence interval for the **population**
  mean. To do so, we plug in values of the sample mean and sample standard 
  deviation in place of their population counterparts. But because these sample
  parameters are inexact estimators of the corresponding population parameters,
  this introduces additional variability. In order to calculate the correct
  critical values for this situation, we need to use a 'relaxed' version of the 
  normal distribution, called the Student's t-distribution. The t-distribution
  describes the probability distribution of the statistic:

`t = sample_mean / (sample_std_dev / sqrt(sample_size))`

The `sample_std_dev` is calculated using the sample for data, but the formula
  used continues to be the population version (without Bessel's correction of 
  the denominator). The t-statistic has the virtue of having a distribution 
  that does not require knowledge of any population parameter (such as the 
  mean or standard deviation). The t-distribution has a single parameter, 
  called the degrees-of-freedom `df`. The degrees-of-freedom is the difference 
  between the size of the sample `n` and the number of population parameters being 
  estimated. Here, we are estimating the mean of a single population, so one 
  population parameter is being estimated, therefore `df <- n - 1`. Below we plot the 
  'standard' normal distribution `N(0, 1)` and t-distributions with various 
  degrees-of-freedom for comparison:

```
rm(list=ls())

x <- seq(from=-5, to=5, by=0.01)
n <- dnorm(x, mean=0, sd=1)
t1 <- dt(x, df=1)
t3 <- dt(x, df=3)
t10 <- dt(x, df=10)
t30 <- dt(x, df=30)
t100 <- dt(x, df=100)

plot(x=range(x), y=range(c(t1, t3, t10, t30, t100)),
  xlab='value (in std devs)', ylab='probability', type='n') 

lines(x=x, y=n, lty=1, col='black')
lines(x=x, y=t1, lty=2, col='cyan')
lines(x=x, y=t3, lty=3, col='orangered')
lines(x=x, y=t10, lty=4, col='magenta')
lines(x=x, y=t30, lty=3, col='cyan')
lines(x=x, y=t100, lty=2, col='orangered')

cv95_n_left <- qnorm(0.025, mean=0, sd=1)
cv95_n_right <- qnorm(0.975, mean=0, sd=1)
cv95_t10_left <- qt(0.025, df=10)
cv95_t10_right <- qt(0.975, df=10)

abline(v=c(cv95_n_left, cv95_n_right), col='black', lty=1)
abline(v=c(cv95_t10_left, cv95_t10_right), col='magenta', lty=4)

legend(
  'topright',
  legend=c("N(0,1)", "t(1)", "t(3)", "t(10)", "t(30)", "t(100)"),
  lty=c(1, 2, 3, 4, 3, 2),
  col=c('black', 'cyan', 'orangered', 'magenta', 'cyan', 'orangered')
)

```

The t-distribution is generally more 'dispersed' (more values tend to 
  occur further from 0) than the standard normal distribution `N(0, 1)`. 
  As the degrees-of-freedom increases, the t-distribution becomes 
  less dispersed, asymptotically approaching the standard normal
  as the degrees-of-freedom approaches infinity.

Fortunately for the non-specialist, R provides the `t.test()` function,
  which performs all the necessary computations we need in order to
  use a sample to estimate a population mean, and calculate theoretical
  (based on the CLT assumption and t-distribution) confidence intervals 
  for the population mean.

Unfortunately for the non-specialist, R provides the `t.test()` function, 
  which allows one to very easily learn to perform a t-test without ever
  investigating the assumptions and limitations associated with the test.

```
rm(list=ls())
set.seed(3)

## 'population' (naive) formula for sd of v; remember, R's 'sd()' 
##   function uses the 'sample' formula, with Bessel's correction:

f.sd.pop <- function(v) {
  d <- v - mean(v)
  s2 <- sum(d ^ 2) / length(v)      ## population variance
  sqrt(s2)
}

x <- rnorm(30, mean=100, sd=10)     ## draw a sample of 10 from N(100, 10)

rslt <- t.test(x)                   ## THIS IS THE ONLY THING YOU NEED!!!
rslt

names(rslt)

rslt$estimate                       ## '$' convenient way to get components
(m <- mean(x))

rslt$parameter
(dof <- length(x) - 1)

rslt$stderr
(se <- f.sd.pop(x) / sqrt(dof))

rslt$statistic
(stat <- m / se)

rslt$conf.int
(ci.lo <- m + qt(0.025, dof) * se)
(ci.hi <- m - qt(0.025, dof) * se)

## what's under the hood:

class(rslt)                        ## a h(ypothesis)test
attributes(rslt)                   ## just 'names' + 'class'

```

The confidence interval constructed above was derived by considering the distribution
  of the sample mean if samples were drawn from the population over and over 
  (without affecting the composition of the population). The resulting 95% confidence
  interval has a similarly structured interpretation: if many, many (approaching 
  infinity) samples of the given size (above n=30 observations per sample) were drawn 
  at random from the population, and used to estimate the confidence interval for 
  that population, 95% of the time, the confidence intervals would contain the true 
  population mean.

[Return to index](#index)

---

### Check your understanding 1

Initialize your variables as follows (you can copy and paste, if you like):

```
rm(list=ls())
set.seed(10)
x <- rnorm(30, 0, 10)

```

1) Use a one-sample t-test to use `x` to estimate the mean and 95% confidence 
   interval for the mean of the population from which `x` was drawn. 

2) What is the standard error of the estimate of the mean? Hint: stored in the result 
   of the t-test in (1). 

3) About how large should a sample size be in order for you to be able to assume
   that the distribution of sample means is normal? Assume you don't know 
   anything about the shape of the population distribution.

4) About what percentage of values fall within two standard deviations of the 
   mean of a normal distribution?

5) The dispersion of the t-distribution is [select 1: smaller or larger] than the N(0, 1) distribution?

6) Increasing sample size [increases or decreases] degrees-of-freedom.

7) Increasing degrees-of-freedom [increases or decreases] how precisely the sample
   mean tends to estimate the population mean. 

[Return to index](#index)

---

### Comparing means with hypothetical values

Above we used the `t.test()` function to estimate a confidence interval for a population 
  mean based on a random sample drawn from the population. You may have heard of the Student's 
  t-test before, and know that it is used for testing hypotheses about means. Above, 
  the default invocation of `t.test()`, in addition to producing a confidence interval 
  for the population mean, also tests the 'null hypothesis' that the population mean is not 
  different from a default comparison value of zero. In general, a null hypothesis 'h0' 
  usually corresponds to a negative result of some type: no difference, no association, 
  etc. We 'test' a null hypothesis about a population by seeing if the null hypothesis is 
  consistent with data collected from a random sample of that population. If our 
  observed sample data would be very unlikely under the null hypothesis (signaled 
  by the test returning a 'p-value' below some preselected cutoff, like 5%) we can 'reject' 
  the null hypothesis. This would typically correspond to a positive or other interesting
  result of some type. In some cases, rejecting the null hypothesis leaves us with only
  one possible alternative hypothesis, but in other cases, rejecting the null hypothesis
  leaves multiple remaining alternative hypotheses, which can complicate interpretation.
  For instance, if the null hypothesis that the mean of a population is zero is rejected,
  that does not tell us if the actual mean is above or below zero, let alone what value
  the mean actually takes. In this sense, reporting confidence intervals is usually far 
  more informative to your audience than reporting a p-value from a statistical test.

In the case above, if the null hypothesis were true (population mean really was `0`), 
  on average, only 5 out of 100 samples drawn from that population would result in 95% 
  confidence intervals that did not include `0`. This logic is used to justify our 
  rejection of the null hypothesis at a p-value (0.05) corresponding to the confidence 
  interval (95%).

Although that is a legitimate way to view the t-test, the actual p-value is calculated
  backwards: instead of comparing the hypothetical value to the critical values for
  a confidence interval with a well defined nominal coverage (e.g. 95%), we ask what 
  is the percent coverage of an interval which ended at the hypothetical value. That is,
  if the hypothetical mean fell exactly at the 95% confidence interval upper bound,
  the p-value would be 0.05. If it fell exactly at the 99% confidence interval lower 
  bound, the p-value would be 0.01. In fact, the p-value returned might be something
  more extreme, like 0.00031. This would correspond to the hypothetical value falling
  just outside a 99.97% confidence interval, a cutoff that probably never occurred to
  us when specifying nominal confidence interval coverage prior to commencing the 
  experiment (you should select confidence interval coverage, null hypotheses, and 
  p-value cutoffs for rejecting null hypotheses prior to looking at the data). The 
  point here is that the p-value is calculated using the same rationale as the 
  confidence interval, but the details differ in a way that allows the p-value to 
  take on a continuum of values between `0` and `1`, rather than simply being one of 
  the two choices 'greater than or equal to 0.05' or 'less than 0.05'.

The interpretation of the p-value is similar to our earlier interpretation of the
  confidence interval. If the experiment were repeated many, many times, each time 
  drawing a random sample of the given size from the same population (without
  affecting the composition of the population) which had the hypothetical mean
  (here `0`), the sample mean would only be as far from the hypothetical mean as 
  was observed with our experimental sample `p-value * 100` percent of the time.
  This way of deriving and interpreting p-values or confidence intervals in terms 
  of the theoretical frequency with which statistics calculated on different 
  random samples from a population would take on a particular range of values is 
  called the 'frequentist' approach to statistics.

The R `t.test()` function takes a parameter called `mu` (defaults to `0`), which can 
  be used to set the hypothetical value for the null hypothesis to whatever value you 
  want. It also takes the parameter `conf.level` (defaults to `0.95`) which can be
  used to calculate confidence bounds at a user-selected level of nominal coverage:

```
rm(list=ls())
set.seed(101)

x <- rnorm(30, mean=10, sd=2)

(rslt <- t.test(x, mu=10))
rslt$conf.int
rslt$p.value

(rslt <- t.test(x, mu=rslt$conf.int[1]))
rslt$conf.int
rslt$p.value

(rslt <- t.test(x, mu=rslt$conf.int[1]-0.01))
rslt$conf.int
rslt$p.value

(rslt <- t.test(x, mu=rslt$conf.int[1]+0.01))
rslt$conf.int
rslt$p.value

(rslt <- t.test(x, conf.level=0.99))
rslt$conf.int
rslt$p.value

(rslt <- t.test(x, mu=rslt$conf.int[2]))
rslt$conf.int
rslt$p.value

```

When comparing a hypothetical value to a population mean with `t.test()`, we 
  can also perform 'one-sided' tests, that only consider the possibility of 
  the population mean being smaller or larger than a hypothetical value.
  For instance, we may want to know if the average US adult would fit into
  a doorway 5'8" (68") tall. In order to answer this question, we could 
  use a random sample from the US population to test the null hypothesis
  that the mean height of the US population was less than or equal to 
  68". Rejecting this null hypothesis would suggest we need to construct
  our doorways taller, but not tell us by how much. To reiterate the 
  earlier vital point: the additional information provided by a 
  confidence interval on the mean population height would give us a much 
  better idea of how to adjust the door height. **Whenever possible, 
  report confidence intervals in addition to or in place of p-values.**

The advantage of using a one-sided p-value instead of the default 
  two-sided p-value is that the one-side test is more powerful: it can
  detect smaller differences between the population mean and the 
  hypothetical comparison value than the two-sided test. Therefore,
  if your interest really only lies in a 'greater' or 'less than' 
  comparison, you should use a one-sided test.
  

```
rm(list=ls())
set.seed(101)

x <- rnorm(30, mean=10, sd=2)

## get two-sided 95% CI:
(rslt <- t.test(x))
rslt$conf.int
(ci.lo <- rslt$conf.int[1])
(ci.hi <- rslt$conf.int[2])

## just inside lower bound of 2-sided:
(rslt <- t.test(x, mu=ci.lo+0.01))
rslt$conf.int
rslt$p.value

## 1-sided CI 'tighter' on 1 side, unconstrained on other;
##   1-sided test p-value more 'powerful':

(rslt <- t.test(x, mu=ci.lo+0.01, alternative='greater'))
rslt$conf.int
rslt$p.value
rslt$p.value * 2

## just inside upper bound of 2-sided:
(rslt <- t.test(x, mu=ci.hi-0.01))
rslt$conf.int
rslt$p.value

## 1-sided CI 'tighter' on 1 side, unconstrained on other;
##   1-sided test p-value more 'powerful':

(rslt <- t.test(x, mu=ci.hi-0.01, alternative='less'))
rslt$conf.int
rslt$p.value
rslt$p.value * 2

```

The p-value tells us how often we expect the population mean to fall 
  outside of the confidence interval calculated based on a random sample.
  That is, at a p-value cutoff of 0.05, we expect that the null hypothesis 
  will be rejected, even if true, about 5% of the time.

In the example below, remember `isTRUE(x)` is the same as:

`is.logical(x) && length(x) == 1 && !is.na(x) && x`

```
rm(list=ls())
set.seed(101)

f <- function(n=30, m=0, s=1, conf.level=0.95, R=1e4) {

  o <- numeric(0)                          ## will hold R results 

  for(i in 1 : R) {                        ## perform R experiments
    x.i <- rnorm(n, mean=m, sd=s)          ## draw random sample of size n from N(m, s)
    rslt.i <- t.test(x.i, conf.level=conf.level)
    ci.lo <- rslt.i$conf.int[1]            ## lower CI bound
    ci.hi <- rslt.i$conf.int[2]            ## upper CI bound
    
    o.i <- F                               ## if pop mean inside CI, o.i=F, else o.i=T
    if(isTRUE(m < ci.lo) || isTRUE(m > ci.hi)) o.i <- T
    o <- c(o, o.i)                         ## store (TRUE/FALSE) result of this experiment
  }

  sum(o) / R                               ## proportion of experiments w/ pop mean outside CI?
}

for(i in 1 : 10) print(f(n=3))             ## try sample size 3
for(i in 1 : 10) print(f(n=10))            ## sample size 10
for(i in 1 : 10) print(f(n=30))            ## sample size 30
for(i in 1 : 10) print(f(n=100))           ## sample size 100
for(i in 1 : 10) print(f(m=-100))          ## at a different mean
for(i in 1 : 10) print(f(s=1000))          ## at a different sd
for(i in 1 : 10) print(f(conf.level=0.90)) ## different CI coverage

```

[Return to index](#index)

---

### Check your understanding 2

Set up your test data as follows:

```
rm(list=ls())
set.seed(101)

x.u <- runif(30, min=0, max=100)
x.r <- rnorm(50, mean=40, sd=5)

t.test(x.u)
t.test(x.r)

```

1) Using the observations in sample `x.u`, generate a 95% two-sided confidence interval 
   for the mean of the population from which sample `x.u` was randomly drawn. What 
   is the estimate of the population mean? What is the standard error of that estimate?

2) Using the observations in sample `x.r` generate a 95% two-sided confidence interval
   for the mean of the population from which sample `x.r` was randomly drawn. What is
   the estimate of the population mean? What is the standard error of that estimate?

3) What is the p-value from the t-test of the null hypothesis that the mean of the 
   population from which `x.r` was drawn is equal to 45?

4) What is the p-value for the null hypothesis that the mean of the population from
   which `x.r` was drawn is greater than 50?

[Return to index](#index)

---


### Estimating proportions with a chi-square test

In addition to making inferences about numeric data, we are often interested in 
  making inferences about categorical data. In the simplest setup, every member
  of a population is assigned to one category from a set of mutually exclusive 
  categories and we are interested in a question such as the proportion of the 
  population in each category. The most common and easily understood scenario
  is when a population is split between two categories. For instance, we might 
  be interested in the proportion of the population that bears a particular 
  genetic allele ('carrier' vs 'non-carrier') or is affected by a viral disease 
  ('infected' vs 'not infected'), or has a sufficient titer of neutralizing 
  antibodies against a virus ('immune' vs 'not immune'). 

A nice 'toy' example is determining if a coin is 'fair'. When flipping coins,
  we expect a fair coin to have an equal probability of landing on a head 
  or a tail with each flip. That is, the probability of getting heads on 
  one flip should be 0.5. This can be restated in the frequentist manner as 
  the expectation that if we started flipping the coin over and over again,
  tabulating outcomes along the way, as the number of flips approaches infinity,
  we expect the proportion of heads in our tally to get closer and closer to 
  0.5. We can test if the coin is fair by using a sample of coin flips to 
  generate a confidence interval on the proportion of heads in the population
  of potential coin flips (including future flips, assuming the coin is not
  tampered with):

```
rm(list=ls())
set.seed(101)

## set up a fair coin and flip it 30 times:

n <- 30                      ## sample size: number of coin flips in experiment
p <- 0.5                     ## the probability of heads (the coin is fair)

(i <- rbinom(n, 1, p))       ## draw 0s (tails) and 1s (heads) from 'population'
table(i)

i[i]
i[!i]

## estimate a 95% confidence interval for the proportion of 
##   'tails' in the population of potential flips of the coin 
##   being tested.

table(i)
rslt <- prop.test(table(i))
rslt
rslt$conf.int                ## 95% CI for proportion of tails (includes 0.5, so 'fair')
rslt$estimate                ## the estimated proportion of tails in the population
sum(!i) / length(i)          ## is pretty easy to estimate directly

## beneath the hood:

class(rslt)                  ## h(ypothesis)test, just like value of t.test()
names(rslt)                  ## lots of familiar (like t.test()) parameters
attributes(rslt)             ## same setup as for t.test()

```

As with the `t.test()`, function, the `prop.test()` function uses a parametric 
  distribution (chi-square distribution in this case) to construct 
  confidence intervals for a population parameter and test null hypotheses 
  about the population parameter. In order to test whether the coin is fair,
  we can use a sample of flips to test the null hypothesis that the proportion
  of heads in the population of potential flips is 0.5. If the sample data
  leads us to reject this hypothesis (p-value returned is less than
  a preselected cutoff like 0.05), we can conclude that the coin is not 
  fair. We could also construct a one-sided test to see if the probability of 
  a head exceeds 0.5 or a one-sided test to see if the probability of a head
  is less than 0.5. Once again, reporting a confidence interval gives us even
  more information about the probability of heads. Therefore, if the
  probability of heads is really of interest, the confidence interval should
  accompany, or be reported in place of, the p-value from the corresponding
  hypothesis test.

```
rm(list=ls())
set.seed(101)

## set up and examine the dataset:

n <- 30                      ## sample size
p <- 0.25                    ## proportion of 'green' in the population

(i <- rbinom(n, 1, p))       ## draw 0s (red) and 1s (green) from population
table(i)

(i <- as.logical(i))         ## 0 -> FALSE; 1 -> TRUE
table(i)

(x <- rep('red', n))         ## initialize all red sample of size n
table(x)

x[i] <- 'green'              ## 1 -> TRUE -> 'red'; 0 -> FALSE -> left 'green'
x
table(x)

x[i]
x[!i]

## estimate a 95% confidence interval for the proportion of 
##   'green' in the population, and test the null hypothesis that the 
##   proportion of 'green' in the population is 0.75.

table(x)
rslt <- prop.test(table(x), p=0.75)
rslt
rslt$estimate                ## estimate of proportion of green in population
rslt$conf.int                ## 95% CI does not include null hypothesis (0.75)
rslt$p.value                 ## so p-value is very low (reject null hypothesis)


## can also do a 1-sided tests, such as a test of the null hypothesis that the 
##   proportion of 'green' in the population is less than 0.4:

rslt <- prop.test(table(x), p=0.4, alternative='less')
rslt
rslt$conf.int                ## proportions constrained between 0 and 1 in any case
rslt$p.value

```

[Return to index](#index)

---

### Estimating proportions with exact tests

The chi-square test in the previous example calculates p-values and confidence 
  intervals based on the parametric chi-square distribution. This calculation 
  makes several assumptions, including that sample sizes are large enough to
  invoke the CLT and that cell counts are large enough (a rough rule of thumb 
  is that your sample should be large enough to provide counts of at least five 
  for each category) for other distributional assumptions to hold. As was 
  mentioned in the context of the t-test, bootstrapping methods provide one 
  approach for deriving confidence intervals without making distributional 
  assumptions. However, when working with proportions, several simpler 
  non-parametric alternatives to the parametric chi-square test are frequently 
  employed. 

These non-parametric exact tests have the virtue of returning confidence intervals 
  guaranteed to be correct regardless of sample size or cell counts. The 
  downside of using these tests is that when the assumptions behind the 
  chi-square test are met, the confidence intervals provided by the chi-square 
  test are often shorter than the corresponding intervals from the 
  non-parametric tests. This means that under those conditions the chi-square 
  test is more powerful than its non-parametric counterparts. Also, the time 
  required for calculations of the exact tests rises with the size of 
  the sample, which can result in prohibitive computational costs for very 
  large samples. Here we demonstrate the popular (and fairly easy to interpret)
  binomial exact test:

```
rm(list=ls())
set.seed(101)

## set up a fair coin and flip it 30 times:

n <- 30                      ## sample size: number of coin flips in experiment
p <- 0.5                     ## the probability of heads (the coin is fair)

(i <- rbinom(n, 1, p))       ## draw 0s (tails) and 1s (heads) from 'population'
table(i)

i[i]
i[!i]

## estimate a 95% confidence interval for the proportion of 
##   'tails' in the population of potential flips of the coin 
##   being tested.

table(i)
rslt.chi <- prop.test(table(i))   ## chi-square test (parametric, approximate)
rslt.chi

rslt.bin <- binom.test(table(i))  ## binomial test ('exact' non-parametric)
rslt.bin

ci.chi <- rslt.chi$conf.int
ci.bin <- rslt.bin$conf.int
ci.chi[2] - ci.chi[1]             ## a bit tighter (more powerful) than non-parametric
ci.bin[2] - ci.bin[1]             ## a bit looser (less powerful) than parametric

rslt.chi$estimate
rslt.bin$estimate 
sum(!i) / length(i) 

## beneath the hood:

class(rslt.bin)                   ## h(ypothesis)test, like t.test() and prop.test()
names(rslt.bin)                   ## lots of familiar (like t.test() and prop.test()) parameters
attributes(rslt.bin)              ## same setup as for t.test() and prop.test()

```

[Return to index](#index)

---

### Check your understanding 3

Set up your data as follows:

```
rm(list=ls())
set.seed(101)

x10a <- rbinom(30, 1, 0.1)
x50a <- rbinom(30, 1, 0.5)
x90a <- rbinom(30, 1, 0.9)

x10b <- rbinom(300, 1, 0.1)
x50b <- rbinom(300, 1, 0.5)
x90b <- rbinom(300, 1, 0.9)

```

1) For each of the random samples `x10a`, `x50a`, `x90a`, `x10b`, `x50b`, and `x90b`:
   use `prop.test()` to estimate a 95% confidence interval for the proportion of 
   0s in the population from which the sample was randomly drawn. Retrieve the 
   estimate and the standard error of the estimate as well.

2) How did changing sample size change the corresponding confidence interval width?

3) Are the estimates centered within their respective confidence intervals?

4) Use the `prop.test()` to test the null hypothesis that the proportion of 1s 
   (not 0s!!!) in the population from which x10a was drawn is equal to 0.4. What is 
   the p-value?

5) Use the `prop.test()` to test the null hypothesis that the proportion of 1s 
   (not 0s!!!) in the population from which x10a was drawn is less than 0.4. What is 
   the p-value?

[Return to index](#index)

---

## FIN!
