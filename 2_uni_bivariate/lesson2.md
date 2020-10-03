# Fundamentals of computational data analysis using R
## Univariate and bivariate statistics: univariate statistics
#### Contact: mitch.kostich@jax.org

---

### Index

- [Lesson goals](#lesson-goals)
- [The Central Limit Theorem](#the-central-limit-theorem)
- [Estimating means with t-test](#estimating-means-with-t-test)
- [Comparing means with prior values](#comparing-means-with-prior-values)
- [Estimating proportions](#estimating-proportions)
- [Comparing proportions with prior values](#comparing-proportions-with-prior-values)

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)

---

### Lesson goals

1) Know how to calculate the mean, variance and standard deviation of a sample.

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
  Conversely, we can calculate a 'critical value' for any probability from this 
  distribution using the R function `qnorm()`. The probability of any sample
  mean having a value further from the population mean than the critical value
  is no more than the specified probability. This idea leads to the calculation
  of confidence intervals for the sample means. Here, we begin by calculating
  the conventional 'two-sided' confidence interval, where we consider the two
  possibilities that the sample mean is higher than the population mean and 
  also the possibility that the sample mean is lower than the population mean.
  This is the most common scenario. If we want to know where the sample mean
  will end up 95% of the time, we can split the remaining 5% (when the sample 
  mean falls outside the interval) evenly between the two 'tails' (left rising 
  tail and right falling tail) of the distribution. An approximation frequently
  employed is that about 95% of the time a normally distributed value will fall
  within two standard deviations of the mean. Therefore the sample mean should
  fall within two standard errors of the population mean about 95% of the time.
  We will calculate the 'critical values' more exactly here:

```

rm(list=ls())
## no random values drawn, so no set.seed() needed

x <- seq(from=60, to=140, by=0.01)
p <- dnorm(x, mean=100, sd=10)                     ## 'density' function for normal

length(x)                                          ## evenly spaced rising series of values
length(p)                                          ## one probability per value

plot(x=x, y=p, main="N(100, 10) distribution", xlab="value", ylab="probability", type="l", lty=2)
abline(v=100, lty=3)
abline(h=0, lty=3)

## a quantile function like 'qnorm' takes a probability as first argument, and returns
##   the value at that probability:

critical_95_left <- qnorm(0.025, mean=100, sd=10)   ## 'quantile' function for N(100, 10)
critical_95_right <- qnorm(0.975, mean=100, sd=10)  ## 1/2 of 5% == 0.025 on each side
critical_99_left <- qnorm(0.005, mean=100, sd=10)   ## 1/2 of 1% == 0.005 on left side
critical_99_right <- qnorm(0.995, mean=100, sd=10)  ## 1/2 of 1% == 0.005 on right side

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

[Return to index](#index)

---

### Estimating means with t-test

Above we learned that if our sample sizes are large enough to justify invoking 
  the CLT, we can use the *population* mean, *population* standard deviation 
  and *sample* size to calculate confidence intervals for the *sample* mean. 

However, in statistical inference we are usually interested in moving in the 
  opposite direction: we want to take a *sample* mean, *sample* standard deviation 
  and *sample* size in order to calculate a confidence interval for the *population* 
  mean. To do so, we plug in values of the sample mean and sample standard 
  deviation in place of their population counterparts. But because these sample
  parameters are inexact estimators of the corresponding population parameters,
  this introduces additional variability. In order to calculate the correct
  critical values for this situation, we need to use a 'relaxed' version of the 
  normal distribution, called the Student's t-distribution. The t-distribution
  describes the probability distribution of the statistic:

  `t = sample_mean / (sample_std_dev / sqrt(sample_size))`

This formula has the virtue of not requiring knowledge of any population 
  parameter. The t-distribution has a single parameter, called the 
  degrees-of-freedom, which is generally the number of samples `n` minus the 
  number of parameters being estimated. Here there are two population
  parameters being implicitly estimated: the population mean using the
  sample mean, and the population standard deviation, using the sample
  standard deviation. Therefore, the degrees-of-freedom, `df = n - 2`.
  We plot the 'standard' normal distribution `N(0, 1)` and t-distributions
  with various degrees-of-freedom for comparison:

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
  xlab='value', ylab='probability', type='n') 

lines(x=x, y=n, lty=1, col='black')
lines(x=x, y=t1, lty=2, col='cyan')
lines(x=x, y=t3, lty=3, col='orangered')
lines(x=x, y=t10, lty=2, col='magenta')
lines(x=x, y=t30, lty=3, col='cyan')
lines(x=x, y=t100, lty=2, col='orangered')

legend(
  'topright',
  legend=c("N(0,1)", "t(1)", "t(3)", "t(10)", "t(30)", "t(100)"),
  lty=c(1, 2, 3, 2, 3, 2),
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
  (based on the CLT and t-distribution) confidence intervals for the 
  population mean.

```
rm(list=ls())
set.seed(3)

x <- rnorm(30, mean=100, sd=10)     ## draw a sample of 10 from N(100, 10)

(rslt <- t.test(x))                 ## one-sample t-test

```

Internal structure and access:

```
names(rslt)
attributes(rslt)
str(rslt)
rslt$statistic

```

[Return to index](#index)

---


### Comparing means with prior values

Some stuff here.

```
some code here

```

[Return to index](#index)

---

### Check your understanding 1

1) some question here

[Return to index](#index)

---


### Estimating proportions

Some stuff here.

```
some code here

```

[Return to index](#index)

---

### Comparing proportions with prior values

Some stuff here.

```
some code here

```

[Return to index](#index)

---

### Check your understanding 2

1) some question here

[Return to index](#index)

---

## FIN!
