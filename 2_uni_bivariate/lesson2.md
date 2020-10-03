# Fundamentals of computational data analysis using R
## Univariate and bivariate statistics: univariate statistics
#### Contact: mitch.kostich@jax.org

---

### Index

- [Lesson goals](#lesson-goals)
- [The Central Limit Theorem](#the-central-limit-theorem)
- [Estimating means with a t-test](#estimating-means-with-a-t-test)
- [Comparing means with prior values](#comparing-means-with-hypothetical-values)
- [Estimating proportions](#estimating-proportions)
- [Comparing proportions with prior values](#comparing-proportions-with-hypothetical-values)

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

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

Invoking the CLT allows us to use a 'parametric' distribution (a distribution 
  defined by defining the parameters of a family of distribution, like 
  specifying the mean and standard deviation for a normal distribution) to
  make various 'parametric' estimates of uncertainty, like the parametric 
  confidence intervals for the mean above. Because such estimates of uncertainty 
  asymptotically approach the 'truth' as the sample size approaches infinity, 
  these estimates are sometimes termed 'asymptotically correct'. 

How large a sample does one require in order to be able to invoke the CLT to 
  justify using parametric confidence intervals for a mean? The speed with 
  which the sample estimate distributions approach their theoretical parametric
  values as sample size increases depends on the population distribution. If the
  sample size is too low, the 'coverage' of confidence intervals will tend to
  be lower than 'nominal'. That is, a parametric confidence interval that 
  is supposed to be 95% (nominally 95%) may actually only tend to capture 
  means of samples about 91% of the time. So the right answer depends both
  on the shape of the population (the closer to normal, the better the 
  parametric confidence intervals will cover the nominal interval) and how
  good you need the coverage to actually be. If the main focus of a study is
  a particular mean, you should ensure you have robust sample sizes. If you
  are using parametric estimates to filter large numbers of largely redundant 
  variables into a smaller set to use in a machine learning procedure, the
  exact coverage of the confidence interval might be less important than 
  getting a quick answer. With small samples, bootstrapping (a non-parametric
  'resampling' procedure covered later in this course) can often provide 
  confidence intervals with better coverage than those based on parametric 
  methods. Bootstrapping also provides good opportunities to evaluate how 
  well the CLT assumption holds. 


[Return to index](#index)

---

### Estimating means with a t-test

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

The `sample_std_dev` is calculated using the sample for data, but the formula
  used continues to be the population version (without Bessel's correction of 
  the denominator). The t-statistic calculation has the virtue of not requiring 
  knowledge of any population parameter. The t-distribution has a single 
  parameter, called the degrees-of-freedom `df`. The degrees-of-freedom 
  is the difference between the size of the sample `n` and the number of 
  population parameters being estimated. Here, we are estimating the mean of a 
  single population, so one population parameter is being estimated, so 
  `df <- n - 1`. Below we plot the 'standard' normal distribution `N(0, 1)` 
  and t-distributions with various degrees-of-freedom for comparison:

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
  investigating the assumptions and limitations behind the test.

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

rslt$estimate
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

2) What is the standard error of the estimate of the mean? Hint: index the result 
   of the t-test in (1). 

3) About how large should a sample size be in order for you to be able to assume
   that the distribution of sample means is normal? Assume you don't know 
   anything about the shape of the population distribution.

4) About what percentage of values fall within two standard deviations of the 
   mean of a normal distribution?

5) The dispersion of the t-distribution is [smaller or larger] than the N(0, 1) distribution?

6) Increasing sample size [increases or decreases] degrees-of-freedom.

7) Increasing degrees-of-freedom [increases or decreases] how precisely the sample
   mean tends to estimate the population mean. 

[Return to index](#index)

---


### Comparing means with hypothetical values

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


### Estimating proportions

Some stuff here.

```
some code here

```

[Return to index](#index)

---

### Comparing proportions with hypothetical values

Some stuff here.

```
some code here

```

[Return to index](#index)

---

### Check your understanding 3

1) some question here

[Return to index](#index)

---

## FIN!
