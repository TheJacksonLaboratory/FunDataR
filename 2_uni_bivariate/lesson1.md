# Fundamentals of computational data analysis using R
## Univariate and bivariate statistics: bias and standard error
#### Contact: mitch.kostich@jax.org

---

### Index

- [Lesson goals](#lesson-goals)
- [What is a mean?](#what-is-a-mean)
- [Populations and samples](#populations-and-samples)
- [Variances and standard deviations](#variances-and-standard-deviations)
- [Standard errors and bias](#standard-errors-and-bias)
- [Distribution of estimates of the mean](#distribution-of-estimates-of-the-mean)

- [Check 1](#check-your-understanding-1)

### Lesson goals:

1) Know how to calculate the mean, variance and standard deviation of a sample.

2) Have feel for the shape of the normal (aka Gaussian) distribution and the
   uniform distribution. Know how to draw random samples from each.

3) Learn how to do basic histograms and 2D plots; get a feel for how changing
   the number of histogram bins affects the output.

3) Understand the difference between a population statistic (e.g. mean or 
   standard deviation) and the corresponding sample statistic.

4) Understand the two components of estimate accuracy: standard error and bias.

5) Get a feel for how increasing sample sizes changes the accuracy of 
   population parameter estimates.

[Return to index](#index)

---


### What is a mean?

You are probably familiar with the notion of the 'mean' or 'average' of a 
  series of numbers as a type of central value (or 'central tendency') for 
  the numbers in the series. But you've probably also heard of the 'median' 
  and may know that it too, is a type of 'central tendency'. You may know the 
  difference between the two in terms of the procedures for calculating their 
  values. The median of the series `x` would be found by sorting `x` then 
  taking the middle value (or the mean of the two central values, if `x` has 
  even length). By contrast, means are calculated using the formula 
  (expressed in R): 

`sum(x) / length(x)`

R also provides the predefined function `mean(x)` for this purpose. Unlike
  our R expression above, the function `mean()` is compiled byte code, the 
  execution of which can often result in faster execution than using an R 
  expression like the one above.

Let's take 1000 random numbers from a normal distribution,
  calculate their mean and plot the results. In this case (plotting a 
  single variable `z`), the horizontal/bottom axis indicates the order 
  in which the numbers occur in `z`. The vertical/left axis indicates
  the magnitude of the numbers in `z`. Since the numbers were drawn 
  at random, we do not expect any relationship between the magnitudes
  (positions along the vertical axis) and order in which numbers were drawn 
  (positions along the horizontal axis):

```
rm(list=ls())

set.seed(1)
str(z <- rnorm(1000, mean=100, sd=10))
mean(z)

## let's see what this distribution looks like:
hist(z)                               ## peaked toward center, much less in tails

## let's make a 2D plot (values on vertical axis, order on horizontal):
plot(z)                               ## 2D-plot; values cluster towards mean
abline(h=mean(z), col='cyan', lty=2)  ## add a h(orizontal) line at mean(z)

```

Now let's repeat the same, but drawing from a uniform distribution in the 
  numeric (includes fractional numbers) closed (includes endpoints) 
  interval [90, 110]. We'll often sample from a uniform distribution, so 
  it is good to get a feel for it:

```
rm(list=ls())

set.seed(1)
str(z <- runif(1000, min=80, max=120))
mean(z)

## let's see what this distribution looks like:
hist(z)                           ## histogram

## let's make a 2D plot (values on vertical axis, order on horizontal):
plot(z)                           ## 2D-plot
abline(h=mean(z), col='cyan')     ## add a h(orizontal) line at mean(z)

```

One distinctive (and conceptually very important) property of a 'mean' 
  which you may not be as familiar with is that the sum of the squared 
  distances between the elements of `x` and `mean(x)` is smaller than 
  it is for any other single number. In that sense, `mean(x)` mimimizes 
  a 'penalty function', which is the sum of squared distances between 
  `mean(x)` and the individual values of `x`.

Let's explore this a bit:

```
rm(list=ls())

## f.ss() returns sum of squared distances of points in v from m.
##   v should be a numeric vector; assumes no NaN or NA values
##   m should be a numeric vector of length 1 or of length length(v)
##   return value is numeric of length 1, or NA on error
##
##   Could just: 'return(sum((v - m) ^ 2))', but we'll break it
##     out for clarity and add a length check in order to improve 
##     robustness to input errors (by preventing unexpected recycling).

f.ss <- function(v, m=0) {

  ## length check: does length(m) equal 1 or length(v)?
  ##   if not, 'stop' execution and return an informative error msg:

  if( (length(m) != 1)  && (length(m) != length(v)) ) {
    stop("length(m) not 1 or length(v)")
  }

  dist <- v - m                    ## distances between v and m; length(dist) == length(v)
  dist.sqr <- dist ^ 2             ## distances squared; length(dist.sqr) == length(v)
  sum.sqr <- sum(dist.sqr)         ## sum of the squared distances; length(sum.sqr) = 1
  return(sum.sqr)                  ## explicit return of sum of squared distances
}

## sanity check f.ss(); essential practice for all your functions!
f.ss(0)                            ## default m==0; (0 - 0)^2 == 0
f.ss(1)                            ## (1 - 0)^2 == 1
f.ss(2)                            ## (2 - 0)^2 == 4
f.ss(0, 0)                         ## (0 - 0)^2 == 0
f.ss(0, 1)                         ## (0 - 1)^2 == 1
f.ss(1, 1)                         ## (1 - 1)^2 == 0
f.ss(0, c(0, 1))                   ## ok: complain if length(m) not 1 or length(v)
f.ss(c(1, 0), 0)                   ## (1 - 0)^2 + (0 - 0)^2 == 1
f.ss(c(1, 0), c(0, 1))             ## (1 - 0)^2 + (0 - 1)^2 == 2
f.ss(c(2, 0), c(0, 2))             ## (2 - 0)^2 + (0 - 2)^2 == 8

## time for an 'experiment':
set.seed(1)
(z <- rnorm(150, mean=100, sd=10))
hist(z, breaks=20)
mean(z)
f.ss(z, mean(z))
f.ss(z, mean(z) + 1)               ## penalty goes up as we move away from mean 
f.ss(z, mean(z) - 1)               ## penalty also rises in other direction
f.ss(z, 0)                         ## worse as you get further?
f.ss(z, 200)                       ## seems like it in this direction?

## f.ss() minimized by mean() even for weird distributions like this
##   3 peaked mixture of 2 uniforms and one normal:

z <- c(
  rnorm(150, mean=100, sd=10),     ## 150 draws from N(100, 10)
  runif(100, min=50, max=75),      ## 100 draws from uniform on interval [50, 75]
  runif(50, min=125, max=150)      ## 50 draws from uniform on interval [125, 150]
)
z
hist(z, breaks=20)                 ## 3-peaks, normal flanked by two uniforms

mean(z)
f.ss(z, mean(z))
f.ss(z, mean(z) + 1)               ## penalty goes up as we move away from mean
f.ss(z, mean(z) - 1)               ## same thing in other direction
f.ss(z, 0)                         ## worse as you get further?
f.ss(z, 200)                       ## seems like it in this direction?

## we can get a more comprehensive view graphically:

f <- function(a, b) f.ss(b, a)     ## flip the order of args so works with 'sapply()'
m <- seq(from=1, to=200, by=0.01)  ## values to try for 'm'; will be 'x' axis
str(m)
penalty <- sapply(m, f, z)         ## calculate the penalty at each value of 'm'; 'y' axis
plot(x=m, y=penalty)               ## single minimum (no local minima), 'convex' shape
abline(v=mean(z), col='cyan')      ## v(ertical) line where m=mean(z)

## or computationally; the 'which()' function returns the
##   integer index of every TRUE value in a logical vector:

tmp <- c(F, T, F, F, T, F, T)      ## a logical vector for demo purposes
which(tmp)                         ## integer indices of every T in 'tmp'

min(penalty)                       ## what is the lowest value of 'penalty'; can 'max()' too
which(penalty == min(penalty))     ## get integer index where 'penalty' is at minimum
m[which(penalty == min(penalty))]  ## what is the value of 'm' at the minimum 'penalty'
round(mean(z), 2)                  ## mean(z) with precision matching 'm'

```

Implications of the mean of a numeric set mimimizing the sum-of-squared distances to all 
  values in the set extend to both summarizing data and predicting/imputing unobserved 
  values. If you were to summarize the values in the set with a single value, the mean would 
  be least incorrect of all possible answers, if correctness is quantified by the total 
  squared distance from (how far 'off') values in the set are from the estimate.
  Similarly, if someone were to draw one value from the set and ask you to predict 
  what the value was, penalizing you with the square of how far off your guess was,
  on average (if we repeated the experiment many, many times and averaged the penalties), 
  your best possible guess would be the mean of the set.

[Return to index](#index)

---

### Populations and samples

There are two types of means that must be distinguished during 
  statistical inference about means: the first is the 'population mean' and 
  the second is the 'sample mean'. In this context, 'population' refers to 
  the population you are interested in understanding. If you wanted to know
  the mean height of everyone in the US, the US population would be the 
  population you are studying. The most direct way to get the mean height for
  the US population would be to measure the height of everyone in the US
  and average the measurements. Then the mean would be known exactly. However, 
  in many cases, it is not practical to measure every member of a population,
  so instead we work with more tractably sized sub-samples from the population. 
  A similar issue arises if we wish to understand a recurring event (something 
  that's happened before and is expected to continue to happen in the future). 
  In this case, the population might be the entire series of events, including 
  future events (if we are interested in prediction). However, future events 
  are not currently available for measurement. So instead we work with samples 
  from the past in order to make our estimates (predictions) of what will happen 
  in the future.

The most important thing about using samples to make estimates about populations
  is that the samples be randomly drawn from the population of interest. 
  Your study would be flawed (even if it happened to yield the correct answer)
  if you tried to estimate the US mean height by only sampling women or only 
  sampling in Baltimore. The issues with non-random sampling are well recognized 
  in the field of medical predictive models: these models tend to predict outcomes 
  better in males of European descent better than other members of the US or world 
  populations. This is because 'in the old days' many medical studies conducted in 
  the US used exclusively white male subjects. When predicting future events, we 
  cannot really randomly sample the entire population (since some events in the 
  population haven't happened yet), so we rely on a usually implicit assumption that 
  the processes and trends of the past will continue in the future without change. 
  In many fields (such as economics and social sciences), this assumption often proves 
  incorrect. However, in the biomedical field, as well as other sciences, we are 
  often studying processes that we can be fairly confident will continue to 
  follow the patterns of the past over any practically important prediction 
  period. For instance, we can assume with substantial confidence that over the
  next million years the earth will continue to rotate (abeit with the current 
  pattern of deceleration), ATP will continue being used for transmitting potential 
  energy within a cell, and carbon will continue to have a valence of four. By 
  contrast, one unexpected remotely related occurrence, such as a trade war or 
  pandemic, can drastically affect the predictive accuracy of even near-term 
  economic models.

As was mentioned earlier, if you could measure e.g. the height of every individual 
  in a population of interest, then you could calculate the mean height of the 
  population exactly. If instead you measure the height of everyone in a
  random sample from the population, you can calculate the mean height of 
  everyone in that sample exactly. But how good of an estimate of the population
  mean will that sample mean be? Much of what we will discuss in this course
  revolves around this and closely related questions. 

[Return to index](#index)

---

### Check your understanding 1:

1) Draw 1000 samples from a uniform distribution on the interval `[-5, 5]` and store them
     in the variable `x`. Make a histogram of `x`. 

2) Draw 1000 samples from a normal distribution with mean `0` and standard deviation `2` 
     and store the samples in the variable `y`. Make a histogram of `y`. Try to fiddle 
     with the number of histograms 'bins' (`bins` parameter to `hist()`). Trying values 
     in `c(3, 10, 30, 100, 300)` should provide a 'feel' for what happens when too
     many or too few.

3) Concatenate `x` and `y` into a single vector of length 2000. Generate
     a 2D plot of the result. From the plot, can you tell where one distribution 
     (say the one from `x`) stops and the other (from `y`) starts?

4) Add a horizontal line to the plot from question #3, at the mean of `z`. 
     Hint: abline parameter `h` for 'horizontal'.

5) Add a vertical line to the plot at the 1000th position in the plot from question #4.
     Hint: abline parameter `v` for 'vertical'; note the bottom axis positions are the
     order (or index positions) of the values in `z`, so the 1000th position marks
     the boundary between the values from `x` (uniform) and those from `y` (normal).
     Are some differences in the distribution of points between the left side and 
     right side of your horizontal line apparent to you? 

[Return to index](#index)

---

### Variances and standard deviations

As we've discussed, the mean is an optimal summary of the 'central tendency' of a set 
  of values in the 'sum-of-squares' (sum-of-squared differences) sense. Another
  aspect of a distribution which we'd often like to quantify is how spread out the
  values in the set are. The 'sum-of-squares' we've been discussing captures how
  far away values are from the mean, so it might be a reasonable measure of spread
  around the mean. However, since it is a sum, it will grow with the number of values
  in the set, rather than converge to a single value. Instead, what we want is an 
  average of the squared differences of the values in the set from the mean. This 
  average squared-distance from values to the mean is called the 'variance' of the 
  set of values.

```
rm(list=ls())

set.seed(3)

## samples of various sizes from a 'population' (the normal distribution N(0, 1)):

(x <- 10 ^ (1 : 6))
names(x) <- as.character(x)
x

y <- sapply(x, rnorm, mean=0, sd=1)      ## one sample for each element of x
d <- sapply(y, function(v) v - mean(v))  ## distances from sample elements to respective sample means
d2 <- sapply(d, function(v) v * v)       ## square the distances

str(y)
str(d)
str(d2)

sapply(d2, sum)                          ## increasing sample size diverges
sapply(d2, mean)                         ## increasing sample size converges towards population variance of 1

```

The variance calculation involves squaring values, which results in the units of
  a variance being the square of whatever unit was being used to measure the
  mean. For instance, if the set of values were heights measured in inches of a 
  sample of US residents, the variance would have units of inches-squared. This 
  is an issue for when combining the mean and variance in your calculations. 
  For instance, if you wanted to know how large the spread was relative to the
  mean you want a unitless number for the ratio, but dividing the variance in
  inches-squared by the mean height in inches would yield a ratio in inches,
  which is hard to interpret. Similary, if you wanted to graph the mean and
  the spread along the same axis, using the variance might be confusing, because 
  it is expressed in different units than the mean. In order to overcome these 
  issues, we often work with the square-root of the variance, which is called 
  the standard deviation. The standard deviation will always be expressed in
  the same units as the mean, so the two can be combined more intuitively in 
  calculations and when plotting.

[Return to index](#index)

---

### Standard errors and bias

Let's take several random samples from a normal distribution in order to
  estimate the mean of an infinite population (the distribution we are 
  drawing from can yield an infinite number of values) based on samples of 
  different sizes. Since we know the population mean and variance 
  (both set as part of specifying the normal distribution), we can evaluate
  how close the sample means are to the population mean and how much spread 
  there is between sample means of different samples drawn from the same 
  population. 

```
rm(list=ls())

set.seed(1)

## 'population' (naive) formula for variance of v:

f.var.pop <- function(v) {
  d <- v - mean(v)
  sum(d ^ 2) / length(v)
}

## 'sample' (theoretically correct) formula for variance of v; uses
##   "Bessel's correction" of denominator:

f.var.smp <- function(v) {
  d <- v - mean(v)
  sum(d ^ 2) / (length(v) - 1)
}

f.stat <- function(n.i, m=0, s=1, R=10000) {

  means <- numeric(length=0)         ## numeric vector of length 0 (empty)
  s2.pop <- numeric(0)               ## empty numeric vector ('length' optional)
  s2.smp <- numeric(0)               ## empty numeric

  for(i in 1:R) {                    ## conduct R 'experiments'
    v.i <- rnorm(n.i, mean=m, sd=s)  ## v.i: random sample of size n.i from N(m, s)

    m.i <- mean(v.i)                 ## m.i: mean of v.i
    s2.pop.i <- f.var.pop(v.i)       ## variance of v.i based on 'population' formula
    s2.smp.i <- f.var.smp(v.i)       ## variance of v.i based on 'sample' formula

    means <- c(means, m.i)           ## add m.i to vector of sample means
    s2.pop <- c(s2.pop, s2.pop.i)    ## add s.pop.i to end of sds.pop
    s2.smp <- c(s2.smp, s2.smp.i)    ## add s.smp.i to end of sds2
  }

  bias.m <- mean(means) - m          ## how far off is average sample estimate from true value
  se.m <- sd(means)                  ## standard error of the mean: sd() of R sample means
  bias.s2.pop <- mean(s2.pop) - s^2  ## apparent bias of sd based on population formula
  se.s2.pop <- sd(s2.pop)            ## standard error of sd based on population formula
  bias.s2.smp <- mean(s2.smp) - s^2  ## apparent bias of sd based on sample formula
  se.s2.smp <- sd(s2.smp)            ## standard error of sd based on sample formula

  c(bias.m=bias.m, se.m=se.m, bias.s2.pop=bias.s2.pop, se.s2.pop=se.s2.pop, bias.s2.smp=bias.s2.smp, se.s2.smp=se.s2.smp)
}

(n <- (2 : 20) ^ 2)                  ## a series of sample sizes to try
(rslt <- sapply(n, f.stat))
t(rslt)
(rslt <- cbind(n=n, t(rslt)))
summary(rslt)

## set up a blank plot with the right axis ranges;
##   "type='n'" means don't actually plot anything:

plot(
  x=range(n),                     ## make sure x axis can accommodate range of n
  y=c(-0.3, 0.9),                 ## y axis spans range of 'bias' and 'se'
  xlab='sample size',             ## label for (horizontal/bottom) x-axis
  ylab='metric',                  ## label for (vertical/left) y-axis
  type='n'                        ## don't actually plot anything yet
)

## more clearly indicate where '0' is:
abline(h=0, lty=3)                ## dotted h(orizontal) line at y=0                    

## connect the dots with line segments:

lines(
  x=rslt[, 'n'],                  ## x-axis: the sample size
  y=rslt[, 'bias.m'],             ## y-axis: bias
  col='cyan',                     ## color of the line segments
  lty=2                           ## dashed line style
)

lines(
  x=rslt[, 'n'],                  ## x-axis: sample size
  y=rslt[, 'se.m'],               ## y-axis: standard error
  col='cyan',                     ## color of line segments
  lty=3                           ## dotted line style
)

lines(
  x=rslt[, 'n'],                  ## x-axis: the sample size
  y=rslt[, 'bias.s2.pop'],        ## y-axis: bias
  col='magenta',                  ## color of the line segments
  lty=2                           ## dashed line style
)

lines(
  x=rslt[, 'n'],                  ## x-axis: sample size
  y=rslt[, 'se.s2.pop'],          ## y-axis: standard error
  col='magenta',                  ## color of line segments
  lty=3                           ## dotted line style
) 

lines(
  x=rslt[, 'n'],                  ## x-axis: the sample size
  y=rslt[, 'bias.s2.smp'],        ## y-axis: bias
  col='orangered',                ## color of the line segments
  lty=2                           ## dashed line style
)

lines(
  x=rslt[, 'n'],                  ## x-axis: sample size
  y=rslt[, 'se.s2.smp'],          ## y-axis: standard error
  col='orangered',                ## color of line segments
  lty=3                           ## dotted line style
) 


## drop a simple legend in 'topright' part of plot:

legend(
  ## where to put legend:
  'topright',
  ## legend labels:
  legend=c('Bias (mean)', 'SE (mean)', 'Bias (var pop)', 'SE (var pop)', 'Bias (var samp)', 'SE (var samp)'),
  ## colors (in register w/ labels)
  col=c('cyan', 'cyan', 'magenta', 'magenta', 'orangered', 'orangered'),
  ## line types (in register w/ labels and colors):
  lty=c(2, 3, 2, 3, 2, 3)
)

## the default formulas in R are the sample formulas:

x <- runif(30, min=0, max=100)
f.var.pop(x)
f.var.smp(x)
var(x)

sqrt(f.var.pop(x))
sqrt(f.var.smp(x))
sd(x)

```

Take home messages: In general, the standard error of a sample estimate of a population
  parameter (such as the mean or standard deviation) decreases with increasing sample 
  size, but with diminishing returns. In fact, the standard error of the mean is 
  directly proportional to the variance of the population parameter being measured and
  inversely proportional to square-root of sample size:

`standard_error_mean = population_variance / square_root(sample_size)`

We also saw that the naive formula (same one you would use for a population) for the 
  sample mean returns an unbiased estimator of the population mean. However, the naive 
  formula for the sample variance actually has a negative bias, which is particularly 
  pronounced for small samples. This bias is eliminated by changing the formula to 
  use `n - 1` instead of `n` in the denominator when averaging.

As an aside: the reason applying the population formula to a sample in order to estimate 
  the population variance results in downwardly biased estimates can be understood in terms 
  of our previous discussion about the mean minimizing the sum of squared distances to the 
  values and the fact that the variance is calculated from the squared distances
  from the sample mean, not the population mean (since the latter is typically unknown). 
  But the sample mean is always a bit off from the population mean, because it is calculated 
  from the sample instead of the whole population. But the sample mean will always 
  have a lower sum of squared distances (and therefore average square distance) to the 
  sample values than any other number, including the population mean. Therefore, the 
  variance calculated by using the population formula plugging and the sample mean
  will always be a bit smaller than if the sd was calculated using the same formula and the 
  population mean. Theoretical analysis of this problem has resulted in proofs that using 
  the sample formula for variance (using a denominator of `n - 1` instead of `n`) results 
  in an unbiased sample-based estimate of the population standard deviation, as we 
  observed in our experiments.

[Return to index](#index)

---

### Distribution of estimates of the mean

So we have now seen that if we repeatedly sample a very large (or infinite, in our case)
  population and calculate the mean for each sample, we have seen that the mean of these
  sample means is centered around the population mean. This means that the sample mean
  is an 'unbiased estimator' of the population mean. We have also seen that the spread of 
  these sample means around the population mean decreases with inceasing sample size. 
  However, the spread of sample means reflects an important aspect of the population:
  the more dispersed the population mean, the more dispersed is the distribution of the
  sample means. However, the shape of the distribution of sample means does not necessarily
  reflect the shape of the underlying population:

```
rm(list=ls())

set.seed(10)

m3 <- numeric(0) 

for(i in 1:1e5) {
  z_i <- runif(3, min=0, max=10)
  m_i <- mean(z_i)
  m3 <- c(m3, m_i)
}

m30 <- numeric(0) 

for(i in 1:1e5) {
  z_i <- runif(30, min=0, max=10)
  m_i <- mean(z_i)
  m30 <- c(m30, m_i)
}

par(mfrow=c(1, 2))        ## plotting area split into 1 row, 2 columns
hist(m3)                  ## in first column
hist(m30)                 ## in second column
par(mfrow=c(1, 1))        ## reset plotting area to 1 row, 1 column

```

Here we can see that even though the shape of the population being sampled is uniform (should look
  more or less like a rectangle with uniform height between 0 and 10), the shape of the 
  distribution of sample means is bell shaped. In fact, the distribution of sample means of any
  continuous distribution will have the familiar bell-shaped normal distribution, with the mean of
  that normal distribution falling around the population mean (unbiased) and with the standard deviation
  of that normal distribution (the other parameter) being equal to the standard error of the mean.

[Return to index](#index)

---

## FIN!
