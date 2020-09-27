# Fundamentals of computational data analysis using R
## A primer on programming using R: lesson 3
#### Contact: mitch.kostich@jax.org

---

### Index

- [What is a mean?](#what-is-a-mean)
- [Populations and samples](#populations-and-samples)
- [Variances and standard deviations](#variances-and-standard-deviations)
- [Standard errors](#standard-errors)

- [Check 1](#check-your-understanding-1)

### What is a mean?

You are probably familiar with the notion of the 'mean' or 'average' of a 
  series of numbers as a type of central value ('central tendency') for 
  the numbers in the series. But you've probably also heard of the 'median' 
  and may know that it too, is a type of 'central tendency'. You may know the 
  difference between the two in terms of the procedures for calculating their 
  values. The median of the series `x` would be found by sorting `x` then 
  taking the middle value (or the mean of the two central values if `x` has 
  even length). By contrast, means are calculated using (expressed in R): 

`sum(x) / length(x)`

R also provides the premade (and compiled, so more
  efficient) function `mean(x)` for this purpose.

Let's take 1000 random numbers from a normal (a.k.a. Gaussian) distribution,
  calculate their mean and plot the results. In this case (plotting a 
  single variable `z`), the horizontal/bottom axis indicates the order 
  in which the numbers occur in `z`. The vertical/left axis indicates
  the magnitude of the numbers in `z`. Since the numbers were drawn 
  at random, we do not expect any relationship between the magnitudes
  (vertical axis) and order in which numbers were drawn (horizontal 
  axis):

```
rm(list=ls())

set.seed(1)
(z <- rnorm(1000, mean=100, sd=10))
mean(z)

## let's see what this distribution looks like:
hist(z)                           ## peaked toward center, much less in tails

## let's make a 2D plot (values on vertical axis, order on horizontal):
plot(z)                           ## 2D-plot; values cluster towards mean
abline(h=mean(z), col='cyan')     ## add a h(orizontal) line at mean(z)

```

Now let's repeat the same, but drawing from a uniform distribution in the 
  numeric (includes fractional numbers) closed (includes endpoints) 
  interval [90, 110]. We'll often sample from this distribution, so 
  it is good to get a feel for it:

```
rm(list=ls())

set.seed(1)
(z <- runif(1000, min=80, max=120))
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
##     out for clarity and add a length check for robustness.

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
f.ss(0, c(0, 1))                   ## oops! length(m) != 1 or length(v)
f.ss(c(1, 0), 0)                   ## (1 - 0)^2 + (0 - 0)^2 == 1
f.ss(c(1, 0), c(0, 1))             ## (1 - 0)^2 + (0 - 1)^2 == 2
f.ss(c(2, 0), c(0, 2))             ## (2 - 0)^2 + (0 - 2)^2 == 8

## time for an 'experiment':
set.seed(1)
(z <- rnorm(150, mean=100, sd=10))
hist(z, breaks=20)
mean(z)
f.ss(z, mean(z))
f.ss(z, mean(z) + 1)
f.ss(z, mean(z) - 1)
f.ss(z, 0)
f.ss(z, 20)

## f.ss() minimized by mean() even for weird distributions like this
##   3 peaked mixture of 2 uniforms and one normal:

z <- c(
  rnorm(50, mean=100, sd=10),      ## 50 draws from normal
  runif(50, min=50, max=75),       ## 50 draws from uniform on interval [50, 75]
  runif(50, min=125, max=150)      ## 50 draws from uniform on interval [125, 150]
)
z
hist(z, breaks=20)                 ## 3-peaks, normal flanked by two uniforms

mean(z)
f.ss(z, mean(z))
f.ss(z, mean(z) + 1)               ## penalty goes up as we move away from mean
f.ss(z, mean(z) - 1)               ## same thing in this direction
f.ss(z, 0)                         ## worse as you get further?
f.ss(z, 200)                       ## seems like it in this direction?

## we can get a comprehensive view graphically:

f <- function(a, b) f.ss(b, a)     ## flip the order of args so works with 'sapply()'
m <- seq(from=1, to=200, by=0.01)  ## values to try for 'm'; will be 'x' axis
penalty <- sapply(m, f, z)         ## calculate the penalty at each value of 'm'; 'y' axis
plot(x=m, y=penalty)               ## single minimum (no local minima), 'convex' shape
abline(v=mean(z), col='cyan')      ## v(ertical) line where m=mean(z)

## or look at it in a less happhazard manner; the 'which()' function returns the
##   integer index of every TRUE value in a logical vector:

tmp <- c(F, T, F, F, T, F, T)      ## a logical vector for demo purposes
which(tmp)                         ## integer indices of every T in 'tmp'

min(penalty)                       ## what is the lowest value of 'penalty'; can 'max()' too
which(penalty == min(penalty))     ## get integer index where 'penalty' is at minimum
m[which(penalty == min(penalty))]  ## what is the value of 'm' at the minimum 'penalty'
round(mean(z), 2)                  ## mean(z) with precision matching 'm'

```

Implications of the mean mimimizing the sum-of-squared distances to all values in a numeric 
  set extend to both summarizing data and making predictions about unobserved values.
  If you were to summarize the values in the set with a single value, the mean would 
  be least incorrect of all possible answers, if correctness is quantified by the total 
  squared distance from (how far 'off') values in the set are from the estimate.
  Similarly, if I were to draw one value from the set and ask you to predict 
  what the value was, penalizing you with the square of how far off your guess was,
  on average (if we repeated the experiment many, many times and averaged the penalties), 
  your best possible guess would be the mean of the set.

---

### Populations and samples

There are two types of means that must be distinguished in order to understand
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
  that's happened before and will continue to happen in the future). In this 
  case, the population might be the entire series of events, including future 
  events (if we are interested in prediction). However, future events are not 
  currently available for measurement. So instead we work with samples from 
  the past in order to make our estimates (predictions) of what will happen 
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
  period. For instance, we can assume with substantial confidence that the earth 
  will continue to rotate (abeit with the current pattern of deceleration), ATP 
  will continue being used for transmitting potential energy within 
  a cell, and carbon will continue to have a valence of four. By contrast, 
  something like the unforseen emergence of a pandemic can have a dramatic
  impact on the performance of previously developed economic model.

As was mentioned earlier, if you could measure e.g. the height of every individual 
  in a population of interest, then you could calculate the mean height of the 
  population exactly. If instead you measure the height of everyone in a
  random sample from the population, you can calculate the mean height of 
  everyone in that sample exactly. But how good of an estimate of the population
  mean will that sample mean be? Much of what we will discuss in this course
  revolves around this and closely related questions. 

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
  in the set, even if the new values are drawn from exactly the same distribution.
  Instead, what we want is an average of the squared differences of the values in
  the set from the mean. This value is called the variance of the set of values.

```
rm(list=ls())

set.seed(33)

## 30, 100, 300, and 1000 samples from normal distribution N(0, 1):
x30 <- rnorm(30, mean=0, sd=1)
x100 <- rnorm(100, mean=0, sd=1)
x300 <- rnorm(300, mean=0, sd=1)
x1000 <- rnorm(1000, mean=0, sd=1) 

## distances of values from respective means:
d30 <- x30 - mean(x30)
d100 <- x100 - mean(x100)
d300 <- x300 - mean(x300)
d1000 <- x1000 - mean(x1000)

## sum-of-squared distances: diverge
sum(d30 ^ 2)
sum(d100 ^ 2)
sum(d300 ^ 2)
sum(d1000 ^ 2)

## mean-of-squared distances: converge
mean(d30 ^ 2)
mean(d100 ^ 2)
mean(d300 ^ 2)
mean(d1000 ^ 2)

```

The variance calculation involves squaring values, which means that the units of
  a variance will be the square of whatever unit was being used to measure the
  mean. For instance, if the set of values were heights measured in inches of a 
  sample of US residents, the variance would have units of inches-squared. This 
  is an issue for when combining the mean and variance in your calculations. 
  For instance, if you wanted to know how large the spread was relative to the
  mean you want a unitless number for the ratio, but dividing the variance in
  inches-squared by the mean height in inches would yield a ratio in inches,
  which is hard to interpret. Similary, if you wanted to graph the mean and
  the spread along the same axis, the variance would not work, because it is
  expressed in different units than the mean. In order to overcome these issues,
  we often work with the square-root of the variance, which is called the
  standard deviation. The standard deviation will always be expressed in
  the same units as the mean, so the two can be combined sensibly in calculations
  and when plotting.

[Return to index](#index)

---

### Standard errors



## FIN!
