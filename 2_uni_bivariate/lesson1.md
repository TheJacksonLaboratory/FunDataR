# Fundamentals of computational data analysis using R
## A primer on programming using R: lesson 3
#### Contact: mitch.kostich@jax.org

---

### Index

- [What is a mean?](#what-is-a-mean)
- [Populations and samples](#populations-and-samples)

- [Check 1](#check-your-understanding-1)

### What is a mean?

You are probably familiar with the notion of the 'mean' or 'average' of a 
  series of numbers as a type of central value for the numbers in the series.
  But you've probably also heard of the 'median' and may know that it too,
  is a type of central value. You may know the distinction based on the 
  difference in the procedures for calculating means and medians.
  The median of the series `x` would be found by sorting `x` then taking the 
  middle value (or the mean of the two central values if `x` has even length). 
  By contrast, means are calculated using (expressed in R): 
  `sum(x) / length(x)`. R also provides the premade (and compiled, so more
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

  ## length check: does length(m) equal 1 or length(v)?:

  if( (length(m) != 1)  && (length(m) != length(v)) ) {
    stop("length(m) not 1 or length(v)")
  }

  dist <- v - m             ## distances between v and m; length(dist) == length(v)
  dist.sqr <- dist ^ 2      ## distances squared; length(dist.sqr) == length(v)
  sum.sqr <- sum(dist.sqr)  ## sum of the squared distances; length(sum.sqr) = 1
  return(sum.sqr)           ## explicit return of sum of squared distances
}

## sanity check f.ss(); essential practice for all your functions!
f.ss(0)                     ## default m==0; (0 - 0)^2 == 0
f.ss(1)                     ## (1 - 0)^2 == 1
f.ss(2)                     ## (2 - 0)^2 == 4
f.ss(0, 0)                  ## (0 - 0)^2 == 0
f.ss(0, 1)                  ## (0 - 1)^2 == 1
f.ss(1, 1)                  ## (1 - 1)^2 == 0
f.ss(0, c(0, 1))            ## oops! length(m) != 1 or length(v)
f.ss(c(1, 0), 0)            ## (1 - 0)^2 + (0 - 0)^2 == 1
f.ss(c(1, 0), c(0, 1))      ## (1 - 0)^2 + (0 - 1)^2 == 2
f.ss(c(2, 0), c(0, 2))      ## (2 - 0)^2 + (0 - 2)^2 == 8

## time for an 'experiment':
set.seed(1)
(y <- rnorm(150, mean=100, sd=10))
hist(y, breaks=20)
mean(y)
f.ss(y, mean(y))
f.ss(y, mean(y) + 1)
f.ss(y, mean(y) - 1)
f.ss(y, 0)
f.ss(y, 20)

## f.ss() minimized by mean() even for weird distributions like this
##   3 peaked mixture of 2 uniforms and one normal:

y <- c(
  rnorm(50, mean=100, sd=10),  ## 50 draws from normal
  runif(50, min=50, max=75),   ## 50 draws from uniform on interval [50, 75]
  runif(50, min=125, max=150)  ## 50 draws from uniform on interval [125, 150]
)
y
hist(y, breaks=20)             ## 3-peaks, normal flanked by two uniforms

mean(y)
f.ss(y, mean(y))
f.ss(y, mean(y) + 1)
f.ss(y, mean(y) - 1)
f.ss(y, 0)
f.ss(y, 200)

```

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
  better in white males than other members of the US or world populations, because 
  'in the old days' many medical studies conducted in the US used exclusively 
  white male subjects. When predicting future events, we cannot really randomly 
  sample the entire population (since some events in the population haven't 
  happened yet), so we rely on a usually implicit assumption that the processes 
  and trends of the past will continue in the future without change. In many 
  fields (such as economics and social sciences), this assumption often proves 
  incorrect. However, in the biomedical field, as well as other sciences, we are 
  usually studying processes that we can be fairly confident will continue to 
  follow the patterns of the past over any practically important prediction 
  period. For instance, we can assume with substantial confidence that the earth 
  will continue to rotate (abeit with the current pattern of deceleration), ATP 
  will continue being used for storing and transmitting potential energy within 
  a cell, and carbon will continue to have a valence of four. By contrast, 
  something like the unpredictable emergence of a pandemic can have a dramatic
  impact on the performance of previously developed economic models.
  
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

## FIN!
