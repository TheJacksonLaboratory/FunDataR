# Fundamentals of computational data analysis using R
## A primer on programming using R: lesson 3
#### Contact: mitch.kostich@jax.org

---

### Index

- [Simple plotting](#simple-plotting)
- [Formulas for plotting and fitting](#formulas-for-plotting-and-fitting)

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

### What is a mean?

You are probably familiar with the notion of the 'mean' or 'average' of a 
  series of numbers as a type of central value for the series.
  But you've probably also heard of the 'median' and may know that it too,
  is a type of central value. You may also be familiar with the distinction
  based on the difference in the formulas for calculating means and medians.
  The median of the series `x` would be found by sorting `x` then taking the 
  middle value (or the mean of the two central values if x has even length). 
  By contrast, means are calculated using (expressed in R): 
  `sum(x) / length(x)`. R also provides the premade (and compiled, so more
  efficient) function `mean()` for this purpose.

```
rm(list=ls())

set.seed(1)
(y <- rnorm(30, mean=10, sd=3))
mean(y)
plot(y)
abline(b=mean(y))

```

A function to compute sum of squared distances
Mean is the single value that minimizes that function

```
rm(list=ls())

## f.ss() returns sum of squared distances of points in v from m.
##   v should be a numeric vector; assumes no NaN or NA values
##   m should be a numeric vector of length 1 or of length length(v)
##   return value is numeric of length 1, or NA on error
##
##   could single line: 'return(sum((v - m) ^ 2))', but we'll break it
##     out for clarity and add a length check for robustness.

f.ss <- function(v, m=0) {

  ## length check: does length(m) equal 1 or length(v)?
  if( (length(m) != 1)  && (length(m) != length(v)) ) {
    stop("length(m) not 1 or length(v)")
  }

  dist <- v - m             ## distances between v and m; length(dist) == length(v)
  dist.sqr <- dist ^ 2      ## distances squared; length(dist.sqr) == length(v)
  sum.sqr <- sum(dist.sqr)  ## sum of the squared distances; length(sum.sqr) = 1
  return(sum.sqr)           ## explicit return of sum of squared distances
}

set.seed(1)
(y <- rnorm(30, mean=10, sd=3))
mean(y)
f.ss(y, mean(y))
f.ss(y, mean(y) + 1)
f.ss(y, mean(y) - 1)
f.ss(y, 0)
f.ss(y, 20)

```

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
  in the field of medical predictive models: these 
  models tend to predict outcomes better in white males than other members of the 
  US or world populations, because 'in the old days' many medical studies 
  conducted in the US used exclusively white male subjects. When predicting 
  future events, we cannot really randomly sample the entire population (since
  some events in the population haven't happened yet), so we rely on a usually
  implicit assumption that the processes and trends of the past will continue
  in the future without change. In many fields (such as economics and social
  sciences), this assumption often proves incorrect. However, in the biomedical 
  field, as well as other sciences, we are usually studying processes that we
  can be fairly confident will continue to follow the patterns of the past
  over any practically important prediction period. For instance, we can 
  assume with substantial confidence that the earth will continue to rotate 
  (abeit with the current pattern of deceleration), ATP will continue being 
  used for storing and transmitting potential energy within a cell, and carbon 
  will continue to have a valence of four.

As was mentioned earlier, if you could measure e.g. the height of every individual 
  in a population of interest, then you could calculate the mean height of the 
  population exactly. If instead you measure the height of everyone in a
  random sample from the population, you can calculate the mean height of 
  everyone in that sample exactly. But how good of an estimate of the population
  mean will that sample mean be? Much of statistics revolves around this and
  closely related questions. 

---

### Check your understanding 1:

1) what is the third-root of 5

2) what is the sum of 500,726 and 324,781, divided by 67?

3) what is 3.14 to the 3.14 power?

[Return to index](#index)

---

### Simple plotting

An extremely important element of data analysis is data visualization. Let's take what you've
  learned thus far and make some simple plots.

```
tm <- 1:100                            ## time
dst <- tm ^ 2                          ## distance, assuming a constant force
tm
dst

plot(x=tm, y=dst)                      ## minimal plot

plot(                                  ## not a complete expression yet
  x=tm,                                ## x positions
  y=dst,                               ## corresponding y positions
  main="My default plot",              ## title for plot
  xlab="time (s)",                     ## how you want the x-axis labeled
  ylab="distance (m)"                  ## how you want the y-axis labeled
)                                      ## finally a complete statement

plot(
  x=tm, 
  y=dst, 
  main="My dot plot", 
  type="p",                            ## specify you want points plotted
  xlab="time (s)", 
  ylab="distance (m)", 
  col="cyan"
)

## now add some dashed lines:
lines(x=tm, y=dst, col='orangered', lty=3)  

plot(
  x=tm, 
  y=dst, 
  main="My line plot", 
  type="l",                            ## specify you want a line plot
  xlab="time (s)", 
  ylab="distance (m)", 
  col="cyan"
)

## now add some '+' points:
points(x=tm, y=dst, col='orangered', pch='+')

```

[Return to index](#index)

---

### Formulas for plotting and fitting

Here we give an example of a common notation used to express functional
  relationships between variables. This notation is widely used when 
  specifying statistical models in R. A basic example would be 
  `weight ~ operator`, which means that `weight` is conidered to be 
  a function of `operator`. For plotting purposes, this means that 
  `weight` ends up plotted on the 'y' (vertical) axis and `operator` 
  ends up plotted on the 'x' (horizontal) axis. This notation is often
  used along with a `data` parameter that specifies a data.frame in 
  which the variables can be found. Simply plotting a data.frame 
  (without a formula) results in a grid of plots in which each 
  variable is plotted against every other variable. This can be 
  useful for exploring a new dataset for potential relationships 
  between variables:

```
dat <- data.frame(
  treatment=factor(c(rep('ctl', 10), rep('trt', 10))),
  weight=c(rnorm(10, mean=10, sd=3), rnorm(10, mean=20, sd=5)),
  operator=factor(rep(c('weichun', 'mitch'), 10))
)
rownames(dat) <- letters[1 : nrow(dat)]
dat

plot(dat)                            ## what do you see?
plot(rock)                           ## 'rock' is a data set included with R

par(mfrow=c(1, 2))                   ## make a plot layout with 1 row and 2 columns
plot(weight ~ operator, data=dat)    ## plot this in first slot (row 1, column 1)
plot(weight ~ treatment, data=dat)   ## plot this in second slot (row 1, column 2)

```

The same type of notation is commonly used to fit a statistical model to a data.frame:

```
fit1 <- lm(weight ~ treatment, data=dat)
summary(fit1)

```

[Return to index](#index)

---

## FIN!
