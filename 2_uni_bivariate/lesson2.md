# Fundamentals of computational data analysis using R
## Univariate and bivariate statistics: lesson 2
#### Contact: mitch.kostich@jax.org

---

### Index

- [Simple plotting](#simple-plotting)
- [Formulas for plotting and fitting](#formulas-for-plotting-and-fitting)

- [Check 1](#check-your-understanding-1)
- [Check 2](#check-your-understanding-2)
- [Check 3](#check-your-understanding-3)

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
