library(ggplot2)
library(ltm)
library(reshape2)
library(dplyr)   
library(mirt)

data <- read.csv("SANS_answer.csv")
data <- na.omit(data)

# Fit the GRM model with all items
model <- mirt(data, 1, itemtype = 'graded')

# Test plotting a single item to ensure it works
plot(model, type = 'trace', items = 1, theta_lim = c(-6, 6), main = colnames(data)[1])

# Set up a 2x2 plotting area
par(mfrow = c(2, 2))

# Plot ICCs for each item
item_names <- colnames(data)
for (i in 1:length(item_names)) {
  plot(model, type = 'trace', items = i, theta_lim = c(-6, 6), main = item_names[i])
  
  # Move to the next plot grid if necessary
  if (i %% 4 == 0 && i != length(item_names)) {
    par(mfrow = c(2, 2))
  }
}
