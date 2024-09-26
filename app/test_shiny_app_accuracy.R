library(tidyverse)
library(mirt)
library(lavaan)
devtools::install_github("masurp/ggmirt")
library(dplyr)

# devtools::install_github("masurp/ggmirt")
library(ggmirt)

data <- read.csv("SAPS_answer.csv")
d <- na.omit(data)


fitGraded <- mirt(d, 1, itemtype = "graded")
fitGraded
summary(fitGraded)

params <- coef(fitGraded, IRTpars = TRUE, simplify = TRUE)
round(params$items, 2)

M2(fitGraded, type = "C2", calcNULL = FALSE)
itemfit(fitGraded)
personfit(fitGraded)
personfit(fitGraded) %>%
  summarize(infit.outside = prop.table(table(z.infit > 1.96 | z.infit < -1.96)),
            outfit.outside = prop.table(table(z.outfit > 1.96 | z.outfit < -1.96))) # lower row = non-fitting people

personfitPlot(fitGraded)
itemfitPlot(fitGraded)

tracePlot(fitGraded) +
  labs(color = "Answer Options")
itemInfoPlot(fitGraded, facet = TRUE,theta_range = c(-6, 6))

testInfoPlot(fitGraded, theta_range = c(-6, 6), adj_factor = .5)

