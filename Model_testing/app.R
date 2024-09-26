# load the package
library(stats4)
library(lattice)
library(mirt)
library(shiny)
library(readxl)
library(catR)
library(writexl)
library(shiny)
library(shinyBS)
library(shinyjs)
library(shinydashboard)
library(DT)
library(readxl)
library(ggplot2)

# load the data
questions <- read.csv("Questions.csv") # Load the SAPS question CSV file
questions2 <- read.csv("Questions2.csv") # Load the SANS question CSV file
load("sapsCoef.Rdata") # Load the SAPS coefficients
load("sapsResultGRM.Rdata") # Load the SAPS GRM results
load("sansCoef.Rdata") # Load the SANS coefficients
load("sansResultGRM.Rdata") # Load the SANS GRM results

answers <- read.csv("SAPS_answer.csv")
answers <- answers[, -1]  # Remove the "SL. no" column
print(head(answers))
answers2 <- read.csv("SANS_answer.csv")
answers2 <- answers2[, -1]  # Remove the "SL. no" column

# Data splitting and training
set.seed(123) # for reproducibility

# Split SAPS responses into training and validation
saps_train_indices <- sample(1:nrow(answers), 0.8 * nrow(answers))
saps_train_responses <- answers[saps_train_indices, ]
saps_val_responses <- answers[-saps_train_indices, ]

# Split SANS responses into training and validation
sans_train_indices <- sample(1:nrow(answers2), 0.8 * nrow(answers2))
sans_train_responses <- answers2[sans_train_indices, ]
sans_val_responses <- answers2[-sans_train_indices, ]

calculate_scores <- function(responses, coef, resultGRM, model="GRM", threshold=0.39) {
  # Convert responses to a numeric vector
  x <- as.numeric(unlist(responses))
  
  theta <- thetaEst(coef, x, model = model, range = c(-6, 6), method = "BM")
  error <- semTheta(theta, coef, x, range = c(-6, 6), model = model, method = "BM")
  stop <- list(rule = "precision", thr = threshold)
  checkstop <- checkStopRule(theta, error, length(x), coef, model = model, stop=stop)
  Theta <- matrix(seq(-6, 6, 0.001))
  tscore <- cbind(Theta, expected.test(resultGRM, Theta))
  total_score <- tscore[findInterval(theta, tscore[, 1]), 2]
  lower_ci <- theta - error
  upper_ci <- theta + error
  lower_score <- tscore[findInterval(round(lower_ci, digits = 2), tscore[, 1]), 2]
  upper_score <- tscore[findInterval(round(upper_ci, digits = 2), tscore[, 1]), 2]
  list(theta = theta, total_score = total_score, lower_score = lower_score, upper_score = upper_score)
}

#Training Model on training Data
# Apply the function to SAPS training data
saps_trainResults <- lapply(saps_train_responses, calculate_scores, coef=sapsCoef, resultGRM=sapsResultGRM)

# Apply the function to SANS training data
sans_trainResults <- lapply(sans_train_responses, calculate_scores, coef=sansCoef, resultGRM=sansResultGRM)

#Validating model on Test Data
# Validate SAPS model
saps_validationResults <- lapply(saps_val_responses, calculate_scores, coef=sapsCoef, resultGRM=sapsResultGRM)

# Extract actual and predicted SAPS scores
# Assuming you have an 'actual_scores' column in the 'answers' data frame
saps_actualScores <- answers$actual_scores
saps_predictedScores <- sapply(saps_validationResults, function(x) x$total_score)

# Calculate SAPS correlation and MAE
saps_correlation <- cor(saps_actualScores, saps_predictedScores)
saps_mae <- mean(abs(saps_actualScores - saps_predictedScores))

# Print SAPS results
cat("SAPS Correlation (r):", saps_correlation, "\n")
cat("SAPS Mean Absolute Error (MAE):", saps_mae, "\n")

# Validate SANS model
sans_validationResults <- lapply(sans_val_responses, calculate_scores, coef=sansCoef, resultGRM=sansResultGRM)

# Extract actual and predicted SANS scores
# Assuming you have an 'actual_scores' column in the 'answers2' data frame
sans_actualScores <- answers2$actual_scores
sans_predictedScores <- sapply(sans_validationResults, function(x) x$total_score)

# Calculate SANS correlation and MAE
sans_correlation <- cor(sans_actualScores, sans_predictedScores)
sans_mae <- mean(abs(sans_actualScores - sans_predictedScores))

# Print SANS results
cat("SANS Correlation (r):", sans_correlation, "\n")
cat("SANS Mean Absolute Error (MAE):", sans_mae, "\n")

#Classification and Accuracy calculation for SAPS
saps_threshold <- 37  

# Classify SAPS based on the threshold
saps_predictedClasses <- ifelse(saps_predictedScores > saps_threshold, 1, 0)
saps_actualClasses <- ifelse(saps_actualScores > saps_threshold, 1, 0)

# Calculate SAPS accuracy
saps_accuracy <- sum(saps_predictedClasses == saps_actualClasses) / length(saps_actualClasses)

cat("SAPS Accuracy:", saps_accuracy, "\n")

#Classification and Accuracy calculation for SANS
sans_threshold <- 37  

# Classify SANS based on the threshold
sans_predictedClasses <- ifelse(sans_predictedScores > sans_threshold, 1, 0)
sans_actualClasses <- ifelse(sans_actualScores > sans_threshold, 1, 0)

# Calculate SANS accuracy
sans_accuracy <- sum(sans_predictedClasses == sans_actualClasses) / length(sans_actualClasses)

cat("SANS Accuracy:", sans_accuracy, "\n")