# ============================================================
# BIKE SHARING DEMAND PREDICTION
# Tree-Based Methods — Academic Approach (ISLR2)
# ============================================================


# --- 1. PACKAGES ------------------------------------------------
#install.packages(c("readxl", "tree", "randomForest", "ggplot2", "corrplot", "caret", "pdp","gridExtra","dplyr"))

library(readxl)
library(tree)
library(randomForest)
library(ggplot2)
library(corrplot)
library(caret)
library(pdp)
library(gridExtra)
library(dplyr)


# --- 2. LOAD DATASET -------------------------------------------
bike <- read_excel("06_bike_sharing.xlsx")

head(bike)
str(bike)
summary(bike)


# --- 3. MISSING VALUES -----------------------------------------
colSums(is.na(bike))


# --- 4. EDA ----------------------------------------------------

# 4a. Rentals over time
ggplot(bike, aes(x = as.Date(date_day), y = rentals)) +
  geom_line(color = "steelblue") +
  labs(x = "Date", y = "Rentals", title = "Hourly Bike Rentals Over Time") +
  theme_minimal(base_size = 13)

# 4b. Rental distribution by season
ggplot(bike, aes(x = as.factor(season), y = rentals, fill = as.factor(season))) +
  geom_boxplot() +
  scale_fill_manual(values = c("1" = "lightblue", "2" = "lightgreen",
                               "3" = "orange",    "4" = "brown"),
                    labels = c("Winter", "Spring", "Summer", "Fall")) +
  scale_x_discrete(labels = c("Winter", "Spring", "Summer", "Fall")) +
  labs(title = "Rental Distribution by Season", x = "Season", y = "Rentals") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

# 4c. Average rentals by hour — working day vs weekend
ggplot(bike, aes(x = hour, y = rentals, color = as.factor(working_day))) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.2) +
  scale_color_manual(values = c("0" = "steelblue", "1" = "darkred"),
                     labels = c("Weekend/Holiday", "Working Day")) +
  labs(title = "Average Rentals by Hour of Day",
       x = "Hour", y = "Average Rentals", color = "") +
  theme_minimal(base_size = 13)

# 4d. Scatter: numerical predictors vs rentals
for (x in c("temperature", "humidity", "wind_speed")) {
  print(
    ggplot(bike, aes(x = .data[[x]], y = rentals)) +
      geom_point(color = "steelblue", alpha = 0.3) +
      labs(x = x, y = "Rentals", title = paste("Rentals vs", x)) +
      theme_minimal(base_size = 13)
  )
}

# 4e. Correlation matrix
corrplot(cor(bike[, c("temperature", "humidity", "wind_speed", "rentals")]),
         type = "upper", method = "circle",
         tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix", mar = c(0, 0, 1, 0))

# 4f. Target variable distribution
ggplot(bike, aes(x = rentals)) +
  geom_histogram(color = "white", fill = "steelblue", bins = 30) +
  labs(x = "Rentals", y = "Frequency", title = "Distribution of Rentals") +
  theme_minimal(base_size = 13)

# 4g. Boxplots: categorical predictors vs rentals
for (var in c("season", "hour", "month", "weather", "working_day")) {
  print(
    ggplot(bike, aes(x = as.factor(.data[[var]]), y = rentals)) +
      geom_boxplot(fill = "steelblue", alpha = 0.7) +
      labs(x = var, y = "Rentals", title = paste("Rentals by", var)) +
      theme_minimal(base_size = 13)
  )
}


# --- 5. DATA PREPARATION ---------------------------------------

# Cast categorical variables to factors
cat_vars <- c("season", "month", "hour", "holiday", "weekday", "working_day", "weather")
bike[cat_vars] <- lapply(bike[cat_vars], as.factor)
# NOTE: 'year' kept numeric — used only for temporal split, then dropped

# Temporal split: train = 2011 (year==0), test = 2012 (year==1)
# Random split is inappropriate for time-series data
train <- subset(bike, year == 0)
test  <- subset(bike, year == 1)

# Remove non-predictor columns after splitting
cols_to_remove <- c("year", "instant", "date_day")
train <- train[, !names(train) %in% cols_to_remove]
test  <- test[,  !names(test)  %in% cols_to_remove]

cat("Training set:", nrow(train), "rows\n")
cat("Test set:    ", nrow(test),  "rows\n")


# --- 6. HELPER: metrics function --------------------------------
# Returns RMSE, MAE, R² for any model prediction
calc_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae  <- mean(abs(actual - predicted))
  r2   <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  c(RMSE = round(rmse, 2), MAE = round(mae, 2), R2 = round(r2, 3))
}


# --- 7. SINGLE REGRESSION TREE ---------------------------------

tree.bike <- tree(rentals ~ ., data = train)
summary(tree.bike)

# Tree visualization
plot(tree.bike)
text(tree.bike, pretty = 0)
title("Regression Tree — Bike Sharing Demand")

# Metrics
pred.tree.train <- predict(tree.bike, newdata = train)
pred.tree       <- predict(tree.bike, newdata = test)

metrics.tree.train <- calc_metrics(train$rentals, pred.tree.train)
metrics.tree       <- calc_metrics(test$rentals,  pred.tree)

cat("Single Tree — Train RMSE:", metrics.tree.train["RMSE"], "\n")
cat("Single Tree — Test  RMSE:", metrics.tree["RMSE"],
    "| MAE:", metrics.tree["MAE"],
    "| R²:",  metrics.tree["R2"], "\n")

# Actual vs Predicted
plot(test$rentals, pred.tree,
     main = "Actual vs Predicted — Single Tree",
     xlab = "Actual rentals", ylab = "Predicted rentals",
     col  = rgb(0.2, 0.4, 0.8, 0.3), pch = 16, cex = 0.6)
abline(0, 1, col = "red", lwd = 2)

# Residuals by hour
plot(test$hour, test$rentals - pred.tree,
     main = "Residuals by Hour — Single Tree",
     xlab = "Hour", ylab = "Residuals")
abline(h = 0, col = "blue")


# --- 8. CROSS-VALIDATION AND PRUNING ---------------------------
# Pruning reduces overfitting by removing splits that add little predictive value

set.seed(42)
cv.bike <- cv.tree(tree.bike)

# CV plots: deviance vs size and vs cost-complexity alpha
par(mfrow = c(1, 2))
plot(cv.bike$size, cv.bike$dev, type = "b",
     main = "CV Deviance vs Tree Size",
     xlab = "Terminal nodes", ylab = "CV Deviance",
     col = "steelblue", pch = 16)
plot(cv.bike$k, cv.bike$dev, type = "b",
     main = "CV Deviance vs Cost-Complexity (k)",
     xlab = "k (alpha)", ylab = "CV Deviance",
     col = "darkred", pch = 16)
par(mfrow = c(1, 1))

# Optimal size selected by CV
best.size <- cv.bike$size[which.min(cv.bike$dev)]
cat("Optimal tree size:", best.size, "terminal nodes\n")

# Pruned tree
prune.bike  <- prune.tree(tree.bike, best = best.size)
pred.pruned <- predict(prune.bike, newdata = test)

plot(prune.bike)
text(prune.bike, pretty = 0)
title(paste("Pruned Tree —", best.size, "Terminal Nodes"))

metrics.pruned <- calc_metrics(test$rentals, pred.pruned)
cat("Pruned Tree — Test RMSE:", metrics.pruned["RMSE"],
    "| MAE:", metrics.pruned["MAE"],
    "| R²:",  metrics.pruned["R2"], "\n")

# Comparison: optimal vs forced 8-node tree
pruned.8    <- prune.tree(tree.bike, best = 8)
pred.8      <- predict(pruned.8, newdata = test)
metrics.8   <- calc_metrics(test$rentals, pred.8)

cat("Pruned Tree (optimal,", best.size, "nodes) RMSE:", metrics.pruned["RMSE"], "\n")
cat("Pruned Tree (forced,  8 nodes)  RMSE:", metrics.8["RMSE"], "\n")


# --- 9. RANDOM FOREST ------------------------------------------
# Ensemble of trees; averaging reduces variance vs single tree

set.seed(42)
rf.bike <- randomForest(rentals ~ .,
                        data       = train,
                        ntree      = 500,
                        importance = TRUE)
rf.bike

# OOB error convergence
plot(rf.bike, main = "Random Forest — OOB Error vs Number of Trees")

# OOB-based training RMSE
cat("Random Forest — Train RMSE (OOB):", round(sqrt(rf.bike$mse[500]), 2), "\n")

# Test metrics
pred.rf        <- predict(rf.bike, newdata = test)
metrics.rf     <- calc_metrics(test$rentals, pred.rf)

cat("Random Forest — Test RMSE:", metrics.rf["RMSE"],
    "| MAE:", metrics.rf["MAE"],
    "| R²:",  metrics.rf["R2"], "\n")

# Variable importance
varImpPlot(rf.bike,
           main = "Variable Importance — Random Forest",
           col = "steelblue", pch = 16)

imp_pct <- importance(rf.bike)[, "IncNodePurity"]
imp_pct <- imp_pct / max(imp_pct) * 100
dotchart(sort(imp_pct),
         main = "Variable Importance (%)",
         xlab = "Relative Importance (%)",
         pch = 19, col = "darkgreen")
print(round(sort(imp_pct, decreasing = TRUE), 2))

# Actual vs Predicted
plot(test$rentals, pred.rf,
     main = "Actual vs Predicted — Random Forest",
     xlab = "Actual rentals", ylab = "Predicted rentals",
     col  = rgb(0, 0.4, 0, 0.3), pch = 16, cex = 0.6)
abline(0, 1, col = "red", lwd = 2)

# Residuals vs Fitted
residui.rf <- test$rentals - pred.rf
plot(pred.rf, residui.rf,
     main = "Residuals vs Fitted — Random Forest",
     xlab = "Fitted values", ylab = "Residuals",
     col  = rgb(0, 0.4, 0, 0.3), pch = 16, cex = 0.6)
abline(h = 0, col = "red", lwd = 2)

# Partial dependence: marginal effect of top predictors
p1_data <- partial(rf.bike, pred.var = "hour", train = train)
p1 <- autoplot(p1_data, main = "Partial Dependence — Hour", rug = TRUE, train = train) +
  theme_minimal() +
  labs(y = "Predicted rentals", x = "Hour")
p2_data <- partial(rf.bike, pred.var = "temperature", train = train)
p2 <- autoplot(p2_data, main = "Partial Dependence — Temperature", rug = TRUE, train = train) +
  theme_minimal() +
  labs(y = "Predicted rentals", x = "Temperature")
grid.arrange(p1, p2, ncol = 2)

# Mtry sensitivity: compare p/3 (default), p/2, p (= bagging)
p         <- ncol(train) - 1
mtry_vals <- c(floor(p / 3), floor(p / 2), p)

rmse_mtry <- sapply(mtry_vals, function(m) {
  set.seed(42)
  rf_m  <- randomForest(rentals ~ ., data = train, ntree = 300, mtry = m)
  pred_m <- predict(rf_m, newdata = test)
  sqrt(mean((test$rentals - pred_m)^2))
})

cat("\n=== Mtry sensitivity ===\n")
print(data.frame(mtry       = mtry_vals,
                 label      = c("p/3 (default)", "p/2", "p (bagging)"),
                 RMSE_test  = round(rmse_mtry, 2)))


# --- 10. LINEAR REGRESSION (BASELINE) -------------------------

num_vars <- c("temperature", "humidity", "wind_speed")

# Standardize numerical predictors using training set parameters
preproc        <- preProcess(train[, num_vars], method = c("center", "scale"))
train_lm       <- train
test_lm        <- test
train_lm[, num_vars] <- predict(preproc, train[, num_vars])
test_lm[, num_vars]  <- predict(preproc, test[, num_vars])

# Fit model on all predictors
formula_lm <- as.formula(
  paste("rentals ~", paste(names(train_lm)[names(train_lm) != "rentals"], collapse = " + "))
)
lm_model <- lm(formula_lm, data = train_lm)
summary(lm_model)

# Metrics
pred.lm.train  <- predict(lm_model, newdata = train_lm)
pred.lm        <- predict(lm_model, newdata = test_lm)

metrics.lm.train <- calc_metrics(train_lm$rentals, pred.lm.train)
metrics.lm       <- calc_metrics(test_lm$rentals,  pred.lm)

cat("Linear Regression — Train RMSE:", metrics.lm.train["RMSE"], "\n")
cat("Linear Regression — Test  RMSE:", metrics.lm["RMSE"],
    "| MAE:", metrics.lm["MAE"],
    "| R²:",  metrics.lm["R2"], "\n")


# --- 11. FINAL MODEL COMPARISON --------------------------------

mean_rentals <- mean(test$rentals)

risultati <- data.frame(
  Model = c("Linear Regression", "Single Tree", "Pruned Tree", "Random Forest"),
  rbind(metrics.lm, metrics.tree, metrics.pruned, metrics.rf),
  row.names = NULL
)
risultati$Error_Pct <- round(risultati$RMSE / mean_rentals * 100, 1)

cat("\n===== Model Comparison =====\n")
print(risultati)

# Bar chart: RMSE with % error labels
ggplot(risultati, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = paste0(RMSE, "\n(", Error_Pct, "%)")),
            vjust = -0.3, size = 3.8) +
  scale_fill_manual(values = c("Linear Regression" = "lightcoral",
                               "Single Tree"       = "steelblue",
                               "Pruned Tree"       = "royalblue",
                               "Random Forest"     = "darkgreen")) +
  labs(title    = "Model comparison: RMSE and relative error",
       subtitle = "Values in brackets = % error relative to mean rentals",
       x = "Model", y = "RMSE (rentals)") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# Training vs Test RMSE summary (overfitting check)
overfitting_check <- data.frame(
  Model         = c("Linear Regression", "Single Tree", "Random Forest"),
  RMSE_Train    = c(metrics.lm.train["RMSE"],
                    metrics.tree.train["RMSE"],
                    round(sqrt(rf.bike$mse[500]), 2)),
  RMSE_Test     = c(metrics.lm["RMSE"],
                    metrics.tree["RMSE"],
                    metrics.rf["RMSE"])
)
cat("\n===== Overfitting Check (Train vs Test RMSE) =====\n")
print(overfitting_check)

