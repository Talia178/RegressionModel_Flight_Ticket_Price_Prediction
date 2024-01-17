# Random Forest Regression Model----
# Load required libraries
install.packages("randomForest")  
install.packages("doParallel")
library(randomForest)
library(doParallel)
library(ggplot2)

# Read data file
file_path <- "C:/Users/trant/OneDrive/Documents/MIT_DA_Tri2_2023/Applied Data Mining/Assignment/processed_flight_data.csv"
flight_data <- read.csv(file_path)

# Define features and target variable
features <- c('Total.Hours', 'Total.Minutes', 'elapsedDays', 'isBasicEconomy',
              'isRefundable', 'isNonStop', 'seatsRemaining', 'totalTravelDistance')
target <- 'totalFare'

# Split data into training set and test set
set.seed(17)  # for reproducibility

sample_indices <- sample(1:nrow(flight_data), 0.5 * nrow(flight_data))

train_data <- flight_data[sample_indices, ]
test_data <- flight_data[-sample_indices, ]

# Set up parallel processing
num_cores <- detectCores()
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Train the Random Forest model with parallel processing
rf_model <- randomForest(train_data[, features], train_data[[target]],
                         ntree = 100, mtry = 2, nodesize = 1, doParallel = TRUE)

# Make predictions on the test data
predictions <- predict(rf_model, test_data[, features])

# Evaluation Metrics----
# Calculate evaluation metrics: R-squared (R²), Mean Absolute Error (MAE), and Mean Percentage Error (MPE)
rsquared <- 1 - sum((predictions - test_data[[target]])^2) / sum((test_data[[target]] - mean(test_data[[target]]))^2)
mae <- mean(abs(predictions - test_data[[target]]))
mpe <- mean((predictions - test_data[[target]]) / test_data[[target]]) * 100

# Print evaluation metrics
print(paste("R-squared (R²) Score:", rsquared))
print(paste("Mean Absolute Error (MAE):", mae))
print(paste("Mean Percentage Error (MPE):", mpe))

# Visualization----
# Visualize residual values with histogram
residuals <- test_data[[target]] - predictions
ggplot(data = data.frame(residuals = residuals), aes(x = residuals)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "black") +
  labs(x = "Residuals", y = "Frequency", title = "Histogram of Residuals")

