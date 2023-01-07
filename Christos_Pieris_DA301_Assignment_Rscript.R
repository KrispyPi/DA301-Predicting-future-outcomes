## LSE Data Analytics Online Career Accelerator 

# DA301:  Advanced Analytics for Organisational Impact

###############################################################################

# Assignment template

## Scenario
## You are a data analyst working for Turtle Games, a game manufacturer and 
## retailer. They manufacture and sell their own products, along with sourcing
## and selling products manufactured by other companies. Their product range 
## includes books, board games, video games and toys. They have a global 
## customer base and have a business objective of improving overall sales 
##performance by utilising customer trends. 

## In particular, Turtle Games wants to understand:
## - how customers accumulate loyalty points (Week 1)
## - how useful are remuneration and spending scores data (Week 2)
## - can social data (e.g. customer reviews) be used in marketing 
##     campaigns (Week 3)
## - what is the impact on sales per product (Week 4)
## - the reliability of the data (e.g. normal distribution, Skewness, Kurtosis)
##     (Week 5)
## - if there is any possible relationship(s) in sales between North America,
##     Europe, and global sales (Week 6).

################################################################################

# Week 4 assignment: EDA using R

## The sales department of Turtle games prefers R to Python. As you can perform
## data analysis in R, you will explore and prepare the data set for analysis by
## utilising basic statistics and plots. Note that you will use this data set 
## in future modules as well and it is, therefore, strongly encouraged to first
## clean the data as per provided guidelines and then save a copy of the clean 
## data for future use.

# Instructions
# 1. Load and explore the data.
##  - Remove redundant columns (Ranking, Year, Genre, Publisher) by creating 
##      a subset of the data frame.
##  - Create a summary of the new data frame.
# 2. Create plots to review and determine insights into data set.
##  - Create scatterplots, histograms and boxplots to gain insights into
##      the Sales data.
##  - Note your observations and diagrams that could be used to provide
##      insights to the business.
# 3. Include your insights and observations.

###############################################################################

# 1. Load and explore the data

# Install and import Tidyverse.
install.packages('tidyverse')
library(tidyverse)
# Import the data set.
setwd(dir='/Users/christospieris/Documents/LSE Data Analytics')
data_og <- data.frame(read.csv(file.choose(),header = T))
# Print the data frame.
View(data_og)
# Create a new data frame from a subset of the sales data frame.
# Remove unnecessary columns. 
sales_data <- subset(data_og, select = -c(Ranking, Year, Genre, Publisher))
# View the data frame.
View(sales_data)
# View the descriptive statistics.
summary(sales_data)

################################################################################
# 2. Review plots to determine insights into the data set.

## 2a) Scatterplots
# Create scatterplots.
qplot(Platform,Global_Sales,data=sales_data)

## 2b) Histograms
# Create histograms.
qplot(Platform,data=sales_data,geom = 'bar')

## 2c) Boxplots
# Create boxplots.
qplot(Platform,Global_Sales,data=sales_data,geom = 'boxplot')

###############################################################################

# 3. Observations and insights

## Your observations and insights here ......




###############################################################################
###############################################################################


# Week 5 assignment: Cleaning and manipulating data using R

## Utilising R, you will explore, prepare and explain the normality of the data
## set based on plots, Skewness, Kurtosis, and a Shapiro-Wilk test. Note that
## you will use this data set in future modules as well and it is, therefore, 
## strongly encouraged to first clean the data as per provided guidelines and 
## then save a copy of the clean data for future use.

## Instructions
# 1. Load and explore the data.
##  - Continue to use the data frame that you prepared in the Week 4 assignment. 
##  - View the data frame to sense-check the data set.
##  - Determine the `min`, `max` and `mean` values of all the sales data.
##  - Create a summary of the data frame.
# 2. Determine the impact on sales per product_id.
##  - Use the group_by and aggregate functions to sum the values grouped by
##      product.
##  - Create a summary of the new data frame.
# 3. Create plots to review and determine insights into the data set.
##  - Create scatterplots, histograms, and boxplots to gain insights into 
##     the Sales data.
##  - Note your observations and diagrams that could be used to provide 
##     insights to the business.
# 4. Determine the normality of the data set.
##  - Create and explore Q-Q plots for all sales data.
##  - Perform a Shapiro-Wilk test on all the sales data.
##  - Determine the Skewness and Kurtosis of all the sales data.
##  - Determine if there is any correlation between the sales data columns.
# 5. Create plots to gain insights into the sales data.
##  - Compare all the sales data (columns) for any correlation(s).
##  - Add a trend line to the plots for ease of interpretation.
# 6. Include your insights and observations.

################################################################################

# 1. Load and explore the data

# View data frame created in Week 4.
View(sales_data)

# Check output: Determine the min, max, and mean values.

# Call the function to calculate the mean.
mean(sales_data$NA_Sales) 
mean(sales_data$EU_Sales) 
mean(sales_data$Global_Sales) 
# Call the function to calculate the median.
median(sales_data$NA_Sales) 
median(sales_data$EU_Sales) 
median(sales_data$Global_Sales)
# Determine the minimum and maximum value.
min(sales_data$NA_Sales) 
min(sales_data$EU_Sales) 
min(sales_data$Global_Sales)
max(sales_data$NA_Sales) 
max(sales_data$EU_Sales) 
max(sales_data$Global_Sales)
# Use the summary() function.
summary(sales_data)
# View the descriptive statistics.


###############################################################################

# 2. Determine the impact on sales per product_id.

## 2a) Use the group_by and aggregate functions.
# Group data based on Product and determine the sum per Product.
grouped_sales_data <- sales_data %>%
  group_by(Product) %>% 
  summarise(sum_NA=sum(NA_Sales),
            sum_EU=sum(EU_Sales),
            sum_Global=sum(Global_Sales))

# View the data frame.
View(grouped_sales_data)

# Explore the data frame.
summary(grouped_sales_data)


## 2b) Determine which plot is the best to compare game sales.
# Create scatterplots.
ggplot(grouped_sales_data, aes(x=sum_EU, y=sum_NA)) + 
  geom_point()
# Create histograms.
hist(grouped_sales_data$sum_EU)
hist(grouped_sales_data$sum_NA)
hist(grouped_sales_data$sum_Global)
# Create boxplots.
boxplot(grouped_sales_data$sum_EU)
boxplot(grouped_sales_data$sum_NA)
boxplot(grouped_sales_data$sum_Global)

###############################################################################


# 3. Determine the normality of the data set.

## 3a) Create Q-Q Plots
# Create Q-Q Plots.
qqnorm(grouped_sales_data$sum_EU)
# Specify qqline function.
qqline(grouped_sales_data$sum_EU)

# Create Q-Q Plots.
qqnorm(grouped_sales_data$sum_NA)
# Specify qqline function.
qqline(grouped_sales_data$sum_NA)

# Create Q-Q Plots.
qqnorm(grouped_sales_data$sum_Global)
# Specify qqline function.
qqline(grouped_sales_data$sum_Global)

## 3b) Perform Shapiro-Wilk test
# Install and import Moments.
library(moments)
# Perform Shapiro-Wilk test.
shapiro.test(grouped_sales_data$sum_EU)
shapiro.test(grouped_sales_data$sum_NA)
shapiro.test(grouped_sales_data$sum_Global)

## 3c) Determine Skewness and Kurtosis
# Skewness and Kurtosis.
skewness(grouped_sales_data$sum_EU) 
kurtosis(grouped_sales_data$sum_EU)
# Skewness and Kurtosis.
skewness(grouped_sales_data$sum_NA) 
kurtosis(grouped_sales_data$sum_NA)
# Skewness and Kurtosis.
skewness(grouped_sales_data$sum_Global) 
kurtosis(grouped_sales_data$sum_Global)

## 3d) Determine correlation
# Determine correlation.
cor(grouped_sales_data$sum_EU,grouped_sales_data$sum_Global)
cor(grouped_sales_data$sum_NA,grouped_sales_data$sum_Global)
cor(grouped_sales_data$sum_NA,grouped_sales_data$sum_EU)
###############################################################################

# 4. Plot the data
# Create plots to gain insights into data.
# Choose the type of plot you think best suits the data set and what you want 
# to investigate. Explain your answer in your report.

ggplot(data=sales_data,
       mapping = aes(x = NA_Sales, y = Global_Sales)) +
  geom_point( alpha = 0.5, size = 1.5) +
  geom_smooth(method = 'lm',se = FALSE, size = 1.5)

ggplot(data=sales_data,
       mapping = aes(x = EU_Sales, y = Global_Sales)) +
  geom_point( alpha = 0.5, size = 1.5) +
  geom_smooth(method = 'lm',se = FALSE, size = 1.5)
###############################################################################

# 5. Observations and insights
# Your observations and insights here...



###############################################################################
###############################################################################

# Week 6 assignment: Making recommendations to the business using R

## The sales department wants to better understand if there is any relationship
## between North America, Europe, and global sales. Therefore, you need to
## investigate any possible relationship(s) in the sales data by creating a 
## simple and multiple linear regression model. Based on the models and your
## previous analysis (Weeks 1-5), you will then provide recommendations to 
## Turtle Games based on:
##   - Do you have confidence in the models based on goodness of fit and
##        accuracy of predictions?
##   - What would your suggestions and recommendations be to the business?
##   - If needed, how would you improve the model(s)?
##   - Explain your answers.

# Instructions
# 1. Load and explore the data.
##  - Continue to use the data frame that you prepared in the Week 5 assignment. 
# 2. Create a simple linear regression model.
##  - Determine the correlation between the sales columns.
##  - View the output.
##  - Create plots to view the linear regression.
# 3. Create a multiple linear regression model
##  - Select only the numeric columns.
##  - Determine the correlation between the sales columns.
##  - View the output.
# 4. Predict global sales based on provided values. Compare your prediction to
#      the observed value(s).
##  - NA_Sales_sum of 34.02 and EU_Sales_sum of 23.80.
##  - NA_Sales_sum of 3.93 and EU_Sales_sum of 1.56.
##  - NA_Sales_sum of 2.73 and EU_Sales_sum of 0.65.
##  - NA_Sales_sum of 2.26 and EU_Sales_sum of 0.97.
##  - NA_Sales_sum of 22.08 and EU_Sales_sum of 0.52.
# 5. Include your insights and observations.

###############################################################################

# 1. Load and explore the data
# View data frame created in Week 5.
View(sales_data)
View(grouped_sales_data)

# Determine a summary of the data frame.
summary(sales_data)
summary(grouped_sales_data)

##############################################################################

# 2. Create a simple linear regression model
## 2a) Determine the correlation between columns
# Create a linear regression model on the original data.
cor(grouped_sales_data)
cor(sales_data$NA_Sales,sales_data$EU_Sales)
cor(grouped_sales_data$sum_EU,grouped_sales_data$sum_NA)


Simple_Linear_Regression_Model <- lm(EU_Sales~NA_Sales,
             data=sales_data)
Simple_Linear_Regression_Model
## 2b) Create a plot (simple linear regression)
# Basic visualization.

# View residuals on a plot.
plot(Simple_Linear_Regression_Model$residuals)

# Plot the relationship with base R graphics.
plot(sales_data$EU_Sales,sales_data$NA_Sales)
coefficients(Simple_Linear_Regression_Model)


# Add line-of-best-fit.
abline(coefficients(Simple_Linear_Regression_Model))

###############################################################################

# 3. Create a multiple linear regression model
# Select only numeric columns from the original data frame.
multi_linear_regression_model <- lm(sum_Global~sum_NA + sum_EU, data = grouped_sales_data)
# Multiple linear regression model.
summary(multi_linear_regression_model)

###############################################################################

# 4. Predictions based on given values
# Compare with observed values for a number of records.

df_test <- data.frame (sum_NA  = c(34.02,3.93,2.73,2.26,22.08),
                       sum_EU = c(23.8,1.56,0.65,0.97,0.52))
df_test 
predict_test = predict(multi_linear_regression_model, newdata = df_test,
                       interval='confidence')
predict_test

###############################################################################

# 5. Observations and insights
# Your observations and insights here...



###############################################################################
###############################################################################




