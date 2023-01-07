# DA301-Predicting-future-outcomes
### LSE Data Analytics Online Career Accelerator 

# DA301:  Advanced Analytics for Organisational Impact

## Assignment template

### Scenario
You are a data analyst working for Turtle Games, a game manufacturer and retailer. They manufacture and sell their own products, along with sourcing and selling products manufactured by other companies. Their product range includes books, board games, video games and toys. They have a global customer base and have a business objective of improving overall sales performance by utilising customer trends. In particular, Turtle Games wants to understand: 
- how customers accumulate loyalty points (Week 1)
- how useful are remuneration and spending scores data (Week 2)
- can social data (e.g. customer reviews) be used in marketing campaigns (Week 3)
- what is the impact on sales per product (Week 4)
- the reliability of the data (e.g. normal distribution, Skewness, Kurtosis) (Week 5)
- if there is any possible relationship(s) in sales between North America, Europe, and global sales (Week 6).

# Week 1 assignment: Linear regression using Python
The marketing department of Turtle Games prefers Python for data analysis. As you are fluent in Python, they asked you to assist with data analysis of social media data. The marketing department wants to better understand how users accumulate loyalty points. Therefore, you need to investigate the possible relationships between the loyalty points, age, remuneration, and spending scores. Note that you will use this data set in future modules as well and it is, therefore, strongly encouraged to first clean the data as per provided guidelines and then save a copy of the clean data for future use.

## Instructions
1. Load and explore the data.
    1. Create a new DataFrame (e.g. reviews).
    2. Sense-check the DataFrame.
    3. Determine if there are any missing values in the DataFrame.
    4. Create a summary of the descriptive statistics.
2. Remove redundant columns (`language` and `platform`).
3. Change column headings to names that are easier to reference (e.g. `renumeration` and `spending_score`).
4. Save a copy of the clean DataFrame as a CSV file. Import the file to sense-check.
5. Use linear regression and the `statsmodels` functions to evaluate possible linear relationships between loyalty points and age/renumeration/spending scores to determine whether these can be used to predict the loyalty points.
    1. Specify the independent and dependent variables.
    2. Create the OLS model.
    3. Extract the estimated parameters, standard errors, and predicted values.
    4. Generate the regression table based on the X coefficient and constant values.
    5. Plot the linear regression and add a regression line.
6. Include your insights and observations.

## 1. Load and explore the data

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.formula.api import ols

# Load the CSV file(s) as reviews.
reviews = pd.read_csv(r'/Users/christospieris/Documents/LSE Data Analytics/Course 3/turtle_reviews.csv')

# View the DataFrame.
reviews

# Any missing values?
print(reviews.isnull().sum())

# Explore the data.




# Descriptive statistics.

reviews.columns

## 2. Drop columns

# Drop unnecessary columns such as language and platform which hold the same value across rows

reviews = reviews.drop(['language','platform'], axis = 1)


# View column names.

reviews.columns

## 3. Rename columns

# Rename the column headers.
new_column_headers = ['Gender','Age','Income','Spend',
                      'Loyalty','Education','Product',
                     'Review','Summary']
reviews.columns = new_column_headers
# View column names.
reviews.columns

## 4. Save the DataFrame as a CSV file

# Create a CSV file as output.

reviews.to_csv('/Users/christospieris/Documents/LSE Data Analytics/Course 3/turtle_reviews_updated.csv')


# Import new CSV file with Pandas.
reviews_updated = pd.read_csv('/Users/christospieris/Documents/LSE Data Analytics/Course 3/turtle_reviews_updated.csv')

# View DataFrame.
reviews_updated.shape
reviews_updated
revs = reviews_updated
revs

## 5. Linear regression

Use linear regression or multiple linear regression and the statsmodels functions to evaluate possible linear relationships between loyalty points and age/remuneration/spending scores to determine whether these can be used to predict the loyalty points.

### 5a) spending vs loyalty

# Independent variable.
y = revs['Loyalty']
# Dependent variable.
x = revs['Spend']

# OLS model and summary.
f = 'y ~ x'
lab = ols(f, data = revs).fit()
lab.summary()


# Extract the estimated parameters
est = lab.params
print(est)
# Extract the standard errors.
err = lab.bse
print(err)
# Extract the predicted values.
predicted = lab.predict()
print(predicted)

# Set the X coefficient and the constant to generate the regression table.
x_co = est[1]
inter = est[0]
y_predicted = inter + x_co * x

# View the output.
y_predicted

# Plot the graph with a regression line.

plt.scatter(x, y)

# Plot the regression line (in black).
plt.plot(x, y_predicted, color='black')

# Set the x and y limits on the axes.
plt.xlim(0)
plt.ylim(0)

# View the plot.
plt.show()

### 5b) renumeration vs loyalty

# Independent variable.
x = revs['Income']
# Dependent variable.
y = revs['Loyalty']
# OLS model and summary.
f = 'y ~ x'
lab = ols(f, data = revs).fit()
lab.summary()

# Extract the estimated parameters
est = lab.params
print(est)
# Extract the standard errors.
err = lab.bse
print(err)
# Extract the predicted values.
predicted = lab.predict()
print(predicted)

# Set the X coefficient and the constant to generate the regression table.
x_co = est[1]
inter = est[0]
y_predicted = inter + x_co * x

# View the output.
y_predicted

# Plot the graph with a regression line.

plt.scatter(x, y)

# Plot the regression line (in black).
plt.plot(x, y_predicted, color='black')

# Set the x and y limits on the axes.
plt.xlim(0)
plt.ylim(0)

# View the plot.
plt.show()

### 5c) age vs loyalty

# Independent variable.
x = revs['Age']
# Dependent variable.
y = revs['Loyalty']
# OLS model and summary.
f = 'y ~ x'
lab = ols(f, data = revs).fit()
lab.summary()

# Extract the estimated parameters
est = lab.params
print(est)
# Extract the standard errors.
err = lab.bse
print(err)
# Extract the predicted values.
predicted = lab.predict()
print(predicted)

# Set the X coefficient and the constant to generate the regression table.
x_co = est[1]
inter = est[0]
y_predicted = inter + x_co * x
# View the output.
y_predicted

# Plot the graph with a regression line.

plt.scatter(x, y)

# Plot the regression line (in black).
plt.plot(x, y_predicted, color='black')

# Set the x and y limits on the axes.
plt.xlim(0)
plt.ylim(0)

# View the plot.
plt.show()

## 6. Observations and insights

***Your observations here...***








# 

# Week 2 assignment: Clustering with *k*-means using Python

The marketing department also wants to better understand the usefulness of renumeration and spending scores but do not know where to begin. You are tasked to identify groups within the customer base that can be used to target specific market segments. Use *k*-means clustering to identify the optimal number of clusters and then apply and plot the data using the created segments.

## Instructions
1. Prepare the data for clustering. 
    1. Import the CSV file you have prepared in Week 1.
    2. Create a new DataFrame (e.g. `df2`) containing the `renumeration` and `spending_score` columns.
    3. Explore the new DataFrame. 
2. Plot the renumeration versus spending score.
    1. Create a scatterplot.
    2. Create a pairplot.
3. Use the Silhouette and Elbow methods to determine the optimal number of clusters for *k*-means clustering.
    1. Plot both methods and explain how you determine the number of clusters to use.
    2. Add titles and legends to the plot.
4. Evaluate the usefulness of at least three values for *k* based on insights from the Elbow and Silhoutte methods.
    1. Plot the predicted *k*-means.
    2. Explain which value might give you the best clustering.
5. Fit a final model using your selected value for *k*.
    1. Justify your selection and comment on the respective cluster sizes of your final solution.
    2. Check the number of observations per predicted class.
6. Plot the clusters and interpret the model.

## 1. Load and explore the data

# Import necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')

# Load the CSV file(s) as df2.
imp = pd.read_csv(r'/Users/christospieris/Documents/LSE Data Analytics/Course 3/turtle_reviews_updated.csv')
df2 = imp[['Income','Spend']]
# View DataFrame.
df2

# Explore the data.


# Descriptive statistics.


## 2. Plot

# Create a scatterplot with Seaborn.
sns.scatterplot(data = df2, x = 'Income', y = 'Spend')

# Create a pairplot with Seaborn.
sns.pairplot(df2)

## 3. Elbow and silhoutte methods

# Determine the number of clusters: Elbow method.
clustersize = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++', 
                    max_iter = 500,
                    n_init = 10,
                    random_state = 42)
    kmeans.fit(df2)
    clustersize.append(kmeans.inertia_)

plt.plot(range(1, 11),
         clustersize,
         marker='o')

plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Cluster Size")

plt.show()

# Determine the number of clusters: Silhouette method.
sil = []
kmax = 10

for k in range(2, kmax+1):
    kmeans_s = KMeans(n_clusters = k).fit(df2)
    labels = kmeans_s.labels_
    sil.append(silhouette_score(df2,
                                labels,
                                metric = 'euclidean'))

# Plot the silhouette method.
plt.plot(range(2, kmax+1),
         sil,
         marker='o')

plt.title("The Silhouette Method")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette")

plt.show()

## 4. Evaluate k-means model at different values of *k*

# Use 3 clusters:
kmeans = KMeans(n_clusters = 3,
                max_iter = 15000,
                init='k-means++',
                random_state=42).fit(df2)

clusters = kmeans.labels_
df2['K-Means Predicted'] = clusters

# Plot the predicted.
sns.pairplot(df2,
             hue='K-Means Predicted',
             diag_kind= 'kde')

# Check the number of observations per predicted class.
df2['K-Means Predicted'].value_counts()

# Use 4 clusters:
kmeans = KMeans(n_clusters = 4,
                max_iter = 15000,
                init='k-means++',
                random_state=42).fit(df2)

clusters = kmeans.labels_
df2['K-Means Predicted'] = clusters

# Plot the predicted.
sns.pairplot(df2,
             hue='K-Means Predicted',
             diag_kind= 'kde')

# Check the number of observations per predicted class.
df2['K-Means Predicted'].value_counts()

# Use 5 clusters:
kmeans = KMeans(n_clusters = 5,
                max_iter = 15000,
                init='k-means++',
                random_state=42).fit(df2)

clusters = kmeans.labels_
df2['K-Means Predicted'] = clusters

# Plot the predicted.
sns.pairplot(df2,
             hue='K-Means Predicted',
             diag_kind= 'kde')

## 5. Fit final model and justify your choice

# Apply the final model.


# Check the number of observations per predicted class.


## 6. Plot and interpret the clusters

# Visualising the clusters.


# View the DataFrame.


# Visualising the clusters.



## 7. Discuss: Insights and observations

***Your observations here...***



# 

# Week 3 assignment: NLP using Python
Customer reviews were downloaded from the website of Turtle Games. This data will be used to steer the marketing department on how to approach future campaigns. Therefore, the marketing department asked you to identify the 15 most common words used in online product reviews. They also want to have a list of the top 20 positive and negative reviews received from the website. Therefore, you need to apply NLP on the data set.

## Instructions
1. Load and explore the data. 
    1. Sense-check the DataFrame.
    2. You only need to retain the `review` and `summary` columns.
    3. Determine if there are any missing values.
2. Prepare the data for NLP
    1. Change to lower case and join the elements in each of the columns respectively (`review` and `summary`).
    2. Replace punctuation in each of the columns respectively (`review` and `summary`).
    3. Drop duplicates in both columns (`review` and `summary`).
3. Tokenise and create wordclouds for the respective columns (separately).
    1. Create a copy of the DataFrame.
    2. Apply tokenisation on both columns.
    3. Create and plot a wordcloud image.
4. Frequency distribution and polarity.
    1. Create frequency distribution.
    2. Remove alphanumeric characters and stopwords.
    3. Create wordcloud without stopwords.
    4. Identify 15 most common words and polarity.
5. Review polarity and sentiment.
    1. Plot histograms of polarity (use 15 bins) for both columns.
    2. Review the sentiment scores for the respective columns.
6. Identify and print the top 20 positive and negative reviews and summaries respectively.
7. Include your insights and observations.

## 1. Load and explore the data

!pip install nltk

!pip install wordcloud

!pip install textblob

# Import all the necessary packages.
import pandas as pd
import numpy as np
import nltk 
import os 
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download ('punkt')
nltk.download ('stopwords')


from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from textblob import TextBlob
from scipy.stats import norm

# Import Counter.
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

# Load the data set as df3.
imp3 = pd.read_csv(r'/Users/christospieris/Documents/LSE Data Analytics/Course 3/turtle_reviews_updated.csv')

# View DataFrame.
imp3


# Explore data set.
imp3.describe

# Keep necessary columns. Drop unnecessary columns.
df3 = imp3[['Review','Summary']]

# View DataFrame.
df3

# Drop any rows with any missing values.
df3 = df3.dropna(axis=0)
df3

## 2. Prepare the data for NLP
### 2a) Change to lower case and join the elements in each of the columns respectively (review and summary)

# Review: Change all to lower case and join with a space.
# Transform 'Review' column to lowercase.
df3['Review'] = df3['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Preview the 'Review'column.
df3['Review'].head()

# Summary: Change all to lower case and join with a space.
# Transform 'Review' column to lowercase.
df3['Summary'] = df3['Summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Preview the 'Review'column.
df3['Summary'].head()

### 2b) Replace punctuation in each of the columns respectively (review and summary)

# Remove punctuation in Review column.
df3['Review'] = df3['Review'].str.replace('[^\w\s]','')
# Preview the result.
df3['Review'].head()

# Remove punctuation in Summary column.
df3['Summary'] = df3['Summary'].str.replace('[^\w\s]','')
# Preview the result.
df3['Summary'].head()

### 2c) Drop duplicates in both columns

# Check the number of duplicate values in the Review column.
print(df3.Review.duplicated().sum())
# Check the number of duplicate values in the Summary column.
print(df3.Summary.duplicated().sum())
# Drop duplicates in both columns.
df4 = df3.drop_duplicates(subset=['Review']).reset_index(drop = True)
df5 = df4.drop_duplicates(subset=['Summary']).reset_index(drop = True)
# View DataFrame.
# Check the number of duplicate values in the Review column.
print(df5.Review.duplicated().sum())
# Check the number of duplicate values in the Summary column.
print(df5.Summary.duplicated().sum())
df5

## 3. Tokenise and create wordclouds

# Create new DataFrame (copy DataFrame).
df6=df5
# View DataFrame.
df6

# Apply tokenisation to both columns.
df6['review_tokens'] = df6['Review'].apply(word_tokenize)
df6['summary_tokens'] = df6['Summary'].apply(word_tokenize)
df6.head()
# Create empty lists for each column.
review_token_list = []
summary_token_list = []
# Generate token lists for each column.
for i in range(df6.shape[0]):
    # Add each token to the list.
    review_token_list = review_token_list + df6['review_tokens'][i]
    summary_token_list = summary_token_list +df6['summary_tokens'][i]



# Create variables holding the entire text of each column
all_reviews = ''
all_summary = ''
for i in range(df6.shape[0]):
    all_reviews = all_reviews + df6['Review'][i]
    all_summary = all_summary + df6['Summary'][i]
# View Variables.
all_reviews
all_summary

# Review: Create a word cloud.

# Set the colour palette.
sns.set(color_codes=True)

# Create a WordCloud object.
word_cloud_review1 = WordCloud(width = 1600, height = 900, 
                background_color ='white',
                colormap = 'plasma', 
                stopwords = 'none',
                min_font_size = 10).generate(all_reviews) 

# Review: Plot the WordCloud image.
                  
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(word_cloud_review1) 
plt.axis('off') 
plt.tight_layout(pad = 0) 
plt.show()



# Summary: Create a word cloud.

# Set the colour palette.
sns.set(color_codes=True)

# Create a WordCloud object.
word_cloud_summary1 = WordCloud(width = 1600, height = 900, 
                background_color ='white',
                colormap = 'plasma', 
                stopwords = 'none',
                min_font_size = 10).generate(all_summary) 

# Summary: Plot the WordCloud image.
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(word_cloud_summary1) 
plt.axis('off') 
plt.tight_layout(pad = 0) 
plt.show()

## 4. Frequency distribution and polarity
### 4a) Create frequency distribution

# Calculate the frequency distribution for text in Review column
fdist_review = FreqDist(review_token_list)
fdist_summary = FreqDist(summary_token_list)
# Preview data.
fdist_summary,fdist_review

### 4b) Remove alphanumeric characters and stopwords

# Delete all the alpanum.
review_tokens = [word for word in review_token_list if word.isalnum()]
summary_tokens = [word for word in summary_token_list if word.isalnum()]

# Remove all the stopwords

# Create a set of English stop words.
english_stopwords = set(stopwords.words('english'))

# Create a filtered list of tokens without stop words.
filtered_review_tokens = [x for x in review_tokens if x.lower() not in english_stopwords]
filtered_summary_tokens = [y for y in summary_tokens if y.lower() not in english_stopwords]
# Define an empty string variable.
review_token_string = ''
summary_token_string = ''
for value in filtered_review_tokens:
    # Add each filtered token word to the string.
    review_token_string = review_token_string + value + ' '
for value in filtered_summary_tokens:
    # Add each filtered token word to the string.
    summary_token_string = summary_token_string + value + ' '

### 4c) Create wordcloud without stopwords

# Create a wordcloud without stop words.
# Set the colour palette.
sns.set(color_codes=True)
# Review: Create a word cloud.
word_cloud_review2 = WordCloud(width = 1600, height = 900, 
                background_color ='white',
                colormap = 'plasma', 
                stopwords = 'none',
                min_font_size = 10).generate(review_token_string) 
# Summary: Create a word cloud.
word_cloud_summary2 = WordCloud(width = 1600, height = 900, 
                background_color ='white',
                colormap = 'plasma', 
                stopwords = 'none',
                min_font_size = 10).generate(summary_token_string) 

# Review: Plot the WordCloud image.
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(word_cloud_review2) 
plt.axis('off') 
plt.tight_layout(pad = 0) 
plt.show()

# Review: Plot the WordCloud image.
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(word_cloud_summary2) 
plt.axis('off') 
plt.tight_layout(pad = 0) 
plt.show()

### 4d) Identify 15 most common words and polarity

# Determine the frequency of words in each column
# Generate a DataFrame from Counter.
review_counts = pd.DataFrame(Counter(filtered_review_tokens).most_common(15),
                      columns=['Word', 'Frequency']).set_index('Word').reset_index()
review_counts.head()
# Determine the 15 most common words in Reviews
# Generate a DataFrame from Counter.
summary_counts = pd.DataFrame(Counter(filtered_summary_tokens).most_common(15),
                      columns=['Word', 'Frequency']).set_index('Word').reset_index()
# Preview data.
counter_outerjoin = pd.concat([review_counts,summary_counts],axis = 1, join='outer')
#counter_innerjoin = pd.concat([review_counts,summary_counts],axis = 1, join='inner')
counter_outerjoin

## 5. Review polarity and sentiment: Plot histograms of polarity (use 15 bins) and sentiment scores for the respective columns.

# Provided function.
def generate_polarity(entry):
    '''Extract polarity score (-1 to +1) for each comment'''
    return TextBlob(entry).sentiment[0]

# Determine polarity of both columns. 
# Populate a new column with polarity scores for each Review.
df6['Review_polarity'] = df6['Review'].apply(generate_polarity)
# Populate a new column with polarity scores for each Summary.
df6['Summary_polarity'] = df6['Summary'].apply(generate_polarity)
# Preview the result.
df6.head()

# Define a function to extract a subjectivity score for the comment.
def generate_subjectivity(entry):
    return TextBlob(entry).sentiment[1]
# Populate a new column with subjectivity scores for each comment.
df6['Review_Subjectivity'] = df6['Review'].apply(generate_subjectivity)
# Populate a new column with polarity scores for each Summary.
df6['Summary_Subjectivity'] = df6['Summary'].apply(generate_subjectivity)
# Preview the result.
df6.head()

# Review: Create a histogram plot with bins = 15.
# Histogram of polarity

# Set the number of bins.
num_bins = 15
# Set the plot area.
plt.figure(figsize=(16,9))
# Define the bars.
n, bins, patches = plt.hist(df6['Review_polarity'], num_bins, facecolor='red', alpha=0.6)
# Set the labels.
plt.xlabel('Polarity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Histogram of sentiment score polarity in Reviews', fontsize=20)
plt.show()
# Histogram of sentiment score
# Set the number of bins.
num_bins = 15
# Set the plot area.
plt.figure(figsize=(16,9))
# Define the bars.
n, bins, patches = plt.hist(df6['Review_Subjectivity'], num_bins, facecolor='red', alpha=0.6)
# Set the labels.
plt.xlabel('Subjectivity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Histogram of sentiment score subjectivity in Reviews', fontsize=20)
plt.show()

# Summary: Create a histogram plot with bins = 15.
# Histogram of polarity

# Set the number of bins.
num_bins = 15
# Set the plot area.
plt.figure(figsize=(16,9))
# Define the bars.
n, bins, patches = plt.hist(df6['Summary_polarity'], num_bins, facecolor='red', alpha=0.6)
# Set the labels.
plt.xlabel('Polarity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Histogram of sentiment score polarity in Summaries', fontsize=20)
plt.show()
# Histogram of sentiment score
# Set the number of bins.
num_bins = 15
# Set the plot area.
plt.figure(figsize=(16,9))
# Define the bars.
n, bins, patches = plt.hist(df6['Summary_Subjectivity'], num_bins, facecolor='red', alpha=0.6)
# Set the labels.
plt.xlabel('Subjectivity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Histogram of sentiment score subjectivity in Summaries', fontsize=20)
plt.show()

## 6. Identify top 20 positive and negative reviews and summaries respectively

# Top 20 positive reviews.
positive_reviews = df6.nlargest(20,'Review_polarity')
# Select relevant columns to view
positive_reviews = positive_reviews[['Review','Review_polarity']]
# Adjust the column width.
positive_reviews.style.set_properties(subset=['Review'], **{'width': '1200px'})
# View output.
positive_reviews

# Top 20 positive summaries.
positive_summaries = df6.nlargest(20,'Summary_polarity')
# Select relevant columns to view
positive_summaries = positive_summaries[['Summary','Summary_polarity']]
# View output.
positive_summaries

# Top 20 negative reviews.
negative_reviews = df6.nsmallest(20,'Review_polarity')
# Select relevant columns to view
negative_reviews = negative_reviews[['Review','Review_polarity']]
# Adjust the column width.
negative_reviews.style.set_properties(subset=['Review'], **{'width': '1200px'})
# View output.
negative_reviews

# Top 20 negative summaries.
negative_summaries = df6.nsmallest(20,'Summary_polarity')
# Select relevant columns to view
negative_summaries = negative_summaries[['Summary','Summary_polarity']]
# Adjust the column width.
negative_summaries.style.set_properties(subset=['Summary'], **{'width': '1200px'})
# View output.
negative_summaries

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
qplot(EU_Sales,Global_Sales,data=sales_data)
qplot(NA_Sales,Global_Sales,data=sales_data)
qplot(NA_Sales,EU_Sales,data=sales_data)
## 2b) Histograms
# Create histograms.
qplot(Product,Global_Sales,data=sales_data)

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
ggplot(grouped_sales_data, aes(x=Product, y=sum_NA)) + 
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
       mapping = aes(x = NA_Sales, y = G
                     lobal_Sales)) +
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





