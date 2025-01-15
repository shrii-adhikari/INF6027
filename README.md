# News Category Analysis and Clickbait Detection

## Overview

This project involves analyzing a news dataset to explore category distributions, author statistics, and detect clickbait. It includes data preprocessing, sentiment analysis, similarity computations, and clustering.

## Dataset

The dataset is a JSON file containing news articles with the following columns: `category`, `authors`, `headline`, `short_description`, and `year`.

## Key Features of the Code

1. **Data Loading and Exploration**  
   - Reads the dataset using `jsonlite::stream_in`.
   - Displays data structure, unique categories, and frequency distributions.

2. **Category and Author Analysis**  
   - Visualizes category frequencies with bar charts.
   - Identifies top 10 authors based on article count.

3. **Data Cleaning**  
   - Handles missing values in `headline` and `short_description`.
   - Cleans text by removing punctuation, emojis, and performing lemmatization.

4. **Text Similarity Calculations**  
   - Computes cosine and Jaccard similarities between `headline` and `short_description`.

5. **Clickbait Detection**  
   - Flags articles as clickbait if headline and description have no similarity and mismatched sentiment.

6. **Sentiment Analysis**  
   - Performs sentiment analysis using the `syuzhet` package.

7. **Clustering**  
   - Applies K-means clustering on similarity metrics to group articles.

8. **Trend Analysis**  
   - Examines publication trends over time for top authors.
   - Identifies authors with high proportions of clickbait articles.

9. **Export**  
   - Saves cleaned data to an Excel file.

## Visualizations

- Bar charts for category and author frequencies.
- Scatter plots for similarity metrics.
- Line plots for publication trends.
- Clustering and silhouette plots.

## Dependencies

The following R packages are required:  
`dplyr`, `ggplot2`, `jsonlite`, `tm`, `stringr`, `textstem`, `syuzhet`, `cluster`, `writexl`, `SnowballC`, and `tidyr`.

## How to Run

1. Install the required R packages:  
   ```R
   install.packages(c("dplyr", "ggplot2", "jsonlite", "tm", "stringr", "textstem", "syuzhet", "cluster", "writexl", "SnowballC", "tidyr"))
   ```

2. Load the script in R and execute it line by line.

3. Ensure the dataset path is correct when loading the JSON file.

4. View the generated plots and outputs.

## Conclusion

This analysis provides insights into news categories, author contributions, and identifies potential clickbait articles, using a combination of statistical and machine learning techniques.
