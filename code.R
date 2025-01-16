# Install necessary libraries
if (!require(dplyr)) install.packages("dplyr")
if (!require(jsonlite)) install.packages("jsonlite")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(tm)) install.packages("tm")
if (!require(stringr)) install.packages("stringr")
if (!require(textstem)) install.packages("textstem")
if (!require(SnowballC)) install.packages("SnowballC")
if (!require(syuzhet)) install.packages("syuzhet")
if (!require(writexl)) install.packages("writexl")


# Load necessary libraries
library(dplyr)  #Data manipulation and transformation
library(jsonlite) #Parsing and generating JSON data
library(ggplot2)  # Data visualisation and plotting
library(tm) # Data visualisation and plotting
library(stringr) #String manipulation and pattern matching
library(textstem) #Stemming and lemmatisation of text data
library(SnowballC) # Stemming
library(syuzhet)  # For sentiment analysis
library(writexl) #To export dataframe into excel file


# Load the dataset
#steam_in(used to read large JOSN files)
news_data <- stream_in(file("C:\\Users\\acer\\Desktop\\UOS\\ITDS\\ASSESSMENT\\New folder\\News_Category_Dataset_v3.json"))

# View structure of the data frame
str(news_data)

# View column names
colnames(news_data)

# View first few rows
head(news_data)


#Data Exploration
# Display the unique categories in new_data
categories <- unique(news_data$category)
print(categories)

# Calculate the frequency of each category
category_frequency <- news_data %>%
  group_by(category) %>% 
  summarize(frequency = n()) %>% 
  arrange(desc(frequency))

# View the frequency table
print(category_frequency)

# Category frequency plot
category_plot <- ggplot(category_frequency, aes(x = reorder(category, -frequency), y = frequency)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Frequency of News Categories", x = "Category", y = "Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = frequency), vjust = -0.5, size = 3.5)

print(category_plot)


#fitering data based on authors
# Filter out rows where 'authors' is NULL or empty before calculating frequency
author_frequency <- news_data %>%
  filter(!is.na(authors) & authors != "") %>%
  group_by(authors) %>%
  summarize(frequency = n()) %>%
  arrange(desc(frequency))

# View the frequency table
print(author_frequency)

# Get the top 10 authors based on frequency using slice_max()
top_10_authors <- author_frequency %>%
  slice_max(frequency, n = 10)
# View the top 10 authors
print(top_10_authors)

# Plot the frequency of the top 10 authors
top_10_author_plot <- ggplot(top_10_authors, aes(x = reorder(authors, -frequency), y = frequency)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  labs(title = "Top 10 Authors", x = "Author", y = "Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = frequency), vjust = -0.5, size = 3.5)

# Display the plot
print(top_10_author_plot)


# Calculate the sum of the frequencies for the top 10 authors
sum_top_10_authors <- sum(top_10_authors$frequency)

# Print the sum
cat("Sum of frequencies for the top 10 authors:", sum_top_10_authors, "\n")

#as the new sample size is 15641, which is a viable dataset
# Filter the original dataset to include only the top 10 authors
top_10_authors_data <- news_data %>%
  filter(authors %in% top_10_authors$authors)



#Extracting "year"
# Convert the 'date' column to Date format (if not already)
top_10_authors_data$date <- as.Date(top_10_authors_data$date, format = "%Y-%m-%d")

# Extract the year from the 'date' column
top_10_authors_data$year <- format(top_10_authors_data$date, "%Y")

# Count the number of articles published by each author in each year
publication_trend <- top_10_authors_data %>%
  group_by(authors, year) %>%
  summarize(article_count = n(), .groups = "drop")


# Plot the trend of article publications over time
publication_trend_plot <- ggplot(publication_trend, aes(x = as.numeric(year), y = article_count, color = authors)) +
  geom_line(size = 1) +  # Line for each author
  geom_point(size = 2) +  # Points for each year
  labs(
    title = "Trend of Article Publications by Top 10 Authors",
    x = "Year",
    y = "Number of Articles Published",
    color = "Authors"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels for readability
    legend.position = "bottom"  # Place legend at the bottom
  ) +
  scale_x_continuous(breaks = unique(as.numeric(publication_trend$year)))  # Adjust x-axis breaks

# Display the plot
print(publication_trend_plot)




#DATA CLEANING

# Check for missing values in 'headline' and 'short_description'
headline_null_count_top_10 <- sum(is.na(top_10_authors_data$headline) | top_10_authors_data$headline == "")
short_description_null_count_top_10 <- sum(is.na(top_10_authors_data$short_description) | top_10_authors_data$short_description == "")

# Print the frequency of null values
cat("Null/empty values in 'headline' for top 10 authors:", headline_null_count_top_10, "\n")
cat("Null/empty values in 'short_description' for top 10 authors:", short_description_null_count_top_10, "\n")

# Replace null or empty values in 'headline' and 'short_description' with 'Not Available' for the top 10 authors
top_10_authors_data$headline[is.na(top_10_authors_data$headline) | top_10_authors_data$headline == ""] <- "Not Available"
top_10_authors_data$short_description[is.na(top_10_authors_data$short_description) | top_10_authors_data$short_description == ""] <- "Not Available"

# Verify the changes by checking the first few rows
head(top_10_authors_data)



# Create a temporary variable that gives the count after removing "Not Available" values
temp_non_null_count <- top_10_authors_data %>%
  filter(headline != "Not Available", short_description != "Not Available") %>%
  summarize(
    headline_count = n(),  # Count of non-"Not Available" values in 'headline'
    short_description_count = n()  # Count of non-"Not Available" values in 'short_description'
  )

# Print the temporary count
print(temp_non_null_count)




# Function to clean text (remove punctuation, emojis, and perform lemmatization)
clean_and_lemmatize_text <- function(text) {
  # Convert text to lowercase
  text <- tolower(text)
  
  # Remove emojis (filter out non-ASCII characters)
  text <- iconv(text, "UTF-8", "ASCII", sub = "")
  
  # Remove punctuation
  text <- removePunctuation(text)
  
  # Perform lemmatization
  text <- lemmatize_strings(text)
  
  # Remove extra whitespace
  text <- stripWhitespace(text)
  
  return(text)
}

# Apply cleaning and lemmatization to 'headline' and 'short_description'
top_10_authors_data$clean_headline <- sapply(top_10_authors_data$headline, clean_and_lemmatize_text)
top_10_authors_data$clean_short_desc <- sapply(top_10_authors_data$short_description, clean_and_lemmatize_text)

# View the cleaned dataframe
head(top_10_authors_data)





#Condition filtering
# Create a new column based on the condition
top_10_authors_data$length_check <- ifelse(
  nchar(top_10_authors_data$headline) <= 2 * nchar(top_10_authors_data$short_description), 
  "Yes", 
  "No"
)

# Modify the data to include only rows that satisfy the condition (length_check = "Yes")
filtered_data <- top_10_authors_data %>%
  filter(length_check == "Yes")

# Count how many rows have "Yes" in the length_check column
yes_count <- nrow(filtered_data)

# Print the count of "Yes"
cat("Count of 'Yes' where length of headline is <= 2 * length of short_description:", yes_count, "\n")


# Count the number of articles per author that satisfy the condition
article_count_by_author <- filtered_data %>%
  group_by(authors) %>%
  summarise(article_count = n())

# Plot the frequency of articles by authors satisfying the condition
filtered_data_plot <- ggplot(article_count_by_author, aes(x = reorder(authors, -article_count), y = article_count)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  labs(title = "Number of Articles Published by Authors (Condition Met)", x = "Author", y = "Number of Articles") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = article_count), vjust = -0.5, size = 3.5)

# Display the plot
print(filtered_data_plot)





#SENTIMENT ANALYSIS
# Function to calculate sentiment
get_sentiment_label <- function(text) {
  sentiment_score <- get_sentiment(text, method = "syuzhet")
  if (sentiment_score > 0) {
    return("Positive")
  } else if (sentiment_score < 0) {
    return("Negative")
  } else {
    return("Neutral")
  }
}

# Apply sentiment analysis to clean_headline and clean_short_desc
filtered_data$headline_sentiment <- sapply(filtered_data$clean_headline, get_sentiment_label)
filtered_data$short_desc_sentiment <- sapply(filtered_data$clean_short_desc, get_sentiment_label)

# Create a new column to compare sentiments
filtered_data$sentiment_match <- ifelse(
  filtered_data$headline_sentiment == filtered_data$short_desc_sentiment,
  "Yes",
  "No"
)

# Define all possible sentiment levels
sentiment_levels <- c("Positive", "Negative", "Neutral")
match_levels <- c("Yes", "No")

# Count the occurrences of each sentiment label for each column
headline_sentiment_count <- table(factor(filtered_data$headline_sentiment, levels = sentiment_levels))
short_desc_sentiment_count <- table(factor(filtered_data$short_desc_sentiment, levels = sentiment_levels))
sentiment_match_count <- table(factor(filtered_data$sentiment_match, levels = match_levels))

# Print the sentiment counts
cat("Headline Sentiment Count:\n")
print(headline_sentiment_count)

cat("\nShort Description Sentiment Count:\n")
print(short_desc_sentiment_count)

cat("\nSentiment Match Count:\n")
print(sentiment_match_count)

# Get the total number of rows in the filtered data
total_rows_filtered_data <- nrow(filtered_data)

# Print the total number of rows
cat("\nTotal number of rows in filtered data:", total_rows_filtered_data, "\n")




#SEMANTIC ANALYSIS-using Cosine Similarity with TF-ID
# Function to preprocess a single text string
clean_text_single <- function(text) {
  text <- tolower(text)                               # Convert to lowercase
  text <- gsub("[[:punct:]]", "", text)               # Remove punctuation
  text <- gsub("[[:digit:]]", "", text)               # Remove numbers
  text <- removeWords(text, stopwords("en"))          # Remove stopwords
  text <- gsub("\\s+", " ", text)                     # Remove extra whitespace
  text <- trimws(text)                                # Trim leading/trailing whitespace
  return(text)
}

# Apply preprocessing to both headline and short description in the filtered data
filtered_data <- filtered_data %>%
  mutate(
    clean_headline = sapply(headline, clean_text_single),
    clean_short_desc = sapply(short_description, clean_text_single)
  )

# Combine cleaned texts into a single dataframe for TF-IDF vectorisation
combined_text <- c(filtered_data$clean_headline, filtered_data$clean_short_desc)

# Create a Corpus for TF-IDF calculation
corpus <- VCorpus(VectorSource(combined_text))

# Preprocess corpus: remove punctuation, numbers, stopwords, and extra whitespace
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

# Create a Document-Term Matrix (DTM) for TF-IDF
dtm <- DocumentTermMatrix(corpus)
tfidf <- weightTfIdf(dtm)

# Extract TF-IDF matrices for headlines and short descriptions
num_rows <- nrow(filtered_data)
headline_tfidf <- as.matrix(tfidf[1:num_rows, ])
short_desc_tfidf <- as.matrix(tfidf[(num_rows + 1):(2 * num_rows), ])

# Function to calculate cosine similarity
calculate_cosine_similarity <- function(vec1, vec2) {
  dot_product <- sum(vec1 * vec2)
  magnitude1 <- sqrt(sum(vec1^2))
  magnitude2 <- sqrt(sum(vec2^2))
  
  if (magnitude1 == 0 || magnitude2 == 0) {
    return(NA)  # Avoid division by zero
  }
  
  cosine_similarity_score <- dot_product / (magnitude1 * magnitude2)
  return(cosine_similarity_score)
}

# Apply cosine similarity row-wise
filtered_data$cosine_similarity <- mapply(
  calculate_cosine_similarity,
  split(headline_tfidf, row(headline_tfidf)),
  split(short_desc_tfidf, row(short_desc_tfidf))
)

# View results
head(filtered_data)

# Calculate average cosine similarity (excluding NA values)
average_similarity <- mean(filtered_data$cosine_similarity, na.rm = TRUE)
cat("Average Cosine Similarity between headlines and short descriptions:", average_similarity, "\n")






#SEMANTIC ANALYSIS- using Jaccard Similarity
# Function to calculate Jaccard Similarity between two sets of words
calculate_jaccard_similarity <- function(text1, text2) {
  # Split the text into words (treat as sets)
  words1 <- unlist(strsplit(text1, " "))
  words2 <- unlist(strsplit(text2, " "))
  
  # Calculate the intersection and union of the two sets
  intersection <- length(intersect(words1, words2))
  union <- length(union(words1, words2))
  
  # Avoid division by zero
  if (union == 0) {
    return(NA)
  }
  
  # Jaccard similarity score
  jaccard_similarity_score <- intersection / union
  return(jaccard_similarity_score)
}

# Apply Jaccard similarity calculation row-wise
filtered_data$jaccard_similarity <- mapply(
  calculate_jaccard_similarity,
  filtered_data$clean_headline,
  filtered_data$clean_short_desc
)

# Frequency of when Jaccard similarity = 0 (excluding NA values)
jaccard_zero_freq <- sum(filtered_data$jaccard_similarity == 0, na.rm = TRUE)
cat("\nFrequency of Jaccard Similarity = 0:", jaccard_zero_freq, "\n")

# Frequency of when Cosine similarity = 0 (excluding NA values)
cosine_zero_freq <- sum(filtered_data$cosine_similarity == 0, na.rm = TRUE)
cat("\nFrequency of Cosine Similarity = 0:", cosine_zero_freq, "\n")




#Semantic mismatched
# Check if there are any NA values in the "semantic_mismatched" column
cat("Number of NAs in semantic_mismatched:", sum(is.na(filtered_data$semantic_mismatched)), "\n")


# Replace NA, NaN, or Inf with 0 in cosine_similarity and jaccard_similarity
filtered_data$cosine_similarity[is.na(filtered_data$cosine_similarity) | 
                                  is.nan(filtered_data$cosine_similarity) | 
                                  is.infinite(filtered_data$cosine_similarity)] <- 0

filtered_data$jaccard_similarity[is.na(filtered_data$jaccard_similarity) | 
                                   is.nan(filtered_data$jaccard_similarity) | 
                                   is.infinite(filtered_data$jaccard_similarity)] <- 0

# Create the "semantic_mismatched" column correctly (again, just in case)
filtered_data$semantic_mismatched <- ifelse(
  is.na(filtered_data$jaccard_similarity) | is.na(filtered_data$cosine_similarity), 
  NA,  # If any of the similarities are NA, mark as NA
  ifelse(
    filtered_data$jaccard_similarity == 0 & filtered_data$cosine_similarity == 0,
    "Yes",
    "No"
  )
)

# Check the first few rows to verify the "semantic_mismatched" column
cat("\nFirst few rows with semantic_mismatched column:\n")
head(filtered_data)

# Frequency of when Semantic mismatch = "Yes" (handle NA values)
semantic_mismatch_yes_freq <- sum(filtered_data$semantic_mismatched == "Yes", na.rm = TRUE)
cat("\nFrequency of Semantic Mismatch = Yes:", semantic_mismatch_yes_freq, "\n")

# Frequency of when Semantic mismatch = "No" (handle NA values)
semantic_mismatch_no_freq <- sum(filtered_data$semantic_mismatched == "No", na.rm = TRUE)
cat("\nFrequency of Semantic Mismatch = No:", semantic_mismatch_no_freq, "\n")




#FINAL COMPARISION- CLICK BAIT DETECTION
# Count the total number of articles and clickbait articles for each author
author_comparison <- filtered_data %>%
  group_by(authors) %>%
  summarize(
    total_articles = n(),
    clickbait_articles = sum(clickbait == "Yes", na.rm = TRUE)
  ) %>%
  mutate(
    clickbait_percentage = (clickbait_articles / total_articles) * 100,
    non_clickbait_percentage = 100 - clickbait_percentage
  )

# Filter authors with at least 40% clickbait articles
authors_40_percent_clickbait <- author_comparison %>%
  filter(clickbait_percentage >= 40) %>%
  select(authors, clickbait_percentage)

# View authors with at least 40% clickbait articles
print(authors_40_percent_clickbait)

# Prepare data for the first plot (Total vs Clickbait Articles Count)
author_comparison_count_long <- author_comparison %>%
  tidyr::pivot_longer(
    cols = c(total_articles, clickbait_articles),
    names_to = "article_type",
    values_to = "count"
  )

# Plot 1: Total Articles vs Clickbait Articles (Count)
ggplot(author_comparison_count_long, aes(x = reorder(authors, -count), y = count, fill = article_type)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = count), position = position_dodge(width = 0.9), vjust = -0.5, size = 3.5) +
  labs(
    title = "Comparison of Total Articles and Clickbait Articles per Author (Count)",
    x = "Author", y = "Number of Articles"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("lightblue", "lightgreen"))

# Prepare data for the second plot (Clickbait vs Non-Clickbait Percentage)
author_comparison_percentage_long <- author_comparison %>%
  tidyr::pivot_longer(
    cols = c(clickbait_percentage, non_clickbait_percentage),
    names_to = "article_type",
    values_to = "percentage"
  )

# Plot 2: Clickbait vs Non-Clickbait Articles (Percentage)
ggplot(author_comparison_percentage_long, aes(x = reorder(authors, -percentage), y = percentage, fill = article_type)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = round(percentage, 1)), 
            position = position_stack(vjust = 0.5), size = 3.5, color = "black") +
  labs(
    title = "Comparison of Clickbait and Non-Clickbait Articles by Author (Percentage)",
    x = "Author", y = "Percentage of Articles"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(
    values = c("clickbait_percentage" = "lightgreen", "non_clickbait_percentage" = "lightblue"),
    labels = c("Clickbait Articles", "Non-Clickbait Articles")
  )



##K-means Clustering 
#install.packages("cluster")
library(cluster)

# Select only the required columns for clustering
clustering_data <- filtered_data %>%
  select(cosine_similarity, jaccard_similarity)

# Scale the data
scaled_data <- scale(clustering_data)
# Scale cosine_similarity and jaccard_similarity columns
scaled_data <- as.data.frame(apply(filtered_data[, c("cosine_similarity", "jaccard_similarity")], 2, scale))


#Validate distance matrix calculation
# Check if the distance matrix is calculated without errors
distance_matrix <- dist(scaled_data)

# Correct indexing of sil_width
sil_width <- vector("numeric", 9)  # Only storing values for k = 2 to 10

# Perform silhouette analysis
for (k in 2:10) {
  km <- kmeans(scaled_data, centers = k, nstart = 25)
  sil <- silhouette(km$cluster, dist(scaled_data))
  sil_width[k - 1] <- mean(sil[, 3])  # Store silhouette score for each k
}

# Plot silhouette widths
plot(2:10, sil_width, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of Clusters (k)",
     ylab = "Average Silhouette Width",
     main = "Silhouette Analysis for Optimal k")



#K-means clustering
# Set the number of clusters
set.seed(123)  # Ensure reproducibility
k <- 4

# Perform k-means clustering
kmeans_result <- kmeans(scaled_data, centers = k, nstart = 25)

# Add the cluster assignments to the original dataset
filtered_data$cluster <- as.factor(kmeans_result$cluster)

# Validate cluster assignments
table(filtered_data$cluster)

# View the cluster assignments
head(filtered_data)

#Evalusting the silhoutte score 
# Compute silhouette scores for k-means clustering with k = 4
silhouette_scores <- silhouette(kmeans_result$cluster, dist(scaled_data))

# Plot the silhouette scores
plot(silhouette_scores, col = c("pink", "lightblue", "lightgreen", "yellow"),
     main = "Silhouette Plot for k = 4", border = NA)



#Visualisation of the clusters in K-means clustering
# Load ggplot2 for visualisation
library(ggplot2)

# Scatter plot of cosine_similarity vs. jaccard_similarity with cluster colouring
ggplot(filtered_data, aes(x = cosine_similarity, y = jaccard_similarity, color = cluster)) +
  geom_point(alpha = 0.6, size = 3) +
  labs(
    title = "Clusters Based on Cosine and Jaccard Similarity",
    x = "Cosine Similarity",
    y = "Jaccard Similarity",
    color = "Cluster"
  ) +
  theme_minimal()


#Download the final data
# Define the file path where you want to save the Excel file
file_path <- "C:\\Users\\acer\\Desktop\\UOS\\ITDS\\ASSESSMENT\\New folder\\top_10_authors_similarity.xlsx"

# Export the dataframe to Excel
write_xlsx(filtered_data, path = file_path)

# Print confirmation
cat("File saved to:", file_path, "\n")
