# Load relevant packages 
library(tidyverse)
library(tidytext)
library(rvest)
library(tibble)
library(stringr)
library(dplyr)
library(tibble)
library(quanteda)
require(wordcloud)
require(stopwords)
require(ggplot2)
require(udpipe)
library(patchwork)
library(gt)
library(tm)
library(textir)
library(SentimentAnalysis)
library(tokenizers)
library(slam)
library(Matrix)
library(udpipe)
library(textdata)
library(igraph)
library(ggraph)
library(knitr)
library(vader)


#
library(tm)
library(topicmodels)
library(tidytext)
library(dplyr)
library(ggplot2)

# Load the final data frame
earnings_calls_df <- read_csv("earnings_calls_df_final.csv")
earnings_calls_df$Transcript[2]
library(tidyverse)
library(stringr)
library(data.table)

# Define ESG keywords
esg_keywords <- list(
  Environmental = c("sustainability", "climate", "carbon", "green", "environment"),
  Social = c("diversity", "inclusion", "safety", "welfare", "community"),
  Governance = c("board", "shareholder rights", "transparency", "leadership", "ethics")
)

# Define the function to extract ESG context
extract_esg_context <- function(df, transcript_col, keywords, window = 35) {
  results <- list()
  
  for (i in seq_len(nrow(df))) {
    text <- df[[transcript_col]][i]
    ticker <- df$Ticker[i]   # Get Ticker symbol
    date <- df$Date[i]       # Get Date
    industry <- df$Industry[i] # Get Industry
    
    words <- unlist(str_split(text, "\\s+"))
    
    for (category in names(keywords)) {
      for (keyword in keywords[[category]]) {
        matches <- which(str_detect(tolower(words), keyword))
        
        for (match in matches) {
          start <- max(1, match - window)
          end <- min(length(words), match + window)
          context <- paste(words[start:end], collapse = " ")
          results <- append(results, list(data.frame(
            Ticker = ticker,
            Date = date,
            Industry = industry,
            Category = category,
            Keyword = keyword,
            Context = context
          )))
        }
      }
    }
  }
  
  # Combine all results into a single data frame
  if (length(results) > 0) {
    return(bind_rows(results))
  } else {
    return(data.frame(Ticker = character(), Date = character(),
                      Industry = character(), Category = character(),
                      Keyword = character(), Context = character()))
  }
}

# Example usage with the dataset
# Assuming 'earnings_calls_df' has columns: 'Transcript', 'Ticker', 'Date', 'Industry'

esg_results <- extract_esg_context(
  df = earnings_calls_df,
  transcript_col = "Transcript",
  keywords = esg_keywords,
  window = 35
)

# Save the results to a CSV
saveRDS(esg_results, "esg_extracted_context.rds")




esg_transcript_35 <- read_rds("esg_extracted_context.rds")

library(dplyr)

# Aggregate data by Date
esg_transcript_35 <- esg_transcript_35 %>%
  group_by(Date) %>%
  summarize(
    Tickers = paste(unique(Ticker), collapse = ", "),         # Combine unique Tickers
    Industries = paste(unique(Industry), collapse = ", "),   # Combine unique Industries
    Categories = paste(unique(Category), collapse = ", "),   # Combine unique Categories
    Keywords = paste(unique(Keyword), collapse = ", "), # Combine unique Keywords
    Context = paste(Context, collapse = " "),       # Concatenate all Contexts
    .groups = "drop"
  )



#clean the data
# Load required libraries
library(stringr)
library(tm)

# Define the cleaning function
clean_data <- function(text) {
  # Remove HTML tags
  text <- str_remove_all(text, "<[^>]+>")
  
  # Remove participant names and roles
  text <- str_remove_all(text, "[A-Za-z\\s-]+\\s-\\s[A-Za-z\\s]+")
  
  # Remove instructions and redundant phrases
  text <- str_remove_all(text, "\\[.*?\\]") # Remove square brackets and their content
  text <- str_remove_all(text, "(Thank you.*?\\.|Ladies and gentlemen,.*?\\.)")
  
  # Normalize text
  text <- str_remove_all(text, "[^a-zA-Z\\s]") # Remove special characters and numbers
  text <- tolower(text) # Convert to lowercase
  text <- str_squish(text) # Remove extra whitespace
  
  # Tokenize and remove stopwords
  words <- unlist(str_split(text, "\\s+"))
  custom_stopwords <- c(stopwords("en"), "will", "can", "call", "one")
  words <- words[!words %in% custom_stopwords]
  
  
  # Join the tokens back to form clean text
  clean_text <- paste(words, collapse = " ")
  return(clean_text)
}

# Apply the cleaning function to the Transcript column
esg_transcript_35$Context <- sapply(esg_transcript_35$Context, clean_data)

# View the cleaned data
#head(earnings_calls_df)
saveRDS(esg_transcript_35, file = "esg_transcript_35.rds")

esg_transcript_35 <- read_rds("esg_transcript_35.rds")


#Remove rows where Industries == "NA" (string NA)
esg_transcript_35 <- esg_transcript_35 %>%
  filter(Industries != "NA")
#create bigrams

# Extract Year from the Date column
esg_transcript_35 <- esg_transcript_35 %>%
  mutate(Year = format(as.Date(Date), "%Y"))



# Tokenize into bigrams and include Industry and Year
bigrams <- esg_transcript_35 %>%
  unnest_tokens(bigram, Context, token = "ngrams", n = 2) %>%
  select(Tickers, Industries, Year, bigram)


# Count bigram frequencies grouped by Industry and Year
bigram_counts <- bigrams %>%
  count(Industries, Year, bigram, sort = TRUE)

bigram_counts <- bigram_counts %>%
  mutate(doc_id = paste(Industries, Year, sep = "_"))

dtm <- bigram_counts %>%
  cast_dtm(doc_id, bigram, n)


dim(dtm)


# Remove terms with sparsity > 99% (appearing in <1% of documents)
dtm_trimmed <- removeSparseTerms(dtm, 0.99)

save(dtm_trimmed, file= "dtm_trimmed.rda")
load("dtm_trimmed.rda")
result <- FindTopicsNumber(
  dtm_trimmed,
  topics = seq(from = 3, to = 33, by = 3),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 6L,
  verbose = TRUE
)

getwd
print(result)

FindTopicsNumber_plot(result)

# estimate topic model
topic <- LDA(dtm_trimmed,  # document term matrix
             k = 15, # specifify number of topics
             method = "Gibbs",
             control = list(
               seed = 1234, # eases replication
               burnin = 100,  # how often sampled before estimation recorded
               iter = 300,  # number of iterations
               keep = 1,    # saves additional data per iteration (such as logLik)
               save = F,     # saves logLiklihood of all iterations
               verbose = 10  # report progress
             ))

# Loglikelihood of the choosen model
topic@loglikelihood             
plot(topic@logLiks, type = "l") 

# Controls defining the model
topic@k
topic@alpha
topic@control
topic@iter

# Document index
topic@documents

# Individual terms
topic@terms

# Term distribution for each topic
beta <- exp(topic@beta) # log of probability reported
dim(beta)

# inspect the most common terms of topic 1
head(topic@terms[order(beta[1,], decreasing = T)], 20)

# inspect top 5 terms for all topics
apply(topic@beta, 1, function(x) head(topic@terms[order(x, decreasing = T)],5))

for (i in 1:15) {
  terms.top.40 <- head(topic@terms[order(beta[i, ], decreasing = TRUE)], 40)
  prob.top.40 <- head(sort(beta[i, ], decreasing = TRUE), 40)
  wordcloud(
    words = terms.top.40,
    freq = prob.top.40,
    random.order = FALSE,
    scale = c(2, 1),
    main = paste("Word Cloud for Topic", i)
  )
}




#temporal analysis--------
# Extract the document-topic distribution
gamma <- as.data.frame(topic@gamma)  # topic@gamma stores document-topic probabilities

# Add document IDs back (to link them to the year and industry)
gamma$doc_id <- rownames(dtm_trimmed)

# View the first few rows
head(gamma)


# Split doc_id back into Industry and Year
gamma <- gamma %>%
  separate(doc_id, into = c("Industries", "Year"), sep = "_", convert = TRUE)

# View the updated gamma dataframe
head(gamma)


# Group by Year and calculate average topic proportions
topic_trends <- gamma %>%
  group_by(Year) %>%
  summarise(across(starts_with("V"), mean))

# View the aggregated topic proportions
print(topic_trends)



# Reshape data to long format
topic_trends_long <- topic_trends %>%
  pivot_longer(cols = starts_with("V"), names_to = "Topic", values_to = "Proportion")

# View the long-format data
head(topic_trends_long,45)



library(ggplot2)

# Plot topic proportions over time
ggplot(topic_trends_long, aes(x = Year, y = Proportion, color = Topic, group = Topic)) +
  geom_line(size = 1) +
  geom_point() +
  theme_minimal() +
  labs(
    title = "Temporal Trends of Topics Over Time",
    x = "Year",
    y = "Average Topic Proportion",
    color = "Topics"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Group by Year and Industries, then calculate average topic proportions
topic_trends_industry <- gamma %>%
  group_by(Year, Industries) %>%
  summarise(across(starts_with("V"), mean))

# Reshape to long format
topic_trends_industry_long <- topic_trends_industry %>%
  pivot_longer(cols = starts_with("V"), names_to = "Topic", values_to = "Proportion")

# Plot trends by Industry
ggplot(topic_trends_industry_long, aes(x = Year, y = Proportion, color = Topic, group = Topic)) +
  geom_line(size = 1) +
  facet_wrap(~Industries) +
  theme_minimal() +
  labs(
    title = "Topic Trends by Industry Over Time",
    x = "Year",
    y = "Average Topic Proportion",
    color = "Topics"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


library(ggplot2)

# Sum proportions for ESG topics
gamma$ESG_Importance <- gamma$V1 + gamma$V5 + gamma$V12 + gamma$V13

# Check updated dataframe
head(gamma)




# Group by Industry and Year to calculate average ESG importance
industry_esg_trends <- gamma %>%
  group_by(Industries, Year) %>%
  summarise(Average_ESG_Importance = mean(ESG_Importance, na.rm = TRUE)) %>%
  arrange(Industries, Year)

# View the trends
print(industry_esg_trends)

# Wrap industry names to a specified width
industry_esg_trends$Industries <- str_wrap(industry_esg_trends$Industries, width = 10)

# Plot ESG importance trends by industry
ggplot(industry_esg_trends, aes(x = Year, y = Average_ESG_Importance, color = Industries, group = Industries)) +
  geom_line(size = 1) +
  geom_point() +
  theme_minimal() +
  labs(
    title = "ESG Discussion Trends by Industry Over Time",
    x = "Year",
    y = "Average ESG Topic Proportion",
    color = "Industries"
  ) +
  theme(axis.text.x = element_text(angle = .5, hjust = 1))

# Calculate change in ESG importance over time for each industry
industry_esg_change <- industry_esg_trends %>%
  group_by(Industries, Year) %>%
  summarise(Change = last(Average_ESG_Importance) - first(Average_ESG_Importance)) %>%
  arrange((Change))

# View industries with increasing ESG discussions
head(industry_esg_change,20)


#------

library(dplyr)

# Calculate year-over-year change in ESG importance
industry_esg_change_yearly <- industry_esg_trends %>%
  group_by(Industries) %>%
  arrange(Year) %>%
  mutate(
    ESG_Change = Average_ESG_Importance - lag(Average_ESG_Importance)
  ) %>%
  na.omit()

# Summarize the largest absolute changes
top_10_esg_change <- industry_esg_change_yearly %>%
  group_by(Industries) %>%
  summarise(Total_Change = sum(ESG_Change, na.rm = TRUE)) %>%
  arrange(desc(abs(Total_Change))) %>%  # Sort by absolute change
  slice(1:10)  # Select top 10 industries

# View the top 10 industries with the largest ESG changes
print(top_10_esg_change)

library(ggplot2)

# Plot top 10 industries with the largest ESG importance changes
ggplot(top_10_esg_change, aes(x = reorder(Industries, Total_Change), y = Total_Change)) +
  geom_col(fill = ifelse(top_10_esg_change$Total_Change > 0, "steelblue", "red")) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 10 Industries with Largest ESG Importance Change",
    x = "Industries",
    y = "Total Change in ESG Importance"
  ) +
  theme(axis.text.x = element_text(size = 10, angle = 0, hjust = 1))

#---------
# Top 5 increasing industries
top_5_increasing <- top_10_esg_change %>%
  filter(Total_Change > 0) %>%
  arrange(desc(Total_Change)) %>%
  slice(1:5)

# Top 5 decreasing industries
top_5_decreasing <- top_10_esg_change %>%
  filter(Total_Change < 0) %>%
  arrange(Total_Change) %>%
  slice(1:5)

# Combine both for visualization
top_5_combined <- rbind(top_5_increasing, top_5_decreasing)

# Filter yearly trends for these industries
top_industries_trend <- industry_esg_change_yearly %>%
  filter(Industries %in% top_5_combined$Industries)

library(ggplot2)

# Plot year-by-year ESG trends for top industries
ggplot(top_industries_trend, aes(x = Year, y = ESG_Change, color = Industries, group = Industries)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  facet_wrap(~Industries, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "Year-by-Year ESG Importance Trends for Top Increasing and Decreasing Industries",
    x = "Year",
    y = "Year-over-Year ESG Importance Change",
    color = "Industries"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")

