# Natural Language Processing

# Impoting the dataset
dataset_original = read.delim("Restaurant_Reviews.tsv", quote = '', stringsAsFactors = FALSE)

# Cleaning the text
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
# Converting all the uppercase letters to lowercase
corpus = tm_map(corpus, content_transformer(tolower))
# Removing the numbers
corpus = tm_map(corpus, removeNumbers)
# Removing all the punctuations 
corpus = tm_map(corpus, removePunctuation)
# Removing all the non-relevant words
corpus = tm_map(corpus, removeWords, stopwords())
# Stemming : only considering the root of a particular word.
# Eg : the word 'loved' will be replaced by 'love'
corpus = tm_map(corpus, stemDocument)
# Removing the extra white spaces
corpus = tm_map(corpus, stripWhitespace)

# Creating a bag of words model
# A huge table, a sparse matrix, with 1000 rows, each row for one review, and 
# columns contain all the words that may be present in the review
dtm = DocumentTermMatrix(corpus)
# Keeping 99% of most frequent words in the sparse matrix 
dtm = removeSparseTerms(dtm, 0.999)

dataset = as.data.frame(as.matrix(dtm))
# Added a new column Liked(the dependent variable) from the original
# dataset to the current dataset
dataset$Liked = dataset_original$Liked

# Encoding the target feature as factor 
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-690],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-690])

# Making the Confusion Matrix
cm = table(test_set[, 690], y_pred)


# 



    