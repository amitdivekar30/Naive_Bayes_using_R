# Build a naive Bayes model on the data set for classifying the ham and spam

#importing dataset
dataset<-read.csv(file.choose(),stringsAsFactors = FALSE)
View(dataset)
str(dataset)

table(dataset$type)


# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset$text))
as.character(corpus[[1]])
corpus = tm_map(corpus, content_transformer(tolower))
as.character(corpus[[1]])
corpus = tm_map(corpus, removeNumbers)
as.character(corpus[[1]])
corpus = tm_map(corpus, removePunctuation)
as.character(corpus[[1]])
corpus = tm_map(corpus, removeWords, stopwords())
as.character(corpus[[1]])
corpus = tm_map(corpus, stemDocument)
as.character(corpus[[1]])
corpus = tm_map(corpus, stripWhitespace)
inspect(corpus[1:2])
as.character(corpus[[1]])


# Creating the Bag of Words model
dtm <- DocumentTermMatrix(corpus)
dtm
dim(dtm)


library(wordcloud)
library(RColorBrewer)
wordcloud(dataset$text, max.words = 40, scale = c(3, 0.5))
spam <- subset(dataset, dataset$type == "spam")
wordcloud(spam$text, max.words = 60, colors = brewer.pal(7, "Paired"), random.order = FALSE)
ham <- subset(dataset, dataset$type == "ham")
wordcloud(ham$text, max.words = 60, colors = brewer.pal(7, "Paired"), random.order = FALSE)

# Splitting the dataset into the Training set and Test set
dim(dtm)
training_set <- dtm[1:4457, ]
test_set <- dtm[4458:5559, ]

#Training & Test Label
sms_train_labels <- dataset[1:4457, ]$type
sms_test_labels <- dataset[4458:5559, ]$type

#Proportion for training & test labels
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

#Creating Indicator Features
threshold <- 0.1
min_freq = round(dtm$nrow*(threshold/100),0)

min_freq

# Create vector of most frequent words
freq_words <- findFreqTerms(x = dtm, lowfreq = min_freq)

str(freq_words)

#Filter the DTM
training_set_freq <- training_set[ , freq_words]
test_set_freq <- test_set[ , freq_words]

dim(training_set_freq)

convert_values <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

sms_train <- apply(training_set_freq, MARGIN = 2, convert_values)
sms_test <- apply(test_set_freq, MARGIN = 2, convert_values)


#Create model from the training dataset
#install.packages("e1071")
library(e1071)

sms_classifier <- naiveBayes(sms_train, factor(sms_train_labels))

#Make predictions on test set
sms_test_pred <- predict(sms_classifier, newdata=sms_test)
sms_test_labels<-factor(sms_test_labels)

# Making the Confusion Matrix
library(caret)
table(sms_test_labels)
confusionMatrix(data = sms_test_pred, reference = sms_test_labels, positive = "spam", dnn = c("Prediction", "Actual"))
mean(sms_test_pred==sms_test_labels)   #0.975
