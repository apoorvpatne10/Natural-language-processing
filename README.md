# Natural-language-processing
An NLP model for classifying a set of reviews into positive (1) or negative categories using Naive-Bayes classifier.

After importing, the first thing I've got to do is to clean the dataset. For that, I'll initialize an empty list which will soon contain all the 1000 cleaned reviews. I'll iterate through all the 1000 reviews, and for each review, I'll include only the words consisting of english letters. I'll eliminate all the numbers/special characters from each review. A simple regex code [^a-zA-Z] will take care of that. Then it'll be converted into lowercase form. I'll split each review to a list of words, eliminate the stopwords, and and apply stemming on every word to reduce it to its simplest form. Then I'll create a bag of words model to only include relevant words in the matrix of features (the reviews). 

Now my aim is to classify the set of reviews into positive and negative ones.

I'll try using all types of classifier that I know and try to determine which classifier will give the best results. Following are the performance metrics based on which I'll judge the model.

Here, TP = # True Positives, TN = # True Negatives, FP = # False Positives, FN = # False Negatives

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)

**Precision(measuring exactness)** = TP / (TP + FP)

**Recall(measuring completeness)** = TP / (TP + FN)

**F1 Score(compromise between Precision and Recall)** = 2 * Precision * Recall / (Precision + Recall)

#### 1. Logistic Regression 
Accuracy : 0.71
Precision : 0.7586206896551724
Recall : 0.6407766990291263
F1 Score : 0.6947368421052632

#### 2. K-Nearest Neigbors(K-NN)
Accuracy : 0.61
Precision : 0.676056338028169
Recall : 0.46601941747572817
F1 Score : 0.5517241379310345

#### 3. Support Vector Machines
Accuracy : 0.805
Precision : 0.8333333333333334
Recall : 0.7766990291262136
F1 Score : 0.8040201005025125

#### 4. Naive Bayes
Accuracy : 0.73
Precision : 0.6842105263157895
Recall : 0.883495145631068
F1 Score : 0.7711864406779663

#### 5. Decision Tree Classification
Accuracy : 0.71
Precision : 0.9245283018867925
Recall : 0.47572815533980584
F1 Score : 0.6282051282051283

#### 6. Random Forest Classification
Accuracy : 0.705
Precision : 0.8235294117647058
Recall : 0.5436893203883495
F1 Score : 0.6549707602339181

