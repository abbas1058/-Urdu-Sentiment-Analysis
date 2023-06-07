# -Urdu-Sentiment-Analysis


# Better Classifier for Urdu Sentiment Analysis

We are making a Urdu Tweets Classifier.We are trying to check which ML- Model is best for Urdu Sentiment Analysis.

we are implementing a part of a research paper in this
project.
Here is the link:
https://ieeexplore.ieee.org/document/9466841/authors

Sentiment analysis is the process of identifying and extracting opinions and emotions expressed in text data. Urdu is a widely spoken language, and sentiment analysis for Urdu text is an important task. In this report, we will discuss different classifiers that can be used for Urdu sentiment analysis and recommend a better classifier based on our analysis.



## Data Source

The first step in building a sentiment analysis model is to gather data.
To do that we searched from the internet but their was not a compiled data set so we reached Professor of ITU Sir Zunnurain and got a data set of 50,000 Urdu tweets of a balanced data sets with two lables 'positive and negative'.


## Importing Data

The Dataset have two files Test Dataset and Train Dataset.
First we imported both and concatenated them.

## Preprocessing Data

We used Urduhack NLP library for urdu language to
- normalize
- remove_punctuation
- remove_accents
- replace_urls
- replace_emails
- replace_phone_numbers
- replace_currency_symbols
- remove_english_alphabets

Documentation: https://urduhack.readthedocs.io/en/stable/reference/preprocessing.html

### Vactorization

To convert tweets to features we vectorized it using TF-IDF

### Label Encoding

We did label encoding to convert features to numbers 

### Train Test Split

We split the data to 30% test data and 70% Train Data randomly.
## Models

The Models we applied were:
- Logistic Regression
- Naive Bayes
- Decision Tree
- SVM
- Random Forest
 
### **Logistic Regression**

Logistic Regression: Logistic regression is a statistical model that is used to analyze the relationship between a dependent variable and one or more independent variables. 

#### Confusion Matrix
We got the following confusion Matrix

![LR_CM](https://drive.google.com/uc?export=view&id=1n_iWCv9NO9orQGo6SOzi5TDl3QdGMhpU)

#### ROC Curve

![ROC Curve](https://drive.google.com/uc?export=view&id=1jGUZ_zz_2oow-FwyiJTMfjiak-5MhIyG)

#### Precision Recall Curve

![Precision Recall Curve](https://drive.google.com/uc?export=view&id=1RjbalJo0qVY1Y1IM8AZvPwcfHpP_B_vD)


### **Naive Bayes**

Naive Bayes: It is a probabilistic classifier that is widely used for text classification tasks. Naive Bayes assumes that the features are independent of each other and calculates the conditional probability of each class given the features.

#### Confusion Matrix
We got the following confusion Matrix

![LR_CM](https://drive.google.com/uc?export=view&id=1n_iWCv9NO9orQGo6SOzi5TDl3QdGMhpU)

#### ROC Curve

![ROC Curve](https://drive.google.com/uc?export=view&id=1ZmheRRL-UK17_kOgzW3WGKOekVKUjOSX)


#### Precision Recall Curve

![Precision Recall Curve](https://drive.google.com/uc?export=view&id=1nP8JqvekbDcyt36ejr1hwu9v8JBX4ez4)

### **Decision Tree**

A decision tree is a popular machine learning algorithm used for both classification and regression tasks. It works by recursively partitioning the input space into smaller regions or subsets, based on the values of the input features, until it reaches a set of leaf nodes that contain the final predictions.
#### Confusion Matrix
We got the following confusion Matrix

![LR_DT](https://drive.google.com/uc?export=view&id=1LSQqVk_rmtUki3N-UzzSn02ieumUPKyh)

#### ROC Curve

![ROC Curve](https://drive.google.com/uc?export=view&id=1hdv4dNVQ11gHoAaHxVjfFxXQLY9DqHSa)


#### Precision Recall Curve

![Precision Recall Curve](https://drive.google.com/uc?export=view&id=1fw-UZDmIdz1JNQuR4Ox-1dZfcc0QYIXZ)

### **SVM**

SVM (Support Vector Machine) is a popular machine learning algorithm that can be used for both classification and regression tasks. It works by finding a hyperplane that separates the input data into different classes, with the largest possible margin between the classes.

#### Confusion Matrix
We got the following confusion Matrix

![SVM_CM](https://drive.google.com/uc?export=view&id=1Iz57Abre6RNKsOk8x70pEdCTAyVUk2r0)

#### ROC Curve

![ROC Curve](https://drive.google.com/uc?export=view&id=1h7zfVNlhK6WRMERMJ25pPzF9bWuFxGQW)


#### Precision Recall Curve

![Precision Recall Curve](https://drive.google.com/uc?export=view&id=1Jakihwz9wonANtyBn7JJTaz0gxpg3hzs)

### **Random Forest**

Random Forest is a popular ensemble learning algorithm used for both classification and regression tasks. It works by building multiple decision trees on randomly sampled subsets of the input data and features, and then combining their predictions to make the final decision.

#### Confusion Matrix
We got the following confusion Matrix

![RandomForest_CM](https://drive.google.com/uc?export=view&id=1Mvx_KnB5Ouf69eI6DkzX3TrJLKUh-HbN)

#### ROC Curve

![ROC Curve](https://drive.google.com/uc?export=view&id=1PMkvgFWaYS_SB9If2xa4V28RsmqCfjjX)

#### Precision Recall Curve

![Precision Recall Curve](https://drive.google.com/uc?export=view&id=16BKffO7JiyOLhqW1Q_wTHy6pe7YOtvMl)
### **KNN**

KNN (K-Nearest Neighbors) is a simple and popular machine learning algorithm used for both classification and regression tasks. It works by finding the K nearest data points to a new input, based on a distance metric such as Euclidean distance or cosine similarity, and then using the labels (for classification) or values (for regression) of those nearest neighbors to make the final prediction.

#### Confusion Matrix
We got the following confusion Matrix

![KNN_CM](https://drive.google.com/uc?export=view&id=1RdzmTTRYxcR0nrAAZgqrfXyLYQ6RPCHm)

#### ROC Curve

![ROC Curve](https://drive.google.com/uc?export=view&id=1nJVnsq2MM6k9CyV-EZ51XzFi0tQnvp4c)

#### Precision Recall Curve

![Precision Recall Curve](https://drive.google.com/uc?export=view&id=1knb5dFlMebfS7_Q3GyiiePnMGA_rE8vE)


## Conclusion
After applying the models we got the following Accuracies:
Logistic Regression : 90%
Naive Bayes : 83%
Decision Tree : 89%
Random Forest : 94%
KNN : 72%

Which concludes that Random Forest is the best model for the Urdu Sentimetal analysis among these models.

