#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Convert json to csv
raw_file = pd.read_json ('Automotive_5.json', lines=True)
raw_file.to_csv ('Automotive_raw.csv', index = None)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("Automotive_raw.csv")
print (df.head(5))


#Data cleaning and visualization
cols_list=list([col for col in df])  
print (cols_list)   #Show all columns in df

df=df[['reviewText','overall']]   # Exlude not needed columns
print (df.head())

# Plot Distribution of comment rate
plt.figure(figsize=(8,5))

plt.hist(df.overall, color='#abcdef')

plt.ylabel('Number of comments')
plt.xlabel('Comment rate')
plt.title("Distribution of comment rate")

plt.show()

#Exluding neutral comment (rated by 3)

df=df.loc[(df['overall'] != 3)]


#Evenly distibute positive and negative comments in order to avoid data bias

negative_comments = df.loc[(df['overall'] <= 2)]     # Sort only negative comments
positive_comments = df.loc[(df['overall'] >= 4)]     # Sort only positive comments
negative_comments.reset_index(drop=True, inplace=True)
positive_comments.reset_index(drop=True, inplace=True)

print('Number of negative comments:',len(negative_comments), 'Number of positive comments:', len(positive_comments))

from random import randint
import random

difference=len(positive_comments)-len(negative_comments)

list_of_random_index=random.sample(range(1, len(positive_comments)), difference)   # Create a list of random index from positive comments

positive_comments=positive_comments.drop(index=list_of_random_index)     # Create new DataFrame with positive comments with length same as negative comments

# Merge positive and negative comments in one DataFrame

dataset=pd.concat([positive_comments, negative_comments])
dataset.sample(frac=1)
dataset.reset_index(drop=True, inplace=True)

print(dataset.head())

print(len(dataset))

# Split labels in 2 classes instead of 4
dataset['overall']=pd.to_numeric(dataset['overall'])
dataset.loc[dataset['overall']<3,'overall']='Negative'
dataset['overall'].astype(str)
dataset.loc[dataset['overall']!='Negative','overall']='Positive'

# Check if there are NaN values

nan_df = dataset[dataset.isna().any(axis=1)]
print (nan_df.head())  # Display rows with NaN values
dataset=dataset.dropna(how='any')
#Split dataset and prepare for training

from sklearn.model_selection import train_test_split
training, test = train_test_split(dataset, test_size=0.25, random_state=42)
train_x=training["reviewText"]
train_y=training["overall"]
test_x=test["reviewText"]
test_y=test["overall"]

print(train_x.head())
print(train_y.head())
print(test_x.head())
print(test_y.head())

#Bag of words vectorization

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

#Train the models

#Linear Support Vector Machine Model
from sklearn import svm

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)


#Logistic Regression Model

from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

#Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

#Random Forest Model
from sklearn.ensemble import RandomForestClassifier
clf_forest=RandomForestClassifier(random_state=42)
clf_forest.fit(train_x_vectors, train_y)

#Comparison between models

#Mean accuracy
print('Accuracy for Linear Support Vector Machine Model:',clf_svm.score(test_x_vectors, test_y))
print('Accuracy for Logistic Regression Model:',clf_log.score(test_x_vectors, test_y))
print('Accuracy for Decision Tree Model:',clf_dec.score(test_x_vectors, test_y))
print('Accuracy for Random Forest Model:',clf_forest.score(test_x_vectors, test_y))

from sklearn.metrics import f1_score

print('F1 score for Linear Support Vector Machine Model:',(f1_score(test_y,clf_svm.predict(test_x_vectors), average=None)))
print('F1 score for Logistic Regression Model:',(f1_score(test_y,clf_log.predict(test_x_vectors), average=None)))
print('F1 score for Decision Tree Model:',(f1_score(test_y,clf_dec.predict(test_x_vectors), average=None)))
print('F1 score for Random Forest Model:',(f1_score(test_y,clf_forest.predict(test_x_vectors), average=None)))

# Fine tuning for best model: Logistic Regression

from sklearn.model_selection import GridSearchCV

parameters = {'max_iter': (100,500,1000)}

clf = GridSearchCV(clf_log, parameters, cv=5)
clf.fit(train_x_vectors, train_y)

print(clf_log.score(test_x_vectors, test_y))
print(f1_score(test_y,clf_log.predict(test_x_vectors), average=None))


import pickle
with open('./models/automotive_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

comment=str(input("Please write a comment about this automotive part: "))

vectorized_comment=vectorizer.transform([comment])

prediction=clf.predict(vectorized_comment)

print (f'Your comment is {prediction}')


