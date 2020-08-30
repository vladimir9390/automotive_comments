import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Convert json to csv
df = pd.read_json ('Automotive_5.json', lines=True)
df.to_csv ('Automotive_raw.csv', index = None)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

raw_file = pd.read_csv("Automotive_raw.csv")
#print (raw_file.head(3))

#Data operations and visualization
not_needed_columns=[]

for column in raw_file:
	if not column=="reviewText" and not column=="overall":
		not_needed_columns.append(column)
	
#print (not_needed_columns)

raw_data = df.drop(columns=not_needed_columns)

#print (raw_data.head(3))



"""plt.figure(figsize=(8,5))

plt.hist(raw_data.overall, color='#abcdef')

plt.ylabel('Number of comments')
plt.xlabel('Comment rate')
plt.title("Distribution of comment rate")

plt.show()"""

#Exluding neutral comment (rated by 3)

raw_data=raw_data.loc[(raw_data['overall'] != 3)]

#print(len(raw_data))

#Evenly distibute positive and negative comments in order to avoid data bias

negative_comments = raw_data.loc[(raw_data['overall'] <= 2)]
positive_comments = raw_data.loc[(raw_data['overall'] >= 4)]
negative_comments.reset_index(drop=True, inplace=True)
positive_comments.reset_index(drop=True, inplace=True)

#print(len(negative_comments), len(positive_comments))

from random import randint
import random

difference=len(positive_comments)-len(negative_comments)

a=random.sample(range(1, len(positive_comments)), difference)

#print (len(a))

reduced_positive_comments=positive_comments.drop(index=a)

#print(reduced_positive_comments.head(10))
#print(len(reduced_positive_comments))

#Making dataset

dataset=pd.concat([reduced_positive_comments, negative_comments])
dataset.sample(frac=1)
dataset.reset_index(drop=True, inplace=True)

#print(dataset.head(5))

#print(len(dataset))

#Split dataset and prepare for training

from sklearn.model_selection import train_test_split
training, test = train_test_split(dataset, test_size=0.33, random_state=42)
train_x=training["reviewText"]
train_y=training["overall"]
test_x=test["reviewText"]
test_y=test["overall"]

"""print(train_x.head(5))
print(train_y.head(5))
print(test_x.head(5))
print(test_y.head(5))"""

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

"""

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
#print(clf_svm.score(test_x_vectors, test_y))
#print(clf_log.score(test_x_vectors, test_y))
#print(clf_dec.score(test_x_vectors, test_y))
#print(clf_forest.score(test_x_vectors, test_y))

from sklearn.metrics import f1_score

#print(f1_score(test_y,clf_svm.predict(test_x_vectors), average=None))
#print(f1_score(test_y,clf_log.predict(test_x_vectors), average=None))
#print(f1_score(test_y,clf_dec.predict(test_x_vectors), average=None))
#print(f1_score(test_y,clf_forest.predict(test_x_vectors), average=None))"""

from sklearn.model_selection import GridSearchCV

"""parameters = {"penalty": ("l1", "l2", "elasticnet", "none"),'max_iter': (100,125,150)}

clf = GridSearchCV(clf_log, parameters, cv=5)
clf.fit(train_x_vectors, train_y)

print(clf_log.score(test_x_vectors, test_y))
print(f1_score(test_y,clf_log.predict(test_x_vectors), average=None))"""

parameters = {'kernel': ('linear', 'rbf'), 'C': (1,4,8,16,32)}
clf = GridSearchCV(clf_svm, parameters, cv=5)
clf.fit(train_x_vectors, train_y)
#print(clf_svm.score(test_x_vectors, test_y))
#print(f1_score(test_y,clf_svm.predict(test_x_vectors), average=None))

import pickle
with open('./models/automotive_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

def rating(number):
	if number<=2:
		return "This is negative comment"
	else:
		return "This is positive comment"

comment=str(input("Please write a comment about this automotive part: "))

vectorized_comment=vectorizer.transform([comment])

number=clf.predict(vectorized_comment)

print (rating(number))

