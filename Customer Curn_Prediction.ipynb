# About Dataset:
It is the dataset of a U.S. bank customer for getting the information that , this particular customer will leave
bank or not.

# Objective:
Develop a model to predict customer churn for a subscription- based service or business. Use historical
customer data, including features like usage behavior and customer demographics, and try algorithms like
Logistic Regression, Random Forests, or Gradient Boosting to predict churn.

# Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import os

import warnings
warnings.filterwarnings("ignore")


# Download Dataset

df = pd.read_csv("Churn_Modelling (1).csv")

print("-- First 5 rows --", "\n")
df.head()

# Remove unwanted rows

df.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace= True)

# Check dataframe information

df.info()

# Null Data

df.isnull().sum().sum()

# Duplicate data

print("No.of duplicates: " + str(df.duplicated().sum()))

df.nunique()

df.describe()

# Numeric feauture

df.describe()

df.iloc[:, :].describe().T.sort_values(by='std' , ascending = False)\
                     .style.background_gradient(cmap='GnBu')\
                     .bar(subset=["max"], color='#BB0000')\
                     .bar(subset=["mean",], color='blue')

# EDA : Lets take "Excited" variable and check all the variable relationship with this.


# 1) Excited 

Excited: 1

Non-Exited : 0

df['Exited'].value_counts()

ex_counts=df.Exited.value_counts()
plt.figure(figsize=(12,6))
plt.title('Exited Distribution',size=16,color='navy')


def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

plt.pie(ex_counts,
        labels=ex_counts.index,
        autopct=lambda pct: func(pct,ex_counts),
        startangle=90,
        colors=("blue","red"),
        shadow=True);


# Conclusions - Exited


1. We can note that 7962 of Customers are in group No-exited and 2037 in group Exited. Therefore, 79.6% of Customers are in the No-exited group.

# 2) CreditScore

df['CreditScore'].value_counts()

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=df, x = 'Exited', y = 'CreditScore', errorbar=None, palette="Set1")
plt.title('Exited vs '+ 'CreditScore',color='blue')
ax.bar_label(ax.containers[0], fmt='%0.2f')
plt.show()

# Conclusions - Exited

CreditScore is practically the same for the 2 groups, with a slightly higher value for group No-exited(0).

 # 3) Geography

df['Geography'].value_counts()

fig, ax = plt.subplots(1, 2) 
fig.suptitle('** ' + 'Geography' + ' **', fontsize=16, color='navy') 
plt.style.use('seaborn')
plt.subplot(1,2,1)
df['Geography'].value_counts().plot(kind='bar',color=sns.color_palette("Set1"))
plt.subplot(1,2,2)
df['Geography'].value_counts().plot(kind='pie',autopct="%.2f%%")
plt.show()

fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(data=df, x = 'Geography', y = 'Exited', errorbar=None, palette="Set1")
plt.title('Exited vs '+ 'Geography',color='navy')
ax.bar_label(ax.containers[0], fmt='%0.2f')
plt.show()

# Conclusions - Geography


The country with the highest number of 'Exited' is Germany, with an average of 0.32. This means that in Germany 32% are from the 'Exited' group. On the other hand, France has the largest number of customers.

# 4) Gender

df['Gender'].value_counts()

fig, ax = plt.subplots(1, 2) 
fig.suptitle('** ' + 'Gender' + ' **', fontsize=16, color='navy') 
plt.style.use('seaborn')
plt.subplot(1,2,1)
df['Gender'].value_counts().plot(kind='bar',color=sns.color_palette("Set1"))
plt.subplot(1,2,2)
df['Gender'].value_counts().plot(kind='pie',autopct="%.2f%%")
plt.show()

fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(data=df, x = 'Gender', y = 'Exited', errorbar=None, palette="Set1")
plt.title('Exited vs '+ 'Gender',color='navy')
ax.bar_label(ax.containers[0], fmt='%0.2f')
plt.show()

# Conclusions - Gender

Gender with the highest number of 'Exited' is Female, with an average of 0.25. The Male gender has the majority of Customers, with 54.57%.

# 5) Age

df['Age'].value_counts()

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('** ' + 'Age' + ' **', fontsize=16, color='navy') 
plt.style.use('seaborn')
df['Age'].value_counts().plot(kind='bar',color=sns.color_palette("Set1"))
plt.show()

fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(data=df, x = 'Exited', y = 'Age', errorbar=None, palette="Set1")
plt.title('Exited vs '+ 'Age',color='navy')
ax.bar_label(ax.containers[0], fmt='%0.2f')
plt.show()

# Conclusion: Age

Age with the highest mean is group 1 (Exited), with an average of 44.84 years. In other words, the 'Exited' group has a higher average age than the 'No-exited' group.

# 6) Tenure

df['Tenure'].value_counts()

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('Tenure', fontsize=16, color='navy') 
plt.style.use('seaborn')
df['Tenure'].value_counts().plot(kind='bar',color=sns.color_palette("Set1"))
plt.show()

fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(data=df, x = 'Tenure', y = 'Exited', errorbar=None, palette="Set1")
plt.title('Exited vs '+ 'Tenure',color='navy')
ax.bar_label(ax.containers[0], fmt='%0.2f')
plt.show()

# Conclusion:  Tenure

Tenure with the highest mean of 'Exited' is Tenure = 0 and the lowest is Tenure = 7. On the other hand, Tenure 2 has the highest number of Customers (1048) and Tenure 0 has the lowest (413).

# 7) Balance

df['Balance'].value_counts()

fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(data=df, x = 'Exited', y = 'Balance', errorbar=None, palette="Set1")
plt.title('Exited vs '+ 'Balance',color='navy')
ax.bar_label(ax.containers[0], fmt='%0.2f')
plt.show()

# Conclusion: Balance

Balance with the highest mean is group 1 (Exited), with average = 91108,54.

# 8) NumOfProducts

df['NumOfProducts'].value_counts()

fig, ax = plt.subplots(1, 2) 
fig.suptitle('** ' + 'NumOfProducts' + ' **', fontsize=16, color='navy') 
plt.style.use('seaborn')
plt.subplot(1,2,1)
df['NumOfProducts'].value_counts().plot(kind='bar',color=sns.color_palette("Set1"))
plt.subplot(1,2,2)
df['NumOfProducts'].value_counts().plot(kind='pie',autopct="%.2f%%")
plt.show()

fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(data=df, x = 'NumOfProducts', y = 'Exited', errorbar=None, palette="Set1")
plt.title('Exited vs '+ 'NumOfProducts',color='navy')
ax.bar_label(ax.containers[0], fmt='%0.2f')
plt.show()

# Conclusions: NumOfProducts


1. Customers with 4 and 3 products have a higher average of Exited, but we must emphasize that there are few customers who have these quantities of products.

2. Customers who have 2 products are the ones with the lowest average for Exited, that is, they are the ones with the lowest propensity to Churn.

3. Reinforcing that the largest number of customers are those with 1 NumOfProducts (5084), followed by customers with 2 NumOfProducts (4590).

# 9) HasCrCard

df['HasCrCard'].value_counts()

fig, ax = plt.subplots(1, 2) 
fig.suptitle('** ' + 'HasCrCard' + ' **', fontsize=16, color='navy') 
plt.style.use('seaborn')
plt.subplot(1,2,1)
df['HasCrCard'].value_counts().plot(kind='bar',color=sns.color_palette("Set1"))
plt.subplot(1,2,2)
df['HasCrCard'].value_counts().plot(kind='pie',autopct="%.2f%%")
plt.show()

fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(data=df, x = 'HasCrCard', y = 'Exited', errorbar=None, palette="Set1")
plt.title('Exited vs '+ 'HasCrCard',color='navy')
ax.bar_label(ax.containers[0], fmt='%0.2f')
plt.show()

# Conclusion: HasCrCard

HasCrCard is practically the same for the 2 groups, with a slightly higher value for group 0. HasCrCard = 1 has the most customers (7055) or 70.55%.

# 10) IsActiveMember

df['IsActiveMember'].value_counts()

fig, ax = plt.subplots(1, 2) 
fig.suptitle('** ' + 'IsActiveMember' + ' **', fontsize=16, color='navy') 
plt.style.use('seaborn')
plt.subplot(1,2,1)
df['IsActiveMember'].value_counts().plot(kind='bar',color=sns.color_palette("Set1"))
plt.subplot(1,2,2)
df['IsActiveMember'].value_counts().plot(kind='pie',autopct="%.2f%%")
plt.show()

fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(data=df, x = 'IsActiveMember', y = 'Exited', errorbar=None, palette="Set1")
plt.title('Exited vs '+ 'IsActiveMember',color='navy')
ax.bar_label(ax.containers[0], fmt='%0.2f')
plt.show()

# Conclusions: IsActiveMember

IsActiveMember with the highest mean of 'Exited' is group 0 (No-Exited), with average = 0,27.

# 11) EstimatedSalary

df['EstimatedSalary'].value_counts()

fig, ax = plt.subplots(figsize=(5, 3))
sns.barplot(data=df, x = 'Exited', y = 'EstimatedSalary', errorbar=None, palette="Set1")
plt.title('Exited vs '+ 'EstimatedSalary',color='navy')
ax.bar_label(ax.containers[0], fmt='%0.2f')
plt.show()

# Conclusion: EstimatedSalary
EstimatedSalary is practically the same for the 2 groups, with a slightly higher value for group 1 (Exited).

# 

# Machine Learning

df.info()

# transforming 'object' to numerical columns with LabelEncoder

LabelEncoder= LabelEncoder()
df2 = df.copy()
df2['Geography']=LabelEncoder.fit_transform(df2['Geography'])
df2['Gender']=LabelEncoder.fit_transform(df2['Gender'])
df2.head()

# Heatmap: insights into the relationships between different features in a dataset

df_corr = df2.corr()
f, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdYlGn',annot_kws={'size': 8}, ax=ax)
plt.show()

# Correlations with "Exited"

limit = -1.0

data = df2.corr()["Exited"].sort_values(ascending=False)
indices = data.index
labels = []
corr = []
for i in range(1, len(indices)):
    if data[indices[i]]>limit:
        labels.append(indices[i])
        corr.append(data[i])
sns.barplot(x=corr, y=labels)
plt.title('Correlations with "Exited"')
plt.show()

# CatBoostClassifier

# assign X and y values
X,y=df2.drop("Exited",axis=1),df2[['Exited']]

from sklearn.model_selection import train_test_split

# split the data to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

!pip install catboost

# Classification Algorithms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from catboost import CatBoostClassifier

ct=CatBoostClassifier(verbose=False)
ct.fit(X_train,y_train)

# Predict the Test set results

y_pred = ct.predict(X_test)

# Check accuracy score 

from sklearn.metrics import accuracy_score

print('Model accuracy score : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# ConfusionMatrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap = plt.cm.Blues, normalize = None, display_labels = ['No-exited', 'Exited'])

# classification_report

print(classification_report(y_test, y_pred))
