#Data Scientist Guide
#import all of these standard libraries
import pandas as pd  # To read data
import numpy as np #math
import matplotlib as plt #graphs
import seaborn as sns #graphics
import sklearn #scipy extra

import scipy #owns datascience
import statsmodels.api as sm #teaching
import matplotlib.pyplot as plt  # To visualize
sns.set() #overrides matplotlib

#Independent variables are features and are on the left of your excel. Like marketing

#dependent are things your features influence,and are usually on the right side of the excel. Like total revenue

data= pd.read_csv('C:/ or C:\\ or Blah.csv', index_col=0)

#Understand The Data by checking it over
data.head()
data.tail()
data.shape
data.info()
data.describe()
data['column'].describe()
data.columns
data.nunique()
data['column name'].unique

#Get total occurence of object
data['column'].value_counts()

##############Cleaning Data##############

#Check for nulls, can replace nulls with mean
data.isnull

#shows which columns have null values
data.isnull().any()

#if Data is not skewed use MEDIAN
data.fillna(data.median())
#if Data is skewed use this MODE one
data['salary'] = data['salary'].fillna(data['salary']

#Final check for nulls
df.isnull().sum()

#Remove Duplicates, Depends on situation.
data.drop_duplicates(inplace= True)


'''
In the event you have a NAN column, like "ages",and its next to a full column 
like,"professions" you can simply group and use median value. 
'''
data['NAN column'].fillna(data.groupby('full column')['NAN column'].transform('median'),inplace= True)


#Drop unneeded cels
newData = data.drop(['trash1'], ['trash2'])

########################Exploratory Data Analysis############################

#Relationship Analysis
correlation = newData.corr()

sns.heatmap(correlation,xticklabels = correlation.columns, yticklabels= correlation.columns, annot=True)

sns.pairplot(newData)

sns.relplot(data =newData, x='columnX',y= 'columnY', hue='BinaryColumn eg gender')

sns.distplot(newData['Column'])

sns.catplot(data=newData, x ='columnX', y='columnY')

data.describe()


#Linear Regression Template
#Y is dependent (Like Revenue), x1 is your feature
y =data['GPA']
x1=data['SAT']

plt.scatter(x1,y)
plt.ylabel('SAT', fontsize=20)
plt.xlabel('GPA', fontsize=20)
plt.show()

x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
results.summary()

'''y=b0+b1x1
From your results get the const = b0
Under that should be your = b1
'''

plt.scatter(x1,y)
'''
Below is a template,uncomment it
Your const b0 will replace the 0.275
Your b1 will replace the 0.0017
'''

yhat=0.275 + 0.0017* x1

fig =plt.plot(x1, yhat, lw=4, c='orange', label='Regression Line')

plt.xlabel('SAT',fontsize =20)
plt.ylabel('GPA', fontsize=20)
plt.show()


'''
Classification

Logistic Regression = 1+ variables to get True or False outcome
    eg: risk factor of diseases, weather

Naive Bayes = Assumes that the presence of a feature in a class is unrelated to the presence of any other feature
    Outperforms other classification Algorithms
    eg: spam filters, disease factors

Stochastic Gradient Descent = Used when n is large
    eg: internet of things, updating parameters in neural networks

K-nearest neighbor = lazy learner, good with large data
    eg: similar tasks, handwirting tasks

Decision Tree = if-then rules, built top down, simple/easy to visualize, may become complex bad tree
    eg: finance, checking for disease check
    
Random Forest = Making many trees, then uses average to improve accuracy, more accurate than one tree, complex/slow
    eg: finding risk, failure of mechanical parts, predicting social media, and performance

Neural Network = neurons taking input and then applies nonlinear function, no feedback, high tolerance to noisy data
    eg: colorization of b/w images, captioning

Support Vector Machine = training data as points in space, memory efficient
    eg: investment decisions, sorting resumes on quality

Evaluation of classifier:
    Holdout Method = most common, test/train set 20/80%
    Cross Validation =    

s

'''






'''
Put a retrospective of what you'd do better
'''