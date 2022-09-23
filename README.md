# Ex.No.1---Data-Preprocessing
##AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


##ALGORITHM:
```
1.Importing the libraries
2.Importing the dataset
3.Taking care of missing data
4.Encoding categorical data
5.Normalizing the data
6.Splitting the data into test and train
```

##PROGRAM:
```
import pandas as pd
import numpy as np

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI411 - Neural Networks/Churn_Modelling.csv")
df

df.isnull().sum()

#Check for Duplicate Values
df.duplicated()

df.describe()

#Detect the Outliers
# Outliers are any abnormal values going beyond
df['Exited'].describe()

""" Normalize the data - There are range of values in different columns of x are different. 

To get a correct ne plot the data of x between 0 and 1 

LabelEncoder can be used to normalize labels.
It can also be used to transform non-numerical labels to numerical labels.
"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()

df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])

'''
MinMaxScaler - Transform features by scaling each feature to a given range. 
When we normalize the dataset it brings the value of all the features between 0 and 1 so that all the columns are in the same range, and thus there is no dominant feature.'''

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))

df1

df1.describe()

# Since values like Row Number, Customer Id and surname  doesn't affect the output y(Exited).
#So those are not considered in the x values
X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)

y = df1.iloc[:,-1].values
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print("Size of X_train: ",len(X_train))

print(X_test)
print("Size of X_test: ",len(X_test))

X_train.shape
```

##OUTPUT:

## Dataset

<img width="575" alt="image" src="https://user-images.githubusercontent.com/114233500/191958281-d2a9b931-82ee-4f2c-9f4b-d306b6ec6aa7.png">

## checking for null values

<img width="224" alt="image" src="https://user-images.githubusercontent.com/114233500/191958476-92f1db8c-12e7-4e3f-9857-85e68ac0ce3d.png">

## checking for duplicate values

<img width="270" alt="image" src="https://user-images.githubusercontent.com/114233500/191958769-51be193a-859c-42a3-a66d-708eaac9fcce.png">

## describing data

<img width="575" alt="image" src="https://user-images.githubusercontent.com/114233500/191958922-c589d59b-8c13-4a47-8510-14940e7d333a.png">

## checking for outliers in exicted column

<img width="298" alt="image" src="https://user-images.githubusercontent.com/114233500/191959247-b4b25404-4b00-41f5-9337-ae3d6696df55.png">


## normalized dataset

<img width="577" alt="image" src="https://user-images.githubusercontent.com/114233500/191959382-4007d283-8c87-413d-aae1-7a5369785324.png">

## describing normalized dataset

<img width="575" alt="image" src="https://user-images.githubusercontent.com/114233500/191959531-ecce5e40-3c71-42f2-be20-dc5972c0359b.png">


## x - values

<img width="575" alt="image" src="https://user-images.githubusercontent.com/114233500/191959728-c6d76efc-61fa-4790-a005-dfe5a6d6d62e.png">

## y - value

<img width="165" alt="image" src="https://user-images.githubusercontent.com/114233500/191959851-e08aab2a-68a8-4cae-904d-b867d0f8734c.png">


## x_train values

<img width="574" alt="image" src="https://user-images.githubusercontent.com/114233500/191960098-1f06bd90-f9a6-4584-988e-90e762ac57f3.png">

## x_train size

<img width="205" alt="image" src="https://user-images.githubusercontent.com/114233500/191960274-1656d4e6-0c49-4cfa-ac1c-cb8f551f73a4.png">


## x_test values

<img width="572" alt="image" src="https://user-images.githubusercontent.com/114233500/191960456-9ae006aa-8afc-41b6-a61d-b0d89928d5aa.png">


## x_test size

<img width="201" alt="image" src="https://user-images.githubusercontent.com/114233500/191960660-cc4124eb-3978-4adf-b432-af8b03c6e768.png">


## x_train shape

<img width="97" alt="image" src="https://user-images.githubusercontent.com/114233500/191960827-6df6faf9-e6f8-44fe-a10d-daad097eaf4e.png">







##RESULT


Data preprocessing is performed in a data set downloaded from kaggle
