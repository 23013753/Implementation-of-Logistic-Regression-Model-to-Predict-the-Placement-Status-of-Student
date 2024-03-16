# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1 Import the required packages and print the present data.
2  Print the placement data and salary data.
3  Find the null and duplicate values.
4  Using logistic regression find the predicted values of accuracy , confusion matrices.
5  Display the results
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.


## Developed by: VISHAL.S
## RegisterNumber: 212223240184

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



 
*/
```

## Output:
PLACEMENT DATA 

![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/ebe1b65d-d249-41ee-9a63-e9f092e15fa8)


SALARY DATA 

![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/e23c1edd-52ca-434e-aeae-1bc8cadc1565)


CHECKING THE NULL() FUNTION
![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/a3ca46d8-18e0-48e2-b758-7b3a40f29c99)


DATA DUPLICATE
![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/53f3576a-6e1d-4598-b522-2d66a984c82e)


PRINT DATA 

![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/b3c9e09f-02c5-4514-9b98-64b00361c927)

DATA-STATUES

![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/70f032a6-6649-481a-846c-3178e1e8fcf8)


Y_prediction array:
![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/7e7b2e97-5122-4c90-81f4-1b90ba26bc42)

Accuracy value:

![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/4d6fdd85-82d2-4837-828f-3b8f04d58a26)

Confusion array:

![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/f4aab30e-b831-4fd4-abab-f2f12b07ba99)

Classification Report

![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/cd9512d8-7c17-404f-a95c-9b144eb01374)

Prediction of LR:
![image](https://github.com/23013753/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145634121/01996c33-11ca-410d-9594-6d4533f465b7)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
