

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
## Developed by: santhosh kumar B
## RegisterNumber: 212223230193

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

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: santhosh kumar B
RegisterNumber: 212223230193 
*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![image ml 1](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/debdc54d-2ed2-4561-b31e-ce8064148400)
![image  ml 2](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/8a38cca3-3e46-4521-a28f-400ccacecfff)
![image ml 3](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/349fb16b-a234-4300-9091-c381d1305b43)
![image ml 4](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/4aace5f8-8563-4d8d-94cd-fa20228716a2)
![image ml 5](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/38597af8-9934-49f4-a174-ce05fa18bf8f)
![image ml 6](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/2889ec1b-c3b2-4eff-8a7e-5112aecd65a7)
![image ml 7](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/e30acbc0-f809-4eb5-a338-d8dacb47fa0c)
![image4 ml 8](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/77a7400f-d3f7-41c3-8dbe-51da358c6c37)
![image ml 9](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/365d9e99-cb37-48ee-b612-fde9eb33221d)
![image ml 10](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/aa4e1291-34c8-4d1e-a111-d7a3fc3e21ac)
![image ml 11](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/0ddd5c05-3e67-4c65-b60c-d2090647026d)













## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
