EX 04: Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2. Print the placement data and salary data.
 
3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.
   
## Program:
Program to implement the the Logistic Regression Model to
Predict the Placement Status of Student.
Developed by: santhosh kumar B
RegisterNumber: 212223230193
import pandas as pd
data=pd.read_csv("C:/Users/admin/OneDrive/Documents/INTRO TO ML/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
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
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


## Output:
![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/9f5f2119-827d-49e0-84a5-b6b3eafd85d8)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/58d0ffd9-9d2b-489c-b737-7e4d319ca032)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/5965f6be-9986-46d4-916e-2a7b821e2462)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/51fde3dc-3d3f-408c-b57e-5b9ba6ffa2a0)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/e4bf44a3-d164-4bf9-abe1-e2d88fa4dfd8)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/875be3a6-0ad8-401b-b89a-76d65338459e)


















## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
