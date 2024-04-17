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

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]
````

## Output:
![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/dcd405d2-c4f7-4f1c-b062-0b426607ae2a)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/09479610-8439-4687-a19f-edf1463bc317)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/d0370ef7-d1b4-4e80-af16-9fc75b518a6b)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/c8e5bc57-f40d-4563-90a1-bbbc050403d0)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/5dff19e8-e97d-4c30-890e-b7e160f69cbb)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/7519bf9f-6115-4755-a92b-7f7c14147fce)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/0feef00e-ee89-48da-b2db-8648c79437dd)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/e1304e58-171d-44d0-beb3-22375b709056)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/cc5c433d-a1fa-48c8-a23a-4fcb0140bef1)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/0cbe1af2-cd65-4cce-ba80-4cf99f2c1891)

![image](https://github.com/Santhoshstudent/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145446853/099dd784-3334-46d8-b596-5aac46d3347f)













## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
