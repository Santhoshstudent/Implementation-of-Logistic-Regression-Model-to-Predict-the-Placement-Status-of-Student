

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

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



## Output:
![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/979e613e-fb9c-4c8d-84a7-7d2db44f84fc)


![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/0920577b-5ea9-4225-834b-aa6711ce1854)


![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/dfde897c-1179-431c-a34f-d8c0a28f5bf1)


![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/33bf7275-8ab9-4354-8e9a-23641b323b63)


![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/15866f74-b4d9-46fd-9f1d-11c7cc4e3c91)


![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/2f2c9810-e1bc-4ba6-bb2f-da6d6a16ca8f)


![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/c493b648-b2d0-4cff-8331-8a072a5c0470)


![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/35524441-44f9-4bfb-aac8-896f67ca4126)


![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/f78e8430-82a1-47af-9e9d-a642ef023f96)


![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/addb70b4-3cbf-4bf3-8982-da5c543cf507)


![image](https://github.com/Kamaleshwa/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144980199/60bbcaeb-bd66-427d-abf3-c0b4341b17b9)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
