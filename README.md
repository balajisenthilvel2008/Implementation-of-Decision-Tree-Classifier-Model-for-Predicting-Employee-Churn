# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: BALAJI S
RegisterNumber: 212225220015
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\Users\acer\Downloads\Employee (1).csv")
print(data.head())
```

<img width="1097" height="557" alt="image" src="https://github.com/user-attachments/assets/20f5d4ab-2c32-419a-9662-146f54e9d755" />

```
data.info()
```

<img width="723" height="478" alt="image" src="https://github.com/user-attachments/assets/3c69fecb-07a2-471a-ae52-07ed4fa9214e" />

```
data.isnull().sum()
```

<img width="380" height="320" alt="image" src="https://github.com/user-attachments/assets/bad67b6b-10f6-49f9-a5da-b2cca0000c02" />

```
data['left'].value_counts()
```

<img width="647" height="107" alt="image" src="https://github.com/user-attachments/assets/6e3c4916-c0c8-42d8-a133-aadefaa5afe3" />

```
le=LabelEncoder()
data["salary"]=le.fit_transform(data['salary'])
print(data.head())
```

<img width="988" height="562" alt="image" src="https://github.com/user-attachments/assets/4f317d38-9305-47b8-9fbc-afe494b2f2e1" />

```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
y=data['left']
print(data.head())
```

<img width="1048" height="562" alt="image" src="https://github.com/user-attachments/assets/17b89646-b83f-43a8-a28d-2fb8e49ffe44" />

```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"The accuracy is : {accuracy:.2f}")
```

<img width="708" height="47" alt="image" src="https://github.com/user-attachments/assets/65c70a41-8cad-4dd8-993a-43e71da11993" />

```
dt.predict([[0.5,0.8,8,260,6,0,1,2]])
```

<img width="641" height="40" alt="image" src="https://github.com/user-attachments/assets/9f9481b8-ee5f-412b-9553-063828959c4c" />

```
plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```

<img width="952" height="682" alt="image" src="https://github.com/user-attachments/assets/6d3a1461-034e-437c-b884-3c1c4e21d778" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
