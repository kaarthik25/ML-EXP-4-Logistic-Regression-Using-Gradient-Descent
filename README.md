# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.<\br>
2. Read the given dataset and assign x and y array.<\br>
3. Split x and y into training and test set.<\br>
4. Scale the x variables.<\b>r
5. Fit the logistic regression for the training set to predict y.<\br>
6. Create the confusion matrix and find the accuracy score, recall sensitivity and specificity.<\br>
7. Plot the training set results.<\br>
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Kaarthikeyan.S
RegisterNumber: 212220040068 
*/
#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading and displaying dataframe
df=pd.read_csv("Social_Network_Ads (1).csv")
df

#assigning x and y and displaying them
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values 

#splitting data into train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

#scaling values and obtaining scaled array of train and test of x
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)

#applying logistic regression to the scaled array
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)

#finding predicted values of y
ypred=c.predict(xtest)
ypred

#calculating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm

#calculating accuracy score
from sklearn import metrics
acc=metrics.accuracy_score(ytest,ypred)
acc

#calculating recall sensitivity and specificity
r_sens=metrics.recall_score(ytest,ypred,pos_label=1)
r_spec=metrics.recall_score(ytest,ypred,pos_label=0)
r_sens,r_spec

#displaying regression 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
xs,ys=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xs[:,0].min()-1,stop=xs[:,0].max()+1,step=0.01),
               np.arange(start=xs[:,1].min()-1,stop=xs[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,c.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,cmap=ListedColormap(("pink","purple")))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x1.max())
for i,j in enumerate(np.unique(ys)):
    plt.scatter(xs[ys==j,0],xs[ys==j,1],
                c=ListedColormap(("white","violet"))(i),label=j)
plt.title("Logistic Regression(Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
```

## Output:

<img width="665" alt="v1" src="https://user-images.githubusercontent.com/94525701/173895989-e0d3fd3a-5eef-4088-bd01-402065818130.png">
<img width="830" alt="v2" src="https://user-images.githubusercontent.com/94525701/173896024-42301e5b-e9f2-4fc1-9876-99df38f68f25.png">
<img width="586" alt="v3" src="https://user-images.githubusercontent.com/94525701/173896044-0ebdb5c2-418c-41c9-a8e4-404122441fdf.png">
<img width="382" alt="v4" src="https://user-images.githubusercontent.com/94525701/173896072-c93f62b2-efb4-4dfb-aee9-1b13baa71534.png">
<img width="575" alt="v5" src="https://user-images.githubusercontent.com/94525701/173896239-25573f8b-27e3-4881-b7fa-cb4498a220ec.png">
<img width="741" alt="v6" src="https://user-images.githubusercontent.com/94525701/173896266-f5847052-1c22-43e2-b952-96389c6967ef.png">


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
