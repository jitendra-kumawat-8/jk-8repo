# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
model=LinearRegression()
model1=Ridge()
model3=Lasso(alpha=2, fit_intercept=False, tol=0.0000001, positive=True)
data = pd.read_csv("Melbourne_housing_FULL.csv",parse_dates=["Date"])
data=data.drop(["Method","Address","Suburb","Postcode","Lattitude","Longtitude","CouncilArea","Bedroom2"],axis=1)
pd.options.display.float_format='{:,.2f}'.format
plt.figure(figsize=(15,15))
data1=data.drop(["SellerG"],axis=1)
sns.heatmap(data1.corr(),annot=True)
plt.show()
print(data1.corr())

dummies=pd.get_dummies(data.Regionname)
dummies2=pd.get_dummies(data.Type)
dummies3=pd.get_dummies(data.SellerG)
data=pd.concat([data,dummies,dummies2,dummies3],axis='columns')
data["Date"+"_year"]=data["Date"].apply(lambda x: x.year)
data["Date"+"_month"]=data["Date"].apply(lambda x: x.month)
data=data.drop(["Date_year","Date","Regionname","Type","SellerG"],axis=1)
bool_series=pd.notnull(data["Price"])
data=data[bool_series]
bool_series=pd.notnull(data["Bathroom"])
data=data[bool_series]
bool_series=pd.notnull(data["Car"])
data=data[bool_series]
bool_series=pd.notnull(data["Landsize"])
data=data[bool_series]
bool_series=pd.notnull(data["YearBuilt"])
data=data[bool_series]
bool_series=pd.notnull(data["BuildingArea"])
data=data[bool_series]
indexNames=data[(data["Landsize"]<100)].index
data.drop(indexNames,inplace=True)
data.drop(data[data['BuildingArea'] > 900].index, inplace = True)
data.drop(data[data['Rooms'] > 5].index, inplace = True)
data.drop(data[data['BuildingArea'] > 900].index, inplace = True)
data.drop(data[data['Landsize'] > 2000].index, inplace = True)
data.drop(data[data['Bathroom'] > 4].index, inplace = True)
data.drop(data[data['Distance'] > 40].index, inplace = True)
data.drop(data[data['YearBuilt'] < 1870].index, inplace = True)
y=data["Price"]
X=data.drop(["Price"],axis=1)
data[["Price","Distance","Bathroom","Car","Landsize","BuildingArea"]].to_csv("new.csv")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
def test(X):
    model.fit(X_train,y_train)
    print("train score/accuracy",model.score(X_train,y_train)*100)
    print("test score/accuracy",model.score(X_test,y_test)*100)
    model1.fit(X_train,y_train)
    print("train Ridge score/accuracy",model1.score(X_train,y_train)*100)
    print("test Ridge score/accuracy",model1.score(X_test,y_test)*100)
    model3.fit(X_train,y_train)
    print("train Lasso score/accuracy",model1.score(X_train,y_train)*100)
    print("test Lasso score/accuracy",model1.score(X_test,y_test)*100)
ar=[]
dum1=list(dummies.columns)
du=du2=[]
dum2=list(dummies2.columns)
def inputdata():
    for i in ['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize','BuildingArea', 'YearBuilt', 'Propertycount']:
        n=float(input("input "+str(i)))
        ar.append(n)
    for i in range(len(dum1)):
        print(i+1,dum1[i])
    n=int(input("select index for region"))
    for i in range(len(dum1)):
        if(i==n-1):
            ar.append(1)
        else:
            ar.append(0)
    for i in range(len(dum2)):
        print(i+1,dum2[i])
    n=int(input("select index for house type"))
    for i in range(len(dum2)):
        if(i==n-1):
            ar.append(1)
        else:
            ar.append(0)
    for i in range(387):
        ar.append(0)
    ar.append(1)
plt.figure(figsize=(10,10))
plt.scatter(X_test["Rooms"],y_test,marker="x",color="red")
plt.xlabel("Rooms")
plt.ylabel("price")
plt.show()
plt.figure(figsize=(10,10))
plt.scatter(X_test["BuildingArea"],y_test,marker="x",color="blue")
plt.xlabel("BuildingArea")
plt.ylabel("price")
plt.show()
plt.figure(figsize=(10,10))
plt.scatter(X_test["Landsize"],y_test,marker="x",color="green")
plt.xlabel("Landsize")
plt.ylabel("price")
plt.show()
plt.figure(figsize=(10,10))
plt.scatter(X_test["Bathroom"],y_test,marker="x",color="purple")
plt.xlabel("Bathroom")
plt.ylabel("price")
plt.show()
plt.figure(figsize=(10,10))
plt.scatter(X_test["Distance"],y_test,marker="x",color="red")
plt.xlabel("Distance")
plt.ylabel("price")
plt.show()
plt.figure(figsize=(10,10))
plt.scatter(X_test["YearBuilt"],y_test,marker="x",color="blue")
plt.xlabel("YearBuilt")
plt.ylabel("price")
plt.show()
plt.figure(figsize=(10,10))
plt.scatter(X_test["Date_month"],y_test,marker="x",color="green")
plt.xlabel("month")
plt.ylabel("price")
plt.show()
test(X)
inputdata()
i=int(input("enter current month"))
ar.append(i)
print(int(model1.predict([ar])))
