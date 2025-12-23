# import the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# read the dataset
dataset=pd.read_csv('emp_sal.csv')

# x and y variables
x=dataset.iloc[: , 1:2].values
y = dataset.iloc[:,2].values

# linear model with 1 degree
lin_reg=LinearRegression()
lin_reg.fit(x,y)

# visualization
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('linear reg')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


# we will be using poly reg 2 degree
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

# visualization
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('linear reg')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# prediction
lin_reg_pred = lin_reg.predict([[6.5]])
lin_reg_pred

poly_reg_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_reg_pred





















