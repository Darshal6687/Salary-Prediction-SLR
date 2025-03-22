# IMPORT LIBRARY
import numpy as np 	#Array		
import matplotlib.pyplot as plt		
import pandas as pd	
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import variation
import scipy.stats as stats
#Load the dataset
dataset = pd.read_csv(r"C:\Users\darsh\OneDrive\Desktop\FSDS\FSDS_20_03\Salary_Data.csv")

# Split the data into INDEPENDENT & DEPENDENT VARIABLE
X = dataset.iloc[:, :-1]    
y = dataset.iloc[:,-1]

#Split the dataset into training and testing sets (80-20%)

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.2, random_state=0) 

# Reshaping the train & test
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)

# Train the model 
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predict the test set
y_pred = regressor.predict(X_test)

# comparison for y_test vs y_pred
comparison = pd.DataFrame({'Actual' : y_test,'Predicted':y_pred})
print(comparison)

# Visualize the training set
plt.scatter(X_test,y_test,color='red')    
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set
plt.scatter(X_test,y_test,color='red')    
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict the salary for 12 & 20 years of experience using the trained model
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

# Check the model performance 
bias = regressor.score(X_train,y_train)
variance = regressor.score(X_test,y_test)
train_mse = mean_squared_error(y_train,regressor.predict(X_train))
test_mse = mean_squared_error(y_test,y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE:{test_mse:.2f}")


# Let`s find out Descriptive statistics

# Mean
dataset.mean()

#Median
dataset.median()

#Mode
dataset['Salary'].mode()

#Variance
dataset.var()

#Standard Deviation
dataset.std()

#Coefficient of variation(cv)
variation(dataset.values)

variation(dataset['Salary'])

#Correlation
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

#Skewness
dataset.skew()

#Standard Error
dataset.sem()

#Z-score
dataset.apply(stats.zscore)

#Degree of Freedom
a = dataset.shape[0]
b = dataset.shape[1]
    
degree_of_freedom = a-b
print(degree_of_freedom)

#Sum of Squares Regression(SSR)

y_mean = np.mean(y)
SSR = np.sum((y_pred - y_mean)**2)
print(SSR)


#Sum of Squares Error(SSE)
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#Sum of Squares Total(SST)
mean_total = np.mean(dataset.values)
SST = np.sum((dataset - mean_total)**2)
print(SST)

# R-Square

r_square = 1 - (SSR/SST)
print(r_square)


