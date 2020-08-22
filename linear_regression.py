# Importing the libraries
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def get_dataset(title):
    return pd.read_csv(title+'.csv')

def data_reshape(data,key):
    return data[key].values.reshape(-1,1)

def data_min_max_norm(data):
    return MinMaxScaler().fit_transform(data)

dataset=get_dataset(title='./data/Salary_Data')
X_ = data_reshape(data=dataset,key='YearsExperience')
y_ = data_reshape(data=dataset,key='Salary')

X = data_min_max_norm(data=X_)
y = data_min_max_norm(data=y_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

def poly_features(data,degree=4):
    return PolynomialFeatures(degree = degree).fit_transform(data)

def lin_reg_fit(x,y):
    return LinearRegression().fit(x,y)

def lin_reg_predict(lin_reg_model,data):
    if lin_reg_model!=None:
        return lin_reg_model.predict(data)
    else:
        print('Model is not valid')
        return None

def shape_write(data,title='data'):
    print('{} shape: {}'.format(title,data.shape))


def accuracy_calculater(lin_reg_model, y_data, y_pred):
    print("Coefficients(slope of the line):", lin_reg_model.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(y_data, y_pred))
    print("R-square: %.2f’" % r2_score(y_data, y_pred), end='\n')


def plot_data(lin_reg_model, x_data, y_data, x_title='x data', y_title='y data', color_data='red', color_line='blue'):
    plt.scatter(x_data, y_data, color=color_data)
    plt.plot(x_data, lin_reg_model.predict(x_data), color=color_line)
    plt.title('Fitted Line:  y = %.2f + %.2f * x' % (lin_reg_model.intercept_, lin_reg_model.coef_))
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.show()

def plot_data_poly(lin_reg_model,x_data,y_data,X,x_title='x data',y_title='y data',color_data='red',color_line='blue'):
    X_grid = np.arange(min(X),max(X),((max(X)-min(X))/len(X))).reshape(-1,1)
    plt.scatter(x_data, y_data, color = color_data)
    plt.plot(X_grid, lin_reg_model.predict(poly_features(X_grid)), color = color_line)
    plt.title('Polynomial Regression results')
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.show()

lin_reg_normal = lin_reg_fit(X_train,y_train)
y_pred_normal = lin_reg_predict(lin_reg_normal,X_test)

shape_write(X_train)
shape_write(y_train)

accuracy_calculater(lin_reg_normal,y_test,y_pred_normal)

plot_data(lin_reg_normal,X_train,y_train)

X_poly_train = poly_features(data=X_train,degree = 4)
y_poly_train = poly_features(data=y_train,degree = 4)
X_poly_test = poly_features(data=X_test,degree = 4)
y_poly_test = poly_features(data=y_test,degree = 4)

lin_reg_poly = lin_reg_fit(X_poly_train,y_train)

y_pred_poly_train = lin_reg_predict(lin_reg_poly,X_poly_train)
y_pred_poly_test = lin_reg_predict(lin_reg_poly,X_poly_test)

print('Train:')
accuracy_calculater(lin_reg_poly,y_train,y_pred_poly_train)
print('\nTest:')
accuracy_calculater(lin_reg_poly,y_test,y_pred_poly_test)

print('Train:')
shape_write(X_poly_train)
shape_write(y_poly_train)
print('\nTest:')
shape_write(X_poly_test)
shape_write(y_poly_test)

print('Train:')
plot_data_poly(lin_reg_poly,X_poly_train,y_poly_train,X_train)

print('Test:')
plot_data_poly(lin_reg_poly,X_poly_test,y_poly_test,X_test)