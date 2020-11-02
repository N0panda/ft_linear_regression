import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

def predict(X, theta0, theta1):
    return theta0 + (theta1 * X)

def normalisation(X, mileage):
    return ((mileage - np.mean(X)) / np.std(X)), ((X - np.mean(X)) / np.std(X))

def reverse_norm(y, price):
    return ((price * np.std(y)) + np.mean(y))

data = pd.read_csv("data.csv")

X = data.values[:, 0]
X = X.reshape(X.shape[0], 1)

y = data.values[:, 1]
y = y.reshape(y.shape[0], 1)

answer = input("Do you want to reset theta ? [yes/no]")
if answer == "yes":
    file = pd.DataFrame({"theta0": [0], "theta1": [0]})
    file.to_csv("theta.csv", index=False)

theta = pd.read_csv("theta.csv")
theta = np.array(theta).T

error = 1
while (error == 1):
    val = input("Enter a positive mileage [ex : (120000)] :")
    if len(val) > 10:
        print("10 char max")
        continue
    try :
        val = int(val)
        if val >= 0 :
            error = 0
        else :
            print("Positive integer")
    except :
        print("Bad value")

norm_x, data_X = normalisation(X, val)
norm_x = np.array(norm_x)
norm_y_pred = predict(norm_x, theta[0, 0], theta[1, 0])
y_pred = reverse_norm(y, norm_y_pred)
if y_pred < 0: y_pred = 0
print("predicted price : ", y_pred)
